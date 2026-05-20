# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import dataclasses
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import safetensors.torch
from ..nn_models import build_encoder, build_forward, build_actor, eval_mode
from .. import config_from_dict, load_model


@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2


@dataclasses.dataclass
class ForwardArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class BackwardArchiConfig:
    hidden_dim: int = 512
    za_dim: int = 256
    enc_horizon:int = 5
    hidden_layers: int = 2
    norm: bool = True


@dataclasses.dataclass
class ArchiConfig:
    z_dim: int = 512
    norm_z: bool = True
    f: ForwardArchiConfig = dataclasses.field(default_factory=ForwardArchiConfig)
    b: BackwardArchiConfig = dataclasses.field(default_factory=BackwardArchiConfig)
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)


@dataclasses.dataclass
class Config:
    obs_dim: int = -1
    action_dim: int = -1
    device: str = "cpu"
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    norm_obs: bool = False

class FBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = config_from_dict(kwargs, Config)
        obs_dim, action_dim = self.cfg.obs_dim, self.cfg.action_dim
        arch = self.cfg.archi
        # create networks
        # self._backward_map = build_backward(obs_dim, arch.z_dim, arch.b)
        self._backward_map = build_encoder(obs_dim,action_dim,arch.z_dim, arch.b.za_dim,arch.z_dim ,arch.b)
        repr_dim = arch.z_dim
        self._forward_map = build_forward(repr_dim, arch.z_dim, action_dim, arch.f)
        # self._forward_psm_map = build_forward(repr_dim, self.max_log_seed,action_dim, arch.f, output_dim=arch.z_dim)
        self._actor = build_actor(repr_dim, arch.z_dim, action_dim, arch.actor)
        self._obs_normalizer = nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        # self._target_forward_psm_map = copy.deepcopy(self._forward_psm_map)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.cfg.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_model(path, device, cls=cls)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        return self._backward_map.features(self._normalize(obs))

    def state_representation_from_normalized_obs(self, obs: torch.Tensor, detach: bool = True):
        if detach:
            with torch.no_grad():
                return self._backward_map.features(obs).detach()
        return self._backward_map.features(obs)

    @torch.no_grad()
    def state_representation(self, obs: torch.Tensor):
        return self.state_representation_from_normalized_obs(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._forward_map(self.state_representation(obs), z, action)

    @torch.no_grad()
    def forward_psm_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._forward_psm_map(self._normalize(obs), z, action)


    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self.state_representation(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    def reward_inference(self, next_obs: torch.Tensor, reward: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        num_batches = int(np.ceil(next_obs.shape[0] / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
            B = self.backward_map(next_obs[start_idx:end_idx].to(self.cfg.device))
            z += torch.matmul(wr[start_idx:end_idx].to(self.cfg.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)

    def save(self, output_folder: str) -> None:
        """
        Saves the model's state dictionary in SafeTensor format.

        Args:
            path (str): The file path where the SafeTensor will be saved.
        """
        torch.save({
            "state_dict": self.state_dict(),
            "cfg": dataclasses.asdict(self.cfg)
        }, output_folder+"/model.pt")
        try:
            # Ensure the model is on CPU before saving to avoid device-related issues
            state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
            safetensors.torch.save_file(state_dict, output_folder+"/model.safetensors")
            print(f"Model successfully saved to {output_folder} in SafeTensor format.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")