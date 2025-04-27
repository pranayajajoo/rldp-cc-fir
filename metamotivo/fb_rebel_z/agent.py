# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from .model import FBModel, config_from_dict
from .model import Config as FBModelConfig
from ..nn_models import weight_init, _soft_update_params, eval_mode
from ..misc.zbuffer import ZBuffer
from pathlib import Path
import json
import safetensors
import copy
from ..pretrained_models import WALKER_OFFLINE_BFM_MODEL, CHEETAH_OFFLINE_BFM_MODEL, QUADRUPED_OFFLINE_BFM_MODEL, POINTMASS_OFFLINE_BFM_MODEL
import math
from torch.distributions import Normal
@dataclasses.dataclass
class TrainConfig:
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float | None = None
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    residual_critic: bool = True
    zero_shot_initialization: bool = True
    q_pi_z: bool = False


@dataclasses.dataclass
class Config:
    model: FBModelConfig = dataclasses.field(default_factory=FBModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    cudagraphs: bool = False
    compile: bool = False


class FBAgent:
    def __init__(self, **kwargs):
        self.cfg = config_from_dict(kwargs, Config)
        self.cfg.train.fb_target_tau = float(min(max(self.cfg.train.fb_target_tau, 0), 1))
        self._model = FBModel(**dataclasses.asdict(self.cfg.model))
        self.num_updates = 0
        


    def setup_pretrained_networks(self,global_cfg):
        if 'walker' in global_cfg.domain_name:
            pretrained_path = WALKER_OFFLINE_BFM_MODEL
        elif 'cheetah' in global_cfg.domain_name:
            pretrained_path = CHEETAH_OFFLINE_BFM_MODEL
        elif 'quadruped' in global_cfg.domain_name:
            pretrained_path =   QUADRUPED_OFFLINE_BFM_MODEL
        elif 'pointmass' in global_cfg.domain_name:
            pretrained_path =   POINTMASS_OFFLINE_BFM_MODEL
        checkpoint = torch.load(pretrained_path+'/model/model.pt', map_location=self.device)
        self._model.load_state_dict(checkpoint['state_dict'],strict=False)
        
        # import ipdb;ipdb.set_trace()
        self.setup_training()
        self.setup_compile()
        # self._model._forward_map.train(False)
        # self._model._backward_map.train(False)
        # # self._model._actor.train(False)
        # Disable gradient computation for pretrained models
        # for p in self._model._forward_map.parameters():
        #     p.requires_grad = False    
        # for p in self._model._backward_map.parameters():
        #     p.requires_grad = False
        # for p in self._model._actor.parameters():
        #     p.requires_grad = False
        # import ipdb;ipdb.set_trace()
        # for name, param in self.agent._model._backward_map.named_parameters():
        #     print(f"Name: {name}")
        #     print(f"Parameter: {param}")
        #     print(f"Shape: {param.shape}")
        self._model.to(self.cfg.model.device)
        
    def setup_zero_shot_initialization(self, unnorm_z_inf, z_inf):
        self.unnorm_z_inf = unnorm_z_inf.detach()
        self.z_inf = z_inf.detach()
        self.z_inf_rep = self.z_inf.repeat(self.cfg.train.batch_size,1)
        self.z_inf.requires_grad = False
        self.unnorm_z_inf.requires_grad = False
        self.scale = torch.linalg.norm(self.unnorm_z_inf.view(-1))/torch.linalg.norm(self.z_inf.view(-1))
        self.unnorm_z_inf_rep = self.unnorm_z_inf.repeat(self.cfg.train.batch_size,1)
        # import ipdb;ipdb.set_trace()
        if self.cfg.train.zero_shot_initialization:
            with torch.no_grad():
                self._model._hierarchical_actor.action.data = copy.deepcopy(self.z_inf.reshape(1,-1))


    @property
    def device(self):
        return self._model.cfg.device

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights


        self.q_optimizer = torch.optim.Adam(
            self._model._qs.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=0,
        )
        self.hierarchical_actor_optimizer = torch.optim.Adam(
            self._model._hierarchical_actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=0,
        )

        # prepare parameter list
        self.qs_paramlist = tuple(x for x in self._model._qs.parameters())
        self.target_qs_paramlist = tuple(x for x in self._model._target_qs.parameters())
        self.hierarchical_actor_paramlist = tuple(x for x in self._model._hierarchical_actor.parameters())
        # self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        # self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        # self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        # self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        # precompute some useful variables
        # self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        # self.off_diag_sum = self.off_diag.sum()

        # self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self.cfg.model.device)

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_rebel_actor = torch.compile(self.update_rebel_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            # self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_rebel_actor = CudaGraphModule(self.update_rebel_actor, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor,  mean: bool = True) -> torch.Tensor:
        return self._model.act(obs,  mean)



    def update(self, replay_buffer, zs, returns) -> Dict[str, torch.Tensor]:
        self.num_updates+=1
        # batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        # obs, action, reward, next_obs, terminated = batch['observation'], batch['action'], batch['reward'], batch['next_observation'], batch['terminated']

        # discount = self.cfg.train.discount * ~terminated

        # self._model._obs_normalizer(obs)
        # self._model._obs_normalizer(next_obs)
        # with torch.no_grad(), eval_mode(self._model._obs_normalizer):
        #     obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()

        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        torch.compiler.cudagraph_mark_step_begin()
        metrics = {}
        metrics.update(
        self.update_actor(
            replay_buffer,
            zs=zs,
            returns = returns,
            clip_grad_norm=clip_grad_norm,
        )
        )
  

        return metrics


    def update_actor(
        self,
        # obs: torch.Tensor,
        replay_buffer,
        zs: torch.Tensor,
        returns: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_rebel_actor(replay_buffer, zs=zs,returns=returns, clip_grad_norm=clip_grad_norm)

    def log_prob_gaussian_dist(self, inputs, means, stds):
        """
        Compute the log probability of each input in the batch under its corresponding Gaussian distribution using torch.distributions.

        Args:
            inputs (torch.Tensor): Tensor of shape (batch_size, d) representing the input data.
            means (torch.Tensor): Tensor of shape (batch_size, d) representing the means of the Gaussians.
            log_vars (torch.Tensor): Tensor of shape (batch_size, d) representing the log variances of the Gaussians.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) containing the log probabilities.
        """
        # Standard deviation is the square root of variance
        std = stds
        
        # Create a Normal distribution for each element in the batch
        dist = Normal(means, std)
        
        # Compute log probabilities for each dimension
        log_probs = dist.log_prob(inputs)
        
        # Sum log probabilities across dimensions to get the total log probability per data point
        total_log_probs = log_probs.sum(dim=1)
        
        return total_log_probs

    def update_rebel_actor(self, replay_buffer, zs: torch.Tensor,returns:torch.Tensor, clip_grad_norm: float | None) -> Dict[str, torch.Tensor]:
        # self._model._actor.train()
        # import ipdb;ipdb.set_trace()
        current_z_mean = self._model.hierarchical_actor()[0]
        cosine_distance = F.cosine_similarity(current_z_mean,self.z_inf)
        mse_distance = F.mse_loss(current_z_mean,self.z_inf)
        
        zs = torch.vstack(zs).to(self.device)
        # self._model._actor = self._model._actor.clone().requires_grad_(True)
        
        # zs_mean = self._model._hierarchical_actor.action.repeat(zs_orig.shape[0],1).detach()
        # noise = (zs_orig.detach() - zs_mean)/torch.exp(self._model._hierarchical_actor.action_log_std)
        # zs = self._model._hierarchical_actor.action.repeat(zs_orig.shape[0],1) + torch.exp(self._model._hierarchical_actor.action_log_std)*noise
        
        # zs.retain_grad()

        returns = torch.vstack(returns).to(self.device)
        
        batch = replay_buffer["train"].sample(zs.shape[0]*zs.shape[0])

        states, action, next_obs, terminated = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
        )
        
        rebel_loss = 0
        temp = 1
        # import ipdb;ipdb.set_trace()
        zs_1 = zs.repeat(len(zs),1)
        zs_2 = torch.repeat_interleave(zs,len(zs),dim=0)
        returns_1 = returns.repeat(len(returns),1)
        returns_2 = torch.repeat_interleave(returns,len(returns),dim=0)

        base_distribution_1 = self._model._actor.get_normal(states, zs_1, 0.2)
        base_distribution_2 = self._model._actor.get_normal(states, zs_2, 0.2)

        policy_distribution_1 = self._model._actor.get_normal(states, zs_1, 0.2)
        policy_distribution_2 = self._model._actor.get_normal(states, zs_2, 0.2)
        # sample_a1 = policy_distribution_1.rsample()
        # sample_a2 = policy_distribution_2.rsample()
        sample_a1 = policy_distribution_1.loc + torch.randn(policy_distribution_1.loc.shape).to(policy_distribution_1.loc.device)*policy_distribution_1.scale
        sample_a2 = policy_distribution_1.loc + torch.randn(policy_distribution_1.loc.shape).to(policy_distribution_1.loc.device)*policy_distribution_1.scale
        sample_a1 = sample_a1.detach()
        sample_a2 = sample_a2.detach()
        # import ipdb;ipdb.set_trace()
        # self.log_prob_gaussian_dist(sample_a1.detach(),policy_distribution_1.loc,policy_distribution_1.scale)
        # for name, param in self._model._actor.named_parameters():
        #     # print(param.requires_grad)
        #     if param.grad is not None:
        #         print(name, param.grad.norm().item())
        #     else:
        #         print(name, "has no grad")

        # dummy = sum(p.sum() for p in self._model._actor.parameters() if p.requires_grad)
        # dummy.backward()
        logpis_base_1 = base_distribution_1.log_prob(sample_a1).sum(-1)
        logpis_base_2 = base_distribution_2.log_prob(sample_a2).sum(-1)

        logpis_policy_1 = policy_distribution_1.log_prob(sample_a1).sum(-1)
        logpis_policy_2 = policy_distribution_2.log_prob(sample_a2).sum(-1)
        # rebel_loss = -logpis_policy_1.mean()
        rebel_loss = F.mse_loss((1/temp*((logpis_policy_1 -logpis_base_1.detach()) -(logpis_policy_2 -logpis_base_2.detach() ))),  (returns_1 - returns_2).reshape(-1))
        # rebel_loss = () - (returns_1 - returns_2).reshape(-1))**2).mean()

        self.hierarchical_actor_optimizer.zero_grad(set_to_none=True)
        
        rebel_loss.backward()
        # for name, param in self._model._actor.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.norm().item())
        #     else:
        #         print(name, "has no grad")
        # import ipdb;ipdb.set_trace()
        
        # z_mean_after = F.normalize(self._model._hierarchical_actor.action,dim=-1)*math.sqrt(self._model.cfg.archi.z_dim)
        # cosine_similarity_after = F.cosine_similarity(z_mean_after, self.z_inf, dim=-1)
        # mse_distance_after = F.mse_loss(z_mean_after, self.z_inf)
        # if clip_grad_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(self._model._hierarchical_actor.parameters(), clip_grad_norm)
        self.hierarchical_actor_optimizer.step()

        return {"actor_loss": rebel_loss.detach(),"mse_distance":mse_distance.detach(), "cosine_similarity":cosine_distance.detach()}

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def maybe_update_rollout_context(self, z: torch.Tensor | None, step_count: torch.Tensor) -> torch.Tensor:
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self.cfg.model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self.cfg.model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self.cfg.model.device))
        else:
            z = self._model.sample_z(step_count.shape[0], device=self.cfg.model.device)
        return z

    @classmethod
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device
        agent = cls(**loaded_config)
        # optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True)
        # agent.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
        # agent.backward_optimizer.load_state_dict(optimizers["backward_optimizer"])
        # agent.forward_optimizer.load_state_dict(optimizers["forward_optimizer"])

        safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device)
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)
        # save optimizer
        torch.save(
            {
                # "actor_optimizer": self.actor_optimizer.state_dict(),
                # "backward_optimizer": self.backward_optimizer.state_dict(),
                # "forward_optimizer": self.forward_optimizer.state_dict(),
            },
            output_folder / "optimizers.pth",
        )
        # save model
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))
