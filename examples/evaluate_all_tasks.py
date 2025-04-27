# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch

torch.set_float32_matmul_precision("high")

import numpy as np
import dataclasses
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.fb import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
from dm_control import suite
import random
from pathlib import Path
import wandb
import json
from typing import List
import mujoco
import warnings
import tyro
from dmc_tasks import dmc
from logging_utils.logx import EpochLogger


ALL_TASKS = {
    "walker": ["walk", "run", "stand", "spin"],
    "cheetah": ["walk", "run", "walk_backward", "run_backward"],
    "pointmass": ["reach_top_left", "reach_top_right", "reach_bottom_right", "reach_bottom_left", "loop", "square", "fast_slow"],
    "quadruped": ["jump", "walk", "run", "stand"],
}



def create_agent(
    domain_name="walker",
    task_name="walk",
    device="cpu",
    compile=False,
    cudagraphs=False,
) -> FBAgent:
    if domain_name not in ["walker", "pointmass", "cheetah", "quadruped"]:
        raise RuntimeError('FB configuration defined only for "walker", "pointmass", "cheetah", "quadruped"')
    env = dmc.make(f"{domain_name}_{task_name}")

    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = env.observation_spec().shape[0]
    # agent_config.model.obs_dim = env.observation_spec()["observations"].shape[0]
    agent_config.model.action_dim = env.action_spec().shape[0]
    agent_config.model.device = device
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1
    agent_config.train.batch_size = 1024
    # archi
    if domain_name in ["walker", "pointmass"]:
        agent_config.model.archi.z_dim = 100
    else:
        agent_config.model.archi.z_dim = 50
    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2
    # optim
    if domain_name == "pointmass":
        agent_config.train.lr_f = 1e-4
        agent_config.train.lr_b = 1e-6
        agent_config.train.lr_actor = 1e-6
    else:
        agent_config.train.lr_f = 1e-4
        agent_config.train.lr_b = 1e-4
        agent_config.train.lr_actor = 1e-4
    agent_config.train.ortho_coef = 1
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5

    if domain_name == "pointmass":
        agent_config.train.discount = 0.99
    else:
        agent_config.train.discount = 0.98
    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    return agent_config

def load_data(dataset_path, expl_agent, domain_name, num_episodes=1):
    path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
    print(f"Data path: {path}")
    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {"observation": [], "terminated": [], "physics": []},
    }
    files = list(path.glob("*.npz"))
    num_episodes = min(num_episodes, len(files))
    for i in tqdm(range(num_episodes)):
        f = files[i]
        data = np.load(str(f))
        storage["observation"].append(data["observation"][:-1].astype(np.float32))
        storage["action"].append(data["action"][1:].astype(np.float32))
        storage["next"]["observation"].append(data["observation"][1:].astype(np.float32))
        storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=np.bool))
        storage["physics"].append(data["physics"][:-1])
        storage["next"]["physics"].append(data["physics"][1:])

    for k in storage:
        if k == "next":
            for k1 in storage[k]:
                storage[k][k1] = np.concat(storage[k][k1])
        else:
            storage[k] = np.concat(storage[k])
    return storage

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass
class TrainConfig:
    pretrained_model:str
    dataset_root: str
    seed: int = 0
    domain_name: str = "walker"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 3_000_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 10_000
    work_dir: str | None = None

    checkpoint_every_steps: int = 1_000_000

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_every_steps: int = 100_000
    eval_tasks: List[str] | None = None

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = "fb_train_dmc"
    wandb_name_prefix: str | None = None

    def __post_init__(self):
        if self.eval_tasks is None:
            self.eval_tasks = ALL_TASKS[self.domain_name]


class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBAgentConfig) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        if self.cfg.work_dir is None:
            import string

            tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
            self.cfg.work_dir = str(self.work_dir)
        else:
            self.work_dir = Path(self.cfg.work_dir)
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print(f"working dir: {self.work_dir}")


        logger_kwargs={'output_dir':self.work_dir, 'exp_name':self.cfg.wandb_pname+'_train', 'output_fname':'train_log.txt'}
        self.train_logger = EpochLogger(**logger_kwargs)
        logger_kwargs={'output_dir':self.work_dir, 'exp_name':self.cfg.wandb_pname+'_eval', 'output_fname':'eval_log.txt'}
        self.eval_logger = EpochLogger(**logger_kwargs)

        self.agent = FBAgent(**dataclasses.asdict(self.agent_cfg))
        set_seed_everywhere(self.cfg.seed)

        if self.cfg.use_wandb:
            exp_name = "fb"
            wandb_name = exp_name
            if self.cfg.wandb_name_prefix:
                wandb_name = f"{self.cfg.wandb_name_prefix}_{exp_name}"
            # fmt: off
            wandb_config = dataclasses.asdict(self.cfg)
            wandb.init(entity=self.cfg.wandb_ename, project=self.cfg.wandb_pname,
                group=self.cfg.agent.name if self.cfg.wandb_gname is None else self.cfg.wandb_gname, name=wandb_name,  # mode="disabled",
                config=wandb_config)  # type: ignore
            # fmt: on

        with (self.work_dir / "config.json").open("w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def evaluate(self):
        self.replay_buffer = {}
        # LOAD DATA FROM EXORL
        data = load_data(
            self.cfg.dataset_root,
            self.cfg.dataset_expl_agent,
            self.cfg.domain_name,
            self.cfg.load_n_episodes,
        )
        self.replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=self.agent.device)}
        self.replay_buffer["train"].extend(data)
        print(self.replay_buffer["train"])
        del data
        checkpoint = torch.load('/project/pi_sniekum_umass_edu/hsikchi/motivo_checkpoints/fb_vanilla_new2/walker/1/checkpoint/model/model.pt', map_location=self.agent.device)
        self.agent._model.load_state_dict(checkpoint['state_dict'])
        # torch.load()
        # self.agent.load(self.cfg.pretrained_model + "/checkpoint")
        # tasks = ALL_TASKS[self.cfg.domain_name]
        eval_dict = self.eval(0)
        print(eval_dict)
        

    def eval(self, t):
        for task in ALL_TASKS[self.cfg.domain_name]:
            z = self.reward_inference(task).reshape(1, -1)
            eval_env = dmc.make(f"{self.cfg.domain_name}_{task}")
            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            for ep in range(num_ep):
                time_step = eval_env.reset()
                t = 0
                while not time_step.last():
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs = torch.tensor(
                            time_step.observation.reshape(1, -1),
                            # time_step.observation["observations"].reshape(1, -1),
                            device=self.agent.device,
                            dtype=torch.float32,
                        )
                        # import ipdb;ipdb.set_trace()
                        action = self.agent.act(obs=obs, z=z, mean=True).cpu().numpy()
                    time_step = eval_env.step(action)
                    t+=1
                    total_reward[ep] += time_step.reward
                print(t)
            # m_dict = {
            #     "reward": np.mean(total_reward),
            #     "reward#std": np.std(total_reward),
            # }
            m_dict = {}
            # m_dict[task] = {
            #     "reward": np.mean(total_reward),
            #     "reward#std": np.std(total_reward),
            # }
            m_dict[task+"_reward"] = np.mean(total_reward)
            m_dict[task+"_reward#std"] = np.std(total_reward)
            
            # {
            #     "reward": np.mean(total_reward),
            #     "reward#std": np.std(total_reward),
            # }
            if self.cfg.use_wandb:
                wandb.log(
                    {f"{task}/{k}": v for k, v in m_dict.items()},
                    step=t,
                )
            
            print(m_dict)
        return m_dict

    def reward_inference(self, task) -> torch.Tensor:
        # set all seeds to 0
        set_seed_everywhere(0)
        env = dmc.make(f"{self.cfg.domain_name}_{task}")
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = []
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(batch["next"]["physics"][i].cpu().numpy())
                env._physics.set_control(batch["action"][i].cpu().detach().numpy())
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            rewards.append(env._task.get_reward(env._physics))
        rewards = np.array(rewards).reshape(-1, 1)
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        import ipdb;ipdb.set_trace()
        return z


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    warnings.warn(
        "Since the original creation of ExORL, mujoco has seen many updates. To rerun all the actions and collect a physics consistent data, you may optionally use the update_data.py utility from MTM (https://github.com/facebookresearch/mtm/tree/main/research/exorl)."
    )
    if config.task_name is None:
        if config.domain_name == "walker":
            config.task_name = "walk"
        elif config.domain_name == "cheetah":
            config.task_name = "run"
        elif config.domain_name == "pointmass":
            config.task_name = "reach_top_left"
        elif config.domain_name == "quadruped":
            config.task_name = "run"
        else:
            raise RuntimeError("Unsupported domain, you need to specify task_name")
    agent_config = create_agent(
        domain_name=config.domain_name,
        task_name=config.task_name,
        device=config.device,
        compile=config.compile,
        cudagraphs=config.cudagraphs,
    )

    ws = Workspace(config, agent_cfg=agent_config)
    ws.evaluate()
