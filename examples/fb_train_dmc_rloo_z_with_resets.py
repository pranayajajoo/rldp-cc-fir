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
from metamotivo.fb_rloo_z   import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from metamotivo.pretrained_models import WALKER_ONLINE_BFM_MODEL
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
from dmc_tasks.dmc import ExtendedTimeStep
from dm_env import StepType
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
    residual_critic = True,
    zero_shot_initialization=True,
    actor_lr = 1e-1,
    expl_logstd = -1,
) -> FBAgent:
    if domain_name not in ["walker", "pointmass", "cheetah", "quadruped"]:
        raise RuntimeError('FB configuration defined only for "walker", "pointmass", "cheetah", "quadruped"')
    env = dmc.make(f"{domain_name}_{task_name}")
    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = env.observation_spec().shape[0]
    agent_config.model.action_dim = env.action_spec().shape[0]
    agent_config.model.device = device
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1
    agent_config.model.expl_logstd = expl_logstd
    agent_config.train.batch_size = 1024
    agent_config.train.residual_critic = residual_critic
    agent_config.train.zero_shot_initialization = zero_shot_initialization
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
    agent_config.model.archi.q.residual_hidden_dim = 64
    # optim
    # if domain_name == "pointmass":
    #     agent_config.train.lr_f = 1e-4
    #     agent_config.train.lr_b = 1e-6
    #     agent_config.train.lr_actor = 1e-6
    # else:

    agent_config.train.fb_target_tau = 0.001 # changed TODO
    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-4
    agent_config.train.lr_actor = actor_lr
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
    dataset_root: str
    seed: int = 0
    domain_name: str = "walker"
    fb_type: str = "offline"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 5_00_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 10_000
    warm_start_timesteps: int = 0
    update_per_timesteps: int = 4
    actor_update_freq: int = 4
    work_dir: str | None = None

    checkpoint_every_steps: int = 1_000_000

    # Algorithm specific
    residual_critic: bool = True
    zero_shot_initialization: bool = True 
    num_actor_updates: int = 1

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_every_steps: int = 10_000
    eval_tasks: List[str] | None = None

    # RLOO hyperparams
    horizon: int = 100
    num_zs: int = 10
    num_trajs_per_state: int = 5
    expl_logstd: float = -1
    actor_lr: float = 5e-2
    initial_state_sample_prob: float = 0.2
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
            self.eval_tasks = [self.task_name]


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

    def train(self):
        self.start_time = time.time()
        self.train_online()

    def train_offline(self) -> None:
        
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

        # Setup pretrained networks
        unnorm_z_inf, z_inf = self.reward_inference_with_projection(self.cfg.task_name)
        self.agent.setup_pretrained_networks(self.cfg, unnorm_z_inf.reshape(1, -1), z_inf.reshape(1, -1))

        total_metrics = None
        fps_start_time = time.time()
        for t in tqdm(range(0, int(self.cfg.num_train_steps))):
            if t % self.cfg.eval_every_steps == 0:
                self.eval(t)

            # torch.compiler.cudagraph_mark_step_begin()
            metrics = self.agent.update(self.replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if t % self.cfg.log_every_updates == 0:
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
            if (t+1) % self.cfg.checkpoint_every_steps == 0:
                # import ipdb;ipdb.set_trace()
                self.agent.save(str(self.work_dir / "checkpoint"))
        self.agent.save(str(self.work_dir / "checkpoint"))
        return


    def train_online(self) -> None:
        

        if self.cfg.fb_type=='online':
            import h5py
            
            if 'walker' in self.cfg.domain_name:
                filename = WALKER_ONLINE_BFM_MODEL+'/online_buffer.h5'

            self.replay_buffer = {}
            with h5py.File(filename, 'r') as h5f:
                for key in h5f.keys():
                    self.replay_buffer[key] = h5f[key][:]
            print(f"Data successfully loaded from {filename}")
            self.agent.setup_pretrained_networks(self.cfg)
            batch_idx = np.random.randint(0,len(self.replay_buffer['observation']),size=(self.cfg.num_inference_samples,))
            batch = {
                'next_observation':self.replay_buffer['next_observation'][batch_idx],
                'next_physics':self.replay_buffer['next_physics'][batch_idx],
                'action':self.replay_buffer['action'][batch_idx],
                
            }
            unnorm_z_inf, z_inf = self.reward_inference_with_projection_given_samples(self.cfg.task_name,batch)
            self.agent.setup_zero_shot_initialization(unnorm_z_inf.reshape(1, -1), z_inf.reshape(1, -1))

        else:
            self.replay_buffer = {}
            # LOAD DATA FROM EXORL
            data = load_data(
                self.cfg.dataset_root,
                self.cfg.dataset_expl_agent,
                self.cfg.domain_name,
                self.cfg.load_n_episodes,
            )
            self.replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=self.agent.device)}
            # import ipdb;ipdb.set_trace()
            self.replay_buffer["train"].extend(data)
            print(self.replay_buffer["train"])
            del data

            # Setup pretrained networks
            # 
            print("Setting pretrained networks for task " ,self.cfg.task_name)

            self.agent.setup_pretrained_networks(self.cfg)
            unnorm_z_inf, z_inf = self.reward_inference_with_projection(self.cfg.task_name)
            self.agent.setup_zero_shot_initialization(unnorm_z_inf.reshape(1, -1), z_inf.reshape(1, -1))


        returns = []
        zs = []
        log_probs = []
        reset_buffer = {'states':[],'physics':[],'actions':[]}
        cum_reward = 0
        train_env = dmc.make(f"{self.cfg.domain_name}_{self.cfg.task_name}")
        time_step = train_env.reset()
        prev_reset_state = {'state':time_step.observation,'physics':time_step.physics,'action':time_step.action}
        total_metrics = None
        fps_start_time = time.time()
        z, logpi = self.agent._model.hierarchical_actor()
        
        for t in tqdm(range(0, int(self.cfg.num_train_steps))):
            if t % self.cfg.eval_every_steps == 0:
                eval_dict = self.eval(t)
                self.eval_logger.log_tabular('timestep', t)
                for key in eval_dict.keys():
                    self.eval_logger.log_tabular(key, eval_dict[key])
                self.eval_logger.dump_tabular()

            if t<=self.cfg.warm_start_timesteps:
                if time_step.last():
                    time_step = train_env.reset()

            if t>self.cfg.warm_start_timesteps and (t)%self.cfg.horizon == 0:
                obs = torch.tensor(
                    time_step.observation.reshape(1, -1),
                    device=self.agent.device,
                    dtype=torch.float32,
                )
                  
                action = self.agent._model.actor(obs,z,0.1).mean
                Fs = self.agent._model._forward_map(obs, z, action)
                terminal_Q = (Fs[0]*self.agent.unnorm_z_inf).sum(-1)* (self.agent_cfg.train.discount**(self.cfg.horizon))
                cum_reward+=terminal_Q
                returns.append(cum_reward.detach())
                
                # returns.append(cum_reward)
                zs.append(z)
                log_probs.append(logpi)
                cum_reward = 0
                z, logpi = self.agent._model.hierarchical_actor() 
                # Reset the state to the prior initial state of the environment used with the same z
                physics_to_reset = prev_reset_state['physics']
                action_to_reset = prev_reset_state['action']
                with train_env._physics.reset_context():
                    train_env._physics.set_state(physics_to_reset)
                    train_env._physics.set_control(action_to_reset)
                mujoco.mj_forward(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_fwdPosition(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_sensorVel(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_subtreeVel(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member        
                # import ipdb;ipdb.set_trace()
                time_step = train_env.step(action_to_reset)

            if t>self.cfg.warm_start_timesteps and t % (self.cfg.horizon*self.cfg.num_trajs_per_state) ==0 :
                

                if len(reset_buffer['states'])>0 and np.random.uniform()<self.cfg.initial_state_sample_prob:
                    time_step = train_env.reset()
                    prev_reset_state = {'state':time_step.observation,'physics':time_step.physics,'action':time_step.action}
                else:
                    idx = np.random.randint(len(reset_buffer['states']))
                    physics_to_reset =  reset_buffer["physics"][idx]
                    action_to_reset = reset_buffer["actions"][idx]
                    prev_reset_state = {'state':reset_buffer["states"][idx],'physics':reset_buffer["physics"][idx],'action':reset_buffer["actions"][idx]}
                    with train_env._physics.reset_context():
                        train_env._physics.set_state(physics_to_reset)
                        train_env._physics.set_control(action_to_reset)
                    mujoco.mj_forward(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member
                    mujoco.mj_fwdPosition(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member
                    mujoco.mj_sensorVel(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member
                    mujoco.mj_subtreeVel(train_env._physics.model.ptr, train_env._physics.data.ptr)  # pylint: disable=no-member        
                    # observation = train_env._task.get_observation(train_env._physics)
                    # reward = 0.0     # or call train_env._task.get_reward(...) if you really want it
                    # discount = self.cfg.train.discount   # or call train_env._task.get_discount(...)
                    # time_step = ExtendedTimeStep(
                    #     step_type=StepType.FIRST,   # or MID, depending on what you are doing
                    #     reward=reward,
                    #     discount=discount,
                    #     observation=observation
                    # )
                                        
                    time_step = train_env.step(action_to_reset)


            with torch.no_grad(), eval_mode(self.agent._model):
                obs = torch.tensor(
                    time_step.observation.reshape(1, -1),
                    device=self.agent.device,
                    dtype=torch.float32,
                )
                # action = self.agent._model.actor(obs,z,0.1).sample().cpu().numpy()
                action = self.agent._model.actor(obs,z,0.1).mean.cpu().numpy()
                # action = self.agent.act(obs=obs, mean=True).cpu().numpy()
                next_time_step = train_env.step(action)
                reset_buffer["states"].append(time_step.observation)
                reset_buffer["actions"].append(next_time_step.action)
                reset_buffer["physics"].append(time_step.physics)
                 
                # import ipdb;ipdb.set_trace()
                reward = next_time_step.reward
                cum_reward+=reward* (self.agent_cfg.train.discount**((t)%self.cfg.horizon))
                time_step=next_time_step


            metrics = None

            if t>self.cfg.warm_start_timesteps and t%(self.cfg.horizon*self.cfg.num_zs) == 0:
                # import ipdb;ipdb.set_trace()

                metrics = self.agent.update(log_probs, returns, num_trajs_per_state = self.cfg.num_trajs_per_state)
                # import ipdb;ipdb.set_trace()
                log_probs = []
                returns = []
                zs = []
                # we need to copy tensors returned by a cudagraph module
                m_dict = {k: metrics[k].clone() for k in metrics.keys()}
             
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
                self.train_logger.log_tabular('timestep', t)
                for key in m_dict.keys():
                    self.train_logger.log_tabular(key, m_dict[key])
                self.train_logger.dump_tabular()

            # if (t+1) % self.cfg.checkpoint_every_steps == 0:
            #     self.agent.save(str(self.work_dir / "checkpoint"))
        # self.agent.save(str(self.work_dir / "checkpoint"))
        return


    def eval(self, t):
        for task in self.cfg.eval_tasks:
            eval_env = dmc.make(f"{self.cfg.domain_name}_{task}")
            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            # import ipdb;ipdb.set_trace()
            for ep in range(num_ep):
                time_step = eval_env.reset()
                while not time_step.last():
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs = torch.tensor(
                            time_step.observation.reshape(1, -1),
                            # time_step.observation["observations"].reshape(1, -1),
                            device=self.agent.device,
                            dtype=torch.float32,
                        )
                        action = self.agent.act(obs=obs, mean=True).cpu().numpy()
                    time_step = eval_env.step(action)
                    total_reward[ep] += time_step.reward
            m_dict = {
                "reward": np.mean(total_reward),
                "reward#std": np.std(total_reward),
            }
            if self.cfg.use_wandb:
                wandb.log(
                    {f"{task}/{k}": v for k, v in m_dict.items()},
                    step=t,
                )
            m_dict["task"] = task
            print(m_dict)
            return m_dict

    def reward_inference_with_projection(self, task) -> torch.Tensor:
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
        unnorm_z, z = self.agent._model.reward_inference_with_projection(
            next_obs=batch["next"]["observation"],
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return unnorm_z, z


    def reward_inference_with_projection_given_samples(self, task, batch) -> torch.Tensor:
        env = dmc.make(f"{self.cfg.domain_name}_{task}")
        num_samples = self.cfg.num_inference_samples
        # batch = self.replay_buffer["train"].sample(num_samples)
        rewards = []
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(batch["next_physics"][i].reshape(-1))
                env._physics.set_control(batch["action"][i].reshape(-1))
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            rewards.append(env._task.get_reward(env._physics))
        rewards = np.array(rewards).reshape(-1, 1)
        unnorm_z, z = self.agent._model.reward_inference_with_projection(
            next_obs=torch.tensor(batch["next_observation"].reshape(num_samples,-1),dtype=torch.float32, device=self.agent.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        return unnorm_z, z

    def reward_inference(self, task) -> torch.Tensor:
        # import ipdb;ipdb.set_trace()
        set_seed_everywhere(0)
        env = dmc.make(f"{self.cfg.domain_name}_{task}")
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)
        rewards = []
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(batch["next"]["physics"][i])
                env._physics.set_control(batch["action"][i])
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            rewards.append(env._task.get_reward(env._physics))
        rewards = np.array(rewards).reshape(-1, 1)
        z = self.agent._model.reward_inference(
            next_obs=torch.tensor(batch["next_observation"],dtype=torch.float32, device=self.agent.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
        )
        # import ipdb;ipdb.set_trace()
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
        residual_critic = config.residual_critic,
        zero_shot_initialization = config.zero_shot_initialization,
        actor_lr = config.actor_lr,
        expl_logstd = config.expl_logstd

    )

    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()
