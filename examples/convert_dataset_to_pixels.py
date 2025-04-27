import os
import platform
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np
import dataclasses
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.mr_sf import FBAgent, FBAgentConfig
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
if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


from absl import app, flags
from pathlib import Path
import numpy as np
from pickle import HIGHEST_PROTOCOL
import torch
from tqdm import tqdm
# from url_benchmark import dmc
# from url_benchmark.in_memory_replay_buffer import ReplayBuffer


FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'walker', '')
flags.DEFINE_string('task', 'run', '')
flags.DEFINE_string('method', 'rnd', '')
flags.DEFINE_string('save_path', '/work/09313/hsikchi/vista/work/exorl', '')
flags.DEFINE_integer('num_episodes', 5000, '')
flags.DEFINE_integer('use_pixels', 0, '')
flags.DEFINE_integer('image_wh', 64, '')




def main(_):
    
    env = FLAGS.env
    method = FLAGS.method
    task = FLAGS.task
    num_episodes = FLAGS.num_episodes
    path = Path(f"{FLAGS.save_path}/datasets/{env}/{method}/buffer/")
    train_env = dmc.make(f'{env}_{task}')
    
    storage = {
        "observation": [],
        "pixel": [],
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
        episode_pixels = []
        for j in range(len(data["physics"][:-1])):
            with train_env.physics.reset_context():
                train_env.physics.set_state(data["physics"][:-1][j])
            camera_id = dict(quadruped=2).get(env, 0)
            pixel = train_env.physics.render(height=FLAGS.image_wh, width=FLAGS.image_wh, camera_id=camera_id)
            episode_pixels.append(pixel.transpose(2, 0, 1))
        episode_pixels = np.stack(episode_pixels, axis=0)
        storage["pixel"].append(episode_pixels.astype(np.uint8))
    with Path(f"/work/09313/hsikchi/vista/work/exorl/datasets/{env}/{method}/pixel64.pt").open('wb') as f:
        torch.save(storage, f, pickle_protocol=HIGHEST_PROTOCOL)

    # for k in storage:
    #     if k == "next":
    #         for k1 in storage[k]:
    #             storage[k][k1] = np.concat(storage[k][k1])
    #     else:
    #         storage[k] = np.concat(storage[k])
    # storage["next_observation_hash"]= np.random.permutation(np.arange(0,len(storage["observation"]))).reshape(-1,1)
    # if not FLAGS.use_pixels:
    #     file_name = 'replay'
    # else:
    #     file_name = f'replay_pixel{FLAGS.image_wh}'
    # with Path(f"{FLAGS.save_path}/datasets/{env}/{method}/{file_name}.pt").open('wb') as f:
    #     torch.save(replay_loader, f, pickle_protocol=HIGHEST_PROTOCOL)
    # return storage
    
    
    # replay_loader = ReplayBuffer(max_episodes=FLAGS.num_episodes, discount=0.99, future=0.99)
    # replay_loader.load(train_env, buffer_dir, relabel=True)
    # if FLAGS.use_pixels:
    #     replay_loader._batch_names.add('pixel')
    #     replay_loader._storage['pixel'] = np.zeros((*replay_loader._storage['action'].shape[:2], 3, FLAGS.image_wh, FLAGS.image_wh), dtype=np.uint8)
    #     for i in tqdm(range(len(replay_loader))):
    #         for j in range(replay_loader._storage['pixel'][i].shape[0]):
    #             with train_env.physics.reset_context():
    #                 train_env.physics.set_state(replay_loader._storage['physics'][i][j])
    #             camera_id = dict(quadruped=2).get(env, 0)
    #             pixel = train_env.physics.render(height=FLAGS.image_wh, width=FLAGS.image_wh, camera_id=camera_id)
    #             replay_loader._storage['pixel'][i][j] = pixel.transpose(2, 0, 1)
    # if not FLAGS.use_pixels:
    #     file_name = 'replay'
    # else:
    #     file_name = f'replay_pixel{FLAGS.image_wh}'
    # with Path(f"{FLAGS.save_path}/datasets/{env}/{method}/{file_name}.pt").open('wb') as f:
    #     torch.save(replay_loader, f, pickle_protocol=HIGHEST_PROTOCOL)


if __name__ == '__main__':
    app.run(main)

# def load_data(dataset_path, expl_agent, domain_name, num_episodes=1):
#     path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
#     print(f"Data path: {path}")
#     storage = {
#         "observation": [],
#         "action": [],
#         "physics": [],
#         "next": {"observation": [], "terminated": [], "physics": []},
#     }
#     files = list(path.glob("*.npz"))
#     num_episodes = min(num_episodes, len(files))
#     for i in tqdm(range(num_episodes)):
#         f = files[i]
#         data = np.load(str(f))
#         storage["observation"].append(data["observation"][:-1].astype(np.float32))
#         storage["action"].append(data["action"][1:].astype(np.float32))
#         storage["next"]["observation"].append(data["observation"][1:].astype(np.float32))
#         storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=np.bool))
#         storage["physics"].append(data["physics"][:-1])
#         storage["next"]["physics"].append(data["physics"][1:])

#     for k in storage:
#         if k == "next":
#             for k1 in storage[k]:
#                 storage[k][k1] = np.concat(storage[k][k1])
#         else:
#             storage[k] = np.concat(storage[k])
#     storage["next_observation_hash"]= np.random.permutation(np.arange(0,len(storage["observation"]))).reshape(-1,1)
#     return storage