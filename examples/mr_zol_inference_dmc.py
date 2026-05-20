from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dmc_tasks import dmc
from metamotivo.mr_sf import FBAgent
from metamotivo.nn_models import eval_mode


ALL_TASKS = {
    "walker": ["walk", "run", "stand", "flip"],
    "cheetah": ["walk", "run", "walk_backward", "run_backward"],
    "pointmass": [
        "reach_top_left",
        "reach_top_right",
        "reach_bottom_right",
        "reach_bottom_left",
    ],
    "quadruped": ["jump", "walk", "run", "stand"],
}


DEFAULT_DATASET_ROOT = "/home/pranayaj/projects/def-whitem/pranayaj/projects/exorl/datasets"
DEFAULT_CHECKPOINT_ROOT = (
    "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/results/"
    "ICLR_Seeds/mr_train_dmc"
)
DEFAULT_OUTPUT_DIR = (
    "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/results/"
    "ICLR_Seeds/mr_train_dmc_zol"
)


@dataclass
class ZOLConfig:
    lr: float = 3e-3
    num_steps: int = 500
    n_mu: int = 512
    early_stop_patience: int = 500
    early_stop_tol: float = 1e-8
    chi2_coef: float = 0.01
    trust_l2_coef: float = 0.001
    weight_clip: float = 20.0
    center_rewards: bool = True
    use_exp_weights: bool = True
    weight_temp: float = 2.0
    mu_reward_top_frac: float = 0.05
    self_normalized_obj: bool = True


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_task_seed(base_seed: int, domain: str, task: str) -> int:
    task_key = f"{domain}/{task}"
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(task_key))
    return base_seed + offset % 1_000_000


def default_checkpoint_path(root: str | Path, domain: str, seed: int) -> Path:
    return (
        Path(root)
        / f"{domain}_"
        / f"rs_2000000_eh_5_enorm_1_edim_512_ortho_loss_1.0_seed_{seed}"
        / "checkpoint"
    )


def load_agent_for_inference(checkpoint: str | Path, device: str) -> FBAgent:
    checkpoint = Path(checkpoint)
    with (checkpoint / "config.json").open() as f:
        config = json.load(f)
    config["model"]["device"] = device
    agent = FBAgent(**config)
    weights = safetensors.torch.load_file(str(checkpoint / "model" / "model.safetensors"), device=device)
    agent._model.load_state_dict(weights)
    agent._model.train(False)
    agent._model.requires_grad_(False)
    return agent


def sample_dataset_transitions(
    dataset_root: str | Path,
    domain: str,
    expl_agent: str,
    num_samples: int,
    seed: int,
    max_files: int | None,
) -> dict[str, np.ndarray]:
    buffer_dir = Path(dataset_root) / domain / expl_agent / "buffer"
    files = sorted(buffer_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {buffer_dir}")
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError("max_files removed all dataset files")

    rng = np.random.default_rng(seed)
    file_ids = rng.integers(0, len(files), size=num_samples)
    grouped: dict[int, list[int]] = {}
    for out_idx, file_id in enumerate(file_ids):
        grouped.setdefault(int(file_id), []).append(out_idx)

    observations: list[np.ndarray] = [None] * num_samples  # type: ignore[list-item]
    actions: list[np.ndarray] = [None] * num_samples  # type: ignore[list-item]
    physics: list[np.ndarray] = [None] * num_samples  # type: ignore[list-item]

    for file_id, out_indices in tqdm(grouped.items(), desc=f"sampling {domain}", leave=False):
        data = np.load(files[file_id])
        obs_arr = data["observation"]
        act_arr = data["action"]
        phys_arr = data["physics"]
        max_t = min(len(obs_arr), len(act_arr), len(phys_arr)) - 1
        if max_t <= 0:
            raise ValueError(f"Dataset file has no valid transitions: {files[file_id]}")
        transition_ids = rng.integers(1, max_t + 1, size=len(out_indices))
        for out_idx, transition_id in zip(out_indices, transition_ids):
            observations[out_idx] = obs_arr[transition_id].astype(np.float32)
            actions[out_idx] = act_arr[transition_id].astype(np.float32)
            physics[out_idx] = phys_arr[transition_id].astype(np.float32)

    return {
        "next_observation": np.stack(observations, axis=0),
        "action": np.stack(actions, axis=0),
        "next_physics": np.stack(physics, axis=0),
    }


def relabel_rewards(env, next_physics: np.ndarray, actions: np.ndarray) -> np.ndarray:
    rewards = np.zeros((next_physics.shape[0],), dtype=np.float32)
    for i in tqdm(range(next_physics.shape[0]), desc="relabel rewards", leave=False):
        with env._physics.reset_context():
            env._physics.set_state(next_physics[i])
            env._physics.set_control(actions[i])
        mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)
        mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)
        mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)
        mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)
        rewards[i] = env._task.get_reward(env._physics)
    return rewards


def actor_mean_action(agent: FBAgent, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    model = agent._model
    obs_norm = model._normalize(obs)
    dist = model._actor(obs_norm, z, model.cfg.actor_std)
    return dist.mean


def forward_features(agent: FBAgent, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    model = agent._model
    obs_norm = model._normalize(obs)
    features = model._forward_map(obs_norm, z, action)
    if features.ndim == 3:
        features = features.mean(dim=0)
    return features


def estimate_mu_pi_z(agent: FBAgent, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    model = agent._model
    if z.ndim == 1:
        z = z.unsqueeze(0)
    z_proj = model.project_z(z)
    z_batch = z_proj.expand(obs.shape[0], -1)
    action = actor_mean_action(agent, obs, z_batch)
    features = forward_features(agent, obs, z_batch, action)
    return features.mean(dim=0)


def zol_latent_search(
    agent: FBAgent,
    batch_obs: torch.Tensor,
    rewards: torch.Tensor,
    initial_z: torch.Tensor,
    cfg: ZOLConfig,
) -> torch.Tensor:
    model = agent._model
    device = agent.device
    gamma = float(agent.cfg.train.discount)
    batch_obs = batch_obs.to(device)
    rewards = rewards.to(device=device, dtype=torch.float32).flatten()

    with torch.no_grad():
        b_s = model.backward_map(batch_obs)

    n = batch_obs.shape[0]
    mu_count = min(cfg.n_mu, n)
    mu_idx = torch.randint(0, n, (mu_count,), device=device)
    mu_obs = batch_obs[mu_idx]
    if cfg.mu_reward_top_frac > 0:
        k = max(1, int(cfg.mu_reward_top_frac * n))
        top_idx = torch.topk(rewards, k=k, largest=True).indices
        m = min(top_idx.shape[0], max(1, cfg.n_mu // 4))
        pick = top_idx[torch.randint(0, top_idx.shape[0], (m,), device=device)]
        mu_obs = torch.cat([mu_obs, batch_obs[pick]], dim=0)

    initial_z = initial_z.detach().to(device).flatten()
    z0_proj = model.project_z(initial_z.unsqueeze(0)).squeeze(0).detach()

    def compute_weights(z_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_proj = model.project_z(z_eval.unsqueeze(0)).squeeze(0)
        mu_pi_z = estimate_mu_pi_z(agent, mu_obs, z_proj)
        mu_col = mu_pi_z.reshape(-1, 1)
        logit = (1.0 - gamma) * torch.matmul(b_s, mu_col).squeeze(1)
        if cfg.use_exp_weights:
            x = cfg.weight_temp * logit
            weights = torch.exp(x - x.max())
        else:
            weights = F.softplus(logit)
        weights = weights / (weights.mean() + 1e-8)
        if cfg.weight_clip is not None:
            weights = torch.clamp(weights, max=float(cfg.weight_clip))
            weights = weights / (weights.mean() + 1e-8)
        return weights, z_proj

    def loss_fn(z_eval: torch.Tensor) -> torch.Tensor:
        weights, z_proj = compute_weights(z_eval)
        local_rewards = rewards
        if cfg.center_rewards:
            local_rewards = local_rewards - local_rewards.mean()
        if cfg.self_normalized_obj:
            objective = torch.sum(weights * local_rewards) / (torch.sum(weights) + 1e-8)
        else:
            objective = torch.mean(weights * local_rewards)
        chi2 = torch.mean((weights - 1.0) ** 2)
        trust = torch.mean((z_proj - z0_proj) ** 2)
        objective = objective - cfg.chi2_coef * chi2 - cfg.trust_l2_coef * trust
        return -objective

    with torch.no_grad():
        best_loss = float(loss_fn(initial_z).item())
        best_z = initial_z.clone()

    z_opt = initial_z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=cfg.lr)
    patience = 0
    for _ in tqdm(range(cfg.num_steps), desc="zol search", leave=False):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(z_opt)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z_opt.copy_(model.project_z(z_opt.unsqueeze(0)).squeeze(0))
        current_loss = float(loss.item())
        if current_loss < best_loss - cfg.early_stop_tol:
            best_loss = current_loss
            best_z = z_opt.detach().clone()
            patience = 0
        else:
            patience += 1
        if patience >= cfg.early_stop_patience:
            break

    return model.project_z(best_z.unsqueeze(0)).squeeze(0).detach()


def evaluate_z(agent: FBAgent, domain: str, task: str, z: torch.Tensor, num_episodes: int) -> dict[str, float]:
    env = dmc.make(f"{domain}_{task}")
    returns = np.zeros((num_episodes,), dtype=np.float64)
    z_eval = z.reshape(1, -1).to(agent.device)
    for ep in tqdm(range(num_episodes), desc=f"eval {task}", leave=False):
        time_step = env.reset()
        while not time_step.last():
            with torch.no_grad(), eval_mode(agent._model):
                obs = torch.as_tensor(
                    time_step.observation.reshape(1, -1),
                    device=agent.device,
                    dtype=torch.float32,
                )
                action = agent.act(obs=obs, z=z_eval, mean=True).cpu().numpy()
            time_step = env.step(action)
            returns[ep] += float(time_step.reward)
    return {"mean": float(np.mean(returns)), "std": float(np.std(returns))}


def run_task(agent: FBAgent, args: argparse.Namespace, domain: str, task: str, out_dir: Path) -> dict:
    env = dmc.make(f"{domain}_{task}")
    samples = sample_dataset_transitions(
        dataset_root=args.dataset_root,
        domain=domain,
        expl_agent=args.dataset_expl_agent,
        num_samples=args.num_inference_samples,
        seed=stable_task_seed(args.seed, domain, task),
        max_files=args.max_dataset_files,
    )
    rewards_np = relabel_rewards(env, samples["next_physics"], samples["action"])
    batch_obs = torch.as_tensor(samples["next_observation"], device=agent.device, dtype=torch.float32)
    rewards = torch.as_tensor(rewards_np.reshape(-1, 1), device=agent.device, dtype=torch.float32)

    with torch.no_grad(), eval_mode(agent._model):
        z_base = agent._model.reward_inference(batch_obs, rewards).squeeze(0)
    z_zol = zol_latent_search(agent, batch_obs, rewards.flatten(), z_base, args.zol_config)

    task_dir = out_dir / domain / task
    task_dir.mkdir(parents=True, exist_ok=True)
    z_base_path = task_dir / "z_base.npy"
    z_zol_path = task_dir / "z_zol.npy"
    np.save(z_base_path, z_base.detach().cpu().numpy())
    np.save(z_zol_path, z_zol.detach().cpu().numpy())

    result = {
        "domain": domain,
        "task": task,
        "num_inference_samples": args.num_inference_samples,
        "reward_mean": float(rewards_np.mean()),
        "reward_std": float(rewards_np.std()),
        "z_base_path": str(z_base_path),
        "z_zol_path": str(z_zol_path),
        "z_base_norm": float(torch.linalg.norm(z_base).item()),
        "z_zol_norm": float(torch.linalg.norm(z_zol).item()),
        "z_delta_norm": float(torch.linalg.norm(z_zol - z_base).item()),
    }
    if args.num_eval_episodes > 0:
        result["baseline_eval"] = evaluate_z(agent, domain, task, z_base, args.num_eval_episodes)
        result["zol_eval"] = evaluate_z(agent, domain, task, z_zol, args.num_eval_episodes)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone ZOL-style latent inference for saved mr_train_dmc checkpoints."
    )
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset_expl_agent", default="rnd")
    parser.add_argument("--checkpoint_root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--domains", default="walker,cheetah,pointmass,quadruped")
    parser.add_argument("--tasks", default="all", help="'all' or comma-separated task names for one domain")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_inference_samples", type=int, default=50_000)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--max_dataset_files", type=int, default=None)
    parser.add_argument("--zol_lr", type=float, default=3e-3)
    parser.add_argument("--zol_steps", type=int, default=500)
    parser.add_argument("--zol_n_mu", type=int, default=512)
    parser.add_argument("--zol_chi2_coef", type=float, default=0.01)
    parser.add_argument("--zol_trust_l2_coef", type=float, default=0.001)
    parser.add_argument("--zol_weight_clip", type=float, default=20.0)
    parser.add_argument("--zol_weight_temp", type=float, default=2.0)
    parser.add_argument("--zol_mu_reward_top_frac", type=float, default=0.05)
    parser.add_argument("--no_center_rewards", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.zol_config = ZOLConfig(
        lr=args.zol_lr,
        num_steps=args.zol_steps,
        n_mu=args.zol_n_mu,
        chi2_coef=args.zol_chi2_coef,
        trust_l2_coef=args.zol_trust_l2_coef,
        weight_clip=args.zol_weight_clip,
        weight_temp=args.zol_weight_temp,
        mu_reward_top_frac=args.zol_mu_reward_top_frac,
        center_rewards=not args.no_center_rewards,
    )

    set_seed(args.seed)
    started_at = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"seed_{args.seed}_{started_at}"
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = parse_csv(args.domains)
    all_results = {
        "config": {
            **{k: v for k, v in vars(args).items() if k != "zol_config"},
            "zol_config": asdict(args.zol_config),
        },
        "results": [],
    }

    for domain in domains:
        checkpoint = default_checkpoint_path(args.checkpoint_root, domain, args.seed)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {domain}: {checkpoint}")
        print(f"\nLoading {domain} checkpoint: {checkpoint}")
        agent = load_agent_for_inference(checkpoint, args.device)

        tasks = ALL_TASKS[domain] if args.tasks == "all" else parse_csv(args.tasks)
        for task in tasks:
            print(f"\n[{domain}/{task}] inferring z")
            result = run_task(agent, args, domain, task, out_dir)
            all_results["results"].append(result)
            with (out_dir / "results.json").open("w") as f:
                json.dump(all_results, f, indent=2)
            print(json.dumps(result, indent=2))

    print(f"\nSaved ZOL inference outputs to {out_dir}")


if __name__ == "__main__":
    main()
