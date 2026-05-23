#!/usr/bin/env python3
"""Aggregate final evaluation returns across random seeds.

By default this script scans the ICLR seed run, infers the completed timestep as
the largest final timestep found in the run, and includes only environments
where every discovered seed has an evaluation row at that timestep.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RUN_DIR = Path(
    "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/"
    "results/ICLR_Seeds/mr_train_dmc_transformer_h5_singlegpu"
)
SEED_RE = re.compile(r"_seed_(\d+)$")


@dataclass(frozen=True)
class SeedRun:
    env: str
    seed: int
    path: Path
    rows: list[dict[str, str]]

    @property
    def last_timestep(self) -> int:
        return int(float(self.rows[-1]["timestep"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Directory containing env_/..._seed_N/eval_log.txt folders. Default: {DEFAULT_RUN_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write CSV outputs. Default: <run-dir>/aggregated_seed_returns",
    )
    parser.add_argument(
        "--completion-timestep",
        type=int,
        default=None,
        help="Timestep required for an env to count as complete. Default: max final timestep found.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Optional env names to consider, e.g. cheetah walker pointmass.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed ids to require. Default: all discovered seeds per env.",
    )
    return parser.parse_args()


def env_name(env_dir: Path) -> str:
    return env_dir.name[:-1] if env_dir.name.endswith("_") else env_dir.name


def read_eval_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def discover_seed_runs(run_dir: Path, env_filter: set[str] | None) -> list[SeedRun]:
    seed_runs: list[SeedRun] = []
    for env_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        env = env_name(env_dir)
        if env_filter is not None and env not in env_filter:
            continue
        for seed_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            match = SEED_RE.search(seed_dir.name)
            if match is None:
                continue
            eval_log = seed_dir / "eval_log.txt"
            if not eval_log.exists():
                continue
            seed_runs.append(
                SeedRun(env=env, seed=int(match.group(1)), path=eval_log, rows=read_eval_log(eval_log))
            )
    return seed_runs


def metric_columns(rows: Iterable[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    metrics: list[str] = []
    for row in rows:
        for key in row:
            if key == "timestep" or key.endswith("#std") or key in seen:
                continue
            seen.add(key)
            metrics.append(key)
    return metrics


def row_at_timestep(seed_run: SeedRun, timestep: int) -> dict[str, str] | None:
    for row in seed_run.rows:
        if int(float(row["timestep"])) == timestep:
            return row
    return None


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def stderr(values: list[float]) -> float:
    return stdev(values) / (len(values) ** 0.5) if values else 0.0


def write_per_seed_csv(path: Path, rows: list[dict[str, object]], metrics: list[str]) -> None:
    fieldnames = ["env", "seed", "timestep", *metrics]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, per_seed_rows: list[dict[str, object]], metrics: list[str]) -> None:
    summary_rows: list[dict[str, object]] = []
    envs = sorted({str(row["env"]) for row in per_seed_rows})
    for env in envs:
        env_rows = [row for row in per_seed_rows if row["env"] == env]
        for metric in metrics:
            values = [float(row[metric]) for row in env_rows if row.get(metric) not in (None, "")]
            if not values:
                continue
            summary_rows.append(
                {
                    "env": env,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean(values),
                    "std": stdev(values),
                    "stderr": stderr(values),
                    "min": min(values),
                    "max": max(values),
                }
            )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env", "metric", "n", "mean", "std", "stderr", "min", "max"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def task_metric_columns(metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if metric.endswith("_reward") and metric != "average_reward"]


def task_name(metric: str) -> str:
    return metric.removesuffix("_reward")


def build_per_seed_task_rows(
    per_seed_rows: list[dict[str, object]], task_metrics: list[str]
) -> list[dict[str, object]]:
    task_rows: list[dict[str, object]] = []
    for row in per_seed_rows:
        for metric in task_metrics:
            if row.get(metric) in (None, ""):
                continue
            task_rows.append(
                {
                    "env": row["env"],
                    "seed": row["seed"],
                    "timestep": row["timestep"],
                    "task": task_name(metric),
                    "metric": metric,
                    "return": row[metric],
                }
            )
    return task_rows


def write_per_seed_task_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "seed", "timestep", "task", "metric", "return"])
        writer.writeheader()
        writer.writerows(rows)


def write_per_task_summary_csv(path: Path, per_seed_task_rows: list[dict[str, object]]) -> None:
    summary_rows: list[dict[str, object]] = []
    groups = sorted({(str(row["env"]), str(row["task"]), str(row["metric"])) for row in per_seed_task_rows})
    for env, task, metric in groups:
        values = [
            float(row["return"])
            for row in per_seed_task_rows
            if row["env"] == env and row["task"] == task and row["metric"] == metric
        ]
        summary_rows.append(
            {
                "env": env,
                "task": task,
                "metric": metric,
                "n": len(values),
                "mean": mean(values),
                "std": stdev(values),
                "stderr": stderr(values),
                "min": min(values),
                "max": max(values),
            }
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["env", "task", "metric", "n", "mean", "std", "stderr", "min", "max"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def print_average_reward_table(per_seed_rows: list[dict[str, object]]) -> None:
    average_rows = [row for row in per_seed_rows if row.get("average_reward") not in (None, "")]
    if not average_rows:
        return

    print("\nFinal average_reward by env")
    print("env,n,mean,std,stderr,seeds")
    for env in sorted({str(row["env"]) for row in average_rows}):
        env_rows = [row for row in average_rows if row["env"] == env]
        values = [float(row["average_reward"]) for row in env_rows]
        seeds = " ".join(str(row["seed"]) for row in sorted(env_rows, key=lambda r: int(r["seed"])))
        print(
            f"{env},{len(values)},{mean(values):.6f},{stdev(values):.6f},"
            f"{stderr(values):.6f},{seeds}"
        )


def print_per_task_table(per_seed_task_rows: list[dict[str, object]]) -> None:
    if not per_seed_task_rows:
        return

    print("\nFinal per-task returns by env")
    print("env,task,n,mean,std,stderr,seeds")
    groups = sorted({(str(row["env"]), str(row["task"])) for row in per_seed_task_rows})
    for env, task in groups:
        rows = [row for row in per_seed_task_rows if row["env"] == env and row["task"] == task]
        values = [float(row["return"]) for row in rows]
        seeds = " ".join(str(row["seed"]) for row in sorted(rows, key=lambda r: int(r["seed"])))
        print(
            f"{env},{task},{len(values)},{mean(values):.6f},{stdev(values):.6f},"
            f"{stderr(values):.6f},{seeds}"
        )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "aggregated_seed_returns").expanduser().resolve()
    env_filter = set(args.envs) if args.envs else None
    required_seeds = set(args.seeds) if args.seeds else None

    seed_runs = discover_seed_runs(run_dir, env_filter)
    if not seed_runs:
        raise SystemExit(f"No eval_log.txt files found under {run_dir}")

    completion_timestep = args.completion_timestep or max(run.last_timestep for run in seed_runs)
    by_env: dict[str, list[SeedRun]] = {}
    for run in seed_runs:
        by_env.setdefault(run.env, []).append(run)

    per_seed_rows: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    completed_envs: list[str] = []
    for env, runs in sorted(by_env.items()):
        runs_by_seed = {run.seed: run for run in runs}
        seeds_to_check = sorted(required_seeds or runs_by_seed)
        missing_seeds = [seed for seed in seeds_to_check if seed not in runs_by_seed]
        if missing_seeds:
            skipped[env] = f"missing seeds {missing_seeds}"
            continue

        rows_for_env: list[dict[str, object]] = []
        for seed in seeds_to_check:
            run = runs_by_seed[seed]
            row = row_at_timestep(run, completion_timestep)
            if row is None:
                skipped[env] = f"seed {seed} stops at {run.last_timestep}"
                rows_for_env = []
                break
            metrics = metric_columns([row])
            rows_for_env.append(
                {
                    "env": env,
                    "seed": seed,
                    "timestep": completion_timestep,
                    **{metric: float(row[metric]) for metric in metrics},
                }
            )
        if rows_for_env:
            completed_envs.append(env)
            per_seed_rows.extend(rows_for_env)

    if not per_seed_rows:
        raise SystemExit(f"No complete envs found at timestep {completion_timestep}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(
        {key: "" for key in row.keys() if key not in {"env", "seed"}} for row in per_seed_rows
    )
    per_seed_csv = output_dir / "per_seed_returns.csv"
    summary_csv = output_dir / "returns_summary.csv"
    per_seed_task_csv = output_dir / "per_seed_task_returns.csv"
    per_task_summary_csv = output_dir / "per_task_returns_summary.csv"
    task_metrics = task_metric_columns(metrics)
    per_seed_task_rows = build_per_seed_task_rows(per_seed_rows, task_metrics)
    write_per_seed_csv(per_seed_csv, per_seed_rows, metrics)
    write_summary_csv(summary_csv, per_seed_rows, metrics)
    write_per_seed_task_csv(per_seed_task_csv, per_seed_task_rows)
    write_per_task_summary_csv(per_task_summary_csv, per_seed_task_rows)

    print(f"Run dir: {run_dir}")
    print(f"Completion timestep: {completion_timestep}")
    print(f"Completed envs: {', '.join(completed_envs)}")
    if skipped:
        print("Skipped envs:")
        for env, reason in sorted(skipped.items()):
            print(f"  {env}: {reason}")
    print(f"Wrote: {per_seed_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {per_seed_task_csv}")
    print(f"Wrote: {per_task_summary_csv}")
    print_average_reward_table(per_seed_rows)
    print_per_task_table(per_seed_task_rows)


if __name__ == "__main__":
    main()
