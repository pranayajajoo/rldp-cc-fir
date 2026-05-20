import itertools
import os
import shlex
import subprocess
import tempfile


jobs_per_node = 1

DOMAINS = ["pointmass", "walker", "cheetah", "quadruped"]

SBATCH_OPTIONS = {
    "job_name": "MR_ZOL_SWEEP",
    "account": "rrg-whitem",
    "time": "24:00:00",
    "nodes": 1,
    "cpus_per_task": "2",
    "mem": "40G",
    "gres": "gpu:nvidia_h100_80gb_hbm3_3g.40gb:1",
    "venv_activate": "/home/pranayaj/projects/def-whitem/pranayaj/scratch/envs/motivo/bin/activate",
    "modules": "mujoco python",
    "mail_user": os.environ.get("SLURM_MAIL_USER", "jajoo@ualberta.ca"),
    "mail_type": os.environ.get("SLURM_MAIL_TYPE", "ALL"),
}

CODEBASE_DIR = "/home/pranayaj/projects/def-whitem/pranayaj/projects/mr_zsrl"
SCRIPT_PATH = "examples/mr_zol_inference_dmc.py"
DATASET_ROOT = "/home/pranayaj/projects/def-whitem/pranayaj/projects/exorl/datasets"
CHECKPOINT_ROOT = (
    "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/results/"
    "ICLR_Seeds/mr_train_dmc"
)
OUTPUT_DIR_BASE = (
    "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/results/"
    "ICLR_Seeds/mr_train_dmc_zol_small_sweep"
)

# Small paper-inspired sweep to test whether ZOL can improve MR latents without
# paying for the full 72-config sweep from the paper.
RUN_ARGS = {
    "--seed": [1],
    "--num_inference_samples": [50_000],
    "--num_eval_episodes": [10],
    "--zol_lr": [5e-4, 1e-4],
    "--zol_steps": [100],
    "--zol_n_mu": [256],
    "--zol_chi2_coef": [0.0, 0.001],
    "--zol_trust_l2_coef": [0.02],
    "--zol_weight_clip": [100.0],
    "--zol_weight_temp": [2.0],
    "--zol_mu_reward_top_frac": [0.05],
}


def generate_combinations(hyperparams):
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def config_label(args):
    return (
        f"lr_{args['--zol_lr']}"
        f"_steps_{args['--zol_steps']}"
        f"_chi2_{args['--zol_chi2_coef']}"
        f"_trust_{args['--zol_trust_l2_coef']}"
        f"_clip_{args['--zol_weight_clip']}"
        f"_nmu_{args['--zol_n_mu']}"
    ).replace(".", "p")


def construct_command(domain, args):
    label = config_label(args)
    cmd_parts = [
        "python",
        SCRIPT_PATH,
        "--dataset_root",
        DATASET_ROOT,
        "--checkpoint_root",
        CHECKPOINT_ROOT,
        "--output_dir",
        f"{OUTPUT_DIR_BASE}/{label}/{domain}",
        "--domains",
        domain,
        "--tasks",
        "all",
        "--device",
        "cuda",
    ]
    for arg, value in args.items():
        cmd_parts.extend([arg, str(value)])
    quoted_cmd = " ".join(shlex.quote(part) for part in cmd_parts)
    return f"cd {shlex.quote(CODEBASE_DIR)} && {quoted_cmd}"


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def submit_chunk(chunk, chunk_idx):
    job_name = f"{SBATCH_OPTIONS['job_name']}_chunk_{chunk_idx}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp_script:
        tmp_script.write("#!/bin/bash\n")
        tmp_script.write(f"#SBATCH --job-name={job_name}\n")
        tmp_script.write(f"#SBATCH --time={SBATCH_OPTIONS['time']}\n")
        tmp_script.write(f"#SBATCH --nodes={SBATCH_OPTIONS['nodes']}\n")
        tmp_script.write(f"#SBATCH --cpus-per-task={SBATCH_OPTIONS['cpus_per_task']}\n")
        tmp_script.write(f"#SBATCH --mem={SBATCH_OPTIONS['mem']}\n")
        tmp_script.write(f"#SBATCH --gres={SBATCH_OPTIONS['gres']}\n")
        tmp_script.write(f"#SBATCH --account={SBATCH_OPTIONS['account']}\n")
        if SBATCH_OPTIONS.get("mail_user"):
            tmp_script.write(f"#SBATCH --mail-user={SBATCH_OPTIONS['mail_user']}\n")
            tmp_script.write(f"#SBATCH --mail-type={SBATCH_OPTIONS.get('mail_type', 'ALL')}\n")
        tmp_script.write("#SBATCH --output=logs/%x_%j.out\n\n")
        tmp_script.write("set -euo pipefail\n")
        tmp_script.write(f"source {shlex.quote(SBATCH_OPTIONS['venv_activate'])}\n")
        tmp_script.write(f"module load {SBATCH_OPTIONS['modules']}\n")
        tmp_script.write('export PYTHONPATH="$PYTHONPATH:."\n')
        tmp_script.write("export MUJOCO_GL=osmesa\n\n")
        tmp_script.write(
            "python -c \"import torch; "
            "p=torch.cuda.get_device_properties(0); "
            "print('GPU:', torch.cuda.get_device_name(0)); "
            "print('VRAM(GB):', p.total_memory/1024**3)\"\n\n"
        )
        for domain, args in chunk:
            command = construct_command(domain, args)
            tmp_script.write(f"{command} &\n")
            tmp_script.write("sleep 60\n")
        tmp_script.write("\nwait\n")

    os.chmod(tmp_script.name, 0o755)
    sbatch_command = ["sbatch", tmp_script.name]
    print("Submitting job chunk:", " ".join(sbatch_command))
    try:
        result = subprocess.run(
            sbatch_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        print("Job submitted successfully:", result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        print("Error submitting job:", exc.stderr)
    finally:
        os.remove(tmp_script.name)


def main():
    os.makedirs("logs", exist_ok=True)
    jobs = [
        (domain, args)
        for args in generate_combinations(RUN_ARGS)
        for domain in DOMAINS
    ]
    chunks = list(chunk_list(jobs, jobs_per_node))
    print(f"Total individual jobs: {len(jobs)}")
    print(f"Number of chunks to submit: {len(chunks)}")
    for chunk_idx, chunk in enumerate(chunks):
        submit_chunk(chunk, chunk_idx)


if __name__ == "__main__":
    main()
