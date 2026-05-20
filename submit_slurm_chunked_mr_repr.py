import itertools
import os
import shlex
import subprocess
import tempfile


jobs_per_node = 2  # Fir: 1 job per 40GB H100 MIG slice

ALL_TASKS = {
    "walker": ["walk", "run", "stand", "flip"],
    "cheetah": ["walk", "run", "walk_backward", "run_backward"],
    "walker_extended": [
        "run_fw_speed1.5",
        "run_fw_speed3.0",
        "run_fw_speed4.5",
        "run_fw_speed6.0",
        "run_fw_speed8.0",
        "run_fw_speed10.0",
        "spin_fw_speed15.0",
        "spin_fw_speed30.0",
        "spin_fw_speed45.0",
        "spin_fw_speed60.0",
        "crawl_fw_speed1.0",
        "crawl_fw_speed4.0",
        "crawl_fw_speed7.0",
        "crawl_fw_speed14.0",
        "run_bw_speed1.5",
        "run_bw_speed3.0",
        "run_bw_speed4.5",
        "run_bw_speed8.0",
        "run_bw_speed6.0",
        "run_bw_speed8.0",
        "run_bw_speed10.0",
        "spin_bw_speed15.0",
        "spin_bw_speed30.0",
        "spin_bw_speed45.0",
        "spin_bw_speed60.0",
        "crawl_bw_speed1.0",
        "crawl_bw_speed4.0",
        "crawl_bw_speed7.0",
        "crawl_bw_speed14.0",
    ],
    "pointmass": [
        "reach_top_left",
        "reach_top_right",
        "reach_bottom_right",
        "reach_bottom_left",
        # "loop",
        # "square",
        # "fast_slow",
    ],
    "quadruped": ["jump", "walk", "run", "stand"],
}


SBATCH_OPTIONS = {
    # ---- Compute Canada (Fir) Slurm settings ----
    "job_name": "MR_REPR",
    "account": "rrg-whitem",
    "time": "48:00:00",
    "nodes": 1,
    "cpus_per_task": "2",
    "mem": "40G",
    # Request one 40GB H100 MIG slice on Fir (from `sinfo -o "%P %G"`).
    "gres": "gpu:nvidia_h100_80gb_hbm3_3g.40gb:1",
    # ---- Required environment setup before running python ----
    "venv_activate": "/home/pranayaj/projects/def-whitem/pranayaj/scratch/envs/motivo/bin/activate",
    "modules": "mujoco python",
    "output": "logs/%x_%j.out",
    "mail_user": os.environ.get("SLURM_MAIL_USER", "jajoo@ualberta.ca"),
    "mail_type": os.environ.get("SLURM_MAIL_TYPE", "ALL"),
}


EXPERIMENT_NAME = "mr_train_dmc_repr"
DATASET_ROOT = "/home/pranayaj/projects/def-whitem/pranayaj/projects/exorl/datasets"
WORK_DIR_BASE = "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/results/ICLR_Seeds/" + EXPERIMENT_NAME
SCRIPT_PATH = "examples/mr_train_dmc_repr.py"
CODEBASE_DIR = "/home/pranayaj/projects/def-whitem/pranayaj/projects/mr_zsrl"


MANDATORY_ARGS = {
    "--dataset_root": DATASET_ROOT,
    "--work_dir": WORK_DIR_BASE,
}


HYPERPARAMETERS = {
    "--seed": [1, 2, 3, 4, 5],
    "--representation_steps": [2000000],
    "--enc_horizon": [5],
    "--encoder_hidden_dim": [512],
    "--encoder_norm": [1],
    "--ortho_coef": [1.0],
}


def generate_hyperparameter_combinations(hyperparams):
    """Generates all combinations of hyperparameters."""
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def construct_command(domain, task, hyperparams):
    """
    Constructs the shell commands (joined by '&&') to set up the environment
    and run the training script for a single hyperparameter combination.
    """
    commands = []
    commands.append(f"cd {shlex.quote(CODEBASE_DIR)}")

    cmd = f"python {SCRIPT_PATH} --domain_name {shlex.quote(domain)} "

    for arg, value in MANDATORY_ARGS.items():
        if arg == "--work_dir":
            value = (
                f"{value}/{domain}_{task}/"
                f"rs_{hyperparams['--representation_steps']}_"
                f"eh_{hyperparams['--enc_horizon']}_"
                f"enorm_{hyperparams['--encoder_norm']}_"
                f"edim_{hyperparams['--encoder_hidden_dim']}_"
                f"ortho_loss_{hyperparams['--ortho_coef']}_"
                f"seed_{hyperparams['--seed']}"
            )
        if value is None:
            cmd += f" {arg}"
        else:
            cmd += f" {arg} {shlex.quote(str(value))}"

    for arg, value in hyperparams.items():
        if value is None:
            cmd += f" {arg}"
        else:
            cmd += f" {arg} {shlex.quote(str(value))}"

    print(cmd)
    commands.append(cmd)
    return " && ".join(commands)


def submit_chunk(chunk, sbatch_opts, chunk_idx):
    """
    For a given chunk of jobs (each job is (domain, task, hyperparams)),
    create one Slurm script that runs all jobs in parallel on a single node.
    """
    job_name = f"{sbatch_opts['job_name']}_chunk_{chunk_idx}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp_script:
        tmp_script.write("#!/bin/bash\n")
        tmp_script.write(f"#SBATCH --job-name={job_name}\n")
        tmp_script.write(f"#SBATCH --time={sbatch_opts['time']}\n")
        tmp_script.write(f"#SBATCH --nodes={sbatch_opts['nodes']}\n")
        tmp_script.write(f"#SBATCH --cpus-per-task={sbatch_opts['cpus_per_task']}\n")
        tmp_script.write(f"#SBATCH --mem={sbatch_opts['mem']}\n")
        tmp_script.write(f"#SBATCH --gres={sbatch_opts['gres']}\n")
        tmp_script.write(f"#SBATCH --account={sbatch_opts['account']}\n")
        if sbatch_opts.get("mail_user"):
            tmp_script.write(f"#SBATCH --mail-user={sbatch_opts['mail_user']}\n")
            tmp_script.write(f"#SBATCH --mail-type={sbatch_opts.get('mail_type', 'ALL')}\n")
        tmp_script.write(f"#SBATCH --output={sbatch_opts['output']}\n\n")

        tmp_script.write("set -euo pipefail\n")
        tmp_script.write(f"source {sbatch_opts['venv_activate']}\n")
        tmp_script.write(f"module load {sbatch_opts['modules']}\n")
        tmp_script.write('export PYTHONPATH="$PYTHONPATH:."\n')
        tmp_script.write("export MUJOCO_GL=osmesa\n\n")
        tmp_script.write(
            "python -c \"import torch; p=torch.cuda.get_device_properties(0); "
            "print('GPU:', torch.cuda.get_device_name(0)); "
            "print('VRAM(GB):', p.total_memory/1024**3)\"\n\n"
        )

        for domain, task, hyperparams in chunk:
            command = construct_command(domain, task, hyperparams)
            tmp_script.write(f"{command} &\n")
            tmp_script.write("sleep 120\n")

        tmp_script.write("\nwait\n")
        print(tmp_script)

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
    except subprocess.TimeoutExpired:
        print("Error: sbatch command timed out.")
    except subprocess.CalledProcessError as e:
        print("Error submitting job:", e.stderr)
    except Exception as e:
        print("Unexpected error:", str(e))
    finally:
        os.remove(tmp_script.name)


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    domains = ["pointmass", "walker", "cheetah", "quadruped"]

    os.makedirs("logs", exist_ok=True)
    os.makedirs(WORK_DIR_BASE, exist_ok=True)

    combinations = list(generate_hyperparameter_combinations(HYPERPARAMETERS))

    all_jobs = []
    for domain in domains:
        for hyperparams in combinations:
            all_jobs.append((domain, "", hyperparams))

    chunked_jobs = list(chunk_list(all_jobs, jobs_per_node))

    total_jobs = len(all_jobs)
    print(f"Total individual jobs: {total_jobs}")
    print(f"Number of chunks to submit (each chunk runs in parallel on one node): {len(chunked_jobs)}")

    for chunk_idx, chunk in enumerate(chunked_jobs):
        submit_chunk(chunk, SBATCH_OPTIONS, chunk_idx)


if __name__ == "__main__":
    main()
