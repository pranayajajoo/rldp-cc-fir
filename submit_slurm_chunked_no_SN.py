import itertools
import subprocess
import shlex
import os
import tempfile
import math

jobs_per_node = 2  # CC/Fir: 1 job per 40GB MIG slice
ALL_TASKS = {
    "walker": ["walk", "run", "stand", "flip"],
    "cheetah": ["walk", "run", "walk_backward", "run_backward"],
    "walker_extended":[
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
        # "fast_slow"
    ],
    "quadruped": ["jump", "walk", "run", "stand"],
}

# Configuration for the Slurm jobs
SBATCH_OPTIONS = {
    # ---- Compute Canada (Fir) Slurm settings ----
    "job_name": "RLDP",                       # Base job name
    "account": "rrg-whitem",                # Slurm account
    # "partition": "gpubase_bygpu_b5",        # Fir GPU partition (adjust as needed)
    "time": "36:00:00",                     # Time limit (HH:MM:SS)
    # "time": "00:30:00",                     # Time limit (HH:MM:SS)
    "nodes": 1,                             # 1 node
    "cpus_per_task": "2",                   # CPU cores per task
    "mem": "40G",                           # CPU RAM per job
    # Request one 40GB H100 MIG slice on Fir (from `sinfo -o "%P %G"`)
    "gres": "gpu:nvidia_h100_80gb_hbm3_3g.40gb:1",
    # ---- Required environment setup before running python ----
    "venv_activate": "/home/pranayaj/projects/def-whitem/pranayaj/scratch/envs/motivo/bin/activate",
    "modules": "mujoco python",
    "output": "logs/%x_%j.out",
    "mail_user": os.environ.get("SLURM_MAIL_USER", "jajoo@ualberta.ca"),
    "mail_type": os.environ.get("SLURM_MAIL_TYPE", "ALL"),
}

# Path configurations
EXPERIMENT_NAME = "ICLR_Seeds"
DATASET_ROOT = "/home/pranayaj/projects/def-whitem/pranayaj/projects/exorl/datasets"
WORK_DIR_BASE = "/home/pranayaj/projects/def-whitem/pranayaj/results/mrzsrl/metamotivo/results/"+EXPERIMENT_NAME
SCRIPT_PATH = "examples/mr_train_dmc_no_SN.py"

# New Configurations
CODEBASE_DIR = "/home/pranayaj/projects/def-whitem/pranayaj/projects/mr_zsrl"

# Mandatory arguments
MANDATORY_ARGS = {
    "--dataset_root": DATASET_ROOT,
    "--work_dir": WORK_DIR_BASE,
    # "--no-zero-shot-initialization": None,
    # "--no-residual-critic": None,
    # "--warm_start_timesteps": 0,
    # "--wandb-pname": "fb_residual_z_runs",
    # "--wandb-name-prefix": "FB_residual_z",
    # "--wandb-gname": "fb_residual_z",
    # "--fb_type": "offline"
    # "--compile": None,
}

# Hyperparameters to sweep over
HYPERPARAMETERS = {
    # "--seed": [1,2,3,4],
    "--seed": [1,2],
    "--representation_steps": [2000000],
    # "--representation_steps": [2000],
    "--enc_horizon":[5],
    "--encoder_hidden_dim" : [512],
    "--encoder_norm" : [1],
    "--ortho_coef": [1.0],
    "--checkpoint_every_steps": [1_00000]
    # Add more hyperparameters if needed
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
    
    # Navigate to the codebase directory
    commands.append(f"cd {shlex.quote(CODEBASE_DIR)}")
    
    # Source the Conda setup script
    # commands.append(f"source {shlex.quote(CONDA_PATH)}")
    
    # Activate the Conda environment
    # commands.append(f"conda activate {shlex.quote(CONDA_ENV)}")

    # Export current path to python path
    # commands.append('export PYTHONPATH="${PYTHONPATH}:."')
    
    # Construct the Python command with arguments
    cmd = f"python {SCRIPT_PATH} --domain_name {shlex.quote(domain)} "
    
    # Add mandatory arguments
    for arg, value in MANDATORY_ARGS.items():
        if arg == "--work_dir":
            # Make work_dir unique per job by appending the domain, task, and seed
            value = f"{value}/{domain}_{task}/rs_{hyperparams['--representation_steps']}_eh_{hyperparams['--enc_horizon']}_enorm_{hyperparams['--encoder_norm']}_edim_{hyperparams['--encoder_hidden_dim']}_ortho_loss_{hyperparams['--ortho_coef']}_seed_{hyperparams['--seed']}"
        if value is None:
            cmd += f" {arg}"
        else:
            cmd += f" {arg} {shlex.quote(str(value))}"
    
    # Add hyperparameter arguments
    for arg, value in hyperparams.items():
        if value is None:
            cmd += f" {arg}"
        else:
            cmd += f" {arg} {shlex.quote(str(value))}"
    
    print(cmd)
    # import ipdb; ipdb.set_trace()

    # Combine environment setup and the python command
    commands.append(cmd)
    
    
    # Return a single shell line that sets up environment AND runs the python script
    return " && ".join(commands)


def submit_chunk(chunk, sbatch_opts, chunk_idx):
    """
    For a given chunk of jobs (each job is (domain, task, hyperparams)),
    create one Slurm script that runs all jobs in parallel on a single node.
    """
    # We'll build a short name that reflects the chunk index
    job_name = f"{sbatch_opts['job_name']}_chunk_{chunk_idx}"
    
    # Create a temporary batch script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_script:
        tmp_script.write("#!/bin/bash\n")
        tmp_script.write(f"#SBATCH --job-name={job_name}\n")
        tmp_script.write(f"#SBATCH --time={sbatch_opts['time']}\n")
        # tmp_script.write(f"#SBATCH --partition={sbatch_opts['partition']}\n")
        tmp_script.write(f"#SBATCH --nodes={sbatch_opts['nodes']}\n")
        tmp_script.write(f"#SBATCH --cpus-per-task={sbatch_opts['cpus_per_task']}\n")
        tmp_script.write(f"#SBATCH --mem={sbatch_opts['mem']}\n")
        tmp_script.write(f"#SBATCH --gres={sbatch_opts['gres']}\n")
        tmp_script.write(f"#SBATCH --account={sbatch_opts['account']}\n")
        if sbatch_opts.get("mail_user"):
            tmp_script.write(f"#SBATCH --mail-user={sbatch_opts['mail_user']}\n")
            tmp_script.write(f"#SBATCH --mail-type={sbatch_opts.get('mail_type', 'ALL')}\n")
        # Output file name includes %j for job ID
        tmp_script.write(f"#SBATCH --output=logs/%x_%j.out\n\n")  
        
        # ---- Compute Canada environment prolog ----
        tmp_script.write("set -euo pipefail\n")
        tmp_script.write(f"source {sbatch_opts['venv_activate']}\n")
        # tmp_script.write("module purge\n")
        tmp_script.write(f"module load {sbatch_opts['modules']}\n")
        tmp_script.write("export PYTHONPATH=\"$PYTHONPATH:.\"\n")
        tmp_script.write("export MUJOCO_GL=osmesa\n")
        tmp_script.write("\n")
        tmp_script.write("python -c \"import torch; p=torch.cuda.get_device_properties(0); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM(GB):', p.total_memory/1024**3)\"\n")
        tmp_script.write("\n")
        # Now, for each job in this chunk, we:
        #   1) Construct its command
        #   2) Run it in background (&)
        #   3) We'll collect logs separately by redirecting to a file if you like,
        #      or you can rely on the merged Slurm output above.
        for idx, (domain, task, hyperparams) in enumerate(chunk):
            command = construct_command(domain, task, hyperparams)
            
            # Redirect each task's stdout/stderr to separate files if desired
            # e.g. >> logs/task_{domain}_{task}_{seed}.log 2>&1
            #
            # For simplicity, just run in the background here
            tmp_script.write(f"{command} &\n")
            tmp_script.write("sleep 120\n")
        
        # Wait for all parallel tasks in this chunk to finish
        tmp_script.write("\nwait\n")

        print(tmp_script)
    
    # Make the script executable
    os.chmod(tmp_script.name, 0o755)
    
    # Submit the batch script
    sbatch_command = ["sbatch", tmp_script.name]
    
    print("Submitting job chunk:", " ".join(sbatch_command))
    
    try:
        result = subprocess.run(
            sbatch_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        print("Job submitted successfully:", result.stdout.strip())
    except subprocess.TimeoutExpired:
        print("Error: sbatch command timed out.")
    except subprocess.CalledProcessError as e:
        print("Error submitting job:", e.stderr)
    except Exception as e:
        print("Unexpected error:", str(e))
    finally:
        # Optionally remove the temporary script
        os.remove(tmp_script.name)


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    # Define the domains you want to train on
    domains = ["cheetah"]  # You can add more: "quadruped","pointmass"
    # domains = ['pointmass', "walker","cheetah","quadruped"]  # You can add more: "quadruped","pointmass"
    # domains = ["walker","cheetah","quadruped"]  # You can add more: "quadruped","pointmass"
    # domains = ['pointmass']
    # Create the logs and work directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs(WORK_DIR_BASE, exist_ok=True)
    
    # Generate all hyperparameter combinations
    combinations = list(generate_hyperparameter_combinations(HYPERPARAMETERS))
    
    # Build a master list of all (domain, task, hyperparams)
    all_jobs = []
    for domain in domains:
        tasks = ALL_TASKS[domain]
        for hyperparams in combinations:
            all_jobs.append((domain, "", hyperparams))
    
    # We now have a list of all individual jobs. We'll group them in chunks of jobs_per_node.
    chunked_jobs = list(chunk_list(all_jobs, jobs_per_node))
    
    total_jobs = len(all_jobs)
    print(f"Total individual jobs: {total_jobs}")
    print(f"Number of chunks to submit (each chunk runs in parallel on one node): {len(chunked_jobs)}")
    
    # Submit each chunk as a single Slurm job
    for chunk_idx, chunk in enumerate(chunked_jobs):
        submit_chunk(chunk, SBATCH_OPTIONS, chunk_idx)


if __name__ == "__main__":
    main()
