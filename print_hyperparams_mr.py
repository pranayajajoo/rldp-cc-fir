import os
import pandas as pd
import numpy as np
from collections import defaultdict

# point this at your cheetah_ directory
# root_dir = "/work/09313/hsikchi/vista/work/metamotivo/results/MR_total_3m_v1_dim_512/quadruped_"
root_dir = "/work/09313/hsikchi/vista/work/metamotivo/results/MR_total_3m_v1_dim_ablate/pointmass_"
# collect last‐timestep rewards by hyperparam
rewards = defaultdict(list)

for sub in os.listdir(root_dir):
    subdir = os.path.join(root_dir, sub)
    if not os.path.isdir(subdir):
        continue
    # expect names like "rs_100000_eh_1_seed_2"
    if "_seed_" not in sub:
        continue
    hyperparam = sub.split("_seed_")[0]
    # hyperparam = "empty"
    log_file = os.path.join(subdir, "eval_log.txt")
    if not os.path.isfile(log_file):
        continue
    # import ipdb;ipdb.set_trace()
    # read the table (tab‐separated)
    # if log file is empty, skip
    if os.path.getsize(log_file) == 0:
        print(f"Skipping empty log file: {log_file}")
        continue
    # if log file is not empty, read it
    df = pd.read_csv(log_file, sep="\t")
    
    
    try:
        last_reward = df["average_reward"].iloc[-1]
    except:
        # load the log
        df = pd.read_csv(log_file, sep="\t")

        # find all columns that end with "_reward" but not "#std"
        reward_cols = [c for c in df.columns if c.endswith("_reward") and not c.endswith("#std")]
        if not reward_cols:
            raise RuntimeError(f"No *_reward columns found in {log_file}")

        # compute average_reward column
        df["average_reward"] = df[reward_cols].mean(axis=1)
        last_reward = df["average_reward"].iloc[-1]
    rewards[hyperparam].append(last_reward)

# build results
results = []
for hp, vals in rewards.items():
    vals = np.array(vals)
    results.append({
        "hyperparam": hp,
        "mean_reward": vals.mean(),
        "std_reward": vals.std(ddof=0)   # population std; use ddof=1 for sample std
    })

# turn into a DataFrame for pretty printing
res_df = pd.DataFrame(results)
res_df = res_df.sort_values("hyperparam").reset_index(drop=True)

# print as a nice table
print(f"{'hyperparam':<25} {'mean':>10} {'std':>10}")
for _, row in res_df.iterrows():
    print(f"{row.hyperparam:<25} {row.mean_reward:10.3f} {row.std_reward:10.3f}")
