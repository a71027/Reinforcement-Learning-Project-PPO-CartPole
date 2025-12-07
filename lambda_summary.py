import pandas as pd
import numpy as np

lambda_sets = {
    "080": ["data/lam080_seed0.csv", "data/lam080_seed42.csv", "data/lam080_seed999.csv"],
    "090": ["data/lam090_seed0.csv", "data/lam090_seed42.csv", "data/lam090_seed999.csv"],
    "095": ["data/baseline_lam095_seed0.csv", "data/baseline_lam095_seed42.csv", "data/baseline_lam095_seed999.csv"],
    "099": ["data/lam099_seed0.csv", "data/lam099_seed42.csv", "data/lam099_seed999.csv"]
}

rows = []

for lam, paths in lambda_sets.items():
    vals = []
    for p in paths:
        df = pd.read_csv(p)
        vals.append(df["mean_reward"].iloc[-1])   # 마지막 epoch 값
    rows.append([lam, np.mean(vals), np.var(vals)])

result = pd.DataFrame(rows, columns=["lambda", "final_mean_reward", "variance"])
print(result)

result.to_csv("lambda_summary.csv", index=False)
