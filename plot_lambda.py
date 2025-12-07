# 이 코드를 plot_lambda.py로 저장
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

lambdas = {
    "λ=0.80": ["data/lam080_seed0.csv","data/lam080_seed42.csv","data/lam080_seed999.csv"],
    "λ=0.90": ["data/lam090_seed0.csv","data/lam090_seed42.csv","data/lam090_seed999.csv"],
    "λ=0.95": ["data/baseline_lam095_seed0.csv","data/baseline_lam095_seed42.csv","data/baseline_lam095_seed999.csv"],
    "λ=0.99": ["data/lam099_seed0.csv","data/lam099_seed42.csv","data/lam099_seed999.csv"]
}

plt.figure(figsize=(12,6))

for label, paths in lambdas.items():
    dfs = [pd.read_csv(p)["mean_reward"] for p in paths]
    avg = np.mean(dfs, axis=0)
    plt.plot(range(len(avg)), avg, label=label, linewidth=2)

plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Mean Reward (3-seed mean)")
plt.title("GAE λ Comparison")
plt.show()
