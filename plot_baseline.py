import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/baseline_lam095_seed0.csv")

plt.plot(df["epoch"], df["mean_reward"], label="baseline (ε=0.2, λ=0.95)")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.title("Baseline PPO : Paper Reproduction")
plt.grid()
plt.legend()
plt.show()
