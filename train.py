# train.py — PPO training with lambda parameter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from ppo import ActorCritic, compute_gae

def train(seed=0, clip_param=0.2, lam=0.95, epochs=50, out="result.csv"):

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    logs = []

    for epoch in range(epochs):
        states, actions, rewards, values, neglogp = [], [], [], [], []

        state, _ = env.reset()
        done = False

        while not done:
            s = torch.tensor(state, dtype=torch.float32)
            pi, v = model(s)
            dist = torch.distributions.Categorical(pi)
            a = dist.sample()

            next_state, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated

            states.append(s)
            actions.append(a)
            rewards.append(r)
            values.append(v.item())
            neglogp.append(dist.log_prob(a).item())

            state = next_state

        returns = compute_gae(rewards, values, lam=lam)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log = torch.tensor(neglogp)

        for _ in range(10):
            pi, v = model(states)
            dist = torch.distributions.Categorical(pi)
            new_log = dist.log_prob(actions)

            ratio = torch.exp(new_log - old_log)
            adv = torch.tensor(returns) - v.squeeze()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (torch.tensor(returns) - v.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        logs.append({
            "epoch": epoch+1,
            "mean_reward": np.sum(rewards),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "lam": lam
        })

    pd.DataFrame(logs).to_csv(out, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out", type=str, default="result.csv")
    args = parser.parse_args()

    train(seed=args.seed, clip_param=args.clip, lam=args.lam,
          epochs=args.epochs, out=args.out)
