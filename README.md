# Reinforcement-Learning-Project-PPO-CartPole
This is an introductory personal project assignment for reinforcement learning. 

It involves reproducing the results of the PPO paper and conducting reproduction experiments using parameters not covered in the paper.


# PPO Reproduction + GAE Lambda Parameter Study 


## Overview 

This project reproduces the baseline PPO experiment from the 2017 paper "Proximal Policy Optimization Algorithms", 
and performs an extended experiment on λ (GAE lambda), a parameter not explored in the original paper. 

The baseline experiment follows the PPO paper settings, while the extended
experiment varies the λ parameter to analyze its effect on stability and
learning performance.


## Baseline (Paper Reproduction)

- clip ε = 0.2 - λ = 0.95 - seeds = 0, 42, 999 - Environment: CartPole-v1 


## Extended Experiment (Not in the Paper)

The PPO paper uses λ = 0.95 but does not test different λ values. 
Extended experiment evaluates: 
- λ = 0.80 
- λ = 0.90 
- λ = 0.95 
- λ = 0.99 

I analyzed how λ influences: 
- variance of advantage estimates 
- stability of training 
- convergence speed


## Installation

python -m venv venv 

venv\Scripts\activate pip 

install torch gymnasium numpy pandas matplotlib


## Running Baseline

python train.py --seed 0 --clip 0.2 --lam 0.95 --out data/baseline_lam095_seed0.csv

python train.py --seed 42 --clip 0.2 --lam 0.95 --out data/baseline_lam095_seed42.csv

python train.py --seed 999 --clip 0.2 --lam 0.95 --out data/baseline_lam095_seed999.csv


## Running Lambda Experiments

python train.py --seed 0 --clip 0.2 --lam 0.80 --out data/lam080_seed0.csv

python train.py --seed 42 --clip 0.2 --lam 0.80 --out data/lam080_seed42.csv

python train.py --seed 999 --clip 0.2 --lam 0.80 --out data/lam080_seed999.csv

python train.py --seed 0 --clip 0.2 --lam 0.90 --out data/lam090_seed0.csv

python train.py --seed 42 --clip 0.2 --lam 0.90 --out data/lam090_seed42.csv

python train.py --seed 999 --clip 0.2 --lam 0.90 --out data/lam090_seed999.csv

python train.py --seed 0 --clip 0.2 --lam 0.99 --out data/lam099_seed0.csv

python train.py --seed 42 --clip 0.2 --lam 0.99 --out data/lam099_seed42.csv

python train.py --seed 999 --clip 0.2 --lam 0.99 --out data/lam099_seed999.csv


## Plotting

python plot_baseline.py 

python plot_lambda.py


## File Structure 

Reinfrocement Learning_Project/ ├── train.py ├── ppo.py ├── plot_baseline.py ├── plot_lambda.py ├── lambda_summary.py ├── data/ │ ├── baseline_lam095_seed0.csv │ ├── lam080_seed0.csv │ ├── ... └── README.md


## Author 

a71027/문정인
