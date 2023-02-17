#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import time

import torch
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_environment import FedEnv
from options import args_parser
from utils import exp_details


def make_env(device):
    return lambda: FedEnv(args=args, device=device)


def evaluate_rl(args):
    print("[RL Main] Beginning policy network training with PPO.")
    env = SubprocVecEnv([make_env(f'cuda:{i}') for i in range(args.n_gpus)])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=args.ppo_n_steps, learning_rate=args.ppo_lr, gamma=args.ppo_gamma)
    model.learn(total_timesteps=args.total_timesteps * args.n_gpus)
    print("[RL Main] Finished training.")
    print("[RL Main] Saving model to save/FedRL")
    model.save("save/FedRL")


if __name__ == '__main__':

    # Set up timer
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('..')
    
    # Parse, validate, and print arguments
    args = args_parser()

    if args.supervision:
        assert args.model in ["cnn", "mlp"]
    else:
        assert args.model in ["autoencoder"]

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    args.device = 'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu'

    # Set random seeds for numpy and stable baselines
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    exp_details(args)

    evaluate_rl(args)