#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import time

import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from options import args_parser
from utils import exp_details

from rl_environment import FedEnv
from rl_local import LocalActions

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

    exp_details(args)
    
    local_actions = LocalActions(args)

    # Set Up RL Agent
    env = FedEnv(args.num_users, args.frac, local_actions, args.epochs)
    
    # check_env(env)

    # model = None
    # if args.rank == 0:
    #     model = PPO("MlpPolicy", env, verbose=1)

    # model = comm.bcast(model, root=0)

    model = PPO("MlpPolicy", env, verbose=1)

    print("Beginning training.")

    model.learn(total_timesteps=1)

    env.reset()

    # for _ in range(1000):
    #     # Random action
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()

    print("Finished training.")

    model.save("FedRL")