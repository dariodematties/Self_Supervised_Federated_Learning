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

from mpi4py import MPI
from rl_environment import FedEnv
from rl_local import LocalActions

if __name__ == '__main__':

    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    
    args.comm = comm
    args.rank = rank
    args.size = size
    args.gpu = rank        

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    args.device = 'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu'

    if args.rank == 0:
        exp_details(args)
        print(f'Environment size: {args.size}')

    print(f'[Rank {args.rank}] Checking in with device {args.device} and GPU number {args.gpu}.')
    
    local_actions = LocalActions(args)

    # Set Up RL Agent
    env = FedEnv(args.num_users, args.frac, args.rank, args.size, args.comm, local_actions, args.epochs)
    # env = gym.make('LunarLander-v2')
    
    # check_env(env)

    # model = None
    # if args.rank == 0:
    #     model = PPO("MlpPolicy", env, verbose=1)

    # model = comm.bcast(model, root=0)

    model = PPO("MlpPolicy", env, verbose=1)

    if args.rank == 0:
        print("Beginning training.")

    # model.learn(total_timesteps=1)

    env.reset()

    for _ in range(1000):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    if args.rank == 0:
        print("Finished training.")

    model.save("FedRL")