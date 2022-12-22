#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import time

import torch
import gym

import stable_baselines3
from stable_baselines3 import PPO

from rl_environment import FedEnv

from options import args_parser
from utils import exp_details

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

    if args.method == "random":
        print("Beginning evaluation, with actions taken at random.")
        env = FedEnv(args, 0)
        obs = env.reset()
        for _ in range(1024):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
        print("Finished evaluation.")

    elif args.method == "rl":
        print("Beginning policy network training with PPO.")
        # envs = [lambda: FedEnv(args, i) for i in range(args.n_gpus)]
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(args.n_gpus))
        # vec_env = stable_baselines3.common.vec_env.SubprocVecEnv(envs)
        # model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=128)
        env = FedEnv(args, 0)
        model = PPO("MlpPolicy", env, verbose=1, n_steps=128)
        model.learn(total_timesteps=512)
        print("Finished training.")
        print("Saving model to save/FedRL")
        model.save("save/FedRL")

        print("Beginning evaluation, with actions selected by policy network.")
        env = FedEnv(args, 0)
        obs = env.reset()
        for _ in range(1024):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        print("Finished evaluation")

    elif args.method == "fedavg":
        print("Beginning evaluation, with FedAvg scheme.")
        env = FedEnv(args, 0)
        obs = env.reset()
        for _ in range(1024):
            # Perform DoNothing action
            action = 0
            obs, reward, done, info = env.step(action)
            if done:
                break
        print("Finished evaluation.")

    else:
        print(f"Invalid method: '{args.method}'")