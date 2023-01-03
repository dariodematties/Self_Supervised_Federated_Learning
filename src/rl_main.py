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

def evaluate_rl(args):
    print("Beginning policy network training with PPO.")
    # save_loss is set to False during training; save_rewards is set to True
    env = FedEnv(args=args, device=0, method="fed_rl", save_loss=False, save_rewards=True)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=args.ppo_n_steps, learning_rate=args.ppo_lr)
    train_steps = args.rl_episodes * args.epochs * (args.num_users * args.frac) ** 2
    model.learn(total_timesteps=train_steps)
    print("Finished training.")
    print("Saving model to save/FedRL")
    model.save("save/FedRL")

    print("Beginning evaluation, with actions selected by policy network.")
    env = FedEnv(args=args, device=0, method="fed_rl")
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    print("Finished evaluation")


def evaluate_random(args):
    print("Beginning evaluation, with actions taken at random.")
    env = FedEnv(args=args, device=0, method="fed_random")
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    print("Finished evaluation.")


def evaluate_avg(args):
    print("Beginning evaluation, with FedAvg scheme.")
    env = FedEnv(args=args, device=0, method="fed_avg")
    obs = env.reset()
    done = False
    while not done:
        # Perform DoNothing action
        action = 0
        obs, reward, done, info = env.step(action)
    print("Finished evaluation.")


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

    if args.method == "fed_rl" or args.method == "all":
        evaluate_rl(args)
    if args.method == "fed_random" or args.method == "all":
        evaluate_random(args)
    if args.method == "fed_avg" or args.method == "all":
        evaluate_avg(args)
