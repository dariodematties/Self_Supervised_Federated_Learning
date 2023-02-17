#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import gym
from gym import spaces

import wandb

import numpy as np
import math

from rl_actions import RLActions


class FedEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, args, device):
        print(f"[RL Environment] Initializing environment for device {device}")

        super(FedEnv, self).__init__()

        # Reward and action tracking
        self.all_rewards = []
        self.all_actions = []
        self.all_steps = []
        self.curr_training_step = 0

        self.device = device
        self.wandb = args.wandb
        self.total_timesteps = args.total_timesteps
        self.target_accuracy = args.target_accuracy
        self.num_users = args.num_users
        self.frac = args.frac
        self.local_ep = args.local_ep
        self.local_bs = args.local_bs
        self.lr = args.lr
        self.supervision = args.supervision
        self.model = args.model
        self.num_channels = args.num_channels
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.optimizer = args.optimizer
        self.iid = args.iid
        self.unequal = args.unequal
        self.verbose = args.verbose
        self.test_fraction = args.test_fraction

        # Set up logging with Weights & Biases
        if self.wandb:
            wandb.init(config=vars(args), group="choose-mb-5")

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
        self.observation_space = spaces.Box(low=-math.inf, high=math.inf, shape=(self.num_users + 1, 5))

    def step(self, action):

        """Take a single step in the environment

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        observation : numpy array
            The observation of the environment after the action has been taken.
        reward : float
            The reward for taking the action.
        done : bool
            A flag indicating whether the episode has finished.
        info : dict
            Additional information about the environment.
        """
        self.curr_step += 1
        self.curr_training_step += 1

        observation, reward, done, info = np.array([0, 0]), 0, False, {}

        num_minibatches = int(action[0] * 29 + 1)

        for usr in self.curr_usrs:
            self.rl_actions.local_training(usr=usr, epoch=0, num_minibatches=num_minibatches)

        # Get observation with PCA
        observation = self.rl_actions.get_pca_reduced_models()
        self.rl_actions.average_into_global(1)

        # Compute global loss
        global_acc = self.rl_actions.local_test_evaluation(-1)
        self.global_accs.append(global_acc)
        reward = 64 ** (global_acc - self.target_accuracy) - 1

        for usr in self.curr_usrs:
            self.rl_actions.drop_model(usr)

        self.episode_rewards.append(reward)
        self.episode_actions.append(action)

        if global_acc > self.target_accuracy:

            self.all_rewards.append(self.episode_rewards)
            self.all_actions.append(self.episode_actions)
            self.all_steps.append(self.curr_step)

            self.episode_rewards = []
            self.episode_actions = []

            done = True

        if self.curr_training_step == self.total_timesteps and self.wandb:
            self.plot_actions_and_rewards()

        # Print out some information about the step taken
        print(
            f"[RL Environment] [Step {self.curr_training_step:{'0'}{4}}] Action: {num_minibatches}, Acc: {global_acc:.3f}, Reward: {reward:.3f}"
        )
        return observation, reward, done, info

    def sample_users(self):
        """Sample a subset of users from the full set of users.
        The subset of users will be used for actions and local training.
        """
        # Calculate the number of users to sample, ensuring that at least one user
        # is sampled
        m = max(int(self.frac * self.num_users), 1)

        # Sample the users without replacement
        self.curr_usrs = np.random.choice(range(self.num_users), m, replace=False)

        # Set the index of the current user to the first user in the sampled list
        self.idx_curr_usr = 0
        self.curr_usr = self.curr_usrs[self.idx_curr_usr]

        # Print out the sampled users
        print(f"[RL Environment] Sampled Users: {self.curr_usrs}")

    def plot_actions_and_rewards(self):
        print("[RL Environment] Plotting actions and rewards")

        # Plot all rewards
        all_rewards_flattened = [reward for episode_rewards in self.all_rewards for reward in episode_rewards]
        data = [[step, reward] for (step, reward) in enumerate(all_rewards_flattened)]
        table = wandb.Table(data=data, columns=["Step", "Reward"])
        wandb.log({"rewards": wandb.plot.line(table, "Step", "Reward", title="Reward vs. Training Step")})

        # Plot total episode rewards
        total_episode_rewards = list(map(lambda l: sum(l), self.all_rewards))
        data = [[episode, reward] for (episode, reward) in enumerate(total_episode_rewards)]
        table = wandb.Table(data=data, columns=["Episode", "Reward"])
        wandb.log(
            {
                "total-episode-rewards": wandb.plot.line(
                    table,
                    "Episode",
                    "Reward",
                    title="Total Reward vs. Training Episode",
                )
            }
        )

        # Plot episode steps
        data = [[episode, steps] for (episode, steps) in enumerate(self.all_steps)]
        table = wandb.Table(data=data, columns=["Episode", "Steps"])
        wandb.log(
            {
                "episode-steps": wandb.plot.line(
                    table,
                    "Episode",
                    "Steps",
                    title="Steps vs. Training Episode",
                )
            }
        )

    def reset(self):
        """Reset the environment.

        Returns
        -------
        observation : numpy array
            The initial observation of the environment after resetting.
        """
        print(f"[RL Environment] Resetting Environment")

        self.curr_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.global_accs = []
        self.rl_actions = RLActions(
            self.num_users,
            self.local_ep,
            self.local_bs,
            self.lr,
            self.optimizer,
            self.supervision,
            self.model,
            self.num_channels,
            self.dataset,
            self.num_classes,
            self.iid,
            self.unequal,
            self.verbose,
            self.test_fraction,
            self.device,
        )

        self.sample_users()

        for usr in self.curr_usrs:
            self.rl_actions.local_training(usr=usr, epoch=0, num_minibatches=23)
        self.rl_actions.compute_pca_loading_vectors()
        observation = self.rl_actions.get_pca_reduced_models()
        self.rl_actions.average_into_global(1)
        for usr in self.curr_usrs:
            self.rl_actions.drop_model(usr)

        return observation
