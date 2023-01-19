#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import gym
from gym import spaces

import wandb

import numpy as np

from rl_actions import RLActions

class FedEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, args, device, method, save_loss=True, save_rewards_and_actions=False):
        print(f"[RL Environment] Initializing environment for device {device}")

        super(FedEnv, self).__init__()

        # Set up logging with Weights & Biases
        wandb.init(config=vars(args), group="train-aggregate-DQN")

        # Reward and action tracking
        self.all_rewards = []
        self.all_actions = []
        self.curr_training_step = 0

        self.args = args
        self.epochs = args.epochs
        self.num_users = args.num_users
        self.frac = args.frac
        self.dummy_environment = args.dummy_environment
        self.total_timesteps = args.total_timesteps
        self.save_loss = save_loss
        self.save_rewards_and_actions = save_rewards_and_actions
        self.episode_steps = args.episode_steps
        self.method = method
        self.device = device

        assert args.dataset == "mnist"

        self.action_space = spaces.Discrete(2)

        self.action_dict = {
            0: "Train",
            1: "Aggregate"
        }

        self.observation_space = spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 100]), shape=(3,), dtype=np.float64)


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

        if not self.dummy_environment:

            pre_action_global_loss = self.rl_actions.local_test_evaluation(-1)
            
            if action == 0:
                self.steps_since_last_aggregation += 1
                for usr in self.curr_usrs:
                    self.rl_actions.local_training(usr, epoch=0)
            if action == 1:
                self.steps_since_last_aggregation = 0
                self.rl_actions.average_into_global(1)
                for usr in self.curr_usrs:
                    self.rl_actions.drop_model(usr)

            # Compute global loss
            global_loss = self.rl_actions.local_test_evaluation(-1)
            
            if action == 1:
                self.global_losses.append(global_loss)
            
            reward = pre_action_global_loss - global_loss

            if self.save_rewards_and_actions:
                self.episode_rewards.append(reward)
                self.episode_actions.append(action)

            if self.curr_step == self.episode_steps:
                if not self.dummy_environment:
                    if self.save_loss:
                        print(f"[RL Environment] Plotting loss")
                        self.rl_actions.plot_global_loss()

                    if self.save_rewards_and_actions:
                        self.all_rewards.append(self.episode_rewards)
                        self.all_actions.append(self.episode_actions)
                        self.episode_rewards = []
                        self.episode_actions = []
                done = True

            if self.curr_training_step == self.total_timesteps:
                self.plot_actions_and_rewards()

            dif1 = self.global_losses[-1] - self.global_losses[-2]
            dif2 = self.global_losses[-2] - self.global_losses[-3]
            observation = [self.steps_since_last_aggregation, dif1, dif1 - dif2]


        # Print out some information about the step taken
        print(f"[RL Environment] [Step {self.curr_training_step}] Action: {self.action_dict[action]}, Reward: {reward:+.8f}")

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

        # Plot mean episode rewards
        mean_episode_rewards = list(map(lambda l: sum(l) / len (l), self.all_rewards))
        data = [[episode, reward] for (episode, reward) in enumerate(mean_episode_rewards)]
        table = wandb.Table(data=data, columns=["Episode", "Reward"])
        wandb.log({"mean-episode-rewards": wandb.plot.line(table, "Episode", "Reward", title="Mean Reward vs. Training Episode")})

        # Plot mean episode action ratios
        for (action, action_name) in self.action_dict.items():
            episode_action_ratios = list(map(lambda l: l.count(action) / len(l), self.all_actions))
            data = [[episode, action_ratio] for (episode, action_ratio) in enumerate(episode_action_ratios)]
            table = wandb.Table(data=data, columns=["Episode", f"{action_name} Action Ratio"])
            plot = wandb.plot.line(table, "Episode", f"{action_name} Action Ratio", title=f"{action_name} Action Ratio vs. Training Episode")
            wandb.log({f"episode-action_{action_name}-ratios": plot})


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
        self.global_losses = []
        self.rl_actions = RLActions(self.args, self.device, self.method)

        # Sample a new set of users
        self.sample_users()

        # Get the observation
        self.steps_since_last_aggregation = 0

        for _ in range(3):
            for usr in self.curr_usrs:
                self.rl_actions.local_training(usr, epoch=0)
            self.rl_actions.average_into_global(1)
            for usr in self.curr_usrs:
                self.rl_actions.drop_model(usr)
            global_loss = self.rl_actions.local_test_evaluation(-1)
            self.global_losses.append(global_loss)

        dif1 = self.global_losses[-1] - self.global_losses[-2]
        dif2 = self.global_losses[-2] - self.global_losses[-3]
        observation = [self.steps_since_last_aggregation, dif1, dif1 - dif2]

        return observation