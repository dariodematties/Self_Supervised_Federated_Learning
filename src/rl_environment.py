#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import gym
from gym import spaces
import numpy as np
from rl_local import LocalActions

class FedEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, args, device, save_loss=True):
        print(f"[RL Environment] Initializing environment for device {device}")

        super(FedEnv, self).__init__()

        # The actions are ShareWeights, SwapWeights, ShareRepresentations, and DoNothing
        # The observations are (src_loss, trg_loss, avg_loss)

        # TODO: Replace high=100 with non-placeholder value

        args.device = device

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=float)

        self.args = args
        self.epoch = 0
        self.local_actions = LocalActions(self.args)

        self.epochs = args.epochs
        self.method = args.method
        self.num_users = args.num_users
        self.frac = args.frac
        self.dummy_environment = args.dummy_environment
        self.save_loss = save_loss


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

        # Get source and destination nodes
        current_src = self.idxs_users[self.usr_1]
        current_dest = self.idxs_users[self.usr_2]
        
        observation, reward, done, info = None, 0, False, {}

        if not self.dummy_environment:
            # If the action is 0, do nothing
            
            # If the action is 1, share the weights between the current source and destination users
            if action == 1:
                self.local_actions.share_weights(current_src, current_dest)
            
            # Compute the losses after action is taken
            self.post_action_losses = self.local_actions.local_evaluation([current_src, current_dest])

            # Calculate the reward as the difference between the sum of the losses before and after the action was taken
            pre_action_loss_sum = self.pre_action_losses[current_src] + self.pre_action_losses[current_dest]
            post_action_loss_sum = self.post_action_losses[current_src] + self.post_action_losses[current_dest]
            
            reward = pre_action_loss_sum - post_action_loss_sum
        
        # Print out some information about the step taken
        print(f"[RL Environment] Source: {current_src}, Destination: {current_dest}, Action: {action}, Reward: {reward}")
        
        # If not all of the pairs in idxs_users have been exhausted, update the users
        if self.update_users():
            if self.dummy_environment:
                observation = np.array([0, 0, 0])      
            else:
                current_src = self.idxs_users[self.usr_1]
                current_dest = self.idxs_users[self.usr_2]
                self.pre_action_losses = self.local_actions.local_evaluation([current_src, current_dest])
                observation = np.array([
                    self.pre_action_losses[current_src],
                    self.pre_action_losses[current_dest],
                    self.loss_avg
                ])
        
        # Otherwise, perform local training and sample new users
        else:
            print("[RL Environment] Performing local training")

            if not self.dummy_environment:
                self.local_actions.local_training(self.idxs_users, self.epoch, save_loss=self.save_loss)
                if self.method == "fedavg":
                    print("[RL Environment] Averaging weights")
                    self.local_actions.average_all_weights(self.idxs_users)

            self.epoch += 1

            self.sample_users()

            if self.dummy_environment:
                observation = np.array([0, 0, 0])
            else:
                losses = self.local_actions.local_evaluation(self.idxs_users, save_loss=self.save_loss)
                self.loss_avg = sum(losses.values()) / len(losses.values())

                current_src = self.idxs_users[self.usr_1]
                current_dest = self.idxs_users[self.usr_2]
                self.pre_action_losses = self.local_actions.local_evaluation([current_src, current_dest])

                observation = np.array([
                    losses[current_src],
                    losses[current_dest],
                    self.loss_avg
                ])   

            if (self.epoch == self.epochs):
                print(f"[RL Environment] Finished {self.epochs} epochs")
                done = True
                if not self.dummy_environment and self.save_loss:
                    print(f"[RL Environment] Plotting losses")
                    self.local_actions.plot_local_losses()  
            
        return observation, reward, done, info
    

    def update_users(self):
        """Update the current source and destination users.

        Returns
        -------
        bool
            A flag indicating whether the users have been updated. If the users
            cannot be updated, returns False.
        """
        # If the destination user is not the last user in the list, increment
        # the destination user index
        if self.usr_2 < len(self.idxs_users) - 1:
            self.usr_2 += 1
        # If the source user is not the last user in the list, increment the
        # source user index and reset the destination user index
        elif self.usr_1 < len(self.idxs_users) - 1:
            self.usr_1 += 1
            self.usr_2 = 0
        # If both the source and destination users are the last user in the list,
        # return False to indicate that the users cannot be updated
        else:
            return False
        # Return True to indicate that the users have been updated
        return True


    def sample_users(self):
        """Sample a subset of users from the full set of users.
        The subset of users will be used for actions and local training.
        """
        # Calculate the number of users to sample, ensuring that at least one user
        # is sampled
        m = max(int(self.frac * self.num_users), 1)
        # Sample the users without replacement
        self.idxs_users = np.random.choice(range(self.num_users), m, replace=False)

        # Print out the sampled users
        print(f"[RL Environment] Sampled Users: {self.idxs_users}")

        # Set the indices of the source and destination users to the first user in
        # the sampled list
        self.usr_1 = 0
        self.usr_2 = 0


    def reset(self):
        """Reset the environment.

        Returns
        -------
        observation : numpy array
            The initial observation of the environment after resetting.
        """
        print(f"[RL Environment] Resetting Environment")

        self.epoch = 0
        self.local_actions = LocalActions(self.args)

        # Sample a new set of users
        self.sample_users()

        # Get the losses for all users and calculate the average loss
        losses = self.local_actions.local_evaluation(self.idxs_users, save_loss=self.save_loss)
        self.loss_avg = sum(losses.values()) / len(losses.values())

        # Set pre-action losses
        current_src = self.idxs_users[self.usr_1]
        current_dest = self.idxs_users[self.usr_2]
        self.pre_action_losses = self.local_actions.local_evaluation([current_src, current_dest])

        # Set the initial observation to be the losses for the source and destination users, as well as the average loss
        observation = np.array([
            losses[self.idxs_users[self.usr_1]],
            losses[self.idxs_users[self.usr_2]],
            self.loss_avg
        ])
            
        return observation
