#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import gym
from gym import spaces
import numpy as np

class FedEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, num_users, frac, rank, size, comm, local_actions, max_epoch):
        if rank == 0:
            print("[RL Agent] Initializing Environment")

        super(FedEnv, self).__init__()

        # The actions are ShareWeights, SwapWeights, ShareRepresentations, and DoNothing
        # The observations are (src_loss, trg_loss, avg_loss)

        # TODO: Replace high=100 with non-placeholder value

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=float)
        self.num_users = num_users
        self.frac = frac
        self.rank = rank
        self.size = size
        self.comm = comm
        self.local_actions = local_actions
        self.epoch = 0
        self.max_epoch = max_epoch

    def get_losses(self):
        my_losses = self.local_actions.local_evaluation(self.idxs_users)
        losses = self.comm.gather(my_losses, root=0)

        if self.rank == 0:
            losses = {idx: l for loss in losses for (idx, l) in loss.items()}

        return losses


    def step(self, action):

        # Step 1: Compute losses before action was taken
        pre_action_losses = self.get_losses()

        current_src = self.idxs_users[self.usr_1]
        current_dest = self.idxs_users[self.usr_2]

        # ShareWeights
        if action == 0:
            if self.rank == current_src % self.size:
                self.local_actions.send_weights(current_src, current_dest)
            if self.rank == current_dest % self.size:
                self.local_actions.receive_and_send_weights(current_src, current_dest)
            if self.rank == current_src % self.size:
                self.local_actions.receive_weights(current_src, current_dest)
        
        # For now, assume no action otherwise
        
        # Step 2: Compute losses after action is taken
        post_action_losses = self.get_losses()

        if self.rank == 0:
            post_action_loss_avg = sum(post_action_losses.values()) / len(post_action_losses.values())
        
        # Observation and reward
        observation, reward, done, info = None, None, False, {}

        # Rank 0 computes reward
        if self.rank == 0:
            reward = - post_action_losses[self.idxs_users[self.usr_1]] - post_action_losses[self.idxs_users[self.usr_2]] + pre_action_losses[self.idxs_users[self.usr_1]] + pre_action_losses[self.idxs_users[self.usr_2]]

            print(f"[RL Agent] Source: {current_src}, Destination: {current_dest}, Action: {action}, Reward: {reward}")
            
        if self.update_users():        
            if self.rank == 0:
                observation = np.array([
                    post_action_losses[self.idxs_users[self.usr_1]],
                    post_action_losses[self.idxs_users[self.usr_2]],
                    post_action_loss_avg
                ])
            
        else:
            print("[RL Agent] Performing Local Training")
            self.local_actions.local_training(self.idxs_users, self.epoch)
            self.epoch += 1
            if (self.epoch == self.max_epoch):
                done = True

            self.sample_users()

            new_losses = self.get_losses()

            if self.rank == 0:
                new_loss_avg = sum(new_losses.values()) / len(new_losses.values())

                observation = np.array([
                    new_losses[self.idxs_users[self.usr_1]],
                    new_losses[self.idxs_users[self.usr_2]],
                    new_loss_avg
                ])            

        observation, reward, done, info = self.comm.bcast((observation, reward, done, info), root=0)

        return observation, reward, done, info
    
    def update_users(self):
        if self.usr_2 != len(self.idxs_users) - 1:
            self.usr_2 += 1
        elif self.usr_1 != len(self.idxs_users) - 1:
            self.usr_1 += 1
            self.usr_2 = 0
        else:
            return False
        return True

    def sample_users(self):
        m = max(int(self.frac * self.num_users), 1)
        self.idxs_users = np.random.choice(range(self.num_users), m, replace=False)

        print(f"[RL Agent] Sampled Users: {self.idxs_users}")

        # The following indices index into the self.idxs_users:
        self.usr_1 = 0
        self.usr_2 = 0

    def reset(self):

        if self.rank == 0:
            print(f"[RL Agent] Resetting Environment")

        self.sample_users()

        observation = None

        losses = self.get_losses()

        if self.rank == 0:
            loss_avg = sum(losses.values()) / len(losses.values())

            observation = np.array([
                losses[self.idxs_users[self.usr_1]],
                losses[self.idxs_users[self.usr_2]],
                loss_avg
            ])
        
        return observation

    # def render(self, mode="human"):
    #     ...

    # def close (self):
    #     ...