#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import mpi4py

import torch
import wandb

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST
from utils import get_dataset, exp_details, average_weights

from mpi4py import MPI
from mpi_communication import gather_weights, gather_losses, bcast_state_dict
from rl_actions import SwapWeights, ShareWeights, ShareRepresentations
from rl_environment import FedEnv

class LocalActions():

    def __init__(self, args):
        
        self.args = args
        self.comm = args.comm
        self.rank = args.rank
        self.size = args.size
        self.device = args.device

        # Load dataset
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(args)

        # Set up local models for each node
        self.local_models = {}
        self.local_train_losses = {}
        self.local_test_losses = {}
        
        for i in range(args.num_users):
            if i % args.size == args.rank:
                if args.supervision:
                    # Supervised learning
                    if args.model == 'cnn':
                        # Convolutional neural netork
                        if args.dataset == 'mnist':
                            local_model = CNNMnist(args=args)
                        elif args.dataset == 'fmnist':
                            local_model = CNNFashion_Mnist(args=args)
                        elif args.dataset == 'cifar':
                            local_model = CNNCifar(args=args)

                    elif args.model == 'mlp':
                        # Multi-layer preceptron
                        img_size = self.train_dataset[0][0].shape
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                            local_model = MLP(dim_in=len_in, dim_hidden=64,
                                            dim_out=args.num_classes)
                    else:
                        exit('Error: unrecognized model')
                else:
                    # Self-supervised learning
                    if args.model == 'autoencoder':
                        # Autoencoder with transpose convolutions
                        if args.dataset == 'mnist':
                            local_model = AutoencoderMNIST(args=args)

                    else:
                        exit('Error: unrecognized unsupervised model')

                local_model.to(self.device)
                local_model.train()
                self.local_models[i] = local_model
                self.local_train_losses[i] = []
                self.local_test_losses[i] = []

        if args.rank == 0:
            print()
            print("Model Information: ")
            print(self.local_models[0])
            print()
            wandb.init(group="federated_mpi_experiment", config=vars(args))

    # DEFINE NECESSARY FUNCTIONS FOR LOCAL TRAINING, LOCAL EVALUATION, AND ALL ACTIONS

    # Functions for ShareWeights

    def dummy(self):
        print(f"I am rank {self.comm.Get_rank()}")

    def send_weights(self, src, dest):
        src_model = copy.deepcopy(self.local_models[src])
        self.comm.isend(src_model, dest=dest % self.size)

    def receive_and_send_weights(self, src, dest):
        dest_model = copy.deepcopy(self.local_models[dest])

        shared_src_model = self.comm.recv(source=src % self.size)
        shared_src_model.to(self.device)
        avg_weights = average_weights([self.local_models[dest].state_dict(), shared_src_model.state_dict()])
        self.local_models[dest].load_state_dict(avg_weights)  

        self.comm.isend(dest_model, dest=src % self.size)

    def receive_weights(self, src, dest):
        shared_dest_model = self.comm.recv(source=dest % self.size)
        shared_dest_model.to(self.device)
        avg_weights = average_weights([self.local_models[src].state_dict(), shared_dest_model.state_dict()])
        self.local_models[src].load_state_dict(avg_weights)

    def local_training(self, idxs_users, epoch):
        for idx in idxs_users:
            if idx % self.size == self.rank:
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx])
                if self.args.supervision:
                    # Supervised learning
                    w, loss = local_model.update_weights(
                        model=self.local_models[idx], global_round=epoch)
                else:
                    # Self-supervised learning
                    w, loss, out = local_model.update_weights(
                            model=self.local_models[idx], global_round=epoch)

                self.local_train_losses[idx].append(loss)

    def local_evaluation(self, idxs_users):
        current_losses = {}
        for idx in idxs_users:
            if idx % self.size == self.rank:
                # this should perform inference, because LocalUpdate splits the passed dataset into train, validation, and test, and the inference() function operates on test data
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx])
                acc, loss = local_model.inference(self.local_models[idx])
                self.local_test_losses[idx].append(loss)
                current_losses[idx] = loss
        return current_losses

    def plot_losses(self):
        train_losses_combined = self.comm.gather(self.local_train_losses, root=0)
        test_losses_combined = self.comm.gather(self.local_test_losses, root=0)

        if self.rank == 0:
            
            train_losses_combined = {idx: loss for local_train_losses in train_losses_combined for (idx, loss) in local_train_losses.items()}
            test_losses_combined = {idx: loss for local_test_losses in test_losses_combined for (idx, loss) in local_test_losses.items()}
        
            train_loss_data = [
                [x, f"User {idx}", y] for idx, ys in train_losses_combined.items() for (x, y) in zip(range(len(ys)), ys)
            ]

            test_loss_data = [
                [x, f"User {idx}", y] for idx, ys in test_losses_combined.items() for (x, y) in zip(range(len(ys)), ys)
            ]
            
            train_loss_table = wandb.Table(data=train_loss_data, columns=["step", "lineKey", "lineVal"])
            test_loss_table = wandb.Table(data=test_loss_data, columns=["step", "lineKey", "lineVal"])

            plot = wandb.plot_table(
                "srajani/fed-users",
                train_loss_table,
                {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
                {"title": "Train Loss vs. Per-Node Local Training Round"}
            )
            wandb.log({"train_loss_plot": plot})

            plot = wandb.plot_table(
                "srajani/fed-users",
                test_loss_table,
                {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
                {"title": "Test Loss vs. Per-Node Local Training Round"}
            )
            wandb.log({"test_loss_plot": plot})