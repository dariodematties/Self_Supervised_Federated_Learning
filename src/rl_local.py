#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch

import multiprocessing

import wandb

from update import LocalUpdate
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST
from utils import get_dataset, average_weights

class LocalActions():

    def __init__(self, args):

        multiprocessing.set_start_method("spawn")
        
        self.args = args
        self.device = args.device

        # Load dataset
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(args)

        # Set up local models for each node
        self.local_models = {}
        self.local_train_losses = {}
        self.local_test_losses = {}
        
        for i in range(args.num_users):
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

            local_model.to(i % self.args.n_gpus)
            local_model.train()
            self.local_models[i] = local_model
            self.local_train_losses[i] = []
            self.local_test_losses[i] = []

        print()
        print("Model Information: ")
        print(self.local_models[0])
        print()
        wandb.init(group="federated_mpi_experiment", config=vars(args))


    def share_weights(self, src, dest):
        """
        Average the weights between the models of two users.

        Parameters
        ----------
            src : int
                the index of the source user
            dest : int
                the index of the destination user
        """
        # Get the models for the source and destination users
        src_model = self.local_models[src]
        dest_model = self.local_models[dest]

        # Compute the average weights of the two models
        avg_weights = average_weights([src_model.state_dict(), dest_model.state_dict()])

        # Load the average weights into the models for the source and destination users
        self.local_models[src].load_state_dict(avg_weights)  
        self.local_models[dest].load_state_dict(avg_weights) 


    def average_all_weights(self, idxs_users):
        """
        Average the weights between idxs_users, and set all of the local models to have these weights.

        Parameters
        ----------
            idxs_users : list
                the indices of the current users selected for training
        """
        # Compute the average weights of the all models for idxs_users
        avg_weights = average_weights([model.state_dict() for idx, model in self.local_models.items() if idx in idxs_users])

        # Load the average weights into all of the models
        for model in self.local_models.values():
            model.load_state_dict(avg_weights)


    def local_training(self, idxs_users, epoch):
        """
        Perform local training for the specified users, evenly split across separate GPUs.

        Parameters
        ----------
        idxs_users: list
            a list of user indices for which to perform local training
        epoch: int
            the current global training round
        """

        torch.multiprocessing.spawn(self.single_gpu_local_training, (idxs_users, epoch), self.args.n_gpus, True)
    

    def single_gpu_local_training(self, rank, idxs_users, epoch):
        """
        Perform local training for the specified users on GPU i. This function will handle some portion of idxs_users for GPU i, depending on how many total GPUs are available.

        Parameters
        ----------
        rank: int
            the rank of the GPU to use for training
        idxs_users: list
            a list of user indices for which to perform local training
        epoch: int
            the current global training round
        """
        idxs_users = [idx_user for idx_user in idxs_users if idx_user % self.args.n_gpus == rank]

        for idx in idxs_users:
            # Create a LocalUpdate object for the current user
            local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                        idxs=self.user_groups[idx])

            # Perform local training for the current user
            if self.args.supervision:
                # Supervised learning
                w, loss = local_model.update_weights(
                    model=self.local_models[idx], global_round=epoch)
            else:
                # Self-supervised learning
                w, loss, out = local_model.update_weights(
                        model=self.local_models[idx], global_round=epoch)

            # Save the training loss for the current user
            self.local_train_losses[idx].append(loss)

    def local_evaluation(self, idxs_users):
        """
        Perform local evaluation for the specified users, evenly split across separate GPUs.

        Parameters
        ----------
        idxs_users: list
            a list of user indices for which to perform local evaluation
        Returns
        -------
        current_losses: dict
            a dictionary mapping user indices to their current test losses
        """

        queue = torch.multiprocessing.Queue()
        torch.multiprocessing.spawn(self.single_gpu_local_evaluation, (idxs_users, queue), self.args.n_gpus, True)
        current_losses = {}
        while not queue.empty():
            current_losses.update(queue.get())
        queue.close()
        return current_losses


    def single_gpu_local_evaluation(self, rank, idxs_users, queue):
        """
        Perform local training for the specified users on GPU rank. This function will handle some portion of idxs_users for GPU rank, depending on how many total GPUs are available.

        Parameters
        ----------
        rank: int
            the rank of the GPU to use for evaluation
        idxs_users: list
            a list of user indices for which to perform local evaluation
        queue: multiprocessing queue
            used to aggregate return values
        """

        my_idxs_users = [idx_user for idx_user in idxs_users if idx_user % self.args.n_gpus == rank]

        current_losses = {}

        for idx in my_idxs_users:
            print(f"Local evaluation on rank {rank} and user {idx}.")
            
            args = self.args
            args.gpu = rank

            # Create a LocalUpdate object for the current user
            local_model = LocalUpdate(args=args, dataset=self.train_dataset,
                                        idxs=self.user_groups[idx])

            # Perform local evaluation for the current user
            acc, loss = local_model.inference(self.local_models[idx])
            self.local_test_losses[idx].append(loss)
            current_losses[idx] = loss

        queue.put(current_losses)


    def plot_local_losses(self):
        train_loss_data = [
            [x, f"User {idx}", y] for idx, ys in self.local_train_losses.items() for (x, y) in zip(range(len(ys)), ys)
        ]

        test_loss_data = [
            [x, f"User {idx}", y] for idx, ys in self.local_test_losses.items() for (x, y) in zip(range(len(ys)), ys)
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