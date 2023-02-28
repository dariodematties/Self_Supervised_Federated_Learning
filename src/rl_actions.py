#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import numpy as np
from sklearn.decomposition import PCA

import torch
import wandb

from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST
from utils import get_dataset, weighted_average_weights, average_weights


class RLActions:
    def __init__(
        self,
        num_users,
        frac,
        local_bs,
        lr,
        optimizer,
        supervision,
        model,
        num_channels,
        dataset,
        num_classes,
        iid,
        unequal,
        dirichlet,
        alpha,
        test_fraction,
        device,
    ):

        self.num_users = num_users
        self.frac = frac
        self.local_bs = local_bs
        self.lr = lr
        self.optimizer = optimizer
        self.supervision = supervision
        self.model = model
        self.num_channels = num_channels
        self.dataset = dataset
        self.num_classes = num_classes
        self.iid = iid
        self.unequal = unequal
        self.dirichlet = dirichlet
        self.alpha = alpha
        self.test_fraction = test_fraction
        self.device = device

        # Load dataset
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(
            num_users, dataset, iid, unequal, dirichlet, alpha
        )

        # Set up local models for each node
        self.local_models = {
            i: self.get_model(supervision, model, dataset, num_channels, num_classes) for i in range(num_users)
        }
        self.global_model = self.get_model(supervision, model, dataset, num_channels, num_classes)

        # Output model information
        print()
        print("Model Information: ")
        print(self.global_model)
        print()

        # Set up tracking for loss and accuracy of global model
        self.global_test_losses = []
        self.global_test_accuracies = []

    def get_model(self, supervision, model, dataset, num_channels, num_classes):
        if supervision:
            # Supervised learning
            if model == "cnn":
                # Convolutional neural netork
                if dataset == "mnist":
                    model = CNNMnist(num_channels, num_classes)
                elif dataset == "fmnist":
                    model = CNNFashion_Mnist()
                elif dataset == "cifar":
                    model = CNNCifar(num_classes)

            elif model == "mlp":
                # Multi-layer preceptron
                img_size = self.train_dataset[0][0].shape
                len_in = 1
                for x in img_size:
                    len_in *= x
                model = MLP(dim_in=len_in, dim_hidden=64, dim_out=num_classes)
            else:
                exit("Error: unrecognized model")
        else:
            # Self-supervised learning
            if model == "autoencoder":
                # Autoencoder with transpose convolutions
                if dataset == "mnist":
                    model = AutoencoderMNIST(num_channels)

            else:
                exit("Error: unrecognized unsupervised model")

        model.to(self.device)
        model.train()

        return model

    def aggregate_models(self, usrs, weights):
        local_model_weights = [model.state_dict() for usr, model in self.local_models.items() if usr in usrs]
        new_global_weights = weighted_average_weights(local_model_weights, weights)
        self.global_model.load_state_dict(new_global_weights)

    def distribute_global_model(self, usrs):
        """
        Drop the global model into the specified local models.

        Parameters
        ----------
            usrs : list[int]
                the user who will receive the global model
        """
        global_model_weights = self.global_model.state_dict()
        for usr in usrs:
            self.local_models[usr].load_state_dict(global_model_weights)

    def local_training(self, usrs, local_ep):
        """
        Perform local training for the specified users.

        Parameters
        ----------
        usrs: list[int]
            the users for which to perform local training
        local_ep: int
            the number of local epochs to train for
        """

        for usr in usrs:
            # Create a LocalUpdate object for the current user
            local_model = LocalUpdate(
                self.train_dataset,
                self.user_groups[usr],
                local_ep,
                self.local_bs,
                self.lr,
                self.optimizer,
                self.supervision,
                self.device,
            )

            # Perform local training for the current user
            if self.supervision:
                # Supervised learning
                w, loss = local_model.update_weights(model=self.local_models[usr])
            else:
                # Self-supervised learning
                w, loss, out = local_model.update_weights(model=self.local_models[usr])

    def local_train_evaluation(self, usr):
        """
        Perform local train evaluation for the specified user.

        Parameters
        ----------
        usr: int
            the user for which to perform local evaluation
            a value of -1 can be used to refer to the global model

        Returns
        -------
        loss: float
            the current loss for the specified user
        """

        # Create a LocalUpdate object for the current user
        local_model = LocalUpdate(
            self.train_dataset,
            self.user_groups[usr],
            self.local_bs,
            self.lr,
            self.optimizer,
            self.supervision,
            self.device,
        )

        # Perform local train evaluation for the current user
        _, loss = local_model.inference(model=self.local_models[usr])
        return loss

    def local_test_evaluation(self, usr, save_loss=False):
        """
        Perform local test evaluation for the specified user.

        Parameters
        ----------
        usrs: int
            the user for which to perform local evaluation
            a value of -1 can be used to refer to the global model

        Returns
        -------
        acc: float
            the current accuracy for the specified user
        """

        # Perform local evaluation for the specified user
        model = self.global_model if usr == -1 else self.local_models[usr]
        acc, loss = test_inference(self.supervision, self.device, model, self.test_dataset, self.test_fraction)

        if usr == -1 and save_loss:
            self.global_test_losses.append(loss)
            if self.supervision:
                self.global_test_accuracies.append(acc)

        return acc

    def plot_global_acc(self):
        data = [[step, acc] for (step, acc) in enumerate(self.global_test_accs)]
        table = wandb.Table(data=data, columns=["Step", "Global Test Loss"])
        wandb.log(
            {
                "global-test-losses": wandb.plot.line(
                    table, "Step", "Global Test Accuracy", title="Global Test Accuracy vs. Training Step"
                )
            }
        )

        if self.supervision:
            data = [[step, accuracy] for (step, accuracy) in enumerate(self.global_test_accuracies)]
            table = wandb.Table(data=data, columns=["Step", "Global Test Accuracy"])
            wandb.log(
                {
                    "global-test-accuracy": wandb.plot.line(
                        table, "Step", "Global Test Accuracy", title="Global Test Accuracy vs. Training Step"
                    )
                }
            )

    def compute_pca_loading_vectors(self, usrs):
        users_per_round = int(self.num_users * self.frac)
        self.pca = PCA(n_components=users_per_round)
        global_parameters = torch.cat([p.flatten() for p in self.global_model.parameters()]).detach().cpu().numpy()
        local_parameters = [
            torch.cat([p.flatten() for p in model.parameters()]).detach().cpu().numpy()
            for usr, model in self.local_models.items()
            if usr in usrs
        ]
        self.pca.fit([global_parameters, *local_parameters])

    def get_pca_reduced_models(self, usrs):
        global_parameters = torch.cat([p.flatten() for p in self.global_model.parameters()]).detach().cpu().numpy()
        local_parameters = [
            torch.cat([p.flatten() for p in model.parameters()]).detach().cpu().numpy()
            for usr, model in self.local_models.items()
            if usr in usrs
        ]
        return self.pca.transform([global_parameters, *local_parameters])
