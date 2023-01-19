#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
import wandb

from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST
from utils import get_dataset, weighted_average_weights, average_weights

class RLActions():

    def __init__(self, args, device, method):
        
        self.args = args
        self.device = device
        self.method = method

        # Load dataset
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(args)

        # Set up local models for each node
        # TODO: make sure this is actually calling self.get_model separate times
        self.local_models = {i: self.get_model(self.args) for i in range(args.num_users)}
        self.global_model = self.get_model(self.args)

        # Output model information
        print()
        print("Model Information: ")
        print(self.global_model)
        print()

        # Set up tracking for loss and accuracy of global model
        self.global_test_losses = []
        self.global_test_accuracies = []


    def get_model(self, args):
        if args.supervision:
            # Supervised learning
            if args.model == 'cnn':
                # Convolutional neural netork
                if args.dataset == 'mnist':
                    model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    model = CNNFashion_Mnist(args=args)
                elif args.dataset == 'cifar':
                    model = CNNCifar(args=args)

            elif args.model == 'mlp':
                # Multi-layer preceptron
                img_size = self.train_dataset[0][0].shape
                len_in = 1
                for x in img_size:
                    len_in *= x
                model = MLP(dim_in=len_in, dim_hidden=64,
                            dim_out=args.num_classes)
            else:
                exit('Error: unrecognized model')
        else:
            # Self-supervised learning
            if args.model == 'autoencoder':
                # Autoencoder with transpose convolutions
                if args.dataset == 'mnist':
                    model = AutoencoderMNIST(args=args)

            else:
                exit('Error: unrecognized unsupervised model')

        model.to(self.device)
        model.train()

        return model


    def lift_model(self, usr, weight):
        """
        Lift a local model into the global model through a weighted average.
        
        Parameters
        ----------
            usr : int
                the user whose model will be lifted into the global model
            weight : float
                the weight of the local model in the averaged model
        """
        local_model_weights = self.local_models[usr].state_dict()
        global_model_weights = self.global_model.state_dict()

        avg_weights = weighted_average_weights([local_model_weights, global_model_weights], [weight, 1 - weight])

        self.global_model.load_state_dict(avg_weights)


    def average_into_global(self, weight):
        local_model_weights = [model.state_dict() for model in self.local_models.values()]
        global_model_weights = self.global_model.state_dict()
        avg_weights = average_weights(local_model_weights)
        new_global_weights = weighted_average_weights([avg_weights, global_model_weights], torch.tensor([weight, 1 - weight]).to(self.device))
        self.global_model.load_state_dict(new_global_weights)      


    def drop_model(self, usr):
        """
        Drop the global model into a user's local model.
        
        Parameters
        ----------
            usr : int
                the user who will receive the global model
        """
        global_model_weights = self.global_model.state_dict()
        self.local_models[usr].load_state_dict(global_model_weights)


    def local_training(self, usr, epoch):
        """
        Perform local training for the specified user.

        Parameters
        ----------
        usr: int
            the user for which to perform local training
        epoch: int
            the current global training round
        """

        # Create a LocalUpdate object for the current user
        local_model = LocalUpdate(args=self.args, device=self.device,
                    dataset=self.train_dataset, idxs=self.user_groups[usr])

        # Perform local training for the current user
        if self.args.supervision:
            # Supervised learning
            w, loss = local_model.update_weights(
                model=self.local_models[usr], global_round=epoch)
        else:
            # Self-supervised learning
            w, loss, out = local_model.update_weights(
                    model=self.local_models[usr], global_round=epoch)

        return loss


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
        local_model = LocalUpdate(args=self.args, device=self.device,
                    dataset=self.train_dataset, idxs=self.user_groups[usr])

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
        loss: float
            the current loss for the specified user
        """

        # Perform local evaluation for the specified user
        model = self.global_model if usr == -1 else self.local_models[usr]
        acc, loss = test_inference(self.args, self.device, model, self.test_dataset)
        
        if usr == -1 and save_loss:
            self.global_test_losses.append(loss)
            if self.args.supervision:
                self.global_test_accuracies.append(acc)

        return loss


    def plot_global_loss(self):
        data = [[step, loss] for (step, loss) in enumerate(self.global_test_losses)]
        table = wandb.Table(data=data, columns=["Step", "Global Test Loss"])
        wandb.log({"global-test-losses": wandb.plot.line(table, "Step", "Global Test Loss", title="Global Test Loss vs. Training Step")})
    
        if self.args.supervision:
            data = [[step, accuracy] for (step, accuracy) in enumerate(self.global_test_accuracies)]
            table = wandb.Table(data=data, columns=["Step", "Global Test Accuracy"])
            wandb.log({"global-test-accuracy": wandb.plot.line(table, "Step", "Global Test Accuracy", title="Global Test Accuracy vs. Training Step")})