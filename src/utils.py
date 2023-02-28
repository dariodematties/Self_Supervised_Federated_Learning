#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
from torchvision import datasets, transforms

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import dirichlet_sampling


def get_dataset(num_users, dataset, iid, unequal, dirichlet=False, alpha=None, shared_sample_ratio=False):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dirichlet and alpha is None:
        raise ValueError("If dirichlet is True, alpha must be specified.")

    if dataset == "cifar":
        data_dir = "../data/cifar/"
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if iid:
            # Sample IID user data from CIFAR
            user_groups = cifar_iid(train_dataset, num_users)
        else:
            # Sample Non-IID user data from CIFAR
            if dirichlet:
                user_groups = dirichlet_sampling(train_dataset, num_users, alpha, shared_sample_ratio)
            elif unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, num_users)

    elif dataset == "mnist" or "fmnist":
        if dataset == "mnist":
            data_dir = "../data/mnist/"
        else:
            data_dir = "../data/fmnist/"

        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, num_users)
        else:
            # Sample Non-IID user data from Mnist
            if dirichlet:
                user_groups = dirichlet_sampling(train_dataset, num_users, alpha, shared_sample_ratio)
            elif unequal:
                # Chose unequal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, num_users)
            else:
                # Chose equal splits for every user
                # user_groups = mnist_noniid_custom(train_dataset, num_users)
                user_groups = mnist_noniid(train_dataset, num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(state_dicts, weights):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(state_dicts[0])
    for key in w_avg.keys():
        s = 0
        for idx, state_dict in enumerate(state_dicts):
            s += torch.mul(state_dict[key], weights[idx])
        w_avg[key] = s
    return w_avg


def exp_details(args):
    print("\nExperimental details:")

    print("\nReinforcement Arguments:")
    print(f"    Steps Before PPO Update : {args.ppo_n_steps}")
    print(f"    PPO Learning Rate       : {args.ppo_lr}")
    print(f"    PPO Discount Factor     : {args.ppo_gamma}")
    print(f"    PPO Batch Size          : {args.ppo_bs}")
    print(f"    PPO Total Timesteps     : {args.total_timesteps}")
    print(f"    Target Accuracy         : {args.target_accuracy}")

    print("\nFederated Arguments:")
    print(f"    Number of Users         : {args.num_users}")
    print(f"    Fraction of Users       : {args.frac}")
    print(f"    Local Epochs            : {args.local_ep}")
    print(f"    Local Batch Size        : {args.local_bs}")
    print(f"    Learning Rate           : {args.lr}")
    print(f"    Momentum                : {args.momentum}")
    print(f"    Optimizer               : {args.optimizer}")

    print("\nModel Arguments:")
    print(f"    Supervision             : {args.supervision}")
    print(f"    Model                   : {args.model}")

    print("\nMisc. Arguments:")
    print(f"    Dataset                 : {args.dataset}")
    print(f"    Number of GPUs          : {args.n_gpus}")
    print(f"    IID                     : {args.iid}")
    print(f"    Unequal                 : {args.unequal}")
    print(f"    Random Seed             : {args.seed}")
    print(f"    Test Fraction           : {args.test_fraction}")

    print()
