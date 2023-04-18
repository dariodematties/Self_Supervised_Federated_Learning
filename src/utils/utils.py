#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
from torchvision import datasets, transforms

from .models import MLPMnist, CNNMnist, CNNCifar


def get_model(arch, dataset, device="cpu"):
    """Returns the model for a particular architecture and dataset, on the specified
    device.

    Args:
        arch (str): the name of the architecture (either "cnn" or "mlp")
        dataset (str): the name of the dataset (either "mnist" or "cifar")
        device (str): the device to keep the model on

    >>> model = get_model("cnn", "mnist")
    >>> model = get_model("cnn", "cifar")
    >>> model = get_model("mlp", "mnist")
    >>> get_model("mlp", "cifar")
    Traceback (most recent call last):
     ...
    ValueError: architecture 'mlp' not supported for 'cifar' dataset
    >>> get_model("resnet1000", "mnist")
    Traceback (most recent call last):
     ...
    ValueError: unrecognized architecture 'resnet1000'
    """
    if arch == "mlp":
        if dataset == "mnist":
            model = MLPMnist()
        else:
            raise ValueError(
                f"architecture '{arch}' not supported for '{dataset}' dataset"
            )

    elif arch == "cnn":
        if dataset == "mnist":
            model = CNNMnist()
        elif dataset == "cifar":
            model = CNNCifar()
        else:
            raise ValueError(
                f"architecture '{arch}' not supported for '{dataset}' dataset"
            )

    else:
        raise ValueError(f"unrecognized architecture '{arch}'")

    model.to(device)
    model.train()

    return model


def get_train_test(dataset, download=True):
    """Returns train and test dataset splits for a given dataset, with appropriate
    normalization applied.

    Args:
        dataset (str): the name of the dataset (either MNIST or CIFAR-10)
        download (bool): if True, downloads dataset from the Internet

    >>> train_dataset, test_dataset = get_train_test("mnist")
    >>> train_dataset, test_dataset = get_train_test("cifar")
    >>> train_dataset, test_dataset = get_train_test("fake_dataset")
    Traceback (most recent call last):
     ...
    ValueError: dataset 'fake_dataset' not supported
    """

    dataset_dir = f"../../data/{dataset}/"
    if dataset == "mnist":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            dataset_dir, train=True, download=False, transform=apply_transform
        )
        test_dataset = datasets.MNIST(
            dataset_dir, train=False, download=False, transform=apply_transform
        )
    elif dataset == "cifar":
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            dataset_dir, train=True, download=False, transform=apply_transform
        )
        test_dataset = datasets.CIFAR10(
            dataset_dir, train=False, download=False, transform=apply_transform
        )
    else:
        raise ValueError(f"dataset '{dataset}' not supported")

    return train_dataset, test_dataset


def average_weights(state_dicts):
    """Return the average of the passed state dicts."""
    w_avg = copy.deepcopy(state_dicts[0])
    for key in w_avg.keys():
        for i in range(1, len(state_dicts)):
            w_avg[key] += state_dicts[i][key]
        w_avg[key] = torch.div(w_avg[key], len(state_dicts))
    return w_avg


def weighted_average_weights(state_dicts, weights):
    """Return the weighted average of the passed state dicts."""
    w_avg = copy.deepcopy(state_dicts[0])
    for key in w_avg.keys():
        s = 0
        for idx, state_dict in enumerate(state_dicts):
            s += torch.mul(state_dict[key], weights[idx])
        w_avg[key] = s
    return w_avg


def exp_details(args):
    """Print out the passed arguments."""
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
    print(f"    Architecture            : {args.arch}")

    print("\nMisc. Arguments:")
    print(f"    Dataset                 : {args.dataset}")
    print(f"    Number of GPUs          : {args.n_gpus}")
    print(f"    IID                     : {args.iid}")
    print(f"    Random Seed             : {args.seed}")
    print(f"    Test Fraction           : {args.test_fraction}")
    print(f"    Save Path               : {args.save_path}")
    print(f"    Data Path               : {args.data_path}")

    print()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
