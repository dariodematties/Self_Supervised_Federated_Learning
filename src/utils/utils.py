#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import csv
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .models import MLPMnist, CNNMnist, CNNCifar, AutoencoderMNIST
from .resnet import resnet50


def get_model(arch, dataset, device="cpu"):
    """Returns the model for a particular architecture and dataset, on the specified
    device.

    Args:
        arch (str): the name of the architecture (either "cnn" or "mlp")
        dataset (str): the name of the dataset (either "mnist", "cifar", or "imagenet")
        device (str): the device to keep the model on

    >>> model = get_model("cnn", "mnist")
    >>> model = get_model("cnn", "cifar")
    >>> model = get_model("mlp", "mnist")
    >>> model = get_model("resnet", "imagenet")
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
    elif arch == "resnet":
        if dataset == "imagenet":
            model, _ = resnet50()
    elif arch == "autoencoder":
        if dataset == "mnist":
            model = AutoencoderMNIST()
        else:
            raise ValueError(
                f"architecture '{arch}' not supported for '{dataset}' dataset"
            )

    else:
        raise ValueError(f"unrecognized architecture '{arch}'")

    model.to(device)
    model.train()

    return model


def get_train_test(dataset, download=False, dataset_dir=None):
    """Returns train and test dataset splits for a given dataset, with appropriate
    normalization applied.

    Args:
        dataset (str): the name of the dataset (either "mnist", "cifar", or "imagenet")
        download (bool): if True, downloads dataset from the Internet
        dataset_dir (str): the path to the dataset; required for ImageNet

    >>> train_dataset, test_dataset = get_train_test("mnist")
    >>> train_dataset, test_dataset = get_train_test("cifar")
    >>> train_dataset, test_dataset = get_train_test("fake_dataset")
    Traceback (most recent call last):
     ...
    ValueError: dataset 'fake_dataset' not supported
    >>> train_dataset, test_dataset = get_train_test("imagenet")
    Traceback (most recent call last):
     ...
    ValueError: to use ImageNet, you must specify the dataset_dir argument
    """

    if dataset_dir is None:
        if dataset == "imagenet":
            raise ValueError(
                "to use ImageNet, you must specify the dataset_dir argument"
            )
        dataset_dir = f"../../data/{dataset}/"

    if dataset == "mnist":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            dataset_dir, train=True, download=download, transform=apply_transform
        )
        test_dataset = datasets.MNIST(
            dataset_dir, train=False, download=download, transform=apply_transform
        )
    elif dataset == "cifar":
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            dataset_dir, train=True, download=download, transform=apply_transform
        )
        test_dataset = datasets.CIFAR10(
            dataset_dir, train=False, download=download, transform=apply_transform
        )
    elif dataset == "imagenet":
        if dataset_dir is None:
            raise ValueError("")
        apply_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_dataset = datasets.ImageFolder(
            os.path.join(dataset_dir, "train"), apply_transform
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(dataset_dir, "val"), apply_transform
        )
    else:
        raise ValueError(f"dataset '{dataset}' not supported")

    return train_dataset, test_dataset


def get_dataset_and_label_names(dataset):
    """Gives the stylized name and label names for each of the classes in the specified
    dataset.

    Args:
        dataset (str): the name of the dataset (either "mnist", "cifar", or "imagenet")

    >>> dataset_name, label_names = get_dataset_and_label_names("imagenet")
    >>> dataset_name
    'ImageNet'
    >>> len(label_names)
    1000
    """
    if dataset == "mnist":
        dataset_name = "MNIST"
        label_names = [str(i) for i in range(10)]
    elif dataset == "cifar":
        dataset_name = "CIFAR-10"
        label_names = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif dataset == "imagenet":
        dataset_name = "ImageNet"
        with open("utils/imagenet_labels.txt") as f:
            label_names = f.read().splitlines()
    return dataset_name, np.array(label_names)


def build_coarse_label_mapping():
    """Returns a list that maps ImageNet labels to coarse labels across 67 classes.

    Coarse labels are assigned according to
    https://github.com/noameshed/novelty-detection/blob/master/imagenet_categories.csv.

    >> fine_to_coarse = build_coarse_label_mapping()
    >> fine_to_coarse[18]
    4
    >> fine_to_coarse[757]
    48
    """

    fine_label_to_idx = {}
    fine_to_coarse = {}

    with open("utils/imagenet_labels.txt") as f:
        fine_labels = f.read().splitlines()
        for idx, fine_label in enumerate(fine_labels):
            fine_label = fine_label.split(",")[0]
            fine_label_to_idx[fine_label] = idx + 1

    with open("utils/imagenet_labels_coarse.csv", "r") as csvfile:
        reader = csv.reader(csvfile)

        # Skip the header row
        next(reader)

        for idx, row in enumerate(reader):
            coarse_idx = idx + 1
            for fine_label in row[1:]:
                if fine_label == "":
                    break
                fine_label = fine_label.replace("_", " ")
                fine_idx = fine_label_to_idx[fine_label]
                fine_to_coarse[fine_idx] = coarse_idx

    return fine_to_coarse


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
