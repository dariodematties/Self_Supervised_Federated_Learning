#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # wandb logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether or not to use logging with wandb",
    )

    # reinforcement arguments
    parser.add_argument(
        "--ppo_n_steps",
        type=int,
        default=256,
        help="n_steps parameter for PPO algorithm; determines number of steps before policy network receives a gradient update",
    )
    parser.add_argument(
        "--ppo_lr",
        type=float,
        default=0.0003,
        help="learning rate for PPO algorithm.",
    )
    parser.add_argument(
        "--ppo_gamma",
        type=float,
        default=0.9,
        help="discount factor for PPO algorithm.",
    )
    parser.add_argument(
        "--ppo_bs",
        type=int,
        default=16,
        help="batch size for PPO algorithm.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5000,
        help="total timesteps for training policy network, per parallel environment",
    )
    parser.add_argument(
        "--target_accuracy",
        type=float,
        default=0.875,
        help="target accuracy for global model",
    )

    # federated arguments
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument("--frac", type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument("--local_ep", type=int, default=5, help="the number of local epochs: E")
    parser.add_argument("--local_bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="type of optimizer",
    )

    # model arguments
    parser.add_argument(
        "--supervision",
        action="store_true",
        help="Whether use supervised models rather than self-supervised ones",
    )
    parser.add_argument("--model", type=str, default="cnn", help="model name")
    parser.add_argument("--kernel_num", type=int, default=9, help="number of each kind of kernel")
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="3,4,5",
        help="comma-separated kernel size to use for convolution",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="number of channels of imgs",
    )
    parser.add_argument("--norm", type=str, default="batch_norm", help="batch_norm, layer_norm, or None")
    parser.add_argument(
        "--num_filters",
        type=int,
        default=32,
        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.",
    )
    parser.add_argument(
        "--max_pool",
        type=str,
        default="True",
        help="Whether use max pooling rather than strided convolutions",
    )

    # other arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="name \
                        of dataset",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number of classes",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="To use cuda, set to a specific GPU ID. Default set to use CPU.",
    )
    parser.add_argument(
        "--n_gpus",
        default=1,
        type=int,
        help="The number of GPUs to use for training.",
    )
    parser.add_argument("--iid", type=int, default=1, help="Default set to IID. Set to 0 for non-IID.")
    parser.add_argument(
        "--unequal",
        type=int,
        default=0,
        help="whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)",
    )
    parser.add_argument("--stopping_rounds", type=int, default=10, help="rounds of early stopping")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=1,
        help="fraction of test dataset to use for test inference (use a smaller value to speed up experiments)",
    )

    args = parser.parse_args()
    return args
