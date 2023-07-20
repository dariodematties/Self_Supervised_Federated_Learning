#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser(default=False):
    """Parses command-line arguments and returns argument Namespace.

    Args:
        default (bool): Ignores command-line arguments and returns Namespace with
            default arguments. Useful for Jupyter notebooks.
    """

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
        help=(
            "n_steps parameter for PPO algorithm; determines number of steps before"
            " policy network receives a gradient update"
        ),
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
        default=15_000,
        help="total timesteps for training policy network, per parallel environment",
    )
    parser.add_argument(
        "--target_accuracy",
        type=float,
        default=0.95,
        help="target accuracy for global model",
    )

    # federated arguments
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=1, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=32, help="local batch size: B")
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
    parser.add_argument("--arch", type=str, default="cnn", help="model name")

    # other arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="name of dataset",
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
    parser.add_argument(
        "--iid",
        type=int,
        default=0,
        help="Default set to non-IID. Set to 1 for IID.",
    )
    parser.add_argument(
        "--dirichlet",
        type=bool,
        default=True,
        help="whether to use a Dirichlet distribution to draw samples for each user",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help=(
            "the alpha value governing the non-IID nature of the Dirichlet distribution"
        ),
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=1,
        help=(
            "fraction of test dataset to use for test inference (use a smaller value to"
            " speed up experiments)"
        ),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../../save",
        help="the path for saving and loading experiment data",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data",
        help="the path for saving and loading datasets",
    )

    if default:
        args = parser.parse_args("")
    else:
        args = parser.parse_args()
    return args
