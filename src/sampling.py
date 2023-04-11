#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from scipy.stats import dirichlet
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


def dirichlet_sampling(
    dataset,
    num_users=100,
    num_samples=30_000,
    alpha=1,
    beta=0,
    subset=False,
    print_labels=True,
    save_path=None,
):
    """
    Sample client data by drawing populations from a Dirichlet distribution
    For more details, see https://arxiv.org/pdf/1909.06335.pdf

    Parameters
    ----------
    dataset: Dataset
        the user for which to perform local training
    num_users: int
        the number of clients participating in FL
    num_samples: int
        the total number of samples to allocate
    alpha: float
        concentration parameter controlling identicalness among clients
        (alpha->0 is highly non-IID; alpha->inf is highly IID)
    beta: float
        the ratio of the total number of samples to include in the shared pool
    subset: bool
        if True, no shared samples will be provided to the client which have the
        client's most prevalent label
    print_labels: bool
        whether or not to print to the console the labels given to each client
    """
    dict_users = {}

    alphas = np.array([alpha * 0.1] * 10)
    labels = np.array(dataset.targets)
    selected = np.array([0] * len(dataset))
    all_samples = np.arange(len(dataset))

    num_unshared_samples = num_samples * (1 - beta)
    num_shared_samples = num_samples * beta

    # Client Samples
    num_samples_per_user = num_unshared_samples // num_users
    for i in range(num_users):
        num_samples_per_label = np.rint(
            (dirichlet.rvs(alphas, size=1)[0] * num_samples_per_user)
        ).astype(int)
        user_samples = set()
        for label, num_samples in enumerate(num_samples_per_label):
            label_samples = all_samples[(selected == 0) & (labels == label)]
            samples = np.random.choice(label_samples, num_samples, replace=False)
            selected[samples] = 1
            user_samples.update(samples)
        dict_users[i] = user_samples

    # Shared Samples
    num_samples_per_label = [int(0.1 * num_shared_samples)] * 10
    max_users = {
        i: sorted(
            set(labels[list(dict_users[i])]),
            key=list(labels[list(dict_users[i])]).count,
            reverse=True,
        )[:1]
        for i in range(num_users)
    }
    for label, num_samples in enumerate(num_samples_per_label):
        label_samples = all_samples[(selected == 0) & (labels == label)]
        samples = np.random.choice(label_samples, num_samples, replace=False)
        selected[samples] = 1
        for i in range(num_users):
            if not (subset and label in max_users[i]):
                dict_users[i].update(samples)

    # Print Labels
    if print_labels:
        for i in range(num_users):
            print(
                f"Labels for User {i}:"
                f" {list(map(lambda n: np.count_nonzero(labels[list(dict_users[i])] == n), range(10)))}"
            )

    if save_path is not None:
        user_labels = [
            [
                i,
                *list(
                    map(
                        lambda n: np.count_nonzero(labels[list(dict_users[i])] == n),
                        range(10),
                    )
                ),
            ]
            for i in range(num_users)
        ]
        plot_label_distribution(user_labels, save_path)

    return dict_users


def dominant_label_sampling(
    dataset,
    num_users=100,
    num_samples=30_000,
    beta=0,
    gamma=0.8,
    subset=False,
    print_labels=True,
    save_path=None,
):
    """
    For each label, assign num_users / num_labels clients whose sample distribution is
    dominated by that label, with the ratio occupied by the dominant label specified by
    gamma.

    Parameters
    ----------
    dataset: Dataset
        the user for which to perform local training
    num_users: int
        the number of clients participating in FL
    beta: float
        the ratio of the total number of samples to include in the shared pool
    gamma: float
        the ratio of a client's samples that should have the dominant label(s)
    subset: bool
        if True, no shared samples will be provided with the client's dominant label(s)
    print_labels: bool
        whether or not to print to the console the labels given to each client
    """
    dict_users = {}

    labels = np.array(dataset.targets)
    selected = np.array([0] * len(dataset))
    all_samples = np.arange(len(dataset))

    num_labels = 10
    num_unshared_samples = num_samples * (1 - beta)
    num_shared_samples = num_samples * beta

    # Client Samples
    num_samples_per_user = num_unshared_samples // num_users
    for i in range(num_users):
        dominant_num_samples = int(num_samples_per_user * gamma)
        other_num_samples = int(
            (num_samples_per_user * (1 - gamma)) // (num_labels - 1)
        )
        user_samples = set()
        dominant_label = i % num_labels
        num_samples_per_label = [
            dominant_num_samples if label == dominant_label else other_num_samples
            for label in range(10)
        ]
        for label, num_samples in enumerate(num_samples_per_label):
            label_samples = all_samples[(selected == 0) & (labels == label)]
            samples = np.random.choice(label_samples, num_samples, replace=False)
            selected[samples] = 1
            user_samples.update(samples)
        dict_users[i] = user_samples

    # Shared Samples
    num_samples_per_label = [int(0.1 * num_shared_samples)] * 10
    max_users = {
        i: sorted(
            set(labels[list(dict_users[i])]),
            key=list(labels[list(dict_users[i])]).count,
            reverse=True,
        )[:1]
        for i in range(num_users)
    }
    for label, num_samples in enumerate(num_samples_per_label):
        label_samples = all_samples[(selected == 0) & (labels == label)]
        samples = np.random.choice(label_samples, num_samples, replace=False)
        selected[samples] = 1
        for i in range(num_users):
            if not (subset and label in max_users[i]):
                dict_users[i].update(samples)

    # Print Labels
    if print_labels:
        for i in range(num_users):
            print(
                f"Labels for User {i}:"
                f" {list(map(lambda n: np.count_nonzero(labels[list(dict_users[i])] == n), range(10)))}"
            )

    if save_path is not None:
        user_labels = [
            [
                i,
                *list(
                    map(
                        lambda n: np.count_nonzero(labels[list(dict_users[i])] == n),
                        range(10),
                    )
                ),
            ]
            for i in range(num_users)
        ]
        plot_label_distribution(user_labels, save_path)

    return dict_users


def missing_label_sampling(
    dataset,
    num_users,
    beta=0,
    num_missing_labels=2,
    subset=False,
    print_labels=True,
    save_path=None,
):
    """
    For each label, assign num_users / num_labels clients whose sample distribution is
    dominated by that label, with the ratio occupied by the dominant label specified by
    gamma.

    Parameters
    ----------
    dataset: Dataset
        the user for which to perform local training
    num_users: int
        the number of clients participating in FL
    beta: float
        the ratio of the total number of samples to include in the shared pool
    num_missing_labels: int
        the number of labels each client should be missing
    subset: bool
        if True, no shared samples will be provided with the client's dominant label(s)
    print_labels: bool
        whether or not to print to the console the labels given to each client
    """
    dict_users = {}

    labels = np.array(dataset.targets)
    selected = np.array([0] * len(dataset))
    all_samples = np.arange(len(dataset))

    num_samples = 50_000
    num_labels = 10
    num_unshared_samples = num_samples * (1 - beta)
    num_shared_samples = num_samples * beta

    # Client Samples
    num_samples_per_user = num_unshared_samples // num_users
    for i in range(num_users):
        num_samples_per_label = int(
            num_samples_per_user // (num_labels - num_missing_labels)
        )
        user_samples = set()
        missing_labels = [
            (num_missing_labels * i + j) % num_labels for j in range(num_missing_labels)
        ]
        num_samples_per_label = [
            0 if label in missing_labels else num_samples_per_label
            for label in range(10)
        ]
        for label, num_samples in enumerate(num_samples_per_label):
            label_samples = all_samples[(selected == 0) & (labels == label)]
            samples = np.random.choice(label_samples, num_samples, replace=False)
            selected[samples] = 1
            user_samples.update(samples)
        dict_users[i] = user_samples

    # Shared Samples
    num_samples_per_label = [int(0.1 * num_shared_samples)] * 10
    max_users = {
        i: sorted(
            set(labels[list(dict_users[i])]),
            key=list(labels[list(dict_users[i])]).count,
            reverse=True,
        )[:8]
        for i in range(num_users)
    }
    for label, num_samples in enumerate(num_samples_per_label):
        label_samples = all_samples[(selected == 0) & (labels == label)]
        samples = np.random.choice(label_samples, num_samples, replace=False)
        selected[samples] = 1
        for i in range(num_users):
            if not (subset and label in max_users[i]):
                dict_users[i].update(samples)

    # Print Labels
    if print_labels:
        for i in range(num_users):
            print(
                f"Labels for User {i}:"
                f" {list(map(lambda n: np.count_nonzero(labels[list(dict_users[i])] == n), range(10)))}"
            )

    if save_path is not None:
        user_labels = [
            [
                i,
                *list(
                    map(
                        lambda n: np.count_nonzero(labels[list(dict_users[i])] == n),
                        range(10),
                    )
                ),
            ]
            for i in range(num_users)
        ]
        plot_label_distribution(user_labels, save_path)

    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset) / num_users)
    num_items = 300
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    labels = dataset.targets.numpy()
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        print(
            f"Labels for User {i}:"
            f" {list(map(lambda n: np.count_nonzero(labels[list(dict_users[i])] == n), range(10)))}"
        )
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_users)
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    )
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def plot_label_distribution(user_labels, save_path):
    print("Plotting label distribution...")

    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    df = pd.DataFrame(user_labels, columns=["User", *[f"Label {i}" for i in range(10)]])
    df.plot.barh(stacked=True, width=0.9, legend=None)

    plt.xlabel("Number of Samples")
    plt.ylabel("User")

    plt.savefig(save_path)

    print(f"Label distribution saved at {save_path}.")


if __name__ == "__main__":
    dataset_train = datasets.MNIST(
        "./data/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    num = 100
    d = mnist_noniid(dataset_train, num)
