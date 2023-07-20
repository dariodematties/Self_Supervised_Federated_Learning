#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from scipy.stats import dirichlet
from tqdm import tqdm
from rich.console import Console


def dirichlet_sampling(
    dataset,
    num_users=100,
    sample_ratio=0.8,
    alpha=1,
    beta=0,
    subset=False,
    print_labels=False,
    num_total_samples=None
):
    """
    Sample client data by drawing populations from a Dirichlet distribution
    For more details, see https://arxiv.org/pdf/1909.06335.pdf

    Args:
        dataset (Dataset): the dataset from which to sample
        num_users (int): the number of clients participating in FL
        sample_ratio (float): the ratio of the total number of samples in the
            dataset from which user samples will be allocated
        alpha (float): concentration parameter controlling identicalness among clients
            (alpha->0 is highly non-IID; alpha->inf is highly IID)
        beta (float): the ratio of the total number of samples to include in the shared
            pool
        subset (bool): if True, no shared samples will be provided to the client which
            have the client's most prevalent label
        print_labels (bool): whether or not to print to the console the labels given to
            each client (supported only for 10-class datasets)
    """
    dict_users = {}

    alphas = np.array([alpha * 0.1] * 10)
    labels = np.array(dataset.targets)
    selected = np.array([0] * len(dataset))
    all_samples = np.arange(len(dataset))
    
    if num_total_samples is None:
        num_total_samples = int(sample_ratio * len(dataset))
    num_unshared_samples = num_total_samples * (1 - beta)
    num_shared_samples = num_total_samples * beta

    # Client Samples
    num_samples_per_user = num_unshared_samples // num_users
    if print_labels:
        print("Performing client sampling...")
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

    if print_labels:
        _print_labels(labels, dict_users)

    return dict_users


def dominant_label_sampling(
    dataset,
    num_users=100,
    sample_ratio=0.8,
    num_labels=None,
    beta=0,
    gamma=0.8,
    subset=False,
    print_labels=True,
):
    """
    For each label, assign num_users / num_labels clients whose sample distribution is
    dominated by that label, with the ratio occupied by the dominant label specified by
    gamma.

    Args:
        dataset (Dataset): the dataset from which to sample
        num_users (int): the number of clients participating in FL
        sample_ratio (float): the ratio of the total number of samples in the
            dataset from which user samples will be allocated
        num_labels (int): the number of dominant labels to use; if None, all
            labels will be used as dominant labels
        beta (float): the ratio of the total number of samples to include in the shared
            pool
        gamma (float): the ratio of a client's samples that should have the dominant
            label(s)
        subset (bool): if True, no shared samples will be provided to the client which
            have the client's most prevalent label
        print_labels (bool): whether or not to print to the console the labels given to
            each client (supported only for 10-class datasets)
    """
    dict_users = {}

    labels = np.array(dataset.targets)
    selected = np.array([0] * len(dataset))
    all_samples = np.arange(len(dataset))

    num_total_samples = int(sample_ratio * len(dataset))
    num_unshared_samples = num_total_samples * (1 - beta)
    num_shared_samples = num_total_samples * beta

    if num_labels is None:
        num_labels = len(np.unique(labels))

    # Client Samples
    num_samples_per_user = num_unshared_samples // num_users
    print("Performing client sampling...")
    for i in tqdm(range(num_users)):
        dominant_label = i % num_labels

        num_dominant_samples = int(num_samples_per_user * gamma)
        num_non_dominant_samples = int(
            (num_samples_per_user * (1 - gamma)) // (num_labels - 1)
        )
        num_remaining_samples = num_samples_per_user - (
            num_dominant_samples + num_non_dominant_samples * (num_labels - 1)
        )

        num_samples_per_label = [
            (
                num_dominant_samples
                if label == dominant_label
                else num_non_dominant_samples
            )
            for label in range(num_labels)
        ]
        remaining_samples = np.random.choice(
            range(num_labels), num_remaining_samples, replace=False
        )
        for sample in remaining_samples:
            num_samples_per_label[sample] += 1

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

    if print_labels:
        _print_labels(labels, dict_users)

    return dict_users


def missing_label_sampling(
    dataset,
    num_users,
    beta=0,
    num_missing_labels=2,
    subset=False,
    print_labels=True,
):
    """
    Perform sampling such that each client is missing num_missing_labels labels.

    Args:
        dataset (Dataset): the dataset from which to sample
        num_users (int): the number of clients participating in FL
        beta (float): the ratio of the total number of samples to include in the shared
            pool
        num_missing_labels (int): the number of labels each client should be missing
        subset (bool): if True, no shared samples will be provided to the client which
            have the client's most prevalent label
        print_labels (bool): whether or not to print to the console the labels given to
            each client (supported only for 10-class datasets)
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
    print("Performing client sampling...")
    for i in tqdm(range(num_users)):
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

    if print_labels:
        _print_labels(labels, dict_users)

    return dict_users


def iid_sampling(dataset, num_users, print_labels=True):
    """
    Sample IID client data from the provided dataset.

    Assigns an random subset of the remaining samples to each user.

    Args:
        dataset (Dataset): the dataset from which to sample
        num_users (int): the number of clients participating in FL
        print_labels (bool): whether or not to print to the console the labels given to
            each client
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    try:
        labels = dataset.targets.numpy()
    except AttributeError:
        # For CIFAR-10, dataset.targets seems to be a list.
        labels = np.array(dataset.targets)
    print("Performing client sampling...")
    for i in tqdm(range(num_users)):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    if print_labels:
        _print_labels(labels, dict_users)

    return dict_users


def non_iid_sampling(dataset, num_users, print_labels=True):
    """
    Sample non-IID client data from provided dataset.

    Divides the dataset into 2 * num_users shards and assigns 2 shards for each users.
    For num_users = 100, this reduces to the scheme used to generate non-IID data in
    McMahan et al. (2017).

    Args:
        dataset (Dataset): the dataset from which to sample
        num_users (int): the number of clients participating in FL
        print_labels (bool): whether or not to print to the console the labels given to
            each client
    """
    num_imgs = 60_000
    num_shards = 2 * num_users
    num_imgs_per_shard = num_imgs // num_shards

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs_per_shard)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    print("Performing client sampling...")
    for i in tqdm(range(num_users)):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (
                    dict_users[i],
                    idxs[rand * num_imgs_per_shard : (rand + 1) * num_imgs_per_shard],
                ),
                axis=0,
            )

    if print_labels:
        _print_labels(labels, dict_users)

    return dict_users


def _check_for_duplicates(dict_users):
    """Checks dictionary of user sample indices to ensure that a given sample belongs
    to at most one user.

    Args:
        dict_users (dict): a dictionary mapping user/client indices to sample indices
    """

    for i, user_samples in dict_users.items():
        user_samples = set(user_samples)
        for j, user_samples_compare in dict_users.items():
            user_samples_compare = set(user_samples_compare)
            if (len(user_samples.intersection(user_samples_compare)) > 0) and i != j:
                return False
    return True


def _print_labels(labels, dict_users):
    """Visualizes client label distributions in the console."""

    console = Console()
    colors = [9, 208, 184, 40, 85, 14, 21, 129, 181, 188]

    for i in range(len(dict_users)):
        user_labels = list(
            map(lambda n: np.count_nonzero(labels[list(dict_users[i])] == n), range(10))
        )
        percentages = _to_percentages(user_labels)

        print(f"User {i}: ", end="")
        for i, percentage in enumerate(percentages):
            console.print(" " * percentage, style=f"on color({colors[i]})", end="")
        print(f" [Total Samples: {sum(user_labels)}]")
        print("\n")


def _to_percentages(nums):
    """Converts a list of numbers to percentages, which are guaranteed to sum to 100%.

    Args:
        nums (list of nums): the list of numbers to convert to percentages

    >>> nums = [1, 1, 1]
    >>> _to_percentages(nums)
    [34, 33, 33]
    >>> nums = [1, 1, 1, 1, 1, 1]
    >>> _to_percentages(nums)
    [17, 17, 17, 17, 16, 16]
    >>> nums = [4, 2, 2]
    >>> _to_percentages(nums)
    [50, 25, 25]
    """

    percentages = np.array(nums) * 100 / sum(nums)
    f, i = np.modf(percentages)
    diff = 100 - np.sum(i)
    while diff > 0:
        max_idx = np.argmax(f)
        i[max_idx] += 1
        f[max_idx] = 0
        diff = 100 - np.sum(i)
    return list(i.astype(int))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
