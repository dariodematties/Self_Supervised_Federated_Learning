#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torch import nn
import torch.nn.functional as F


class MLPMnist(nn.Module):
    """The MLP used for MNIST in McMahan et al. (2017)

    >>> model = MLPMnist()
    >>> get_parameter_count(model)
    199210
    """

    def __init__(self):
        super(MLPMnist, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)


class CNNMnist(nn.Module):
    """The MLP used for MNIST in McMahan et al. (2017)

    >>> model = CNNMnist()
    >>> get_parameter_count(model)
    1663370
    """

    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, padding=1)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, padding=1)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class CNNCifar(nn.Module):
    """A CNN similar to the one used for CIFAR-10 in McMahan et al. (2017).

    Note that McMahan et al. provide a link to a TensorFlow tutorial from which
    their network comes, but the link is dead. We use a network from a similar
    (and possibly the same) tutorial found here:
    https://www.tensorflow.org/tutorials/images/cnn.

    >>> model = CNNCifar()
    >>> get_parameter_count(model)
    122570
    """

    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2)
        x = self.conv3(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_parameter_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == "__main__":
    import doctest

    doctest.testmod()
