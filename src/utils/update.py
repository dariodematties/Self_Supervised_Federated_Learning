#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(
        self, dataset, idxs, local_ep, local_bs, lr, optimizer, supervision, device
    ):
        self.local_ep = local_ep
        self.local_bs = local_bs
        self.lr = lr
        self.optimizer = optimizer
        self.supervision = supervision
        self.device = device

        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs)
        )

        if self.supervision:
            # Supervised learning
            # Default criterion set to CrossEntropy loss function
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            # Self-Supervised learning
            self.criterion = nn.MSELoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[: int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)) :]

        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train), batch_size=self.local_bs, shuffle=True
        )
        validloader = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=int(len(idxs_val) / 10),
            shuffle=False,
        )
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=int(len(idxs_test) / 10),
            shuffle=False,
        )
        return trainloader, validloader, testloader

    def update_weights(self, model):
        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.5)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr
            )

        if self.supervision:
            # Supervised learning
            for iter in range(self.local_ep):
                losses = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item() / images.shape[0])
            return model.state_dict(), sum(losses) / len(losses)

        else:
            # Self-Supervised learning
            out = []
            for iter in range(self.local_ep):
                losses = []
                for batch_idx, (images, _) in enumerate(self.trainloader):
                    images = images.to(self.device)

                    model.zero_grad()
                    outputs = model(images)
                    loss = self.criterion(outputs, images)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item() / images.shape[0])
                out.append(
                    (iter, images, outputs),
                )

            return model.state_dict(), sum(losses) / len(losses), out

    def inference(self, model):
        """Returns the inference accuracy and loss."""

        model.eval()

        if self.supervision:
            # Supervised learning
            losses = []
            total, correct = 0.0, 0.0

            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item() / images.shape[0])

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct / total
            loss = sum(losses) / len(losses)
            return accuracy, loss

        else:
            # Self-Supervised learning
            losses = []
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images = images.to(self.device)

                # Inference
                outputs = model(images)
                loss = self.criterion(outputs, images)
                losses.append(loss.item() / images.shape[0])

            loss = sum(losses) / len(losses)
            return None, loss

    def process_samples(self, dataset, idxs, model):
        dataloader = DataLoader(
            DatasetSplit(dataset, idxs), batch_size=int(len(idxs) / 10), shuffle=False
        )

        if self.supervision:
            # Supervised learning
            output_list = []
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                output_list.extend(outputs)

        else:
            # Self-Supervised learning

            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)

                # Inference
                outputs = model(images)
                output_list.extend(outputs)

        return output_list


def test_inference(supervision, device, model, test_dataset, test_fraction=1):
    """Returns the test accuracy and loss."""

    test_batch_size = 256
    trimmed_test_dataset = Subset(
        test_dataset, range(int(len(test_dataset) * test_fraction))
    )
    testloader = DataLoader(
        trimmed_test_dataset, batch_size=test_batch_size, shuffle=False
    )

    model.eval()

    if supervision:
        # Supervised learning
        # Default criterion set to CrossEntropy loss function
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        # Self-Supervised learning
        criterion = nn.MSELoss().to(device)

    if supervision:
        # Supervised learning
        losses = []
        total, correct = 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item() / images.shape[0])

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        loss = sum(losses) / len(losses)

        return accuracy, loss

    else:
        # Self-Supervised learning
        losses = []
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)

            # Inference
            outputs = model(images)
            loss = criterion(outputs, images)
            losses.append(loss.item() / images.shape[0])

        loss = sum(losses) / len(losses)
        return None, loss
