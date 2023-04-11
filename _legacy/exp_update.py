#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import copy

import numpy as np


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, device, dataset, idxs):
        self.args = args
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = device
        # self.device = 'cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu'
        # self.device = 'cuda' if args.gpu else 'cpu'
        if args.supervision:
            # Supervised learning
            # Default criterion set to NLL loss function
            self.criterion = nn.NLLLoss().to(self.device)
        else:
            # Self-Supervised learning
            self.criterion = nn.MSELoss().to(self.device)
    

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader


    def update_weights(self, model, global_round, global_grad, global_grad_ratio):
        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        if self.args.supervision:
            # Supervised learning
            model.zero_grad()
            grad_avg = copy.deepcopy(list(model.parameters()))

            total_batches = 0

            for iter in range(self.args.local_ep):
                losses = []

                for batch_idx, (images, labels) in enumerate(self.trainloader):

                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    for param_avg, param_global, param in zip(grad_avg, global_grad, model.parameters()):
                        if param_global.grad is not None:
                            param.grad = param.grad * (1 - global_grad_ratio) + param_global.grad * global_grad_ratio
                        if param_avg.grad is None:
                            param_avg.grad = param.grad
                        else:
                            param_avg.grad += param.grad
                    total_batches += 1
                    optimizer.step()
                    

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round + 1, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    losses.append(loss.item() / images.shape[0])
            
            for param in grad_avg:
                param.grad /= total_batches

            return model.state_dict(), sum(losses) / len(losses), grad_avg

        else:
            # Self-Supervised learning
            out = []
            for iter in range(self.args.local_ep):
                losses = []
                for batch_idx, (images, _) in enumerate(self.trainloader):
                    images = images.to(self.device)

                    model.zero_grad()
                    outputs = model(images)
                    loss = self.criterion(outputs, images)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round + 1, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    losses.append(loss.item() / images.shape[0])
                out.append((iter, images, outputs),)

            return model.state_dict(), sum(losses) / len(losses), out


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()

        if self.args.supervision:
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
        dataloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=int(len(idxs)/10), shuffle=False)

        if self.args.supervision:
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


def test_inference(args, device, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    test_batch_size = 256
    trimmed_test_dataset = Subset(test_dataset, range(int(len(test_dataset) * args.test_fraction)))
    testloader = DataLoader(trimmed_test_dataset, batch_size=test_batch_size, shuffle=False)

    model.eval()

    if args.supervision:
        # Supervised learning
        # Default criterion set to NLL loss function
        criterion = nn.NLLLoss().to(device)
    else:
        # Self-Supervised learning
        criterion = nn.MSELoss().to(device)

    if args.supervision:
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