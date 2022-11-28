#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu'
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

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        if self.args.supervision:
            # Supervised learning
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        else:
            # Self-Supervised learning
            Outputs = []
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, _) in enumerate(self.trainloader):
                    images = images.to(self.device)

                    model.zero_grad()
                    outputs = model(images)
                    loss = self.criterion(outputs, images)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                Outputs.append((iter, images, outputs),)

            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), Outputs

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if self.args.supervision:
            # Supervised learning
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct/total
        else:
            # Self-Supervised learning
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images = images.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, images)
                loss += batch_loss.item()

                total += len(labels)

            accuracy = loss/total
        return accuracy, loss
    
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
                


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda' if args.gpu else 'cpu'
    #criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)



    if args.supervision:
        # Supervised learning
        # Default criterion set to NLL loss function
        criterion = nn.NLLLoss().to(device)
    else:
        # Self-Supervised learning
        criterion = nn.MSELoss().to(device)



    if args.supervision:
        # Supervised learning
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
    else:
        # Self-Supervised learning
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, images)
            loss += batch_loss.item()

            total += len(labels)

        accuracy = loss/total
    return accuracy, loss
