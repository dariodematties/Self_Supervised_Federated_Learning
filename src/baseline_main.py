#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# How to run it?
# Just do: python src/baseline_main.py --model=autoencoder --dataset=mnist --epochs=20 --gpu=0 --local_bs 64 --lr 1e-3 --optimizer adam

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST

if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    #train_dataset = list(train_dataset)[:4096]
    #test_dataset = list(test_dataset)[:4096]

    # BUILD MODEL
    if args.supervision:
        # Supervised learning
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_model = CNNCifar(args=args)
        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=args.num_classes)
        else:
            exit('Error: unrecognized supervised model')
    else:
        # Self supervised learning
        if args.model == 'autoencoder':
            # Transpose Convolution and Autoencoder
            if args.dataset == 'mnist':
                global_model = AutoencoderMNIST(args=args)

        else:
            exit('Error: unrecognized unsupervised model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    if args.supervision:
        # Supervised learning
        criterion = torch.nn.NLLLoss().to(device)
        epoch_loss = []

        for epoch in tqdm(range(args.epochs)):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            epoch_loss.append(loss_avg)

        # Plot loss
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                     args.epochs))

        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
    else:
        # Self-Supervised learning
        criterion = torch.nn.MSELoss().to(device)
        epoch_loss = []
        Outputs = []

        for epoch in tqdm(range(args.epochs)):
            batch_loss = []

            for batch_idx, (images, _) in enumerate(trainloader):
                images = images.to(device)

                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            epoch_loss.append(loss_avg)
            Outputs.append((epoch, images, outputs),)

        # Plot loss
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                     args.epochs))
        # Plot outputs
        for k in range(0, args.epochs, 5):
            plt.figure(figsize=(9, 2))
            imgs = Outputs[k][1].cpu().detach().numpy()
            recon = Outputs[k][2].cpu().detach().numpy()
            for i, item in enumerate(imgs):
                if i >= 9: break
                plt.subplot(2, 9, i+1)
                plt.axis('off')
                plt.imshow(item[0])
                
            for i, item in enumerate(recon):
                if i >= 9: break
                plt.subplot(2, 9, 9+i+1)
                plt.axis('off')
                plt.imshow(item[0])

            plt.savefig('./image_{}.png'.format(k))
            plt.close()

