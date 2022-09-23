#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST
from utils import get_dataset, average_weights, exp_details

from mpi4py import MPI
from mpi_communication import gather_weights, gather_losses, bcast_state_dict

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    args.rank=rank
    args.size=size

    if args.rank == 0:
        print('The size of the environment is {}' .format(args.size))

    print('Here rank {} saying hello!' .format(args.rank))

    args.gpu=rank
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    # device = 'cuda:'+str(args.gpu) if args.gpu else 'cpu'
    device = 'cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu'

    print('From rank {} device is {} and gpu number is {}' .format(args.rank, device, args.gpu))

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

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
            exit('Error: unrecognized model')
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

    if args.rank == 0:
        print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # local_outputs = []
        if args.rank==0:
            print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for i in range(len(idxs_users)):
            idx=idxs_users[i]

            if i%args.size==args.rank:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                if args.supervision:
                    # Supervised learning
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                else:
                    # Self-Supervised learning
                    w, loss, out = local_model.update_weights(
                            model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                    # local_outputs.append(out)

        local_weights=gather_weights(comm, local_weights, args)
        local_losses=gather_losses(comm, local_losses, args)
                    
        # update global weights
        if args.rank==0:
            global_weights = average_weights(local_weights)
        else:
            global_weights = {}

        if args.rank==0:
            number_of_keys=len(global_weights.keys())
        else:
            number_of_keys=None

        number_of_keys = comm.bcast(number_of_keys, root=0)

        keys=[]
        for key_number in range(number_of_keys):
            if args.rank==0:
                key=list(global_weights.keys())[key_number]
            else:
                key=None

            keys.append(comm.bcast(key, root=0))
            
        # bcast_state_dict
        global_weights = bcast_state_dict(comm, global_weights, keys, root=0)
 
        # update global weights
        global_model.load_state_dict(global_weights)

        if args.rank==0:
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)








        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for i in range(len(idxs_users)):
        # for c in range(args.num_users):
            idx=idxs_users[i]

            if i%args.size==args.rank:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)

        list_acc=gather_losses(comm, list_acc, args)
        list_loss=gather_losses(comm, list_loss, args)
        if args.rank==0:
            train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if args.rank==0:
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                if args.supervision:
                    print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                else:
                    print('Train MSE: {:.8f} \n'.format(train_accuracy[-1]))







    # Test inference after completion of training
    if args.rank==0:
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {args.epochs} global rounds of training:')
        if args.supervision:
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        else:
            print("|---- Avg Train MSE: {:.8f}".format(train_accuracy[-1]))
            print("|---- Test MSE: {:.8f}".format(test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # PLOTTING (optional)
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')

        # Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(args.dataset, args.model, args.epochs, args.frac,
                           args.iid, args.local_ep, args.local_bs))
        
        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                    format(args.dataset, args.model, args.epochs, args.frac,
                           args.iid, args.local_ep, args.local_bs))



        # # Plot outputs
        # #for k in range(0, args.epochs, 5):
        # for k in range(0, args.local_ep, 5):
            # plt.figure(figsize=(9, 2))
            # imgs = local_outputs[0][k][1].cpu().detach().numpy()
            # recon = local_outputs[0][k][2].cpu().detach().numpy()
            # for i, item in enumerate(imgs):
                # if i >= 9: break
                # plt.subplot(2, 9, i+1)
                # plt.axis('off')
                # plt.imshow(item[0])
                
            # for i, item in enumerate(recon):
                # if i >= 9: break
                # plt.subplot(2, 9, 9+i+1)
                # plt.axis('off')
                # plt.imshow(item[0])

            # plt.savefig('./image_{}.png'.format(k))
            # plt.close()

