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

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from models import AutoencoderMNIST
from utils import get_dataset, exp_details, average_weights

from mpi4py import MPI
from mpi_communication import gather_weights, gather_losses, bcast_state_dict
from rl_actions import SwapWeights, ShareWeights, ShareRepresentations

import wandb

if __name__ == '__main__':

    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up timer
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('..')
    
    # Parse, validate, and print arguments
    args = args_parser()

    if args.supervision:
        assert args.model in ["cnn", "mlp"]
    else:
        assert args.model in ["autoencoder"]
    
    args.rank = rank
    args.size = size
    args.gpu = rank        

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    device = 'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu'

    if args.rank == 0:
        exp_details(args)
        print(f'Environment size: {args.size}')

    print(f'[Rank {args.rank}] Checking in with device {device} and GPU number {args.gpu}.')

    # Load dataset
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Set up local models for each node
    local_models = {}
    local_losses = {}

    for i in range(args.num_users):
        if i % args.size == args.rank:
            if args.supervision:
                # Supervised learning
                if args.model == 'cnn':
                    # Convolutional neural netork
                    if args.dataset == 'mnist':
                        local_model = CNNMnist(args=args)
                    elif args.dataset == 'fmnist':
                        local_model = CNNFashion_Mnist(args=args)
                    elif args.dataset == 'cifar':
                        local_model = CNNCifar(args=args)

                elif args.model == 'mlp':
                    # Multi-layer preceptron
                    img_size = train_dataset[0][0].shape
                    len_in = 1
                    for x in img_size:
                        len_in *= x
                        local_model = MLP(dim_in=len_in, dim_hidden=64,
                                        dim_out=args.num_classes)
                else:
                    exit('Error: unrecognized model')
            else:
                # Self-supervised learning
                if args.model == 'autoencoder':
                    # Autoencoder with transpose convolutions
                    if args.dataset == 'mnist':
                        local_model = AutoencoderMNIST(args=args)

                else:
                    exit('Error: unrecognized unsupervised model')

            local_model.to(device)
            local_model.train()
            local_models[i] = local_model
            local_losses[i] = []

    if args.rank == 0:
        print()
        print("Model Information: ")
        print(local_models[0])
        print()
    
    wandb.init(group="federated_mpi_experiment", config=vars(args))

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    if args.rank == 0:
        pbar = tqdm(total=args.epochs)

    for epoch in range(args.epochs):
            
        # [ACTIONS]  
        # Decide on actions; here, rank 0 acts as RL agent
        actions = []
        if args.rank == 0:
            # Add actions to list; something like: 
            # actions.append(SwapWeights(2, 3))
            # actions.append(SwapWeights(7, 4))
            actions.append(ShareWeights(1, 2))
            actions.append(ShareWeights(3, 4))
            actions.append(ShareWeights(5, 6))
            actions.append(ShareWeights(7, 8))
            actions.append(ShareWeights(9, 10))
            # pass

        # Broadcast actions
        actions = comm.bcast(actions, root=0)

        # Perform actions
        for action in actions:

            if isinstance(action, SwapWeights):
                src_idxs = user_groups[action.src]
                user_groups[action.src] = user_groups[action.dest]
                user_groups[action.dest] = src_idxs

            elif isinstance(action, ShareRepresentations):
                if action.src % args.size == args.rank:
                    local_update = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                    output_list = local_update.process_samples(train_dataset, action.indices, local_models[action.src])
                    comm.Isend(output_list, dest=action.dest % args.size, tag=action.tag)

                if action.dest % args.size == args.rank:
                    output_list = comm.recv(source=action.src % args.size, tag=action.tag)
                    # TODO: Here, the destination node should train to match the representations. This can also be implemented via a new function in update.py.
            
            elif isinstance(action, ShareWeights):
                
                if action.src % args.size == args.rank:
                    src_local_weights = local_models[action.src].state_dict()
                    comm.isend(src_local_weights, dest=action.dest % args.size)

                if action.dest % args.size == args.rank:
                    shared_weights = comm.recv(source=action.src % args.size)
                    avg_weights = average_weights([local_models[action.dest].state_dict(), shared_weights])
                    local_models[action.dest].load_state_dict(avg_weights)  

                    dest_local_weights = local_models[action.src].state_dict()
                    comm.isend(dest_local_weights, dest=action.src % args.size)                            

                if action.src % args.size == args.rank:
                    shared_weights = comm.recv(source=action.dest % args.size)
                    avg_weights = average_weights([local_models[action.src].state_dict(), shared_weights])
                    local_models[action.src].load_state_dict(avg_weights)

            

            else:
                exit("Error: Undefined action")
        
        
        # [USER SELECTION]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # [PRINT INFO]
        if args.rank == 0:
            print()
            print()
            print(f'| Global Training Round : {epoch + 1} |')
            print()
            print("Actions Selected by RL Agent: ")
            if len(actions) == 0:
                print("None")
            else: 
                for action in actions:
                    print(action)
            print()
            print("Users Chosen: ")
            print(idxs_users)
            print()

        # [LOCAL TRAINING]
        for idx in idxs_users:
            if idx % args.size == args.rank:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                if args.supervision:
                    # Supervised learning
                    w, loss = local_model.update_weights(
                        model=local_models[idx], global_round=epoch)
                else:
                    # Self-supervised learning
                    w, loss, out = local_model.update_weights(
                            model=local_models[idx], global_round=epoch)

                local_losses[idx].append(loss)
            
        # [EVALUATION]
        # if args.rank == 0:
        #     loss_avg = sum(local_losses) / len(local_losses)
        #     train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        for idx in idxs_users:
            if idx % args.size == args.rank:
                local_models[idx].eval()
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                acc, loss = local_model.inference(model=local_models[idx])
                list_acc.append(acc)
                list_loss.append(loss)

        # Gather accuracies and losses
        list_acc_all = comm.gather(list_acc, root=0)
        list_loss_all =  comm.gather(list_loss, root=0)
        
        if args.rank == 0:
            # Flatten accuracies and losses
            list_acc_all = [acc for list_acc in list_acc_all for acc in list_acc]
            list_loss_all = [loss for list_loss in list_loss_all for loss in list_loss]
            train_accuracy.append(sum(list_acc_all) / len(list_acc_all))
            train_loss.append(sum(list_loss_all) / len(list_loss_all))

        # Print global training loss after every 'i' rounds
        if args.rank == 0:
            if (epoch + 1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                if args.supervision:
                    print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                else:
                    print('Train MSE: {:.8f} \n'.format(train_accuracy[-1]))
            print()
            pbar.update(1)
            print()

    # wandb plotting
    
    data = [
        [x, f"User {idx}", y] for idx, ys in local_losses.items() for (x, y) in zip(range(len(ys)), ys)
    ]
    
    table = wandb.Table(data=data, columns=["step", "lineKey", "lineVal"])

    plot = wandb.plot_table(
        "srajani/fed-users",
        table,
        {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
    )

    wandb.log({"loss_plot": plot})
    

    # Test inference after completion of training

    # if args.rank == 0:
    #     test_acc, test_loss = test_inference(args, global_model, test_dataset)

    list_acc, list_loss = [], []
    for idx in idxs_users:
        if idx % args.size == args.rank:
            acc, loss = test_inference(args, local_models[idx], test_dataset)
            list_acc.append(acc)
            list_loss.append(loss)

    list_acc_all = comm.gather(list_acc, root=0)
    list_loss_all = comm.gather(list_loss, root=0)
    # list_acc = gather_losses(comm, list_acc, args)
    # list_loss = gather_losses(comm, list_loss, args)
    
    if args.rank == 0:
        list_acc_all = [acc for list_acc in list_acc_all for acc in list_acc]
        list_loss_all = [loss for list_loss in list_loss_all for loss in list_loss]
        test_accuracy = sum(list_acc_all) / len(list_acc_all)
        test_loss = sum(list_loss_all) / len(list_loss_all)

    if args.rank == 0:
        print(f' \n Results after {args.epochs} global rounds of training:')
        if args.supervision:
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_accuracy))
        else:
            print("|---- Avg Train MSE: {:.8f}".format(train_accuracy[-1]))
            print("|---- Test MSE: {:.8f}".format(test_accuracy))

        # Saving the objects train_loss and train_accuracy:
        file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # PLOTTNG (optional)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('Agg')

        # Plot Loss curve
        # plt.figure()
        # plt.title('Training Loss vs Communication rounds')
        # plt.plot(range(len(train_loss)), train_loss, color='r')
        # plt.ylabel('Training loss')
        # plt.xlabel('Communication Rounds')
        # plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
        #             format(args.dataset, args.model, args.epochs, args.frac,
        #                    args.iid, args.local_ep, args.local_bs))
        
        # Plot Average Accuracy vs Communication rounds
        # plt.figure()
        # plt.title('Average Accuracy vs Communication rounds')
        # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        # plt.ylabel('Average Accuracy')
        # plt.xlabel('Communication Rounds')
        # plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
        #             format(args.dataset, args.model, args.epochs, args.frac,
        #                    args.iid, args.local_ep, args.local_bs))



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

