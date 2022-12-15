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

import mpi4py

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
    local_train_losses = {}
    local_test_losses = {}

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
            local_train_losses[i] = []
            local_test_losses[i] = []

    if args.rank == 0:
        print()
        print("Model Information: ")
        print(local_models[0])
        print()
        wandb.init(group="federated_mpi_experiment", config=vars(args))

    # Training
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    if args.rank == 0:
        pbar = tqdm(total=args.epochs)

    for epoch in range(args.epochs):
        
        # [USER SELECTION]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # [ACTIONS]  
        # Decide on actions; here, rank 0 acts as RL agent
        actions = []
        if args.rank == 0:
            # Add actions to list; something like: 
            actions.append(ShareWeights(idxs_users[0], idxs_users[1]))
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
                    src_model = copy.deepcopy(local_models[action.src])
                    comm.isend(src_model, dest=action.dest % args.size)

                if action.dest % args.size == args.rank:
                    dest_model = copy.deepcopy(local_models[action.dest])

                    shared_src_model = comm.recv(source=action.src % args.size)
                    shared_src_model.to(device)
                    avg_weights = average_weights([local_models[action.dest].state_dict(), shared_src_model.state_dict()])
                    local_models[action.dest].load_state_dict(avg_weights)  

                    comm.isend(dest_model, dest=action.src % args.size)                            

                if action.src % args.size == args.rank:
                    shared_dest_model = comm.recv(source=action.dest % args.size)
                    shared_dest_model.to(device)
                    avg_weights = average_weights([local_models[action.src].state_dict(), shared_dest_model.state_dict()])
                    local_models[action.src].load_state_dict(avg_weights)

            else:
                exit("Error: Undefined action")
        
        comm.barrier()

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

                local_train_losses[idx].append(loss)

        # [LOCAL TEST EVALUATION]
        for idx in idxs_users:
            if idx % args.size == args.rank:
                # this should perform inference, because LocalUpdate splits the passed dataset into train, validation, and test, and the inference() function operates on test data
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                acc, loss = local_model.inference(local_models[idx])
                local_test_losses[idx].append(loss)
        

        # Print training and evaluation loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            last_train_losses = [local_train_loss[-1] for local_train_loss in local_train_losses.values() if len(local_train_loss) > 0]
            last_test_losses = [local_test_loss[-1] for local_test_loss in local_test_losses.values() if len(local_test_loss) > 0]

            last_train_losses_combined = comm.gather(last_train_losses, root=0)
            last_test_losses_combined = comm.gather(last_test_losses, root=0)

            if args.rank == 0:
                
                last_train_losses_combined = [loss for last_train_losses in last_train_losses_combined for loss in last_train_losses]
                last_test_losses_combined = [loss for last_test_losses in last_test_losses_combined for loss in last_test_losses]

                print(f'\nAvg Stats after {epoch + 1} global rounds:')
                print(f'Avg. Local Training Loss: {sum(last_train_losses_combined) / len(last_train_losses_combined)}')
                print(f'Avg. Local Testing Loss: {sum(last_test_losses_combined) / len(last_test_losses_combined)}')
        
        comm.barrier()

        if args.rank == 0:
            print()
            pbar.update(1)
            print()

    # wandb plotting

    train_losses_combined = comm.gather(local_train_losses, root=0)
    test_losses_combined = comm.gather(local_test_losses, root=0)

    if args.rank == 0:
        
        train_losses_combined = {idx: loss for local_train_losses in train_losses_combined for (idx, loss) in local_train_losses.items()}
        test_losses_combined = {idx: loss for local_test_losses in test_losses_combined for (idx, loss) in local_test_losses.items()}
    
        train_loss_data = [
            [x, f"User {idx}", y] for idx, ys in train_losses_combined.items() for (x, y) in zip(range(len(ys)), ys)
        ]

        test_loss_data = [
            [x, f"User {idx}", y] for idx, ys in test_losses_combined.items() for (x, y) in zip(range(len(ys)), ys)
        ]
        
        train_loss_table = wandb.Table(data=train_loss_data, columns=["step", "lineKey", "lineVal"])
        test_loss_table = wandb.Table(data=test_loss_data, columns=["step", "lineKey", "lineVal"])

        plot = wandb.plot_table(
            "srajani/fed-users",
            train_loss_table,
            {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
            {"title": "Train Loss vs. Per-Node Local Training Round"}
        )
        wandb.log({"train_loss_plot": plot})

        plot = wandb.plot_table(
            "srajani/fed-users",
            test_loss_table,
            {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"},
            {"title": "Test Loss vs. Per-Node Local Training Round"}
        )
        wandb.log({"test_loss_plot": plot})
    

    # Test inference after completion of training
    last_train_losses = [local_train_loss[-1] for local_train_loss in local_train_losses.values() if len(local_train_loss) > 0]
    last_test_losses = [local_test_loss[-1] for local_test_loss in local_test_losses.values() if len(local_test_loss) > 0]

    last_train_losses_combined = comm.gather(last_train_losses, root=0)
    last_test_losses_combined = comm.gather(last_test_losses, root=0)

    if args.rank == 0:
        last_train_losses_combined = [loss for last_train_losses in last_train_losses_combined for loss in last_train_losses]
        last_test_losses_combined = [loss for last_test_losses in last_test_losses_combined for loss in last_test_losses]

        print(f' \n Results after {args.epochs} global rounds of training:')
        print(f'Avg. Local Training Loss: {sum(last_train_losses_combined) / len(last_train_losses_combined)}')
        print(f'Avg. Local Testing Loss: {sum(last_test_losses_combined) / len(last_test_losses_combined)}')

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

