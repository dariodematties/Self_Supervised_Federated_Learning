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
from rl_actions import SwapWeights, ShareWeights, ShareRepresentations

if __name__ == '__main__':

    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up timer
    start_time = time.time()

    # Define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    
    # Parse, validate, and print arguments
    args = args_parser()

    if args.supervision:
        assert args.model in ["cnn", "mlp"]
    else:
        assert args.model in ["autoencoder"]
    
    exp_details(args)
    
    args.rank = rank
    args.size = size
    args.gpu = rank

    if args.rank == 0:
        print(f'Environment size {args.size}')
    print(f'[Rank {args.rank}] Checking in!')

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    device = 'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu'

    print(f'Rank {args.rank} checking in with device {device} and gpu number is {args.gpu}')

    # Load dataset
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # train_dataset = list(train_dataset)[:4096]
    # test_dataset = list(test_dataset)[:4096]

    
    # Set up model
    global_model = None

    if args.rank == 0:
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

    # Broadcast global model
    global_model = comm.bcast(global_model, root=0)
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

        if args.rank == 0:
            print(f'\n | Global Training Round : {epoch + 1} |\n')

        # Action Phase
            
        # Decide on actions; here, rank 0 acts as RL agent
        actions = []
        if args.rank == 0:
            # Add actions to list; something like: 
            actions.append(SwapWeights(2, 3))
            pass

        # Broadcast actions
        actions = comm.bcast(actions, root=0)


        for action in actions:

            if isinstance(action, SwapWeights):
                src_idxs = user_groups[action.src]
                user_groups[action.src] = user_groups[action.dest]
                user_groups[action.dest] = src_idxs

            elif isinstance(action, ShareRepresentations):
                if args.rank == action.src:
                    # Perform inference with process_samples; then send output representations via MPI
                    # NOTE: Using global model to process samples for now, because local model does not exist
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                    output_list = local_model.process_samples(train_dataset, action.indices, copy.deepcopy(global_model))
                    comm.send(output_list, dest=action.dest, tag="ShareRepresentations")
                if args.rank == action.dest:
                    output_list = comm.recv(source=action.src, tag="ShareRepresentations")
                    # Here, the destination node should train to match the representations. This can also be implemented via a new function in update.py, but it doesn't make sense right now since there is only the global model at this point.
                    pass
            
            # NOTE: Sharing weights is redundant in the context of FedAvg, because we're sending out a global model each epoch anyway. However, it will make sense as a means to generate global agreement once we remove this, because the local weights will no longer be ephemeral.
            if isinstance(action, ShareWeights):
                # Send src local model weights to dest local model (these models do not currently exist, since we're sending out global models at the start of each epoch).
                pass
        

        # User Selection & Local Training

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
        global_weights = {}
        if args.rank == 0:
            global_weights = average_weights(local_weights)
        global_weights = comm.bcast(global_weights, root=0)
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

        # PLIOTTNG (optional)
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

