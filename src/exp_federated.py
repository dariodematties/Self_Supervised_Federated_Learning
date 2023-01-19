import os
import time

import torch
import numpy as np
import copy

from rl_environment import FedEnv
from options import args_parser
from utils import exp_details, get_dataset, weighted_average_weights, average_weights
from exp_update import LocalUpdate, test_inference

from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, AutoencoderMNIST

from matplotlib import pyplot as plt


def get_model(args, train_dataset):
    if args.supervision:
        # Supervised learning
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                model = CNNCifar(args=args)

        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            model = MLP(dim_in=len_in, dim_hidden=64,
                        dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')
    else:
        # Self-supervised learning
        if args.model == 'autoencoder':
            # Autoencoder with transpose convolutions
            if args.dataset == 'mnist':
                model = AutoencoderMNIST(args=args)

        else:
            exit('Error: unrecognized unsupervised model')

    model.to(args.device)
    model.train()

    return model


if __name__ == "__main__":

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

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    args.device = 'cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu'

    # Set random seed for numpy
    np.random.seed(args.seed)

    exp_details(args)

    train_dataset, test_dataset, user_groups = get_dataset(args)

    tests = {
        "lambda=0.4": 0.4,
    }

    for name, global_grad_ratio in tests.items():

        local_models = {i: get_model(args, train_dataset) for i in range(args.num_users)}
        global_model = get_model(args, train_dataset)

        # Set up tracking for loss and accuracy of global model
        global_test_losses = []
        global_test_accuracies = []

        local_models[0].zero_grad()
        global_grad = copy.deepcopy(list(local_models[0].parameters()))
        new_global_grad = copy.deepcopy(global_grad)

        for epoch in range(100):
            for usr, model in local_models.items():
                # Create a LocalUpdate object for the current user
                local_model = LocalUpdate(args=args, device=args.device,
                            dataset=train_dataset, idxs=user_groups[usr])

                # Perform local training for the current user
                if args.supervision:
                    # Supervised learning
                    w, loss, grad_avg = local_model.update_weights(
                        model=local_models[usr], global_round=epoch, global_grad=global_grad, global_grad_ratio=global_grad_ratio)
                    for param_copy, param_global in zip(grad_avg, new_global_grad):
                        if param_global.grad is None:
                            param_global.grad = param_copy.grad
                        else:
                            param_global.grad += param_copy.grad
                else:
                    # Self-supervised learning
                    w, loss, out = local_model.update_weights(
                            model=local_models[usr], global_round=epoch)

            for param in new_global_grad:
                param.grad /= args.num_users

            global_grad = new_global_grad

            acc, loss = test_inference(args, args.device, global_model, test_dataset)

            print(f"Epoch {epoch} | Acc: {acc}")

            local_model_weights = [model.state_dict() for model in local_models.values()]
            avg_weights = average_weights(local_model_weights)
            global_model.load_state_dict(avg_weights)  

            global_test_accuracies.append(acc)
            global_test_losses.append(loss)
            
        plt.plot(global_test_accuracies, label=name)
        
    plt.xlabel("Global Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Global Epoch")
    plt.legend()
    plt.savefig(f"test_acc_plot")