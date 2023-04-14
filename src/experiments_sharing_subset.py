from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import copy

import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from options import args_parser
from utils import exp_details, get_dataset, average_weights
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, AutoencoderMNIST
from resnet import resnet20


def get_model(args, train_dataset, device):
    if args.supervision:
        # Supervised learning
        if args.model == "resnet":
            if args.dataset == "cifar":
                model = resnet20()
            else:
                exit("ResNet only implemented for CIFAR-10 dataset")
        elif args.model == "cnn":
            # Convolutional neural netork
            if args.dataset == "mnist":
                model = CNNMnist(
                    num_channels=args.num_channels, num_classes=args.num_classes
                )
            elif args.dataset == "fmnist":
                model = CNNFashion_Mnist()
            elif args.dataset == "cifar":
                model = CNNCifar(num_classes=args.num_classes)

        elif args.model == "mlp":
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
        else:
            exit("Error: unrecognized model")
    else:
        # Self-supervised learning
        if args.model == "autoencoder":
            # Autoencoder with transpose convolutions
            if args.dataset == "mnist":
                model = AutoencoderMNIST(args=args)

        else:
            exit("Error: unrecognized unsupervised model")

    model.to(device)
    model.train()

    return model


def evaluate_model(alpha, beta, subset, global_epochs, args, device):
    global_test_accuracies = []

    train_dataset, test_dataset, user_groups = get_dataset(
        args.num_users,
        args.dataset,
        False,
        False,
        "missing_label",
        alpha,
        beta,
        0.8,
        2,
        subset,
        save_path=f"save/labels_b_{int(beta * 100)}%_s_{subset}",
    )

    global_model = get_model(args, train_dataset, device)
    local_models = {i: copy.deepcopy(global_model) for i in range(args.num_users)}

    for epoch in tqdm(range(global_epochs)):
        curr_users = np.random.choice(args.num_users, int(args.num_users * args.frac))
        for usr in curr_users:
            # Create a LocalUpdate object for the current user
            local_model = LocalUpdate(
                train_dataset,
                user_groups[usr],
                args.local_ep,
                args.local_bs,
                args.lr,
                args.optimizer,
                args.supervision,
                device,
            )

            # Perform local training for the current user
            if args.supervision:
                # Supervised learning
                w, loss = local_model.update_weights(local_models[usr])
            else:
                # Self-supervised learning
                w, loss, out = local_model.update_weights(local_models[usr])

        acc, loss = test_inference(
            args.supervision, device, global_model, test_dataset, args.test_fraction
        )

        # print(f"(Rank {args.rank}) Epoch {epoch}/{global_epochs}")

        local_model_weights = [local_models[usr].state_dict() for usr in curr_users]
        avg_weights = average_weights(local_model_weights)
        global_model.load_state_dict(avg_weights)
        for model in local_models.values():
            model.load_state_dict(avg_weights)

        global_test_accuracies.append((alpha, beta, subset, epoch, acc))

    return global_test_accuracies


if __name__ == "__main__":
    # Parse, validate, and print arguments
    args = args_parser()

    # MPI setup
    comm = MPI.COMM_WORLD

    args.world_size = comm.Get_size()
    args.rank = comm.Get_rank()
    args.hostname = MPI.Get_processor_name()

    local_rank = 0
    for i, name in enumerate(MPI.COMM_WORLD.allgather(args.hostname)):
        if i >= args.rank:
            break
        if name == args.hostname:
            local_rank += 1
    args.local_rank = local_rank

    torch.cuda.set_device(args.local_rank)

    print(f"(Rank {args.rank}) {args.hostname}, GPU {args.local_rank}")

    comm.Barrier()

    if args.rank == 0:
        exp_details(args)

    comm.Barrier()

    # Set random seed for numpy
    np.random.seed(args.seed)

    # Set up experimental values
    global_epochs = 200
    alpha_list = [1]
    beta_list = [0, 0.05, 0.1, 0.2]
    subset_list = [True, False]

    tests = [(a, b, s) for a in alpha_list for b in beta_list for s in subset_list]
    tests_args = [(a, b, s, global_epochs, args, "cuda") for (a, b, s) in tests]

    accs = []
    for i, test_args in enumerate(tests_args):
        if i % (args.world_size) == args.rank:
            a, b, s, *_ = test_args
            print(f"(Rank {args.rank}) Evaluating model: a = {a}, b = {b}, s = {s}")
            accs.extend(evaluate_model(*test_args))

    all_accs = comm.gather(accs, root=0)

    if args.rank == 0:
        all_accs = [acc for accs in all_accs for acc in accs]
        print("Got test accuracy data.")

        df = pd.DataFrame.from_records(
            all_accs,
            columns=["alpha", "beta", "subset", "Global Epoch", "Global Test Accuracy"],
        )

        df.to_csv(f"save/test_acc_data_{args.dataset}_e_{args.local_ep}.csv")

        print("Saved.")
