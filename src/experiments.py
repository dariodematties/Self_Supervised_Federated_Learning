import os
import multiprocessing as mp

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

import torch

from options import args_parser
from utils import exp_details, get_dataset, average_weights
from update import LocalUpdate, test_inference

from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, AutoencoderMNIST


def get_model(args, train_dataset, device):
    if args.supervision:
        # Supervised learning
        if args.model == "cnn":
            # Convolutional neural netork
            if args.dataset == "mnist":
                model = CNNMnist(num_channels=args.num_channels, num_classes=args.num_classes)
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


def evaluate_model(alpha, num_minibatches, sharing, device, args):
    global_test_accuracies = []

    train_dataset, test_dataset, user_groups = get_dataset(
        args.num_users, args.dataset, False, False, True, alpha, sharing
    )

    local_models = {i: get_model(args, train_dataset, device) for i in range(args.num_users)}
    global_model = get_model(args, train_dataset, device)

    for epoch in range(150):
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
                args.verbose,
                device,
            )

            # Perform local training for the current user
            if args.supervision:
                # Supervised learning
                w, loss = local_model.update_weights(
                    model=local_models[usr], global_round=epoch, num_minibatches=num_minibatches
                )
            else:
                # Self-supervised learning
                w, loss, out = local_model.update_weights(model=local_models[usr], global_round=epoch)

        acc, loss = test_inference(args.supervision, device, global_model, test_dataset, args.test_fraction)

        print(f"Epoch {epoch} | Acc: {acc} | Loss: {loss}")

        local_model_weights = [local_models[usr].state_dict() for usr in curr_users]
        avg_weights = average_weights(local_model_weights)
        global_model.load_state_dict(avg_weights)
        for model in local_models.values():
            model.load_state_dict(avg_weights)

        global_test_accuracies.append((alpha, num_minibatches, sharing, epoch, acc))

    return global_test_accuracies
    # q.put(global_test_accuracies)


if __name__ == "__main__":

    # Define paths
    path_project = os.path.abspath("..")

    # Parse, validate, and print arguments
    args = args_parser()

    if args.supervision:
        assert args.model in ["cnn", "mlp"]
    else:
        assert args.model in ["autoencoder"]

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))

    # Set random seed for numpy
    np.random.seed(args.seed)

    # Set seaborn theme
    sns.set_theme()

    exp_details(args)

    # Set up experimental values
    # alpha_list = [0.1, 1, 10]
    # alpha_list = [1, 10]
    alpha_list = [0.1, 1, 10]
    # num_minibatches_list = [10, 50]
    num_minibatches_list = [250]
    sharing_list = [False, True]

    tests = [(a, m, s) for a in alpha_list for m in num_minibatches_list for s in sharing_list]
    tests_args = [(a, m, s, "cuda:" + str(i % args.n_gpus), args) for i, (a, m, s) in enumerate(tests)]

    ctx = mp.get_context("forkserver")
    with ctx.Pool(len(tests)) as p:
        accs = p.starmap(evaluate_model, tests_args)
    accs = np.reshape(accs, (-1, 5))

    print("Got test accuracy data.")

    df = pd.DataFrame.from_records(
        accs, columns=["alpha", "num_minibatches", "sharing", "Global Epoch", "Global Test Accuracy"]
    )

    sns.lineplot(
        data=df,
        x="Global Epoch",
        y="Global Test Accuracy",
        # style="num_minibatches",
        style="sharing",
        hue="alpha",
        palette="flare",
        hue_norm=matplotlib.colors.LogNorm(),
    )

    print("Plotted.")

    df.to_csv("test_acc_data.csv")
    plt.savefig(f"test_acc_plot.png")

    print("Saved.")
