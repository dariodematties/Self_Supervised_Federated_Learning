import copy

import torch
from torchvision import datasets, transforms

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from tqdm import tqdm

from options import args_parser
from utils import exp_details, get_dataset, average_weights
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, AutoencoderMNIST
from sampling import dominant_label_sampling
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


if __name__ == "__main__":

    epochs = 80
    optimal_epochs = 10

    # Parse, validate, and print arguments
    args = args_parser()
    exp_details(args)

    # Set random seed for numpy
    np.random.seed(args.seed)

    device = "cuda"

    data_dir = f"../data/{args.dataset}/"
    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    base_model = get_model(args, train_dataset, device)
    optimal_model = copy.deepcopy(base_model)
    global_model = copy.deepcopy(base_model)
    local_models = [copy.deepcopy(base_model) for _ in range(args.num_users)]
    dict_users = dominant_label_sampling(train_dataset, num_users=100, gamma=0.1, print_labels=False)

    print("Training optimal model...")
    epoch_optimal_params = []
    
    acc, loss = test_inference(supervision=True, device="cuda", model=optimal_model, test_dataset=test_dataset, test_fraction=1)
    optimal_params = torch.cat([p.flatten() for p in optimal_model.parameters()]).detach().cpu().numpy()
    epoch_optimal_params.append(optimal_params)
    print(f"Accuracy of optimal model: {acc}")

    for _ in tqdm(range(optimal_epochs)):
        local_update = LocalUpdate(
            train_dataset,
            range(50_000),
            args.local_ep,
            args.local_bs,
            args.lr,
            args.optimizer,
            args.supervision,
            device,
        )
        w, loss = local_update.update_weights(optimal_model)

        acc, loss = test_inference(supervision=True, device="cuda", model=optimal_model, test_dataset=test_dataset, test_fraction=1)
        print(f"Accuracy of optimal model: {acc}")
        optimal_params = torch.cat([p.flatten() for p in optimal_model.parameters()]).detach().cpu().numpy()
        epoch_optimal_params.append(optimal_params)
    
    print("Training client fitting models...")
    epoch_user_params = []
    for _ in tqdm(range(epochs)):
        user_params = []
        curr_users = np.random.choice(args.num_users, int(args.num_users * args.frac))
        for usr in curr_users:
            local_model = local_models[usr]

            local_update = LocalUpdate(
                train_dataset,
                dict_users[usr],
                args.local_ep,
                args.local_bs,
                args.lr,
                args.optimizer,
                args.supervision,
                device,
            )
            w, loss = local_update.update_weights(local_model)

            
        for i, user_samples in dict_users.items():
            local_model = local_models[i]
            params = torch.cat([p.flatten() for p in local_model.parameters()]).detach().cpu().numpy()
            user_params.append(params)
        epoch_user_params.append(user_params)

        local_model_weights = [local_models[usr].state_dict() for usr in curr_users]
        avg_weights = average_weights(local_model_weights)
        global_model.load_state_dict(avg_weights)
        for model in local_models:
            model.load_state_dict(avg_weights)

        acc, loss = test_inference(supervision=True, device="cuda", model=global_model, test_dataset=test_dataset, test_fraction=1)
        print(f"Accuracy of global model: {acc}")

        user_params = []
        for i, user_samples in dict_users.items():
            local_model = local_models[i]
            params = torch.cat([p.flatten() for p in local_model.parameters()]).detach().cpu().numpy()
            user_params.append(params)
        epoch_user_params.append(user_params)
    

    # PCA for fitting model parameters
    pca = PCA(n_components=2)
    pca.fit(epoch_optimal_params)
    pca.fit([params for user_params in epoch_user_params for params in user_params])

    epoch_optimal_params_pca = pca.transform(epoch_optimal_params)
    
    epoch_user_params_pca = np.array([pca.transform(user_params) for user_params in epoch_user_params])

    # tsne = TSNE() 
    # epoch_all_params = tsne.fit_transform([*epoch_optimal_params, *[user_params for user_params in epoch_user_params]])
    # epoch_optimal_params_tsne = tsne.fit(epoch_all_params[:len(optimal_epochs)])
    # epoch_user_params_tsne = tsne.fit(epoch_all_params[:len(optimal_epochs)])

    # labels = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    labels = range(10)

    all_x = [*epoch_optimal_params_pca[:, 0], *epoch_user_params_pca[:, :, 0].flatten()]
    all_y = [*epoch_optimal_params_pca[:, 1], *epoch_user_params_pca[:, :, 1].flatten()]

    xlim = (min(all_x), max(all_x))
    ylim = (min(all_y), max(all_y))

    for epoch in range(epochs * 2):
        fig, ax = plt.subplots()
        pth = ax.scatter(epoch_optimal_params_pca[:, 0], epoch_optimal_params_pca[:, 1], c=range(optimal_epochs + 1), cmap="plasma")
        fig.colorbar(pth)

        ax.scatter(epoch_user_params_pca[epoch, :, 0], epoch_user_params_pca[epoch, :, 1], c="g")
        ax.set_title("Principal Components of MNIST Models")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.savefig(f"pca_models_epoch_{epoch}.png")
        plt.close()