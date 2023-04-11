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
from tqdm import tqdm

from options import args_parser
from utils import exp_details, get_dataset, average_weights
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, AutoencoderMNIST
from sampling import dominant_label_sampling, missing_label_sampling
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


if __name__ == "__main__":
    # Parse, validate, and print arguments
    args = args_parser()
    exp_details(args)

    # Set random seed for numpy
    np.random.seed(args.seed)

    device = "cuda"

    data_dir = f"../data/{args.dataset}/"
    apply_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=apply_transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=apply_transform
    )

    base_model = get_model(args, train_dataset, device)

    model_params_fit = []
    dict_users = missing_label_sampling(
        train_dataset, num_users=100, print_labels=False
    )

    print("Training client fitting models...")
    for i, user_samples in dict_users.items():
        local_model = copy.deepcopy(base_model)
        local_update = LocalUpdate(
            train_dataset,
            user_samples,
            args.local_ep,
            args.local_bs,
            args.lr,
            args.optimizer,
            args.supervision,
            device,
        )
        w, loss = local_update.update_weights(local_model)

        params = (
            torch.cat([p.flatten() for p in local_model.parameters()])
            .detach()
            .cpu()
            .numpy()
        )
        model_params_fit.append(params)

    print("Training client prediction models...")
    scores = []
    for gamma in tqdm([0.8]):
        # trials = []
        # 3 trials for each ratio
        # for _ in range(1):
        #     model_params_predict = []
        #     dict_users = dominant_label_sampling(train_dataset, 100, gamma=gamma, print_labels=True)

        #     for i, user_samples in dict_users.items():
        #         local_model = copy.deepcopy(base_model)
        #         local_update = LocalUpdate(
        #             train_dataset,
        #             user_samples,
        #             args.local_ep,
        #             args.local_bs,
        #             args.lr,
        #             args.optimizer,
        #             args.supervision,
        #             device,
        #         )
        #         w, loss = local_update.update_weights(local_model)

        #         params = torch.cat([p.flatten() for p in local_model.parameters()]).detach().cpu().numpy()
        #         model_params_predict.append(params)
        #     trials.append(model_params_predict)

        for n_neighbors in [3]:
            # PCA for fitting model parameters
            pca = PCA(n_components=2)
            pca_params_fit = pca.fit_transform(model_params_fit)

            # KNN for fitting model parameters
            # neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
            # neigh.fit(model_params_fit, [i // 10 for i in range(100)])

            # KNN for predicting model parameters
            # trial_scores = []
            # for model_params_predict in trials:
            #     score = neigh.score(model_params_predict, [i // 10 for i in range(100)])
            #     trial_scores.append(score)
            # avg_score = sum(trial_scores) / len(trial_scores)
            # scores.append((gamma, n_neighbors, avg_score))

            # model_params_fit_avg = []
            # for label in range(10):
            #     model_params_fit_avg.append([params for i, params in enumerate(model_params_fit) if i % 10 == label])

            # for model_params_predict in trials:
            #     for i, params in enumerate(model_params_predict):
            #         true_label = i % 10
            #         pred = [1 / np.linalg.norm(params - model_params_fit_avg[label]) for label in range(10)]
            #         pred = np.array([math.exp(p) / 0.01 for p in pred])
            #         pred = pred / sum(pred)
            #         # pred = torch.nn.Softmax()(torch.tensor(pred))
            #         print(f"True Label: {true_label}, Predicted Distribution: {pred}")

            # print(f"Dominant label prediction accuracy, k={n_neighbors}: {score}")

        for i, params in enumerate(pca_params_fit):
            plt.scatter(
                params[0],
                params[1],
                c=i % 10,
                vmin=0,
                vmax=9,
                label=i % 10 if i < 10 else "",
                cmap="tab10",
            )

        plt.title(
            "Principal Components of Client Models, Colored by Dominant MNIST Label"
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.savefig(f"pca_models_{gamma * 100}.png")

    # df = pd.DataFrame.from_records(
    #     scores, columns=["Dominant Sample Ratio", "Nearest Neighbors", "Prediction Accuracy"]
    # )

    # df.to_csv(f"prediction_accs.csv")

    # plt.figure(figsize=(10,10))
    # sns.heatmap(df.pivot_table("Prediction Accuracy", "Dominant Sample Ratio", "Nearest Neighbors"), annot=True)
    # plt.savefig("prediction_accs")
