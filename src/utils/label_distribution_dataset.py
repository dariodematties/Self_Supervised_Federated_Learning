import copy
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

from .update import LocalUpdate
from .sampling import dirichlet_sampling


class LabelDistributionDataset(Dataset):
    """A meta-dataset that maps PCA-reduced model parameters of clients trained on a
    given dataset to distributions over the labels of the samples on which they were
    trained.
    """

    def __init__(
        self,
        dataset,
        local_ep,
        local_bs,
        lr,
        optimizer,
        supervision,
        train,
        data_dir,
        base_model_path,
        device,
    ):
        """
        Args:
            dataset (str): the dataset on which the client models will be trained
            local_ep (int): the number of local epochs for which to train client models
            local_bs (int): the batch size for local model training
            lr (float): the learning rate for local model training
            optimizer (str): the optimizer for local model training
            supervision (bool): whether to use supervision for local model training
            train (bool): if True, training set is used; else, test set is used
            data_dir (str): the path at which both the dataset and meta-dataset should
                be saved or loaded
            base_model_path (str): the path to the base model, which will be the
                starting point for the client models
            device (str): the device on which to perform local model training
        """
        self._device = device
        self._dataset = dataset
        self._local_ep = local_ep
        self._local_bs = local_bs
        self._lr = lr
        self._optimizer = optimizer
        self._supervision = supervision
        self._train = train
        self._base_model = torch.load(base_model_path)

        self._alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
        dataset_dir = os.path.join(data_dir, self._dataset)

        if self._dataset == "mnist":
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            self._train_dataset = datasets.MNIST(
                dataset_dir, train=True, download=True, transform=apply_transform
            )
            self._test_dataset = datasets.MNIST(
                dataset_dir, train=False, download=True, transform=apply_transform
            )
        if self._dataset == "cifar":
            apply_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
            self._train_dataset = datasets.CIFAR10(
                dataset_dir, train=True, download=True, transform=apply_transform
            )
            self._test_dataset = datasets.CIFAR10(
                dataset_dir, train=False, download=True, transform=apply_transform
            )

        meta_dataset_dir = os.path.join(
            data_dir, "label_distribution", f"ep_{self._local_ep}_{self._dataset}"
        )
        train_dir = os.path.join(meta_dataset_dir, "train")
        test_dir = os.path.join(meta_dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        self._input_save_path_train = os.path.join(train_dir, "inputs.pt")
        self._label_save_path_train = os.path.join(train_dir, "labels.pt")
        self._input_save_path_test = os.path.join(test_dir, "inputs.pt")
        self._label_save_path_test = os.path.join(test_dir, "labels.pt")

        self._save_samples()
        self._load_samples()

    def _check_exists(self):
        paths = [
            self._input_save_path_train,
            self._label_save_path_train,
            self._input_save_path_test,
            self._label_save_path_test,
        ]
        return all([os.path.exists(path) for path in paths])

    def _save_samples(self):
        if self._check_exists():
            print("Loading from saved dataset...")
            return

        print("Creating dataset...")
        user_samples_train, user_labels_train = self._get_samples_and_labels(train=True)
        user_samples_test, user_labels_test = self._get_samples_and_labels(train=False)
        user_params_train = self._train_models(user_samples_train, train=True)
        user_params_test = self._train_models(user_samples_test, train=False)
        pca = self._fit_pca(user_params_train)
        user_params_train_pca = self._transform_pca(user_params_train, pca)
        user_params_test_pca = self._transform_pca(user_params_test, pca)

        print("Saving dataset...")
        torch.save(user_params_train_pca, self._input_save_path_train)
        torch.save(user_labels_train, self._label_save_path_train)
        torch.save(user_params_test_pca, self._input_save_path_test)
        torch.save(user_labels_test, self._label_save_path_test)

    def _load_samples(self):
        print("Loading dataset...")
        self._user_params_pca = torch.load(
            self._input_save_path_train if self._train else self._input_save_path_test
        )
        self._user_labels = torch.load(
            self._label_save_path_train if self._train else self._label_save_path_test
        )

    def _get_samples_and_labels(self, train):
        print("Getting samples and labels...")
        dataset = self._train_dataset if train else self._test_dataset
        labels = np.array(dataset.targets)
        user_samples = []
        user_labels = []
        for alpha in tqdm(self._alphas):
            num_samples = 2_000 if train else 400
            for i in range(num_samples):
                samples = dirichlet_sampling(
                    dataset,
                    num_users=1,
                    num_samples=500,
                    alpha=alpha,
                    print_labels=False,
                )[0]
                user_samples.append(samples)
                label_counts = torch.tensor(
                    [
                        np.count_nonzero(labels[list(samples)] == label)
                        for label in range(10)
                    ],
                    dtype=torch.float32,
                )
                label_ratios = label_counts / sum(label_counts)
                user_labels.append(label_ratios)
        return user_samples, user_labels

    def _train_models(self, user_samples, train):
        print("Training client models...")
        user_params = []
        for samples in tqdm(user_samples):
            local_model = copy.deepcopy(self._base_model).to(self._device)

            local_update = LocalUpdate(
                self._train_dataset if train else self._test_dataset,
                samples,
                self._local_ep,
                self._local_bs,
                self._lr,
                self._optimizer,
                self._supervision,
                self._device,
            )
            local_update.update_weights(local_model)

            params = (
                torch.cat([p.flatten() for p in local_model.parameters()])
                .detach()
                .cpu()
                .numpy()
            )
            user_params.append(params)
        return user_params

    def _fit_pca(self, user_params):
        print("Fitting PCA...")
        pca = PCA(n_components=10)
        pca.fit(user_params)
        return pca

    def _transform_pca(self, user_params, pca):
        print("Transforming PCA...")
        return torch.tensor(pca.transform(user_params), dtype=torch.float32)

    def __len__(self):
        return len(self._user_labels)

    def __getitem__(self, idx):
        return self._user_params_pca[idx], self._user_labels[idx]
