import copy

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

from .options import args_parser
from .utils import exp_details, get_train_test, get_model, get_dataset_and_label_names, add_noise, average_weights
from .update import LocalUpdate, test_inference
from .sampling import dominant_label_sampling, dirichlet_sampling, iid_sampling

from itertools import product


def train_models(args, device):
    noise_types = ['gaussian', 'laplacian']
    noise_scales = [1e-1, 1e-2, 1e-3, 0.0]

    all_accs = []
    base_model = get_model(args.arch, args.dataset, device)

    for noise_type, noise_scale in product(noise_types, noise_scales):

        accs = [] 

        train_dataset, test_dataset = get_train_test(args.dataset)

        global_model = copy.deepcopy(base_model)
        local_models = [copy.deepcopy(base_model) for _ in range(args.num_users)]

        dict_users = iid_sampling(train_dataset, num_users=args.num_users, print_labels=False)
        
        print(f"{device} | starting training for {noise_type}, {noise_scale}")
        for epoch in range(500):
            curr_users = np.random.choice(args.num_users, int(args.num_users * args.frac))
            for usr in curr_users:
                user_samples = dict_users[usr]
                local_model = local_models[usr]
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
                add_noise(local_model.state_dict(), noise_type=noise_type, scale=noise_scale)

            acc, loss = test_inference(
                args.supervision, device, global_model, test_dataset, args.test_fraction
            )

            local_model_weights = [local_models[usr].state_dict() for usr in curr_users]
            avg_weights = average_weights(local_model_weights)
            global_model.load_state_dict(avg_weights)
            for model in local_models:
                model.load_state_dict(avg_weights)

            accs.append(acc)
        all_accs.append(accs)
        
    return all_accs