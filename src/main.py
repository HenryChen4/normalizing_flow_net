import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from ranger_adabelief import RangerAdaBelief as Ranger
import os

from matplotlib import pyplot as plt

from model_loading import sample_arm_input, get_cartesian, generate_data, load_data
from model_batched import Normalizing_Flow_Net
from visualize import visualize

from model_eval import evaluate, evaluate_seeds
from tqdm import tqdm, trange

# data generation vars
arm_dim = 10
num_train_samples = 64000
batch_size = 16

# seeds
train_sample_gen_seed = 6571678561
c_seed = 682627359
PERMUTE_SEED = 12415385 # constant for both training and testing

# sampling and loading dataset
training_data = generate_data(arm_dim=arm_dim,
                              num_rows=num_train_samples,
                              random_sample_seed=train_sample_gen_seed)

data_loader = load_data(data=training_data,
                        batch_size=batch_size)

# different hyperparameters
num_iters = 100
learning_rate = 1e-4
num_coupling_layers = 10
noise_scale = 1e-3

# model creation
conditional_net_config = {
    "layer_specs": [(arm_dim//2 + 3, 256),
                    (256, 256),
                    (256, arm_dim)],
    "activation": nn.LeakyReLU,
}

for i in num_coupling_layers:
    # main experiment loop
    normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
                                                noise_scale=noise_scale,
                                                num_layers=i)

    # model training
    all_epoch_loss, all_mean_dist = normalizing_flow_net.train(arm_dim=arm_dim,
                                                            data_loader=data_loader,
                                                            num_iters=num_iters,
                                                            optimizer=Ranger,
                                                            learning_rate=learning_rate,
                                                            batch_size=batch_size,
                                                            c_seed=c_seed,
                                                            permute_seed=PERMUTE_SEED)

    all_mean_dist = [dist.cpu().numpy() for dist in all_mean_dist]

    # save results
    save_dir = f"results/2d_arm/"
    os.makedirs(save_dir, exist_ok=True)
    epoch_loss_save_path = os.path.join(save_dir, f'epoch_loss_{i}_cl.png')
    dist_save_path = os.path.join(save_dir, f'mean_dist_{i}_cl.png')
    model_save_path = os.path.join(save_dir, f'model_{i}_cl.pth')

    plt.plot(np.arange(num_iters), all_epoch_loss)
    plt.savefig(epoch_loss_save_path)
    plt.show()
    plt.clf()

    plt.plot(np.arange(num_iters), all_mean_dist)
    plt.savefig(dist_save_path)
    plt.show()
    plt.clf()

    torch.save(normalizing_flow_net, model_save_path)