import numpy as np
import torch.nn as nn
import torch

from matplotlib import pyplot as plt

from model_loading import sample_arm_input, get_cartesian, generate_data, load_data
from model import Normalizing_Flow_Net
from visualize import visualize

# data generation vars
arm_dim = 10
num_train_samples = 2.5e6

# seeds
train_sample_gen_seed = 41895
c_seed = 1345135
permute_seed = 415523

# global (hyper) params ??
num_coupling_layers = 20
noise_scale = 1e-3
num_iters = 1000
learning_rate = 5e-5

# sampling dataset
training_data = generate_data(arm_dim=arm_dim,
                              num_rows=num_train_samples,
                              random_sample_seed=train_sample_gen_seed)

# initialize models
# 3 hidden layers of 1028 units, output is split in 2 for s and t vectors
conditional_net_config = {
    "layer_specs": [(arm_dim//2 + 3, 1028),
                    (1028, 1028),
                    (1028, 1028),
                    (1028, arm_dim)],
    "activation": nn.LeakyReLU,
}

normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
                                            noise_scale=noise_scale,
                                            num_layers=num_coupling_layers)

normalizing_flow_net.train(arm_dim=arm_dim,
                           data=training_data,
                           num_iters=num_iters,
                           learning_rate=learning_rate,
                           c_seed=c_seed,
                           permute_seed=permute_seed)