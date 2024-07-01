import os

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from model_loading import (
    generate_data,
    create_loader,
)

from ikflows_model import (
    create_flow,
    train
)

from ranger_adabelief import RangerAdaBelief as Ranger

# seeds
permute_seed = 234682346
random_sample_seed = 2346789234

arm_dim = 10
num_rows = 3200

# hyper params
hypernet_config = {
    "hidden_features": (1024, 1024, 1024, 1024),
    "activation": nn.LeakyReLU
}
num_coupling_layers = 12
batch_size = 32
num_iters = 100
learning_rate = 5e-4

arm_data = generate_data(arm_dim=arm_dim,
                         num_rows=num_rows,
                         random_sample_seed=random_sample_seed)

train_loader = create_loader(data=arm_data,
                            batch_size=batch_size)

# create flow network
flow_network = create_flow(arm_dim=arm_dim,
                           num_coupling_layers=num_coupling_layers,
                           hypernet_config=hypernet_config,
                           permute_seed=permute_seed)

# train the flow network
all_epoch_loss, all_mean_dist = train(flow_network=flow_network,
                                train_loader=train_loader,
                                num_iters=num_iters,
                                optimizer=Ranger,
                                learning_rate=learning_rate)

# save results and model
save_dir = f"results/dummy_test"
os.makedirs(save_dir, exist_ok=True)
loss_and_dist_save_path = os.path.join(save_dir, f'loss_and_dist.png')
model_save_path = os.path.join(save_dir, f'model_test.pth')

plt.plot(np.arange(num_iters), all_epoch_loss, color="green", label="loss")
plt.plot(np.arange(num_iters), all_mean_dist, color="blue", label="dist")
plt.legend()
plt.savefig(loss_and_dist_save_path)
plt.clf()

torch.save(flow_network, model_save_path)
