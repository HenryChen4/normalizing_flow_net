# TODO: 1. Create the archive and grab all solutions from archive
# TODO: 2. Train an NFN on that data over a number of iterations
# TODO: 3. Write functions that can be used in main.py

import torch.nn as nn
import torch
from src.ikflows_model import create_flow, train_archive_distill

from src.create_archive import gather_solutions
import matplotlib.pyplot as plt
import os
import numpy as np

arm_dim = 10
train_batch_size = 16

# QD training hyperparameters (bigger numbers == more training data)
# These settings produce 112898 total training samples
num_qd_iters = 700
cells = 10000
sigma0 = 0.1
batch_size = 30
num_emitters = 5

print(">Starting QD loop to generate training samples")
train_loader = gather_solutions(arm_dim=arm_dim,
                                num_qd_iters=num_qd_iters,
                                train_batch_size=train_batch_size,
                                cells=cells,
                                sigma0=sigma0,
                                batch_size=batch_size,
                                num_emitters=num_emitters)
print(">Ending QD loop to generate training samples")
# create ik flow archive model
# archive model hyperparameters
num_coupling_layers = 12
num_context = 3
hypernet_config = {
    "hidden_features": (512, 512, 512, 512),
    "activation": nn.LeakyReLU
}
permute_seed = 1357981375
num_iters = 1
learning_rate = 5e-5
optimizer = torch.optim.Adam

flow = create_flow(arm_dim=arm_dim,
                   num_coupling_layers=num_coupling_layers,
                   num_context=num_context,
                   hypernet_config=hypernet_config,
                   permute_seed=permute_seed)

# train archive model on archive data
print(">Starting archive model training loop to distill archive data")
all_epoch_loss, all_mean_dist, all_mean_obj_diff = train_archive_distill(flow_network=flow,
                                                                         train_loader=train_loader,
                                                                         num_iters=num_iters,
                                                                         optimizer=optimizer,
                                                                         learning_rate=learning_rate)

cpu_epoch_loss = []
cpu_mean_dist = []
cpu_mean_obj_diff = []
for i in all_epoch_loss:
    cpu_epoch_loss.append(i)

for i in all_mean_dist:
    cpu_mean_dist.append(i.cpu().numpy())

for i in all_mean_obj_diff:
    cpu_mean_obj_diff.append(i.cpu().numpy())

# save results and model
save_dir = f"results/archive_distill3"
os.makedirs(save_dir, exist_ok=True)
loss_and_dist_save_path = os.path.join(save_dir, f'loss_and_dist.png')
model_save_path = os.path.join(save_dir, f'model_test.pth')

torch.save(flow, model_save_path)

plt.plot(np.arange(num_iters), cpu_epoch_loss, color="green", label="loss")
plt.plot(np.arange(num_iters), cpu_mean_dist, color="blue", label="dist")
plt.plot(np.arange(num_iters), cpu_mean_obj_diff, color="red", label="diff")
plt.legend()
plt.savefig(loss_and_dist_save_path)
plt.clf()