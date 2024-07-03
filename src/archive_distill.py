import time
import sys

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import cvt_archive_heatmap

from src.model_loading import create_loader_better
from src.ikflows_model import create_flow, train_archive_distill

def simulate(solutions, link_lengths):
    objs = -np.std(solutions, axis=1)

    # theta_1, theta_1 + theta_2, ...
    cum_theta = np.cumsum(solutions, axis=1)
    # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
    x_pos = link_lengths[None] * np.cos(cum_theta)
    # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
    y_pos = link_lengths[None] * np.sin(cum_theta)

    meas = np.concatenate(
        (
            np.sum(x_pos, axis=1, keepdims=True),
            np.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs, meas

def create_scheduler(arm_dim):
    link_lengths = np.ones(arm_dim)  # 12 links, each with length 1.
    max_pos = np.sum(link_lengths)
    archive = CVTArchive(
        solution_dim=arm_dim,
        cells=10000,
        # The x and y coordinates are bound by the maximum arm position.
        ranges=[(-max_pos, max_pos), (-max_pos, max_pos)],
        # The archive will use a k-D tree to search for the cell a solution
        # belongs to.
        use_kd_tree=True,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=np.zeros(arm_dim),
            # Initial step size of 0.1 seems reasonable based on the bounds.
            sigma0=0.1,
            ranker="2imp",
            bounds=[(-np.pi, np.pi)] * arm_dim,
            batch_size=30,
        ) for _ in range(5)  # Create 5 separate emitters.
    ]

    scheduler = Scheduler(archive, emitters)

    return archive, scheduler

def fill_archive(arm_dim, scheduler, archive, num_iters):
    """Runs QD algorithm"""
    metrics = {
        "Archive Size": {
            "itrs": [0],
            "vals": [0],  # Starts at 0.
        },
        "Max Objective": {
            "itrs": [],
            "vals": [],  # Does not start at 0.
        },
    }

    for itr in trange(1, num_iters + 1, desc='Iterations', file=sys.stdout):
        sols = scheduler.ask()

        # Keep track of these solutions for more training data

        objs, meas = simulate(sols, link_lengths=np.ones(arm_dim))
        scheduler.tell(objs, meas)

        # Logging.
        if itr % 50 == 0:
            metrics["Archive Size"]["itrs"].append(itr)
            metrics["Archive Size"]["vals"].append(len(archive))
            metrics["Max Objective"]["itrs"].append(itr)
            metrics["Max Objective"]["vals"].append(archive.stats.obj_max)

    return archive

def get_training_loader(archive, batch_size):
    all_train_samples = []
    i = 0
    for elite in archive:
        arm_pose = elite["solution"]
        objective = elite["objective"]
        measures = elite["measures"]

        single_train_tuple = (
            torch.tensor(arm_pose, dtype=torch.float64),
            torch.cat((torch.tensor(measures), torch.tensor(objective).unsqueeze(dim=0)))
        )
        
        all_train_samples.append(single_train_tuple)
    train_loader = create_loader_better(all_train_samples, batch_size)
    return train_loader

# QD Loop hyperparams
arm_dim = 10
num_iters_qd = 700

archive, scheduler = create_scheduler(arm_dim=arm_dim)
archive = fill_archive(arm_dim=arm_dim,
                       scheduler=scheduler,
                       archive=archive,
                       num_iters=num_iters_qd)

# load data from archive into a train loader
batch_size = 16
train_loader = get_training_loader(archive=archive,
                                   batch_size=batch_size)

# create flow network
# hyperparams
num_coupling_layers = 12
num_context = 3
hyper_net_config = {
    "hidden_features": (512, 512, 512, 512),
    "activation": nn.LeakyReLU
}
permute_seed = 27135981375
num_iters = 50
optimizer = torch.optim.Adam
learning_rate = 5e-5

flow_network = create_flow(arm_dim=arm_dim,
                           num_coupling_layers=num_coupling_layers,
                           num_context=num_context,
                           hypernet_config=hyper_net_config,
                           permute_seed=permute_seed)

all_epoch_loss, all_mean_dist, all_mean_obj_diff = train_archive_distill(flow_network=flow_network,
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
save_dir = f"results/archive_distill1"
os.makedirs(save_dir, exist_ok=True)
loss_and_dist_save_path = os.path.join(save_dir, f'loss_and_dist.png')
model_save_path = os.path.join(save_dir, f'model_test.pth')

torch.save(flow_network, model_save_path)

plt.plot(np.arange(num_iters), cpu_epoch_loss, color="green", label="loss")
plt.plot(np.arange(num_iters), cpu_mean_dist, color="blue", label="dist")
plt.plot(np.arange(num_iters), cpu_mean_obj_diff, color="red", label="diff")
plt.legend()
plt.savefig(loss_and_dist_save_path)
plt.clf()