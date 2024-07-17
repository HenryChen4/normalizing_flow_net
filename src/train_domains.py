import torch.nn as nn
import torch
from src.nfn_model import create_flow, train_archive_distill

from src.create_archive import gather_solutions
from ranger_adabelief import RangerAdaBelief as Ranger

import matplotlib.pyplot as plt
import os
import numpy as np

import fire

from src.domains import DOMAIN_CONFIGS

def get_qd_config(qd_name):
    qd_configs = {
        "normal": {
            "grid_cells": (100, 100),
            "sigma0": 0.1,
            "batch_size": 30,
            "num_emitters": 5,
            "num_qd_iters": 700
        }, 
        "more": {
            "grid_cells": (100, 100),
            "sigma0": 0.1,
            "batch_size": 30,
            "num_emitters": 6,
            "num_qd_iters": 700
        }, 
        "most": {
            "grid_cells": (100, 100),
            "sigma0": 0.1,
            "batch_size": 30,
            "num_emitters": 7,
            "num_qd_iters": 700
        }
    }
    return qd_configs[qd_name]

def get_flow_config(flow_name):
    flows = {
        "arm_10d_v1": {
            "solution_dim": 10,
            "num_coupling_layers": 15,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (1024, 1024, 1024),
                "activation": nn.LeakyReLU
            },
            "permute_seed": 13513
        },
        "arm_10d_v2": {
            "solution_dim": 10,
            "num_coupling_layers": 12,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (1024, 1024, 1024),
                "activation": nn.LeakyReLU
            },
            "permute_seed": 81562
        },
        "sphere_100d_v1": {
            "solution_dim": 100,
            "num_coupling_layers": 120,
            "num_context": 3,
            "hypernet_config": {
                "hidden_features": (512, 512, 512),
                "activation": nn.LeakyReLU
            },
            "permute_seed": 41488
        }
    }

    return flows[flow_name]


def main(
    domain_name: str,
    flow_name: str,
    qd_name: str,
    optimizer_name: str,
    batch_size=16,
    num_iters=100,
    learning_rate=5e-5
):
    """Runs experiment"""
    # gather domain training solutions
    print("> Starting QD training loop")
    domain = DOMAIN_CONFIGS[domain_name]
    qd_config = get_qd_config(qd_name)

    train_loader = gather_solutions(qd_config=qd_config,
                                    batch_size=batch_size,
                                    **domain)
    print("> Ending QD training loop")
    
    # train flow model
    optimizer = None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam
    elif optimizer_name == "ranger":
        optimizer = Ranger
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD
    else: 
        print(f"{optimizer_name} optimizer not yet implemented!")

    flow = create_flow(**get_flow_config(flow_name))
    all_epoch_loss, all_mean_dist = train_archive_distill(flow_network=flow,
                                                          train_loader=train_loader,
                                                          num_iters=num_iters,
                                                          optimizer=optimizer,
                                                          learning_rate=learning_rate)
    
    # saving plots
    cpu_epoch_loss = []
    cpu_mean_dist = []

    for i in all_epoch_loss:
        cpu_epoch_loss.append(i)

    for i in all_mean_dist:
        cpu_mean_dist.append(i.cpu().numpy())

    # save results and model
    save_dir = f"results/archive_distill/{domain_name}/{flow_name}"
    os.makedirs(save_dir, exist_ok=True)
    loss_and_dist_save_path = os.path.join(save_dir, f'loss_and_dist.png')
    model_save_path = os.path.join(save_dir, f'model_test.pth')

    torch.save(flow, model_save_path)

    plt.plot(np.arange(num_iters), cpu_epoch_loss, color="green", label="loss")
    plt.plot(np.arange(num_iters), cpu_mean_dist, color="blue", label="dist")
    plt.legend()
    plt.savefig(loss_and_dist_save_path)
    plt.clf()

# usage:
# python -m src.train_domains --domain_name=sphere_100d --flow_name=sphere_100d_v1 --qd_name=normal --optimizer_name=ranger --num_iters=1
if __name__ == "__main__":
    fire.Fire(main)