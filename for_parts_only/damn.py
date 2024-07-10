"""
DAMT
Input: simulate func, iterations, batch size, tradeoff
Initialize archive model (NFN)
Loop:
    1. Uniformly sample desired features
    2. Collect solutions from NFN in this case
    3. Evaluate obj and features
    4. Grad ascent DAM loss
"""

import torch
import torch.nn as nn
import numpy as np
from src.nfn_model import create_flow
from src.create_archive import torch_simulate
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# only works for arm domain rn and nfn models
def damt(archive_model, 
         num_iters, 
         batch_size, 
         trade_off,
         arm_dim,
         feature_sample_seed,
         optimizer,
         learning_rate):
    optimizer = optimizer(archive_model.parameters(), lr=learning_rate)
    max_feature_dist = torch.linalg.norm(torch.tensor([arm_dim], dtype=torch.float64) - 
                                         torch.tensor([-arm_dim], dtype=torch.float64))
    # main training loop
    all_loss = []
    for i in trange(num_iters):
        # step 1: sample desired features
        rng = np.random.default_rng(seed=feature_sample_seed + i)
        true_features = rng.uniform(low=-arm_dim, high=arm_dim, size=(batch_size, 2))
        true_features = torch.tensor(true_features, dtype=torch.float64)
        
        context = torch.cat((true_features, torch.zeros(batch_size, 1)), dim=1)

        # step 2: collect arm solutions from nfn, conditioned on best arm obj and features
        arm_poses = archive_model(context).rsample()

        # step 3: evaluate arm solutions
        objectives, sampled_features = torch_simulate(arm_poses, link_lengths=torch.ones(arm_dim))

        # step 4: optimize
        feature_errors = torch.linalg.norm(sampled_features - true_features, axis=1)
        training_objective = torch.sum(objectives - feature_errors/max_feature_dist)
        loss = -training_objective
        all_loss.append(loss)

        # grad ascent so loss is negated
        print(f"epoch {i} loss: {-loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

    return all_loss
        

# hyperparams for archive model (NFN)
arm_dim = 10
num_coupling_layers = 14
num_context = 3 # i think we can try passing in max obj for now
hyper_net_config = {
    "hidden_features": (124, 124, 124),
    "activation": nn.LeakyReLU
}
permute_seed = 1235091783590

# create archive model
archive_model = create_flow(arm_dim=arm_dim,
                            num_coupling_layers=num_coupling_layers,
                            num_context=num_context,
                            hypernet_config=hyper_net_config,
                            permute_seed=permute_seed)

# hyperparams for damt
num_iters = 100
batch_size = 20
trade_off = 0.2
feature_sample_seed = 5172035
optimizer = torch.optim.Adam
learning_rate = 1e-5

damt_loss = damt(archive_model=archive_model,
                 num_iters=num_iters,
                 batch_size=batch_size,
                 trade_off=trade_off,
                 arm_dim=arm_dim,
                 feature_sample_seed=feature_sample_seed,
                 optimizer=optimizer,
                 learning_rate=learning_rate)

print("Final loss value:")
print(damt_loss[-1])

# plt.plot(np.arange(num_iters), [loss for loss.detach().numpy() in damt_loss])
# plt.show()