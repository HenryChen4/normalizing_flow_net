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
num_train_samples = 2
batch_size = 1

# seeds
train_sample_gen_seed = 41895
c_seed = 1345135
PERMUTE_SEED = 415523 # constant for both training and testing

# sampling and loading dataset
training_data = generate_data(arm_dim=arm_dim,
                              num_rows=num_train_samples,
                              random_sample_seed=train_sample_gen_seed)

data_loader = load_data(data=training_data,
                        batch_size=batch_size)

# different hyperparameters
num_iters = 2
learning_rate = 5e-7
num_coupling_layers = 6
noise_scale = 1e-2

# model creation
conditional_net_config = {
    "layer_specs": [(arm_dim//2 + 3, 1024),
                    (1024, 1024),
                    (1024, arm_dim)],
    "activation": nn.LeakyReLU,
}

normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
                                            noise_scale=noise_scale,
                                            num_layers=num_coupling_layers)

# model training
all_epoch_loss, all_batch_loss, all_mean_dist = normalizing_flow_net.train(arm_dim=arm_dim,
                                                                        data_loader=data_loader,
                                                                        num_iters=num_iters,
                                                                        optimizer=Ranger,
                                                                        learning_rate=learning_rate,
                                                                        batch_size=batch_size,
                                                                        c_seed=c_seed,
                                                                        permute_seed=PERMUTE_SEED)

all_mean_dist = [dist.cpu().numpy() for dist in all_mean_dist]

# save results
save_dir = f"results/result3"
os.makedirs(save_dir, exist_ok=True)
epoch_loss_save_path = os.path.join(save_dir, 'epoch_loss.png')
batch_loss_save_path = os.path.join(save_dir, 'batch_loss.png')
dist_save_path = os.path.join(save_dir, 'mean_dist.png')
model_save_path = os.path.join(save_dir, 'model.pth')

plt.plot(np.arange(num_iters), all_epoch_loss)
plt.savefig(epoch_loss_save_path)
plt.show()

plt.plot(np.arange((num_train_samples/batch_size) * num_iters), all_batch_loss)
plt.savefig(batch_loss_save_path)
plt.show()

plt.plot(np.arange((num_train_samples/batch_size) * num_iters), all_mean_dist)
plt.savefig(dist_save_path)
plt.show()

torch.save(normalizing_flow_net, model_save_path)

# """START of comparing trained vs untrained models"""

# base_seed = 5723759
# num_seeds = 1

# all_untrained_dist = []

# for i in trange(num_seeds):
#     test_data = generate_data(arm_dim=arm_dim,
#                           num_rows=10,
#                           random_sample_seed=245823+i)

#     mean_l2_error = evaluate(test_data=test_data,
#                          model=normalizing_flow_net,
#                          permute_seed=PERMUTE_SEED)
    
#     all_untrained_dist.append(mean_l2_error)

# all_trained_dist = evaluate_seeds(base_seed=base_seed,
#                                   num_rows=10,
#                                   num_seeds=num_seeds)

# save_dir = f"results/test"
# os.makedirs(save_dir, exist_ok=True)
# test_dist_path = os.path.join(save_dir, 'test_dist2.png')

# plt.plot(np.arange(num_seeds), all_untrained_dist, color='blue')
# plt.plot(np.arange(num_seeds), all_trained_dist, color="orange")
# plt.xlabel("seeds")
# plt.ylabel("average dist")
# plt.legend(['untrained dist', 'trained dist'])


# plt.savefig(test_dist_path)
# plt.show()

# """END of comparing trained vs untrained models"""