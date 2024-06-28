import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from ranger_adabelief import RangerAdaBelief as Ranger
import os

from matplotlib import pyplot as plt

from model_loading import sample_arm_input, get_cartesian, generate_data, load_data, normalize_data
from model import Normalizing_Flow_Net
from visualize import visualize

from model_eval import compare
from tqdm import tqdm, trange

# data generation vars
arm_dim = 10
num_train_samples = 6400
batch_size = 16

# seeds
train_sample_gen_seed = 7234
c_seed = 2349579
PERMUTE_SEED = 283623450 # constant for both training and testing

# sampling and loading dataset
training_data = generate_data(arm_dim=arm_dim,
                              num_rows=num_train_samples,
                              random_sample_seed=train_sample_gen_seed)

normalized_training_data = normalize_data(training_data)

data_loader = load_data(data=normalized_training_data,
                        batch_size=batch_size)

# different hyperparameters
num_iters = 500
learning_rate = 1e-8
num_coupling_layers = 1

# model creation
conditional_net_config = {
    "layer_specs": [(arm_dim//2 + 2, 256),
                    (256, 256),
                    (256, arm_dim)],
    "activation": nn.LeakyReLU,
}

# main experiment loop
normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
                                            num_layers=num_coupling_layers,
                                            arm_dim=arm_dim,
                                            permute_seed=PERMUTE_SEED)

# model training
all_epoch_loss = normalizing_flow_net.train(data_loader=data_loader,
                                            num_iters=num_iters,
                                            optimizer=Ranger,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size)

# save results
# save_dir = f"results/invertible_model/"
# os.makedirs(save_dir, exist_ok=True)
# epoch_loss_save_path = os.path.join(save_dir, f'epoch_loss_test.png')
# model_save_path = os.path.join(save_dir, f'model_test.pth')

# plt.plot(np.arange(num_iters), all_epoch_loss)
# plt.savefig(epoch_loss_save_path)
# plt.show()
# plt.clf()

# torch.save(normalizing_flow_net, model_save_path)
