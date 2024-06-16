import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from ranger_adabelief import RangerAdaBelief as Ranger
import os

from matplotlib import pyplot as plt

from model_loading import sample_arm_input, get_cartesian, generate_data, load_data
from model import Normalizing_Flow_Net
from visualize import visualize

# data generation vars
arm_dim = 10
num_train_samples = 5000

# seeds
train_sample_gen_seed = 41895
c_seed = 1345135
permute_seed = 415523

# sampling dataset
training_data = generate_data(arm_dim=arm_dim,
                              num_rows=num_train_samples,
                              random_sample_seed=train_sample_gen_seed)

num_iters = 2000

# different model configs
model_configs = {
    "medium": {
        "layer_specs": [(arm_dim//2 + 3, 32),
                        (32, 32),
                        (32, 32),
                        (32, arm_dim)],
        "activation": nn.LeakyReLU,
    },
    "bigger": {
        "layer_specs": [(arm_dim//2 + 3, 64),
                        (64, 64),
                        (64, 64),
                        (64, arm_dim)],
        "activation": nn.LeakyReLU,
    },
    "large": {
        "layer_specs": [(arm_dim//2 + 3, 256),
                        (256, 256),
                        (256, 256),
                        (256, arm_dim)],
        "activation": nn.LeakyReLU,
    },
    "smaller": {
        "layer_specs": [(arm_dim//2 + 3, 16),
                        (16, 16),
                        (16, 16),
                        (16, arm_dim)],
        "activation": nn.LeakyReLU,
    },
    "smallest": {
        "layer_specs": [(arm_dim//2 + 3, 8),
                        (8, 8),
                        (8, arm_dim)],
        "activation": nn.LeakyReLU,
    },
}

# different optimizers
optimizers = {
    "ranger": Ranger,
    "adam": optim.Adam,
    "sgd": optim.SGD,
}

# different hyperparameters
learning_rates = [1e-5, 5e-6, 1e-6, 5e-7]
num_coupling_layers = [3, 6, 9, 12]
noise_scales = [1e-3, 1e-5, 1e-7]

# main experiment loop
for a, model_config in enumerate(model_configs):
    for b, optimizer in enumerate(optimizers):
        for c, lr in enumerate(learning_rates):
            for d, num_coupling_layer in enumerate(num_coupling_layers):
                for e, noise_scale in enumerate(noise_scales):
                    normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=model_configs[model_config],
                                                                noise_scale=noise_scale,
                                                                num_layers=num_coupling_layer)
                    
                    all_loss = normalizing_flow_net.train(arm_dim=arm_dim,
                                                          data=training_data,
                                                          num_iters=num_iters,
                                                          optimizer=optimizers[optimizer],
                                                          learning_rate=lr,
                                                          c_seed=c_seed,
                                                          permute_seed=permute_seed)
                    
                    save_dir = f"results/model_{a}/optimizer_{b}/lr_{c}/ncl_{d}/ns_{e}"
                    os.makedirs(save_dir, exist_ok=True)
                    loss_save_path = os.path.join(save_dir, 'loss.png')
                    model_save_path = os.path.join(save_dir, 'model.pth')

                    plt.plot(np.arange(num_iters), all_loss)
                    plt.savefig(loss_save_path)
                    plt.show()
                    torch.save(normalizing_flow_net, model_save_path)
