import numpy as np
import torch.nn as nn
import torch

from matplotlib import pyplot as plt

from model import Normalizing_Flow_Net
from visualize import visualize

def sample_arm_input(arm_dim, seed):
    """Sample initial arm pose from normal distribution
    
    Args:
        arm_dim (int): Number of arm joints.
        seed (int): Rng seed.
    Returns:
        arm_solution (np.ndarray): Single sampled arm solution
    """
    rng = np.random.default_rng(seed=seed)
    return rng.normal(loc=0.0, scale=np.pi/3, size=(arm_dim, ))

def get_cartesian(arm_pos):
    """Get cartesian coords from arm pos. Link lengths are
    assumed to be 1.
    
    Args:
        arm_pos (np.ndarray): Arm joint position.
    """
    theta_sum = 0
    x_final = 0
    y_final = 0
    for theta in arm_pos:
        theta_sum += theta
        x_final += np.cos(theta_sum)
        y_final += np.sin(theta_sum)
    return (x_final, y_final)

def generate_data(arm_dim, num_rows, random_sample_seed):
    """Generates training data for neural net.

    Args:
        arm_dim (int): Number of arm joints.
        num_rows (int): Size of dataset.
        random_sample_seed (int): Random sample seed.
    Returns:
        data (tuple): (arm_soln, x, y)
    """
    data = []
    for i in range(num_rows):
        arm_solution = sample_arm_input(arm_dim=arm_dim, 
                                        seed=random_sample_seed)
        car_x, car_y = get_cartesian(arm_solution)
        data.append((arm_solution, car_x, car_y))
    return data

"""START of testing"""

# rng = np.random.default_rng(seed=315)

# # (hyper) parameters ??
# arm_dim = 10
# target_cartesian = np.array((5, 7))
# num_coupling_layers = 15

# # seeds
# arm_gen_seed = 124
# permute_seed = 432

# # sample the arm
# arm_link_lens = np.repeat(1, arm_dim)
# initial_arm_sample = sample_arm_input(arm_dim=arm_dim,
#                                       seed=arm_gen_seed)

# print(initial_arm_sample)

# # sample c, supposed to be sampled every epoch
# c = rng.uniform(0, 1)

# # initialize the flow net
# # you take half the arm, cartesian (x, y), and random c as input
# # you output 2 things, s and t, and functionally you just split the final vector
# conditional_net_config = {
#     "layer_specs": [(arm_dim//2 + 3, 128),
#                     (128, 128),
#                     (128, 128),
#                     (128, arm_dim)],
#     "activation": nn.LeakyReLU,
# }

# normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
#                                             num_layers=num_coupling_layers)

# arm_solutions = normalizing_flow_net(initial_arm_solutions=initial_arm_sample,
#                              car_x=target_cartesian[0],
#                              car_y=target_cartesian[1],
#                              c=c,
#                              permute_seed=permute_seed)

# # visualize final arm output
# fig, ax = plt.subplots(figsize=(8, 8))

# visualize(solution=arm_solutions[0].detach().numpy(),
#           link_lengths=arm_link_lens,
#           objective=10,
#           ax=ax)

# plt.show()

"""END of testing"""

"""START of training"""

rng = np.random.default_rng(seed=315)

# (hyper) parameters ??
arm_dim = 10
target_cartesian = np.array((3, 4))

num_coupling_layers = 20

num_data = 3000
num_iters = 1000
learning_rate = 5e-4

# seeds
arm_gen_seed = 413
permute_seed = 149914
c_seed = 413234

# generate data
data = generate_data(arm_dim=arm_dim,
                     num_rows=num_data,
                     random_sample_seed=arm_gen_seed)

# model configs
conditional_net_config = {
    "layer_specs": [(arm_dim//2 + 3, 128),
                    (128, 128),
                    (128, 128),
                    (128, arm_dim)],
    "activation": nn.LeakyReLU,
}

normalizing_flow_net = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
                                            num_layers=num_coupling_layers)

normalizing_flow_net.train(arm_dim=arm_dim,
                           data=data,
                           num_iters=num_iters,
                           learning_rate=learning_rate,
                           c_seed=c_seed,
                           permute_seed=permute_seed)

"""END of training"""