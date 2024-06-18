# from model_unbatched import Conditional_Net, Coupling_Layer, Permute_Layer, Normalizing_Flow_Net
from model_loading import sample_arm_input, get_cartesian, generate_data, Arm_Dataset, load_data
from model_batched import Conditional_Net, Coupling_Layer, Permute_Layer, Normalizing_Flow_Net

import torch
import torch.nn as nn
import numpy as np

arm_dim = 10
num_rows = 6
batch_size = 3

data = generate_data(arm_dim=arm_dim,
                     num_rows=num_rows,
                     random_sample_seed=24144)

data_loader = load_data(data=data,
                        batch_size=batch_size)

conditional_net_config = {
    "layer_specs": [(arm_dim//2 + 3, 10),
                    (10, 10),
                    (10, 10)],
    "activation": nn.LeakyReLU,
}

conditional_net = Conditional_Net(**conditional_net_config)
conditional_net = conditional_net.double()

coupling_layer = Coupling_Layer(conditional_net=conditional_net,
                                noise_scale=1e-5)

c = torch.tensor(0.1, dtype=torch.float64)

nfn = Normalizing_Flow_Net(conditional_net_config=conditional_net_config,
                           noise_scale=1e-5,
                           num_layers=1)

for i, data_tuple in enumerate(data_loader):
    arm_poses = data_tuple[0]
    cart_poses = data_tuple[1]

    new_arm_poses, log_det_jac = nfn.forward(arm_poses=arm_poses,
                cart_poses=cart_poses,
                c=c,
                permute_seed=241084)

    loss = nfn.ikflow_loss(og_sampled_arms=arm_poses,
                           log_det_jacobian=log_det_jac)
    
    print(loss)