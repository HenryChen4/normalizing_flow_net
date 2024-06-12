import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Conditional_Net(nn.Module):
    def __init__(self, layer_specs, activation):
        super().__init__()
        layers = []
        for i, shape in enumerate(layer_specs):
            layers.append(
                nn.Linear(shape[0],
                          shape[1],
                          bias=shape[2] if len(shape) == 3 else True)
            )
            if i != len(layer_specs) - 1:
                layers.append(activation())
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)
    
class Coupling_Layer:
    def __init__(self, conditional_net):
        self.conditional_net = conditional_net

    def get_s_t(self, in_arm_poses, in_cartesian_pose):
        """Feed inputs into the conditional_net to get s and t.
        Args:
            cartesian_pose (np.ndarray): (x, y) coordinates for arms
            in_arm_poses (np.ndarray): D/2 arms poses
        Returns:
            s, t gaussian transformation
        """
        in_cartesian_pose = torch.squeeze(in_cartesian_pose)
        conditional_input = torch.cat((in_arm_poses, in_cartesian_pose), dim=-1)
        layer_out = self.conditional_net(conditional_input)
        s, t = layer_out.chunk(2, dim=-1)
        return s, t

    def forward(self, arm_solutions, car_x, car_y):
        d = arm_solutions.shape[0]
        const_arm_soln = arm_solutions[0:d//2]
        altered_arm_soln = arm_solutions[d//2:d]

        car_pose = torch.tensor([car_x, car_y], dtype=torch.float32).unsqueeze(0)
        
        s, t = self.get_s_t(const_arm_soln, car_pose)
        altered_arm_soln = altered_arm_soln * torch.exp(s) + t

        return torch.cat((const_arm_soln, altered_arm_soln), dim=0), s

class Permute_Layer:
    def __init__(self, arm_soln, seed):
        self.arm_soln = arm_soln
        self.seed = seed

    def forward(self):
        rng = np.random.default_rng(seed=self.seed)
        permuted = rng.permutation(self.arm_soln.detach().numpy())
        return torch.tensor(permuted, dtype=torch.float32)
    
class Normalizing_Flow_Net(nn.Module):
    def __init__(self, conditional_net_config, num_layers):
        super().__init__()
        self.conditional_net = Conditional_Net(**conditional_net_config)
        self.num_layers = num_layers
    
    def forward(self, initial_arm_solutions, car_x, car_y, permute_seed):
        arm_solutions = torch.tensor(initial_arm_solutions, dtype=torch.float32)

        for i in range(self.num_layers):
            # first feed through coupling
            c_layer = Coupling_Layer(self.conditional_net)
            arm_solutions, s = c_layer.forward(arm_solutions, car_x, car_y)

            # permute the solutions
            p_layer = Permute_Layer(arm_solutions, permute_seed)
            arm_solutions = p_layer.forward()

        return arm_solutions
    
    def train(self, data, num_iters, noise_seed, permute_seed, lr=0.001):
        pass
                