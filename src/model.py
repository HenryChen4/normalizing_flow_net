import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

class Conditional_Net(nn.Module):
    def __init__(self, layer_specs, activation):
        super().__init__()
        layers = []
        for i, shape in enumerate(layer_specs):
            layers.append(
                nn.Linear(shape[0],
                          shape[1],
                          bias=shape[2] if len(shape) == 3 else False)
            )
            if i != len(layer_specs) - 1:
                layers.append(activation())
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        # first normalize inputs
        input = (input - input.mean()) / input.std()
        # forward prop
        return self.model(input)

    def initialize(self, func):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Biases init to zero.

        self.apply(init_weights)

        return self
    
class Coupling_Layer:
    """Transforms sampled arm solution through an affine
    normalizing flow into a new arm solution.

    Implemented as described in: "https://arxiv.org/pdf/2111.08933"

    Args:
        conditional_net (Conditional_Net): Neural network
        for retrieving scaling (s) and transformation (t) 
        values.
    """
    def __init__(self, conditional_net):
        self.conditional_net = conditional_net

    def get_s_t(self, in_arm_poses, in_cartesian_pose, c):
        """Performs affine flow calculations.

        Args:
            in_arm_poses (torch.tensor): Input arm solutions.
            in_cartesian_pose (torch.tensor): Input cartesian poses.
            c (torch.tensor): Input c.
        Returns:
            s (torch.float32): Scaling factor.
            t (torch.float32): Translation factor.
        """
        in_cartesian_pose = torch.squeeze(in_cartesian_pose)
        c = c.unsqueeze(dim=0)
        conditional_input = torch.cat((in_arm_poses, in_cartesian_pose, c), dim=-1)
        layer_out = self.conditional_net(conditional_input)
        s, t = layer_out.chunk(2, dim=-1)
        return s, t

    def forward(self, arm_solution, car_x, car_y, c):
        """Forward propagate single input.
        
        Args:
            arm_solution (torch.tensor): Single sampled arm solution.
            car_x (float): Target cartesian x.
            car_y (float): Target cartesian y.
        Returns:
            new_arm_solution (torch.tensor): Transformed arm solution.
            s (torch.float32): Scaling factor.
        """
        d = arm_solution.shape[0]
        const_arm_soln = arm_solution[0:d//2]
        altered_arm_soln = arm_solution[d//2:d]

        car_pose = torch.tensor([car_x, car_y], dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        
        s, t = self.get_s_t(const_arm_soln, car_pose, c)

        altered_arm_soln = altered_arm_soln * torch.exp(s) + t

        return torch.cat((const_arm_soln, altered_arm_soln), dim=0), s

class Permute_Layer:
    """Permutes arm solutions for input into next layer/unit.
    
    Args:
        arm_soln (torch.tensor): Arm solution to be permuted.
        seed (int): Permute seed
    """
    def __init__(self, arm_soln, seed):
        self.arm_soln = arm_soln
        self.seed = seed

    def forward(self):
        """Feed arm solution forward.

        Returns:
            permuted_arm_solution (torch.tensor): Permuted arm solution.
        """
        rng = np.random.default_rng(seed=self.seed)
        permuted = rng.permutation(self.arm_soln.detach().numpy())
        return torch.tensor(permuted, dtype=torch.float32)
    
class Normalizing_Flow_Net(nn.Module):
    def __init__(self, conditional_net_config, num_layers):
        super().__init__()
        self.conditional_net = Conditional_Net(**conditional_net_config)
        self.conditional_net.initialize(nn.init.kaiming_normal_)
        self.num_layers = num_layers
    
    def forward(self, initial_arm_solutions, car_x, car_y, c, permute_seed):
        arm_solutions = torch.tensor(initial_arm_solutions, dtype=torch.float32)
        log_det_jacobian = 0.0

        for i in range(self.num_layers):
            # first feed through coupling
            c_layer = Coupling_Layer(self.conditional_net)
            arm_solutions, s = c_layer.forward(arm_solutions, car_x, car_y, c)

            # derivative of f wrt to x. only s remains.
            log_det_jacobian += torch.mean(torch.log(torch.abs(s)))

            # permute the solutions
            p_layer = Permute_Layer(arm_solutions, permute_seed)
            arm_solutions = p_layer.forward()

        return arm_solutions, log_det_jacobian
    
    def ikflow_loss(self, og_sampled_arm, log_det_jacobian):
        # compute -log(P_z(z))
        z_l2_norm = torch.sum(torch.tensor(og_sampled_arm ** 2))
        arm_dim = len(og_sampled_arm)
        log_pz = arm_dim * torch.log(torch.tensor(2*torch.pi)) + z_l2_norm

        return -0.5 * log_pz - log_det_jacobian

    def train(self, arm_dim, data, num_iters, learning_rate, c_seed, permute_seed):
        optimizer = optim.RAdam(self.conditional_net.parameters(), lr=learning_rate)
        for epoch in trange(num_iters):
            epoch_loss = 0.0
            # making sure to generate new c each epoch
            rng = np.random.default_rng(c_seed + epoch)
            c = rng.uniform(0, 1)
            v = rng.normal(0, c, size=arm_dim)
            for i, (og_sampled_arm, car_x, car_y) in enumerate(tqdm(data)):
                modified_arm = og_sampled_arm + v
                new_arm_solution, log_det_jacobian = self.forward(initial_arm_solutions=modified_arm,
                                                                  car_x=car_x,
                                                                  car_y=car_y,
                                                                  c=c,
                                                                  permute_seed=permute_seed)
                
                # compute loss and backpropagate
                single_loss = -self.ikflow_loss(modified_arm, log_det_jacobian)

                optimizer.zero_grad()
                single_loss.backward()
                optimizer.step()

                epoch_loss += single_loss.item()
            print(f"epoch: {epoch}, loss: {epoch_loss/(len(data))}")

# TODO: Look into what the softflow thing does and exactly what the scaling means.
# TODO: Set up a test function to generate multiple solutions