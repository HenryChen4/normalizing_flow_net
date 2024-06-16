import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ranger_adabelief import RangerAdaBelief as Ranger
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    normalizing flow into a new arm solution. Make sure every 
    input to every function is a float64 for higher precision and 
    less instability.

    Implemented as described in: "https://arxiv.org/pdf/2111.08933"

    Args:
        conditional_net (Conditional_Net): Neural network
        for retrieving scaling (s) and transformation (t) 
        values.
    """
    def __init__(self, conditional_net, noise_scale):
        self.conditional_net = conditional_net
        self.noise_scale = noise_scale

    def get_s_t(self, in_arm_poses, in_cartesian_pose, c):
        """Performs affine flow calculations.

        Args:
            in_arm_poses (torch.tensor): Input arm solutions.
            in_cartesian_pose (torch.tensor): Input cartesian poses.
            c (torch.tensor): Input c.
        Returns:
            s (torch.tensor): Scaling factor.
            t (torch.tensor): Translation factor.
        """
        in_cartesian_pose = torch.squeeze(in_cartesian_pose).to(device)
        c = c.unsqueeze(dim=0).to(device)
        conditional_input = torch.cat((in_arm_poses, in_cartesian_pose, c), dim=-1)
        
        # adding softflow noise
        noise = torch.randn_like(conditional_input) * self.noise_scale
        noise = noise.double().to(device)
        conditional_input += noise

        layer_out = self.conditional_net(conditional_input)
        s, t = layer_out.chunk(2, dim=-1)

        s = torch.clamp(s, min=-10, max=10)
        t = torch.clamp(t, min=-10, max=10)

        return s, t

    def forward(self, arm_solution, car_x, car_y, c):
        """Forward propagate single input.
        
        Args:
            arm_solution (torch.tensor): Single sampled arm solution.
            car_x (float): Target cartesian x.
            car_y (float): Target cartesian y.
        Returns:
            new_arm_solution (torch.tensor): Transformed arm solution.
            s (torch.float64): Scaling factor.
        """
        d = arm_solution.shape[0]
        const_arm_soln = arm_solution[0:d//2].to(device)
        altered_arm_soln = arm_solution[d//2:d].to(device)

        car_pose = torch.tensor([car_x, car_y], dtype=torch.float64).to(device)
        c = torch.tensor(c, dtype=torch.float64).to(device)
        
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
        permuted = rng.permutation(self.arm_soln.cpu().detach().numpy())
        return torch.tensor(permuted, dtype=torch.float32)
    
class Normalizing_Flow_Net(nn.Module):
    def __init__(self, conditional_net_config, noise_scale, num_layers):
        super().__init__()
        self.conditional_net = Conditional_Net(**conditional_net_config).to(device)
        self.conditional_net.initialize(nn.init.kaiming_normal_)
        self.noise_scale = noise_scale
        self.num_layers = num_layers
        # convert weights to float64
        self.double()
    
    def forward(self, arm_solutions, car_x, car_y, c, permute_seed):
        log_det_jacobian = 0.0

        for i in range(self.num_layers):
            # first feed through coupling
            c_layer = Coupling_Layer(self.conditional_net, self.noise_scale)

            arm_solutions, s = c_layer.forward(arm_solutions, car_x, car_y, c)

            # derivative of f wrt to x. only s remains.
            log_det_jacobian += torch.mean(torch.log(torch.abs(s)))

            # permute the solutions
            p_layer = Permute_Layer(arm_solutions, permute_seed)
            arm_solutions = p_layer.forward()

        return arm_solutions, log_det_jacobian
    
    def ikflow_loss(self, og_sampled_arm, log_det_jacobian):
        # compute -log(P_z(z))
        z_l2_norm = torch.sum(og_sampled_arm ** 2)
        arm_dim = len(og_sampled_arm)
        log_pz = arm_dim * torch.log(torch.tensor(2*torch.pi, device=device)) + z_l2_norm

        loss = -0.5 * log_pz - log_det_jacobian

        # if loss is torch.nan or loss < 0:
        #     print(f"z_l2: {z_l2_norm}")
        #     print(f"log_pz: {log_pz}")
        #     print(f"log_det_jac: {log_det_jacobian}")

        return loss

    def train(self, arm_dim, data, num_iters, learning_rate, c_seed, permute_seed):
        optimizer = Ranger(self.conditional_net.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(self.conditional_net.parameters(), lr=learning_rate)

        for epoch in trange(num_iters):
            epoch_loss = 0.0
            # making sure to generate new c each epoch
            rng = np.random.default_rng(c_seed + epoch)
            c = rng.uniform(0, 1)
            v = torch.tensor(rng.normal(0, c, size=arm_dim), dtype=torch.float64).to(device)
            for i, (og_sampled_arm, car_x, car_y) in enumerate(tqdm(data)):
                og_sampled_arm = og_sampled_arm.to(device)
                modified_arm = og_sampled_arm + v
                new_arm_solution, log_det_jacobian = self.forward(arm_solutions=modified_arm,
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

#TODO rewrite code to support data batching