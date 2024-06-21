import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from model_loading import get_cartesian_batched

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
                layers.append(nn.BatchNorm1d(shape[1]))
                layers.append(activation())
        self.model = nn.Sequential(*layers)

        self.input_mean = None
        self.input_std = None
    
    def forward(self, input):
        # first normalize inputs
        self.input_mean = input.mean().to(device)
        self.input_std = input.std().to(device)
        input = (input - self.input_mean) / self.input_std

        # forward prop (OUTPUT IS NORMALIZED)
        return self.model(input)

    def unnormalize(self, output_arm_soln):
        return (output_arm_soln * self.input_std) + self.input_mean

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
    def __init__(self, 
                 conditional_net, 
                 noise_scale):
        self.conditional_net = conditional_net
        self.noise_scale = noise_scale

    def get_s_t(self, 
                in_arm_poses, 
                in_cartesian_poses, 
                c):
        """Performs batched affine flow calculations
        
        Args:
            in_arm_pose (torch.tensor): Input arm solutions.
            in_cartesian_pose (torch.tensor): Input cartesian poses.
            c (torch.tensor): Input c value (unbatched).
        Returns:
            s (torch.tensor): Scaling factor.
            t (torch.tensor): Translation factor.
        """
        
        batch_size = len(in_arm_poses)
        c = c.repeat(batch_size).unsqueeze(dim=1)
        conditional_input = torch.cat(tensors=(in_arm_poses.to(device), in_cartesian_poses.to(device), c.to(device)),
                                      dim=1)

        # adding softflow noise
        noise = torch.randn_like(conditional_input) * self.noise_scale
        noise = noise.double().to(device)
        conditional_input += noise
        
        layer_out = self.conditional_net(input=conditional_input)
        s, t = layer_out.chunk(2, dim=1)

        s = torch.clamp(s, min=-s/2, max=s/2)
        t = torch.clamp(t, min=-t/2, max=t/2)

        return s, t

    def forward(self, 
                arm_poses, 
                cart_poses, 
                c):
        """Forward propagate batched input.
        
        Args:
            arm_solution (torch.tensor): Batched sample arm solution.
            car_pose (torch.tensor): Batched cartesian pose.
            c (torch.tensor): Input c value (unbatched).
        Returns:
            new_arm_solutions (torch.tensor): Batched transformed arm solution.
            s (torch.tensor): Scaling factor.
        """
        arm_dim = arm_poses.shape[1]
        const_arm_soln = arm_poses[:, :arm_dim//2].to(device)
        altered_arm_soln = arm_poses[:, arm_dim//2:arm_dim].to(device)

        s, t = self.get_s_t(const_arm_soln, cart_poses, c)

        altered_arm_soln = altered_arm_soln * torch.exp(s) + t
        return torch.cat(tensors=(const_arm_soln, altered_arm_soln),
                         dim=1), s
    
class Permute_Layer():
    """Permutes arm solutions for input into next layer/unit.
    
    Args:
        arm_poses (torch.tensor): Batch of arm poses to permute.
        seed (int): Permute seed
    """
    def __init__(self, arm_poses, permute_seed):
        self.arm_poses = arm_poses
        self.permute_seed = permute_seed

    def forward(self):
        """Feed arm solution forward.

        Returns:
            permuted_arm_solution (torch.tensor): Permuted arm solution.
        """
        self.arm_poses = self.arm_poses.cpu().detach().numpy()
        rng = np.random.default_rng(seed=self.permute_seed)
        permuted = np.array([rng.permutation(arm_pose) for arm_pose in self.arm_poses])

        return torch.tensor(permuted, dtype=torch.float64)
    
class Normalizing_Flow_Net(nn.Module):
    def __init__(self, conditional_net_config, noise_scale, num_layers):
        super().__init__()
        self.conditional_net = Conditional_Net(**conditional_net_config).to(device)
        self.conditional_net.initialize(nn.init.kaiming_normal_)
        self.noise_scale = noise_scale
        self.num_layers = num_layers
        # convert weights to float64
        self.double()
    
    def forward(self, 
                arm_poses, 
                cart_poses, 
                c, 
                permute_seed,
                epsilon=1e-9):
        batch_size = len(arm_poses)
        log_det_jacobian = torch.zeros((batch_size, 1), dtype=torch.float64).to(device)

        for i in range(self.num_layers):
            # first feed through coupling
            c_layer = Coupling_Layer(self.conditional_net, self.noise_scale)

            arm_poses, s = c_layer.forward(arm_poses=arm_poses,
                                           cart_poses=cart_poses,
                                           c=c)
            
            # derivative of f wrt to x. only s remains.
            log_det_jacobian += torch.mean(torch.log(torch.abs(s.to(device)) + epsilon), dim=1).unsqueeze(dim=1).to(device)

            # permute the solutions
            p_layer = Permute_Layer(arm_poses=arm_poses,
                                    permute_seed=permute_seed+i)
            arm_poses = p_layer.forward()

        return arm_poses, log_det_jacobian
    
    def ikflow_loss(self, og_sampled_arms, log_det_jacobian):
        # compute -log(P_z(z))
        z_l2_norm  = torch.sum(input=og_sampled_arms ** 2, 
                               dim=1,
                               dtype=torch.float64).unsqueeze(dim=1)
        arm_dim = len(og_sampled_arms[0])
        log_pz = -0.5 * (arm_dim * torch.log(torch.tensor(2*torch.pi, device=device)) + z_l2_norm)
        loss = log_pz + log_det_jacobian
        return -loss

    def train(self, 
              arm_dim,
              data_loader,
              num_iters,
              optimizer,
              learning_rate,
              batch_size,
              c_seed,
              permute_seed):
        """Trains instance of normalizing flow network.

        Args:
            arm_dim (int): Dimension of arm.
            data_loader (torch.DataLoader): Batched dataloader.
            num_iters (int): Number of iterations.
            optimizer (torch.optim): Optimizer used for training.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            c_seed (int): Seed for initial generation of c.
            permute_seed(int): Seed for initial permutation.
        Returns:
            all_epoch_loss (list): Epoch loss acquired over num iters.
            all_mean_dist (list): Mean dist between sampled points acquired over num iters.
        """
        optimizer = optimizer(self.conditional_net.parameters(), lr=learning_rate)
        all_epoch_loss = []
        all_mean_dist = []
        for epoch in trange(num_iters):
            epoch_loss = 0.0
            epoch_mean_dist = 0.0
            # making sure to generate new c each epoch
            rng = np.random.default_rng(c_seed + epoch)
            c = torch.tensor(rng.uniform(0, 1), dtype=torch.float64)
            v = torch.tensor(rng.normal(0, c, size=(batch_size, arm_dim)), dtype=torch.float64).to(device)
            for i, (data_tuple) in enumerate(tqdm(data_loader)):
                sampled_arm_poses = data_tuple[0].to(device)
                cart_poses = data_tuple[1].to(device)
                modified_arm_poses = sampled_arm_poses + v
                
                new_arm_poses, log_det_jacobian = self.forward(arm_poses=modified_arm_poses,
                                                               cart_poses=cart_poses,
                                                               c=c,
                                                               permute_seed=permute_seed+i)

                new_arm_poses = self.conditional_net.unnormalize(new_arm_poses.to(device))
                
                # compute loss and backprop
                # grab initial normed arm input for consistency
                single_loss = self.ikflow_loss(og_sampled_arms=sampled_arm_poses,
                                               log_det_jacobian=log_det_jacobian)
                
                batch_loss = torch.mean(single_loss)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()

                # compute euclidean distance between sampled and target cartesian positions
                # averaged over batch size
                sampled_cart_poses = get_cartesian_batched(new_arm_poses.cpu().detach().numpy()).to(device)

                batch_dist = (sampled_cart_poses - cart_poses).pow(2).sum(dim=1).sqrt()
                batch_mean_dist = batch_dist.mean()

                epoch_mean_dist += batch_mean_dist
                
            print(f"epoch: {epoch}, loss: {epoch_loss/(len(data_loader))}, mean_dist: {epoch_mean_dist/len(data_loader)}")
            all_mean_dist.append(epoch_mean_dist/len(data_loader))
            all_epoch_loss.append(epoch_loss/len(data_loader))

        return all_epoch_loss, all_mean_dist
