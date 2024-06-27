import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm, trange

# for testing only:
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
        self.double()

    def forward(self, input):
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
    def __init__(self,
                 conditional_net):
        self.conditional_net = conditional_net
    
    def get_s_t(self,
                in_arm_poses,
                conditions):
        # batch_size = len(in_arm_poses) use for calculating c dim
        conditional_input = torch.cat(tensors=(in_arm_poses.to(device), conditions.to(device)),
                                      dim=1)

        layer_out = self.conditional_net(input=conditional_input)
        s, t = layer_out.chunk(2, dim=1)

        return s, t
    
    def generate(self,
                 arm_poses,
                 conditions):
        arm_dim = arm_poses.shape[1]
        const_arm_soln = arm_poses[:, :arm_dim//2].to(device)
        altered_arm_soln = arm_poses[:, arm_dim//2:].to(device)

        s, t = self.get_s_t(const_arm_soln, conditions)

        # compute log det jacobian
        log_det_jac = torch.sum(s, dim=1)

        # affine transformation
        altered_arm_soln = altered_arm_soln * torch.exp(s) + t

        return torch.cat(tensors=(const_arm_soln, altered_arm_soln),
                         dim=1), s, t, log_det_jac
    
    def normalize(self,
                  transformed_arm_poses,
                  s,
                  t):
        arm_dim = transformed_arm_poses.shape[1]
        inverted_const_arm_soln = transformed_arm_poses[:, :arm_dim//2]
        inverted_altered_arm_soln = transformed_arm_poses[:, arm_dim//2:].to(device) 

        # inverted affine transformation
        inverted_altered_arm_soln = (inverted_altered_arm_soln - t) * torch.exp(-s)

        return torch.cat(tensors=(inverted_const_arm_soln, inverted_altered_arm_soln),
                         dim=1)
    
class Permute_Layer():
    def __init__(self, arm_dim, permute_seed):
        self.permute_seed = permute_seed

        # create permutation array indices (easier to reverse permutation)
        rng = np.random.default_rng(seed=permute_seed)
        self.permute_indices = np.arange(arm_dim)
        rng.shuffle(self.permute_indices)
    
    def permute(self, arm_poses):
        arm_poses = arm_poses.cpu().detach().numpy()
        permuted_arm = arm_poses[:, self.permute_indices]

        return torch.tensor(permuted_arm, dtype=torch.float64)
    
    def unpermute(self, permuted_arm):
        self.permute_indices = np.argsort(self.permute_indices)
        unpermuted_arm = permuted_arm[:, self.permute_indices]

        return unpermuted_arm

class Coupling_Unit():
    def __init__(self, coupling_layer, permute_layer):
        self.coupling_layer = coupling_layer
        self.permute_layer = permute_layer
    
    def generate(self, in_arm_poses, conditions):
        new_arm_poses, s, t, log_det_jac = self.coupling_layer.generate(in_arm_poses, conditions)
        permuted_arm_poses = self.permute_layer.permute(new_arm_poses)

        return permuted_arm_poses, s, t, log_det_jac

    def normalize(self, arm_poses, s, t):
        unpermuted_arm_poses = self.permute_layer.unpermute(arm_poses)
        normalized_arm_poses = self.coupling_layer.normalize(unpermuted_arm_poses, s, t)
        
        return normalized_arm_poses

class Normalizing_Flow_Net(nn.Module):
    def __init__(self,
                 conditional_net_config,
                 num_layers,
                 arm_dim,
                 permute_seed):
        super().__init__()
        self.conditional_net = Conditional_Net(**conditional_net_config)
        self.conditional_net.initialize(nn.init.kaiming_normal_)

        self.arm_dim = arm_dim

        self.coupling_units = []
        # initialize units for model
        for i in range(num_layers):
            coupling_layer = Coupling_Layer(conditional_net=self.conditional_net)
            # permutation is different for each coupling layer 
            permute_layer = Permute_Layer(arm_dim=arm_dim,
                                          permute_seed=permute_seed+i)
            coupling_unit = Coupling_Unit(coupling_layer=coupling_layer,
                                          permute_layer=permute_layer)
            self.coupling_units.append(coupling_unit)
            
        # keep track of s and t for normalization
        self.s_hist = []
        self.t_hist = []

        # convert weights to float64
        self.double()
    
    def forward(self, 
                initial_arm_poses,
                cart_poses):
        generated_arm_poses = initial_arm_poses
        conditions = cart_poses # subject to change if I need to use the c variable
        for i, coupling_unit in enumerate(self.coupling_units):
            generated_arm_poses, s, t, log_det_jac = coupling_unit.generate(in_arm_poses=generated_arm_poses,
                                                                            conditions=conditions)
            self.s_hist.append(s)
            self.t_hist.append(t)           
        return generated_arm_poses, log_det_jac

    def backward(self,
                 final_arm_poses):
        normalized_arm_poses = final_arm_poses
        for i, coupling_unit in reversed(list(enumerate(self.coupling_units))):
            normalized_arm_poses = coupling_unit.normalize(arm_poses=normalized_arm_poses,
                                                           s=self.s_hist[i],
                                                           t=self.t_hist[i])
        return normalized_arm_poses
    
    def mean_neg_log_loss(self, 
                          normalized_arm_poses, 
                          log_det_jacobian):
        log_pz = -0.5 * (self.arm_dim * torch.log(torch.tensor(2 * torch.pi, device=device))) + torch.sum(normalized_arm_poses ** 2, dim=1)
        loss = log_pz + log_det_jacobian
        return -torch.mean(loss)

    def train(self,
              data_loader,
              num_iters,
              optimizer,
              learning_rate,
              batch_size):
        optimizer = optimizer(self.conditional_net.parameters(), lr=learning_rate)
        all_epoch_loss = []
        for epoch in trange(num_iters):
            epoch_loss = 0.0

            # do the c thing later if training is too hard
            for i, (data_tuple) in enumerate(tqdm(data_loader)):
                sampled_arm_poses = data_tuple[0].to(device)
                cart_poses = data_tuple[1].to(device)

                # first feed foward in the generative direction
                generated_arm_poses, log_det_jac = self.forward(initial_arm_poses=sampled_arm_poses,
                                                   cart_poses=cart_poses)
                
                # normalize the generated arm poses
                normalized_arm_poses = self.backward(generated_arm_poses)

                # compute mean batch loss
                mean_batch_loss = self.mean_neg_log_loss(normalized_arm_poses=normalized_arm_poses,
                                                         log_det_jacobian=log_det_jac)
                
                # back propagate
                optimizer.zero_grad()
                mean_batch_loss.backward()
                optimizer.step()

                epoch_loss += mean_batch_loss.item()
            
            print(f"epoch: {epoch}, loss: {epoch_loss/len(data_loader)}")
            all_epoch_loss.append(epoch_loss/len(data_loader))
        
        return all_epoch_loss
