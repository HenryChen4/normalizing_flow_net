import torch
import numpy as np

from zuko.distributions import DiagNormal
from zuko.flows import (
    Flow, 
    GeneralCouplingTransform, 
    UnconditionalTransform, 
    UnconditionalDistribution
)
from zuko.transforms import (
    MonotonicAffineTransform,
    PermutationTransform
)

from tqdm import (
    trange,
    tqdm
)

from src.model_loading import get_cartesian_batched

def create_flow(solution_dim,
                num_coupling_layers,
                num_context,
                hypernet_config,
                permute_seed):
    """Creates an IKFlow normalizing flow network
    
    Args:
        arm_dim (int): Number of arm joints.
        num_coupling_layers (int): Number of coupling layers.
        hypernet_config (dict): Dictionary for conditional neural network.
        permute_seed (permute_seed): Seed for coupling layer permutation.
    Returns:
        flow (Flow): Flow object (normalizing flow network).
    """
    transforms = []
    for i in range(num_coupling_layers):
        torch.manual_seed(permute_seed + i)

        single_transform = GeneralCouplingTransform(
            features=solution_dim,
            context=num_context,
            univariate=MonotonicAffineTransform,
            **hypernet_config
        ).double()
        permute_transform = UnconditionalTransform(
            PermutationTransform,
            torch.randperm(solution_dim),
            buffer=True
        ).double()
        transforms.append(single_transform)
        transforms.append(permute_transform)

    flow = Flow(
        transform=transforms,
        base=UnconditionalDistribution(
            DiagNormal,
            loc=torch.full((solution_dim, ), 0, dtype=torch.float64),
            scale=torch.full((solution_dim, ), torch.pi/3, dtype=torch.float64),
            buffer=True
        )
    )
    
    return flow

def train(flow_network,
          train_loader,
          num_iters,
          optimizer,
          learning_rate):
    """Trains normalizing flow network with only cartesian poses at context.

    Args:
        flow_network (Flow): Normalizing flow network.
        train_loader (DataLoader): Torch dataloader for easy batching.
        num_iters (int): Number of training iterations.
        optimizer (torch.optim): Optimizer used for training.
        learning_rate (float): Learning rate for optimizer.
    Returns:
        all_epoch_loss (list): Loss acquired every epoch.
        all_mean_dist (list): Mean euclidean distance between sampled arm and original arm.   
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow_network.to(device)
    optimizer = optimizer(flow_network.parameters(), lr=learning_rate)

    all_epoch_loss = []
    all_mean_dist = []

    for epoch in trange(num_iters):
        epoch_loss = 0.
        mean_dist = 0.

        for i, (data_tuple) in enumerate(tqdm(train_loader)):
            original_arm_poses = data_tuple[0].to(device)
            original_cart_poses = data_tuple[1].to(device)
            
            batch_loss = -flow_network(original_cart_poses).log_prob(original_arm_poses)
            batch_loss = batch_loss.mean()

            generated_arm_poses = flow_network(original_cart_poses).sample().to(device)  

            generated_cart_poses = get_cartesian_batched(generated_arm_poses.cpu().detach().numpy()).to(device)
            all_distances = torch.norm(generated_cart_poses - original_cart_poses, p=2, dim=1)
            mean_distance = all_distances.mean().to(device)
            mean_dist += mean_distance

            optimizer.zero_grad()
            batch_loss.backward()

            clip_value = 0.5  # set the clip value threshold
            torch.nn.utils.clip_grad_value_(flow_network.parameters(), clip_value)

            optimizer.step()

            epoch_loss += batch_loss.item()
        
        print(f"epoch: {epoch}, loss: {epoch_loss/len(train_loader)}, mean dist: {mean_dist/len(train_loader)}")
        all_epoch_loss.append(epoch_loss/len(train_loader))
        all_mean_dist.append(mean_dist/len(train_loader))
    
    return all_epoch_loss, all_mean_dist

def train_archive_distill(flow_network,
                          train_loader,
                          num_iters,
                          optimizer,
                          learning_rate):
    """Train normalizing flow network with cartesian pose and objective as context.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow_network.to(device)
    optimizer = optimizer(flow_network.parameters(), lr=learning_rate)

    all_epoch_loss = []
    all_mean_dist = []

    for epoch in trange(num_iters):
        epoch_loss = 0.
        mean_dist = 0.

        for i, (data_tuple) in enumerate(tqdm(train_loader)):
            original_arm_poses = data_tuple[0].to(device)
            original_context = data_tuple[1].to(device)
            
            original_cart_poses = original_context[:,:-1]

            batch_loss = -flow_network(original_context).log_prob(original_arm_poses)
            batch_loss = batch_loss.mean()

            generated_arm_poses = flow_network(original_context).sample().to(device)  

            # calculate measure diff
            generated_cart_poses = get_cartesian_batched(generated_arm_poses.cpu().detach().numpy()
                                                         ).to(device)
            all_distances = torch.norm(generated_cart_poses - original_cart_poses, p=2, dim=1)
            mean_distance = all_distances.mean().to(device)
            mean_dist += mean_distance

            optimizer.zero_grad()
            batch_loss.backward()

            clip_value = 0.5  # anything > is risky (according to tests)
            torch.nn.utils.clip_grad_value_(flow_network.parameters(), clip_value)

            optimizer.step()

            epoch_loss += batch_loss.item()
        
        print(f"epoch: {epoch}, loss: {epoch_loss/len(train_loader)}, mean dist: {mean_dist/len(train_loader)}")#, mean_obj_diff: {mean_obj_diff/len(train_loader)}")
        
        all_epoch_loss.append(epoch_loss/len(train_loader))
        all_mean_dist.append(mean_dist/len(train_loader))
    
    return all_epoch_loss, all_mean_dist
