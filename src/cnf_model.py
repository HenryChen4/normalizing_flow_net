import torch
import torch.nn as nn

from zuko.flows.continuous import CNF
from zuko.flows import Flow, UnconditionalDistribution
from zuko.distributions import DiagNormal

from tqdm import trange, tqdm

# TODO: make this work for all domains

def create_cnf(solution_dim,
               num_transforms,
               num_context,
               hypernet_config,
               sigma):
    """Creates continuous normalizing flow

    Args:
        solution_dim (int): Dimension of domain solutions.
        num_transforms (int): Number of transformations.
        num_context (int): Number of context variables.
        hypernet_config (dict): Dictionary for ODE neural network.
    """
    transforms = []
    for i in range(num_transforms):
        single_transform = CNF(features=solution_dim,
                               context=num_context,
                               **hypernet_config)
        transforms.append(single_transform)
    flow = Flow(
        transform=transforms,
        base=UnconditionalDistribution(
            DiagNormal,
            loc=torch.full((solution_dim, ), 0, dtype=torch.float32),
            scale=torch.full((solution_dim, ), sigma, dtype=torch.float32),
            buffer=True
        )
    )

    return flow

def train(cnf_network,
          train_loader,
          num_iters,
          optimizer,
          learning_rate,
          domain_func):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnf_network.to(device)
    optimizer = optimizer(cnf_network.parameters(), lr=learning_rate)
    
    all_epoch_loss = []
    all_feature_err = []

    for epoch in trange(num_iters):
        epoch_loss = 0.
        mean_err = 0.

        for i, (data_tuple) in enumerate(tqdm(train_loader)):
            original_soln = data_tuple[0].to(device)
            original_context = data_tuple[1].to(device)

            original_measures = original_context[:, :-1]

            batch_loss = -cnf_network(original_context).log_prob(original_soln)
            batch_loss = batch_loss.mean()

            generated_soln = cnf_network(original_context).sample().to(device)
            generated_measures, _ = domain_func(generated_soln.cpu().detach().numpy()).to(device)
            all_feature_err = torch.norm(generated_measures - original_measures, p=2, dim=1)
            mean_feature_err = all_feature_err.mean().to(device)
            mean_err += mean_feature_err

            optimizer.zero_grad()
            batch_loss.backward()

            clip_value = 0.5
            torch.nn.utils.clip_grad_value_(cnf_network.parameters(), clip_value)

            optimizer.step()

            epoch_loss += batch_loss.item()

        print(f"epcoh: {epoch}, loss: {epoch_loss/len(train_loader)}, feature error: {mean_err/len(train_loader)}")

        all_epoch_loss.append(epoch_loss/len(train_loader))
        all_feature_err.append(mean_err/len(train_loader))
    
    return all_epoch_loss, all_feature_err

hypernet_config = {
    "hidden_features": (128, 128, 128),
    "activation": nn.ELU
}

flow = create_cnf(solution_dim=10,
                  num_transforms=5,
                  num_context=3,
                  hypernet_config=hypernet_config,
                  sigma=torch.pi/3)

