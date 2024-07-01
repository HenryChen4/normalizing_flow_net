import os
import torch.nn as nn
import torch
from model_loading import (
    generate_data, 
    get_cartesian,
    get_cartesian_batched
)

from tqdm import (
    tqdm, trange
)

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    return model

def test_model(flow_model, 
               arm_dim,
               num_test_samples, 
               test_sample_seed):
    # generate test samples
    torch.manual_seed(test_sample_seed)
    random_arm_sample = torch.rand((num_test_samples, arm_dim))
    original_cart_coords = get_cartesian_batched(arm_poses=random_arm_sample)
    
    # sample arms from flow model
    generated_arm_sample = flow_model(original_cart_coords).sample()
    generated_cart_coords = get_cartesian_batched(generated_arm_sample)

    # compute mean distance between generated cart coords and original
    return generated_arm_sample, torch.norm(generated_cart_coords - original_cart_coords, p=2, dim=1).mean()

flow_model = load_model("results/20_iters/model_test.pth")

arm_dim = 10
num_test_samples = 20
test_sample_seed = 13570

generated_arm_sample, mean_dist = test_model(flow_model=flow_model,
                       arm_dim=arm_dim,
                       num_test_samples=num_test_samples,
                       test_sample_seed=test_sample_seed)

print(generated_arm_sample)
print(mean_dist)