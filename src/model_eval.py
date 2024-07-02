import os
import torch.nn as nn
import torch
from model_loading import (
    generate_data, 
    get_cartesian,
    get_cartesian_batched
)

from visualize import visualize
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    return model

def test_model(flow_model, 
               arm_dim,
               num_test_samples, 
               test_sample_seed):
    '''Tests model performance on unseen test samples
    '''
    # generate test samples
    torch.manual_seed(test_sample_seed)
    random_arm_sample = torch.rand((num_test_samples, arm_dim))
    original_cart_coords = get_cartesian_batched(arm_poses=random_arm_sample)
    
    # sample arms from flow model
    generated_arm_sample = flow_model(original_cart_coords).sample()
    generated_cart_coords = get_cartesian_batched(generated_arm_sample)

    # compute mean distance between generated cart coords and original
    return generated_arm_sample, torch.norm(generated_cart_coords - original_cart_coords, p=2, dim=1).mean()

def visualize_point(model, cart_pose, num_arms):
    all_arm_poses = []
    for i in range(num_arms):
        all_arm_poses.append(model(cart_pose).sample())

    link_lengths = np.ones(shape=(num_arms, ))
    objectives = np.ones(shape=(num_arms, ))
    fig, ax = plt.subplots()

    visualize(all_arm_poses, link_lengths, objectives, ax, cart_pose)
    plt.show()

# visualize model behavior
flow_model = load_model("../results/dummy_test/model_test.pth")
visualize_point(flow_model, torch.tensor([-7, 6]), 20)
