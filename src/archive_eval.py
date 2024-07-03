import os
import torch.nn as nn
import torch
from model_loading import (
    generate_data, 
    get_cartesian,
    get_cartesian_batched
)

from model_eval import load_model
from visualize import visualize

import matplotlib.pyplot as plt
import numpy as np
from ribs.emitters import EmitterBase

def visualize_point(model, context, num_arms):
    all_arm_poses = []
    for i in range(num_arms):
        all_arm_poses.append(model(context).sample())

    link_lengths = np.ones(shape=(num_arms, ))
    objectives = np.ones(shape=(num_arms, ))
    objectives = np.repeat(context[-1], num_arms)
    _, ax = plt.subplots()

    visualize(all_arm_poses, link_lengths, objectives, ax, context[:-1])
    plt.show()

# visualize model behavior
flow_model = load_model("../results/archive_distill2/model_test.pth")
visualize_point(flow_model, torch.tensor([-7, 6, -0.01]), 20)
