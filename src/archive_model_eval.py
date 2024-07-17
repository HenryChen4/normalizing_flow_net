"""Evaluate archive model
"""

from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch

from src.create_archive import simulate
from src.model_eval import load_model, visualize_point

def get_evaluation_archives(archive_model, 
                            arm_dim, 
                            device):
    """Creates a grid archive and evaluates the archive model at the center of
    every grid cell."""
    archives = {
        # Shows the objective in every cell. However, the features may be
        # incorrect.
        "objective": None,
        # Shows the feature error in every cell.
        "feature_error": None,
        # Places each solution into the correct cell. Elitism rules are
        # enforced, i.e., if two solutions land in the same cell we only take
        # the better one.
        "corrected": None,
    }

    for key in archives:
        archives[key] = GridArchive(
            solution_dim=arm_dim,
            dims=[100, 100],
            ranges=[[-arm_dim, arm_dim], [-arm_dim, arm_dim]]
        )

    centers = [(b[:-1] + b[1:]) / 2.0 for b in archives["objective"].boundaries]

    feature_coords = np.meshgrid(*centers)
    feature_grid = np.stack([x.ravel() for x in feature_coords], axis=1)
    
    # set archive model to eval mode
    archive_model.eval()
    with torch.no_grad():
        feature_grid_torch = torch.tensor(feature_grid,
                                          dtype=torch.float32,
                                          device=device)
        
        # set objective to max, in this case, 1.0
        objective = torch.ones(feature_grid_torch.shape[0], 1)
        context = torch.cat((feature_grid_torch, objective), dim=1)

        # pylint: disable-next = not-callable
        solutions: torch.Tensor = archive_model(context).sample()

        solutions = solutions.detach().cpu().numpy()

        objectives, features = simulate(solutions, link_lengths=np.ones(arm_dim))

        sorted_indices = np.argsort(objectives)
        objectives_sorted = objectives[sorted_indices] 

        i = 0
        while objectives_sorted[i] < 0:
            print(f"setting {objectives_sorted[i]} to 0")
            objectives_sorted[i] = 0
            i += 1
        print("beautifying done!")

        objectives = np.zeros_like(objectives_sorted)
        objectives[sorted_indices] = objectives_sorted

    # set archive model to train mode
    archive_model.train()

    archives["objective"].add(solutions, objectives, feature_grid)
    archives["feature_error"].add(
        solutions,
        # pylint: disable-next = not-callable
        np.linalg.norm(features - feature_grid, axis=1),
        feature_grid)
    archives["corrected"].add(solutions, objectives, features)

    return archives

def visualize_archives(archives):
    # Create a figure with 1 row and 3 columns of subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Iterate through your archives
    for i, archive_type in enumerate(archives):
        grid_archive_heatmap(archives[archive_type], ax=axes[i])
        print(archives[archive_type].stats)
        axes[i].set_title(f"Archive: {archive_type}")
        axes[i].set_aspect('equal')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()

# testing distill 3 model
model = load_model("./results/archive_distill100d/model_test.pth")

archives = get_evaluation_archives(archive_model=model,
                                   arm_dim=10,
                                   device="cpu")

visualize_archives(archives)
