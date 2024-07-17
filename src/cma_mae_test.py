from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from tqdm import tqdm, trange
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# basically instead of having a discrete cma-mae archive representing the solution
# sample, we have an "archive model" that represents the solution sample.
# Goal is to achieve the same coverage as cma-mae for a discrete archive while
# also working in the continuos state of things.

# objective and measure function for arm
def simulate(solutions: torch.Tensor):
    objectives = 1.0 - torch.var(solutions, axis=1)
    cum_theta = torch.cumsum(solutions, axis=1)

    x_pos = torch.cos(cum_theta)
    y_pos = torch.sin(cum_theta)

    features = torch.cat(
        (
            torch.sum(x_pos, axis=1, keepdims=True),
            torch.sum(y_pos, axis=1, keepdims=True)
        ),
        axis=1,
    )

    return objectives, features

def train_models(archive_model, 
                discount_model, 
                total_iters,
                num_emitters=5,
                batch_size=30):
    emitters = [
        EvolutionStrategyEmitter(
            archive=None,
            x0=np.zeros(100),
            sigma0=0.5,
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=batch_size,
        ) for _ in range(num_emitters)
    ]

    

from ribs.archives import GridArchive

max_bound = 100 / 2 * 5.12

archive = GridArchive(solution_dim=100,
                      dims=(100, 100),
                      ranges=[(-max_bound, max_bound), (-max_bound, max_bound)],
                      learning_rate=0.01,
                      threshold_min=0.0)

result_archive = GridArchive(solution_dim=100,
                             dims=(100, 100),
                             ranges=[(-max_bound, max_bound), (-max_bound, max_bound)])

from ribs.emitters import EvolutionStrategyEmitter

emitters = [
    EvolutionStrategyEmitter(
        archive,
        x0=np.zeros(100),
        sigma0=0.5,
        ranker="imp",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=36,
    ) for _ in range(15)
]

from ribs.schedulers import Scheduler

scheduler = Scheduler(archive, emitters, result_archive=result_archive)

total_itrs = 10_000

for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
    solution_batch = scheduler.ask()
    objective_batch, measure_batch = sphere(solution_batch)
    scheduler.tell(objective_batch, measure_batch)

    # Output progress every 500 iterations or on the final iteration.
    if itr % 500 == 0 or itr == total_itrs:
        tqdm.write(f"Iteration {itr:5d} | "
                   f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}%  "
                   f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")