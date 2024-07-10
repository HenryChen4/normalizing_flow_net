import sys
import numpy as np
from tqdm import trange

import torch

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from src.model_loading import create_loader

def simulate(solutions, link_lengths):
    """Objective and measure function"""
    objs = 1.0 - np.var(solutions, axis=1)

    cum_theta = np.cumsum(solutions, axis=1)
    x_pos = link_lengths[None] * np.cos(cum_theta)
    y_pos = link_lengths[None] * np.sin(cum_theta)

    meas = np.concatenate(
        (
            np.sum(x_pos, axis=1, keepdims=True),
            np.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs, meas

def torch_simulate(solutions, link_lengths):
    """simulate but with torch"""
    objs = -torch.std(solutions, dim=1)

    cum_theta = torch.cumsum(solutions, dim=1)
    x_pos = link_lengths[None] * torch.cos(cum_theta)
    y_pos = link_lengths[None] * torch.sin(cum_theta)

    meas = torch.stack(
        (
            torch.sum(x_pos, dim=1),
            torch.sum(y_pos, dim=1)
        ),
        axis=1
    )

    return objs, meas

def create_scheduler(arm_dim, 
                     cells=10000,
                     sigma0=0.1,
                     batch_size=30,
                     num_emitters=5):
    link_lengths = np.ones(arm_dim)
    max_pos = np.sum(link_lengths)
    archive = CVTArchive(
        solution_dim=arm_dim,
        cells=cells,
        ranges=[(-max_pos, max_pos), (-max_pos, max_pos)],
        use_kd_tree=True,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive=archive,
            x0=np.zeros(arm_dim),
            sigma0=sigma0,
            ranker="2imp",
            bounds=[(-np.pi, np.pi)] * arm_dim,
            batch_size=batch_size,
        ) for _ in range(num_emitters) 
    ]

    scheduler = Scheduler(archive, emitters)

    return archive, scheduler

def fill_archive(arm_dim, scheduler, archive, num_iters):
    en_route_sols = []

    """Runs QD algorithm"""
    for itr in trange(1, num_iters + 1, desc='Iterations', file=sys.stdout):
        sols = scheduler.ask()
        objs, meas = simulate(sols, link_lengths=np.ones(arm_dim))

        if len(sols) != len(objs) or len(sols) != len(meas):
            raise("Size mismatch, ribs issue.")
        
        for i in range(len(sols)):
            solution_dict = {
                "solution": None,
                "objective": None,
                "measures": None
            }

            solution_dict["solution"] = sols[i]
            solution_dict["objective"] = objs[i]
            solution_dict["measures"] = meas[i]

            en_route_sols.append(solution_dict)

        scheduler.tell(objs, meas)

    return archive, en_route_sols

def gather_solutions(arm_dim, 
                     num_qd_iters, 
                     train_batch_size,
                     cells=10000,
                     sigma0=0.1,
                     batch_size=30,
                     num_emitters=5):
    all_sols = []
    
    archive, scheduler = create_scheduler(arm_dim=arm_dim,
                                          cells=cells,
                                          sigma0=sigma0,
                                          batch_size=batch_size,
                                          num_emitters=num_emitters)
    archive, en_route_solns = fill_archive(arm_dim=arm_dim,
                                           scheduler=scheduler,
                                           archive=archive,
                                           num_iters=num_qd_iters)
    
    for sol in en_route_solns:
        arm_pose = sol["solution"]
        objective = sol["objective"]
        measures = sol["measures"]

        train_tuple = (torch.tensor(arm_pose, dtype=torch.float64),
                       torch.cat((torch.tensor(measures), torch.tensor(objective).unsqueeze(dim=0))))

        all_sols.append(train_tuple)

    for elite in archive:
        arm_pose = elite["solution"]
        objective = elite["objective"]
        measures = elite["measures"]

        single_train_tuple = (
            torch.tensor(arm_pose, dtype=torch.float64),
            torch.cat((torch.tensor(measures), torch.tensor(objective).unsqueeze(dim=0)))
        )
        
        all_sols.append(single_train_tuple)
    
    train_loader = create_loader(all_sols, train_batch_size, shuffle=True)
    return train_loader