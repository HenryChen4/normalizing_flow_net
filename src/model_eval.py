import os
import torch.nn as nn
import torch
from model_loading import generate_data, get_cartesian

from tqdm import tqdm, trange

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    return model

def evaluate(test_data, model, permute_seed):
    mean_dist = 0.0

    for i, (sampled_arm, target_car_x, target_car_y) in enumerate(tqdm(test_data)):
        # c set to 0 for testing
        new_arm_sol, _ = model.forward(arm_solutions=sampled_arm,
                                    car_x=target_car_x,
                                    car_y=target_car_y,
                                    c=0,
                                    permute_seed=permute_seed)
        
        sampled_car_x_y = get_cartesian(new_arm_sol.detach().numpy())
        target_car_x_y = torch.cat((target_car_x.unsqueeze(dim=0), target_car_y.unsqueeze(dim=0)))
        
        mean_dist += (sampled_car_x_y - target_car_x_y).pow(2).sum().sqrt()
    
    return mean_dist/len(test_data)

def evaluate_seeds(base_seed, num_seeds, num_rows, permute_seed):
    # load the model
    arm_dim = 10

    save_dir = f"results/test"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'model.pth')

    model = load_model(model_save_path=model_save_path)

    all_mean_dist = []

    for i in trange(num_seeds):
        test_data = generate_data(arm_dim=arm_dim,
                            num_rows=num_rows,
                            random_sample_seed=base_seed+i)

        mean_dist = evaluate(test_data=test_data,
                            model=model,
                            permute_seed=permute_seed)

        all_mean_dist.append(mean_dist)
    
    return all_mean_dist