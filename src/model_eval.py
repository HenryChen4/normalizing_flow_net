import os
import torch.nn as nn
import torch
from model_loading import generate_data, get_cartesian, load_data

from tqdm import tqdm, trange

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    return model

def compare(untrained_model, trained_model, permute_seed, model_settings):
    """Compare untrained vs trained model
    
    Args:
        untrained_model (Normalizing_Flow_Net): Untrained nfn model.
        untrained_model (Normalizing_Flow_Net): Trained nfn model.
        permute_seed (int): Seed used to permute layers in both models.
        model_settings (dict): HP and architecture configurations for both models.
    """

    

def evaluate(test_data, model, permute_seed, batch_size):
    test_loader = load_data(test_data, batch_size)

    mean_dist = 0.0

    for i, (sampled_arm, target_car_x, target_car_y) in enumerate(tqdm(test_loader)):
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

    save_dir = f"more_results/result4/"
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

# TODO: Clean this up and change to loading batches instead of single data