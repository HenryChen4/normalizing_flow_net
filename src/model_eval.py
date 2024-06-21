import os
import torch.nn as nn
import torch
from model_loading import generate_data, get_cartesian, load_data

from tqdm import tqdm, trange

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    return model

def compare(untrained_model, 
            trained_model_name, 
            permute_seed,
            arm_dim, 
            batch_size, 
            num_rows,
            random_sample_seed):
    """Compares untrained vs trained model
    
    Args:

    """
    save_dir = "results/2d_arm"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, trained_model_name)
    trained_model = load_model(model_save_path)

    data = generate_data(arm_dim=arm_dim,
                         num_rows=num_rows,
                         random_sample_seed=random_sample_seed)
    data_loader = load_data(data=data,
                            batch_size=batch_size)
    
    for i, data_tuple in enumerate(tqdm(data_loader)):
        print(data_tuple)