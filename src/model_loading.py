import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def sample_arm_input(arm_dim, seed):
    """Sample initial arm pose from normal distribution
    
    Args:
        arm_dim (int): Number of arm joints.
        seed (int): Rng seed.
    Returns:
        arm_solution (torch.tensor): Single sampled arm solution
    """
    rng = np.random.default_rng(seed=seed)
    return torch.tensor(rng.normal(loc=0.0, scale=np.pi/3, size=(arm_dim, )), dtype=torch.float64)

def get_cartesian(arm_pos):
    """Get cartesian coords from arm pos. Link lengths are
    assumed to be 1.
    
    Args:
        arm_pos (torch.tensor): Arm joint position.
    """
    theta_sum = 0
    x_final = 0
    y_final = 0
    for theta in arm_pos:
        theta_sum += theta
        x_final += np.cos(theta_sum)
        y_final += np.sin(theta_sum)
    return torch.tensor((x_final, y_final), dtype=torch.float64)

def generate_data(arm_dim, num_rows, random_sample_seed):
    """Generates training data for neural net.

    Args:
        arm_dim (int): Number of arm joints.
        num_rows (int): Size of dataset.
        random_sample_seed (int): Random sample seed.
    Returns:
        data (tuple): (arm_soln (torch.tensor), x (torch.float64), y (torch.float64))
    """
    data = []
    for i in range(num_rows):
        arm_solution = sample_arm_input(arm_dim=arm_dim, 
                                        seed=random_sample_seed+i)
        car_x, car_y = get_cartesian(arm_solution)
        data.append((arm_solution, 
                     car_x.type(torch.float64), 
                     car_y.type(torch.float64)))
    return data

# TODO: Code batching in model.py
class Arm_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def load_data(data, batch_size):
    x = [item[0] for item in data]
    y = [torch.cat((item[1].unsqueeze(dim=0), item[2].unsqueeze(dim=0))) for item in data]
    
    dataset = Arm_Dataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader