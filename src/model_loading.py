import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def sample_arm_input(arm_dim, seed):
    rng = np.random.default_rng(seed=seed)
    return torch.tensor(rng.normal(loc=0.0, scale=np.pi/3, size=(arm_dim, )), dtype=torch.float64)

def get_cartesian(arm_pos):
    theta_sum = 0
    x_final = 0
    y_final = 0
    for theta in arm_pos:
        theta_sum += theta
        x_final += np.cos(theta_sum)
        y_final += np.sin(theta_sum)
    return torch.tensor((x_final, y_final), dtype=torch.float64)

def get_cartesian_batched(arm_poses):
    all_arm_poses = []
    for arm_pose in arm_poses:
        all_arm_poses.append(get_cartesian(arm_pose))
    return torch.stack(all_arm_poses)     

def generate_data(arm_dim, num_rows, random_sample_seed):
    data = []
    for i in range(num_rows):
        arm_solution = sample_arm_input(arm_dim=arm_dim, 
                                        seed=random_sample_seed+i)
        car_x, car_y = get_cartesian(arm_solution)
        
        data.append((arm_solution, 
                     torch.cat((car_x.unsqueeze(dim=0), 
                                car_y.unsqueeze(dim=0)))))
    return data

class Arm_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def create_loader(data, batch_size):
    x = [item[0] for item in data]
    y = [item[1] for item in data]

    dataset = Arm_Dataset(x, y)
    dataloader = DataLoader(dataset, batch_size)

    return dataloader