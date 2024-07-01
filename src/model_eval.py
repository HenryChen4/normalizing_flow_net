import os
import torch.nn as nn
import torch
from model_loading import generate_data, get_cartesian

from tqdm import tqdm, trange

def load_model(model_save_path):
    model = torch.load(model_save_path, map_location=torch.device("cpu"))
    return model

model = load_model("./results/dummy_test")