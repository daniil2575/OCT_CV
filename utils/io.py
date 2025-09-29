import torch

def load_checkpoint(path: str):
    return torch.load(path, map_location='cpu')
