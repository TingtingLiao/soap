import torch 
import numpy as np


def detach(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach()
    return data

def to_device(data, device='cpu'): 
    new_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if device == 'cpu':
                new_data[k] = v.detach().cpu()
            else:
                new_data[k] = v.to(device)
        elif isinstance(v, dict):
            new_data[k] = to_device(v, device)
        else:
            new_data[k] = v
    return new_data

def to_tensor(data, device='cpu'):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = torch.tensor(v).float()
            if device != 'cpu':
                data[k] = data[k].to(device)
    return data

def save2pth(data, path): 
    torch.save(to_device(data, 'cpu'), path)

def printf(data):
    for k, v in data.items():
        print(k, v.shape)