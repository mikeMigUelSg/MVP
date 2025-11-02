
import torch
import math

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def mape(pred, true, threshold=0.1):
    """
    MAPE que ignora valores abaixo do threshold para evitar divisão por valores próximos de zero
    """
    mask = true.abs() > threshold
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return torch.mean(torch.abs((pred[mask] - true[mask]) / true[mask])) * 100.0
