
import torch
import math

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def mape(pred, true, eps=1e-6):
    return torch.mean(torch.abs((pred - true) / (true.abs() + eps))) * 100.0
