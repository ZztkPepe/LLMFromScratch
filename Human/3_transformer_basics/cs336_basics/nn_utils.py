import math
import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)