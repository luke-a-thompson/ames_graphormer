import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Implementation of https://ar5iv.labs.arxiv.org/html/1910.07467"""

    def __init__(self, input_dim: int, eps: float = 1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_mean = (x**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(square_mean + self.eps)
        x_normalized = x / rms
        return self.gamma * x_normalized + self.beta


class CRMSNorm(nn.Module):
    """Implementation of https://ar5iv.labs.arxiv.org/html/2305.14858"""

    def __init__(self, input_dim: int, eps: float = 1e-8):
        super(CRMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = (x**2).mean(dim=-1, keepdim=True)
        sq_mean = x.mean(dim=-1, keepdim=True) ** 2
        crms = torch.sqrt(mean_sq + sq_mean + self.eps)
        x_normalized = x / crms
        return self.gamma * x_normalized + self.beta


class MaxNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-8):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = x.abs().max(dim=-1, keepdim=True).values + self.eps
        x_normalized = x / max_val
        return self.gamma * x_normalized + self.beta
