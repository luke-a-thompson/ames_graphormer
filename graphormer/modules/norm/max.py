import torch
import torch.nn as nn


class MaxNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = x.abs().max(dim=-1, keepdim=True).values + self.eps
        x_normalized = x / max_val
        return self.gamma * x_normalized + self.beta
