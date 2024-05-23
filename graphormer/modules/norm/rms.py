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
        square_mean = x.square().mean(dim=-1, keepdim=True)
        rms = (square_mean + self.eps).rsqrt()
        x_normalized = x * rms
        return self.gamma * x_normalized + self.beta
