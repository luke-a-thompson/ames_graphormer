import torch
import torch.nn as nn


class CRMSNorm(nn.Module):
    """Implementation of https://ar5iv.labs.arxiv.org/html/2305.14858"""

    def __init__(self, input_dim: int, eps: float = 1e-8):
        super(CRMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        discarded_element_sq = x.sum(dim=-1, keepdim=True).square()
        sum_sq = x.square().sum(dim=-1, keepdim=True)
        crms = (((sum_sq + discarded_element_sq) / (x.shape[-1] + 1)) + self.eps).rsqrt()
        x_normalized = x * crms

        return self.gamma * x_normalized + self.beta
