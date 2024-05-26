import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class CRMSNorm(nn.Module):
    """Implementation of https://ar5iv.labs.arxiv.org/html/2305.14858"""

    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, data: ModelData) -> ModelData:
        discarded_element_sq = data.x.sum(dim=-1, keepdim=True).square()
        sum_sq = data.x.square().sum(dim=-1, keepdim=True)
        crms = (((sum_sq + discarded_element_sq) / (data.x.shape[-1] + 1)) + self.eps).rsqrt()
        x_normalized = data.x * crms
        data.normalized_input = self.gamma * x_normalized + self.beta
        return data
