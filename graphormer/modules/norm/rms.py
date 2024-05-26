import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class RMSNorm(nn.Module):
    """Implementation of https://ar5iv.labs.arxiv.org/html/1910.07467"""

    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, data: ModelData) -> ModelData:
        square_mean = data.x.square().mean(dim=-1, keepdim=True)
        rms = (square_mean + self.eps).rsqrt()
        x_normalized = data.x * rms
        data.normalized_input = self.gamma * x_normalized + self.beta
        return data
