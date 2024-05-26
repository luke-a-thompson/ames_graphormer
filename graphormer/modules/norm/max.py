import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class MaxNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, data: ModelData) -> ModelData:
        max_val = data.x.abs().max(dim=-1, keepdim=True).values + self.eps
        x_normalized = data.x / max_val
        data.normalized_input = self.gamma * x_normalized + self.beta
        return data
