import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class Output(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: ModelData) -> torch.Tensor:
        vnode_outputs = data.x[:, 0, :]
        output = self.out(vnode_outputs)
        return output.squeeze()
