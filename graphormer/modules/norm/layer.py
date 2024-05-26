import torch.nn as nn

from graphormer.modules.model_data import ModelData


class LayerNorm(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, data: ModelData) -> ModelData:
        data.normalized_input = self.layer_norm(data.x)
        return data
