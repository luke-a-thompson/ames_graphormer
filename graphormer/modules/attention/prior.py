import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class AttentionPrior(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: ModelData) -> ModelData:
        assert data.spatial_encoding is not None
        assert data.edge_encoding is not None
        data.attention_prior = (data.spatial_encoding + data.edge_encoding).unsqueeze(1).contiguous()
        return data
