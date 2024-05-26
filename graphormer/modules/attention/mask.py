import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class AttentionPadMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: ModelData) -> ModelData:
        data.x[data.x_pad_mask] = torch.zeros(data.x.shape[-1]).to(data.device)
        return data
