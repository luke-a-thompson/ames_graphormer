import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class RezeroResidual(nn.Module):
    def __init__(self, attention: nn.Module, ffn: nn.Module):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.alpha = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, data: ModelData) -> ModelData:
        data.normalized_input = data.x
        # data.attention_output <- attention(x)
        self.attention(data)

        data.x = data.x + self.alpha * data.attention_output

        data.normalized_input = data.x
        # data.ffn_output <- ffn(x)
        self.ffn(data)
        data.x = data.x + self.alpha * data.ffn_output

        return data
