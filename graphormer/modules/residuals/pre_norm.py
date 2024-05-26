import torch.nn as nn

from graphormer.modules.model_data import ModelData


class PreNormResidual(nn.Module):
    def __init__(
        self, input_normalization: nn.Module, attention: nn.Module, attention_normalization: nn.Module, ffn: nn.Module
    ):
        super().__init__()
        self.input_normalization = input_normalization
        self.attention = attention
        self.attention_normalization = attention_normalization
        self.ffn = ffn

    def forward(self, data: ModelData) -> ModelData:
        # data.normalized_input <- normalized data.x
        self.input_normalization(data)
        # data.attention_output <- attention(normalized_input)
        self.attention(data)

        data.x = data.attention_output + data.x

        # data.normalized_input <- normalized data.x
        self.attention_normalization(data)
        # data.ffn_output <- ffn(normalized_input)
        self.ffn(data)

        data.x = data.ffn_output + data.x
        return data
