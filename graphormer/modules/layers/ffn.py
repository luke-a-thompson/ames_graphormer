import torch.nn as nn

from graphormer.modules.model_data import ModelData


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: ModelData) -> ModelData:
        assert data.normalized_input is not None

        x = self.layer1(data.normalized_input)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        data.ffn_output = x
        return data
