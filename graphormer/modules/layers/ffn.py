import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
