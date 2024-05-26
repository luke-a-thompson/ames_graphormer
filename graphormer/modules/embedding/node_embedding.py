import torch.nn as nn

from graphormer.modules.model_data import ModelData


class NodeEmbedding(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(self.feature_dim, self.embedding_dim)

    def forward(self, data: ModelData) -> ModelData:
        x_pad_mask = (data.x == -2).all(dim=-1)
        data.x_pad_mask = x_pad_mask
        data.x = self.embedding(data.x)
        data.x[x_pad_mask, :] = -2
        return data
