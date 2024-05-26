import torch.nn as nn

from graphormer.modules.model_data import ModelData


class EdgeEmbedding(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(feature_dim, embedding_dim)

    def forward(self, data: ModelData) -> ModelData:
        edge_attr = data.edge_attr
        edge_pad_mask = (edge_attr == -1).all(dim=-1)
        edge_embedding = self.embedding(edge_attr)
        edge_embedding[edge_pad_mask, :] = -1
        data.edge_embedding = edge_embedding
        return data
