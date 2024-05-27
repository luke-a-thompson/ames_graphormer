import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class EdgeEncoding(nn.Module):
    def __init__(self, edge_embedding_dim: int, max_path_distance: int):
        """
        Initializes a new instance of the EdgeEncoding.

        Args:
            edge_embedding_dim (int): The dimension of the edge embeddings.
            max_path_distance (int): The maximum path distance.

        """
        super().__init__()
        self.edge_embedding_dim = edge_embedding_dim
        self.max_path_distance = max_path_distance
        # Prior that further edges are less important than closer ones
        self.edge_vector = nn.Parameter(torch.zeros(self.max_path_distance, self.edge_embedding_dim))
        self.eps = 1e-9

    def forward(self, data: ModelData) -> ModelData:
        """
        :param data.edge_embedding: edge feature matrix, shape (batch_size, num_edges, edge_dim)
        :param data.edge_paths: pairwise node paths in edge indexes, shape (batch_size, num_nodes, num_nodes, path of edge indexes to traverse from node_i to node_j where len(edge_paths) = max_path_length)
        :return: torch.Tensor, Edge Encoding
        """
        batch_size = data.edge_paths.shape[0]
        edge_mask = data.edge_paths == -1
        edge_paths_clamped = data.edge_paths.clamp(min=0)
        batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand_as(data.edge_paths)

        # Get the edge embeddings for each edge in the paths (when defined)
        assert data.edge_embedding is not None
        edge_path_embeddings = data.edge_embedding[batch_indices, edge_paths_clamped, :]
        edge_path_embeddings[edge_mask] = 0.0

        path_lengths = (~edge_mask).sum(dim=-1) + self.eps

        # Get sum of embeddings * self.edge_vector for edge in the path,
        # then sum the result for each path
        # b: batch_size
        # n, m: padded num_nodes
        # l: max_path_length
        # d: edge_emb_dim
        # (batch_size, padded_num_nodes**2)
        edge_path_encoding = torch.einsum("bnmld,ld->bnm", edge_path_embeddings, self.edge_vector)

        # Find the mean embedding based on the path lengths
        # shape: (batch_size, padded_num_node_pairs)
        data.edge_encoding = edge_path_encoding.div(path_lengths)
        return data
