import torch
import torch.nn as nn


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
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_embedding_dim))
        self.eps = 1e-9

    def forward(
        self,
        edge_embedding: torch.Tensor,
        edge_paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (batch_size, num_nodes, hidden_dim)
        :param edge_embedding: edge feature matrix, shape (batch_size, num_edges, edge_dim)
        :param edge_paths: pairwise node paths in edge indexes, shape (batch_size, num_nodes ** 2 + padding, path of edge indexes to traverse from node_i to node_j where len(edge_paths) = max_path_length)
        :return: torch.Tensor, Edge Encoding
        """
        batch_size = edge_paths.shape[0]
        edge_mask = edge_paths == -1
        edge_paths_clamped = edge_paths.clamp(min=0)
        batch_indices = torch.arange(batch_size).view(batch_size, 1, 1).expand_as(edge_paths)

        # Get the edge embeddings for each edge in the paths (when defined)
        edge_path_embeddings = edge_embedding[batch_indices, edge_paths_clamped, :]
        edge_path_embeddings[edge_mask] = 0.0

        path_lengths = (~edge_mask).sum(dim=-1) + self.eps

        # Get sum of embeddings * self.edge_vector for edge in the path,
        # then sum the result for each path
        # b: batch_size
        # e: padded num_nodes**2
        # l: max_path_length
        # d: edge_emb_dim
        # (batch_size, padded_num_nodes**2)
        edge_path_encoding = torch.einsum("beld,ld->be", edge_path_embeddings, self.edge_vector)

        # Find the mean embedding based on the path lengths
        # shape: (batch_size, padded_num_node_pairs)
        edge_path_encoding = edge_path_encoding.div(path_lengths)
        return edge_path_encoding
