import torch
from torch import nn
from torch_geometric.data import Data

from graphormer.functional import shortest_path_distance
from graphormer.layers import (
    CentralityEncoding,
    EdgeEncoding,
    GraphormerEncoderLayer,
    SpatialEncoding,
    flatten_paths_tensor,
)


class Graphormer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_node_dim: int,
        node_dim: int,
        input_edge_dim: int,
        edge_dim: int,
        output_dim: int,
        n_heads: int,
        max_in_degree: int,
        max_out_degree: int,
        max_path_distance: int,
    ):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim,
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.edge_encoding = EdgeEncoding(self.edge_dim, self.max_path_distance)

        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    n_heads=self.n_heads,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None

        x = data.x.float()

        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        node_paths, edge_paths = shortest_path_distance(data)

        flattened_edge_paths = flatten_paths_tensor(edge_paths)
        flattened_node_paths = flatten_paths_tensor(node_paths)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)
        # (num, edge_dim)

        x = self.centrality_encoding(x, edge_index)
        edge_encoding = self.edge_encoding(x, edge_attr, flattened_edge_paths)
        spatial_encoding = self.spatial_encoding(x, flattened_node_paths)

        for layer in self.layers:
            x = layer(x, spatial_encoding, edge_encoding)

        x = self.node_out_lin(x)

        return x
