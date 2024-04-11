from typing import Dict, List

import torch
from torch import nn
from torch_geometric.utils import degree

from graphormer.utils import decrease_to_max_value


class FeedForwardNetwork(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float
    ):
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


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, hidden_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.hidden_dim = hidden_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, hidden_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, hidden_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node embedding
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        in_degree = decrease_to_max_value(
            degree(index=edge_index[1], num_nodes=num_nodes).long(),
            self.max_in_degree - 1,
        )
        out_degree = decrease_to_max_value(
            degree(index=edge_index[0], num_nodes=num_nodes).long(),
            self.max_out_degree - 1,
        )

        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """
        :param x: node embedding
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial encoding
        """

        paths_flat = paths.flatten(0, 1).to(x.device)
        paths_mask = (paths_flat != -1).to(x.device)
        path_lengths = paths_mask.sum(dim=1)
        length_mask = path_lengths != 0
        max_lengths = torch.full_like(path_lengths, self.max_path_distance)
        b_idx = torch.minimum(path_lengths, max_lengths) - 1
        spatial_encoding = torch.zeros_like(b_idx, dtype=torch.float)
        spatial_encoding[length_mask] = self.b[b_idx][length_mask]
        return spatial_encoding.reshape((x.shape[0], x.shape[0]))


def flatten_paths_tensor(
    paths: Dict[int, Dict[int, List[int]]], max_path_length: int = 5
) -> torch.Tensor:
    nodes = paths.keys()
    num_nodes = len(nodes)

    tensor_paths = torch.full(
        (num_nodes, num_nodes, max_path_length), -1, dtype=torch.int
    )
    for src, dsts in paths.items():
        for dst, path in dsts.items():
            path_tensor = torch.tensor(
                path[:max_path_length] + [-1] * (max_path_length - len(path)),
                dtype=torch.int,
            )
            tensor_paths[src, dst] = path_tensor
    return tensor_paths


class EdgeEncoding(nn.Module):
    def __init__(self, edge_embedding_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_embedding_dim = edge_embedding_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(
            torch.randn(self.max_path_distance, self.edge_embedding_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (num_nodes, hidden_dim)
        :param edge_attr: edge feature matrix, shape (num_edges, edge_dim)
        :param edge_paths: pairwise node paths in edge indexes, shape (num_nodes, num_nodes, max_path_len)
        :return: torch.Tensor, Edge Encoding
        """
        # shape (num_nodes**2, max_path_len)
        edge_paths_flat = edge_paths.flatten(0, 1).to(x.device)

        edge_mask = (edge_paths_flat != -1).to(x.device)
        edge_indices = torch.where(edge_mask, edge_paths_flat, 0).to(x.device)
        path_lengths = edge_mask.sum(dim=1)

        selected_edges = edge_embedding[edge_indices].to(x.device)

        path_attrs = torch.full(selected_edges.shape, -1, dtype=torch.float).to(
            x.device
        )
        path_attrs[edge_mask] = selected_edges[edge_mask].to(x.device)

        valid_row_mask = (path_attrs != -1).any(dim=2).to(x.device)

        extended_edge_vector = (
            self.edge_vector.unsqueeze(0).expand_as(path_attrs).to(x.device)
        )

        # shape (num_nodes ** 2, max_path_len, edge_dim)
        masked_path_attrs = torch.where(
            valid_row_mask.unsqueeze(-1), path_attrs, 0.0
        ).to(x.device)

        edge_encoding = torch.full((edge_paths_flat.shape[0], self.edge_embedding_dim), 0.0).to(
            x.device
        )
        edge_encoding = (
            (extended_edge_vector * masked_path_attrs).sum(dim=2).to(x.device)
        )
        # Find the mean based on the path lengths
        edge_encoding = edge_encoding.sum(dim=1)

        non_empty_paths = path_lengths != 0
        edge_encoding[non_empty_paths] = edge_encoding[non_empty_paths].div(
            path_lengths[non_empty_paths]
        )

        return edge_encoding.reshape((x.shape[0], x.shape[0])).to(x.device)


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float = 0.1
    ):
        """
        :param num_heads: number of attention heads
        :param d_x: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.num_heads = num_heads

        self.scale = hidden_dim ** -0.5
        self.hidden_dim = hidden_dim
        self.linear_q = nn.Linear(
            hidden_dim, hidden_dim * num_heads, bias=False)
        self.linear_k = nn.Linear(
            hidden_dim, hidden_dim * num_heads, bias=False)
        self.linear_v = nn.Linear(
            hidden_dim, hidden_dim * num_heads, bias=False)
        self.att_dropout = nn.Dropout(dropout_rate)

        self.linear_out = nn.Linear(
            hidden_dim * num_heads, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: node embedding
        :param spatial_encoding: spatial Encoding matrix
        :param edge_encoding: edge encoding matrix
        :return: torch.Tensor, node embeddings after all attention heads
        """
        q = self.linear_q(x).view(x.shape[1], x.shape[0], self.num_heads)
        k = self.linear_k(x).view(x.shape[1], x.shape[0], self.num_heads)
        v = self.linear_v(x).view(x.shape[1], x.shape[0], self.num_heads)

        k = k.transpose(1, 2)
        a = (q @ k) * self.scale
        a = a + spatial_encoding + edge_encoding
        a = torch.softmax(a, dim=2)
        a = self.att_dropout(a)
        out = a @ v
        out = out.reshape((x.shape[0], self.num_heads * x.shape[1]))
        return self.linear_out(out)


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim=80, ffn_dropout=0.1):
        """
        :param hidden_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.att_norm = nn.LayerNorm(hidden_dim)
        self.attention = GraphormerMultiHeadAttention(
            num_heads=n_heads,
            hidden_dim=hidden_dim,
        )
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(
            hidden_dim, ffn_dim, hidden_dim, ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements forward pass of the Graphormer encoder layer.

        The correct sequence for operations with residual connections and layer normalization is:
        1. LayerNorm (LN) is applied to the input.
        2. The LayerNorm'd input is passed to MHA or FFN.
        3. The MHA or FFN output is added to the original input, x (residual connection).
        4. The combined (MHA_out + x) output goes through the ffn.

        This results in the following operations:
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node embedding
        :param spatial_encoding: spatial encoding
        :param edge_encoding: encoding of the edges
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        att_input = self.att_norm(x)
        att_output = self.attention(
            att_input, spatial_encoding, edge_encoding) + x

        ffn_input = self.ffn_norm(att_output)
        ffn_output = self.ffn(ffn_input) + att_output

        return ffn_output

