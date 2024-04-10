from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import degree

from graphormer.utils import decrease_to_max_value


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
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
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """

        paths_flat = paths.flatten(0, 1).to(x.device)
        paths_mask = (paths_flat != -1).to(x.device)
        path_lengths = paths_mask.sum(dim=1)
        length_mask = path_lengths != 0
        max_lengths = torch.full_like(path_lengths, self.max_path_distance)
        b_idx = torch.minimum(path_lengths, max_lengths) - 1
        spatial_matrix = torch.zeros_like(b_idx, dtype=torch.float)
        spatial_matrix[length_mask] = self.b[b_idx][length_mask]
        spatial_matrix = spatial_matrix.reshape((x.shape[0], x.shape[0]))
        return spatial_matrix


def flatten_paths_tensor(
    paths: Dict[int, Dict[int, List[int]]], max_path_length: int = 5
) -> torch.Tensor:
    nodes = paths.keys()

    tensor_paths = torch.full(
        (len(nodes), len(nodes), max_path_length), -1, dtype=torch.int
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
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(
            torch.randn(self.max_path_distance, self.edge_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (num_nodes, node_dim)
        :param edge_attr: edge feature matrix, shape (num_edges, edge_dim)
        :param edge_paths: pairwise node paths in edge indexes, shape (num_nodes, num_nodes, max_path_len)
        :return: torch.Tensor, Edge Encoding matrix
        """
        # shape (num_nodes**2, max_path_len)
        edge_paths_flat = edge_paths.flatten(0, 1).to(x.device)

        edge_mask = (edge_paths_flat != -1).to(x.device)
        edge_indices = torch.where(edge_mask, edge_paths_flat, 0).to(x.device)
        path_lengths = edge_mask.sum(dim=1)

        selected_edges = edge_attr[edge_indices].to(x.device)

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

        edge_embeddings = torch.full((edge_paths_flat.shape[0], self.edge_dim), 0.0).to(
            x.device
        )
        edge_embeddings = (
            (extended_edge_vector * masked_path_attrs).sum(dim=2).to(x.device)
        )
        # Find the mean based on the path lengths
        edge_embeddings = edge_embeddings.sum(dim=1)

        non_empty_paths = path_lengths != 0
        edge_embeddings[non_empty_paths] = edge_embeddings[non_empty_paths].div(
            path_lengths[non_empty_paths]
        )

        cij = edge_embeddings.reshape((x.shape[0], x.shape[0])).to(x.device)
        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_q: int,
        dim_k: int,
        edge_dim: int,
        max_path_distance: int,
        attention_dropout_rate: float = 0.1,
    ):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.att_size = att_size = dim_in  # // num_heads # make multiheaded & vectorise
        self.scale = att_size**-0.5

        self.q = nn.Linear(dim_in, dim_in)
        self.k = nn.Linear(dim_in, dim_in)
        self.v = nn.Linear(dim_in, dim_in)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(dim_in * att_size, dim_in)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        b: torch.Tensor,
        edge_paths: torch.Tensor,
        ptr=None,
    ) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(
            size=(x.shape[0], x.shape[0]), fill_value=-1e6, device=x.device
        )
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0]), device=x.device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(
                next(self.parameters()).device
            )
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, edge_paths)
        a = self.compute_a(key, query, ptr)
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i] : ptr[i + 1], ptr[i] : ptr[i + 1]] = (
                    query[ptr[i] : ptr[i + 1]].mm(
                        key[ptr[i] : ptr[i + 1]].transpose(0, 1)
                    )
                    / query.size(-1) ** 0.5
                )

        return a


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_in: int,
        dim_q: int,
        dim_k: int,
        edge_dim: int,
        max_path_distance: int,
    ):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                GraphormerAttentionHead(
                    dim_in, dim_q, dim_k, edge_dim, max_path_distance
                )
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        b: torch.Tensor,
        edge_paths: torch.Tensor,
        ptr,
    ) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat(
                [
                    attention_head(x, edge_attr, b, edge_paths, ptr)
                    for attention_head in self.heads
                ],
                dim=-1,
            )
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        b: torch,
        edge_paths: torch.Tensor,
        ptr,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x

        x_new = self.ff(self.ln_2(x_prime)) + x_prime
        x_new = F.gelu(x_new, approximate="tanh")

        return x_new
