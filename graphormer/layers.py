import torch
from torch import nn
from torch_geometric.utils import degree

from graphormer.norm import CRMSNorm, MaxNorm, RMSNorm
from graphormer.utils import decrease_to_max_value
from graphormer.config.options import NormType


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
        self.t1 = nn.Parameter(torch.randn(1))
        self.t2 = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        """
        :param x: node embedding, shape: (num_nodes, hidden_dim)
        :param paths: pairwise node paths, shape: (num_pairwaise_paths, max_path_length)
        :return: torch.Tensor, spatial encoding
        """

        vnode_out_mask = paths[:, 0] == 0
        vnode_in_mask = paths[:, 1] == 0

        paths_mask = (paths != -1).to(x.device)
        path_lengths = paths_mask.sum(dim=-1)
        length_mask = path_lengths != 0
        max_lengths = torch.full_like(path_lengths, self.max_path_distance)
        b_idx = torch.minimum(path_lengths, max_lengths) - 1
        spatial_encoding = torch.zeros_like(b_idx, dtype=torch.float)
        spatial_encoding[length_mask] = self.b[b_idx][length_mask]
        # Reset VNODE -> Node encodings
        spatial_encoding[vnode_out_mask] = self.t1
        # Reset Node -> VNODE encodings
        spatial_encoding[vnode_in_mask] = self.t2

        return spatial_encoding


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

    def forward(
        self,
        x: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: node feature matrix, shape (num_nodes, hidden_dim)
        :param edge_embedding: edge feature matrix, shape (num_edges, edge_dim)
        :param edge_paths: pairwise node paths in edge indexes, shape (sum(num_nodes ** 2 for each graph in the batch), path of edge indexes to traverse from node_i to node_j where len(edge_paths) = max_path_length)
        :return: torch.Tensor, Edge Encoding
        """
        edge_mask = (edge_paths != -1).to(x.device)
        path_lengths = edge_mask.sum(dim=-1)

        # Get the edge embeddings for each edge in the paths (when defined)
        edge_path_embeddings = torch.full(
            (edge_paths.shape[0], self.max_path_distance, x.shape[1]), 0.0, dtype=edge_embedding.dtype
        ).to(x.device)
        edge_path_embeddings[edge_mask] = edge_embedding[edge_paths].to(x.device)[edge_mask]

        # Get sum of embeddings * self.edge_vector for edge in the path,
        # then sum the result for each path
        edge_path_encoding = (edge_path_embeddings * self.edge_vector).sum(dim=1).sum(dim=-1)

        # Find the mean embedding based on the path lengths
        # shape: (num_node_pairs)
        non_empty_paths = path_lengths != 0

        edge_path_encoding[non_empty_paths] = edge_path_encoding[non_empty_paths].div(path_lengths[non_empty_paths])

        return edge_path_encoding.to(x.device)


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        """
        :param num_heads: number of attention heads
        :param d_x: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.num_heads = num_heads

        self.scale = hidden_dim**-0.5
        self.hidden_dim = hidden_dim
        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim {
            hidden_dim} must be divisible by num_heads {num_heads}"
        self.head_size = hidden_dim // num_heads
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_dropout = nn.Dropout(dropout_rate)

        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: node embedding, shape: (batch_size, num_nodes, hidden_dim)
        :param spatial_encoding: spatial encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :param edge_encoding: edge encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :return: torch.Tensor, node embeddings after all attention heads
        """
        batch_size = x.shape[0]
        max_subgraph_size = x.shape[1]
        hidden_dim = x.shape[2]
        bias = (spatial_encoding + edge_encoding).unsqueeze(1)

        q_x = self.linear_q(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k_x = self.linear_k(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v_x = self.linear_v(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k_x_t = k_x.transpose(-2, -1)
        a = (q_x @ k_x_t) * self.scale
        a += bias
        pad_mask = torch.any(a != 0, dim=-1)
        a[~pad_mask] = float("-inf")
        a = torch.softmax(a, dim=-1)
        a = torch.nan_to_num(a)
        a @= v_x
        a = self.att_dropout(a)
        attn = a.reshape(batch_size, max_subgraph_size, hidden_dim)
        return self.linear_out(attn)


class GraphormerEncoderLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, n_heads: int, ffn_dim=80, ffn_dropout=0.1, norm_type: NormType = NormType.LAYER
    ):
        """
        :param hidden_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.attention = GraphormerMultiHeadAttention(
            num_heads=n_heads,
            hidden_dim=hidden_dim,
        )

        match norm_type:
            case NormType.LAYER:
                self.n1 = nn.LayerNorm(hidden_dim)
                self.n2 = nn.LayerNorm(hidden_dim)
            case NormType.RMS:
                self.n1 = RMSNorm(hidden_dim)
                self.n2 = RMSNorm(hidden_dim)
            case NormType.CRMS:
                self.n1 = CRMSNorm(hidden_dim)
                self.n2 = CRMSNorm(hidden_dim)
            case NormType.MAX:
                self.n1 = MaxNorm(hidden_dim)
                self.n2 = MaxNorm(hidden_dim)
            case NormType.NONE:
                self.n1 = nn.Identity()
                self.n2 = nn.Identity()

        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim, hidden_dim, ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        spatial_encoding: torch.Tensor,
        edge_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements forward pass of the Graphormer encoder layer.

        The correct sequence for operations with residual connections and layer rescaling is:
        1. Normalization (N) is applied to the input.
        2. The rescaled input is passed to MHA or FFN.
        3. The MHA or FFN output is added to the original input, x (residual connection).
        4. The combined (MHA_out + x) output goes through the ffn.

        This results in the following operations:
        h′(l) = MHA(N(h(l−1))) + h(l−1)
        h(l) = FFN(N(h′(l))) + h′(l)

        :param x: node embedding
        :param spatial_encoding: spatial encoding
        :param edge_encoding: encoding of the edges
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        att_input = self.n1(x)
        att_output = self.attention(att_input, spatial_encoding, edge_encoding) + x
        pad_mask = torch.any(att_output == 0, dim=-1)

        ffn_input = self.n2(x)
        ffn_output = self.ffn(ffn_input) + att_output
        ffn_output[pad_mask] = 0

        return ffn_output
