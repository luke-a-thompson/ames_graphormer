from typing import List, Optional
import torch
import torch.nn.utils.rnn as rnn
from torch import nn

from graphormer.modules.encoding import CentralityEncoding, EdgeEncoding, SpatialEncoding
from graphormer.modules.layers import GraphormerEncoderLayer
from graphormer.config.options import NormType, AttentionType

import warnings
from torch.jit import TracerWarning  # type: ignore

warnings.filterwarnings("ignore", category=TracerWarning)


class Graphormer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        node_feature_dim: int,
        hidden_dim: int,
        edge_feature_dim: int,
        edge_embedding_dim: int,
        ffn_hidden_dim: int,
        output_dim: int,
        max_in_degree: int,
        max_out_degree: int,
        max_path_distance: int,
        dropout: float = 0.05,
        norm_type: NormType = NormType.LAYER,
        attention_type: AttentionType = AttentionType.MHA,
        n_heads: Optional[int] = None,
        heads_by_layer: Optional[List[int]] = None,
        local_heads_by_layer: Optional[List[int]] = None,
        global_heads_by_layer: Optional[List[int]] = None,
        n_local_heads: Optional[int] = None,
        n_global_heads: Optional[int] = None,
    ):
        """
        :param num_layers: number of Graphormer layers
        :param node_feature_dim: input dimension of node features
        :param hidden_dim: hidden dimensions of node features
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
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.output_dim = output_dim
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance
        self.n_heads = n_heads
        self.norm_type = norm_type
        self.heads_by_layer = heads_by_layer
        self.n_local_heads = n_local_heads
        self.n_global_heads = n_global_heads
        self.global_heads_by_layer = global_heads_by_layer
        self.local_heads_by_layer = local_heads_by_layer
        self.dropout = dropout
        self.attention_type = attention_type

        self.node_embedding = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.edge_feature_dim, self.edge_embedding_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            hidden_dim=self.hidden_dim,
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.edge_encoding = EdgeEncoding(self.edge_embedding_dim, self.max_path_distance)

        layers = None
        match self.attention_type:
            case AttentionType.MHA | AttentionType.LINEAR:
                layers = self.mha_layers()
            case AttentionType.FISH:
                layers = self.fish_layers()

        self.layers = nn.ModuleList(layers)

        self.out_lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.apply(Graphormer._init_weights)

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def fish_layers(self):
        if self.n_global_heads is None and self.global_heads_by_layer is None:
            raise AttributeError("global_heads or global_heads_by_layer must be defined")
        if self.n_local_heads is None and self.local_heads_by_layer is None:
            raise AttributeError("local_heads or local_heads_by_layer must be defined")

        if self.global_heads_by_layer is None or len(self.global_heads_by_layer) == 0:
            assert self.n_global_heads is not None
            self.global_heads_by_layer = [self.n_global_heads for _ in range(self.num_layers)]

        if self.local_heads_by_layer is None or len(self.local_heads_by_layer) == 0:
            assert self.n_local_heads is not None
            self.local_heads_by_layer = [self.n_local_heads for _ in range(self.num_layers)]

        layers = []
        for n_global_heads, n_local_heads in zip(self.global_heads_by_layer, self.local_heads_by_layer):
            if n_global_heads != n_local_heads:
                layers.append(
                    GraphormerEncoderLayer(
                        hidden_dim=self.hidden_dim,
                        n_global_heads=n_global_heads,
                        n_local_heads=n_local_heads,
                        ffn_dim=self.ffn_hidden_dim,
                        ffn_dropout=self.dropout,
                        attn_dropout=self.dropout,
                        norm_type=self.norm_type,
                        attention_type=self.attention_type,
                    )
                )
            else:
                layers.append(
                    GraphormerEncoderLayer(
                        hidden_dim=self.hidden_dim,
                        n_heads=n_global_heads,
                        ffn_dim=self.ffn_hidden_dim,
                        ffn_dropout=self.dropout,
                        attn_dropout=self.dropout,
                        norm_type=self.norm_type,
                        attention_type=AttentionType.MHA,
                    )
                )
        return layers

    def mha_layers(
        self,
    ):
        if self.n_heads is None and self.heads_by_layer is None:
            raise ValueError("n_heads or heads_by_layer must be defined.")
        if self.heads_by_layer is None or len(self.heads_by_layer) == 0:
            assert self.n_heads is not None
            self.heads_by_layer = [self.n_heads for _ in range(self.num_layers)]

        if len(self.heads_by_layer) != self.num_layers:
            raise ValueError(
                f"The length of heads_by_layer {len(self.heads_by_layer)} must equal the number of layers {self.num_layers}"
            )

        return [
            GraphormerEncoderLayer(
                hidden_dim=self.hidden_dim,
                n_heads=n_heads,
                ffn_dim=self.ffn_hidden_dim,
                ffn_dropout=self.dropout,
                attn_dropout=self.dropout,
                norm_type=self.norm_type,
                attention_type=self.attention_type,
            )
            for n_heads in self.heads_by_layer
        ]

    @classmethod
    def _init_weights(cls, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_paths: torch.Tensor,
        edge_paths: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: A list of nodes and their corresponding input features
        :param edge_index: Two lists of nodes, where pair of matching indices represents an edge
        :param edge_attr: A list of all the edges in the batch and their corresponding features
        :param ptr: A set of half-open ranges, where each range indicates the specific index of the starting node of the graph in the batch
        :param node_paths: The paths from each node in the subgraph to every other node in the subgraph as defined by the node indices.
        :param edge_paths: The paths from each node in the subgraph to every other node in the subgraph as defined by the edge indices.
        :return: torch.Tensor, output node embeddings
        """

        x = self.node_embedding(x)
        x = self.centrality_encoding(x, edge_index)

        edge_embedding = self.edge_embedding(edge_attr)
        spatial_encoding = self.spatial_encoding(x, node_paths)
        edge_encoding = self.edge_encoding(edge_embedding, edge_paths)

        padded_encoding_bias = spatial_encoding + edge_encoding

        for layer in self.layers:
            x = layer(x, padded_encoding_bias)

        # Output for each VNODE for each graph
        vnode_outputs = x[:, 0, :]
        out = self.out_lin(vnode_outputs)

        return out.squeeze()
