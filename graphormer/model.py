from typing import List, Optional
import torch
import torch.nn.utils.rnn as rnn
from torch import nn

from graphormer.layers import CentralityEncoding, EdgeEncoding, GraphormerEncoderLayer, SpatialEncoding
from graphormer.config.options import NormType, AttentionType

import warnings
from torch.jit import TracerWarning

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
            case AttentionType.MHA:
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

        return [
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
            for n_global_heads, n_local_heads in zip(self.global_heads_by_layer, self.local_heads_by_layer)
        ]

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
        ptr: torch.Tensor,
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

        x = x.float()

        edge_index = edge_index.long()
        edge_attr = edge_attr.float()
        subgraph_idxs = torch.stack([ptr[:-1], ptr[1:]], dim=1)
        # Create VNODE
        vnode = torch.full_like(x[0], -1).unsqueeze(0)

        # Prepare expanded list to accommodate new vnodes
        num_insertions = subgraph_idxs.size(0)
        new_size = x.size(0) + num_insertions
        original_indices = torch.ones(new_size, dtype=torch.bool)
        original_indices[subgraph_idxs[:, 0] + torch.arange(num_insertions).to(x.device)] = 0
        expanded_x = torch.zeros((new_size, x.size(1)), dtype=x.dtype).to(x.device)

        expanded_x[original_indices] = x
        expanded_x[~original_indices] = vnode
        x = expanded_x

        # Adjust ptrs
        new_ptr = subgraph_idxs[:, 0] + torch.arange(num_insertions).to(x.device)
        new_ptr = torch.cat([new_ptr, torch.tensor([new_size]).to(x.device)])
        ptr = new_ptr.int().to(x.device)

        x = self.node_embedding(x)
        x = self.centrality_encoding(x, edge_index)
        spatial_encoding = self.spatial_encoding(x, node_paths)
        edge_embedding = self.edge_embedding(edge_attr)
        edge_encoding = self.edge_encoding(x, edge_embedding, edge_paths)

        # Pad graphs to max graph size
        subgraph_idxs = torch.stack([ptr[:-1], ptr[1:]], dim=1)
        max_size = int(torch.max(subgraph_idxs[:, 1] - subgraph_idxs[:, 0]).item())
        subgraph_sq_ptr = torch.cat(
            [torch.tensor([0]).to(x.device), ((subgraph_idxs[:, 1] - subgraph_idxs[:, 0]) ** 2).cumsum(dim=0)]
        )
        subgraph_sq_idxs = torch.stack([subgraph_sq_ptr[:-1], subgraph_sq_ptr[1:]], dim=1)

        subgraphs = []
        spatial_subgraphs = []
        edge_subgraphs = []
        for idx_range, idx_sq_range in zip(subgraph_idxs.tolist(), subgraph_sq_idxs.tolist()):
            subgraph_size = idx_range[1] - idx_range[0]
            subgraph = x[idx_range[0] : idx_range[1]]
            subgraphs.append(subgraph)

            spatial_subgraph = torch.zeros((max_size, max_size))
            spatial_subgraph[:subgraph_size, :subgraph_size] = spatial_encoding[
                idx_sq_range[0] : idx_sq_range[1]
            ].reshape(subgraph_size, subgraph_size)
            spatial_subgraphs.append(spatial_subgraph.unsqueeze(0))

            edge_subgraph = torch.zeros((max_size, max_size))
            edge_subgraph[:subgraph_size, :subgraph_size] = edge_encoding[idx_sq_range[0] : idx_sq_range[1]].reshape(
                subgraph_size, subgraph_size
            )
            edge_subgraphs.append(edge_subgraph.unsqueeze(0))

        x = rnn.pad_sequence(subgraphs, batch_first=True)
        padded_spatial_subgraphs = torch.cat(spatial_subgraphs).to(x.device)
        padded_edge_subgraphs = torch.cat(edge_subgraphs).to(x.device)
        padded_mask = torch.all(x == 0, dim=-1)

        for layer in self.layers:
            x = layer(x, padded_spatial_subgraphs, padded_edge_subgraphs)

        x = x.flatten(0, 1)[~padded_mask.flatten()]
        # Output for each VNODE for each graph
        vnode_outputs = x[ptr[:-1]]
        out = self.out_lin(vnode_outputs)

        return out.squeeze()
