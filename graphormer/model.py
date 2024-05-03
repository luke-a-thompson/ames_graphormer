import torch
import torch.nn.utils.rnn as rnn
from torch import nn

from graphormer.layers import CentralityEncoding, EdgeEncoding, GraphormerEncoderLayer, SpatialEncoding


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
        n_heads: int,
        max_in_degree: int,
        max_out_degree: int,
        max_path_distance: int,
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
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

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

        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    hidden_dim=self.hidden_dim,
                    n_heads=self.n_heads,
                    ffn_dim=self.ffn_hidden_dim,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.out_lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.apply(Graphormer._init_weights)

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
        subgraph_sq_ptr = torch.cat(
            [torch.tensor([0]).to(x.device), ((subgraph_idxs[:, 1] - subgraph_idxs[:, 0]) ** 2).cumsum(dim=0)]
        )
        subgraph_sq_idxs = torch.stack([subgraph_sq_ptr[:-1], subgraph_sq_ptr[1:]], dim=1)

        subgraphs = []
        spatial_subgraphs = []
        edge_subgraphs = []
        for idx_range, idx_sq_range in zip(subgraph_idxs.tolist(), subgraph_sq_idxs.tolist()):
            subgraph = x[idx_range[0] : idx_range[1]]
            subgraphs.append(subgraph)
            spatial_subgraph = spatial_encoding[idx_sq_range[0] : idx_sq_range[1]]
            spatial_subgraphs.append(spatial_subgraph)
            edge_subgraph = edge_encoding[idx_sq_range[0] : idx_sq_range[1]]
            edge_subgraphs.append(edge_subgraph)

        x = rnn.pad_sequence(subgraphs, batch_first=True)
        padded_spatial_subgraphs = rnn.pad_sequence(spatial_subgraphs, batch_first=True)
        padded_edge_subgraphs = rnn.pad_sequence(edge_subgraphs, batch_first=True)
        padded_mask = torch.all(x == 0, dim=-1)

        for layer in self.layers:
            x = layer(x, padded_spatial_subgraphs, padded_edge_subgraphs)

        x = x.flatten(0, 1)[~padded_mask.flatten()]
        # Output for each VNODE for each graph
        vnode_outputs = x[ptr[:-1]]
        out = self.out_lin(vnode_outputs)

        return out.squeeze()
