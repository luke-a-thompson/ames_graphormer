import torch
import torch.nn as nn
from torch_geometric.utils import degree


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
        :param x: node embedding (batch_size, max_subgraph_size, node_embedding_dim)
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[1]
        in_pad_mask = edge_index[:, 1] == -1
        out_pad_mask = edge_index[:, 0] == -1

        in_degree = torch.clamp_max(
            degree(index=edge_index[:, 1][~in_pad_mask], num_nodes=num_nodes).long(),
            self.max_in_degree - 1,
        )
        out_degree = torch.clamp_max(
            degree(index=edge_index[:, 0][~out_pad_mask], num_nodes=num_nodes).long(),
            self.max_out_degree - 1,
        )

        # Exclude adding any centrality info to the VNODE
        x[:, 1:] += self.z_in[in_degree][1:] + self.z_out[out_degree][1:]

        return x
