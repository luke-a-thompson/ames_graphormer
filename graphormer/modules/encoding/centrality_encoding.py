import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


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
        # Prior that more central nodes hold more important than less central nodes
        self.z_in = torch.nn.Parameter(torch.zeros(self.max_in_degree + 1, self.hidden_dim))
        self.z_out = torch.nn.Parameter(torch.zeros(self.max_out_degree + 1, self.hidden_dim))

    def forward(self, data: ModelData) -> ModelData:
        """
        :param degrees: degrees of graph (batch_size, 2, num_nodes)
        :return: torch.Tensor, centrality encoding (batch_size, num_nodes, hidden_dim)
        """
        centrality_encoding = torch.zeros(data.degrees.shape[0], data.degrees.shape[2], self.hidden_dim).to(data.device)
        pad_mask = (data.degrees == -1)[:, 0]

        in_degree = torch.clamp_max(
            data.degrees[:, 1],
            self.max_in_degree,
        )
        out_degree = torch.clamp_max(
            data.degrees[:, 0],
            self.max_out_degree,
        )

        z_in = self.z_in[in_degree[~pad_mask]]
        z_out = self.z_out[out_degree[~pad_mask]]

        centrality_encoding[~pad_mask] = z_in + z_out
        data.x += centrality_encoding

        return data
