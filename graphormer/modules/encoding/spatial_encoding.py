import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        b = torch.zeros(self.max_path_distance)
        self.b = nn.Parameter(b)
        self.t1 = nn.Parameter(torch.randn(1))
        self.t2 = nn.Parameter(torch.randn(1))

    def forward(self, data: ModelData) -> ModelData:
        """
        :param paths: pairwise node paths, shape: (batch_size, max_graph_size, max_graph_size, max_path_length)
        :return: torch.Tensor, spatial encoding
        """

        vnode_out_mask = data.node_paths[:, :, :, 0] == 0
        vnode_in_mask = data.node_paths[:, :, :, 1] == 0

        paths_mask = (data.node_paths != -1).to(data.device)
        path_lengths = paths_mask.sum(dim=-1)
        length_mask = path_lengths != 0
        b_idx = torch.minimum(path_lengths, torch.Tensor([self.max_path_distance]).long().to(data.device)) - 1
        spatial_encoding = torch.zeros_like(b_idx, dtype=torch.float)
        spatial_encoding[length_mask] = self.b[b_idx][length_mask]
        # Reset VNODE -> Node encodings
        spatial_encoding[vnode_out_mask] = self.t1
        # Reset Node -> VNODE encodings
        spatial_encoding[vnode_in_mask] = self.t2
        data.spatial_encoding = spatial_encoding

        return data
