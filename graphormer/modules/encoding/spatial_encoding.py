import torch
import torch.nn as nn


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
        :param x: node embedding, shape: (batch_size, num_nodes, hidden_dim)
        :param paths: pairwise node paths, shape: (batch_size, num_pairwaise_paths, max_path_length)
        :return: torch.Tensor, spatial encoding
        """

        vnode_out_mask = paths[:, :, 0] == 0
        vnode_in_mask = paths[:, :, 1] == 0

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
