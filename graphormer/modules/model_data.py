from typing import Optional
import torch


class ModelData:
    """
    A mutable version of `GraphormerBatch` with additional fields that may be needed for graphormer modules to communicate
    """

    def __init__(
        self,
        x: torch.Tensor,
        degrees: torch.Tensor,
        edge_attr: torch.Tensor,
        node_paths: torch.Tensor,
        edge_paths: torch.Tensor,
        device: torch.device,
    ):
        assert isinstance(x, torch.Tensor), "x must be a Tensor"
        assert isinstance(degrees, torch.Tensor), "degrees must be a Tensor"
        assert isinstance(edge_attr, torch.Tensor), "edge_attr must be a Tensor"
        assert isinstance(node_paths, torch.Tensor), "node_paths must be a Tensor"
        assert isinstance(edge_paths, torch.Tensor), "node_paths must be a Tensor"

        self.x = x
        self.degrees = degrees
        self.edge_attr = edge_attr
        self.node_paths = node_paths
        self.edge_paths = edge_paths
        self.device = device
        self.x_pad_mask: Optional[torch.Tensor] = None
        self.edge_embedding: Optional[torch.Tensor] = None
        self.spatial_encoding: Optional[torch.Tensor] = None
        self.edge_encoding: Optional[torch.Tensor] = None
        self.attention_prior: Optional[torch.Tensor] = None
        self.normalized_input: Optional[torch.Tensor] = None
        self.attention_output: Optional[torch.Tensor] = None
        self.ffn_output: Optional[torch.Tensor] = None
