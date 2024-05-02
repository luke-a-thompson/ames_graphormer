from typing import Tuple

import gnn_tools
import torch


def shortest_path_distance(edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    edges = [(x.item() + 1, y.item() + 1) for x, y in zip(edge_index[0], edge_index[1])]

    # Insert VNODE, disconnected to allow physical paths between nodes in SPD
    edges.insert(0, (0, 0))

    node_paths, edge_paths = gnn_tools.shortest_paths(edges, 5)  # type: ignore
    node_paths_tensor = torch.Tensor(node_paths).int()
    edge_paths_tensor = torch.Tensor(edge_paths).int()

    # Connect VNODE node paths
    # Set VNODE paths to [0, node, -1...]
    physical_nodes = torch.arange(1, node_paths_tensor.shape[1]).view(-1, 1)
    zeros = torch.zeros_like(physical_nodes)
    vnode_paths = torch.cat((zeros, physical_nodes), dim=1)
    # VNODE -> Node
    node_paths_tensor[0, 1:, :2] = vnode_paths
    # Node -> VNODE
    node_paths_tensor[1:, 0, :2] = torch.flip(vnode_paths, dims=[1])

    # Connect VNODE edge paths
    # Set VNODE edge paths to [new_edge, -1 ... ]
    extra_edge_idxs = torch.arange(
        torch.max(edge_paths_tensor).item() + 1,
        torch.max(edge_paths_tensor).item() + node_paths_tensor.shape[1],
    ).unsqueeze(-1)
    # VNODE -> Node
    edge_paths_tensor[0, 1:, :1] = extra_edge_idxs
    # Node -> VNODE
    edge_paths_tensor[1:, 0, :1] = extra_edge_idxs

    node_paths_tensor_flat = node_paths_tensor.flatten(0, 1)
    edge_paths_tensor_flat = edge_paths_tensor.flatten(0, 1)

    return node_paths_tensor_flat, edge_paths_tensor_flat
