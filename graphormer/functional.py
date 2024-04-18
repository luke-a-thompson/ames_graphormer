from typing import Tuple

import gnn_tools
import torch
from torch_geometric.data import Data


def shortest_path_distance(data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idxs = torch.Tensor([i for i in range(data.ptr[-1])]).long().to("cuda")

    # I.e. ptr = [0, 9, 18, 27, 36] -> [[0, 9], [9, 18], [18, 27], [27, 36]]
    subgraph_idxs = [data.ptr[i : i + 2] for i in range(data.ptr.shape[0] - 1)]
    subgraphs = [data.subgraph(idxs[x[0] : x[1]]) for x in subgraph_idxs]
    sub_node_paths_list = []
    sub_edge_paths_list = []
    for graph in subgraphs:
        assert graph.edge_index is not None
        edges = [(x.item() + 1, y.item() + 1) for x, y in zip(graph.edge_index[0], graph.edge_index[1])]
        if len(edges) == 0:
            continue

        # Insert VNODE, disconnected to allow physical paths between nodes in SPD
        edges.insert(0, (0, 0))

        sub_node_paths, sub_edge_paths = gnn_tools.shortest_paths(edges, 5)
        sub_node_paths_tensor = torch.Tensor(sub_node_paths).int()
        sub_edge_paths_tensor = torch.Tensor(sub_edge_paths).int()

        # Connect VNODE node paths
        # Set VNODE paths to [0, node, -1...]
        physical_nodes = torch.arange(1, sub_node_paths_tensor.shape[1]).view(-1, 1)
        zeros = torch.zeros_like(physical_nodes)
        vnode_paths = torch.cat((zeros, physical_nodes), dim=1)
        sub_node_paths_tensor[0, 1:, :2] = vnode_paths
        sub_node_paths_tensor[1:, 0, :2] = torch.flip(vnode_paths, dims=[1])

        # Connect VNODE edge paths
        # Set VNODE edge paths to [new_edge, -1 ... ]
        extra_edge_idxs = torch.arange(
            torch.max(sub_edge_paths_tensor).item() + 1,
            torch.max(sub_edge_paths_tensor).item() + sub_node_paths_tensor.shape[1],
        ).unsqueeze(-1)
        sub_edge_paths_tensor[0, 1:, :1] = extra_edge_idxs
        sub_edge_paths_tensor[1:, 0, :1] = extra_edge_idxs

        sub_node_paths_list.append(sub_node_paths_tensor.flatten(0, 1))
        sub_edge_paths_list.append(sub_edge_paths_tensor.flatten(0, 1))

    # TODO:Need to return the indices of the VNODEs for spatial encoding

    # shape (sum(g.num_nodes ** 2 for g in graphs), max_path_len)
    node_paths = torch.cat(sub_node_paths_list)
    edge_paths = torch.cat(sub_edge_paths_list)
    return node_paths, edge_paths
