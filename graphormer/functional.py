from typing import Tuple

import gnn_tools
import torch
from torch_geometric.data import Data


def shortest_path_distance(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    idxs = torch.Tensor([i for i in range(data.ptr[-1])]).long().to("cuda")

    # I.e. ptr = [0, 9, 18, 27, 36] -> [[0, 9], [9, 18], [18, 27], [27, 36]]
    subgraph_idxs = [data.ptr[i : i + 2] for i in range(data.ptr.shape[0] - 1)]
    subgraphs = [data.subgraph(idxs[x[0] : x[1]]) for x in subgraph_idxs]
    sub_node_paths_list = []
    sub_edge_paths_list = []
    for graph in subgraphs:
        assert graph.edge_index is not None
        edges = [
            (x.item(), y.item())
            for x, y in zip(graph.edge_index[0], graph.edge_index[1])
        ]
        if len(edges) == 0:
            continue

        sub_node_paths, sub_edge_paths = gnn_tools.shortest_paths(edges, 5)
        sub_node_paths_list.append(torch.Tensor(sub_node_paths).int().flatten(0, 1))
        sub_edge_paths_list.append(torch.Tensor(sub_edge_paths).int().flatten(0, 1))

    # shape (sum(g.num_nodes ** 2 for g in graphs), max_path_len)
    node_paths = torch.cat(sub_node_paths_list)
    edge_paths = torch.cat(sub_edge_paths_list)
    return node_paths, edge_paths
