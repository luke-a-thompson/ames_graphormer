from typing import List, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def floyd_warshall_source_to_all(
    G: nx.Graph, source: int, max_path_length: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")

    num_nodes = G.number_of_nodes()

    # Allocate one extra to get accurate edge information up to the cap
    node_paths = torch.full((num_nodes, max_path_length + 1), -1)
    edge_paths = torch.full((num_nodes, max_path_length + 1), -1)

    if num_nodes == 0:
        return node_paths[:, :max_path_length], edge_paths[:, :max_path_length]

    start_idx = list(G.nodes())[0]
    node_paths[source - start_idx][0] = source
    edges = {edge: i for i, edge in enumerate(G.edges())}
    visited_nodes = set([source])

    node_queue = [source]

    while node_queue:
        # FIFO queue
        v = node_queue.pop(0)
        for w in G[v]:
            if w in visited_nodes:
                continue
            visited_nodes.add(w)

            # reindex start to 0 as subgraphs from megagraph retain original node indices (e.g. 9 if its the 2nd graph)
            v_row_idx = v - start_idx
            w_row_idx = w - start_idx
            node_paths[w_row_idx] = node_paths[v_row_idx]
            edge_paths[w_row_idx] = edge_paths[v_row_idx]
            # Where the first "free" slot in the path at node w is
            w_node_col_idx = (node_paths[w_row_idx]
                              == -1).nonzero(as_tuple=True)[0]
            # Where the first "free" slot in the path at edge w is
            w_edge_col_idx = (edge_paths[w_row_idx]
                              == -1).nonzero(as_tuple=True)[0]
            # If there is space in node paths list
            if len(w_node_col_idx) > 0:
                node_paths[w_row_idx, w_node_col_idx[0]] = w
                # If there is space in the edge paths list
                if len(w_edge_col_idx) > 0:
                    edge_nodes = node_paths[w_row_idx][
                        w_node_col_idx[0] - 1: w_node_col_idx[0] + 1
                    ]
                    edge = edges[tuple(edge_nodes.tolist())]
                    edge_paths[w_row_idx, w_edge_col_idx[0]] = edge
            node_queue.append(w)

    # Chop off extra column
    edge_paths = edge_paths[:, :max_path_length]
    node_paths = node_paths[:, :max_path_length]

    return node_paths, edge_paths


def all_pairs_shortest_path(
    G: nx.Graph, max_path_length: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = G.number_of_nodes()
    node_paths = torch.full((num_nodes, num_nodes, max_path_length), -1)
    edge_paths = torch.full((num_nodes, num_nodes, max_path_length), -1)
    src_index = None
    for src in G:
        if src_index is None:
            src_index = src
            n, e = floyd_warshall_source_to_all(G, src)
            node_paths[src - src_index] = n
            edge_paths[src - src_index] = e
    return node_paths, edge_paths


def shortest_path_distance(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    G: nx.DiGraph = to_networkx(data)
    idxs: List[int] = [i for i in range(data.ptr[-1])]
    subgraph_idxs: List[int, int] = [data.ptr[i: i + 2] for i in range(data.ptr.shape[0] - 1)]  # I.e. ptr = [0, 9, 18, 27, 36] -> [[0, 9], [9, 18], [18, 27], [27, 36]]
    subgraphs: torch.Tensor = [G.subgraph(idxs[x[0]: x[1]]) for x in subgraph_idxs]
    sub_node_paths_list = []
    sub_edge_paths_list = []
    for graph in subgraphs:
        sub_node_paths, sub_edge_paths = all_pairs_shortest_path(graph)
        sub_node_paths_list.append(sub_node_paths)
        sub_edge_paths_list.append(sub_edge_paths)
    node_paths = insert_diagonal_tensors(sub_node_paths_list)
    edge_paths = insert_diagonal_tensors(sub_edge_paths_list)
    return node_paths, edge_paths


def insert_diagonal_tensors(tensor_list: List[torch.Tensor]):
    if len(tensor_list) == 1:
        return tensor_list[0]

    height, width, depth = 0, 0, tensor_list[0].size(2)
    for tensor in tensor_list:
        height += tensor.size(0)
        width += tensor.size(1)

    result = torch.full((height, width, depth), -1, dtype=tensor_list[0].dtype)
    current_row, current_col = 0, 0
    for tensor in tensor_list:
        end_row = current_row + tensor.size(0)
        end_col = current_col + tensor.size(1)
        result[current_row:end_row, current_col:end_col, :] = tensor
        current_row = end_row
        current_col = end_col

    return result

