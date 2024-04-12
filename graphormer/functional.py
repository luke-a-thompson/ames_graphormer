from typing import Dict, List, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def floyd_warshall_source_to_all(G: nx.Graph, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    # paths dictionary  (paths to key from source)
    node_paths = {source: [source]}
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if cutoff is not None and cutoff <= level:
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(
    G: nx.Graph,
) -> Tuple[torch.Tensor, torch.Tensor]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    node_paths = flatten_paths_tensor(node_paths)
    edge_paths = flatten_paths_tensor(edge_paths)
    return node_paths, edge_paths


def shortest_path_distance(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    G: nx.DiGraph = to_networkx(data)
    idxs = [i for i in range(data.ptr[-1])]
    subgraph_idxs = [data.ptr[i : i + 2] for i in range(data.ptr.shape[0] - 1)]
    subgraphs = [G.subgraph(idxs[x[0] : x[1]]) for x in subgraph_idxs]
    sub_node_paths_list = []
    sub_edge_paths_list = []
    for graph in subgraphs:
        sub_node_paths, sub_edge_paths = all_pairs_shortest_path(graph)
        sub_node_paths_list.append(sub_node_paths)
        sub_edge_paths_list.append(sub_edge_paths)

    node_paths = insert_diagonal_tensors(sub_node_paths_list)
    edge_paths = insert_diagonal_tensors(sub_edge_paths_list)
    return node_paths, edge_paths


def flatten_paths_tensor(
    paths: Dict[int, Dict[int, List[int]]], max_path_length: int = 5
) -> torch.Tensor:
    nodes = list(paths.keys())
    num_nodes = len(nodes)

    tensor_paths = torch.full(
        (num_nodes, num_nodes, max_path_length), -1, dtype=torch.int
    )
    if len(nodes) == 0:
        return tensor_paths
    start_idx = nodes[0]
    for src, dsts in paths.items():
        for dst, path in dsts.items():
            path_tensor = torch.tensor(
                path[:max_path_length] + [-1] * (max_path_length - len(path)),
                dtype=torch.int,
            )
            tensor_paths[src - start_idx, dst - start_idx] = path_tensor
    return tensor_paths


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

