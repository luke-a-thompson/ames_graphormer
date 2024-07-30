from typing import Optional

import torch
from rdkit import Chem
from torch_geometric.utils import degree

from graphormer.functional import shortest_path_distance


def check_smiles_and_label(smiles, label) -> Optional[str]:
    if torch.isnan(label):
        return f"WARN: No label for {smiles}, skipped"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"WARN: Invalid SMILES {smiles}, skipped"

    return None


def process(data, max_distance: int):
    data.x = torch.cat((torch.ones(1, data.x.shape[1]) * -1, data.x), dim=0)
    new_idxs = torch.stack((torch.zeros(data.x.shape[0]), torch.arange(0, data.x.shape[0])), dim=0).transpose(0, 1)

    # Check if there are any isolated nodes not represented in the edge index
    for node_index in range(0, data.x.shape[0]):
        if node_index not in data.edge_index[0] and node_index not in data.edge_index[1]:
            data.edge_index = torch.cat((data.edge_index, torch.tensor([[node_index], [node_index]])), dim=1)

    node_paths, edge_paths, extra_edge_idxs = shortest_path_distance(data.edge_index, max_distance)
    original_edge_index = data.edge_index
    data.edge_index = torch.cat((new_idxs, data.edge_index.transpose(0, 1)), dim=0).transpose(0, 1).long()
    data.degrees = torch.stack(
        [degree(data.edge_index[:, 1], data.x.shape[0]), degree(data.edge_index[:, 0], data.x.shape[0])],
    ).transpose(0, 1)
    data.node_paths = node_paths.flatten(0, 1)
    data.edge_paths = edge_paths.flatten(0, 1)
    data.edge_attr = torch.cat(
        (
            torch.ones(1, data.edge_attr.shape[1]) * -1,
            data.edge_attr,
            torch.ones(extra_edge_idxs.shape[0], data.edge_attr.shape[1]) * -1,
        ),
        dim=0,
    )

    assert len(data.node_paths) == len(data.edge_paths)
    assert (
        node_paths.shape[0] - 1 == data.x.shape[0]
    ), f"{node_paths.shape[0]}, {data.x.shape[0]}, {original_edge_index}"
    assert data.degrees.shape[0] == data.x.shape[0]

    return data
