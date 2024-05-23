import torch
from torch_geometric.utils import from_smiles, degree
from graphormer.functional import shortest_path_distance
from rdkit import Chem


def check_smiles_and_label(smiles, label):
    if torch.isnan(label):
        return f"WARN: No label for {smiles}, skipped"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"WARN: Invalid SMILES {smiles}, skipped"

    return None


def process(smiles, label, max_distance):
    data = from_smiles(smiles)
    data.y = label
    node_paths, edge_paths, extra_edge_idxs = shortest_path_distance(data.edge_index, max_distance)

    data.x = torch.cat((torch.ones(1, data.x.shape[1]) * -1, data.x), dim=0)
    new_idxs = torch.stack((torch.zeros(data.x.shape[0]), torch.arange(0, data.x.shape[0])), dim=0).transpose(0, 1)

    data.edge_index = torch.cat((new_idxs, data.edge_index.transpose(0, 1)), dim=0).transpose(0, 1).long()
    data.degrees = torch.stack(
        [degree(data.edge_index[:, 1], data.x.shape[0]), degree(data.edge_index[:, 0], data.x.shape[0])],
    ).transpose(0, 1)
    data.node_paths = node_paths
    data.edge_paths = edge_paths
    data.edge_attr = torch.cat(
        (
            torch.ones(1, data.edge_attr.shape[1]) * -1,
            data.edge_attr,
            torch.ones(extra_edge_idxs.shape[0], data.edge_attr.shape[1]) * -1,
        ),
        dim=0,
    )

    assert data.degrees.shape[0] == data.x.shape[0]
    assert data.edge_attr.shape[0] - 1 == torch.max(
        data.edge_paths
    ), f"Missing edge attrs for graph!  edge_attr.shape: {data.edge_attr.shape}, max_edge_index: {torch.max(data.edge_paths)}"
    return data
