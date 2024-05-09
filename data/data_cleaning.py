import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

from graphormer.functional import shortest_path_distance


class HonmaDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Honma_New.xlsx"]

    @property
    def processed_file_names(self):
        return ["honma.pt"]

    def process(self):
        """
        Process the raw data and save the processed data.

        This method cleans the raw data, converts it into a format suitable for training,
        and saves the processed data to a .pt file.

        Returns:
            None
        """
        honma = pd.read_excel(self.raw_paths[0])

        data_list = []

        for smiles, ames in zip(honma["smiles"], honma["ames"]):
            label = torch.tensor([ames], dtype=torch.float)
            if torch.isnan(label):
                print(f"WARN: Entry {smiles} has no label, skipping")
                continue
            data = from_smiles(smiles)
            data.y = label
            node_paths, edge_paths = shortest_path_distance(data.edge_index)
            data.node_paths = node_paths
            data.edge_paths = edge_paths
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class HansenDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Hansen_New.csv"]

    @property
    def processed_file_names(self):
        return ["hansen.pt"]

    def process(self):
        """
        Process the raw data and save the processed data.

        This method cleans the raw data, converts it into a format suitable for training,
        and saves the processed data to a .pt file.

        Returns:
            None
        """
        honma = pd.read_csv(self.raw_paths[0])

        data_list = []

        for smiles, ames in zip(honma["smiles"], honma["ames"]):
            label = torch.tensor([ames], dtype=torch.float)
            if torch.isnan(label):
                print(f"WARN: Entry {smiles} has no label, skipping")
                continue
            data = from_smiles(smiles)
            data.y = label
            node_paths, edge_paths = shortest_path_distance(data.edge_index)
            data.node_paths = node_paths
            data.edge_paths = edge_paths
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    dataset = HonmaDataset("data")
    print(len(dataset))
