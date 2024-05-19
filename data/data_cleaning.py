import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

from graphormer.functional import shortest_path_distance
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm


def check_smiles_and_label(smiles, label):
    if torch.isnan(label):
        return f"WARN: No label for {smiles}, skipped"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"WARN: Invalid SMILES {smiles}, skipped"

    return None

class HonmaDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
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
        warnings = []

        for smiles, ames in tqdm(zip(honma["smiles"], honma["ames"]), total=len(honma), desc="Processing"):
            label = torch.tensor([ames], dtype=torch.float)

            warning = check_smiles_and_label(smiles, label)
            if warning:
                warnings.append(warning)
                continue

            data = from_smiles(smiles)
            data.y = label
            node_paths, edge_paths = shortest_path_distance(data.edge_index, self.max_distance)
            data.node_paths = node_paths
            data.edge_paths = edge_paths
            data.graph_feats = torch.tensor(QED.properties(Chem.MolFromSmiles(smiles)))
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        # Print all warnings at the end
        for warning in warnings:
            print(warning)


class HansenDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
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
        warnings = []

        for smiles, ames in tqdm(zip(honma["smiles"], honma["ames"]), total=len(honma), desc="Processing Hansen Dataset"):
            label = torch.tensor([ames], dtype=torch.float)

            warning = check_smiles_and_label(smiles, label)
            if warning:
                warnings.append(warning)
                continue

            data = from_smiles(smiles)
            data.y = label
            node_paths, edge_paths = shortest_path_distance(data.edge_index, self.max_distance)
            data.node_paths = node_paths
            data.edge_paths = edge_paths
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        # Print all warnings at the end
        for warning in warnings:
            print(warning)

class CombinedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Combined.xlsx"]

    @property
    def processed_file_names(self):
        return ["combined.pt"]

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
        warnings = []

        for smiles, ames in tqdm(zip(honma["smiles"], honma["ames"]), total=len(honma), desc="Processing Combined Dataset"):
            label = torch.tensor([ames], dtype=torch.float)

            warning = check_smiles_and_label(smiles, label)
            if warning:
                warnings.append(warning)
                continue

            data = from_smiles(smiles)
            data.y = label
            node_paths, edge_paths = shortest_path_distance(data.edge_index, self.max_distance)
            data.node_paths = node_paths
            data.edge_paths = edge_paths
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        # Print all warnings at the end
        for warning in warnings:
            print(warning)


if __name__ == "__main__":
    dataset = HonmaDataset("data")
    dataset = HansenDataset("data")
    dataset = CombinedDataset("data")
    print("Datasets built")
