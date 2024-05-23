import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from graphormer.data.data_cleaning import check_smiles_and_label, process


class GraphormerDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raise NotImplementedError()

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def process(self):
        """
        Process the raw data and save the processed data.

        This method cleans the raw data, converts it into a format suitable for training,
        and saves the processed data to a .pt file.

        Returns:
            None
        """
        df = pd.read_excel(self.raw_paths[0])

        data_list = []
        warnings = []

        for smiles, ames in tqdm(
            zip(df["smiles"], df["ames"]), total=len(df), desc="Processing dataset", unit="SMILES"
        ):
            label = torch.tensor([ames], dtype=torch.float)

            warning = check_smiles_and_label(smiles, label)
            if warning:
                warnings.append(warning)
                continue

            data = process(smiles, label, self.max_distance)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        # Print all warnings at the end
        for warning in warnings:
            print(warning)
