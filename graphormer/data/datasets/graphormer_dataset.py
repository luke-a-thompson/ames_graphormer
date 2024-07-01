import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
from tqdm import tqdm
import os

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
        file_path = self.raw_paths[0]
        file_extension = os.path.splitext(file_path)[1]

        if file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

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

            data = from_smiles(smiles)
            data.y = label
            data = process(data, self.max_distance)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        # Print all warnings at the end
        for warning in warnings:
            print(warning)
