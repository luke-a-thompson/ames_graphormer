from pathlib import Path
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_smiles
import torch


def clean_hansen(hansen_dataset: Path | pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Hansen dataset by removing unnecessary columns and renaming the remaining columns.

    Parameters:
    - hansen_dataset: Path or pd.DataFrame
        The path to the Hansen dataset CSV file or a DataFrame containing the dataset.

    Returns:
    - pd.DataFrame
        The cleaned Hansen dataset with columns 'smiles' and 'ames'.
    """
    hansen = pd.read_csv(hansen_dataset)

    hansen = hansen.drop(hansen.columns[1], axis=1)
    hansen.columns = ["smiles", "ames"]

    return hansen


def clean_honma(honma_dataset: Path | pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Honma dataset by reading an Excel file, selecting specific columns,
    renaming columns, and applying a transformation to the 'ames' column.

    Args:
        honma_dataset (Path or pd.DataFrame): The path to the Honma dataset Excel file
            or a DataFrame containing the dataset.

    Returns:
        pd.DataFrame: The cleaned Honma dataset with columns 'smiles' and 'ames'.

    """
    honma = pd.read_excel(honma_dataset, sheet_name=0, skiprows=[1, 2], usecols="D,G")

    honma = honma[[honma.columns[1], honma.columns[0]]]
    honma.columns = ["smiles", "ames"]
    honma["ames"] = honma["ames"].apply(lambda x: 1 if x == "A" or x == "B" else 0)

    return honma


class AmesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Honma_Ames.xlsx"]

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
        honma = clean_honma(self.raw_paths[0])  # Clean the raw_file_names, idx[0]

        data_list = []

        for smiles, ames in zip(honma["smiles"], honma["ames"]):
            data = from_smiles(smiles)
            data.y = torch.tensor([ames], dtype=torch.float)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    dataset = AmesDataset("data")
    print(len(dataset))
