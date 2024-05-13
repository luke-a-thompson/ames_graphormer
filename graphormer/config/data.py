from typing import Self, Tuple
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from graphormer.config.options import DatasetType
from torch_geometric.loader import DataLoader


class DataConfig:
    def __init__(self, dataset_type: DatasetType, batch_size: int, data_dir: str, max_path_distance: int):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.max_path_distance = max_path_distance
        self.test_size = None
        self.random_state = None
        self.num_node_features = None
        self.num_edge_features = None
        self.num_classes = None

    def with_random_state(self, random_state: int) -> Self:
        self.random_state = random_state
        return self

    def with_test_size(self, test_size: float) -> Self:
        self.test_size = test_size
        return self

    def build(self) -> None:
        dataloader_optimization_params = {
            "pin_memory": True,
            "num_workers": 4,
            "prefetch_factor": 4,
            "persistent_workers": True,
        }
        match self.dataset_type:
            case DatasetType.HANSEN:
                if self.test_size is None:
                    raise AttributeError("test_size is not defined for HansenDataset")
                if self.random_state is None:
                    raise AttributeError("random_state is not defined for HansenDataset")
                from data.data_cleaning import HansenDataset

                dataset = HansenDataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes
                test_ids, train_ids = train_test_split(
                    range(len(dataset)), test_size=self.test_size, random_state=self.random_state
                )
                self.train_loader = DataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                self.test_loader = DataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )
            case DatasetType.HONMA:
                from data.data_cleaning import HonmaDataset

                dataset = HonmaDataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes
                self.train_loader = DataLoader(dataset[:12140], batch_size=self.batch_size, shuffle=True, **dataloader_optimization_params)  # type: ignore
                self.test_loader = DataLoader(dataset[12140:], batch_size=self.batch_size, **dataloader_optimization_params)  # type: ignore

    def for_training(self) -> Tuple[DataLoader, DataLoader]:
        if not self.train_loader:
            raise ValueError("Train loader not initialized. Call build method first.")
        if not self.test_loader:
            raise ValueError("Test loader not initialized. Call build method first.")
        return self.train_loader, self.test_loader

    def for_inference(self) -> DataLoader:
        if not self.test_loader:
            raise ValueError("Test loader not initialized. Call build method first.")
        return self.test_loader

    def str(self) -> str:
        return f"{self.dataset_type} dataset"
