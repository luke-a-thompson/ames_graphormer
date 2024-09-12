from typing import Self, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from graphormer.config.options import DatasetRegime, DatasetType
from graphormer.data.dataloader import GraphormerDataLoader


class DataConfig:
    def __init__(
        self,
        dataset_type: DatasetType,
        dataset_regime: DatasetRegime,
        batch_size: int,
        data_dir: str,
        max_path_distance: int,
    ):
        self.dataset_type = dataset_type
        self.dataset_regime = dataset_regime
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.max_path_distance = max_path_distance
        self.test_size = None
        self.random_state = None
        self.num_node_features = None
        self.num_edge_features = None
        self.num_classes = None
        self.num_workers = 4
        self.prefetch_factor = 4

    def with_random_state(self, random_state: int) -> Self:
        self.random_state = random_state
        return self

    def with_test_size(self, test_size: float) -> Self:
        self.test_size = test_size
        return self

    def with_num_workers(self, num_workers: int) -> Self:
        self.num_workers = num_workers
        return self

    def with_prefetch_factor(self, prefect_factor: int) -> Self:
        self.prefetch_factor = prefect_factor
        return self

    def build(self) -> Tuple[GraphormerDataLoader, GraphormerDataLoader] | GraphormerDataLoader:
        """
        Builds the dataloaders for the dataset.

        Returns:
            IF TEST:
                A single test dataloader.
            ELSE:
                A tuple of train and validation dataloaders.
        """
        dataloader_optimization_params = {
            "pin_memory": True,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": True,
        }
        match self.dataset_type:
            case DatasetType.HANSEN:
                if self.test_size is None:
                    raise AttributeError("test_size is not defined for HansenDataset")
                if self.random_state is None:
                    raise AttributeError("random_state is not defined for HansenDataset")
                from graphormer.data.datasets import HansenDataset

                dataset = HansenDataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                if self.dataset_regime == DatasetRegime.TEST:
                    test_loader = GraphormerDataLoader(
                        dataset[6513:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:6513])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

            case DatasetType.HONMA:
                from graphormer.data.datasets import HonmaDataset

                dataset = HonmaDataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                # len(Honma) = 13730
                if self.dataset_regime == DatasetRegime.TEST:
                    test_loader = GraphormerDataLoader(
                        dataset[12140:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:12140])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

            case DatasetType.COMBINED:
                from graphormer.data.datasets import CombinedDataset

                dataset = CombinedDataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                # len(Combined) = 20242
                if self.dataset_regime == DatasetRegime.TEST:
                    test_loader = GraphormerDataLoader(
                        dataset[18652:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:18652])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

            case DatasetType.OGBG_MOLPCBA:
                from graphormer.data.datasets import OGBDataset

                dataset = OGBDataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features

                split_idx = dataset.get_idx_split()

                train_loader = GraphormerDataLoader(
                    dataset[split_idx["train"]],  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                valid_loader = GraphormerDataLoader(
                    dataset[split_idx["valid"]],  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, valid_loader

            case DatasetType.TOX24:
                from graphormer.data.datasets import Tox24Dataset

                dataset = Tox24Dataset(self.data_dir, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                # len(Tox24) = 1513, no labels 1014:
                if self.dataset_regime == DatasetRegime.TEST:
                    test_loader = GraphormerDataLoader(
                        dataset[1014:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:1013])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

    def __str__(self) -> str:
        return f"{self.dataset_type} dataset"
