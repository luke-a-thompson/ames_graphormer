import torch

from graphormer.data.datasets.graphormer_dataset import GraphormerDataset


class Tox24Dataset(GraphormerDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["tox24_all.csv"]

    @property
    def processed_file_names(self):
        return ["tox24_all.pt"]
