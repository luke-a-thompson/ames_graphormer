import torch
from ogb.graphproppred import PygGraphPropPredDataset
from graphormer.data.datasets.graphormer_dataset import process
from tqdm import tqdm
import numpy as np
import os.path as osp
from ogb.io.read_graph_pyg import read_graph_pyg
import pandas as pd


class OGBDataset(PygGraphPropPredDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
        super().__init__("ogbg-molpcba", root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["ogb-molpcba.pt"]

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info["add_inverse_edge"] == "True"

        if self.meta_info["additional node files"] == "None":
            additional_node_files = []
        else:
            additional_node_files = self.meta_info["additional node files"].split(",")

        if self.meta_info["additional edge files"] == "None":
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info["additional edge files"].split(",")

        data_list = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=add_inverse_edge,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files,
            binary=self.binary,
        )

        if self.task_type == "subtoken prediction":
            graph_label_notparsed = pd.read_csv(
                osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip", header=None
            ).values
            graph_label = [str(graph_label_notparsed[i][0]).split(" ") for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, "graph-label.npz"))["graph_label"]
            else:
                graph_label = pd.read_csv(
                    osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip", header=None
                ).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if "classification" in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        processed_data_list = []

        for g in tqdm(data_list, total=len(data_list), desc="Processing OGB Dataset"):
            try:
                g = process(g, self.max_distance)
                processed_data_list.append(g)
            except Exception as e:
                print(f"{e} occurred when processing {g}")

            # Might need to fix split index if many rows are dropped

        data, slices = self.collate(processed_data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
