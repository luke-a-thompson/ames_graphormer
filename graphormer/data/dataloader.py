from typing import Any, List, Union, Sequence, Optional, override
import torch
import torch.nn.utils.rnn as rnn
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import BaseData, Collater, DataLoader, Dataset, DatasetAdapter


class GraphormerDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )
        self.collate_fn = GraphormerCollater(dataset, follow_batch, exclude_keys)


class GraphormerBatch(Data):
    def __init__(self, *args, **kwargs):
        self.node_paths: Optional[torch.Tensor] = None
        self.edge_paths: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)


class GraphormerCollater(Collater):
    @override
    def __call__(self, batch: List[Any]) -> GraphormerBatch:
        data: GraphormerBatch = super().__call__(batch)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None
        assert data.node_paths is not None
        assert data.edge_paths is not None
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        node_paths = data.node_paths
        edge_paths = data.edge_paths
        ptr = data.ptr

        node_subgraphs = []
        edge_index_subgraphs = []
        edge_attr_subgraphs = []
        node_paths_subgraphs = []
        edge_paths_subgraphs = []
        subgraph_idxs = torch.stack([ptr[:-1], ptr[1:]], dim=1)
        subgraph_sq_ptr = torch.cat(
            [torch.tensor([0]).to(x.device), (subgraph_idxs[:, 1] - subgraph_idxs[:, 0]).square().cumsum(dim=0)]
        )
        subgraph_idxs_sq = torch.stack([subgraph_sq_ptr[:-1], subgraph_sq_ptr[1:]], dim=1)
        for idx_range, idx_range_sq in zip(subgraph_idxs.tolist(), subgraph_idxs_sq.tolist()):
            subgraph = x[idx_range[0] : idx_range[1]]
            node_subgraphs.append(subgraph)
            start_edge_index = (edge_index[0] < idx_range[0]).sum()
            stop_edge_index = (edge_index[0] < idx_range[1]).sum()
            start_node_index = idx_range[0]
            edge_index_subgraphs.append(
                (edge_index[:, start_edge_index:stop_edge_index] - start_node_index).transpose(0, 1)
            )
            edge_attr_subgraphs.append(edge_attr[start_edge_index:stop_edge_index, :])
            node_paths_subgraphs.append(node_paths[idx_range_sq[0] : idx_range_sq[1]])
            edge_paths_subgraphs.append(edge_paths[idx_range_sq[0] : idx_range_sq[1]])

        data.x = rnn.pad_sequence(node_subgraphs, batch_first=True)
        data.edge_index = (
            rnn.pad_sequence(edge_index_subgraphs, batch_first=True, padding_value=-1).transpose(1, 2).long()
        )
        data.edge_attr = rnn.pad_sequence(edge_attr_subgraphs, batch_first=True, padding_value=-1)
        data.node_paths = rnn.pad_sequence(node_paths_subgraphs, batch_first=True, padding_value=-1)
        data.edge_paths = rnn.pad_sequence(edge_paths_subgraphs, batch_first=True, padding_value=-1)
        return data
