from typing import OrderedDict
import torch
from torch import nn

from graphormer.data.dataloader import GraphormerBatch

import warnings
from torch.jit._trace import TracerWarning

from graphormer.modules.model_data import ModelData  # type: ignore

warnings.filterwarnings("ignore", category=TracerWarning)


class Graphormer(nn.Module):
    def __init__(self, modules: OrderedDict[str, nn.Module]):
        """
        :param modules: The list of modules to use
        """

        super().__init__()
        self.module_dict = modules
        self.net = nn.Sequential(modules)
        self.apply(Graphormer._init_weights)

    def enable_dropout(self, dropout_rate: float) -> None:
        """Function to enable the dropout layers during test-time for Monte Carlo dropout."""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
                if dropout_rate is not None:
                    m.p = dropout_rate

    @classmethod
    def _init_weights(cls, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(
        self,
        data: GraphormerBatch,
    ) -> torch.Tensor:
        """
        :param data: The batch data to pass in to the model
        :return: torch.Tensor, output node embeddings
        """
        model_data = ModelData(data.x, data.degrees, data.edge_attr, data.node_paths, data.edge_paths, data.x.device)  # type: ignore
        return self.net(model_data)
