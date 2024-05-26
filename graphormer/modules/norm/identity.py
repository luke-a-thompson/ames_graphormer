import torch.nn as nn

from graphormer.modules.model_data import ModelData


class IdentityNorm(nn.Module):
    """
    Placeholder that doesn't do any normalization, just populates the "normalized_input" field to avoid branching in downstream modules
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: ModelData) -> ModelData:
        data.normalized_input = data.x
        return data
