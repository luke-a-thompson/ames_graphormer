from typing import Dict, Self
import torch
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss

from graphormer.config.options import LossReductionType


class LossConfig:
    def __init__(self, loss_reduction_type: LossReductionType, device: torch.device):
        self.loss_reduction_type = loss_reduction_type
        self.device = device
        self.pos_weight = None
        self.state_dict = None

    def with_pos_weight(self, pos_weight: torch.Tensor) -> Self:
        self.pos_weight = pos_weight
        return self

    def with_state_dict(self, state_dict: Dict) -> Self:
        self.state_dict = state_dict
        return self

    def build(self) -> _Loss:
        if self.pos_weight is None:
            raise AttributeError("train_dataloader is not defined for BCEWithLogitsLoss")

        loss_params = {
            "reduction": self.loss_reduction_type.value,
            "pos_weight": self.pos_weight,
        }
        loss = BCEWithLogitsLoss(**loss_params).to(self.device)
        if self.state_dict is not None:
            loss.load_state_dict(self.state_dict)
            self.state_dict = None
        return loss
