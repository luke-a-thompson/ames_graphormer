from typing import Dict, Self, Tuple
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, MSELoss

from graphormer.config.options import LossFunction, LossReductionType


class LossConfig:
    def __init__(
        self,
        loss_function: LossFunction | tuple[LossFunction, ...],
        loss_reduction_type: LossReductionType,
        device: torch.device,
    ):
        self.loss_function = loss_function
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

    def with_weights(self, weights: float | Tuple[float, ...]) -> Self:
        self.weights = weights
        return self

    def build(self) -> _Loss:
        if self.loss_function is None:
            raise AttributeError("loss_function is not defined for LossConfig")
        if (
            isinstance(self.loss_function, LossFunction)
            and self.loss_function == LossFunction.BCE_LOGITS
            and self.pos_weight is None
        ):
            raise AttributeError("pos_weight is not defined for BCEWithLogitsLoss")

        if isinstance(self.loss_function, tuple):
            if not isinstance(self.weights, tuple) or len(self.weights) != len(self.loss_function):
                raise ValueError("Weights must be provided for all loss functions")
            losses = [self._build_single_loss(lf) for lf in self.loss_function]
            return self._combine_losses(losses)
        else:
            return self._build_single_loss(self.loss_function)

    def _build_single_loss(self, loss_fn: LossFunction) -> _Loss:
        loss_params = {
            "reduction": self.loss_reduction_type.value,
        }
        if loss_fn == LossFunction.BCE_LOGITS and self.pos_weight is not None:
            loss_params["pos_weight"] = self.pos_weight

        match loss_fn:
            case LossFunction.BCE_LOGITS:
                loss = BCEWithLogitsLoss(**loss_params)
            case LossFunction.MSE:
                loss = MSELoss(**loss_params)
            case _:
                raise ValueError(f"Unsupported loss function: {loss_fn}")

        loss = loss.to(self.device)

        if self.state_dict is not None:
            loss.load_state_dict(self.state_dict)
            self.state_dict = None

        return loss

    def _combine_losses(self, losses: list[_Loss]) -> _Loss:
        class CombinedLoss(nn.Module):
            def __init__(self, losses, weights):
                super().__init__()
                self.losses = nn.ModuleList(losses)
                self.weights = weights

            def forward(self, *args, **kwargs):
                return sum(w * loss(*args, **kwargs) for loss, w in zip(self.losses, self.weights))

        return CombinedLoss(losses, self.weights)
