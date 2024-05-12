from typing import Dict, Self, Tuple
from graphormer.config.options import OptimizerType, LossReductionType
from torch.optim import SGD, AdamW, Optimizer
import torch.nn as nn


class OptimizerConfig:
    def __init__(
        self,
        optimizer_type: OptimizerType,
        loss_reduction_type: LossReductionType,
        accumulation_steps: int,
        batch_size: int,
        lr: float,
    ):
        self.loss_reduction_type = loss_reduction_type
        self.optimizer_type = optimizer_type
        self.accumulation_steps = accumulation_steps
        self.batch_size = batch_size
        self.effective_batch_size = accumulation_steps * batch_size
        self.lr = lr
        self.effective_lr = (
            self.lr / self.accumulation_steps
            if self.loss_reduction_type == LossReductionType.MEAN
            else self.lr / self.effective_batch_size
        )

        self.betas = None
        self.weight_decay = None
        self.eps = None
        self.momentum = None
        self.nesterov = None
        self.dampening = None
        self.state_dict = None

    def with_betas(self, betas: Tuple[float, float]) -> Self:
        self.betas = betas
        return self

    def with_weight_decay(self, weight_decay: float) -> Self:
        self.weight_decay = weight_decay
        return self

    def with_eps(self, eps: float) -> Self:
        self.eps = eps
        return self

    def with_momentum(self, momentum: float) -> Self:
        self.momentum = momentum
        return self

    def with_nesterov(self, nesterov: bool) -> Self:
        self.nesterov = nesterov
        return self

    def with_dampening(self, dampening: float) -> Self:
        self.dampening = dampening
        return self

    def with_state_dict(self, state_dict: Dict) -> Self:
        self.state_dict = state_dict
        return self

    def build(self, model: nn.Module) -> Optimizer:
        match self.optimizer_type:
            case OptimizerType.ADAMW:
                if self.betas is None:
                    raise AttributeError("betas are not defined for AdamW optimizer")
                if self.eps is None:
                    raise AttributeError("eps is not defined for AdamW optimizer")
                if self.weight_decay is None:
                    raise AttributeError("weight_decay is not defined for AdamW optimizer")

                adam_params = {
                    "lr": self.effective_lr,
                    "betas": self.betas,
                    "eps": self.eps,
                    "weight_decay": self.weight_decay,
                }
                optimizer = AdamW(model.parameters(), **adam_params)
                if self.state_dict is not None:
                    optimizer.load_state_dict(self.state_dict)
                    self.state_dict = None
                return optimizer
            case OptimizerType.SGD:
                if self.momentum is None:
                    raise AttributeError("momentum is not defined for SGD optimizer")
                if self.nesterov is None:
                    raise AttributeError("nesterov is not defined for SGD optimizer")
                if self.weight_decay is None:
                    raise AttributeError("weight_decay is not defined for SGD optimizer")
                if self.dampening is None:
                    raise AttributeError("dampening is not defined for SGD optimizer")
                sgd_params = {
                    "lr": self.effective_lr,
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                    "dampening": self.dampening,
                    "weight_decay": self.weight_decay,
                }
                optimizer = SGD(model.parameters(), **sgd_params)
                if self.state_dict is not None:
                    optimizer.load_state_dict(self.state_dict)
                    self.state_dict = None
                return optimizer
