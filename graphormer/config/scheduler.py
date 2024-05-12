from typing import Dict, Self
from graphormer.config.options import SchedulerType, LossReductionType
from torch.optim.lr_scheduler import LRScheduler, PolynomialLR, ReduceLROnPlateau
from torch.optim import Optimizer
from graphormer.schedulers import GreedyLR


class SchedulerConfig:
    def __init__(self, scheduler_type: SchedulerType, accumulation_steps: int, batch_size: int):
        self.scheduler_type = scheduler_type
        self.accumulation_steps = accumulation_steps
        self.batch_size = batch_size
        self.effective_batch_size = accumulation_steps * batch_size
        self.patience = None
        self.cooldown = None
        self.min_lr = None
        self.max_lr = None
        self.warmup = None
        self.smooth = None
        self.window_size = None
        self.reset = None
        self.factor = None
        self.loss_reduction = None
        self.total_iters = None
        self.state_dict = None

    def with_patience(self, patience: int) -> Self:
        self.patience = patience
        return self

    def with_cooldown(self, cooldown: int) -> Self:
        self.cooldown = cooldown
        return self

    def with_min_lr(self, min_lr: float) -> Self:
        self.min_lr = min_lr
        return self

    def with_max_lr(self, max_lr: float) -> Self:
        self.max_lr = max_lr
        return self

    def with_warmup(self, warmup: int) -> Self:
        self.warmup = warmup
        return self

    def with_smooth(self, smooth: bool) -> Self:
        self.smooth = smooth
        return self

    def with_window_size(self, window_size: int) -> Self:
        self.window_size = window_size
        return self

    def with_reset(self, reset: int) -> Self:
        self.reset = reset
        return self

    def with_factor(self, factor: float) -> Self:
        self.factor = factor
        return self

    def with_batch_size(self, batch_size: int) -> Self:
        self.batch_size = batch_size
        return self

    def with_loss_reduction(self, loss_reduction: LossReductionType) -> Self:  # noqa: F821
        self.loss_reduction = loss_reduction
        return self

    def with_total_iters(self, total_iters: int) -> Self:
        self.total_iters = total_iters
        return self

    def with_power(self, power: float) -> Self:
        self.power = power
        return self

    def with_state_dict(self, state_dict: Dict) -> Self:
        self.state_dict = state_dict
        return self

    def build(self, optimizer: Optimizer) -> LRScheduler:
        match self.scheduler_type:
            case SchedulerType.POLYNOMIAL:
                if self.total_iters is None:
                    raise AttributeError("Total Iters not defined for PolynomialLR scheduler")
                if self.power is None:
                    raise AttributeError("Power not defined for PolynomialLR scheduler")
                polynomial_lr_params = {
                    "total_iters": self.total_iters,
                    "power": self.power,
                }
                scheduler = PolynomialLR(optimizer, **polynomial_lr_params)
                if self.state_dict is not None:
                    scheduler.load_state_dict(self.state_dict)
                    self.state_dict = None
                return scheduler
            case SchedulerType.GREEDY:
                if self.factor is None:
                    raise AttributeError("factor not defined for GreedyLR scheduler")
                if self.min_lr is None:
                    raise AttributeError("min_lr is not defined for GreedyLR scheduler")
                if self.max_lr is None:
                    raise AttributeError("max_lr is not defined for GreedyLR scheduler")
                if self.cooldown is None:
                    raise AttributeError("cooldown is not defined for GreedyLR scheduler")
                if self.patience is None:
                    raise AttributeError("patience is not defined for GreedyLR scheduler")
                if self.warmup is None:
                    raise AttributeError("warmup is not defined for GreedyLR scheduler")
                if self.smooth is None:
                    raise AttributeError("smooth is not defined for GreedyLR scheduler")
                if self.window_size is None:
                    raise AttributeError("window is not defined for GreedyLR scheduler")
                if self.reset is None:
                    raise AttributeError("reset is not defined for GreedyLR scheduler")

                effective_min_lr = (
                    self.min_lr / self.accumulation_steps
                    if self.loss_reduction == LossReductionType.MEAN
                    else self.min_lr / self.effective_batch_size
                )
                effective_max_lr = (
                    self.max_lr / self.accumulation_steps
                    if self.loss_reduction == LossReductionType.MEAN
                    else self.max_lr / self.effective_batch_size
                )

                greedy_lr_params = {
                    "factor": self.factor,
                    "min_lr": effective_min_lr,
                    "max_lr": effective_max_lr,
                    "cooldown": self.cooldown,
                    "patience": self.patience,
                    "warmup": self.warmup,
                    "smooth": self.smooth,
                    "window_size": self.window_size,
                    "reset": self.reset,
                }
                scheduler = GreedyLR(optimizer, **greedy_lr_params)
                if self.state_dict is not None:
                    scheduler.load_state_dict(self.state_dict)
                    self.state_dict = None
                return scheduler
            case SchedulerType.PLATEAU:
                if self.factor is None:
                    raise AttributeError("factor is not defined for ReduceLROnPlateau scheduler")
                if self.patience is None:
                    raise AttributeError("patience is not defined for ReduceLROnPlateau scheduler")
                if self.cooldown is None:
                    raise AttributeError("cooldown is not defined for ReduceLROnPlateau scheduler")
                if self.min_lr is None:
                    raise AttributeError("min_lr is not defined for ReduceLROnPlateau scheduler")

                effective_min_lr = (
                    self.min_lr / self.accumulation_steps
                    if self.loss_reduction == LossReductionType.MEAN
                    else self.min_lr / self.effective_batch_size
                )
                plateau_lr_params = {
                    "factor": self.factor,
                    "patience": self.patience,
                    "cooldown": self.cooldown,
                    "min_lr": effective_min_lr,
                }
                scheduler = ReduceLROnPlateau(optimizer, mode="min", **plateau_lr_params)
                if self.state_dict is not None:
                    scheduler.load_state_dict(self.state_dict)
                    self.state_dict = None
                return scheduler
