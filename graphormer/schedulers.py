# pyright: reportIncompatibleMethodOverride=false
from typing import List, Optional, Tuple

import torch
from torch.optim.lr_scheduler import LRScheduler


class GreedyLR(LRScheduler):
    """
        Implements the Greedy LR scheduler described in "Zeroth Order GreedyLR: An Adaptive Learning
    Rate Scheduler for Deep Neural Network Training" by Subramanian & Ganapathiraman 2023.
    https://www.amazon.science/publications/zeroth-order-greedylr-an-adaptive-learning-rate-scheduler-for-deep-neural-network-training
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float | List[float] | Tuple[float, ...] = 1e-7,
        max_lr: float | List[float] | Tuple[float, ...] = 1e-3,
        cooldown: int = 0,
        warmup: int = 0,
        patience: int = 0,
        smooth: bool = True,
        window_size: int = 10,
        factor: float = 0.1,
        reset: int = 0,
        threshold: float = 1e-4,
        eps: float = 1e-8,
    ):
        self.optimizer = optimizer
        assert 0 < factor < 1, f"factor must be in the range (0, 1), got: {
            factor}"
        self.init_lrs = []
        for group in optimizer.param_groups:
            self.init_lrs.append(group['lr'])
        self.smooth = smooth
        self.factor = factor
        self.cooldown = cooldown
        self.warmup = warmup
        self.patience = patience
        self.reset = reset
        self.threshold = threshold
        self.warmup_counter = 0
        self.cooldown_counter = 0
        self.num_good_epochs = 0
        self.num_bad_epochs = 0
        self.window = []
        self.window_size = window_size
        self.eps = eps
        self.best_loss = float("inf")
        self._last_lr = [x['lr'] for x in optimizer.param_groups]
        self.last_epoch = 0
        self.has_reset = False

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

    def step(self, total_eval_loss: float, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        val_loss = total_eval_loss
        if self.smooth:
            self.window.append(total_eval_loss)
            if len(self.window) > self.window_size:
                self.window.pop(0)
            val_loss = sum(self.window) / len(self.window)

        loss_is_improved = val_loss < self.best_loss - self.threshold
        if loss_is_improved:
            self.best_loss = val_loss
            self.num_good_epochs += 1
            self.num_bad_epochs = 0
        else:
            self.num_good_epochs = 0
            self.num_bad_epochs += 1

        if self.cooldown_counter < self.cooldown:
            self.cooldown_counter += 1
            self.num_good_epochs = 0

        if self.warmup_counter < self.warmup:
            self.warmup_counter += 1
            self.num_bad_epochs = 0

        if self.num_good_epochs > self.patience:
            self._increase_lr()
            self.cooldown_counter = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.warmup_counter = 0

        if self.reset > 0 and epoch > self.reset and not self.has_reset:
            self._reset_lr()
            self.num_good_epochs = 0
            self.num_bad_epochs = 0
            self.warmup_counter = 0
            self.cooldown_counter = 0
            self.best_loss = float("-inf")
            self.has_reset = True
            if self.smooth:
                self.window = []

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.last_epoch = epoch

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr

    def _increase_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(old_lr / self.factor, self.max_lrs[i])
            if new_lr - old_lr > self.eps:
                param_group['lr'] = new_lr

    def _reset_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.init_lrs[i]
