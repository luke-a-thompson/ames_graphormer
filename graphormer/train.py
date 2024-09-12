from typing import List, Optional, Self

import optuna
import torch
from optuna.trial import Trial
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, PolynomialLR, ReduceLROnPlateau
from tqdm import tqdm

from graphormer.config.hparams import HyperparameterConfig
from graphormer.config.options import AttentionType, LossReductionType, ResidualType, SchedulerType, LossFunction
from graphormer.config.utils import calculate_pos_weight, model_init_print, save_checkpoint
from graphormer.data.dataloader import GraphormerBatch, GraphormerDataLoader
from graphormer.model_analysis import (
    plot_attention_sigma,
    plot_centrality_in_degree_bias,
    plot_centrality_out_degree_bias,
    plot_edge_path_length_bias,
    plot_layer_residual_weights,
    plot_node_path_length_bias,
)
from graphormer.modules.model import Graphormer
from graphormer.schedulers import GreedyLR
import numpy as np


class Trainer:
    def __init__(
        self,
        hparam_config: HyperparameterConfig,
        scheduler: LRScheduler,
        model: Graphormer,
        train_loader: GraphormerDataLoader,
        test_loader: GraphormerDataLoader,
        device: torch.device,
        optimizer: Optimizer,
        loss: _Loss,
        writer: SummaryWriter,
        effective_batch_size: int,
    ):
        self.hparam_config = hparam_config
        self.scheduler = scheduler
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.writer = writer
        self.effective_batch_size = effective_batch_size

    @classmethod
    def build(
        cls,
        hparam_config: HyperparameterConfig,
        train_loader: Optional[GraphormerDataLoader] = None,
        test_loader: Optional[GraphormerDataLoader] = None,
    ) -> Self:
        device = torch.device(hparam_config.torch_device)
        logging_config = hparam_config.logging_config()
        data_config = hparam_config.data_config()
        if train_loader is None or test_loader is None:
            train_loader, test_loader = data_config.build()

        assert train_loader is not None
        assert test_loader is not None
        model_config = hparam_config.model_config()
        loss_config = hparam_config.loss_config()
        optimizer_config = hparam_config.optimizer_config()
        scheduler_config = hparam_config.scheduler_config()
        assert hparam_config.batch_size is not None
        assert hparam_config.last_effective_batch_num is not None

        writer = logging_config.build()
        assert data_config.num_node_features is not None
        assert data_config.num_edge_features is not None
        model = (
            model_config.with_node_feature_dim(data_config.num_node_features)
            .with_edge_feature_dim(data_config.num_edge_features)
            .with_output_dim(1)
            .build()
            .to(hparam_config.torch_device)
        )
        loss_weights = hparam_config.loss_weights
        if LossFunction.MSE not in hparam_config.loss_function:
            pos_weight = calculate_pos_weight(train_loader)
            loss = loss_config.with_pos_weight(pos_weight).with_weights(loss_weights).build()
        else:
            loss = loss_config.with_weights(loss_weights).build()
        optimizer = optimizer_config.build(model)
        if scheduler_config.scheduler_type == SchedulerType.ONE_CYCLE:
            scheduler_config = scheduler_config.with_train_batches_per_epoch(len(train_loader))
        scheduler = scheduler_config.build(optimizer)
        return cls(
            hparam_config,
            scheduler,
            model,
            train_loader,
            test_loader,
            device,
            optimizer,
            loss,
            writer,
            optimizer_config.effective_batch_size,
        )

    def fit(
        self,
        trial: Optional[Trial] = None,
        train_loader: Optional[GraphormerDataLoader] = None,
        test_loader: Optional[GraphormerDataLoader] = None,
        optimized_model: bool = False,
    ) -> float:
        train_loader = train_loader if train_loader is not None else self.train_loader
        test_loader = test_loader if test_loader is not None else self.test_loader
        model_init_print(self.hparam_config, self.model, train_loader, test_loader)
        self.model.train()

        if optimized_model:
            self.model: Graphormer = torch.compile(self.model, mode="max_autotune")  # type: ignore

        progress_bar = tqdm(total=0, desc="Initializing...", unit="batch")
        train_batches_per_epoch = len(train_loader)
        eval_batches_per_epoch = len(test_loader)

        avg_eval_loss = float("inf")

        assert self.hparam_config.accumulation_steps is not None
        assert self.hparam_config.last_effective_batch_num is not None
        assert self.hparam_config.batch_size is not None

        for epoch in range(self.hparam_config.start_epoch, self.hparam_config.epochs):
            total_train_loss = 0.0
            total_eval_loss = 0.0

            # Set total length for training phase and update description
            progress_bar.reset(total=len(train_loader))
            progress_bar.set_description(f"Epoch {epoch+1}/{self.hparam_config.epochs} Train")

            self.model.train()

            avg_loss = 0.0
            train_batch_num = epoch * train_batches_per_epoch
            loss_values: list[float] = []
            for batch_idx, batch in enumerate(train_loader):
                if (
                    self.hparam_config.tune_size is not None
                    and batch_idx / train_batches_per_epoch > self.hparam_config.tune_size
                    and trial is not None
                ):
                    break

                if train_batch_num == 0 and trial is None:
                    self.optimizer.zero_grad()

                batch_loss = self.train_step(batch, batch_idx, train_batch_num, loss_values, train_batches_per_epoch)

                total_train_loss += batch_loss
                interleaved_list = ", ".join(
                    [
                        f"{w}x '{f.value}'"
                        for w, f in zip(self.hparam_config.loss_weights, self.hparam_config.loss_function)
                    ]
                )
                progress_bar.set_postfix_str(f"Avg Loss: {avg_loss:.3f}. Weights: {interleaved_list}")
                progress_bar.update()  # Increment the progress bar
                train_batch_num += 1

                avg_loss = total_train_loss / (batch_idx + 1)
                if self.hparam_config.loss_reduction == LossReductionType.SUM:
                    avg_loss /= self.hparam_config.batch_size
                self.writer.add_scalar("train/avg_train_loss", avg_loss, epoch)

            if isinstance(self.scheduler, PolynomialLR):
                self.scheduler.step()
            self.writer.add_scalar(
                "train/lr",
                (
                    self.scheduler.get_last_lr()[0] * self.hparam_config.accumulation_steps
                    if self.hparam_config.loss_reduction == LossReductionType.MEAN
                    else self.scheduler.get_last_lr()[0] * self.effective_batch_size
                ),
                epoch,
            )

            # Prepare for the evaluation phase
            progress_bar.reset(total=len(test_loader))
            progress_bar.set_description(f"Epoch {epoch+1}/{self.hparam_config.epochs} Eval")

            all_eval_labels = []
            all_eval_preds = []

            self.model.eval()
            eval_batch_num = epoch * eval_batches_per_epoch
            for batch in test_loader:
                batch_loss = self.eval_step(batch, eval_batch_num, all_eval_preds, all_eval_labels)
                total_eval_loss += batch_loss
                progress_bar.update()
                eval_batch_num += 1

            if isinstance(self.scheduler, (ReduceLROnPlateau, GreedyLR)):
                self.scheduler.step(total_eval_loss)

            avg_eval_loss = total_eval_loss / len(test_loader)
            if self.hparam_config.loss_reduction == LossReductionType.SUM:
                avg_eval_loss /= float(self.hparam_config.batch_size)
            progress_bar.set_postfix_str(f"Avg Eval Loss: {avg_eval_loss:.4f}")
            if LossFunction.MSE not in self.hparam_config.loss_function:
                bac = balanced_accuracy_score(all_eval_labels, all_eval_preds)
                ac = accuracy_score(all_eval_labels, all_eval_preds)
                bac_adj = balanced_accuracy_score(all_eval_labels, all_eval_preds, adjusted=True)
                self.writer.add_scalar("eval/acc", ac, epoch)
                self.writer.add_scalar("eval/bac", bac, epoch)
                self.writer.add_scalar("eval/bac_adj", bac_adj, epoch)
                print(
                    f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f} | Avg Eval Loss: {
                    avg_eval_loss:.4f} | Eval BAC: {bac:.4f} | Eval ACC: {ac:.4f}"
                )
            else:
                mae = mean_absolute_error(all_eval_labels, all_eval_preds)
                rmse = np.sqrt(mean_squared_error(all_eval_labels, all_eval_preds))
                self.writer.add_scalar("eval/mae", mae, epoch)
                self.writer.add_scalar("eval/rmse", rmse, epoch)
                print(
                    f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f} | Avg Eval Loss: {
                    avg_eval_loss:.4f} | Eval RMSE: {rmse:.4f} | Eval MAE: {mae:.4f}"
                )
            self.writer.add_scalar("eval/avg_eval_loss", avg_eval_loss, epoch)
            self.writer.add_figure(
                # type: ignore
                "plot/edge_encoding_bias",
                plot_edge_path_length_bias(self.model),
                epoch,
            )
            self.writer.add_figure(
                # type: ignore
                "plot/node_encoding_bias",
                plot_node_path_length_bias(self.model),
                epoch,
            )
            self.writer.add_figure(
                "plot/centrality_in_degree_bias",
                # type: ignore
                plot_centrality_in_degree_bias(self.model),
                epoch,
            )
            self.writer.add_figure(
                "plot/centrality_out_degree_bias",
                # type: ignore
                plot_centrality_out_degree_bias(self.model),
                epoch,
            )
            if self.hparam_config.residual_type == ResidualType.REZERO:
                self.writer.add_figure(
                    # type: ignore
                    "plot/residual_weigths",
                    plot_layer_residual_weights(self.model),
                    epoch,
                )
            if self.hparam_config.attention_type == AttentionType.FISH:
                self.writer.add_figure(
                    # type: ignore
                    "plot/sigma_strength",
                    plot_attention_sigma(self.model),
                    epoch,
                )

            if total_eval_loss < self.hparam_config.best_loss and trial is None:
                self.hparam_config.best_loss = avg_eval_loss
                save_checkpoint(
                    epoch,
                    self.hparam_config,
                    self.model,
                    self.optimizer,
                    self.loss,
                    self.scheduler,
                    "best",
                )

            if epoch % self.hparam_config.checkpt_save_interval == 0 and trial is None:
                save_checkpoint(
                    epoch,
                    self.hparam_config,
                    self.model,
                    self.optimizer,
                    self.loss,
                    self.scheduler,
                )

            if trial is not None:
                trial.report(avg_eval_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        progress_bar.close()
        return avg_eval_loss

    def train_step(
        self,
        batch: GraphormerBatch,
        batch_idx: int,
        train_batch_num: int,
        loss_values: List[float],
        train_batches_per_epoch: int,
    ):
        batch.to(self.device)  # type: ignore
        y = batch.y.to(self.device)  # type: ignore

        output = self.model(batch)

        loss = self.loss(output, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.hparam_config.clip_grad_norm, error_if_nonfinite=True
        )
        if should_step(batch_idx, self.hparam_config.accumulation_steps, train_batches_per_epoch):  # type: ignore
            self.optimizer.step()
            self.optimizer.zero_grad()
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
                self.hparam_config.last_effective_batch_num += 1  # type: ignore

        batch_loss = loss.item()
        loss_values.append(batch_loss)
        self.writer.add_scalar("train/batch_loss", batch_loss, train_batch_num)
        self.writer.add_scalar(
            "train/sample_loss",
            (
                batch_loss / output.shape[0]
                if self.hparam_config.loss_reduction == LossReductionType.SUM
                else batch_loss
            ),
            train_batch_num,
        )
        return batch_loss

    def eval_step(
        self, batch: GraphormerBatch, eval_batch_num: int, all_eval_preds: List[float], all_eval_labels: List[int]
    ):
        batch.to(self.device)  # type: ignore
        y = batch.y.to(self.device)  # type: ignore
        with torch.no_grad():
            output = self.model(batch)
            loss = self.loss(output, y)
        batch_loss: float = loss.item()
        self.writer.add_scalar("eval/batch_loss", batch_loss, eval_batch_num)

        if LossFunction.MSE not in self.hparam_config.loss_function:
            eval_preds = torch.round(torch.sigmoid(output)).tolist()
            eval_labels = y.cpu().numpy()
            if sum(eval_labels) > 0:
                batch_bac = balanced_accuracy_score(eval_labels, eval_preds)
                self.writer.add_scalar("eval/batch_bac", batch_bac, eval_batch_num)
        else:
            eval_preds = output.cpu().numpy()
            eval_labels = y.cpu().numpy()

        all_eval_preds.extend(eval_preds)
        all_eval_labels.extend(eval_labels)
        return batch_loss


def should_step(batch_idx: int, accumulation_steps: int, train_batches_per_epoch: int) -> bool:
    if accumulation_steps <= 1:
        return True
    if batch_idx > 0 and (batch_idx + 1) % accumulation_steps == 0:
        return True
    if batch_idx >= train_batches_per_epoch - 1:
        return True
    return False
