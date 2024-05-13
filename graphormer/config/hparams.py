import os
from typing import Any, Dict, Optional, Self
import torch
import datetime
from random import random
from graphormer.config.data import DataConfig
from graphormer.config.logging import LoggingConfig
from graphormer.config.loss import LossConfig
from graphormer.config.model import ModelConfig
from graphormer.config.optimizer import OptimizerConfig
from graphormer.config.scheduler import SchedulerConfig
from graphormer.config.options import LossReductionType, DatasetType, SchedulerType, OptimizerType


class HyperparameterConfig:
    def __init__(
        self,
        datadir: Optional[str] = None,
        dataset: Optional[DatasetType] = None,
        batch_size: Optional[int] = None,
        max_path_distance: Optional[int] = None,
        num_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        n_heads: Optional[int] = None,
        max_in_degree: Optional[int] = None,
        max_out_degree: Optional[int] = None,
        output_dim: Optional[int] = None,
        node_feature_dim: Optional[int] = None,
        edge_feature_dim: Optional[int] = None,
        test_size: Optional[float] = None,
        tune_size: float = 1.0,
        random_state: int = int(random() * 1e9),
        lr: Optional[float] = None,
        b1: Optional[float] = None,
        b2: Optional[float] = None,
        weight_decay: Optional[float] = None,
        eps: Optional[float] = None,
        clip_grad_norm: float = 5.0,
        torch_device: str = "cuda",
        epochs: int = 10,
        lr_power: Optional[float] = None,
        scheduler_type: Optional[SchedulerType] = None,
        lr_patience: Optional[int] = None,
        lr_cooldown: Optional[int] = None,
        lr_min: Optional[float] = None,
        lr_max: Optional[float] = None,
        lr_warmup: Optional[int] = None,
        lr_smooth: Optional[bool] = None,
        lr_window: Optional[int] = None,
        lr_reset: Optional[int] = None,
        lr_factor: Optional[float] = None,
        checkpt_save_interval: int = 1,
        accumulation_steps: Optional[int] = None,
        loss_reduction: Optional[LossReductionType] = None,
        name: Optional[str] = None,
        optimizer_type: Optional[OptimizerType] = None,
        momentum: Optional[float] = None,
        nesterov: Optional[bool] = None,
        dampening: Optional[float] = None,
        logdir: Optional[str] = None,
        flush_secs: Optional[int] = None,
        purge_step: Optional[int] = None,
        comment: Optional[str] = None,
        max_queue: Optional[int] = None,
        write_to_disk: Optional[bool] = None,
        filename_suffix: Optional[str] = None,
        start_epoch: int = 0,
        checkpoint_dir: str = "pretrained_models",
        dropout: Optional[float] = None,
        rescale: Optional[bool] = None,
    ):
        if name is None:
            name = datetime.datetime.now().strftime("%d-%m-%y")
        self.datadir = datadir
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_path_distance = max_path_distance
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.output_dim = output_dim
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.test_size = test_size
        self.tune_size = tune_size
        self.random_state = random_state
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.eps = eps
        self.clip_grad_norm = clip_grad_norm
        self.torch_device = torch_device
        self.epochs = epochs
        self.lr_power = lr_power
        self.scheduler_type = scheduler_type
        self.lr_patience = lr_patience
        self.lr_cooldown = lr_cooldown
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_warmup = lr_warmup
        self.lr_smooth = lr_smooth
        self.lr_window = lr_window
        self.lr_reset = lr_reset
        self.lr_factor = lr_factor
        self.checkpt_save_interval = checkpt_save_interval
        self.accumulation_steps = accumulation_steps
        self.loss_reduction = loss_reduction
        self.name = name
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening
        self.logdir = logdir
        self.flush_secs = flush_secs
        self.purge_step = purge_step
        self.comment = comment
        self.max_queue = max_queue
        self.write_to_disk = write_to_disk
        self.filename_suffix = filename_suffix
        self.start_epoch = start_epoch
        self.checkpoint_dir = checkpoint_dir
        self.dropout = dropout
        self.rescale = rescale
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.loss_state_dict = None

    def data_config(self) -> DataConfig:
        if self.dataset is None:
            raise AttributeError("dataset not defined for DataConfig")
        if self.batch_size is None:
            raise AttributeError("batch_size not defined for DataConfig")
        if self.datadir is None:
            raise AttributeError("datadir is not defined for DataConfig")
        if self.max_path_distance is None:
            raise AttributeError("max_path_distance is not defined for DataConfig")

        config = DataConfig(self.dataset, self.batch_size, self.datadir, self.max_path_distance)

        if self.test_size is not None:
            config = config.with_test_size(self.test_size)
        if self.random_state is not None:
            config = config.with_random_state(self.random_state)

        return config

    def model_config(self) -> ModelConfig:
        config = ModelConfig()
        if self.num_layers is not None:
            config = config.with_num_layers(self.num_layers)
        if self.hidden_dim is not None:
            config = config.with_hidden_dim(self.hidden_dim)
        if self.edge_embedding_dim is not None:
            config = config.with_edge_embedding_dim(self.edge_embedding_dim)
        if self.ffn_hidden_dim is not None:
            config = config.with_ffn_hidden_dim(self.ffn_hidden_dim)
        if self.n_heads is not None:
            config = config.with_num_heads(self.n_heads)
        if self.max_in_degree is not None:
            config = config.with_max_in_degree(self.max_in_degree)
        if self.max_out_degree is not None:
            config = config.with_max_out_degree(self.max_out_degree)
        if self.output_dim is not None:
            config = config.with_output_dim(self.output_dim)
        if self.node_feature_dim is not None:
            config = config.with_node_feature_dim(self.node_feature_dim)
        if self.edge_feature_dim is not None:
            config = config.with_edge_feature_dim(self.edge_feature_dim)
        if self.max_path_distance is not None:
            config = config.with_max_path_distance(self.max_path_distance)
        if self.dropout is not None:
            config = config.with_dropout(self.dropout)
        if self.rescale is not None:
            config = config.with_rescale(self.rescale)
        if self.model_state_dict is not None:
            config = config.with_state_dict(self.model_state_dict)
            self.model_state_dict = None

        return config

    def scheduler_config(self) -> SchedulerConfig:
        if self.scheduler_type is None:
            raise AttributeError("scheduler_type is not defined for SchedulerConfig")
        if self.accumulation_steps is None:
            raise AttributeError("accumulation_steps is not defined for SchedulerConfig")
        if self.batch_size is None:
            raise AttributeError("batch_size is not defined for SchedulerConfig")

        config = SchedulerConfig(self.scheduler_type, self.accumulation_steps, self.batch_size)
        if self.lr_factor is not None:
            config = config.with_factor(self.lr_factor)
        if self.lr_warmup is not None:
            config = config.with_warmup(self.lr_warmup)
        if self.lr_cooldown is not None:
            config = config.with_cooldown(self.lr_cooldown)
        if self.lr_power is not None:
            config = config.with_power(self.lr_power)
        if self.lr_min is not None:
            config = config.with_min_lr(self.lr_min)
        if self.lr_max is not None:
            config = config.with_max_lr(self.lr_max)
        if self.lr_smooth is not None:
            config = config.with_smooth(self.lr_smooth)
        if self.lr_window is not None:
            config = config.with_window_size(self.lr_window)
        if self.epochs is not None:
            config = config.with_total_iters(self.epochs)
        if self.batch_size is not None:
            config = config.with_batch_size(self.batch_size)
        if self.loss_reduction is not None:
            config = config.with_loss_reduction(self.loss_reduction)
        if self.lr_patience is not None:
            config = config.with_patience(self.lr_patience)
        if self.lr_reset is not None:
            config = config.with_reset(self.lr_reset)
        if self.scheduler_state_dict is not None:
            config = config.with_state_dict(self.scheduler_state_dict)
            self.scheduler_state_dict = None
        return config

    def optimizer_config(self) -> OptimizerConfig:
        if self.optimizer_type is None:
            raise AttributeError("optimizer_type is not defined for OptimizerConfig")
        if self.loss_reduction is None:
            raise AttributeError("loss_reduction is not defined for OptimizerConfig")
        if self.accumulation_steps is None:
            raise AttributeError("accumulation_steps is not defined for OptimizerConfig")
        if self.batch_size is None:
            raise AttributeError("batch_size is not defined for OptimizerConfig")
        if self.lr is None:
            raise AttributeError("lr is not defined for OptimizerConfig")

        config = OptimizerConfig(
            self.optimizer_type, self.loss_reduction, self.accumulation_steps, self.batch_size, self.lr
        )

        if self.b1 is not None and self.b2 is not None:
            config = config.with_betas((self.b1, self.b2))
        if self.eps is not None:
            config = config.with_eps(self.eps)
        if self.momentum is not None:
            config = config.with_momentum(self.momentum)
        if self.nesterov is not None:
            config = config.with_nesterov(self.nesterov)
        if self.dampening is not None:
            config = config.with_dampening(self.dampening)
        if self.weight_decay is not None:
            config = config.with_weight_decay(self.weight_decay)
        if self.optimizer_state_dict is not None:
            config = config.with_state_dict(self.optimizer_state_dict)
            self.optimizer_state_dict = None

        return config

    def loss_config(self) -> LossConfig:
        if self.loss_reduction is None:
            raise AttributeError("loss_reduction is not defined for LossConfig")
        if self.torch_device is None:
            raise AttributeError("torch_device is not defined for LossConfig")

        config = LossConfig(self.loss_reduction, torch.device(self.torch_device))

        if self.loss_state_dict is not None:
            config = config.with_state_dict(self.loss_state_dict)
            self.loss_state_dict = None
        return config

    def logging_config(self) -> LoggingConfig:
        if self.logdir is None:
            raise AttributeError("logdir is not defined for LoggingConfig")

        config = LoggingConfig(f"{self.logdir}/{self.name}")

        if self.flush_secs is not None:
            config = config.with_flush_secs(self.flush_secs)
        if self.purge_step is not None:
            config = config.with_purge_step(self.purge_step)
        if self.comment is not None:
            config = config.with_comment(self.comment)
        if self.max_queue is not None:
            config = config.with_max_queue(self.max_queue)
        if self.write_to_disk is not None:
            config = config.with_write_to_disk(self.write_to_disk)
        if self.filename_suffix is not None:
            config = config.with_filename_suffix(self.filename_suffix)

        return config

    def load_from_checkpoint(self) -> Self:
        device = torch.device(self.torch_device)

        if self.checkpoint_dir is None:
            return self
        checkpoint_path = f"{self.checkpoint_dir}/{self.name}.pt"
        if not os.path.exists(checkpoint_path):
            return self
        checkpoint = torch.load(checkpoint_path, map_location=device)

        hparams: Dict[str, Any] = checkpoint["hyperparameters"]
        for key, value in hparams.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_state_dict = checkpoint["model_state_dict"]
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        self.loss_state_dict = checkpoint["loss_state_dict"]
        self.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Successfully loaded model {self.name} at epoch {self.start_epoch}")
        del checkpoint
        return self

    def load_for_inference(self) -> Self:
        device = torch.device(self.torch_device)

        checkpoint_path = f"{self.checkpoint_dir}/{self.name}.pt"
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {self.checkpoint_path} does not exist - Inference requires a trained model")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        hparams: Dict[str, Any] = checkpoint["hyperparameters"]
        for key, value in hparams.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_state_dict = checkpoint["model_state_dict"]
        print(f"Successfully loaded model {self.name} for inference")
        del checkpoint
        return self
