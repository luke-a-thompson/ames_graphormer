import os
from typing import Any, Dict, List, Tuple, Optional, Self
import click
import torch
import datetime
from random import random
from graphormer.config.data import DataConfig
from graphormer.config.logging import LoggingConfig
from graphormer.config.loss import LossConfig
from graphormer.config.model import ModelConfig
from graphormer.config.optimizer import OptimizerConfig
from graphormer.config.scheduler import SchedulerConfig
from graphormer.config.options import (
    AttentionType,
    LossFunction,
    LossReductionType,
    DatasetType,
    DatasetRegime,
    ResidualType,
    SchedulerType,
    OptimizerType,
    NormType,
)
from graphormer.utils import create_composite_decorator, configure


hyperparameters = create_composite_decorator(
    click.option(
        "-c",
        "--config",
        type=click.Path(dir_okay=False),
        is_eager=True,
        expose_value=False,
        help="Read option values from the specified config file",
        callback=configure,
        default="default_hparams.toml",
    ),
    click.option("--datadir", default="data"),
    click.option("--logdir", default="runs"),
    click.option("--dataset", type=click.Choice(DatasetType, case_sensitive=False), default=DatasetType.HONMA),  # type: ignore
    click.option(
        "--dataset_regime", type=click.Choice(DatasetRegime, case_sensitive=False), default=DatasetRegime.TRAIN
    ),  # type: ignore
    click.option("--num_layers", default=3),
    click.option("--hidden_dim", default=128),
    click.option("--edge_embedding_dim", default=128),
    click.option("--ffn_hidden_dim", default=80),
    click.option("--n_heads", default=4),
    click.option("--heads_by_layer", multiple=True, default=[], type=click.INT),
    click.option("--max_in_degree", default=5),
    click.option("--max_out_degree", default=5),
    click.option("--max_path_distance", default=5),
    click.option(
        "--residual_type", type=click.Choice(ResidualType, case_sensitive=False), default=ResidualType.PRENORM
    ),  # type: ignore
    click.option("--test_size", default=0.8),
    click.option("--random_state", default=None, type=click.INT),
    click.option("--batch_size", default=16),
    click.option("--lr", default=3e-4),
    click.option("--b1", default=0.9),
    click.option("--b2", default=0.999),
    click.option("--weight_decay", default=0.0),
    click.option("--eps", default=1e-8),
    click.option("--nesterov", default=False),
    click.option("--momentum", default=0.0),
    click.option("--dampening", default=0.0),
    click.option("--clip_grad_norm", default=5.0),
    click.option("--torch_device", default="cuda"),
    click.option("--epochs", default=10),
    click.option("--lr_power", default=0.5),
    click.option(
        "--scheduler_type",
        type=click.Choice(SchedulerType, case_sensitive=False),  # type: ignore
        default=SchedulerType.FIXED,
    ),
    click.option(
        "--optimizer_type", type=click.Choice(OptimizerType, case_sensitive=False), default=OptimizerType.ADAMW
    ),  # type: ignore
    click.option("--lr_patience", default=4),
    click.option("--lr_cooldown", default=2),
    click.option("--lr_min", default=1e-6),
    click.option("--lr_max", default=1e-3),
    click.option("--lr_warmup", default=2),
    click.option("--lr_smooth", default=True),
    click.option("--lr_window", default=10),
    click.option("--lr_reset", default=0),
    click.option("--lr_factor", default=0.5),
    click.option("--pct_start", default=0.3),
    click.option("--div_factor", default=25),
    click.option("--final_div_factor", default=1e4),
    click.option("--cycle_momentum", default=True),
    click.option("--three_phase", default=False),
    click.option("--max_momentum", default=0.95),
    click.option("--base_momentum", default=0.85),
    click.option("--last_effective_batch_num", default=-1),
    click.option("--anneal_strategy", default="cos"),
    click.option("--name", default=None),
    click.option("--checkpt_save_interval", default=5),
    click.option("--accumulation_steps", default=1),
    click.option(
        "--loss_function",
        type=click.Choice(LossFunction, case_sensitive=False),
        multiple=True,
        default=[LossFunction.BCE_LOGITS],
    ),  # type: ignore
    click.option(
        "--loss_reduction", type=click.Choice(LossReductionType, case_sensitive=False), default=LossReductionType.MEAN
    ),  # type: ignore
    click.option("--loss_weights", type=click.FLOAT, multiple=True, default=(1,)),  # type: ignore
    click.option("--checkpoint_dir", default="pretrained_models"),
    click.option("--dropout", default=0.05),
    click.option("--temperature", default=1.0),
    click.option("--norm_type", type=click.Choice(NormType, case_sensitive=False), default=NormType.LAYER),  # type: ignore
    click.option("--attention_type", type=click.Choice(AttentionType, case_sensitive=False), default=AttentionType.MHA),  # type: ignore
    click.option("--n_global_heads", default=4),
    click.option("--n_local_heads", default=8),
    click.option("--global_heads_by_layer", multiple=True, default=[], type=click.INT),
    click.option("--local_heads_by_layer", multiple=True, default=[], type=click.INT),
    click.option("--flush_secs", default=5),
    click.option("--num_workers", default=4),
    click.option("--prefetch_factor", default=16),
)


class HyperparameterConfig:
    def __init__(
        self,
        # Global Parameters
        name: Optional[str] = None,
        random_state: int = int(random() * 1e9),
        torch_device: str = "cuda",
        best_loss: float = float("inf"),
        # Data Parameters
        datadir: Optional[str] = None,
        dataset: Optional[DatasetType] = None,
        dataset_regime: Optional[DatasetRegime] = None,
        batch_size: Optional[int] = None,
        max_path_distance: Optional[int] = None,
        node_feature_dim: Optional[int] = None,
        edge_feature_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        test_size: Optional[float] = None,
        tune_size: float = 1.0,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        # Model Parameters
        num_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_global_heads: Optional[int] = None,
        n_local_heads: Optional[int] = None,
        heads_by_layer: Optional[List[int]] = None,
        global_heads_by_layer: Optional[List[int]] = None,
        local_heads_by_layer: Optional[List[int]] = None,
        max_in_degree: Optional[int] = None,
        max_out_degree: Optional[int] = None,
        dropout: Optional[float] = None,
        temperature: Optional[float] = None,
        norm_type: Optional[NormType] = None,
        attention_type: Optional[AttentionType] = None,
        residual_type: Optional[ResidualType] = None,
        # Optimizer Parameters
        optimizer_type: Optional[OptimizerType] = None,
        momentum: Optional[float] = None,
        nesterov: Optional[bool] = None,
        dampening: Optional[float] = None,
        lr: Optional[float] = None,
        b1: Optional[float] = None,
        b2: Optional[float] = None,
        weight_decay: Optional[float] = None,
        eps: Optional[float] = None,
        clip_grad_norm: float = 5.0,
        loss_function: Optional[Tuple[LossFunction, ...]] = None,
        loss_reduction: Optional[LossReductionType] = None,
        loss_weights: Optional[Tuple[float, ...]] = None,
        # Scheduler Parameters
        scheduler_type: Optional[SchedulerType] = None,
        lr_power: Optional[float] = None,
        lr_patience: Optional[int] = None,
        lr_cooldown: Optional[int] = None,
        lr_min: Optional[float] = None,
        lr_max: Optional[float] = None,
        lr_warmup: Optional[int] = None,
        lr_smooth: Optional[bool] = None,
        lr_window: Optional[int] = None,
        lr_reset: Optional[int] = None,
        lr_factor: Optional[float] = None,
        pct_start: Optional[float] = None,
        div_factor: Optional[float] = None,
        final_div_factor: Optional[float] = None,
        cycle_momentum: Optional[bool] = None,
        three_phase: Optional[bool] = None,
        max_momentum: Optional[float | List[float]] = None,
        base_momentum: Optional[float | List[float]] = None,
        last_effective_batch_num: Optional[int] = None,
        anneal_strategy: Optional[str] = None,
        # Training Parameters
        start_epoch: int = 0,
        epochs: int = 10,
        accumulation_steps: Optional[int] = None,
        checkpt_save_interval: int = 1,
        checkpoint_dir: str = "pretrained_models",
        # Logging Parameters
        logdir: Optional[str] = None,
        flush_secs: Optional[int] = None,
        purge_step: Optional[int] = None,
        comment: Optional[str] = None,
        max_queue: Optional[int] = None,
        write_to_disk: Optional[bool] = None,
        filename_suffix: Optional[str] = None,
    ):
        # Global Parameters
        if name is None:
            name = f"model_{datetime.datetime.now().strftime("%d-%m-%y-%H-%M")}"
        self.name = name
        self.random_state = random_state
        self.torch_device = torch_device
        self.best_loss = best_loss
        # Data Parameters
        self.datadir = datadir
        self.dataset = dataset
        self.dataset_regime = dataset_regime
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.max_path_distance = max_path_distance
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.test_size = test_size
        self.tune_size = tune_size
        self.output_dim = output_dim
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        # Model Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.n_heads = n_heads
        self.n_global_heads = n_global_heads
        self.n_local_heads = n_local_heads
        self.heads_by_layer = heads_by_layer
        self.global_heads_by_layer = global_heads_by_layer
        self.local_heads_by_layer = local_heads_by_layer
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.norm_type = norm_type
        self.attention_type = attention_type
        self.residual_type = residual_type
        self.dropout = dropout
        self.temperature = temperature
        # Optimizer Parameters
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.eps = eps
        self.clip_grad_norm = clip_grad_norm
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction
        self.loss_weights = loss_weights
        # Scheduler Parameters
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
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.cycle_momentum = cycle_momentum
        self.three_phase = three_phase
        self.max_momentum = max_momentum
        self.base_momentum = base_momentum
        self.last_effective_batch_num = last_effective_batch_num
        self.anneal_strategy = anneal_strategy
        # Training Parameters
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.checkpt_save_interval = checkpt_save_interval
        self.checkpoint_dir = checkpoint_dir
        # Logging Parameters
        self.logdir = logdir
        self.flush_secs = flush_secs
        self.purge_step = purge_step
        self.comment = comment
        self.max_queue = max_queue
        self.write_to_disk = write_to_disk
        self.filename_suffix = filename_suffix

        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.loss_state_dict = None

    def data_config(self) -> DataConfig:
        if self.dataset is None or self.dataset not in DatasetType:
            raise AttributeError(f"dataset '{self.dataset}' not defined for DataConfig")
        if self.dataset_regime is None or self.dataset_regime not in DatasetRegime:
            raise AttributeError(f"dataset_regime '{self.dataset_regime}' not defined for DataConfig")
        if self.batch_size is None:
            raise AttributeError("batch_size not defined for DataConfig")
        if self.datadir is None:
            raise AttributeError("datadir is not defined for DataConfig")
        if self.max_path_distance is None:
            raise AttributeError("max_path_distance is not defined for DataConfig")

        config = DataConfig(self.dataset, self.dataset_regime, self.batch_size, self.datadir, self.max_path_distance)

        if self.test_size is not None:
            config = config.with_test_size(self.test_size)
        if self.random_state is not None:
            config = config.with_random_state(self.random_state)
        if self.num_workers is not None:
            config = config.with_num_workers(self.num_workers)
        if self.prefetch_factor is not None:
            config = config.with_prefetch_factor(self.prefetch_factor)

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
        if self.temperature is not None:
            config = config.with_temperature(self.temperature)
        if self.norm_type is not None:
            config = config.with_norm_type(self.norm_type)
        if self.attention_type is not None:
            config = config.with_attention_type(self.attention_type)
        if self.attention_type == AttentionType.MHA or self.attention_type == AttentionType.LINEAR:
            if self.n_heads is not None:
                config = config.with_num_heads(self.n_heads)
            if self.heads_by_layer is not None:
                config = config.with_heads_by_layer(self.heads_by_layer)
        if self.attention_type == AttentionType.FISH:
            if self.n_local_heads is not None:
                config = config.with_n_local_heads(self.n_local_heads)
            if self.n_global_heads is not None:
                config = config.with_n_global_heads(self.n_global_heads)
            if self.global_heads_by_layer is not None:
                config = config.with_global_heads_by_layer(self.global_heads_by_layer)
            if self.local_heads_by_layer is not None:
                config = config.with_local_heads_by_layer(self.local_heads_by_layer)
        if self.residual_type is not None:
            config = config.with_residual_type(self.residual_type)

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
        if self.pct_start is not None:
            config = config.with_pct_start(self.pct_start)
        if self.div_factor is not None:
            config = config.with_div_factor(self.div_factor)
        if self.three_phase is not None:
            config = config.with_three_phase(self.three_phase)
        if self.max_momentum is not None:
            config = config.with_max_momentum(self.max_momentum)
        if self.final_div_factor is not None:
            config = config.with_final_div_factor(self.final_div_factor)
        if self.base_momentum is not None:
            config = config.with_base_momentum(self.base_momentum)
        if self.last_effective_batch_num is not None:
            config = config.with_last_effective_batch_num(self.last_effective_batch_num)
        if self.cycle_momentum is not None:
            config = config.with_cycle_momentum(self.cycle_momentum)
        if self.anneal_strategy is not None:
            config = config.with_anneal_strategy(self.anneal_strategy)

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
        if self.loss_function is None:
            raise AttributeError("loss_function is not defined for LossConfig")
        if self.loss_reduction is None:
            raise AttributeError("loss_reduction is not defined for LossConfig")
        if self.torch_device is None:
            raise AttributeError("torch_device is not defined for LossConfig")

        config = LossConfig(self.loss_function, self.loss_reduction, torch.device(self.torch_device))

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

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Consider setting --name to train a new model.")
            raise

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
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist - Inference requires a trained model")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        hparams: Dict[str, Any] = checkpoint["hyperparameters"]
        for key, value in hparams.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_state_dict = checkpoint["model_state_dict"]
        print(f"Successfully loaded model {self.name} for inference")
        del checkpoint
        return self
