from typing import List, Optional, Tuple
import click
from random import random
from graphormer.config.data import DataConfig
from graphormer.config.hparams import HyperparameterConfig, hyperparameters
from graphormer.config.options import (
    AttentionType,
    DatasetType,
    DatasetRegime,
    LossReductionType,
    OptimizerType,
    ResidualType,
    SchedulerType,
    NormType,
    LossFunction,
)
from optuna.trial import Trial

from graphormer.utils import create_composite_decorator, configure

tuning_hyperparameters = create_composite_decorator(
    hyperparameters,
    click.option(
        "-c",
        "--config",
        type=click.Path(dir_okay=False),
        is_eager=True,
        expose_value=False,
        help="Read option values from the specified config file",
        callback=configure,
        default="hparams/default_tuning_hparams.toml",
    ),
    click.option("--logdir", default="optuna_runs"),
    click.option("--max_in_degree", default=5),
    click.option("--max_out_degree", default=5),
    click.option("--max_path_distance", default=5),
    click.option("--min_b1", default=0.7),
    click.option("--max_b1", default=0.99),
    click.option("--min_b2", default=0.99),
    click.option("--max_b2", default=0.9999),
    click.option("--min_weight_decay", default=0.0),
    click.option("--max_weight_decay", default=1e-3),
    click.option("--min_eps", default=1e-10),
    click.option("--max_eps", default=1e-7),
    click.option("--min_momentum", default=0.0),
    click.option("--max_momentum", default=1.0),
    click.option("--min_max_momentum", default=0.9),
    click.option("--max_max_momentum", default=0.99),
    click.option("--min_base_momentum", default=0.1),
    click.option("--max_base_momentum", default=1.0),
    click.option("--min_dampening", default=0.0),
    click.option("--max_dampening", default=1.0),
    click.option("--min_clip_grad_norm", default=1.0),
    click.option("--max_clip_grad_norm", default=7.0),
    click.option("--min_lr_power", default=0.1),
    click.option("--max_lr_power", default=0.9),
    click.option("--min_lr_patience", default=0),
    click.option("--max_lr_patience", default=5),
    click.option("--min_lr_cooldown", default=0),
    click.option("--max_lr_cooldown", default=3),
    click.option("--min_lr_min", default=1e-7),
    click.option("--max_lr_min", default=3e-5),
    click.option("--min_lr_max", default=1e-4),
    click.option("--max_lr_max", default=1e-2),
    click.option("--min_lr_warmup", default=0),
    click.option("--max_lr_warmup", default=3),
    click.option("--min_lr_window", default=1),
    click.option("--max_lr_window", default=5),
    click.option("--min_lr_reset", default=0),
    click.option("--max_lr_reset", default=7),
    click.option("--min_lr_factor", default=0.1),
    click.option("--max_lr_factor", default=0.9),
    click.option("--min_pct_start", default=0.1),
    click.option("--max_pct_start", default=0.9),
    click.option("--min_div_factor", default=1.0),
    click.option("--max_div_factor", default=1e2),
    click.option("--min_final_div_factor", default=1.0),
    click.option("--max_final_div_factor", default=1e6),
    click.option("--study_name", default=None),
    click.option("--min_dropout", default=0.0),
    click.option("--max_dropout", default=0.5),
    click.option("--tune_size", default=0.25),
    click.option("--n_trials", default=10000),
)


class TuningHyperparameterConfig:
    def __init__(
        self,
        # Tuning Parameters
        study_name: str,
        random_state: int = int(random() * 1e9),
        torch_device: str = "cuda",
        checkpoint_dir: str = "optuna_models",
        n_trials: Optional[int] = None,
        # Data Parameters
        datadir: Optional[str] = None,
        dataset: Optional[DatasetType] = None,
        dataset_regime: Optional[DatasetRegime] = None,
        node_feature_dim: Optional[int] = None,
        edge_feature_dim: Optional[int] = None,
        max_path_distance: Optional[int] = None,
        batch_size: Optional[int] = None,
        test_size: Optional[float] = None,
        tune_size: Optional[float] = None,
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
        output_dim: Optional[int] = None,
        max_dropout: Optional[float] = None,
        min_dropout: Optional[float] = None,
        norm_type: Optional[NormType] = None,
        attention_type: Optional[AttentionType] = None,
        residual_type: Optional[ResidualType] = None,
        # Optimizer Parameters
        optimizer_type: Optional[OptimizerType] = None,
        loss_reduction_type: Optional[LossReductionType] = None,
        lr: Optional[float] = None,
        b1: Optional[float] = None,
        b2: Optional[float] = None,
        max_b1: Optional[float] = None,
        min_b1: Optional[float] = None,
        max_b2: Optional[float] = None,
        min_b2: Optional[float] = None,
        weight_decay: Optional[float] = None,
        max_weight_decay: Optional[float] = None,
        min_weight_decay: Optional[float] = None,
        eps: Optional[float] = None,
        max_eps: Optional[float] = None,
        min_eps: Optional[float] = None,
        clip_grad_norm: float = 3.0,
        max_clip_grad_norm: float = 5.0,
        min_clip_grad_norm: float = 1.0,
        momentum: Optional[float] = None,
        max_momentum: Optional[float] = None,
        min_momentum: Optional[float] = None,
        dampening: Optional[float] = None,
        max_dampening: Optional[float] = None,
        min_dampening: Optional[float] = None,
        nesterov: Optional[float] = None,
        loss_function: Optional[LossFunction | tuple[LossFunction, ...]] = None,
        loss_weights: Optional[Tuple[float, ...]] = None,
        # Scheduler Parameters
        scheduler_type: Optional[SchedulerType] = None,
        lr_power: Optional[float] = None,
        lr_min: Optional[float] = None,
        lr_max: Optional[float] = None,
        max_lr_power: Optional[float] = None,
        min_lr_power: Optional[float] = None,
        lr_patience: Optional[int] = None,
        max_lr_patience: Optional[int] = None,
        min_lr_patience: Optional[int] = None,
        lr_cooldown: Optional[int] = None,
        max_lr_cooldown: Optional[int] = None,
        min_lr_cooldown: Optional[int] = None,
        max_lr_min: Optional[float] = None,
        min_lr_min: Optional[float] = None,
        max_lr_max: Optional[float] = None,
        min_lr_max: Optional[float] = None,
        lr_warmup: Optional[int] = None,
        lr_smooth: Optional[bool] = None,
        max_lr_warmup: Optional[int] = None,
        min_lr_warmup: Optional[int] = None,
        lr_window: Optional[int] = None,
        max_lr_window: Optional[int] = None,
        min_lr_window: Optional[int] = None,
        lr_reset: Optional[float] = None,
        max_lr_reset: Optional[int] = None,
        min_lr_reset: Optional[int] = None,
        lr_factor: Optional[float] = None,
        max_lr_factor: Optional[float] = None,
        min_lr_factor: Optional[float] = None,
        pct_start: Optional[float] = None,
        min_pct_start: Optional[float] = None,
        max_pct_start: Optional[float] = None,
        div_factor: Optional[float] = None,
        min_div_factor: Optional[float] = None,
        max_div_factor: Optional[float] = None,
        final_div_factor: Optional[float] = None,
        min_final_div_factor: Optional[float] = None,
        max_final_div_factor: Optional[float] = None,
        cycle_momentum: Optional[bool] = None,
        three_phase: Optional[bool] = None,
        min_max_momentum: Optional[float] = None,
        max_max_momentum: Optional[float] = None,
        base_momentum: Optional[float | List[float]] = None,
        min_base_momentum: Optional[float] = None,
        max_base_momentum: Optional[float] = None,
        anneal_strategy: Optional[str] = None,
        last_effective_batch_num: Optional[int] = None,
        # Logging Parameters
        logdir: Optional[str] = None,
        # Training Parameters
        epochs: int = 40,
        accumulation_steps: Optional[int] = None,
        **kwargs,
    ):
        self.n_trials = n_trials
        self.study_name = study_name
        self.random_state = random_state
        self.torch_device = torch_device
        self.checkpoint_dir = checkpoint_dir
        self.datadir = datadir
        self.dataset = dataset
        self.dataset_regime = dataset_regime
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.max_path_distance = max_path_distance
        self.batch_size = batch_size
        self.test_size = test_size
        self.tune_size = tune_size

        # Model Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.n_heads = n_heads
        self.heads_by_layer = heads_by_layer
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.output_dim = output_dim
        self.max_dropout = max_dropout
        self.min_dropout = min_dropout
        self.norm_type = norm_type
        self.attention_type = attention_type
        self.n_global_heads = n_global_heads
        self.n_local_heads = n_local_heads
        self.global_heads_by_layer = global_heads_by_layer
        self.local_heads_by_layer = local_heads_by_layer
        self.residual_type = residual_type

        # Optimizer Parameters
        self.optimizer_type = optimizer_type
        self.loss_reduction_type = loss_reduction_type
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.max_b1 = max_b1
        self.min_b1 = min_b1
        self.max_b2 = max_b2
        self.min_b2 = min_b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_weight_decay = max_weight_decay
        self.min_weight_decay = min_weight_decay
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.clip_grad_norm = clip_grad_norm
        self.max_clip_grad_norm = max_clip_grad_norm
        self.min_clip_grad_norm = min_clip_grad_norm
        self.momentum = momentum
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        self.dampening = dampening
        self.max_dampening = max_dampening
        self.min_dampening = min_dampening
        self.nesterov = nesterov

        self.loss_function = loss_function
        self.loss_weights = loss_weights

        # Scheduler Parameters
        self.scheduler_type = scheduler_type
        self.lr_power = lr_power
        self.max_lr_power = max_lr_power
        self.min_lr_power = min_lr_power
        self.lr_patience = lr_patience
        self.max_lr_patience = max_lr_patience
        self.min_lr_patience = min_lr_patience
        self.lr_cooldown = lr_cooldown
        self.max_lr_cooldown = max_lr_cooldown
        self.min_lr_cooldown = min_lr_cooldown
        self.lr_min = lr_min
        self.max_lr_min = max_lr_min
        self.min_lr_min = min_lr_min
        self.lr_max = lr_max
        self.max_lr_max = max_lr_max
        self.min_lr_max = min_lr_max
        self.lr_warmup = lr_warmup
        self.max_lr_warmup = max_lr_warmup
        self.min_lr_warmup = min_lr_warmup
        self.lr_smooth = lr_smooth
        self.lr_window = lr_window
        self.max_lr_window = max_lr_window
        self.min_lr_window = min_lr_window
        self.lr_reset = lr_reset
        self.max_lr_reset = max_lr_reset
        self.min_lr_reset = min_lr_reset
        self.lr_factor = lr_factor
        self.max_lr_factor = max_lr_factor
        self.min_lr_factor = min_lr_factor
        self.pct_start = pct_start
        self.min_pct_start = min_pct_start
        self.max_pct_start = max_pct_start
        self.div_factor = div_factor
        self.min_div_factor = min_div_factor
        self.max_div_factor = max_div_factor
        self.final_div_factor = final_div_factor
        self.min_final_div_factor = min_final_div_factor
        self.max_final_div_factor = max_final_div_factor
        self.cycle_momentum = cycle_momentum
        self.three_phase = three_phase
        self.min_max_momentum = min_max_momentum
        self.max_max_momentum = max_max_momentum
        self.base_momentum = base_momentum
        self.min_base_momentum = min_base_momentum
        self.max_base_momentum = max_base_momentum
        self.anneal_strategy = anneal_strategy
        self.last_effective_batch_num = last_effective_batch_num

        # Logging Parameters
        self.logdir = logdir

        # Training Parameters
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps

    def create_hyperparameters(self, trial: Trial) -> HyperparameterConfig:
        if self.batch_size is None:
            raise AttributeError("batch_size not defined for TuningHyperparameterConfig")
        if self.min_dropout is None:
            raise AttributeError("min_dropout not defined for TuningHyperparameterConfig")
        if self.max_dropout is None:
            raise AttributeError("max_dropout not defined for TuningHyperparameterConfig")
        if self.lr is None:
            raise AttributeError("min_lr not defined for TuningHyperparameterConfig")
        if self.min_b1 is None:
            raise AttributeError("min_b1 not defined for TuningHyperparameterConfig")
        if self.max_b1 is None:
            raise AttributeError("max_b1 not defined for TuningHyperparameterConfig")
        if self.min_b2 is None:
            raise AttributeError("min_b2 not defined for TuningHyperparameterConfig")
        if self.max_b2 is None:
            raise AttributeError("max_b2 not defined for TuningHyperparameterConfig")
        if self.min_weight_decay is None:
            raise AttributeError("min_weight_decay not defined for TuningHyperparameterConfig")
        if self.max_weight_decay is None:
            raise AttributeError("max_weight_decay not defined for TuningHyperparameterConfig")
        if self.min_eps is None:
            raise AttributeError("min_eps not defined for TuningHyperparameterConfig")
        if self.max_eps is None:
            raise AttributeError("max_eps not defined for TuningHyperparameterConfig")
        if self.min_clip_grad_norm is None:
            raise AttributeError("min_clip_grad_norm not defined for TuningHyperparameterConfig")
        if self.max_clip_grad_norm is None:
            raise AttributeError("max_clip_grad_norm not defined for TuningHyperparameterConfig")
        if self.min_momentum is None:
            raise AttributeError("min_momentum not defined for TuningHyperparameterConfig")
        if self.max_momentum is None:
            raise AttributeError("max_momentum not defined for TuningHyperparameterConfig")
        if self.min_dampening is None:
            raise AttributeError("min_dampening not defined for TuningHyperparameterConfig")
        if self.max_dampening is None:
            raise AttributeError("max_dampening not defined for TuningHyperparameterConfig")
        if self.min_lr_power is None:
            raise AttributeError("min_lr_power not defined for TuningHyperparameterConfig")
        if self.max_lr_power is None:
            raise AttributeError("max_lr_power not defined for TuningHyperparameterConfig")
        if self.min_lr_patience is None:
            raise AttributeError("min_lr_patience not defined for TuningHyperparameterConfig")
        if self.max_lr_patience is None:
            raise AttributeError("max_lr_patience not defined for TuningHyperparameterConfig")
        if self.min_lr_cooldown is None:
            raise AttributeError("min_lr_cooldown not defined for TuningHyperparameterConfig")
        if self.max_lr_cooldown is None:
            raise AttributeError("max_lr_cooldown not defined for TuningHyperparameterConfig")
        if self.min_lr_min is None:
            raise AttributeError("min_lr_min not defined for TuningHyperparameterConfig")
        if self.max_lr_min is None:
            raise AttributeError("max_lr_min not defined for TuningHyperparameterConfig")
        if self.min_lr_max is None:
            raise AttributeError("min_lr_max not defined for TuningHyperparameterConfig")
        if self.max_lr_max is None:
            raise AttributeError("max_lr_max not defined for TuningHyperparameterConfig")
        if self.min_lr_warmup is None:
            raise AttributeError("min_lr_warmup not defined for TuningHyperparameterConfig")
        if self.max_lr_warmup is None:
            raise AttributeError("max_lr_warmup not defined for TuningHyperparameterConfig")
        if self.min_lr_window is None:
            raise AttributeError("min_lr_window not defined for TuningHyperparameterConfig")
        if self.max_lr_window is None:
            raise AttributeError("max_lr_window not defined for TuningHyperparameterConfig")
        if self.min_lr_reset is None:
            raise AttributeError("min_lr_reset not defined for TuningHyperparameterConfig")
        if self.max_lr_reset is None:
            raise AttributeError("max_lr_reset not defined for TuningHyperparameterConfig")
        if self.min_lr_factor is None:
            raise AttributeError("min_lr_factor not defined for TuningHyperparameterConfig")
        if self.max_lr_factor is None:
            raise AttributeError("max_lr_factor not defined for TuningHyperparameterConfig")
        if self.accumulation_steps is None:
            raise AttributeError("accumulation_steps not defined for TuningHyperparameterConfig")
        if self.tune_size is None:
            raise AttributeError("tune_size not defined for TuningHyperparameterConfig")
        if self.min_pct_start is None:
            raise AttributeError("min_pct_start not defined for TuningHyperparameterConfig")
        if self.max_pct_start is None:
            raise AttributeError("max_pct_start not defined for TuningHyperparameterConfig")
        if self.min_div_factor is None:
            raise AttributeError("min_div_factor not defined for TuningHyperparameterConfig")
        if self.max_div_factor is None:
            raise AttributeError("max_div_factor not defined for TuningHyperparameterConfig")
        if self.min_final_div_factor is None:
            raise AttributeError("min_final_div_factor not defined for TuningHyperparameterConfig")
        if self.max_final_div_factor is None:
            raise AttributeError("max_final_div_factor not defined for TuningHyperparameterConfig")
        if self.min_base_momentum is None:
            raise AttributeError("min_base_momentum not defined for TuningHyperparameterConfig")
        if self.max_base_momentum is None:
            raise AttributeError("max_base_momentum not defined for TuningHyperparameterConfig")
        if self.min_max_momentum is None:
            raise AttributeError("min_max_momentum not defined for TuningHyperparameterConfig")
        if self.max_max_momentum is None:
            raise AttributeError("max_max_momentum not defined for TuningHyperparameterConfig")

        if self.residual_type is None:
            self.residual_type = ResidualType(
                trial.suggest_categorical("residual_type", [ResidualType.PRENORM, ResidualType.REZERO])
            )
        if self.optimizer_type is None:
            self.optimizer_type = OptimizerType(
                trial.suggest_categorical("optimizer_type", [OptimizerType.SGD, OptimizerType.ADAMW])
            )
        if self.scheduler_type is None:
            self.scheduler_type = SchedulerType(
                trial.suggest_categorical(
                    "scheduler_type",
                    [
                        SchedulerType.GREEDY,
                        SchedulerType.POLYNOMIAL,
                        SchedulerType.PLATEAU,
                        SchedulerType.ONE_CYCLE,
                        SchedulerType.FIXED,
                    ],
                )
            )
        if self.loss_reduction_type is None:
            loss_reduction_type = LossReductionType(
                trial.suggest_categorical("loss_reduction", [LossReductionType.MEAN, LossReductionType.SUM])
            )

        if self.norm_type is None:
            self.norm_type = NormType(
                trial.suggest_categorical("norm_type", [NormType.LAYER, NormType.MAX, NormType.RMS, NormType.CRMS])
            )

        match self.optimizer_type:
            case OptimizerType.SGD:
                if self.momentum is None:
                    self.momentum = trial.suggest_float("momentum", self.min_momentum, self.max_momentum)

                if self.dampening is None:
                    self.dampening = trial.suggest_float("dampening", self.min_dampening, self.max_dampening)

                if self.nesterov is None:
                    self.nesterov = trial.suggest_categorical("nesterov", [True, False])
            case OptimizerType.ADAMW:
                if self.b1 is None:
                    self.b1 = trial.suggest_float("b1", self.min_b1, self.max_b1)
                if self.b2 is None:
                    self.b2 = trial.suggest_float("b2", self.min_b2, self.max_b2)
                if self.eps is None:
                    self.eps = trial.suggest_float("eps", self.min_eps, self.max_eps)

        match self.scheduler_type:
            case SchedulerType.GREEDY:
                self.lr_smooth = trial.suggest_categorical("lr_smooth", [True, False])
                self.lr_warmup = trial.suggest_int("lr_warmup", self.min_lr_warmup, self.max_lr_warmup)
                self.lr_cooldown = trial.suggest_int("lr_cooldown", self.min_lr_cooldown, self.max_lr_cooldown)
                self.lr_window = trial.suggest_int("lr_window", self.min_lr_window, self.max_lr_window)
                self.lr_patience = trial.suggest_int("lr_patience", self.min_lr_patience, self.max_lr_patience)
                self.lr_reset = trial.suggest_int("lr_reset", self.min_lr_reset, self.max_lr_reset)
                self.lr_max = trial.suggest_float("lr_max", self.min_lr_max, self.max_lr_max)
                self.lr_min = trial.suggest_float("lr_min", self.min_lr_min, self.max_lr_min)
                self.lr_factor = trial.suggest_float("lr_factor", self.min_lr_factor, self.max_lr_factor)
            case SchedulerType.PLATEAU:
                self.lr_min = trial.suggest_float("lr_min", self.min_lr_min, self.max_lr_min)
                self.lr_factor = trial.suggest_float("lr_factor", self.min_lr_factor, self.max_lr_factor)
                self.lr_patience = trial.suggest_int("lr_patience", self.min_lr_patience, self.max_lr_patience)
                self.lr_cooldown = trial.suggest_int("lr_cooldown", self.min_lr_cooldown, self.max_lr_cooldown)
            case SchedulerType.POLYNOMIAL:
                self.lr_power = trial.suggest_float("lr_power", self.min_lr_power, self.max_lr_power)
            case SchedulerType.ONE_CYCLE:
                self.pct_start = trial.suggest_float("pct_start", self.min_pct_start, self.max_pct_start)
                self.div_factor = trial.suggest_float("div_factor", self.min_div_factor, self.max_div_factor)
                self.final_div_factor = trial.suggest_float(
                    "final_div_factor", self.min_final_div_factor, self.max_final_div_factor
                )
                if self.cycle_momentum is None:
                    self.cycle_momentum = trial.suggest_categorical("cycle_momentum", [True, False])
                if self.three_phase is None:
                    self.three_phase = trial.suggest_categorical("three_phase", [True, False])
                self.max_momentum = trial.suggest_float("max_momentum", self.min_max_momentum, self.max_max_momentum)
                self.base_momentum = trial.suggest_float(
                    "base_momentum", self.min_base_momentum, self.max_base_momentum
                )
                if self.anneal_strategy is None:
                    self.anneal_strategy = trial.suggest_categorical("anneal_strategy", ["cos", "linear"])

        if self.weight_decay is None:
            self.weight_decay = trial.suggest_float("weight_decay", self.min_weight_decay, self.max_weight_decay)

        if self.clip_grad_norm is None:
            self.clip_grad_norm = trial.suggest_float(
                "clip_grad_norm", self.min_clip_grad_norm, self.max_clip_grad_norm
            )

        return HyperparameterConfig(
            datadir=self.datadir,
            dataset=self.dataset,
            dataset_regime=self.dataset_regime,
            random_state=self.random_state,
            torch_device=self.torch_device,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            test_size=self.test_size,
            tune_size=self.tune_size,
            logdir=self.logdir,
            start_epoch=0,
            max_path_distance=self.max_path_distance,
            # Model Parameters
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            edge_embedding_dim=self.edge_embedding_dim,
            ffn_hidden_dim=self.ffn_hidden_dim,
            n_heads=self.n_heads,
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            output_dim=self.output_dim,
            dropout=trial.suggest_float("dropout", self.min_dropout, self.max_dropout),
            batch_size=self.batch_size,
            norm_type=self.norm_type,
            attention_type=self.attention_type,
            heads_by_layer=self.heads_by_layer,
            n_global_heads=self.n_global_heads,
            n_local_heads=self.n_local_heads,
            global_heads_by_layer=self.global_heads_by_layer,
            local_heads_by_layer=self.local_heads_by_layer,
            residual_type=self.residual_type,
            # Optimizer Parameters
            optimizer_type=self.optimizer_type,
            lr=self.lr,
            b1=self.b1,
            b2=self.b2,
            weight_decay=self.weight_decay,
            eps=self.eps,
            clip_grad_norm=self.clip_grad_norm,
            momentum=self.momentum,
            dampening=self.dampening,
            nesterov=self.nesterov,
            loss_function=self.loss_function,
            loss_weights=self.loss_weights,
            # Scheduler Parameters
            scheduler_type=self.scheduler_type,
            last_effective_batch_num=self.last_effective_batch_num,
            lr_power=self.lr_power,
            lr_patience=self.lr_patience,
            lr_cooldown=self.lr_cooldown,
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            lr_warmup=self.lr_warmup,
            lr_window=self.lr_window,
            lr_reset=self.lr_reset,
            lr_factor=self.lr_factor,
            lr_smooth=self.lr_smooth,
            pct_start=self.pct_start,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            cycle_momentum=self.cycle_momentum,
            three_phase=self.three_phase,
            max_momentum=self.max_momentum,
            base_momentum=self.base_momentum,
            anneal_strategy=self.anneal_strategy,
            # Training Parameters
            epochs=40,
            accumulation_steps=self.accumulation_steps,
            loss_reduction=loss_reduction_type,
            checkpoint_dir=self.checkpoint_dir,
            checkpt_save_interval=1000,
            write_to_disk=False,
        )

    def data_config(self) -> DataConfig:
        if self.dataset is None:
            raise AttributeError("dataset not defined for DataConfig")
        if self.dataset_regime is None:
            raise AttributeError("dataset_regime not defined for DataConfig")
        if self.batch_size is None:
            raise AttributeError("batch_size not defined for DataConfig")
        if self.datadir is None:
            raise AttributeError("datadir not defined for DataConfig")
        if self.max_path_distance is None:
            raise AttributeError("max_path_distance not defined for DataConfig")

        config = DataConfig(self.dataset, self.dataset_regime, self.batch_size, self.datadir, self.max_path_distance)

        if self.test_size is not None:
            config = config.with_test_size(self.test_size)
        return config
