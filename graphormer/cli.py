import click
import tomllib
from typing import Optional, List
from pathlib import Path
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
import torch
import optuna

from graphormer.config.hparams import HyperparameterConfig
from graphormer.config.options import (
    AttentionType,
    LossReductionType,
    NormType,
    OptimizerType,
    SchedulerType,
    DatasetType,
)
from graphormer.config.tuning_hparams import TuningHyperparameterConfig
from graphormer.train import train_model
from graphormer.inference import inference_model
from graphormer.results import save_results, friedman_from_bac_csv


def configure(ctx, param, filename):
    with open(filename, "rb") as f:
        config = tomllib.load(f)
    ctx.default_map = config


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    is_eager=True,
    expose_value=False,
    help="Read option values from the specified config file",
    callback=configure,
    default="default_hparams.toml",
)
@click.option("--datadir", default="data")
@click.option("--logdir", default="runs")
@click.option("--dataset", type=click.Choice(DatasetType, case_sensitive=False), default=DatasetType.HONMA)  # type: ignore
@click.option("--num_layers", default=3)
@click.option("--hidden_dim", default=128)
@click.option("--edge_embedding_dim", default=128)
@click.option("--ffn_hidden_dim", default=80)
@click.option("--n_heads", default=4)
@click.option("--heads_by_layer", multiple=True, default=[], type=click.INT)
@click.option("--max_in_degree", default=5)
@click.option("--max_out_degree", default=5)
@click.option("--max_path_distance", default=5)
@click.option("--test_size", default=0.8)
@click.option("--random_state", default=None, type=click.INT)
@click.option("--batch_size", default=16)
@click.option("--lr", default=3e-4)
@click.option("--b1", default=0.9)
@click.option("--b2", default=0.999)
@click.option("--weight_decay", default=0.0)
@click.option("--eps", default=1e-8)
@click.option("--nesterov", default=False)
@click.option("--momentum", default=0.0)
@click.option("--dampening", default=0.0)
@click.option("--clip_grad_norm", default=5.0)
@click.option("--torch_device", default="cuda")
@click.option("--epochs", default=10)
@click.option("--lr_power", default=0.5)
@click.option(
    "--scheduler_type",
    type=click.Choice(SchedulerType, case_sensitive=False),  # type: ignore
    default=SchedulerType.GREEDY,
)
@click.option("--optimizer_type", type=click.Choice(OptimizerType, case_sensitive=False), default=OptimizerType.ADAMW)  # type: ignore
@click.option("--lr_patience", default=4)
@click.option("--lr_cooldown", default=2)
@click.option("--lr_min", default=1e-6)
@click.option("--lr_max", default=1e-3)
@click.option("--lr_warmup", default=2)
@click.option("--lr_smooth", default=True)
@click.option("--lr_window", default=10)
@click.option("--lr_reset", default=0)
@click.option("--lr_factor", default=0.5)
@click.option("--pct_start", default=0.3)
@click.option("--div_factor", default=25)
@click.option("--final_div_factor", default=1e4)
@click.option("--cycle_momentum", default=True)
@click.option("--three_phase", default=False)
@click.option("--max_momentum", default=0.95)
@click.option("--base_momentum", default=0.85)
@click.option("--last_effective_batch_num", default=-1)
@click.option("--anneal_strategy", default="cos")
@click.option("--name", default=None)
@click.option("--checkpt_save_interval", default=5)
@click.option("--accumulation_steps", default=1)
@click.option("--loss_reduction", type=click.Choice(LossReductionType, case_sensitive=False), default=LossReductionType.MEAN)  # type: ignore
@click.option("--checkpoint_dir", default="pretrained_models")
@click.option("--dropout", default=0.05)
@click.option("--norm_type", type=click.Choice(NormType, case_sensitive=False), default=NormType.LAYER)  # type: ignore
@click.option("--attention_type", type=click.Choice(AttentionType, case_sensitive=False), default=AttentionType.MHA)  # type: ignore
@click.option("--n_global_heads", default=4)
@click.option("--n_local_heads", default=8)
@click.option("--global_heads_by_layer", multiple=True, default=[], type=click.INT)
@click.option("--local_heads_by_layer", multiple=True, default=[], type=click.INT)
def train(**kwargs):
    hparam_config = HyperparameterConfig(**kwargs)
    hparam_config.load_from_checkpoint()
    torch.manual_seed(hparam_config.random_state)
    train_model(hparam_config)


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    is_eager=True,
    expose_value=False,
    help="Read option values from the specified config file",
    callback=configure,
    default="default_tuning_hparams.toml",
)
@click.option("--datadir", default="data")
@click.option("--logdir", default="optuna_runs")
@click.option("--dataset", type=click.Choice(DatasetType, case_sensitive=False), default=DatasetType.HONMA)  # type: ignore
@click.option("--num_layers", default=3)
@click.option("--hidden_dim", default=128)
@click.option("--edge_embedding_dim", default=128)
@click.option("--ffn_hidden_dim", default=80)
@click.option("--n_heads", default=None)
@click.option("--heads_by_layer", multiple=True, default=[], type=click.INT)
@click.option("--max_in_degree", default=5)
@click.option("--max_out_degree", default=5)
@click.option("--max_path_distance", default=5)
@click.option("--test_size", default=0.8)
@click.option("--random_state", default=None, type=click.INT)
@click.option("--batch_size", default=128)
@click.option("--lr", default=2.5e-3)
@click.option("--min_b1", default=0.7)
@click.option("--max_b1", default=0.99)
@click.option("--min_b2", default=0.99)
@click.option("--max_b2", default=0.9999)
@click.option("--min_weight_decay", default=0.0)
@click.option("--max_weight_decay", default=1e-3)
@click.option("--min_eps", default=1e-10)
@click.option("--max_eps", default=1e-7)
@click.option("--min_momentum", default=0.0)
@click.option("--max_momentum", default=1.0)
@click.option("--min_dampening", default=0.0)
@click.option("--max_dampening", default=1.0)
@click.option("--min_clip_grad_norm", default=1.0)
@click.option("--max_clip_grad_norm", default=7.0)
@click.option("--torch_device", default="cuda")
@click.option("--epochs", default=20)
@click.option("--min_lr_power", default=0.1)
@click.option("--max_lr_power", default=0.9)
@click.option("--min_lr_patience", default=0)
@click.option("--max_lr_patience", default=5)
@click.option("--min_lr_cooldown", default=0)
@click.option("--max_lr_cooldown", default=3)
@click.option("--min_lr_min", default=1e-7)
@click.option("--max_lr_min", default=3e-5)
@click.option("--min_lr_max", default=1e-4)
@click.option("--max_lr_max", default=1e-2)
@click.option("--min_lr_warmup", default=0)
@click.option("--max_lr_warmup", default=3)
@click.option("--min_lr_window", default=1)
@click.option("--max_lr_window", default=5)
@click.option("--min_lr_reset", default=0)
@click.option("--max_lr_reset", default=7)
@click.option("--min_lr_factor", default=0.1)
@click.option("--max_lr_factor", default=0.9)
@click.option("--study_name", default=None)
@click.option("--accumulation_steps", default=1)
@click.option("--checkpoint_dir", default="optuna_models")
@click.option("--min_dropout", default=0.0)
@click.option("--max_dropout", default=0.5)
@click.option("--norm_type", type=click.Choice(NormType, case_sensitive=False), default=None)  # type: ignore
@click.option("--tune_size", default=0.25)
@click.option("--n_trials", default=10000)
@click.option("--optimizer_type", type=click.Choice(OptimizerType, case_sensitive=False), default=None)  # type: ignore
@click.option(
    "--scheduler_type",
    type=click.Choice(SchedulerType, case_sensitive=False),  # type: ignore
    default=None,
)
@click.option("--loss_reduction_type", type=click.Choice(LossReductionType, case_sensitive=False), default=None)  # type: ignore
@click.option("--attention_type", type=click.Choice(AttentionType, case_sensitive=False), default=None)  # type: ignore
def tune(**kwargs):
    hparam_config = TuningHyperparameterConfig(**kwargs)
    data_config = hparam_config.data_config()
    train_loader, test_loader = data_config.build()
    study = optuna.create_study(
        direction="minimize",
        study_name=hparam_config.study_name,
        storage="sqlite:///db.sqlite3",
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )

    starting_points = []

    match hparam_config.optimizer_type:
        case OptimizerType.SGD:
            # Baseline
            starting_points.append(
                {
                    "nesterov": False,
                    "momentum": 0.0,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                }
            )
            starting_points.append(
                {
                    "nesterov": False,
                    "momentum": 0.0,
                    "dampening": 0.0,
                    "weight_decay": 0.0001,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                }
            )
            starting_points.append(
                {
                    "nesterov": False,
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                },
            )
            starting_points.append(
                {
                    "nesterov": False,
                    "momentum": 0.9,
                    "dampening": 0.1,
                    "weight_decay": 0.0,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                },
            )
            starting_points.append(
                {
                    "nesterov": True,
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                },
            )
            starting_points.append(
                {
                    "nesterov": True,
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "weight_decay": 0.0001,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                },
            )
        case OptimizerType.ADAMW:
            # Baseline
            starting_points.append(
                {
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                }
            )
            starting_points.append(
                {
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                    "dropout": 0.1,
                    "clip_grad_norm": 5.0,
                }
            )
            # Discovered good parameters
            starting_points.append(
                {
                    "b1": 0.867,
                    "b2": 0.9977,
                    "eps": 1e-09,
                    "dropout": 0.0848,
                    "weight_decay": 0.066,
                    "clip_grad_norm": 3.0767,
                }
            )
            starting_points.append(
                {
                    "b1": 0.8255,
                    "b2": 0.99755,
                    "eps": 9.0837e-08,
                    "dropout": 0.26312,
                    "weight_decay": 0.01895,
                    "clip_grad_norm": 3.20889,
                }
            )

    for starting_params in starting_points:
        study.enqueue_trial(starting_params, skip_if_exists=True)

    def objective(trial: Trial) -> float:
        trial_hparams = hparam_config.create_hyperparameters(trial)
        return train_model(trial_hparams, trial, train_loader, test_loader, data_config)

    study.optimize(objective, n_trials=hparam_config.n_trials)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


@click.command()
@click.option("--datadir", default="data")
@click.option("--dataset", type=click.Choice(DatasetType, case_sensitive=False), default=DatasetType.HONMA)  # type: ignore
@click.option("--name", default=None)
@click.option("--checkpoint_dir", default="pretrained_models")
@click.option("--mc_samples", default=None, type=click.INT)
@click.option("--max_path_distance", default=5)
@click.option("--test_size", default=0.2)
@click.option("--random_state", default=42)
@click.option("--batch_size", default=4)
@click.option("--torch_device", default="cuda")
def inference(mc_samples: Optional[int], **kwargs):
    hparam_config = HyperparameterConfig(**kwargs)
    hparam_config.load_for_inference()
    print(hparam_config)
    torch.manual_seed(hparam_config.random_state)
    results = inference_model(hparam_config, mc_samples=mc_samples)

    mc_dropout = mc_samples is not None
    save_results(results, hparam_config.name, mc_dropout)


# Example: poetry run analyze --models results,results2,results3
@click.command()
@click.option("--bac_csv_path", type=click.Path(exists=True), default="results/MC_BACs.csv")
@click.option("--models", type=click.STRING, callback=lambda ctx, param, value: value.split(","), required=True)
@click.option("--alpha", default=0.05)
def analyze(bac_csv_path: Path, models: List[str], alpha: float):
    assert len(models) >= 3, "The Friedman test requires at least 3 models to compare."
    friedman_from_bac_csv(bac_csv_path, models, alpha)
