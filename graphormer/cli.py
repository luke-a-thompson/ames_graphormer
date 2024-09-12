import click
from typing import Optional
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
import torch
import optuna

from graphormer.config.hparams import HyperparameterConfig, hyperparameters
from graphormer.config.options import (
    OptimizerType,
    DatasetRegime,
)
from graphormer.config.tuning_hparams import TuningHyperparameterConfig, tuning_hyperparameters
from graphormer.inference import inference_model
from graphormer.optimise_temperature import optimise_temperature
from graphormer.results import save_results, friedman_from_bac_csv
from graphormer.train import Trainer


@click.command()
@hyperparameters
def train(**kwargs):
    hparam_config = HyperparameterConfig(**kwargs)
    hparam_config.load_from_checkpoint()
    hparam_config.epochs = kwargs.get("epochs", hparam_config.epochs)  # Allow max epochs override to extend training
    torch.manual_seed(hparam_config.random_state)
    trainer = Trainer.build(hparam_config)
    trainer.fit()


@click.command()
@tuning_hyperparameters
def tune(**kwargs):
    hparam_config = TuningHyperparameterConfig(**kwargs)
    study = optuna.create_study(
        direction="minimize",
        study_name=hparam_config.study_name,
        storage="sqlite:///db.sqlite3",
        sampler=TPESampler(),
        pruner=HyperbandPruner(max_resource=60),
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
        trainer = Trainer.build(trial_hparams)
        return trainer.fit()

    study.optimize(objective, n_trials=hparam_config.n_trials)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


@click.command()
@hyperparameters
@click.option("--checkpoint_dir", default="pretrained_models")
@click.option("--mc_samples", default=None, type=click.INT)
@click.option("--mc_dropout_rate", default=0.1, type=click.FLOAT)
def inference(mc_samples: Optional[int], mc_dropout_rate: Optional[float], **kwargs):
    hparam_config = HyperparameterConfig(**kwargs)
    hparam_config.load_for_inference()
    hparam_config.dataset_regime = DatasetRegime.TEST
    hparam_config.temperature = kwargs.get("temperature", hparam_config.temperature)
    torch.manual_seed(hparam_config.random_state)
    results = inference_model(hparam_config, mc_samples=mc_samples, mc_dropout_rate=mc_dropout_rate)

    save_results(results, hparam_config.name, mc_samples)


@click.command()
@hyperparameters
@click.option("--max_iter", default=50, type=click.INT)
def tune_temperature(max_iter: int, **kwargs):
    hparam_config = HyperparameterConfig(**kwargs)
    hparam_config.dataset_regime = DatasetRegime.TRAIN  # We optimise temperature on the validation set
    hparam_config.load_for_inference()
    print(hparam_config)
    optimise_temperature(hparam_config, max_iter=max_iter)


@click.command()
@hyperparameters
def estimate_noise_scale(**kwargs):
    hparam_config = HyperparameterConfig(**kwargs)
    hparam_config.load_from_checkpoint()
    torch.manual_seed(hparam_config.random_state)


# # Example: poetry run analyze --models results,results2,results3
# @click.command()
# @click.option("--bac_csv_path", type=click.Path(exists=True), default="results/MC_BACs.csv")
# @click.option("--models", type=click.STRING, callback=lambda ctx, param, value: value.split(","), required=True)
# @click.option("--alpha", default=0.05)
# def analyze(bac_csv_path: Path, models: List[str], alpha: float):
#     assert len(models) >= 3, "The Friedman test requires at least 3 models to compare."
#     friedman_from_bac_csv(bac_csv_path, models, alpha)
