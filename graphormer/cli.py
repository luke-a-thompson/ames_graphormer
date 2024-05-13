
import click
import tomllib
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
import torch
from torch_geometric.loader import DataLoader
import optuna

from graphormer.config.hparams import HyperparameterConfig
from graphormer.config.options import LossReductionType, OptimizerType, SchedulerType, DatasetType
from graphormer.config.tuning_hparams import TuningHyperparameterConfig
from graphormer.model import Graphormer
from graphormer.train import train_model


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
    default = "default_hparams.toml"
)
@click.option("--datadir", default="data")
@click.option("--logdir", default="runs")
@click.option("--dataset", type=click.Choice(DatasetType, case_sensitive=False), default=DatasetType.HONMA) # type: ignore
@click.option("--num_layers", default=3)
@click.option("--hidden_dim", default=128)
@click.option("--edge_embedding_dim", default=128)
@click.option("--ffn_hidden_dim", default=80)
@click.option("--n_heads", default=4)
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
    type=click.Choice(SchedulerType, case_sensitive=False), # type: ignore
    default=SchedulerType.GREEDY,
)
@click.option("--optimizer_type", type=click.Choice(OptimizerType, case_sensitive=False), default=OptimizerType.ADAMW) # type: ignore
@click.option("--lr_patience", default=4)
@click.option("--lr_cooldown", default=2)
@click.option("--lr_min", default=1e-6)
@click.option("--lr_max", default=1e-3)
@click.option("--lr_warmup", default=2)
@click.option("--lr_smooth", default=True)
@click.option("--lr_window", default=10)
@click.option("--lr_reset", default=0)
@click.option("--lr_factor", default=0.5)
@click.option("--name", default=None)
@click.option("--checkpt_save_interval", default=5)
@click.option("--accumulation_steps", default=1)
@click.option("--loss_reduction", type=click.Choice(LossReductionType, case_sensitive=False), default=LossReductionType.MEAN) #type: ignore
@click.option("--checkpoint_dir", default="pretrained_models")
@click.option("--dropout", default=0.05)
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
    default = "default_tuning_hparams.toml"
)
@click.option("--datadir", default="data")
@click.option("--logdir", default="optuna_runs")
@click.option("--dataset", type=click.Choice(DatasetType, case_sensitive=False), default=DatasetType.HONMA) # type: ignore
@click.option("--num_layers", default=3)
@click.option("--hidden_dim", default=128)
@click.option("--edge_embedding_dim", default=128)
@click.option("--ffn_hidden_dim", default=80)
@click.option("--n_heads", default=4)
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
@click.option("--tune_size", default=0.25)
def tune(**kwargs):
    hparam_config = TuningHyperparameterConfig(**kwargs)
    study = optuna.create_study(
        direction="minimize",
        study_name=hparam_config.study_name, 
        storage="sqlite:///db.sqlite3", 
        sampler=TPESampler(),
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    def objective(trial: Trial) -> float:
        trial_hparams = hparam_config.create_hyperparameters(trial)
        return train_model(trial_hparams, trial)
    study.optimize(objective, n_trials=hparam_config.n_trials)
    print(f"Best value: {study.best_value} (params: {study.best_params})")



@click.command()
@click.option("--data", default="data")
@click.option("--ames_dataset", default="Honma")
@click.option("--max_path_distance", default=5)
@click.option("--test_size", default=0.2)
@click.option("--monte_carlo_dropout", default=False)
@click.option("--state_dict", default="pretrained_models/Graphormer_checkpoint-1_15-04-24.pt")
@click.option("--random_state", default=42)
@click.option("--batch_size", default=4)
@click.option("--torch_device", default="cuda")
def inference(
    data: str,
    ames_dataset: str,
    max_path_distance: int,
    test_size: float,
    monte_carlo_dropout: bool,
    state_dict: str,
    random_state: int,
    batch_size: int,
    torch_device: str,
) -> torch.Tensor:
    state_dict = torch.load(state_dict)

    if ames_dataset == "Honma":
        from data.data_cleaning import HonmaDataset
        dataset = HonmaDataset(data, max_distance=max_path_distance)
        dataset = dataset[12140:]
    elif ames_dataset == "Hansen":
        from data.data_cleaning import HansenDataset
        dataset = HansenDataset(data, max_distance=max_path_distance)
        raise NotImplementedError("Hansen dataset not implemented yet")
    else:
        raise ValueError(f"Unknown dataset {data}")

    device = torch.device(torch_device)
    model = Graphormer(
        node_feature_dim=dataset.num_node_features,
        edge_feature_dim=dataset.num_edge_features,
        output_dim=dataset[0].y.shape[0],
        **state_dict["hyperparameters"],
    )
    Graphormer.load_state_dict(state_dict["state_dict"], strict=False)
    torch.manual_seed(random_state)

    inference_loader = DataLoader(dataset, batch_size, device)

    if not monte_carlo_dropout:
        model.eval()

        with torch.no_grad:
            output = model(inference_loader)
    else:
        from utils import monte_carlo_dropout

        model.eval()
        model.apply(monte_carlo_dropout)

        raise NotImplementedError("Monte Carlo Dropout not implemented yet")

    return output
