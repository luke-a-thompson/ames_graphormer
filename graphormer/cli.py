
import click
import tomllib
import torch
from torch_geometric.loader import DataLoader

from graphormer.config.hparams import HyperparameterConfig
from graphormer.config.options import LossReductionType, OptimizerType, SchedulerType, DatasetType
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


<<<<<<< HEAD
def create_loaders(
    dataset: InMemoryDataset, test_size: float, batch_size: int, random_state: int, ames_dataset: str | None = None
) -> Tuple[DataLoader, DataLoader]:
    dataloader_optimization_params = {
        "pin_memory": True,
        "num_workers": 4,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }
    if ames_dataset == "Honma":
        train_loader = DataLoader(dataset[:12140], batch_size=batch_size, shuffle=True, **dataloader_optimization_params) #type: ignore
        test_loader = DataLoader(dataset[12140:], batch_size=batch_size, **dataloader_optimization_params)# type: ignore
    else:
        test_ids, train_ids = train_test_split(
            range(len(dataset)), test_size=test_size, random_state=random_state
        )
        train_loader = DataLoader(
            Subset(dataset, train_ids),  # type: ignore
            batch_size=batch_size,
            shuffle=True,
            **dataloader_optimization_params,
        )
        test_loader = DataLoader(
            Subset(dataset, test_ids),  # type: ignore
            batch_size=batch_size,
            **dataloader_optimization_params,
        )
    return train_loader, test_loader


def calculate_pos_weight(loader: DataLoader):
    num_neg_samples = 0
    num_pos_samples = 0
    for sample in loader:
        num_pos_samples += torch.sum(sample.y).item()
        num_neg_samples += torch.sum(sample.y == 0).item()
    return torch.tensor([num_neg_samples / num_pos_samples])


def get_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    epochs: int,
    lr_power: float,
    lr_factor: float,
    lr_min: float,
    lr_max: float,
    lr_cooldown: int,
    lr_patience: int,
    lr_warmup: int,
    lr_smooth: bool,
    lr_window: int,
    lr_reset: int,
) -> LRScheduler:
    scheduler = None
    if scheduler_type == Scheduler.POLYNOMIAL.value:
        scheduler = PolynomialLR(optimizer, total_iters=epochs, power=lr_power)
    elif scheduler_type == Scheduler.GREEDY.value:
        scheduler = GreedyLR(
            optimizer,
            factor=lr_factor,
            min_lr=lr_min,
            max_lr=lr_max,
            cooldown=lr_cooldown,
            patience=lr_patience,
            warmup=lr_warmup,
            smooth=lr_smooth,
            window_size=lr_window,
            reset=lr_reset,
        )
    elif scheduler_type == Scheduler.PLATEAU.value:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=lr_patience, cooldown=lr_cooldown, min_lr=lr_min)
    assert scheduler is not None
    return scheduler


def load_from_checkpoint(
    name: str,
    dataset: InMemoryDataset,
    scheduler_type: str,
    epochs: int,
    lr_params: Dict,
    optimizer_params: Dict,
    test_size: float,
    batch_size: int,
    device,
) -> Tuple[torch.nn.Module, Optimizer, LRScheduler, DataLoader, DataLoader, _Loss, int]:
    checkpoint = torch.load(
        f"pretrained_models/{name}.pt", map_location=device)
    hparams = checkpoint["hyperparameters"]
    model = Graphormer(
        **hparams,
        node_feature_dim=dataset.num_node_features,
        edge_feature_dim=dataset.num_edge_features,
        output_dim=dataset[0].y.shape[0],  # type: ignore
    )
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    scheduler = get_scheduler(scheduler_type, optimizer, epochs, **lr_params)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    random_state = checkpoint["random_state"]
    assert random_state is not None
    train_loader, test_loader = create_loaders(
        dataset, test_size, batch_size, random_state)

    pos_weight = calculate_pos_weight(train_loader).to(device)
    loss_function = nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=pos_weight)
    loss_function.load_state_dict(checkpoint["loss_state_dict"])

    print(f"Successfully loaded model {name} at epoch {start_epoch}")
    del checkpoint
    return model, optimizer, scheduler, train_loader, test_loader, loss_function, start_epoch

def should_step(batch_idx: int, accumulation_steps: int, train_batches_per_epoch: int) -> bool:
    if accumulation_steps <= 1:
        return True
    if batch_idx > 0 and (batch_idx + 1) % accumulation_steps == 0:
        return True
    if batch_idx >= train_batches_per_epoch - 1:
        return True
    return False
=======
>>>>>>> ea67f57 (feat: improve configuration)
