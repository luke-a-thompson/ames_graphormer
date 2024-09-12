from typing import Optional
import torch
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader
from graphormer.config.hparams import HyperparameterConfig
from tqdm import tqdm


def save_checkpoint(
    epoch: int,
    hparams: HyperparameterConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: _Loss,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    suffix: Optional[str] = None,
) -> None:
    """
    Save the model weights, optimizer state, and other necessary information to a checkpoint file.

    Args:
        epoch (int): The current epoch index.
        hparams (HyperparameterConfig): The hyperparameters for the training session
        model (torch.nn.Module): The model whose weights need to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss (torch.nn.modules.loss._Loss): The loss function used for training the model
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The learning rate scheduler used during training. Defaults to None.

    Returns:
        None
    """

    import os

    if not os.path.exists(hparams.checkpoint_dir):
        os.makedirs(hparams.checkpoint_dir)

    checkpoint = {
        "epoch": epoch,
        "hyperparameters": vars(hparams),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "loss_state_dict": loss.state_dict(),
    }

    name = hparams.name
    if suffix is not None:
        name = f"{hparams.name}_{suffix}"

    try:
        torch.save(checkpoint, f"{hparams.checkpoint_dir}/{name}.pt")
        print(f"Checkpoint successfully saved to {hparams.checkpoint_dir}/{name}.pt")
    except Exception as e:
        print(f"Failed to save {hparams.checkpoint_dir}/{name}. Error: {e}")


def model_init_print(
    hparams: HyperparameterConfig,
    model: Optional[torch.nn.Module] = None,
    train_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None,
):
    """
    Display an overview of the training parameters.

    Args:
        parameters_dict (Dict[str, int | float]): A dictionary containing the training parameters.

    Returns:
        None
    """

    from torchinfo import summary

    if model:
        summary(model)

    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(
        show_header=True,
        header_style="green",
        box=box.MINIMAL,
    )
    table.add_column(
        "Parameter",
        style="dim",
        overflow="fold",
    )
    table.add_column("Value", overflow="fold")

    if train_dataloader is not None:
        table.add_row("Train Dataset Size", str(len(train_dataloader.dataset)))  # type: ignore
    if test_dataloader is not None:
        table.add_row("Validation Dataset Size", str(len(test_dataloader.dataset)))  # type: ignore

    for name, value in vars(hparams).items():
        table.add_row(name, str(value))

    console.print(table)


def calculate_pos_weight(loader: DataLoader):
    num_neg_samples = 0
    num_pos_samples = 0
    print("Calculating positive weight...")
    for sample in tqdm(loader):
        num_pos_samples += int(torch.sum(sample.y).item())
        num_neg_samples += int(torch.sum(sample.y == 0).item())
    return torch.tensor([num_neg_samples / num_pos_samples])
