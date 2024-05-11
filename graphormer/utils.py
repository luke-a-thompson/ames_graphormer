from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


def monte_carlo_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def difference_idxs(a, b, epsilon=1e-6) -> torch.Tensor:
    differences = torch.abs(a - b)
    indices = torch.nonzero(differences > epsilon)

    for idx in indices:
        idx_tuple = tuple(idx.tolist())
        print(
            f"Index: {idx_tuple}, Tensor1 Value: {a[idx_tuple]}, Tensor2 Value: {
                b[idx_tuple]}, Diff: {a[idx_tuple] - b[idx_tuple]}"
        )
    return indices


def save_model_weights(
    model: torch.nn.Module,
    hyperparameters: Dict[str, int | float],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: _Loss,
    random_state: int,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    last_train_loss: Optional[float] = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Save the model weights, optimizer state, and other necessary information to a checkpoint file.

    Args:
        model (torch.nn.Module): The model whose weights need to be saved.
        epoch (int): The current epoch index.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss (torch.nn.modules.loss._Loss): The loss function used for training the model
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The learning rate scheduler used during training. Defaults to None.
        last_train_loss (float, optional): The last recorded training loss. Defaults to None.
        model_name (str, optional): The name of the model. Defaults to "Graphormer".

    Returns:
        None
    """

    import os
    from datetime import datetime

    folder_save_path = "pretrained_models"
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)

    c_date = datetime.now().strftime("%d-%m-%y")

    if model_name is None:
        name = f"{folder_save_path}/Graphormer_checkpoint-{epoch}_{c_date}.pt"
    else:
        name = f"{folder_save_path}/{model_name}.pt"

    checkpoint = {
        "state_dict": model.state_dict(),
        "hyperparameters": hyperparameters,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "last_train_loss": last_train_loss,
        "loss_state_dict": loss.state_dict(),
        "random_state": random_state,
    }

    try:
        torch.save(checkpoint, name)
        print(f"Checkpoint successfully saved to {name}")
    except Exception as e:
        print(f"Failed to save {name}. Error: {e}")


def model_init_print(model_parameters: Dict[str, int | float], dataloaders: tuple, model=None, input_size=None) -> Dict[str, int | float]:
    from torchinfo import summary

    if model and input_size:
        summary(model)

    print_model_parameters_table(model_parameters, dataloaders)
    save_model_parameters(model_parameters)

    return model_parameters


def save_model_parameters(model_parameters: dict[str, int | float]) -> dict[str, int | float]:
    """
    Cleans the specified model parameters by removing any keys that are not in the keep_keys list.

    Args:
        model_parameters (dict[str, int | float]): The model parameters to be saved.

    Returns:
        dict[str, int | float]: The saved model parameters.

    """
    keep_keys = [
        "num_layers",
        "hidden_dim",
        "edge_embedding_dim",
        "ffn_hidden_dim",
        "n_heads",
        "max_in_degree",
        "max_out_degree",
        "max_path_distance",
    ]

    keys_to_remove = [k for k in model_parameters if k not in keep_keys]

    for k in keys_to_remove:
        model_parameters.pop(k, None)

    return model_parameters


def print_model_parameters_table(parameters_dict: Dict[str, int | float], dataloaders: tuple) -> None:
    """
    Display an overview of the training parameters.

    Args:
        parameters_dict (Dict[str, int | float]): A dictionary containing the training parameters.

    Returns:
        None
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(
        show_header=True,
        header_style="green",
        box=box.MINIMAL,  # Use a minimal box style to save space
    )
    table.add_column(
        "Parameter",
        style="dim",
        overflow="fold",
    )
    table.add_column("Value", overflow="fold")

    if dataloaders:
        table.add_row("Train Dataset Size", str(len(dataloaders[0].dataset)))
        table.add_row("Test Dataset Size", str(len(dataloaders[1].dataset)))

    for name, value in parameters_dict.items():
        table.add_row(name, str(value))


    console.print(table)
