from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
from typing import Dict


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


def eval_metrics(y_true, y_pred):
    y_true, y_pred = y_true.detach().cpu(), y_pred.detach().cpu()
    return {balanced_accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred)}


def difference_idxs(a, b, epsilon=1e-6) -> torch.Tensor:
    differences = torch.abs(a - b)
    indices = torch.nonzero(differences > epsilon)

    for idx in indices:
        idx_tuple = tuple(idx.tolist())
        print(
            f"Index: {idx_tuple}, Tensor1 Value: {a[idx_tuple]}, Tensor2 Value: {b[idx_tuple]}, Diff: {a[idx_tuple] - b[idx_tuple]}"
        )
    return indices


def save_model_weights(
    model: torch.nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    last_train_loss: float = None,
    model_name: str = "Graphormer",
) -> None:
    """
    Save the model weights, optimizer state, and other necessary information to a checkpoint file.

    Args:
        model (torch.nn.Module): The model whose weights need to be saved.
        epoch (int): The current epoch index.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The learning rate scheduler used during training. Defaults to None.
        last_train_loss (float, optional): The last recorded training loss. Defaults to None.
        model_name (str, optional): The name of the model. Defaults to "Graphormer".

    Returns:
        None
    """

    from datetime import datetime
    import os

    folder_save_path = "pretrained_models"
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)

    c_date = datetime.now().strftime("%d-%m-%y")

    name: str = f"{folder_save_path}/{model_name}_checkpoint-{epoch}_{c_date}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,  # Assuming epoch indexing starts at 0
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": [
            None if lr_scheduler is None else lr_scheduler.state_dict()
        ],  # If using a learning rate scheduler
        "loss": last_train_loss,  # Save the last loss value
    }

    try:
        torch.save(checkpoint, name)
        print(f"Checkpoint successfully saved to {name}")
    except Exception as e:
        print(f"Failed to save {name}. Error: {e}")


def print_model_parameters_table(parameters_dict: Dict[str, int | float]):
    """
    Display an overview of the training parameters.

    Args:
        parameters_dict (Dict[str, int | float]): A dictionary containing the training parameters.

    Returns:
        None
    """
    from rich.table import Table
    from rich.console import Console
    from rich import box

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

    for name, value in parameters_dict.items():
        table.add_row(name, str(value))

    console.print(table)
