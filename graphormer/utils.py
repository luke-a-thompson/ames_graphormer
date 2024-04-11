from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch


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
