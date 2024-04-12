import click
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from data.data_cleaning import HonmaDataset
from graphormer.model import Graphormer
from graphormer.utils import (
    save_model_weights,
    print_model_parameters_table,
)


@click.command()
@click.option("--data", default="data")
@click.option("--num_layers", default=3)
@click.option("--hidden_dim", default=128)
@click.option("--edge_embedding_dim", default=128)
@click.option("--ffn_hidden_dim", default=80)
@click.option("--n_heads", default=4)
@click.option("--max_in_degree", default=5)
@click.option("--max_out_degree", default=5)
@click.option("--max_path_distance", default=5)
@click.option("--test_size", default=0.8)
@click.option("--random_state", default=42)
@click.option("--batch_size", default=4)
@click.option("--lr", default=3e-4)
@click.option("--b1", default=0.9)
@click.option("--b2", default=0.999)
@click.option("--weight_decay", default=0.0)
@click.option("--eps", default=1e-8)
@click.option("--clip_grad_norm", default=5.0)
@click.option("--torch_device", default="cuda")
@click.option("--epochs", default=10)
def train(
    data: str,
    num_layers: int,
    hidden_dim: int,
    edge_embedding_dim: int,
    ffn_hidden_dim: int,
    n_heads: int,
    max_in_degree: int,
    max_out_degree: int,
    max_path_distance: int,
    test_size: float,
    random_state: int,
    batch_size: int,
    lr: float,
    b1: float,
    b2: float,
    weight_decay: float,
    eps: float,
    clip_grad_norm: float,
    torch_device: str,
    epochs: int,
):
    model_parameters = locals().copy()
    print_model_parameters_table(model_parameters)
    torch.manual_seed(random_state)
    device = torch.device(torch_device)
    dataset = HonmaDataset(data)
    model = Graphormer(
        num_layers=num_layers,
        node_feature_dim=dataset.num_node_features,
        hidden_dim=hidden_dim,
        edge_feature_dim=dataset.num_edge_features,
        edge_embedding_dim=edge_embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        output_dim=dataset[0].y.shape[0],  # type: ignore
        n_heads=n_heads,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        max_path_distance=max_path_distance,
    )

    test_ids, train_ids = train_test_split(
        [i for i in range(len(dataset))], test_size=test_size, random_state=random_state
    )
    train_loader = DataLoader(
        Subset(dataset, train_ids), batch_size=batch_size  # type: ignore
    )
    test_loader = DataLoader(
        Subset(dataset, test_ids), batch_size=batch_size  # type: ignore
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay, eps=eps
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    model.to(device)

    progress_bar = tqdm(total=0, desc="Initializing...", unit="batch")

    for epoch in range(epochs):
        total_loss = 0.0
        total_eval_loss = 0.0

        # Set total length for training phase and update description
        progress_bar.reset(total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Train")

        model.train()
        avg_loss = 0.0
        for batch in train_loader:
            batch.to(device)
            y = batch.y.to(device)
            optimizer.zero_grad()
            model_out = model(batch)
            output = global_mean_pool(model_out, batch.batch)
            loss = loss_function(output, y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix_str(f"Avg Loss: {avg_loss:.4f}")
            progress_bar.update()  # Increment the progress bar

        # Prepare for the evaluation phase
        progress_bar.reset(total=len(test_loader))
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Eval")

        model.eval()
        for batch in test_loader:
            batch.to(device)
            y = batch.y.to(device)
            with torch.no_grad():
                output = global_mean_pool(model(batch), batch.batch)
                loss = loss_function(output, y.unsqueeze(1))
            total_eval_loss += loss.item()

            progress_bar.update()  # Manually increment for each batch in eval

        avg_eval_loss = total_eval_loss / len(test_loader)
        progress_bar.set_postfix_str(f"Avg Eval Loss: {avg_eval_loss:.4f}")

        print(
            f"Epoch {epoch+1} | Avg Train Loss: {avg_loss:.4f} | Avg Eval Loss: {avg_eval_loss:.4f} | Eval BAC: {balanced_accuracy_score(y.cpu().numpy(), output.cpu().numpy() > 0.5):.4f}"
        )

        if epoch % 20 == 0 and epoch != 0:
            save_model_weights(model, epoch, optimizer, last_train_loss=avg_loss)

    progress_bar.close()


def inference(
    model_path: str,
    dataset: str,
    torch_device: str,
) -> torch.Tensor:
    device = torch.device(torch_device)
    dataset = HonmaDataset("data")
    model = torch.load(model_path, map_location=device)

    inference_loader = DataLoader(dataset, device)

    model.eval()

    with torch.no_grad:
        output = model(inference_loader)

    return output
