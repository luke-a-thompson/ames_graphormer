import math

import click
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm

from data.data_cleaning import AmesDataset
from graphormer.model import Graphormer


@click.command()
@click.option("--data", default="data")
@click.option("--num_layers", default=3)
@click.option("--node_dim", default=128)
@click.option("--edge_dim", default=128)
@click.option("--n_heads", default=4)
@click.option("--max_in_degree", default=5)
@click.option("--max_out_degree", default=5)
@click.option("--max_path_distance", default=5)
@click.option("--test_size", default=0.8)
@click.option("--random_state", default=42)
@click.option("--batch_size", default=4)
@click.option("--lr", default=3e-4)
@click.option("--torch_device", default="cuda")
@click.option("--epochs", default=10)
def train(
    data: str,
    num_layers: int,
    node_dim: int,
    edge_dim: int,
    n_heads: int,
    max_in_degree: int,
    max_out_degree: int,
    max_path_distance: int,
    test_size: float,
    random_state: int,
    batch_size: int,
    lr: float,
    torch_device: str,
    epochs: int,
):
    torch.manual_seed(random_state)
    device = torch.device(torch_device)
    dataset = AmesDataset(data)
    model = Graphormer(
        num_layers=num_layers,
        input_node_dim=dataset.num_node_features,
        node_dim=node_dim,
        input_edge_dim=dataset.num_edge_features,
        edge_dim=edge_dim,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_function = nn.L1Loss(reduction="sum")
    model.to(device)

    for _ in range(epochs):
        model.train()
        batch_loss = 0.0
        prog = tqdm(train_loader)
        for batch in prog:
            batch.to(device)
            y = batch.y
            optimizer.zero_grad()
            output = global_mean_pool(model(batch), batch.batch)
            loss = loss_function(output, y)
            batch_loss += loss.item()
            prog.set_description(f"batch_loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            assert not math.isnan(batch_loss)

        print("TRAIN LOSS", batch_loss / len(train_ids))

        model.eval()
        batch_loss = 0.0
        prog = tqdm(test_loader)
        for batch in prog:
            batch.to(device)
            y = batch.y
            with torch.no_grad():
                output = global_mean_pool(model(batch), batch.batch)
                loss = loss_function(output, y)
            batch_loss += loss.item()
            prog.set_description(f"batch_loss: {loss.item()}")
            assert not math.isnan(batch_loss)
        print("EVAL LOSS", batch_loss / len(test_ids))

