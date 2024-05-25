from typing import List
import torch
from matplotlib.figure import Figure
from graphormer.config.options import AttentionType
from graphormer.modules.layers.encoder import GraphormerEncoderLayer
from graphormer.modules.model import Graphormer
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def plot_edge_path_length_bias(model: Graphormer) -> Figure:
    length_bias = [x.mean().item() for x in model.edge_encoding.edge_vector]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(length_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Edge Encoding Path Length Bias")
    ax.set_xlabel("Edge Path Length")
    ax.set_ylabel("Bias (Mean)")
    ax.grid(True)

    return fig


def plot_node_path_length_bias(model: Graphormer) -> Figure:
    length_bias = [x.item() for x in model.spatial_encoding.b]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(length_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Spatial Encoding Path Length Bias")
    ax.set_xlabel("Node Path Length")
    ax.set_ylabel("Bias")
    ax.grid(True)

    return fig


def plot_centrality_in_degree_bias(model: Graphormer) -> Figure:
    z_in_bias = [x.mean().item() for x in model.centrality_encoding.z_in]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_in_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Centrality Encoding In Degree Bias")
    ax.set_xlabel("In Degree")
    ax.set_ylabel("Bias (Mean)")
    ax.grid(True)

    return fig


def plot_centrality_out_degree_bias(model: Graphormer) -> Figure:
    z_out_bias = [x.mean().item() for x in model.centrality_encoding.z_out]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_out_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Centrality Encoding Out Degree Bias")
    ax.set_xlabel("Out Degree")
    ax.set_ylabel("Bias (Mean)")
    ax.grid(True)

    return fig


def plot_layer_residual_gates(model: Graphormer) -> Figure:
    res_gates = [x.item() for x in model.residual_gates]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res_gates, marker="o", linestyle="-", color="b")
    ax.set_title("Layer Residual Gates")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual Gate")
    ax.grid(True)

    return fig


def plot_attention_sigma(model: Graphormer) -> Figure:
    assert model.attention_type == AttentionType.FISH
    layers: List[GraphormerEncoderLayer] = model.layers  # type: ignore
    sigma: List[torch.Tensor] = [layer.attention.sigma for layer in layers]
