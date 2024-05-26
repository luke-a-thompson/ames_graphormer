from typing import List
import torch
from matplotlib.figure import Figure
from torch.nn.utils.rnn import pad_sequence
from graphormer.config.options import AttentionType
from graphormer.modules.attention import GraphormerFishAttention
from graphormer.modules.model import Graphormer
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def plot_edge_path_length_bias(model: Graphormer) -> Figure:
    length_bias = [x.mean().item() for x in model.module_dict["edge_encoding"].edge_vector]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(length_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Edge Encoding Path Length Bias")
    ax.set_xlabel("Edge Path Length")
    ax.set_ylabel("Bias (Mean)")
    ax.grid(True)

    return fig


def plot_node_path_length_bias(model: Graphormer) -> Figure:
    length_bias = [x.item() for x in model.module_dict["spatial_encoding"].b]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(length_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Spatial Encoding Path Length Bias")
    ax.set_xlabel("Node Path Length")
    ax.set_ylabel("Bias")
    ax.grid(True)

    return fig


def plot_centrality_in_degree_bias(model: Graphormer) -> Figure:
    z_in_bias = [x.mean().item() for x in model.module_dict["centrality_encoding"].z_in]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_in_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Centrality Encoding In Degree Bias")
    ax.set_xlabel("In Degree")
    ax.set_ylabel("Bias (Mean)")
    ax.grid(True)

    return fig


def plot_centrality_out_degree_bias(model: Graphormer) -> Figure:
    z_out_bias = [x.mean().item() for x in model.module_dict["centrality_encoding"].z_out]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_out_bias, marker="o", linestyle="-", color="b")
    ax.set_title("Centrality Encoding Out Degree Bias")
    ax.set_xlabel("Out Degree")
    ax.set_ylabel("Bias (Mean)")
    ax.grid(True)

    return fig


def plot_layer_residual_weights(model: Graphormer) -> Figure:
    res_gates = [x.alpha.item() for k, x in model.module_dict.items() if k.startswith("residual_layer")]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res_gates, marker="o", linestyle="-", color="b")
    ax.set_title("Layer Residual Weights")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual Weight")
    ax.grid(True)

    return fig


def plot_attention_sigma(model: Graphormer) -> Figure:
    attention_layers: List[GraphormerFishAttention] = [x.attention for k, x in model.module_dict.items() if k.startswith("residual_layer") and isinstance(x.attention, GraphormerFishAttention)]  # type: ignore
    sigma: torch.Tensor = pad_sequence([layer.sigma for layer in attention_layers], padding_value=0.0)
    fig, ax = plt.subplots()
    cax = ax.imshow(sigma.detach().cpu().numpy(), cmap="viridis")
    fig.colorbar(cax)
    ax.set_title("Sigma Strength by Layer and Attention Head")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Global Attention Head")

    return fig
