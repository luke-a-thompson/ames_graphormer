from collections import OrderedDict
from typing import Dict, List, Self

import torch.nn as nn
from graphormer.modules.attention import (
    AttentionPrior,
    AttentionPadMask,
    GraphormerMultiHeadAttention,
    GraphormerLinearAttention,
    GraphormerFishAttention,
)
from graphormer.modules.embedding import EdgeEmbedding, NodeEmbedding
from graphormer.modules.encoding import SpatialEncoding, EdgeEncoding
from graphormer.modules.layers import FeedForwardNetwork, Output
from graphormer.modules.model import Graphormer
from graphormer.modules.encoding import CentralityEncoding
from graphormer.config.options import AttentionType, NormType, ResidualType
from graphormer.modules.norm import LayerNorm, RMSNorm, CRMSNorm, MaxNorm
from graphormer.modules.residuals import PreNormResidual, RezeroResidual


class ModelConfig:
    def __init__(self):
        self.num_layers = None
        self.node_feature_dim = None
        self.hidden_dim = None
        self.edge_feature_dim = None
        self.edge_embedding_dim = None
        self.ffn_hidden_dim = None
        self.output_dim = None
        self.n_heads = None
        self.heads_by_layer = None
        self.max_in_degree = None
        self.max_out_degree = None
        self.max_path_distance = None
        self.dropout = None
        self.temperature = None
        self.state_dict = None
        self.norm_type = None
        self.attention_type = None
        self.global_heads_by_layer = None
        self.local_heads_by_layer = None
        self.n_global_heads = None
        self.n_local_heads = None
        self.residual_type = None

    def with_num_layers(self, num_layers: int) -> Self:
        self.num_layers = num_layers
        return self

    def with_node_feature_dim(self, node_feature_dim: int) -> Self:
        self.node_feature_dim = node_feature_dim
        return self

    def with_hidden_dim(self, hidden_dim: int) -> Self:
        self.hidden_dim = hidden_dim
        return self

    def with_edge_feature_dim(self, edge_feature_dim: int) -> Self:
        self.edge_feature_dim = edge_feature_dim
        return self

    def with_edge_embedding_dim(self, edge_embedding_dim: int) -> Self:
        self.edge_embedding_dim = edge_embedding_dim
        return self

    def with_ffn_hidden_dim(self, ffn_hidden_dim: int) -> Self:
        self.ffn_hidden_dim = ffn_hidden_dim
        return self

    def with_output_dim(self, output_dim: int) -> Self:
        self.output_dim = output_dim
        return self

    def with_num_heads(self, n_heads: int) -> Self:
        self.n_heads = n_heads
        return self

    def with_heads_by_layer(self, heads_by_layer: List[int]) -> Self:
        self.heads_by_layer = heads_by_layer
        return self

    def with_max_in_degree(self, max_in_degree: int) -> Self:
        self.max_in_degree = max_in_degree
        return self

    def with_max_out_degree(self, max_out_degree: int) -> Self:
        self.max_out_degree = max_out_degree
        return self

    def with_max_path_distance(self, max_path_distance: int) -> Self:
        self.max_path_distance = max_path_distance
        return self

    def with_dropout(self, dropout: float) -> Self:
        self.dropout = dropout
        return self

    def with_temperature(self, temperature: float) -> Self:
        self.temperature = temperature
        return self

    def with_norm_type(self, norm_type: NormType) -> Self:
        self.norm_type = norm_type
        return self

    def with_attention_type(self, attention_type: AttentionType) -> Self:
        self.attention_type = attention_type
        return self

    def with_n_global_heads(self, n_global_heads: int) -> Self:
        self.n_global_heads = n_global_heads
        return self

    def with_global_heads_by_layer(self, global_heads_by_layer: List[int]) -> Self:
        self.global_heads_by_layer = global_heads_by_layer
        return self

    def with_n_local_heads(self, n_local_heads: int) -> Self:
        self.n_local_heads = n_local_heads
        return self

    def with_local_heads_by_layer(self, local_heads_by_layer: List[int]) -> Self:
        self.local_heads_by_layer = local_heads_by_layer
        return self

    def with_residual_type(self, residual_type: ResidualType) -> Self:
        self.residual_type = residual_type
        return self

    def with_state_dict(self, state_dict: Dict) -> Self:
        self.state_dict = state_dict
        return self

    def build(self) -> Graphormer:
        if self.num_layers is None:
            raise AttributeError("num_layers is not defined for Graphormer")
        if self.node_feature_dim is None:
            raise AttributeError("node_feature_dim is not defined for Graphormer")
        if self.hidden_dim is None:
            raise AttributeError("hidden_dim is not defined for Graphormer")
        if self.edge_feature_dim is None:
            raise AttributeError("edge_feature_dim is not defined for Graphormer")
        if self.edge_embedding_dim is None:
            raise AttributeError("edge_embedding_dim is not defined for Graphormer")
        if self.ffn_hidden_dim is None:
            raise AttributeError("ffn_hidden_dim is not defined for Graphormer")
        if self.output_dim is None:
            raise AttributeError("output_dim is not defined for Graphormer")
        if self.max_in_degree is None:
            raise AttributeError("max_in_degree is not defined for Graphormer")
        if self.max_out_degree is None:
            raise AttributeError("max_out_degree is not defined for Graphormer")
        if self.max_path_distance is None:
            raise AttributeError("max_path_distance is not defined for Graphormer")
        if self.dropout is None:
            raise AttributeError("dropout is not defined for Graphormer")
        if self.temperature is None:
            raise AttributeError("temperature is not defined for Graphormer")
        if self.norm_type is None:
            raise AttributeError("norm_type is not defined for Graphormer")
        if self.attention_type is None:
            raise AttributeError("attention_type is not defined for Graphormer")
        if self.n_heads is None and self.heads_by_layer is None and self.attention_type == AttentionType.MHA:
            raise AttributeError("n_heads or heads_by_layer must be defined for Graphormer")
        if (
            self.n_global_heads is None
            and self.global_heads_by_layer is None
            and self.attention_type == AttentionType.FISH
        ):
            raise AttributeError("n_global_heads or global_heads_by_layer must be defined for Graphormer")
        if (
            self.n_local_heads is None
            and self.local_heads_by_layer is None
            and self.attention_type == AttentionType.FISH
        ):
            raise AttributeError("n_local_heads or local_heads_by_layer must be defined for Graphormer")
        if self.residual_type is None:
            raise AttributeError("residual_type must be defined for Graphormer")

        residual_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(self.num_layers):
            input_norm = None
            match self.norm_type:
                case NormType.LAYER:
                    input_norm = LayerNorm(self.hidden_dim)
                case NormType.RMS:
                    input_norm = RMSNorm(self.hidden_dim)
                case NormType.CRMS:
                    input_norm = CRMSNorm(self.hidden_dim)
                case NormType.MAX:
                    input_norm = MaxNorm(self.hidden_dim)
                case NormType.NONE:
                    pass

            attention = None
            match self.attention_type:
                case AttentionType.MHA:
                    if self.heads_by_layer is None or len(self.heads_by_layer) == 0:
                        assert self.n_heads is not None
                        self.heads_by_layer = [self.n_heads for _ in range(self.num_layers)]
                    if len(self.heads_by_layer) != self.num_layers:
                        raise ValueError(
                            f"The length of heads_by_layer {len(self.heads_by_layer)} must equal the number of layers {self.num_layers}"
                        )

                    num_heads = self.heads_by_layer[i]
                    attention = GraphormerMultiHeadAttention(num_heads, self.hidden_dim, self.dropout)
                case AttentionType.LINEAR:
                    if self.heads_by_layer is None or len(self.heads_by_layer) == 0:
                        assert self.n_heads is not None
                        self.heads_by_layer = [self.n_heads for _ in range(self.num_layers)]
                    if len(self.heads_by_layer) != self.num_layers:
                        raise ValueError(
                            f"The length of heads_by_layer {len(self.heads_by_layer)} must equal the number of layers {self.num_layers}"
                        )

                    num_heads = self.heads_by_layer[i]
                    attention = GraphormerLinearAttention(num_heads, self.hidden_dim, self.dropout)
                case AttentionType.FISH:
                    if self.global_heads_by_layer is None or len(self.global_heads_by_layer) == 0:
                        assert self.n_global_heads is not None
                        self.global_heads_by_layer = [self.n_global_heads for _ in range(self.num_layers)]
                    if self.local_heads_by_layer is None or len(self.local_heads_by_layer) == 0:
                        assert self.n_local_heads is not None
                        self.local_heads_by_layer = [self.n_local_heads for _ in range(self.num_layers)]
                    num_global_heads = self.global_heads_by_layer[i]
                    num_local_heads = self.local_heads_by_layer[i]
                    if num_global_heads != num_local_heads:
                        attention = GraphormerFishAttention(
                            num_global_heads, num_local_heads, self.hidden_dim, self.dropout
                        )
                    else:
                        attention = GraphormerMultiHeadAttention(num_global_heads, self.hidden_dim, self.dropout)

            attention_norm = None
            match self.norm_type:
                case NormType.LAYER:
                    attention_norm = LayerNorm(self.hidden_dim)
                case NormType.RMS:
                    attention_norm = RMSNorm(self.hidden_dim)
                case NormType.CRMS:
                    attention_norm = CRMSNorm(self.hidden_dim)
                case NormType.MAX:
                    attention_norm = MaxNorm(self.hidden_dim)
                case NormType.NONE:
                    pass

            ffn = FeedForwardNetwork(self.hidden_dim, self.ffn_hidden_dim, self.hidden_dim, self.dropout)
            residual = None
            match self.residual_type:
                case ResidualType.PRENORM:
                    assert input_norm is not None
                    assert attention_norm is not None
                    residual = PreNormResidual(input_norm, attention, attention_norm, ffn)
                case ResidualType.REZERO:
                    residual = RezeroResidual(attention, ffn)

            assert residual is not None
            residual_layers[f"residual_layer_{i}"] = residual

        modules = OrderedDict()

        modules["node_embedding"] = NodeEmbedding(self.node_feature_dim, self.hidden_dim)
        modules["edge_embedding"] = EdgeEmbedding(self.edge_feature_dim, self.edge_embedding_dim)
        modules["centrality_encoding"] = CentralityEncoding(self.max_in_degree, self.max_out_degree, self.hidden_dim)
        modules["spatial_encoding"] = SpatialEncoding(self.max_path_distance)
        modules["edge_encoding"] = EdgeEncoding(self.edge_embedding_dim, self.max_path_distance)
        modules["attention_prior"] = AttentionPrior()
        modules["attention_pad_mask"] = AttentionPadMask()
        for key, residual_layer in residual_layers.items():
            modules[key] = residual_layer
        modules["output"] = Output(self.hidden_dim, self.output_dim)

        model = Graphormer(modules)

        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
            self.state_dict = None

        return model
