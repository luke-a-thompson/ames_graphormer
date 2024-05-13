from typing import Dict, Self
from graphormer.model import Graphormer


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
        self.max_in_degree = None
        self.max_out_degree = None
        self.max_path_distance = None
        self.dropout = None
        self.state_dict = None
        self.rescale = None

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

    def with_rescale(self, rescale: bool) -> Self:
        self.rescale = rescale
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
        if self.n_heads is None:
            raise AttributeError("n_heads is not defined for Graphormer")
        if self.max_in_degree is None:
            raise AttributeError("max_in_degree is not defined for Graphormer")
        if self.max_out_degree is None:
            raise AttributeError("max_out_degree is not defined for Graphormer")
        if self.max_path_distance is None:
            raise AttributeError("max_path_distance is not defined for Graphormer")
        if self.dropout is None:
            raise AttributeError("dropout is not defined for Graphormer")
        if self.rescale is None:
            raise AttributeError("rescale is not defined for Graphormer")

        model = Graphormer(
            num_layers=self.num_layers,
            node_feature_dim=self.node_feature_dim,
            hidden_dim=self.hidden_dim,
            edge_feature_dim=self.edge_feature_dim,
            edge_embedding_dim=self.edge_embedding_dim,
            ffn_hidden_dim=self.ffn_hidden_dim,
            output_dim=self.output_dim,
            n_heads=self.n_heads,
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            max_path_distance=self.max_path_distance,
            dropout=self.dropout,
            rescale=self.rescale,
        )

        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
            self.state_dict = None

        return model
