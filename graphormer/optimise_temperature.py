import torch
from graphormer.config.hparams import HyperparameterConfig
from graphormer.modules.model import Graphormer
from graphormer.results import calculate_ece
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple


class TemperatureOptimizer:
    def __init__(self, hparam_config: HyperparameterConfig):
        self.device = torch.device(hparam_config.torch_device)
        self.data_config = hparam_config.data_config()
        self.model_config = hparam_config.model_config()
        _, self.validation_loader = self.data_config.build()  # type: ignore

    def get_predictions(self, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        assert self.data_config.num_node_features is not None
        assert self.data_config.num_edge_features is not None
        model: Graphormer = (
            self.model_config.with_node_feature_dim(self.data_config.num_node_features)
            .with_edge_feature_dim(self.data_config.num_edge_features)
            .with_output_dim(1)
            .with_temperature(temperature)
            .build()
            .to(self.device)
        )
        model.eval()

        labels = []
        preds = []

        for batch in self.validation_loader:
            batch.to(self.device)
            y = batch.y.to(self.device)

            with torch.no_grad():
                output = model(batch)

            batch_eval_preds = torch.sigmoid(output/temperature).cpu().numpy()
            batch_eval_labels = y.cpu().numpy()
            labels.extend(batch_eval_labels)
            preds.extend(batch_eval_preds)

        return np.array(labels), np.array(preds)

    def objective(self, temperature: float) -> float:
        labels, preds = self.get_predictions(temperature)
        return calculate_ece(labels, preds)

    def optimize(self, max_iter: int = 50) -> float:
        result = minimize_scalar(self.objective, bounds=(0.1, 5.0), method="bounded", options={"maxiter": max_iter})

        optimal_temperature = result.x
        optimal_ece = result.fun

        print(f"Optimal Temperature: {optimal_temperature:.4f}, ECE: {optimal_ece:.4f}")

        return optimal_temperature


def optimise_temperature(hparam_config: HyperparameterConfig, max_iter: int = 50) -> float:
    optimizer = TemperatureOptimizer(hparam_config)
    return optimizer.optimize(max_iter)
