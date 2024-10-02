import torch
from typing import Optional
from graphormer.config.hparams import HyperparameterConfig
from torch_geometric.loader import DataLoader
from graphormer.config.data import DataConfig
from graphormer.modules.model import Graphormer
from graphormer.config.utils import model_init_print
from tqdm import tqdm
from typing import Dict, List


def inference_model(
    hparam_config: HyperparameterConfig,
    inference_loader: Optional[DataLoader] = None,
    data_config: Optional[DataConfig] = None,
    mc_samples: Optional[int] = None,
    mc_dropout_rate: Optional[float] = None,
) -> List[Dict[str, List[int | float]]] | Dict[str, List[int | float]]:
    """
    Inference model.

    Returns:
        List[Dict[str, List[int | float]]] if mc_samples is not None,
        Dict[str, List[int | float]] if mc_samples is None.
    """
    if data_config is None:
        data_config = hparam_config.data_config()
    model_config = hparam_config.model_config()

    inference_loader = data_config.build()  # type: ignore

    assert hparam_config.batch_size is not None
    assert data_config.num_node_features is not None
    assert data_config.num_edge_features is not None

    device = torch.device(hparam_config.torch_device)
    model: Graphormer = (
        model_config.with_node_feature_dim(data_config.num_node_features)
        .with_edge_feature_dim(data_config.num_edge_features)
        .with_output_dim(1)
        .build()
        .to(device)
    )

    model_init_print(hparam_config, model, test_dataloader=inference_loader)

    model.eval()
    if mc_samples is not None and mc_dropout_rate is not None:
        mc_results = []
        model.enable_dropout(mc_dropout_rate)
        labels = []
        logits = []
        for mc_sample in tqdm(range(mc_samples), desc="MC Dropout Inference", unit="mc_sample"):

            for batch in inference_loader:  # type: ignore
                batch.to(device)
                y = batch.y.to(device)

                with torch.no_grad():
                    output = model(batch)

                batch_eval_logits: List[float] = torch.sigmoid(output/hparam_config.temperature).tolist()
                batch_eval_labels: List[float] = y.cpu().tolist()

                labels.extend(batch_eval_labels)
                logits.extend(batch_eval_logits)

            mc_results.append({"labels": labels, "logits": logits})

        return mc_results
    else:
        labels = []
        logits = []
        for batch in inference_loader:  # type: ignore

            batch.to(device)
            y = batch.y.to(device)

            with torch.no_grad():
                output = model(batch)

            batch_eval_logits: List[float] = torch.sigmoid(output/hparam_config.temperature).tolist()
            batch_eval_labels: List[float] = y.cpu().tolist()

            labels.extend(batch_eval_labels)
            logits.extend(batch_eval_logits)

        results = {"labels": labels, "logits": logits}

        return results
