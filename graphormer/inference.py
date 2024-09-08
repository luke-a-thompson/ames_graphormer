import torch
from typing import Optional
from graphormer.config.hparams import HyperparameterConfig
from torch_geometric.loader import DataLoader
from graphormer.config.data import DataConfig
from graphormer.config.options import DatasetRegime
from graphormer.modules.model import Graphormer
from graphormer.config.utils import model_init_print
from tqdm import tqdm
from typing import Dict, List


def inference_model(
    hparam_config: HyperparameterConfig,
    inference_loader: Optional[DataLoader] = None,
    data_config: Optional[DataConfig] = None,
    mc_samples: Optional[int] = None,
) -> Dict[int, Dict[str, List[float] | int]]:
    if data_config is None:
        data_config = hparam_config.data_config()
    model_config = hparam_config.model_config()

    mc_dropout = mc_samples is not None
    mc_dropout_rate = 0.1

    data_config.dataset_regime = DatasetRegime.TEST
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

    results = {}

    model.eval()
    if mc_dropout:
        model.enable_dropout(mc_dropout_rate)
        for mc_sample in tqdm(range(mc_samples), desc="MC Dropout Inference", unit="mc_sample"):
            for batch_idx, batch in enumerate(inference_loader):  # type: ignore
                sample_idx: int = batch_idx * hparam_config.batch_size

                batch.to(device)
                y = batch.y.to(device)
                with torch.no_grad():
                    output = model(batch)

                batch_eval_preds = torch.sigmoid(output).tolist()
                batch_eval_labels = y.cpu().numpy()

                for pred, label in zip(batch_eval_preds, batch_eval_labels):
                    if sample_idx not in results:
                        results[sample_idx] = {"preds": [], "label": label}

                    results[sample_idx]["preds"].append(pred)

                    sample_idx += 1

        return results

    for batch_idx, batch in enumerate(inference_loader):  # type: ignore
        sample_idx: int = batch_idx * hparam_config.batch_size
        batch.to(device)
        y = batch.y.to(device)
        with torch.no_grad():
            output = model(batch)

            batch_eval_preds = torch.sigmoid(output).tolist()
            batch_eval_labels = y.cpu().numpy()

            for pred, label in zip(batch_eval_preds, batch_eval_labels):
                if sample_idx not in results:
                    results[sample_idx] = {"preds": [], "label": label}

                results[sample_idx]["preds"].append(pred)

                sample_idx += 1

    return results
