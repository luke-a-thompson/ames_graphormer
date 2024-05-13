import torch
from graphormer.config.hparams import HyperparameterConfig
from torch_geometric.loader import DataLoader
from graphormer.config.data import DataConfig
from graphormer.config.utils import model_init_print
from sklearn.metrics import balanced_accuracy_score


def inference_model(
    hparam_config: HyperparameterConfig, inference_loader: DataLoader = None, data_config: DataConfig = None
):
    if data_config is None:
        data_config = hparam_config.data_config()
    model_config = hparam_config.model_config()

    data_config.build()
    inference_loader = data_config.for_inference()
    assert hparam_config.batch_size is not None
    assert data_config.num_node_features is not None
    assert data_config.num_edge_features is not None

    device = torch.device(hparam_config.torch_device)
    model = (
        model_config.with_num_layers(hparam_config.num_layers)
        .with_node_feature_dim(data_config.num_node_features)
        .with_edge_feature_dim(data_config.num_edge_features)
        .with_output_dim(1)
        .build()
        .to(device)
    )

    model_init_print(hparam_config, model, test_dataloader=inference_loader)

    all_eval_labels = []
    all_eval_preds = []

    model.eval()
    for batch in inference_loader:
        batch.to(device)
        y = batch.y.to(device)
        with torch.no_grad():
            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.ptr,
                batch.node_paths,
                batch.edge_paths,
            )

        eval_preds = torch.round(torch.sigmoid(output)).tolist()
        eval_labels = y.cpu().numpy()

    all_eval_preds.extend(eval_preds)
    all_eval_labels.extend(eval_labels)

    bac = balanced_accuracy_score(all_eval_labels, all_eval_preds)

    print(f"Inference balanced accuracy: {bac:.3f}")
