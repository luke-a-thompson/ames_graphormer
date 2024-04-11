from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch


def eval_metrics(y_true, y_pred):
    y_true, y_pred = y_true.detach().cpu(), y_pred.detach().cpu()
    return {balanced_accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred)}


def difference_idxs(a, b, epsilon=1e-6) -> torch.Tensor:
    differences = torch.abs(a - b)
    indices = torch.nonzero(differences > epsilon)

    for idx in indices:
        idx_tuple = tuple(idx.tolist())
        print(
            f"Index: {idx_tuple}, Tensor1 Value: {a[idx_tuple]}, Tensor2 Value: {b[idx_tuple]}, Diff: {a[idx_tuple] - b[idx_tuple]}"
        )
    return indices
