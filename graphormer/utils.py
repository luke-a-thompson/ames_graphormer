import torch
import torch.nn as nn


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


def monte_carlo_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def difference_idxs(a, b, epsilon=1e-6) -> torch.Tensor:
    differences = torch.abs(a - b)
    indices = torch.nonzero(differences > epsilon)

    for idx in indices:
        idx_tuple = tuple(idx.tolist())
        print(
            f"Index: {idx_tuple}, Tensor1 Value: {a[idx_tuple]}, Tensor2 Value: {
                b[idx_tuple]}, Diff: {a[idx_tuple] - b[idx_tuple]}"
        )
    return indices
