import torch
import tomllib


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


def parse_models(ctx, param, value):
    return value.split(",")


def create_composite_decorator(*decorators):
    def composite_decorator(f):
        for d in reversed(decorators):
            f = d(f)
        return f

    return composite_decorator


def configure(ctx, param, filename):
    with open(filename, "rb") as f:
        config = tomllib.load(f)
    ctx.default_map = config
