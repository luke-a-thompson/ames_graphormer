import torch
from graphormer.modules.norm import MaxNorm


class TestMaxNormGroup:
    def test_handle_all_ones(self):
        torch.set_grad_enabled(False)
        shape = (2, 3, 4, 5)
        x = torch.ones(shape)
        max_norm = MaxNorm(input_dim=shape[3])
        norm_x = max_norm(x)
        assert (norm_x == 1).all()

    def test_handle_all_zeros(self):
        torch.set_grad_enabled(False)
        shape = (2, 3, 4, 5)
        x = torch.zeros(shape)
        max_norm = MaxNorm(input_dim=shape[3])
        norm_x = max_norm(x)
        assert (norm_x == 0).all()

    def test_handle_normal_case(self):
        torch.set_grad_enabled(False)
        x = torch.Tensor([-3, -2, -1, 0, 1, 2, 3])
        max_norm = MaxNorm(input_dim=x.shape[0])
        norm_x = max_norm(x)
        expected_norm = x / 3
        assert torch.allclose(norm_x, expected_norm)
