import torch
from graphormer.modules.model_data import ModelData
from graphormer.modules.norm import MaxNorm


class TestMaxNormGroup:
    def test_handle_all_ones(self):
        torch.set_grad_enabled(False)
        shape = (2, 3, 4, 5)
        x = torch.ones(shape)
        device = torch.device("cpu")
        data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
        max_norm = MaxNorm(input_dim=shape[3])
        data = max_norm(data)

        assert (data.normalized_input == 1).all()

    def test_handle_all_zeros(self):
        torch.set_grad_enabled(False)
        shape = (2, 3, 4, 5)
        x = torch.zeros(shape)
        device = torch.device("cpu")
        data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
        max_norm = MaxNorm(input_dim=shape[3])
        data = max_norm(data)
        assert (data.normalized_input == 0).all()

    def test_handle_normal_case(self):
        torch.set_grad_enabled(False)
        x = torch.Tensor([-3, -2, -1, 0, 1, 2, 3])
        device = torch.device("cpu")
        data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
        max_norm = MaxNorm(input_dim=x.shape[0])
        data = max_norm(data)
        expected_norm = x / 3
        assert torch.allclose(data.normalized_input, expected_norm)
