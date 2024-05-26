import torch
from graphormer.modules.encoding import SpatialEncoding
from graphormer.modules.model_data import ModelData


class TestSpatialEncodingGroup:
    def test_spatial_encoding(self):
        torch.set_grad_enabled(False)
        max_path_length = 5
        spatial_encoding = SpatialEncoding(max_path_length)
        spatial_encoding.b = torch.nn.Parameter(torch.arange(max_path_length) * -1.0)
        spatial_encoding.t1 = torch.nn.Parameter(torch.Tensor([2.0]))
        spatial_encoding.t2 = torch.nn.Parameter(torch.Tensor([3.0]))
        batch_size = 2
        num_pairwise_paths = 36
        paths = torch.ones(batch_size, num_pairwise_paths, max_path_length) * -1

        paths[0][:6] = torch.Tensor([[0.0, x, -1.0, -1.0, -1.0] for x in range(1, 7)])
        paths[0][6:12] = torch.Tensor([[x, 0.0, -1.0, -1.0, -1.0] for x in range(1, 7)])
        paths[0][13] = torch.Tensor([1.0, 2.0, 3.0, -1.0, -1.0])
        paths[0][14] = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        paths[1][:6] = torch.Tensor([[0.0, x, -1.0, -1.0, -1.0] for x in range(1, 7)])
        paths[1][6:12] = torch.Tensor([[x, 0.0, -1.0, -1.0, -1.0] for x in range(1, 7)])
        paths[1][13] = torch.Tensor([3.0, 2.0, -1.0, -1.0, -1.0])
        paths[1][14] = torch.Tensor([5.0, 4.0, 3.0, 2.0, -1.0])

        device = torch.device("cpu")
        data = ModelData(torch.zeros(3), torch.zeros(3), torch.zeros(3), paths, torch.zeros(3), device)

        data: ModelData = spatial_encoding(data)
        assert data.spatial_encoding is not None

        assert data.spatial_encoding.shape == (batch_size, num_pairwise_paths)
        assert (data.spatial_encoding[0, :6] == spatial_encoding.t1).all()
        assert (data.spatial_encoding[0, 6:12] == spatial_encoding.t2).all()

        expected_path_0_13_encoding = torch.Tensor([-2.0])
        expected_path_0_14_encoding = torch.Tensor([-4.0])
        expected_path_1_13_encoding = torch.Tensor([-1.0])
        expected_path_1_14_encoding = torch.Tensor([-3.0])
        assert data.spatial_encoding[0, 13] == expected_path_0_13_encoding
        assert data.spatial_encoding[0, 14] == expected_path_0_14_encoding
        assert data.spatial_encoding[1, 13] == expected_path_1_13_encoding
        assert data.spatial_encoding[1, 14] == expected_path_1_14_encoding
        assert (data.spatial_encoding[0, 15:] == 0.0).all()
        assert (data.spatial_encoding[1, 15:] == 0.0).all()
