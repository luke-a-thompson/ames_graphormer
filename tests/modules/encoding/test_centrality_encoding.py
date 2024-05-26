import torch

from graphormer.modules.encoding import CentralityEncoding
from graphormer.modules.model_data import ModelData


class TestCentralityEncodingGroup:
    def test_centrality_encoding(self):
        torch.set_grad_enabled(False)
        batch_size = 2
        num_nodes = 3
        hidden_dim = 5
        max_in_degree = 7
        max_out_degree = 9
        centrality_encoding = CentralityEncoding(max_in_degree, max_out_degree, hidden_dim)
        centrality_encoding.z_in = torch.nn.Parameter(
            torch.Tensor([[x * 2.0 for _ in range(hidden_dim)] for x in range(max_in_degree + 1)])
        )
        centrality_encoding.z_out = torch.nn.Parameter(
            torch.Tensor([[x * 3.0 for _ in range(hidden_dim)] for x in range(max_out_degree + 1)])
        )

        degrees = torch.ones(batch_size, 2, num_nodes) * -1.0
        # In degrees
        degrees[0, 1] = torch.Tensor([2, 6, 12])
        degrees[1, 1] = torch.Tensor([5, 6, -1])
        # Out degrees
        degrees[0, 0] = torch.Tensor([1, 5, 9])
        degrees[1, 0] = torch.Tensor([3, 4, -1])
        degrees = degrees.long()

        device = torch.device("cpu")
        data = ModelData(
            torch.zeros(batch_size, num_nodes, hidden_dim),
            degrees,
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            device,
        )
        data: ModelData = centrality_encoding(data)
        assert data.x.shape == (batch_size, num_nodes, hidden_dim)

        assert (data.x[0, 0, :] == 7.0).all(), data.x[0, 0, :]
        assert (data.x[0, 1, :] == 27.0).all(), data.x[0, 1, :]
        assert (data.x[0, 2, :] == 41.0).all(), data.x[0, 2, :]

        assert (data.x[1, 0, :] == 19.0).all(), data.x[1, 0, :]
        assert (data.x[1, 1, :] == 24.0).all(), data.x[1, 1, :]
        assert (data.x[1, 2, :] == 0.0).all(), data.x
