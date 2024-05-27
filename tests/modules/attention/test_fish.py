import torch

from graphormer.modules.attention.fish import GraphormerFishAttention
from graphormer.modules.model_data import ModelData


class TestFishGroup:
    def test_forward(self):
        torch.set_grad_enabled(False)
        device = torch.device("cpu")

        embedding_dim = 4
        num_nodes = 3
        batch_size = 2
        num_global_heads = 2
        num_local_heads = 4
        global_weights = torch.nn.Parameter(
            torch.concat([torch.arange(embedding_dim) * x for x in range(embedding_dim)])
            .reshape(embedding_dim, embedding_dim)
            .float()
        )
        fish = GraphormerFishAttention(num_global_heads, num_local_heads, embedding_dim, 0.0)
        fish.global_q.weight = global_weights
        fish.global_k.weight = global_weights
        fish.local_v.weight = torch.nn.Parameter(torch.ones_like(fish.local_v.weight) * 0.1)
        fish.sigma = torch.nn.Parameter(torch.zeros_like(fish.sigma))
        fish.linear_out.weight = torch.nn.Parameter(torch.ones_like(fish.linear_out.weight) * 0.1)

        x = torch.Tensor(
            [
                [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]],
                [[0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        )

        data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
        data.normalized_input = data.x
        data.attention_prior = torch.zeros(batch_size, num_nodes, num_nodes)
        data: ModelData = fish(data)
        assert data.attention_output is not None
        assert data.attention_output.shape == (batch_size, num_nodes, embedding_dim)
