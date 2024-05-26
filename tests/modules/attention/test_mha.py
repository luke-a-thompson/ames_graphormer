import torch
from graphormer.modules.attention import GraphormerMultiHeadAttention
from graphormer.modules.model_data import ModelData


class TestMultiHeadAttentionGroup:
    def test_forward(self):
        embedding_dim = 4
        num_nodes = 3
        num_heads = 2
        batch_size = 1
        with torch.no_grad():
            weights = torch.nn.Parameter(
                torch.concat([torch.arange(embedding_dim) * x for x in range(embedding_dim)])
                .reshape(embedding_dim, embedding_dim)
                .float()
            )
            mha = GraphormerMultiHeadAttention(num_heads, embedding_dim, 0.0)
            mha.linear_q.weight = weights
            mha.linear_k.weight = weights
            mha.linear_v.weight = weights
            mha.linear_out.weight = torch.nn.Parameter(torch.ones_like(weights))

            ref_mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, bias=False, batch_first=True)
            ref_mha.in_proj_weight[:embedding_dim, :] = weights
            ref_mha.in_proj_weight[embedding_dim : 2 * embedding_dim, :] = weights
            ref_mha.in_proj_weight[2 * embedding_dim :, :] = weights

            ref_mha.out_proj.weight = torch.nn.Parameter(torch.ones_like(weights))

            x = (
                torch.arange(batch_size * num_nodes * embedding_dim)
                .reshape(batch_size, num_nodes, embedding_dim)
                .float()
            )
            device = torch.device("cpu")
            data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
            data.normalized_input = x
            data.attention_prior = torch.zeros(batch_size, num_nodes, num_nodes)
            data = mha.forward(data)
            assert data.attention_output is not None
            ref_mha_out = ref_mha.forward(x, x, x, need_weights=False)[0]
            assert torch.allclose(data.attention_output, ref_mha_out, rtol=0.3)
