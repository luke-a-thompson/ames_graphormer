import torch
from graphormer.modules.attention import GraphormerMultiHeadAttention
from graphormer.modules.model_data import ModelData


class TestMultiHeadAttentionGroup:
    def test_forward(self):
        torch.set_grad_enabled(False)
        embedding_dim = 4
        num_nodes = 3
        num_heads = 2
        batch_size = 1
        weights = torch.nn.Parameter(
            torch.concat([torch.arange(embedding_dim) * x for x in range(embedding_dim)])
            .reshape(embedding_dim, embedding_dim)
            .float()
        )
        mha = GraphormerMultiHeadAttention(num_heads, embedding_dim, 0.0)
        mha.linear_q.weight = weights
        mha.linear_k.weight = weights
        mha.linear_v.weight = weights
        mha.linear_out.weight = torch.nn.Parameter(weights)

        ref_mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, bias=False, batch_first=True)
        ref_mha.in_proj_weight[:embedding_dim, :] = weights
        ref_mha.in_proj_weight[embedding_dim : 2 * embedding_dim, :] = weights
        ref_mha.in_proj_weight[2 * embedding_dim :, :] = weights

        ref_mha.out_proj.weight = torch.nn.Parameter(weights)

        x = torch.arange(batch_size * num_nodes * embedding_dim).reshape(batch_size, num_nodes, embedding_dim).float()
        device = torch.device("cpu")
        data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
        data.normalized_input = x
        data.attention_prior = torch.zeros(batch_size, num_nodes, num_nodes)
        data = mha.forward(data)
        assert data.attention_output is not None
        ref_mha_out = ref_mha.forward(x, x, x, need_weights=False)[0]
        assert torch.allclose(data.attention_output, ref_mha_out, rtol=0.3)

    def test_with_padding(self):
        torch.set_grad_enabled(False)
        embedding_dim = 4
        max_num_nodes = 3
        num_heads = 2
        batch_size = 2
        weights = torch.nn.Parameter(
            torch.concat([torch.arange(embedding_dim) * x for x in range(embedding_dim)])
            .reshape(embedding_dim, embedding_dim)
            .float()
        )

        mha = GraphormerMultiHeadAttention(num_heads, embedding_dim, 0.0)
        mha.linear_q.weight = weights
        mha.linear_k.weight = weights
        mha.linear_v.weight = weights
        mha.linear_v.bias = torch.nn.Parameter(torch.zeros_like(mha.linear_v.bias))
        mha.linear_out.weight = torch.nn.Parameter(weights)

        ref_mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, bias=False, batch_first=True)
        ref_mha.in_proj_weight[:embedding_dim, :] = weights
        ref_mha.in_proj_weight[embedding_dim : 2 * embedding_dim, :] = weights
        ref_mha.in_proj_weight[2 * embedding_dim :, :] = weights

        ref_mha.out_proj.weight = torch.nn.Parameter(weights)

        x = torch.Tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]],
                [[0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]
        )

        device = torch.device("cpu")
        data = ModelData(x, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), device)
        data.normalized_input = x
        data.attention_prior = torch.zeros(batch_size, max_num_nodes, max_num_nodes)
        data = mha.forward(data)
        assert data.attention_output is not None
        pad_mask = (x == 0.0).all(dim=-1)
        ref_mha_out = ref_mha.forward(x, x, x, need_weights=False, key_padding_mask=pad_mask)[0]
        ref_mha_out[pad_mask] = 0.0
        assert torch.allclose(
            data.attention_output, ref_mha_out, rtol=0.3
        ), f"mha: out {data.attention_output}\nref_mha_out: {ref_mha_out}"
