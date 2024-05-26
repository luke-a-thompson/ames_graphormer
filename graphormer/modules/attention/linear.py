import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class GraphormerLinearAttention(nn.Module):
    """
    @inproceedings{katharopoulos_et_al_2020,
        author = {Katharopoulos, A. and Vyas, A. and Pappas, N. and Fleuret, F.},
        title = {Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention},
        booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
        year = {2020}
    }
    """

    def __init__(self, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1, eps: float = 1e-09):
        """
        :param num_heads: number of attention heads
        :param hidden_dim: node feature matrix input number of dimension
        """
        super().__init__()
        self.num_heads = num_heads

        self.hidden_dim = hidden_dim
        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim {
            hidden_dim} must be divisible by num_heads {num_heads}"
        self.head_size = hidden_dim // num_heads
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att_dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()
        self.feature_map = lambda x: self.elu(x) + 1.0
        self.eps = eps

        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, data: ModelData) -> ModelData:
        """
        :param x: input, shape (batch_size, max_subgraph_size, node_embed_dim)
        :param encoding_bias: the bias term, shape (batch_size, max_subgraph_size, max_subgraph_size)
        """
        assert data.normalized_input is not None
        assert data.attention_prior is not None
        x = data.normalized_input
        prior = data.attention_prior
        batch_size = x.shape[0]
        max_subgraph_size = x.shape[1]
        # (batch_size, max_subgraph_size, 1, 1)
        bias = prior.mean(dim=-1).view(batch_size, max_subgraph_size, 1, 1)

        q_x = self.feature_map(
            self.linear_q(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size) + bias
        ).contiguous()
        padding_mask = torch.all(q_x == 0, dim=-1)
        k_x = self.feature_map(
            self.linear_k(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size)
        ).contiguous()
        v_x = self.feature_map(
            self.linear_v(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size)
        ).contiguous()

        # b: batch_size
        # n: max_subgraph_size
        # h: num_heads
        # d: k_head_size
        # m: v_head_size
        # (batch_size, num_heads, v_head_size, k_head_size)
        k_v = torch.einsum("bnhd,bnhm->bhmd", k_x, v_x)

        # b: batch_size
        # n: max_subgraph_size
        # h: num_heads
        # d: head_size
        # (batch_size, max_subgraph_size, num_heads)
        z = (torch.einsum("bnhd,bhd->bnh", q_x, k_x.sum(dim=1)) + self.eps).reciprocal()

        # b: batch_size
        # n: max_subgraph_size
        # h: num_heads
        # d: head_size
        # m: v_head_size
        # (batch_size, max_subgraph_size, num_heads, v_head_size)
        attn = torch.einsum("bnhd,bhmd,bnh->bnhm", q_x, k_v, z)
        attn[padding_mask] = 0.0

        attn = self.att_dropout(attn)
        attn = attn.contiguous().view(batch_size, max_subgraph_size, self.num_heads * self.head_size)
        data.attention_output = self.linear_out(attn)
        return data
