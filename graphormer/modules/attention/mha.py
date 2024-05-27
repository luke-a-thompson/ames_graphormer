import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        """
        :param num_heads: number of attention heads
        :param d_x: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.scale = self.hidden_dim**-0.5
        assert (
            self.hidden_dim % self.num_heads == 0
        ), f"hidden_dim {
            self.hidden_dim} must be divisible by num_heads {self.num_heads}"
        self.head_size = self.hidden_dim // self.num_heads
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.linear_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.att_dropout = nn.Dropout(dropout_rate)

        self.linear_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, data: ModelData) -> ModelData:
        """
        :param x: node embedding, shape: (batch_size, num_nodes, hidden_dim)
        :param encoding_bias: spatial encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :return: torch.Tensor, node embeddings after all attention heads
        """
        assert data.normalized_input is not None
        assert data.attention_prior is not None

        x = data.normalized_input
        prior = data.attention_prior
        batch_size = x.shape[0]
        max_subgraph_size = x.shape[1]

        q_x = self.linear_q(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size)
        k_x = self.linear_k(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size)
        v_x = self.linear_v(x).view(batch_size, max_subgraph_size, self.num_heads, self.head_size)

        # b: batch_size
        # n, m: max_subgraph_size
        # h: num_heads
        # d: head_size
        # (batch_size, num_heads, max_subgraph_size, max_subgraph_size)
        a = torch.einsum("bnhd,bmhd->bhnm", q_x, k_x)
        pad_mask = torch.all(a == 0, dim=-1)
        a = a * self.scale

        a = a + prior
        a[pad_mask] = float("-inf")
        a = torch.softmax(a, dim=-1)
        a = torch.nan_to_num(a)
        # b: batch_size
        # n, m: max_subgraph_size
        # h: num_heads
        # d: head_size
        # (batch_size, max_subgraph_size, num_heads, head_size)
        a = torch.einsum("bhnm,bmhd->bnhd", a, v_x)
        a = self.att_dropout(a)
        attn = a.contiguous().view(batch_size, max_subgraph_size, self.num_heads * self.head_size)
        data.attention_output = self.linear_out(attn)
        return data
