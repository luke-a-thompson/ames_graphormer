import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class GraphormerFishAttention(nn.Module):
    """
    This class implements techniques from the following paper:
    Tan M. Nguyen, Tam Nguyen, Hai Do, Khai Nguyen, Vishwanath Saragadam, Minh Pham, Duy Khuong Nguyen, Nhat Ho, and Stanley J. Osher.
    "Improving Transformer with an Admixture of Attention Heads."
    Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS 2022).
    Available at: https://proceedings.neurips.cc/paper_files/paper/2022/file/b2e4edd53059e24002a0c916d75cc9a3-Paper-Conference.pdf
    """

    def __init__(
        self,
        num_global_heads: int,
        num_local_heads: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_global_heads = num_global_heads
        self.num_local_heads = num_local_heads

        self.scale = hidden_dim**-0.5
        self.hidden_dim = hidden_dim
        assert (
            self.hidden_dim % self.num_global_heads == 0
        ), f"hidden_dim {
            self.hidden_dim} must be divisible by num_global_heads {self.num_global_heads}"
        assert (
            self.num_global_heads <= self.num_local_heads
        ), f"num_global_heads {self.num_global_heads} should be less than num_local_heads {self.num_local_heads}"

        self.head_size = self.hidden_dim // self.num_global_heads
        self.global_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.global_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.local_v = nn.Linear(self.hidden_dim, self.num_local_heads * self.head_size, bias=True)
        self.att_dropout = nn.Dropout(dropout_rate)

        self.sigma = nn.Parameter(0.1 * torch.ones(self.num_global_heads, requires_grad=True))
        self.global_to_local_proj = nn.Sequential(
            nn.Linear(self.num_global_heads, self.num_local_heads),
            nn.Mish(),
            nn.Linear(self.num_local_heads, self.num_local_heads),
        )

        self.linear_out = nn.Linear(self.num_local_heads * self.head_size, self.hidden_dim, bias=False)

    def forward(
        self,
        data: ModelData,
    ):
        """
        :param x: node embedding, shape: (batch_size, num_nodes, hidden_dim)
        :param spatial_encoding: spatial encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :param edge_encoding: edge encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :return: torch.Tensor, node embeddings after all attention heads
        """

        assert data.normalized_input is not None
        assert data.attention_prior is not None
        x = data.normalized_input
        prior = data.attention_prior.permute(0, 2, 3, 1).contiguous()  # Permute to (batch_size, max_subgraph_size, max_subgraph_size, num_global_heads)
        batch_size = x.shape[0]
        max_subgraph_size = x.shape[1]

        global_q_x = (
            self.global_q(x).contiguous().view(batch_size, max_subgraph_size, self.num_global_heads, self.head_size)
        )
        global_k_x = (
            self.global_k(x).contiguous().view(batch_size, max_subgraph_size, self.num_global_heads, self.head_size)
        )
        v_x = self.local_v(x).contiguous().view(batch_size, max_subgraph_size, self.num_local_heads, self.head_size)

        # (1, num_global_heads, 1, 1)
        sigma_sq = self.sigma.square().reshape(1, 1, 1, self.num_global_heads).contiguous()

        # b: batch_size
        # n, m: max_subgraph_size
        # g: num_global_heads
        # (batch_size, max_seq_len, max_seq_len, num_global_heads)
        g_k = torch.einsum("bngd,bmgd->bnmg", global_q_x, global_k_x)
        eps = torch.randn_like(g_k).to(x.device)
        pad_mask = torch.all(g_k == 0, dim=-1)
        sigma_eps = sigma_sq * eps
        a = g_k + sigma_eps
        a[pad_mask, :] = 0.0

        # b: batch_size
        # n, m: max_subgraph_size
        # g: num_global_heads
        # l: num_local_heads
        # d: head_size
        # (batch_size, max_seq_len, max_seq_len, num_local_heads)
        a = self.global_to_local_proj(a)
        a *= self.scale
        a += prior
        a[pad_mask] = float("-inf")

        a = torch.softmax(a, dim=-1)
        a = torch.nan_to_num(a)

        # b: batch_size
        # n, m: max_subgraph_size
        # g: num_global_heads
        # l: num_local_heads
        # d: head_size
        # (batch_size, max_subgraph_size, num_local_heads, head_size)
        a = torch.einsum("bnml,bmld->bnld", a, v_x)
        a = self.att_dropout(a)
        attn = a.reshape(batch_size, max_subgraph_size, self.num_local_heads * self.head_size).contiguous()
        data.attention_output = self.linear_out(attn)
        return data
