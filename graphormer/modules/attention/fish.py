import torch
import torch.autograd as autograd
import torch.nn as nn


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
        self.p = nn.Parameter(torch.randn(self.num_global_heads, self.num_local_heads, requires_grad=True))
        self.mish = nn.Mish()

        self.linear_out = nn.Linear(self.num_local_heads * self.head_size, self.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        encoding_bias: torch.Tensor,
    ):
        """
        :param x: node embedding, shape: (batch_size, num_nodes, hidden_dim)
        :param spatial_encoding: spatial encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :param edge_encoding: edge encoding matrix, shape (batch_size, max_graph_size, max_graph_size)
        :return: torch.Tensor, node embeddings after all attention heads
        """

        batch_size = x.shape[0]
        max_subgraph_size = x.shape[1]
        # (batch_size, 1, max_seq_len, max_seq_len)
        bias = encoding_bias.view(batch_size, 1, max_subgraph_size, max_subgraph_size).contiguous()

        global_q_x = (
            self.global_q(x).contiguous().view(batch_size, max_subgraph_size, self.num_global_heads, self.head_size)
        )
        global_k_x = (
            self.global_k(x).contiguous().view(batch_size, max_subgraph_size, self.num_global_heads, self.head_size)
        )
        v_x = self.local_v(x).contiguous().view(batch_size, max_subgraph_size, self.num_local_heads, self.head_size)

        # b: batch_size
        # n, m: max_subgraph_size
        # g: num_global_heads
        # (batch_size, num_global_heads, max_seq_len, max_seq_len)
        g_k = torch.einsum("bngd,bmgd->bgnm", global_q_x, global_k_x)
        eps = torch.randn_like(g_k).to(x.device)
        pad_mask = torch.all(g_k == 0, dim=-1)
        # (1, num_global_heads, 1, 1)
        sigma_sq = self.sigma.square().reshape(1, self.num_global_heads, 1, 1).contiguous()
        # (batch_size, num_global_heads, head_size, max_seq_len, max_seq_len)
        sigma_eps = sigma_sq * eps
        a = g_k + sigma_eps
        a[pad_mask, :] = 0.0

        # b: batch_size
        # n, m: max_subgraph_size
        # g: num_global_heads
        # l: num_local_heads
        # d: head_size
        # (batch_size, num_local_heads, max_seq_len, max_seq_len)
        a = torch.einsum("bgnm,gl->blnm", a, self.p)

        pad_mask = torch.all(a == 0, dim=-1)
        a = self.mish(a)
        a *= self.scale
        a += bias

        a[pad_mask] = float("-inf")
        a = torch.softmax(a, dim=-1)
        a = torch.nan_to_num(a)

        # b: batch_size
        # n, m: max_subgraph_size
        # g: num_global_heads
        # l: num_local_heads
        # d: head_size
        # (batch_size, max_subgraph_size, num_local_heads, head_size)
        a = torch.einsum("blnm,bmld->bnld", a, v_x)
        a = self.att_dropout(a)
        attn = a.reshape(batch_size, max_subgraph_size, self.num_local_heads * self.head_size).contiguous()
        return self.linear_out(attn)
