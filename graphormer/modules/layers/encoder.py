from typing import Optional
import torch
from torch import nn

from graphormer.modules.norm import CRMSNorm, MaxNorm, RMSNorm
from graphormer.modules.attention import (
    GraphormerMultiHeadAttention,
    GraphormerLinearAttention,
    GraphormerFishAttention,
)
from graphormer.modules.layers import FeedForwardNetwork
from graphormer.config.options import AttentionType, NormType


class GraphormerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: Optional[int] = None,
        n_global_heads: Optional[int] = None,
        n_local_heads: Optional[int] = None,
        ffn_dim=80,
        ffn_dropout=0.1,
        attn_dropout=0.1,
        norm_type: NormType = NormType.LAYER,
        attention_type: AttentionType = AttentionType.MHA,
    ):
        """
        :param hidden_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_global_heads = n_global_heads
        self.n_local_heads = n_local_heads
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.ffn_dim = ffn_dim

        match attention_type:
            case AttentionType.MHA:
                if self.n_heads is None:
                    raise AttributeError("n_heads must be defined for GraphormerMultiHeadAttention")
                self.attention = GraphormerMultiHeadAttention(
                    num_heads=self.n_heads,
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.attn_dropout,
                )
            case AttentionType.LINEAR:
                if self.n_heads is None:
                    raise AttributeError("n_heads must be defined for GraphormerMultiHeadAttention")
                self.attention = GraphormerLinearAttention(
                    num_heads=self.n_heads,
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.attn_dropout,
                )

            case AttentionType.FISH:
                if self.n_global_heads is None:
                    raise AttributeError("n_global_heads must be defined for GraphormerFishAttention")
                if self.n_local_heads is None:
                    raise AttributeError("n_local_heads must be defined for GraphormerFishAttention")
                self.attention = GraphormerFishAttention(
                    num_global_heads=self.n_global_heads,
                    num_local_heads=self.n_local_heads,
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.attn_dropout,
                )

        match norm_type:
            case NormType.LAYER:
                self.n1 = nn.LayerNorm(self.hidden_dim)
                self.n2 = nn.LayerNorm(self.hidden_dim)
            case NormType.RMS:
                self.n1 = RMSNorm(self.hidden_dim)
                self.n2 = RMSNorm(self.hidden_dim)
            case NormType.CRMS:
                self.n1 = CRMSNorm(self.hidden_dim)
                self.n2 = CRMSNorm(self.hidden_dim)
            case NormType.MAX:
                self.n1 = MaxNorm(self.hidden_dim)
                self.n2 = MaxNorm(self.hidden_dim)
            case NormType.NONE:
                self.n1 = nn.Identity()
                self.n2 = nn.Identity()

        self.ffn = FeedForwardNetwork(self.hidden_dim, self.ffn_dim, self.hidden_dim, self.ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoding_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements forward pass of the Graphormer encoder layer.

        The correct sequence for operations with residual connections and layer rescaling is:
        1. Normalization (N) is applied to the input.
        2. The rescaled input is passed to MHA or FFN.
        3. The MHA or FFN output is added to the original input, x (residual connection).
        4. The combined (MHA_out + x) output goes through the ffn.

        This results in the following operations:
        h′(l) = MHA(N(h(l−1))) + h(l−1)
        h(l) = FFN(N(h′(l))) + h′(l)

        :param x: node embedding
        :param spatial_encoding: spatial encoding
        :param edge_encoding: encoding of the edges
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        att_input = self.n1(x)
        att_output = self.attention(att_input, encoding_bias) + x
        pad_mask = torch.any(att_output == 0, dim=-1)

        ffn_input = self.n2(x)
        ffn_output = self.ffn(ffn_input) + att_output
        ffn_output[pad_mask] = 0

        return ffn_output
