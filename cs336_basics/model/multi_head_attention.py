import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

from cs336_basics.model.linear import Linear
from cs336_basics.model.rotary_positional_embedding import RotaryPositionalEmbedding
from cs336_basics.nn_utils import utils

class MultiHeadAttention(torch.nn.Module):
    '''Implements multi-head attention.

    MHA is a *concatenation* over multiple HA's, where
    each HA can be thought of as a low-rank projection
    over the entire attention matrix. Specifically, per-head
    Q K V are all d_model(in) by d_k(out), whereas full attention
    Q K V are all d_model by d_model, where d_model = d_k * num_heads.
    '''
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float | None=None,
        max_seq_len: int | None=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        # For each Wq_i.
        self.q_heads = torch.nn.ModuleList([
            Linear(self.d_model, self.d_k)
            for _ in range(num_heads)
        ])
        # For each Wk_i.
        self.k_heads = torch.nn.ModuleList([
            Linear(self.d_model, self.d_k)
            for _ in range(num_heads)
        ])
        # For each Wv_i.
        self.v_heads = torch.nn.ModuleList([
            Linear(self.d_model, self.d_k)
            for _ in range(num_heads)
        ])
        # For Wo.
        self.output_layer = Linear(d_model, d_model)
        # For RoPE.
        if rope_theta is not None and max_seq_len is not None:
            self.rope_layer= RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope_layer = None

    def forward(
        self,
        x: Float[torch.Tensor, '... sequence_length d_model'],
        token_positions:\
            Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, '... sequence_length d_model']:
        seq_len = x.shape[-2]
        single_head_attention_mask = (
            torch
                .ones(seq_len, seq_len)
                .tril()
                .bool()
                .to(x.device)
        )

        # Calculate each head attentions.
        attentions = []
        for h in range(self.num_heads):
            q, k, v = self.q_heads[h](x), self.k_heads[h](x), self.v_heads[h](x)
            
            # print(f'''[single head attention internal tensor shapes]
            # x.shape: {x.shape}
            # q.shape: {q.shape}
            # k.shape: {k.shape}
            # v.shape: {v.shape}
            # mask.shape: {single_head_attention_mask.shape}
            # ''')
            
            # Apply RoPE if applicable.
            if token_positions is not None:
                assert self.rope_layer is not None
                q, k = (
                    self.rope_layer(q, token_positions),
                    self.rope_layer(k, token_positions),
                )
            single_head_attention: Float[torch.Tensor, '... d_k d_k'] =\
                utils.scaled_dot_product_attention(
                    q, k, v,
                    single_head_attention_mask)
            attentions.append(single_head_attention)
        # Concat single head attentions. Heads were splitted earlier at the
        # last dim, so concat at dim=-1.
        muti_head = torch.concat(attentions, dim=-1)
        # print(f'!!! x.shape: {x.shape}')
        # print(f'!!! attentions[0].shape: {attentions[0].shape}')
        # print(f'!!! muti_head.shape: {muti_head.shape}')
        # print(f'!!! output_layer.weights.shape: {self.output_layer.weights.shape}')
        muti_head_self_attention = self.output_layer(muti_head)
        return muti_head_self_attention
