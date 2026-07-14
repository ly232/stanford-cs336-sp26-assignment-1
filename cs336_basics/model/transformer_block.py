import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

from cs336_basics.model.rms_norm import RmsNorm
from cs336_basics.model.multi_head_attention import MultiHeadAttention
from cs336_basics.model.positionwise_feedforward import PositionwiseFeedforward


class TransformerBlock(torch.nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.mha_prenorm_layer = RmsNorm(d_model)
        self.mha_layer = MultiHeadAttention(
            d_model, num_heads, theta, max_seq_len)
        self.swiglu_prenorm_layer = RmsNorm(d_model)
        self.swiglu_layer = PositionwiseFeedforward(d_model, d_ff)

    def initialize_weights(
        self,
        weights: dict[str, torch.Tensor]) -> None:
        '''Initializes weights in transformer block.
        
        Args:
            weights (dict[str, Tensor]):
                State dict of our reference implementation.
                The keys of this dictionary are:
                - `attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is (d_model, d_model).
                - `ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
        '''
        self.mha_layer.initialize_weights(
            q_weight=weights['attn.q_proj.weight'],
            k_weight=weights['attn.k_proj.weight'],
            v_weight=weights['attn.v_proj.weight'],
            o_weight=weights['attn.output_proj.weight'],
        )
        with torch.no_grad():
            self.mha_prenorm_layer.gains.copy_(weights['ln1.weight'])
            self.swiglu_prenorm_layer.gains.copy_(weights['ln2.weight'])
            self.swiglu_layer.w1.weights.copy_(weights['ffn.w1.weight'])
            self.swiglu_layer.w2.weights.copy_(weights['ffn.w2.weight'])
            self.swiglu_layer.w3.weights.copy_(weights['ffn.w3.weight'])

    def forward(
        self,
        x: Float[torch.Tensor, 'batch seq d_model'],
        token_positions: Float[torch.Tensor, '... seq'] | None = None,
    ) -> Float[torch.Tensor, '...']:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len).to(x.device)
        mha = x + self.mha_layer(
            self.mha_prenorm_layer(x),
            token_positions)
        swiglu = mha + self.swiglu_layer(
            self.swiglu_prenorm_layer(mha))
        return swiglu
