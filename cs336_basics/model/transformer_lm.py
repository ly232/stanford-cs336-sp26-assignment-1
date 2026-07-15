import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.transformer_block import TransformerBlock
from cs336_basics.model.rms_norm import RmsNorm
from cs336_basics.model.linear import Linear


class TransformerLanguageModel(torch.nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_layer =\
            Embedding(vocab_size=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ])
        self.norm_layer = RmsNorm(d_model=d_model)
        self.output_embedding_layer = Linear(
            in_features=d_model,
            out_features=vocab_size,
        )

    def initialize_weights(
        self,
        weights: dict[str, torch.Tensor],
    ) -> None:
        '''Initializes weights in this Transformer LM.

        Args:
            weights (dict[str, Tensor]):
                State dict of our reference implementation. {num_layers} refers to an
                integer between `0` and `num_layers - 1` (the layer index).
                The keys of this dictionary are:
                - `token_embeddings.weight`
                    Token embedding matrix. Shape is (vocab_size, d_model).
                - `layers.{num_layers}.attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is ((d_model / num_heads) * num_heads, d_model).
                - `layers.{num_layers}.ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `layers.{num_layers}.ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `layers.{num_layers}.ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `layers.{num_layers}.ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `layers.{num_layers}.ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ln_final.weight`
                    Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                    Shape is (d_model, ).
                - `lm_head.weight`
                    Weights of the language model output embedding.
                    Shape is (vocab_size, d_model).
        '''
        with torch.no_grad():
            self.embedding_layer.weights.copy_(
                weights['token_embeddings.weight'],
            )
            self.norm_layer.gains.copy_(
                weights['ln_final.weight'],
            )
            self.output_embedding_layer.weights.copy_(
                weights['lm_head.weight'],
            )
        for i in range(self.num_layers):
            block_weights = {
                'attn.q_proj.weight': ...,
                'attn.k_proj.weight': ...,
                'attn.v_proj.weight': ...,
                'attn.output_proj.weight': ...,
                'ln1.weight': ...,
                'ffn.w1.weight': ...,
                'ffn.w2.weight': ...,
                'ffn.w3.weight': ...,
                'ln2.weight': ...,
            }
            for key in block_weights:
                block_weights[key] = weights[f'layers.{i}.{key}']
            self.transformer_blocks[i].initialize_weights(block_weights)

    def forward(self, in_indices: Float[torch.Tensor, 'batch seq'])\
        -> Float[torch.Tensor, 'batch seq vocab_size']:
        x: Float[torch.Tensor, 'batch seq d_model'] =\
            self.embedding_layer(in_indices)
        for block in self.transformer_blocks:
            x: Float[torch.Tensor, 'batch seq d_model'] = block(x)
        x: Float[torch.Tensor, 'batch seq d_model'] = self.norm_layer(x)
        x: Float[torch.Tensor, 'batch seq vocab_size'] =\
            self.output_embedding_layer(x)
        return x
