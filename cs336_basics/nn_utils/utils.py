import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

def softmax(input: torch.Tensor, dim: int) -> torch.Tensor:
    offset = torch.max(input, dim=dim, keepdim=True).values
    input = input - offset
    numerator = torch.exp(input)
    denominator = torch.sum(torch.exp(input), dim=dim, keepdim=True)
    return numerator / denominator

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    '''Implements scaled dot product attention.

    Formula:
      Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
    '''
    d_k = Q.shape[-1]
    input = einsum(
        Q, K,
        '... queries d_k, ... keys d_k -> ... queries keys'
    ) / math.sqrt(d_k)
    if mask is not None:
        input = input.masked_fill(~mask, float('-inf'))
    sm = softmax(
        input=input,
        dim=-1,
    )
    return einsum(
        sm, V,
        '... queries keys, ... keys dv -> ... queries dv'
    )
