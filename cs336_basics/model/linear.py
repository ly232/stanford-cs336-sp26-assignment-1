import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        super().__init__()
        mu = 0.0
        sigma = math.sqrt(2.0 / (in_features + out_features))
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                tensor=torch.randn(
                    out_features,
                    in_features,
                    device=device,
                    dtype=dtype,) * sigma,
                a=-3*sigma,
                b=3*sigma,
            ),
        )

    def forward(
        self,
        x: Float[torch.Tensor, '... d_in'])\
        -> Float[torch.Tensor, '... d_out']:
        return einsum(
            self.weights, x,
            'd_out d_in, ... d_in -> ... d_out',
        )
