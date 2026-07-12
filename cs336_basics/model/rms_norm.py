import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

class RmsNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains: Float[torch.Tensor, 'd_model'] =\
            torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[torch.Tensor, 'batch seq d_model'])\
        -> Float[torch.Tensor, 'batch seq d_model']:
        rms: Float[torch.Tensor, 'batch seq 1'] =\
            torch.sqrt(
                1 / self.d_model * torch.sum(
                    torch.square(x),
                    dim=-1,
                    keepdims=True,
                ) + self.eps
            )
        return einsum(
            x / rms, self.gains,
            'batch seq d_model, d_model -> batch seq d_model'
        )
