import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

class PositionwiseFeedforward(torch.nn.Module):
    '''Positionwise FFN, aka SWIGLU.'''

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = self._gaussian_init_weights(d_ff, d_model)
        self.w2 = self._gaussian_init_weights(d_model, d_ff)
        self.w3 = self._gaussian_init_weights(d_ff, d_model)

    def _gaussian_init_weights(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None) -> torch.nn.Parameter:
        mu = 0.0
        sigma = math.sqrt(2.0 / (in_features + out_features))
        return torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                tensor=torch.randn(
                    in_features,
                    out_features,
                    device=device,
                    dtype=dtype,) * sigma,
                a=-3*sigma,
                b=3*sigma,
            ),
        )

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def _swiglu(self, x: Float[torch.Tensor, 'batch seq d_model'])\
        -> Float[torch.Tensor, 'batch seq d_model']:
        w1_x = einsum(
            self.w1, x,
            'd_ff d_model, batch seq d_model -> batch seq d_ff'
        )
        silu_w1_x = self._silu(w1_x)
        w3_x = einsum(
            self.w3, x,
            'd_ff d_model, batch seq d_model -> batch seq d_ff'
        )
        return einsum(
            self.w2, silu_w1_x * w3_x,
            'd_model d_ff, batch seq d_ff -> batch seq d_model'
        )


    def forward(self, x: Float[torch.Tensor, 'batch seq d_model'])\
        -> Float[torch.Tensor, 'batch seq d_model']:
        return self._swiglu(x)
