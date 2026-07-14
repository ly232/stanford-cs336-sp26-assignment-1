import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

from cs336_basics.model.linear import Linear

class PositionwiseFeedforward(torch.nn.Module):
    '''Positionwise FFN, aka SWIGLU.'''

    def __init__(
        self,
        d_model: int,
        d_ff: int):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def _swiglu(self, x: Float[torch.Tensor, 'batch seq d_model'])\
        -> Float[torch.Tensor, 'batch seq d_model']:
        silu_w1_x = self._silu(self.w1(x))
        w3_x = self.w3(x)
        return self.w2(silu_w1_x * w3_x)

    def forward(self, x: Float[torch.Tensor, 'batch seq d_model'])\
        -> Float[torch.Tensor, 'batch seq d_model']:
        return self._swiglu(x)
