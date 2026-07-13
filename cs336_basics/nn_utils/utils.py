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
