import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import numpy as np
import numpy.typing as npt
import random
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[
    Float[torch.Tensor, 'batch_size context_length'],
    Float[torch.Tensor, 'batch_size context_length'],
]:
    inputs = []
    targets = []
    for _ in range(batch_size):
        start = random.randint(0, len(dataset) - context_length - 1)
        inputs.append(dataset[start:start+context_length])
        targets.append(dataset[start+1:start+context_length+1])
    inputs = np.array(inputs)
    targets = np.array(targets)

    return (
        torch.tensor(inputs, device=device),
        torch.tensor(targets, device=device),
    )
