import math
import os
import typing

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import numpy as np
import numpy.typing as npt
import random
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    obj = {
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']
