from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

class Embedding(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                tensor=torch.randn(vocab_size, embedding_dim),
                a=-3,
                b=3,
            ),
        )
        self.vocab_size = vocab_size

    def forward(self, x: Float[torch.Tensor, 'batch seq'])\
        -> Float[torch.Tensor, 'batch seq d_model']:
        # Onehot encode the input token ids.
        onehot: Float[torch.Tensor, 'batch seq vocab_size']\
            = torch.nn.functional.one_hot(
                x, num_classes=self.vocab_size)
        # print(f'''shapes and dtype:
        # x: {x.shape}, {x.dtype}
        # onehot: {onehot.shape}, {onehot.dtype}
        # weights: {self.weights.shape}, {self.weights.dtype}
        # ''')
        return einsum(
            # NOTE: einsum requires 2 tensors to share same dtype.
            self.weights, onehot.to(self.weights.dtype),
            'vocab_size d_model, batch seq vocab_size -> batch seq d_model'
        )
