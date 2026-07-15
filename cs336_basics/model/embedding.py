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
        # # Inefficient implementation (kept for learning purpose):
        # # Onehot encode the input token ids.
        # onehot: Float[torch.Tensor, 'batch seq vocab_size']\
        #     = torch.nn.functional.one_hot(
        #         x, num_classes=self.vocab_size)
        # return einsum(
        #     # NOTE: einsum requires 2 tensors to share same dtype.
        #     self.weights, onehot.to(self.weights.dtype),
        #     'vocab_size d_model, batch seq vocab_size -> batch seq d_model'
        # )

        # Directly apply tensor fancy indexing:
        # * weights.shape is (vocab_size, d_model)
        # * x.shape is (batch, seq)
        # * weights[x] is thus (batch, seq, d_model)
        #
        # See tensor_indexing_memo.ipynb. weights[x] can
        # be thought of as function composition, where
        # the output of x (actual indices of each vocab
        # position) can be directly feed into weights as
        # query to find the corresponding embedding value.
        return self.weights[x]
