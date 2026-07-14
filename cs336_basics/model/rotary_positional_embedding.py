import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array

import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        # NOTE: max_seq_len >= seq_len. max_seq_len is the max position value
        # we'll ever see from the whole document, not just a single input
        # tensor. For example, max_seq_len may be 4096, but for train/test the
        # input tensor may be chunked into seq_len of much smaller size, e.g.
        # seq_len=12. In this case, each input tensor x would have seq
        # numbered 0..11, and no way for us to tie to position of the document.
        # That's also why in the `forward` method below, we also need an arg
        # for `token_positions`.
        max_seq_len: int,
    ):
        super().__init__()
        assert d_k % 2 == 0, f'RoPE only supports even dimenions; got {d_k}'
        # Space out angle params: angle[i][k] = i / theta ^ ((2*k-2)/d_k)
        # Note there are d_k // 2 different angle values, repeated for
        # max_seq_len times for each positional encoding.
        i_range = torch.linspace(0, max_seq_len - 1, max_seq_len).unsqueeze(-1)
        k_range = torch.linspace(1, d_k // 2, d_k // 2).unsqueeze(0)
        theta_exponent = (2 * k_range - 2) / d_k
        angle: Float[torch.Tensor, 'max_seq_len d_k_half'] =\
            i_range / torch.pow(theta, theta_exponent)
        # Calculate rotational matrix elements then register each quadrant into
        # buffer. Each buffered tensor has shape (max_seq_len, d_k/2)
        #
        # PyTorch specific notes:
        # - register_buffer(<name>, <tensor>, persistent=False) means we can
        #   later retrieve the registered tensor via self.<name>.
        # - persistent=False means when we call torch.save(model.state_dict())
        #   we won't store them.
        # - register_buffer only suports tensor elements, so we cannot store a
        #   single rot_mat python object.
        # - use register_buffer instead of free-floating python object or
        #   nn.Parameter, because (a) free-floating python object won't respond
        #   to model.to_device('cuda'), making them less efficient during
        #   retrival, and (b) nn.Parameter would make them part of model
        #   parameter and by default participates in autograd, and even if we
        #   can manually disable gradient tracking, it still stores to
        #   state_dict which we don't want.
        self.register_buffer('cos_table', torch.cos(angle), persistent=False)
        self.register_buffer('sin_table', torch.sin(angle), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, '... seq d_k'],
        token_positions: Float[torch.Tensor, '... seq'],
    ) -> Float[torch.Tensor, '... seq d_k']:
        x = rearrange(
            x,
            '... seq (d_k half) -> ... seq d_k half',
            half=2
        )  # shape = (..., seq, d_k/2, 2)
        # Partition x into first and second halfs, then take linear combination
        # with respect to rotation matrix.
        first, second = x[..., 0].squeeze(-1), x[..., 1].squeeze(-1)
        # Note the following shapes:
        # * self.cos_table: (max_seq_len, d_k/2)
        # * self.sin_table: (max_seq_len, d_k/2)
        # * token_positions: (..., seq)
        #
        # Directly applying token_positions as an index against the source
        # tensors cos_table and sin_table works by broadcast. In general if:
        #   result = source[index]
        # then:
        #   result.shape = (*index.shape, *source[1:].shape)
        # here:
        #   index is token_positions, shape is (..., seq)
        #   source is cos_table/sin_table, shape is (max_seq_len, d_k/2)
        # so after indexing, shape becomes (..., seq, d_k/2)
        #
        # Another way to think about it:
        #   for each (b, s) in token_positions:
        #     # b goes from 1..batch_size, s goes from 1..seq_len
        #     pos = token_positions[b, s]  # pos is now a scalar
        #     # CRUCIAL #1: out preserved prefix dimensions b & s.
        #     # CRUCIAL #2: tensor indexing always pivots at axis 0.
        #     out[b, s, :] = cos_table[pos, :]
        # this is also why out has shape (*index.shape, *source[1:].shape)
        #
        # A more mathematical mental model: FUNCTIONAL COMPOSITION!!!:
        # * Each Tensor can be thought of as a mathematical function, where the
        #   input is the first n dimensions, and range is the last remaining
        #   dimensions (n can vary from 1 to n-1).
        # * Since we queried `source[.]` (and not `source[., ., ., ...]`), we
        #   only consider axis-0 as the input of source tensor function, which
        #   means the remainder dimensions are function outputs.
        # * This also implies for the `index` tensor, the function *output*
        #   must match the function input of the source tensor, which is the
        #   scalar value of the index tensor (when indexed over *all* axis).
        # * Combining these arguments, we arrive at the following functions:
        #   - source: Z (from axis-0) -> [axis-1, ..., axis-n]
        #   - index: [axis-0, ..., axis-(m-1)] -> Z (index tensor values)
        #   - source o index: [axis-0, ..., axis-(m-1)] -> [axis-1, ..., axis-n]
        # which is why output has shape (*index.shape, *source[1:].shape)
        cos = self.cos_table[token_positions]  # shape = (..., seq, d_k/2)
        sin = self.sin_table[token_positions]  # shape = (..., seq, d_k/2)
        first, second = (
            cos * first - sin * second,
            sin * first + cos * second,
        )
        x = rearrange(
            torch.stack((first, second), dim=-1),
            '... seq d_k half -> ... seq (d_k half)'
        )
        return x
