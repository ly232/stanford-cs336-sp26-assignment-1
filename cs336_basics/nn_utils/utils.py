import math

from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool, UInt8, Array
from typing import Iterable

import torch

def softmax(
    input: Float[torch.Tensor, '... vocab_size'],
    dim: int = -1) -> Float[torch.Tensor, '... vocab_size']:
    offset = torch.max(input, dim=dim, keepdim=True).values
    input = input - offset
    numerator = torch.exp(input)
    denominator = torch.sum(torch.exp(input), dim=dim, keepdim=True)
    return numerator / denominator

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    '''Implements scaled dot product attention.

    Formula:
      Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
    '''
    d_k = Q.shape[-1]
    input = einsum(
        Q, K,
        '... queries d_k, ... keys d_k -> ... queries keys'
    ) / math.sqrt(d_k)
    if mask is not None:
        input = input.masked_fill(~mask, float('-inf'))
    sm = softmax(
        input=input,
        dim=-1,
    )
    return einsum(
        sm, V,
        '... queries keys, ... keys dv -> ... queries dv'
    )

def cross_entropy(
    logits: Float[torch.Tensor, 'batch ... vocab'],
    targets: Int[torch.Tensor, 'batch ...']
) -> Float[torch.Tensor, '']:
    '''Calculates the cross entropy of a batch of *tokens*.

    Args:
        logits: token's tensor, shape is (batch, vocab).
        targets: vocab index of the ground truth, shape is (batch,)

    Returns:
        Cross entropy loss *of the single token*. averaged over batch.
        Shape is (1,).
    '''
    # Subtract baseline for numerical stability, aka "log sum exp trick".
    # See numeric_stability_memo.ipynb for details.
    logits = logits - torch.max(input=logits, dim=-1, keepdim=True).values
    logits_at_targets: Float[torch.Tensor, 'batch 1'] = torch.gather(
        input=logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    )
    log_probs = (
        logits_at_targets
        - torch.log(
            torch.sum(
                input=torch.exp(logits),
                dim=-1,
                keepdim=True,
            )
        )
    ).squeeze(-1)
    return -torch.mean(log_probs)

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if warmup_iters <= it <= cosine_cycle_iters:
        return (
            min_learning_rate
            + 0.5 * (
                1 + math.cos(
                    (it - warmup_iters)
                    / (cosine_cycle_iters - warmup_iters)
                    * math.pi)
            ) * (max_learning_rate - min_learning_rate)
        )
    return min_learning_rate

def clip_gradient(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
) -> None:
    '''Clips gradient in-place.'''
    global_norm = math.sqrt(
        sum(
            p.grad.norm() ** 2
            for p in parameters
            if p.grad is not None)
        )
    if global_norm > max_l2_norm:
        scale = max_l2_norm / (global_norm + 1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(scale)
