from collections.abc import Callable
from typing import Optional

import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            # Args are consistent with torch.optim.AdamW
            params: list[torch.nn.Parameter],
            lr: float | torch.Tensor = 1e-3,
            betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError('Invalid learning rate: {lr}')
        defaults = {
            'alpha': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'lmda': weight_decay,
        }
        # Note: momentum and velocity are kept as part of
        # self.state.
        super().__init__(params, defaults)

    def count_trainable_params(self):
        '''Returns number of trainable params for this optimizer.'''
        return sum(
            p.numel()
            for pg in self.param_groups
            for p in pg['params']
            if p.requires_grad
        )

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group['alpha']
            beta1, beta2 = group['beta1'], group['beta2']
            eps = group['eps']
            lmda = group['lmda']
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data

                # Extract state for this parameter.
                # torch.Tensor is hashable, so self.state[p]
                # returns the optimizer state tied to a particular
                # tensor param.
                state = self.state[p]
                if 't' not in state:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                state['t'] += 1
                t, m, v = state['t'], state['m'], state['v']

                # p.data -= alpha * lmda * p
                p.data.mul_(1 - alpha * lmda)  # in-place is more efficient.
                alpha_t = (
                    alpha 
                    * math.sqrt(1 - beta2 ** t)
                    / (1 - beta1 ** t)
                )
                # state['m'] = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_((1 - beta1) * g)
                # state['v'] = beta2 * v + (1 - beta2) * torch.square(g)
                v.mul_(beta2).add_((1 - beta2) * torch.square(g))
                # p.data -= alpha_t * state['m'] / (torch.sqrt(state['v']) + eps)
                p.data.add_(-alpha_t * m / (torch.sqrt(v) + eps))

        return loss
