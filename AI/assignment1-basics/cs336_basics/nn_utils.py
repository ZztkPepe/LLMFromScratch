from __future__ import annotations

from collections.abc import Iterable
import math

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_normalizer = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(grad.detach() ** 2) for grad in grads))
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)


def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
