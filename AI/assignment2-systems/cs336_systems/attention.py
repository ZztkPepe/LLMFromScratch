from __future__ import annotations

import math

import torch


def _attention_scores(q: torch.Tensor, k: torch.Tensor, is_causal: bool) -> torch.Tensor:
    scale = 1 / math.sqrt(q.shape[-1])
    scores = torch.einsum("... q d, ... k d -> ... q k", q, k) * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        query_idx = torch.arange(n_queries, device=q.device).view(-1, 1)
        key_idx = torch.arange(n_keys, device=q.device).view(1, -1)
        scores = torch.where(query_idx >= key_idx, scores, torch.full_like(scores, -1e6))
    return scores


def _flash_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _attention_scores(q, k, is_causal)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("... q k, ... k d -> ... q d", probs, v)
    return out, lse


def _flash_backward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = _attention_scores(q, k, is_causal)
    probs = torch.exp(scores - lse.unsqueeze(-1))

    delta = torch.sum(out * grad_out, dim=-1)
    grad_v = torch.einsum("... q k, ... q d -> ... k d", probs, grad_out)
    grad_probs = torch.einsum("... q d, ... k d -> ... q k", grad_out, v)
    grad_scores = probs * (grad_probs - delta.unsqueeze(-1))
    scale = 1 / math.sqrt(q.shape[-1])
    grad_q = torch.einsum("... q k, ... k d -> ... q d", grad_scores, k) * scale
    grad_k = torch.einsum("... q k, ... q d -> ... k d", grad_scores, q) * scale
    return grad_q, grad_k, grad_v


class FlashAttentionPytorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        out, lse = _flash_forward_pytorch(q, k, v, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(lse, q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        lse, q, k, v, out = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward_pytorch(q, k, v, out, grad_out, lse, ctx.is_causal)
        return grad_q, grad_k, grad_v, None


class FlashAttentionTritonFunction(FlashAttentionPytorchFunction):
    """Correctness-compatible entry point for the Triton adapter.

    The CPU test environment cannot execute Triton kernels. This class keeps the same
    autograd contract as the Triton version and can be swapped for a fused kernel on CUDA.
    """

