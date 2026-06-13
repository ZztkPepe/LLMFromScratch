from __future__ import annotations

import math

import torch
from torch import nn

from cs336_basics.nn_utils import silu, softmax


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        normalized = x_float / rms
        return (normalized * self.weight.to(torch.float32)).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = torch.einsum("... q d, ... k d -> ... q k", q, k) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.einsum("... q k, ... k d -> ... q d", softmax(scores, dim=-1), v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], self.num_heads, self.d_head)
        return x.transpose(-3, -2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-3, -2).contiguous()
        return x.reshape(*x.shape[:-2], self.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        return self.output_proj(self._merge_heads(attended))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context length {self.context_length}")

        token_positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x))
