from __future__ import annotations

import torch

from cs336_basics.nn_utils import softmax


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> list[int]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    device = next(model.parameters()).device
    generated = list(prompt_ids)
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        context = generated[-context_length:] if context_length is not None else generated
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1] / temperature
        probs = softmax(logits, dim=-1)

        if top_p is not None:
            probs = _apply_top_p(probs, top_p)

        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

    return generated


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[0] = True
    first_above = torch.nonzero(cumulative >= top_p, as_tuple=False)
    if len(first_above) > 0:
        keep[first_above[0].item()] = True

    filtered = torch.zeros_like(probs)
    kept_probs = sorted_probs * keep
    filtered.scatter_(0, sorted_indices, kept_probs)
    return filtered / filtered.sum()
