from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    offsets = np.arange(context_length)
    x = dataset[starts[:, None] + offsets]
    y = dataset[starts[:, None] + offsets + 1]
    return (
        torch.as_tensor(x, dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )
