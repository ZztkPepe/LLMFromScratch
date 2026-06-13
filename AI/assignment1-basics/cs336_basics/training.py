from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.nn_utils import clip_gradients, cosine_lr_schedule, cross_entropy
from cs336_basics.serialization import save_checkpoint


@dataclass
class TrainingConfig:
    batch_size: int
    context_length: int
    max_iters: int
    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int
    eval_interval: int = 100
    eval_iters: int = 10
    grad_clip: float | None = None
    checkpoint_path: str | None = None
    device: str = "cpu"


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    config: TrainingConfig,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        x, y = get_batch(dataset, config.batch_size, config.context_length, config.device)
        logits = model(x)
        losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    valid_data: np.ndarray | None,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    model.to(config.device)
    model.train()
    log: list[dict[str, float]] = []
    start_time = time.time()

    for iteration in range(config.max_iters):
        lr = cosine_lr_schedule(
            iteration,
            config.max_lr,
            config.min_lr,
            config.warmup_iters,
            config.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.batch_size, config.context_length, config.device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip is not None:
            clip_gradients(model.parameters(), config.grad_clip)
        optimizer.step()

        if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
            entry = {
                "iteration": float(iteration),
                "train_loss": float(loss.item()),
                "learning_rate": float(lr),
                "elapsed_seconds": time.time() - start_time,
            }
            if valid_data is not None:
                entry["valid_loss"] = estimate_loss(model, valid_data, config)
            log.append(entry)
            print(entry)

    if config.checkpoint_path is not None:
        save_checkpoint(model, optimizer, config.max_iters, config.checkpoint_path)
    return log
