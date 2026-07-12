from __future__ import annotations

import argparse
import statistics
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_step(model, optimizer, inputs, targets, mode: str) -> None:
    if mode == "forward":
        with torch.no_grad():
            model(inputs)
        return

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    if mode == "full":
        optimizer.step()


def run_benchmark(args) -> list[float]:
    device = torch.device(args.device)
    config = MODEL_SIZES[args.model_size] | {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "rope_theta": args.rope_theta,
    }
    model = BasicsTransformerLM(**config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

    for _ in range(args.warmup_steps):
        timed_step(model, optimizer, inputs, targets, args.mode)
        synchronize(device)

    timings = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        timed_step(model, optimizer, inputs, targets, args.mode)
        synchronize(device)
        timings.append(timeit.default_timer() - start)
    return timings


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CS336 Assignment 2 Transformer steps.")
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), default="small")
    parser.add_argument("--mode", choices=["forward", "backward", "full"], default="forward")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timings = run_benchmark(args)
    print(f"mean_seconds={statistics.mean(timings):.6f}")
    print(f"std_seconds={statistics.stdev(timings) if len(timings) > 1 else 0.0:.6f}")
    print(f"raw_seconds={timings}")


if __name__ == "__main__":
    main()
