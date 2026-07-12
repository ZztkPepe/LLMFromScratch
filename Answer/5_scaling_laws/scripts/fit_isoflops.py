"""Fit simple IsoFLOPs scaling trends from the assignment's synthetic curves.

The handout's data gives validation loss for several model sizes at fixed
compute budgets. This script extracts the best point on each IsoFLOPs curve and
fits three small laws that are useful for planning experiments:

    N_opt(C) = a_N * C ** b_N
    D_opt(C) = a_D * C ** b_D
    L_opt(C) = E + A * C ** (-alpha)

where C is training FLOPs, N is non-embedding parameters, and D is training
tokens. The token estimate uses the dense-transformer approximation C = 6ND.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "isoflops_curves.json"
)


@dataclass(frozen=True)
class IsoFLOPsPoint:
    parameters: float
    compute_budget: float
    final_loss: float

    @property
    def train_tokens(self) -> float:
        return self.compute_budget / (6.0 * self.parameters)


@dataclass(frozen=True)
class PowerLaw:
    coefficient: float
    exponent: float

    def predict(self, x: float) -> float:
        return self.coefficient * x**self.exponent


@dataclass(frozen=True)
class LossLaw:
    floor: float
    coefficient: float
    exponent: float

    def predict(self, compute_budget: float) -> float:
        return self.floor + self.coefficient * compute_budget ** (-self.exponent)


@dataclass(frozen=True)
class ScalingFit:
    best_points: list[IsoFLOPsPoint]
    parameter_law: PowerLaw
    token_law: PowerLaw
    loss_law: LossLaw


def load_points(path: Path) -> list[IsoFLOPsPoint]:
    raw_points = json.loads(path.read_text())
    return [
        IsoFLOPsPoint(
            parameters=float(raw_point["parameters"]),
            compute_budget=float(raw_point["compute_budget"]),
            final_loss=float(raw_point["final_loss"]),
        )
        for raw_point in raw_points
    ]


def best_points_by_compute(points: list[IsoFLOPsPoint]) -> list[IsoFLOPsPoint]:
    grouped_points: dict[float, list[IsoFLOPsPoint]] = defaultdict(list)
    for point in points:
        grouped_points[point.compute_budget].append(point)

    return [
        min(grouped_points[compute_budget], key=lambda point: point.final_loss)
        for compute_budget in sorted(grouped_points)
    ]


def fit_log_log_power_law(xs: list[float], ys: list[float]) -> PowerLaw:
    log_xs = [math.log(x) for x in xs]
    log_ys = [math.log(y) for y in ys]
    mean_x = sum(log_xs) / len(log_xs)
    mean_y = sum(log_ys) / len(log_ys)
    denominator = sum((x - mean_x) ** 2 for x in log_xs)
    if denominator == 0:
        raise ValueError("need more than one distinct x value to fit a power law")

    exponent = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(log_xs, log_ys, strict=True))
        / denominator
    )
    intercept = mean_y - exponent * mean_x
    return PowerLaw(coefficient=math.exp(intercept), exponent=exponent)


def fit_offset_loss_law(
    compute_budgets: list[float],
    losses: list[float],
    *,
    floor_candidates: int = 5_000,
) -> LossLaw:
    """Fit L(C) = E + A*C^-alpha with a simple one-dimensional grid over E."""

    min_loss = min(losses)
    lower_floor = max(0.0, min_loss - 2.0)
    upper_floor = min_loss - 1e-6
    step = (upper_floor - lower_floor) / floor_candidates

    best: tuple[float, float, float, float] | None = None
    for candidate_idx in range(floor_candidates):
        floor = lower_floor + step * candidate_idx
        shifted_losses = [loss - floor for loss in losses]
        if any(shifted_loss <= 0 for shifted_loss in shifted_losses):
            continue

        shifted_law = fit_log_log_power_law(compute_budgets, shifted_losses)
        predicted_losses = [
            floor + shifted_law.predict(compute_budget)
            for compute_budget in compute_budgets
        ]
        squared_error = sum(
            (actual - predicted) ** 2
            for actual, predicted in zip(losses, predicted_losses, strict=True)
        )
        if best is None or squared_error < best[0]:
            best = (
                squared_error,
                floor,
                shifted_law.coefficient,
                -shifted_law.exponent,
            )

    if best is None:
        raise ValueError("could not fit loss law")

    _, floor, coefficient, exponent = best
    return LossLaw(floor=floor, coefficient=coefficient, exponent=exponent)


def fit_scaling_laws(points: list[IsoFLOPsPoint]) -> ScalingFit:
    best_points = best_points_by_compute(points)
    compute_budgets = [point.compute_budget for point in best_points]
    parameters = [point.parameters for point in best_points]
    tokens = [point.train_tokens for point in best_points]
    losses = [point.final_loss for point in best_points]

    return ScalingFit(
        best_points=best_points,
        parameter_law=fit_log_log_power_law(compute_budgets, parameters),
        token_law=fit_log_log_power_law(compute_budgets, tokens),
        loss_law=fit_offset_loss_law(compute_budgets, losses),
    )


def print_fit(fit: ScalingFit, *, target_compute_budget: float | None) -> None:
    print("Best point on each IsoFLOPs curve:")
    print("compute_budget,parameters,train_tokens,final_loss")
    for point in fit.best_points:
        print(
            f"{point.compute_budget:.6e},"
            f"{point.parameters:.6e},"
            f"{point.train_tokens:.6e},"
            f"{point.final_loss:.6f}"
        )

    print()
    print("Fitted scaling laws:")
    print(
        "N_opt(C) = "
        f"{fit.parameter_law.coefficient:.6e} * C^{fit.parameter_law.exponent:.6f}"
    )
    print(
        f"D_opt(C) = {fit.token_law.coefficient:.6e} * C^{fit.token_law.exponent:.6f}"
    )
    print(
        "L_opt(C) = "
        f"{fit.loss_law.floor:.6f} + "
        f"{fit.loss_law.coefficient:.6e} * C^-{fit.loss_law.exponent:.6f}"
    )

    if target_compute_budget is None:
        return

    print()
    print(f"Prediction for C={target_compute_budget:.6e}:")
    print(f"parameters={fit.parameter_law.predict(target_compute_budget):.6e}")
    print(f"train_tokens={fit.token_law.predict(target_compute_budget):.6e}")
    print(f"final_loss={fit.loss_law.predict(target_compute_budget):.6f}")


def fit_to_json(
    fit: ScalingFit, *, target_compute_budget: float | None
) -> dict[str, object]:
    result: dict[str, object] = {
        "best_points": [
            {
                "compute_budget": point.compute_budget,
                "parameters": point.parameters,
                "train_tokens": point.train_tokens,
                "final_loss": point.final_loss,
            }
            for point in fit.best_points
        ],
        "parameter_law": {
            "coefficient": fit.parameter_law.coefficient,
            "exponent": fit.parameter_law.exponent,
        },
        "token_law": {
            "coefficient": fit.token_law.coefficient,
            "exponent": fit.token_law.exponent,
        },
        "loss_law": {
            "floor": fit.loss_law.floor,
            "coefficient": fit.loss_law.coefficient,
            "exponent": fit.loss_law.exponent,
        },
    }
    if target_compute_budget is not None:
        result["prediction"] = {
            "compute_budget": target_compute_budget,
            "parameters": fit.parameter_law.predict(target_compute_budget),
            "train_tokens": fit.token_law.predict(target_compute_budget),
            "final_loss": fit.loss_law.predict(target_compute_budget),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to isoflops_curves.json.",
    )
    parser.add_argument(
        "--target-compute-budget",
        type=float,
        default=None,
        help="Optional FLOPs budget to extrapolate from the fitted laws.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable fit results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit = fit_scaling_laws(load_points(args.data))
    print_fit(fit, target_compute_budget=args.target_compute_budget)
    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(
                fit_to_json(fit, target_compute_budget=args.target_compute_budget),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
