"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, TypeVar

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 when x is less than y, otherwise 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 when x equals y, otherwise 0.0."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check whether two numbers are within 1e-2 of each other."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    """Return x when positive, otherwise 0.0."""
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """Compute the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute e raised to x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal of x."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log(x) multiplied by d."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of 1/x multiplied by d."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of ReLU(x) multiplied by d."""
    return d if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[A], B]) -> Callable[[Iterable[A]], List[B]]:
    """Apply a one-argument function to every item in an iterable."""

    def apply(ls: Iterable[A]) -> List[B]:
        return [fn(x) for x in ls]

    return apply


def zipWith(fn: Callable[[A, B], C]) -> Callable[[Iterable[A], Iterable[B]], List[C]]:
    """Combine two iterables elementwise with a two-argument function."""

    def apply(ls1: Iterable[A], ls2: Iterable[B]) -> List[C]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply


def reduce(fn: Callable[[B, A], B], start: B) -> Callable[[Iterable[A]], B]:
    """Reduce an iterable to one value, starting from an initial value."""

    def apply(ls: Iterable[A]) -> B:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return apply


negList: Callable[[Iterable[float]], List[float]] = map(neg)  # noqa: N816
addLists: Callable[[Iterable[float], Iterable[float]], List[float]] = zipWith(add)  # noqa: N816
sum: Callable[[Iterable[float]], float] = reduce(add, 0.0)
prod: Callable[[Iterable[float]], float] = reduce(mul, 1.0)
