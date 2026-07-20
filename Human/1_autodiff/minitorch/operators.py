"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, TypeVar, List

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


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y:float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return float(x < y)


def eq(x: float, y: float) -> bool:
    return float(x == y)

def max(x: float, y: float) -> float:
    return y if lt(x, y) else x


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return max(0.0, x)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    return 1.0 / x * d


def inv_back(x: float, d: float) -> float:
    return -1.0 * (1.0 / x**2)  * d


def relu_back(x: float, d: float) -> float:
    return 0.0 if x <= 0 else 1.0 * d


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


# TODO: Implement for Task 0.3.
def map(fn: Callable[[A], B]) -> Callable[[Iterable[A]], List[B]]:

    def apply(ls: Iterable[A]) -> List[B]:
        return [fn(ele) for ele in ls]

    return apply


def zipWith(fn: Callable[[A, B], C]) -> Callable[[Iterable[A], Iterable[B]], List[C]]:
    """Combine two iterables elementwise with a two-argument function."""

    def apply(ls1: Iterable[A], ls2: Iterable[B]) -> List[B]:
        return [fn(ele1, ele2) for ele1, ele2 in zip(ls1, ls2)]

    return apply


def reduce(fn: Callable[[B, A], B], start: B) -> Callable[[Iterable[A]], B]:

    def apply(ls: Iterable[A]) -> B:
        result = start
        for ele in ls:
            result = fn(result, ele)
        return result

    return apply

negList: Callable[[Iterable[float]], List[float]] = map(neg)
addLists: Callable[[Iterable[float], Iterable[float]], List[float]] = zipWith(add)
sum: Callable[[Iterable[float], float], float] = reduce(add, 0.0)
prod: Callable[[Iterable[float], float], float] = reduce(mul, 1.0)
