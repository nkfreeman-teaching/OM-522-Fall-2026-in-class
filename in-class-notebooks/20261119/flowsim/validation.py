"""Checks for student-supplied numbers and sampled durations.

This module imports no other FlowSim module, so every module can use it.
"""

from __future__ import annotations

import inspect
import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from numbers import Integral, Real

import numpy as np

Sampler = Callable[[np.random.Generator], float]
Duration = float | Sampler


def number(value: object, label: str, *, positive: bool = False) -> float:
    """Require a finite nonnegative scalar, or a positive one if positive is set."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number, not {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        comparison = "greater than zero" if positive else "at least zero"
        raise ValueError(f"{label} must be finite and {comparison}, not {value!r}.")
    return result


def integer(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer of at least {minimum}.")
    if value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}.")
    return int(value)


def duration(value: object, label: str, *, positive: bool = False) -> None:
    """Check a duration at definition time: a number or a one-argument sampler."""
    if not callable(value):
        number(
            value=value,
            label=label,
            positive=positive,
        )
        return
    try:
        signature = inspect.signature(value)
    except ValueError:
        return
    try:
        signature.bind(object())
    except TypeError:
        raise ValueError(
            f"{label} must be a number or a function of one argument, the "
            "random generator, e.g. lambda rng: rng.exponential(scale=4) "
            "or flowsim.exponential(mean=4)."
        ) from None


def sample(
    distribution: Duration,
    rng: np.random.Generator,
    label: str,
    *,
    positive: bool = False,
) -> float:
    value = distribution(rng) if callable(distribution) else distribution
    # Fast path for the common case, a finite float in range; anything else
    # takes the full checks below, which raise the same errors as before.
    if type(value) is float and 0 <= value < math.inf and (value or not positive):
        return value
    if isinstance(value, np.ndarray):
        raise ValueError(
            f"{label} came back as an array {value!r}. A sampler returns one "
            "number, so drop size= from the NumPy call, e.g. "
            "rng.exponential(scale=4)."
        )
    return number(
        value=value,
        label=label,
        positive=positive,
    )


def cumulative_shares(probabilities: Sequence[float] | np.ndarray) -> list[float]:
    """Return the running totals of the probabilities, ending at one.

    These are the totals that NumPy's ``Generator.choice(n, p=...)`` builds on
    every call. ``draw_index`` searches them with one uniform draw, which gives
    the index ``choice`` gives at about a twentieth of the cost per draw.
    tests/test_distributions.py compares the two, so a NumPy release that
    changes ``choice`` fails that test before it changes a seeded run.
    """
    totals = np.asarray(probabilities, dtype=float).cumsum()
    totals /= totals[-1]
    return totals.tolist()


def draw_index(shares: list[float], rng: np.random.Generator) -> int:
    """Draw an index with the probabilities behind ``cumulative_shares``."""
    return bisect_right(shares, rng.random())
