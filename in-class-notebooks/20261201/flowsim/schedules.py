"""Piecewise constant arrival rates and staffing levels."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

import numpy as np

from flowsim.validation import integer, number

Points = tuple[tuple[float, float], ...]


def validate_schedule(
    points: Sequence[tuple[float, float]],
    *,
    label: str,
    counts: bool = False,
    maximum: int | None = None,
) -> Points:
    if not points:
        raise ValueError(f"{label} must have at least one time and value pair.")
    result = []
    prior = -1.0
    shape = "(start time, level)" if counts else "(start time, rate)"
    example = "staffing=[(0, 2), (120, 3)]" if counts else "rates=[(0, 10), (120, 20)]"
    for index, pair in enumerate(points):
        is_pair = (
            isinstance(pair, (Sequence, np.ndarray))
            and not isinstance(pair, (str, bytes))
            and len(pair) == 2
        )
        if not is_pair:
            raise ValueError(
                f"{label} entry {index} must be a {shape} pair, e.g. {example}, "
                f"not {pair!r}."
            )
        time = number(pair[0], f"{label} change time")
        if time <= prior:
            raise ValueError(f"{label} change times must increase strictly.")
        if not result and time != 0:
            raise ValueError(f"{label} must begin at time zero.")
        value = (
            integer(
                value=pair[1],
                label=f"{label} level",
                minimum=0,
            )
            if counts
            else number(pair[1], f"{label} rate")
        )
        if maximum is not None and value > maximum:
            raise ValueError(f"{label} cannot exceed installed capacity {maximum}.")
        result.append((time, value))
        prior = time
    return tuple(result)


def next_arrival(
    points: Points,
    now: float,
    horizon: float,
    rng: np.random.Generator,
) -> float | None:
    """Invert integrated intensity for a piecewise constant Poisson process."""
    # The exponential draw is dimensionless cumulative hazard. Rate has units
    # customers/time, and rate * elapsed time is the hazard spent in a segment.
    hazard = float(rng.exponential(scale=1))
    index = bisect_right(points, now, key=lambda point: point[0]) - 1
    cursor = now
    while cursor < horizon:
        rate = points[index][1]
        end = min(
            horizon,
            points[index + 1][0] if index + 1 < len(points) else horizon,
        )
        spent = rate * (end - cursor)
        if rate > 0 and hazard < spent:
            arrival = cursor + hazard / rate
            if arrival <= now:
                raise ValueError("Arrival rate is too large for time to advance.")
            return arrival
        hazard -= spent
        cursor = end
        index += 1
    return None
