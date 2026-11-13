"""Forecasts and ordering policies for stock points, independent of the engine.

A stock point reviews once per period, after that period's demand. Its target
is cover times the forecast plus safety stock, and it orders the shortfall of
its inventory position below the target, never a negative quantity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from flowsim.validation import integer, number


@dataclass(frozen=True)
class MovingAverage:
    """Forecast one period's demand as the mean of the last window observations."""

    window: int

    def forecast(self, history: Sequence[float]) -> float:
        # Before window periods have passed, the mean uses every observation so
        # far. A review always follows at least one observation.
        recent = history[-self.window :]
        return sum(recent) / len(recent)


@dataclass(frozen=True)
class OrderUpTo:
    """Raise the inventory position to cover * forecast + safety stock."""

    forecast: MovingAverage
    safety_factor: float
    cover: float | None = None

    def target(self, history: Sequence[float], cover: float) -> tuple[float, float]:
        """Return the forecast and the target for a known cover in periods.

        Safety stock is safety_factor * sqrt(cover) times the sample standard
        deviation (n - 1 divisor) of the demands in the forecast window. It is
        zero until the window is full, since a deviation from two or three early
        observations can set a target far above steady demand.
        """
        forecast = self.forecast.forecast(history)
        safety = 0.0
        if self.safety_factor > 0 and len(history) >= self.forecast.window:
            recent = history[-self.forecast.window :]
            mean = sum(recent) / len(recent)
            variance = sum((value - mean) ** 2 for value in recent) / (len(recent) - 1)
            safety = self.safety_factor * math.sqrt(cover) * math.sqrt(variance)
        return forecast, cover * forecast + safety


def moving_average(*, window: int) -> MovingAverage:
    """Forecast one period's demand as the mean of the last window observations.

    Before window periods have been observed, the mean uses every observation
    so far. Pass the result to order_up_to as forecast.

    Parameters
    ----------
    window : int
        Number of most recent review periods averaged, at least 1.

    Returns
    -------
    MovingAverage
        An immutable forecast definition.

    Example
    -------
    >>> forecast = flowsim.moving_average(window=4)
    """
    return MovingAverage(
        integer(
            value=window,
            label="Moving-average window",
            minimum=1,
        )
    )


def order_up_to(
    *,
    forecast: MovingAverage,
    safety_factor: float,
    cover: float | None = None,
) -> OrderUpTo:
    """Define an order-up-to policy.

    At each review, a stock point orders the shortfall of its inventory
    position (on hand plus on order minus backorders) below the target, never a
    negative quantity. The target is cover times the forecast plus safety
    stock. Safety stock is safety_factor * sqrt(cover) times the sample
    standard deviation of the demands in the forecast window, and it is zero
    until that window is full.

    Parameters
    ----------
    forecast : MovingAverage
        The demand forecast, from flowsim.moving_average.
    safety_factor : float
        Number of standard deviations of safety stock, at least 0. A positive
        value requires a moving-average window of at least 2.
    cover : float, optional
        Periods of forecast demand the target covers, greater than 0. Leave it
        out for a constant lead time, where cover is lead_time / period. Give
        it for a sampled lead time (whose mean FlowSim does not estimate) or
        for a stock point that produces at stations (lead_time=None).

    Returns
    -------
    OrderUpTo
        An immutable policy to pass to Model.stock_point as policy.

    Example
    -------
    >>> policy = flowsim.order_up_to(
    ...     forecast=flowsim.moving_average(window=4),
    ...     safety_factor=1.0,
    ... )
    """
    if not isinstance(forecast, MovingAverage):
        raise ValueError("forecast must come from flowsim.moving_average.")
    factor = number(safety_factor, "Safety factor")
    if factor > 0 and forecast.window < 2:
        raise ValueError(
            "Safety stock uses the spread of the forecast window, so a positive "
            "safety factor needs a moving-average window of at least 2."
        )
    periods = (
        None
        if cover is None
        else number(
            value=cover,
            label="Cover",
            positive=True,
        )
    )
    return OrderUpTo(
        forecast=forecast,
        safety_factor=factor,
        cover=periods,
    )
