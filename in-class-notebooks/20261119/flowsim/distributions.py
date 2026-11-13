"""Named duration and demand distributions.

Each constructor (``exponential``, ``gamma``, ...) validates its arguments and
returns a frozen, stateless ``Distribution``. A distribution is a function of
the random generator, ``f(rng) -> float``, so it can be passed anywhere FlowSim
accepts a sampler. The engine supplies its own generator, so a distribution
never holds random state.
"""

# Each Distribution subclass supplies mean and sd as a dataclass field or a
# read-only property. Pyright rejects both against the base's declaration, and
# a base property would block the frozen dataclass from setting a field.
# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

import functools
import html
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from numbers import Real

import numpy as np
from scipy.special import zeta

from flowsim.validation import cumulative_shares, draw_index


def _bad(fn: str, rule: str, example: str, got: dict[str, object]) -> ValueError:
    shown = ", ".join(f"{name}={value!r}" for name, value in got.items())
    return ValueError(f"{fn} needs {rule}, e.g. {example}, not {shown}.")


def _real(fn: str, example: str, **arguments: object) -> list[float]:
    """Require finite real numbers and return them as floats, in order."""
    result = []
    for name, value in arguments.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise _bad(fn, f"{name} to be a finite number", example, {name: value})
        number = float(value)
        if not math.isfinite(number):
            raise _bad(fn, f"{name} to be a finite number", example, {name: value})
        result.append(number)
    return result


def _has_bool(values: object) -> bool:
    """Whether a sequence holds booleans, which float conversion would hide."""
    try:
        items = np.asarray(values, dtype=object).ravel()
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in items)


def _fmt(value: object) -> str:
    return f"{value:g}"


class Distribution(ABC):
    """A named duration or demand distribution, callable as ``f(rng) -> float``.

    Subclasses are frozen dataclasses built by the lowercase constructors in
    this module. Every instance has ``mean`` and ``sd`` (floats, read-only),
    a one-sentence ``describe()``, and a constructor-like ``repr``.

    Example
    -------
    >>> exponential(mean=4)(np.random.default_rng(1))  # one draw, a float
    """

    # Declared only: each subclass supplies these as a field or a property.
    mean: float
    sd: float

    @abstractmethod
    def __call__(self, rng: np.random.Generator) -> float:
        """Draw one value using the given generator."""

    @abstractmethod
    def describe(self) -> str:
        """Return one plain sentence describing the distribution.

        Returns
        -------
        str
            The sentence, with the parameters in the model's units.

        Example
        -------
        >>> flowsim.exponential(mean=4).describe()
        'Exponential with mean 4: memoryless gaps, as for random arrivals.'
        """

    def __repr__(self) -> str:
        # Every subclass that keeps this repr is a dataclass.
        shown = ", ".join(
            f"{item.name}={_fmt(getattr(self, item.name))}"
            for item in fields(self)  # pyright: ignore[reportArgumentType]
        )
        return f"{type(self).__name__}({shown})"

    def _repr_html_(self) -> str:
        return (
            f"<code>{html.escape(repr(self))}</code><br>{html.escape(self.describe())}"
        )

    def spec(self) -> dict:
        """Return the constructor name and its arguments at full precision.

        ``getattr(flowsim, spec["name"])(**arguments)`` rebuilds the same
        distribution, where arguments are the other keys, so export metadata
        can record it without rounding.

        Returns
        -------
        dict
            The key "name" (the lowercase constructor name) and one key per
            constructor argument.

        Example
        -------
        >>> flowsim.exponential(mean=4).spec()
        {'name': 'exponential', 'mean': 4.0}
        """
        # Every concrete distribution is a frozen dataclass.
        assert is_dataclass(self)
        arguments = {}
        for field in fields(self):
            value = getattr(self, field.name)
            arguments[field.name] = list(value) if isinstance(value, tuple) else value
        return {"name": type(self).__name__.lower(), **arguments}


@dataclass(frozen=True, repr=False)
class Exponential(Distribution):
    mean: float

    @property
    def sd(self) -> float:
        return self.mean

    def __call__(self, rng):
        return float(rng.exponential(scale=self.mean))

    def describe(self) -> str:
        return (
            f"Exponential with mean {_fmt(self.mean)}: memoryless gaps, "
            "as for random arrivals."
        )


def exponential(*, mean: float) -> Exponential:
    """Exponential duration, the memoryless gap between random events.

    Parameters
    ----------
    mean : float
        Mean duration, greater than zero.

    Returns
    -------
    Exponential
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> exponential(mean=4)
    Exponential(mean=4)
    """
    example = "exponential(mean=4)"
    (m,) = _real("exponential", example, mean=mean)
    if m <= 0:
        raise _bad("exponential", "mean > 0", example, {"mean": mean})
    return Exponential(mean=m)


@dataclass(frozen=True, repr=False)
class Uniform(Distribution):
    low: float
    high: float

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2

    @property
    def sd(self) -> float:
        return (self.high - self.low) / math.sqrt(12)

    def __call__(self, rng):
        return float(rng.uniform(low=self.low, high=self.high))

    def describe(self) -> str:
        return (
            f"Uniform between {_fmt(self.low)} and {_fmt(self.high)}: "
            "every value in the range is equally likely."
        )


def uniform(*, low: float, high: float) -> Uniform:
    """Uniform duration, equally likely anywhere between low and high.

    Parameters
    ----------
    low : float
        Smallest value, at least zero.
    high : float
        Largest value, greater than low.

    Returns
    -------
    Uniform
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> uniform(low=1, high=2)
    Uniform(low=1, high=2)
    """
    example = "uniform(low=1, high=2)"
    lo, hi = _real("uniform", example, low=low, high=high)
    if not 0 <= lo < hi:
        raise _bad("uniform", "0 <= low < high", example, {"low": low, "high": high})
    return Uniform(low=lo, high=hi)


def _three_points(fn: str, low: float, mode: float, high: float) -> list[float]:
    example = f"{fn}(low=1, mode=2, high=6)"
    lo, mo, hi = _real(fn, example, low=low, mode=mode, high=high)
    if not (0 <= lo <= mo <= hi and lo < hi):
        raise _bad(
            fn,
            "0 <= low <= mode <= high with low < high",
            example,
            {"low": low, "mode": mode, "high": high},
        )
    return [lo, mo, hi]


@dataclass(frozen=True, repr=False)
class Triangular(Distribution):
    low: float
    mode: float
    high: float

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    @property
    def sd(self) -> float:
        # (a^2 + b^2 + c^2 - ab - ac - bc) / 18 written with differences, which
        # keeps its precision far from zero.
        a, b, c = self.low, self.mode, self.high
        return math.hypot(a - b, a - c, b - c) / 6

    def __call__(self, rng):
        return float(rng.triangular(left=self.low, mode=self.mode, right=self.high))

    def describe(self) -> str:
        return (
            f"Triangular from {_fmt(self.low)} to {_fmt(self.high)}, most likely "
            f"near {_fmt(self.mode)}: a rough low/likely/high estimate."
        )


def triangular(*, low: float, mode: float, high: float) -> Triangular:
    """Triangular duration from a low, most likely, and high estimate.

    Parameters
    ----------
    low : float
        Smallest value, at least zero.
    mode : float
        Most likely value, between low and high.
    high : float
        Largest value, greater than low.

    Returns
    -------
    Triangular
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> triangular(low=1, mode=2, high=6)
    Triangular(low=1, mode=2, high=6)
    """
    lo, mo, hi = _three_points("triangular", low, mode, high)
    return Triangular(low=lo, mode=mo, high=hi)


@dataclass(frozen=True, repr=False)
class Pert(Distribution):
    low: float
    mode: float
    high: float

    @functools.cached_property
    def _alpha(self) -> float:
        return 1 + 4 * (self.mode - self.low) / (self.high - self.low)

    @functools.cached_property
    def _beta(self) -> float:
        return 1 + 4 * (self.high - self.mode) / (self.high - self.low)

    @property
    def mean(self) -> float:
        return (self.low + 4 * self.mode + self.high) / 6

    @property
    def sd(self) -> float:
        a, b = self._alpha, self._beta
        spread = self.high - self.low
        return spread * math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def __call__(self, rng):
        spread = self.high - self.low
        return float(self.low + spread * rng.beta(a=self._alpha, b=self._beta))

    def describe(self) -> str:
        return (
            f"PERT from {_fmt(self.low)} to {_fmt(self.high)}, most likely "
            f"near {_fmt(self.mode)}: a smooth low/likely/high estimate."
        )


def pert(*, low: float, mode: float, high: float) -> Pert:
    """PERT duration, a smooth low, most likely, and high estimate.

    Parameters
    ----------
    low : float
        Smallest value, at least zero.
    mode : float
        Most likely value, between low and high.
    high : float
        Largest value, greater than low.

    Returns
    -------
    Pert
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> pert(low=1, mode=2, high=6)
    Pert(low=1, mode=2, high=6)
    """
    lo, mo, hi = _three_points("pert", low, mode, high)
    return Pert(low=lo, mode=mo, high=hi)


@dataclass(frozen=True, repr=False)
class Normal(Distribution):
    mean: float
    sd: float

    def __call__(self, rng):
        return float(rng.normal(loc=self.mean, scale=self.sd))

    def describe(self) -> str:
        text = (
            f"Normal with mean {_fmt(self.mean)} and sd {_fmt(self.sd)}: "
            "symmetric around the mean."
        )
        # Phi(-mean / sd), the chance of a draw below zero.
        negative = 0.5 * (1 + math.erf(-self.mean / (self.sd * math.sqrt(2))))
        if negative > 1e-6:
            text += (
                f" A draw is negative with probability {negative:.3g}, "
                "and a negative draw stops the run."
            )
        return text


def normal(*, mean: float, sd: float) -> Normal:
    """Normal duration. A negative draw stops the run, so keep mean well above sd.

    Parameters
    ----------
    mean : float
        Mean of the distribution.
    sd : float
        Standard deviation, greater than zero.

    Returns
    -------
    Normal
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> normal(mean=10, sd=2)
    Normal(mean=10, sd=2)
    """
    example = "normal(mean=10, sd=2)"
    m, s = _real("normal", example, mean=mean, sd=sd)
    if s <= 0:
        raise _bad("normal", "sd > 0", example, {"sd": sd})
    return Normal(mean=m, sd=s)


@dataclass(frozen=True, repr=False)
class Lognormal(Distribution):
    mean: float
    sd: float

    @functools.cached_property
    def _sigma(self) -> float:
        # Divide before squaring and use log1p, so a tiny or huge scale and a
        # small coefficient of variation keep their precision.
        return math.sqrt(math.log1p((self.sd / self.mean) ** 2))

    @functools.cached_property
    def _mu(self) -> float:
        return math.log(self.mean) - self._sigma**2 / 2

    def __call__(self, rng):
        return float(rng.lognormal(mean=self._mu, sigma=self._sigma))

    def describe(self) -> str:
        return (
            f"Lognormal with mean {_fmt(self.mean)} and sd {_fmt(self.sd)}: "
            "right-skewed, never negative."
        )


def lognormal(*, mean: float, sd: float) -> Lognormal:
    """Lognormal duration, right-skewed and never negative.

    The mean and sd are those of the duration itself, not of its logarithm.

    Parameters
    ----------
    mean : float
        Mean of the duration, greater than zero.
    sd : float
        Standard deviation of the duration, greater than zero.

    Returns
    -------
    Lognormal
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> lognormal(mean=4, sd=2)
    Lognormal(mean=4, sd=2)
    """
    example = "lognormal(mean=4, sd=2)"
    m, s = _real("lognormal", example, mean=mean, sd=sd)
    if m <= 0 or s <= 0:
        raise _bad(
            "lognormal",
            "mean > 0 and sd > 0",
            example,
            {"mean": mean, "sd": sd},
        )
    return Lognormal(mean=m, sd=s)


@dataclass(frozen=True, repr=False)
class Gamma(Distribution):
    shape: float
    scale: float

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    @property
    def sd(self) -> float:
        return math.sqrt(self.shape) * self.scale

    def __call__(self, rng):
        return float(rng.gamma(shape=self.shape, scale=self.scale))

    def describe(self) -> str:
        return (
            f"Gamma with shape {_fmt(self.shape)} and scale {_fmt(self.scale)}: "
            "right-skewed, flexible for service times."
        )


def gamma(*, shape: float, scale: float) -> Gamma:
    """Gamma duration, right-skewed with mean shape * scale.

    Parameters
    ----------
    shape : float
        Shape, greater than zero.
    scale : float
        Scale, greater than zero.

    Returns
    -------
    Gamma
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> gamma(shape=2, scale=3)
    Gamma(shape=2, scale=3)
    """
    example = "gamma(shape=2, scale=3)"
    sh, sc = _real("gamma", example, shape=shape, scale=scale)
    if sh <= 0 or sc <= 0:
        raise _bad(
            "gamma",
            "shape > 0 and scale > 0",
            example,
            {"shape": shape, "scale": scale},
        )
    return Gamma(shape=sh, scale=sc)


@dataclass(frozen=True, repr=False)
class Weibull(Distribution):
    shape: float
    scale: float

    @property
    def mean(self) -> float:
        return self.scale * math.gamma(1 + 1 / self.shape)

    @property
    def sd(self) -> float:
        # The variance is scale^2 * (Gamma(1 + 2x) - Gamma(1 + x)^2) for
        # x = 1 / shape, i.e., scale^2 * Gamma(1 + x)^2 * expm1(L) with
        # L = lgamma(1 + 2x) - 2 lgamma(1 + x). Subtracting the gamma values
        # cancels for a large shape (shape 1e8 gave a negative variance), so
        # a small x takes L from the series of lgamma(1 + z), whose terms are
        # (-1)^n zeta(n) z^n / n for n >= 2; they sum to 1e-15 accuracy here.
        x = 1 / self.shape
        if x >= 0.01:
            log_ratio = math.lgamma(1 + 2 * x) - 2 * math.lgamma(1 + x)
            return self.scale * math.gamma(1 + x) * math.sqrt(math.expm1(log_ratio))
        # L = x^2 * series, and sqrt(expm1(L)) = x * sqrt(series * expm1(L) / L).
        # x^2 enters only the expm1(L) / L factor, which tends to 1, so a huge
        # shape (1e170, where x^2 underflows) still gives a positive sd.
        series = sum(
            (-1) ** n * float(zeta(n)) * (2**n - 2) * x ** (n - 2) / n
            for n in range(2, 12)
        )
        log_ratio = x * x * series
        growth = math.expm1(log_ratio) / log_ratio if log_ratio > 0 else 1.0
        return self.scale * math.gamma(1 + x) * x * math.sqrt(series * growth)

    def __call__(self, rng):
        return float(self.scale * rng.weibull(a=self.shape))

    def describe(self) -> str:
        return (
            f"Weibull with shape {_fmt(self.shape)} and scale {_fmt(self.scale)}: "
            "a lifetime model, wearing out when shape exceeds 1."
        )


def weibull(*, shape: float, scale: float) -> Weibull:
    """Weibull duration, a common model for time to failure.

    Parameters
    ----------
    shape : float
        Shape, greater than zero.
    scale : float
        Scale, greater than zero.

    Returns
    -------
    Weibull
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> weibull(shape=1.5, scale=10)
    Weibull(shape=1.5, scale=10)
    """
    example = "weibull(shape=1.5, scale=10)"
    sh, sc = _real("weibull", example, shape=shape, scale=scale)
    if sh <= 0 or sc <= 0:
        raise _bad(
            "weibull",
            "shape > 0 and scale > 0",
            example,
            {"shape": shape, "scale": scale},
        )
    return Weibull(shape=sh, scale=sc)


@dataclass(frozen=True, repr=False)
class Erlang(Distribution):
    k: int
    mean: float

    @property
    def sd(self) -> float:
        return self.mean / math.sqrt(self.k)

    def __call__(self, rng):
        return float(rng.gamma(shape=self.k, scale=self.mean / self.k))

    def describe(self) -> str:
        return (
            f"Erlang with {self.k} stages and mean {_fmt(self.mean)}: "
            "the sum of exponential stages, less variable than exponential."
        )


def erlang(*, k: int, mean: float) -> Erlang:
    """Erlang duration, the sum of k exponential stages.

    Parameters
    ----------
    k : int
        Number of stages, an integer of at least 1.
    mean : float
        Mean of the total duration, greater than zero.

    Returns
    -------
    Erlang
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> erlang(k=3, mean=6)
    Erlang(k=3, mean=6)
    """
    example = "erlang(k=3, mean=6)"
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise _bad("erlang", "k to be an integer >= 1", example, {"k": k})
    (m,) = _real("erlang", example, mean=mean)
    if m <= 0:
        raise _bad("erlang", "mean > 0", example, {"mean": mean})
    return Erlang(k=int(k), mean=m)


@dataclass(frozen=True, repr=False)
class Beta(Distribution):
    a: float
    b: float
    low: float
    high: float

    @property
    def mean(self) -> float:
        return self.low + (self.high - self.low) * self.a / (self.a + self.b)

    @property
    def sd(self) -> float:
        total = self.a + self.b
        spread = self.high - self.low
        return spread * math.sqrt(self.a * self.b / (total**2 * (total + 1)))

    def __call__(self, rng):
        spread = self.high - self.low
        return float(self.low + spread * rng.beta(a=self.a, b=self.b))

    def describe(self) -> str:
        return (
            f"Beta({_fmt(self.a)}, {_fmt(self.b)}) stretched to run from "
            f"{_fmt(self.low)} to {_fmt(self.high)}: a bounded, flexible shape."
        )


def beta(*, a: float, b: float, low: float = 0, high: float = 1) -> Beta:
    """Beta duration scaled to run from low to high.

    Parameters
    ----------
    a : float
        First shape, greater than zero.
    b : float
        Second shape, greater than zero.
    low : float
        Smallest value, at least zero (default 0).
    high : float
        Largest value, greater than low (default 1).

    Returns
    -------
    Beta
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> beta(a=2, b=5, low=1, high=4)
    Beta(a=2, b=5, low=1, high=4)
    """
    example = "beta(a=2, b=5, low=1, high=4)"
    a_, b_, lo, hi = _real("beta", example, a=a, b=b, low=low, high=high)
    if a_ <= 0 or b_ <= 0 or not 0 <= lo < hi:
        raise _bad(
            "beta",
            "a > 0, b > 0, and 0 <= low < high",
            example,
            {"a": a, "b": b, "low": low, "high": high},
        )
    return Beta(a=a_, b=b_, low=lo, high=hi)


@dataclass(frozen=True, repr=False)
class Empirical(Distribution):
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def sd(self) -> float:
        return float(np.std(self.values, ddof=0))

    def __call__(self, rng):
        return self.values[int(rng.integers(len(self.values)))]

    def describe(self) -> str:
        return (
            f"Empirical: resamples one of {len(self.values)} observed values "
            "with equal chance."
        )

    def spec(self) -> dict:
        return {"name": "empirical", "data": list(self.values)}

    def __repr__(self) -> str:
        return (
            f"Empirical(n={len(self.values)}, mean={_fmt(self.mean)}, "
            f"min={_fmt(min(self.values))}, max={_fmt(max(self.values))})"
        )


def empirical(data: Sequence[float]) -> Empirical:
    """Resample observed values, each with equal chance.

    Parameters
    ----------
    data : sequence of float
        Observed durations: at least one value, each finite and at least zero.

    Returns
    -------
    Empirical
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> empirical([2.0, 3.5, 4.1]).mean
    3.2
    """
    example = "empirical([2.0, 3.5, 4.1])"
    if _has_bool(data):
        raise _bad("empirical", "numbers, not True or False", example, {"data": data})
    try:
        array = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        raise _bad("empirical", "a list of numbers", example, {"data": data}) from None
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array < 0)
    ):
        raise _bad(
            "empirical",
            "a nonempty 1-D list of finite values >= 0",
            example,
            {"data": data},
        )
    return Empirical(values=tuple(float(x) for x in array))


@dataclass(frozen=True, repr=False)
class Poisson(Distribution):
    mean: float

    @property
    def sd(self) -> float:
        return math.sqrt(self.mean)

    def __call__(self, rng):
        return float(rng.poisson(lam=self.mean))

    def describe(self) -> str:
        return (
            f"Poisson with mean {_fmt(self.mean)}: whole-number counts, "
            "as for random demand."
        )


def poisson(*, mean: float) -> Poisson:
    """Poisson demand, a random whole-number count.

    Parameters
    ----------
    mean : float
        Mean count, greater than zero.

    Returns
    -------
    Poisson
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> poisson(mean=20)
    Poisson(mean=20)
    """
    example = "poisson(mean=20)"
    (m,) = _real("poisson", example, mean=mean)
    if m <= 0:
        raise _bad("poisson", "mean > 0", example, {"mean": mean})
    return Poisson(mean=m)


@dataclass(frozen=True, repr=False)
class Discrete(Distribution):
    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(p * v for p, v in zip(self.probabilities, self.values))

    @property
    def sd(self) -> float:
        m = self.mean
        return math.sqrt(
            sum(p * (v - m) ** 2 for p, v in zip(self.probabilities, self.values))
        )

    @functools.cached_property
    def _shares(self) -> list[float]:
        return cumulative_shares(self.probabilities)

    def __call__(self, rng):
        return self.values[draw_index(self._shares, rng)]

    def describe(self) -> str:
        return (
            f"Discrete: one of {len(self.values)} listed values, "
            "each with its own probability."
        )

    def spec(self) -> dict:
        return {
            "name": "discrete",
            "values": list(self.values),
            "weights": list(self.probabilities),
        }

    def __repr__(self) -> str:
        values = ", ".join(_fmt(v) for v in self.values)
        probabilities = ", ".join(_fmt(p) for p in self.probabilities)
        return f"Discrete(values=({values}), probabilities=({probabilities}))"


def discrete(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
) -> Discrete:
    """Pick one of a few listed values, equally likely or by weight.

    Parameters
    ----------
    values : sequence of float
        The possible values, each finite and at least zero.
    weights : sequence of float, optional
        Relative chances, one per value, each at least zero with a positive
        sum. Equal chances if omitted.

    Returns
    -------
    Discrete
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> discrete([1, 2, 5], weights=[2, 2, 1]).mean
    2.2
    """
    example = "discrete([1, 2, 5], weights=[2, 2, 1])"
    if _has_bool(values) or (weights is not None and _has_bool(weights)):
        raise _bad(
            "discrete",
            "numbers, not True or False",
            example,
            {"values": values, "weights": weights},
        )
    try:
        v = np.asarray(values, dtype=float)
        w = np.ones(v.size) if weights is None else np.asarray(weights, dtype=float)
    except (TypeError, ValueError):
        raise _bad(
            "discrete",
            "numeric values and weights",
            example,
            {"values": values},
        ) from None
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) or np.any(v < 0):
        raise _bad(
            "discrete",
            "a nonempty list of finite values >= 0",
            example,
            {"values": values},
        )
    if (
        w.shape != v.shape
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
        or w.max(initial=0) <= 0
    ):
        raise _bad(
            "discrete",
            "one weight per value, each >= 0, with a positive sum",
            example,
            {"weights": weights},
        )
    # Scale by the largest weight first, so huge finite weights cannot
    # overflow the sum.
    w = w / w.max()
    return Discrete(
        values=tuple(float(x) for x in v),
        probabilities=tuple(float(x) for x in w / w.sum()),
    )


@dataclass(frozen=True, repr=False)
class NegativeBinomial(Distribution):
    mean: float
    sd: float

    @functools.cached_property
    def _n(self) -> float:
        # Variance = mean + mean^2 / n, so n = mean^2 / (variance - mean).
        return self.mean**2 / (self.sd**2 - self.mean)

    @functools.cached_property
    def _p(self) -> float:
        # NumPy counts failures before the n-th success with success chance p,
        # whose mean n (1 - p) / p equals mean when p = mean / variance.
        return self.mean / self.sd**2

    def __call__(self, rng):
        return float(rng.negative_binomial(n=self._n, p=self._p))

    def describe(self) -> str:
        return (
            f"Negative binomial with mean {_fmt(self.mean)} and sd {_fmt(self.sd)}: "
            "whole-number counts, more variable than Poisson."
        )

    def spec(self) -> dict:
        return {"name": "negative_binomial", "mean": self.mean, "sd": self.sd}


def negative_binomial(*, mean: float, sd: float) -> NegativeBinomial:
    """Negative binomial demand, a whole-number count more variable than Poisson.

    A Poisson count has variance equal to its mean. The negative binomial adds
    variability, so its sd must satisfy sd**2 > mean. Draws use NumPy's
    ``rng.negative_binomial(n, p)`` with n = mean**2 / (sd**2 - mean) and
    p = mean / sd**2, i.e., the number of failures before the n-th success.

    Parameters
    ----------
    mean : float
        Mean count, greater than zero.
    sd : float
        Standard deviation of the count, with sd**2 greater than mean.

    Returns
    -------
    NegativeBinomial
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> negative_binomial(mean=20, sd=8)
    NegativeBinomial(mean=20, sd=8)
    """
    example = "negative_binomial(mean=20, sd=8)"
    m, s = _real("negative_binomial", example, mean=mean, sd=sd)
    if m <= 0 or s <= 0 or s**2 <= m:
        raise _bad(
            "negative_binomial",
            "mean > 0 and sd**2 > mean (use poisson(mean=...) when sd**2 = mean)",
            example,
            {"mean": mean, "sd": sd},
        )
    return NegativeBinomial(mean=m, sd=s)


@dataclass(frozen=True, repr=False)
class Shifted(Distribution):
    distribution: Distribution
    by: float

    @property
    def mean(self) -> float:
        return self.distribution.mean + self.by

    @property
    def sd(self) -> float:
        return self.distribution.sd

    def __call__(self, rng):
        return float(self.by + self.distribution(rng))

    def describe(self) -> str:
        text = (
            f"{self.distribution.describe()} Every draw is then shifted by "
            f"{_fmt(self.by)}."
        )
        if self.by < 0:
            text += " A draw can be negative, and a negative draw stops the run."
        return text

    def spec(self) -> dict:
        return {
            "name": "shifted",
            "distribution": self.distribution.spec(),
            "by": self.by,
        }

    def __repr__(self) -> str:
        return f"Shifted({self.distribution!r}, by={_fmt(self.by)})"


def shifted(distribution: Distribution | dict, *, by: float) -> Shifted:
    """Add a fixed amount to every draw of another distribution.

    A shift models a minimum time, e.g., a 2-minute setup followed by an
    exponential task. ``fit_all(data, shifted=True)`` returns shifted fits.

    Parameters
    ----------
    distribution : Distribution or dict
        A FlowSim distribution, or its ``spec()`` dictionary, which lets an
        exported specification rebuild the shifted distribution.
    by : float
        Amount added to every draw. A negative shift can produce negative
        draws, which stop a run.

    Returns
    -------
    Shifted
        An immutable distribution, callable with a random generator.

    Example
    -------
    >>> shifted(exponential(mean=4), by=2)
    Shifted(Exponential(mean=4), by=2)
    """
    example = "shifted(exponential(mean=4), by=2)"
    if isinstance(distribution, dict):
        arguments = dict(distribution)
        name = arguments.pop("name", None)
        if name not in _CONSTRUCTORS:
            raise _bad(
                "shifted",
                "a distribution or the spec() of one",
                example,
                {"distribution": distribution},
            )
        distribution = _CONSTRUCTORS[name](**arguments)
    if not isinstance(distribution, Distribution):
        raise _bad(
            "shifted",
            "a FlowSim distribution such as exponential(mean=4)",
            example,
            {"distribution": distribution},
        )
    (b,) = _real("shifted", example, by=by)
    return Shifted(distribution=distribution, by=b)


# Constructors by spec name, so shifted() can rebuild an exported inner spec.
_CONSTRUCTORS = {
    "beta": beta,
    "discrete": discrete,
    "empirical": empirical,
    "erlang": erlang,
    "exponential": exponential,
    "gamma": gamma,
    "lognormal": lognormal,
    "negative_binomial": negative_binomial,
    "normal": normal,
    "pert": pert,
    "poisson": poisson,
    "shifted": shifted,
    "triangular": triangular,
    "uniform": uniform,
    "weibull": weibull,
}
