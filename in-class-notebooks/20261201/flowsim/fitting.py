"""Fit distribution families to observed durations or counts.

``fit_all(data)`` fits each family by maximum likelihood, ranks the fits by
AIC, and returns a ``Fits`` table. ``best_fit(data)`` returns the winner, a
``Fit`` that drives a station or source like any FlowSim distribution.
"""

# Each Distribution subclass supplies mean and sd as a dataclass field or a
# read-only property. Pyright rejects both against the base's declaration, and
# a base property would block the frozen dataclass from setting a field.
# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

import html
import math
import os
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
import scipy.stats as st
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar

from flowsim.distributions import (
    Distribution,
    Shifted,
    beta,
    exponential,
    gamma,
    lognormal,
    negative_binomial,
    normal,
    poisson,
    triangular,
    uniform,
    weibull,
)
from flowsim.distributions import shifted as _shifted
from flowsim.plots import BLUE, GRAY, GREEN, ORANGE, PURPLE, style

# The default search covers interarrival times (exponential), right-skewed
# durations (gamma, lognormal, Weibull), and symmetric durations far from zero
# (normal). The four skewed families are fitted with location fixed at zero,
# since a duration cannot start below zero, and the normal is skipped by
# `Fits.best` whenever its chance of a negative draw exceeds NEGATIVE_LIMIT. Uniform, triangular, and
# beta can be fitted by naming them in `families=`, but they are not searched:
# with free endpoints that hug the sample's minimum and maximum, they win AIC on
# data from unrelated families (beta wins on most exponential samples). Johnson,
# Pearson, and log-logistic families are left out because an operations course
# cannot explain their parameters, and their small AIC gains would hide the
# families students can reason about.
FAMILIES = ("exponential", "gamma", "lognormal", "weibull", "normal")

# Every family `families=` accepts for durations, and for counts.
ALL_FAMILIES = (*FAMILIES, "uniform", "triangular", "beta")
COUNT_FAMILIES = ("poisson", "negative_binomial")

# `Fits.best` skips a fit whose chance of a negative draw exceeds this.
NEGATIVE_LIMIT = 1e-6

_SCIPY = {
    "exponential": st.expon,
    "gamma": st.gamma,
    "lognormal": st.lognorm,
    "weibull": st.weibull_min,
    "normal": st.norm,
    "uniform": st.uniform,
    "triangular": st.triang,
    "beta": st.beta,
}
_ZERO_LOCATION = ("exponential", "gamma", "lognormal", "weibull")
_ROUNDED = "values look rounded; record finer times or use empirical(data)"
# Parameter columns in `Fits.table`, in display order. Mean and sd parameters
# are already the table's mean and sd columns, so they are not repeated.
_PARAMETER_COLUMNS = ("shape", "scale", "low", "mode", "high", "a", "b", "by")
_COLORS = {
    "exponential": BLUE,
    "gamma": ORANGE,
    "lognormal": GREEN,
    "weibull": PURPLE,
    "normal": "#E69F00",
    "uniform": "#56B4E9",
    "triangular": "#000000",
    "beta": "#F0E442",
    "poisson": BLUE,
    "negative_binomial": ORANGE,
}


class _NoFit(Exception):
    """A family that cannot be fitted to these data, with the reason."""


def _short(distribution: Distribution) -> str:
    """A constructor-like repr with three significant digits."""
    if isinstance(distribution, Shifted):
        return f"Shifted({_short(distribution.distribution)}, by={distribution.by:.3g})"
    shown = ", ".join(
        f"{item.name}={getattr(distribution, item.name):.3g}"
        for item in fields(distribution)  # pyright: ignore[reportArgumentType]
    )
    return f"{type(distribution).__name__}({shown})"


@dataclass(frozen=True, repr=False, eq=False)
class Fit(Distribution):
    """One fitted family, usable anywhere FlowSim accepts a sampler.

    A fit draws exactly what its fitted ``distribution`` draws under the same
    generator, and its ``spec()`` is that distribution's, so an export records
    it as rebuildable.

    Attributes
    ----------
    family : str
        FlowSim family name, e.g. ``"gamma"``.
    distribution : Distribution
        The fitted FlowSim distribution, e.g. ``gamma(shape=4.4, scale=1.3)``.
    aic : float
        Akaike information criterion, 2k - 2 log-likelihood, where k counts
        only the fitted parameters. Lower is better.
    ks : float or None
        Kolmogorov-Smirnov statistic, i.e., the largest vertical gap between
        the data's empirical CDF and the fitted CDF. None for count fits.
    p_negative : float
        Chance that a draw is below zero.

    Example
    -------
    >>> fit = best_fit(np.random.default_rng(7).gamma(4, 1.5, 200))
    >>> fit.params  # e.g. {'shape': 4.39, 'scale': 1.26}
    """

    family: str
    distribution: Distribution
    aic: float
    ks: float | None
    p_negative: float
    _frozen: Any = field(repr=False)

    @property
    def params(self) -> dict[str, float]:
        """Fitted parameters by constructor keyword; a shift adds ``by``."""
        if isinstance(self.distribution, Shifted):
            inner = self.distribution.distribution.spec()
            inner.pop("name")
            return {**inner, "by": self.distribution.by}
        arguments = self.distribution.spec()
        arguments.pop("name")
        return arguments

    @property
    def mean(self) -> float:
        """Return the fitted distribution's mean, in the data's units.

        Example
        -------
        >>> fit.mean
        """
        return self.distribution.mean

    @property
    def sd(self) -> float:
        """Return the fitted distribution's standard deviation, in the data's units.

        Example
        -------
        >>> fit.sd
        """
        return self.distribution.sd

    def __call__(self, rng):
        return self.distribution(rng)

    def describe(self) -> str:
        """Return one plain sentence describing the fitted distribution.

        Returns
        -------
        str
            The fitted distribution's description.

        Example
        -------
        >>> fit.describe()
        """
        return self.distribution.describe()

    def spec(self) -> dict:
        """Return the fitted distribution's constructor name and arguments.

        Returns
        -------
        dict
            The key "name" and one key per constructor argument, at full
            precision.

        Example
        -------
        >>> fit.spec()
        """
        return self.distribution.spec()

    def __repr__(self) -> str:
        text = f"{_short(self.distribution)}  AIC={self.aic:.1f}"
        if self.ks is not None:
            text += f"  KS={self.ks:.3f}"
        if self.p_negative > NEGATIVE_LIMIT:
            text += f"  P(X<0)={self.p_negative:.3g}"
        return text


class Fits:
    """Fitted families ranked by AIC, returned by ``fit_all``.

    ``fits.best`` is the top-ranked fit whose chance of a negative draw is at
    most 1e-6 (such a draw still stops a run),
    ``fits.pick("gamma")`` or ``fits.pick(2)`` chooses another,
    ``fits.table`` compares them, ``fits.failed`` lists families that did not
    fit and why, and ``fits.plot()`` overlays the fits on the data. Iterating
    yields fits in rank order.

    Example
    -------
    >>> fits = fit_all(np.random.default_rng(7).gamma(4, 1.5, 200))
    >>> fits.best.family
    'gamma'
    """

    def __init__(
        self,
        data: np.ndarray,
        fits: list[Fit],
        failed: list[tuple[str, str]],
        discrete: bool,
    ):
        self._data = data
        # A stable sort keeps the search order among equal AICs.
        self._fits = tuple(sorted(fits, key=lambda fit: fit.aic))
        self._failed = tuple(failed)
        self._discrete = discrete

    def __len__(self) -> int:
        return len(self._fits)

    def __iter__(self) -> Iterator[Fit]:
        return iter(self._fits)

    @property
    def best(self) -> Fit:
        """The top-ranked fit whose chance of a negative draw is at most 1e-6.

        Example
        -------
        >>> fit_all([1.0, 2.0, 3.0, 6.0]).best
        """
        for fit in self._fits:
            if fit.p_negative <= NEGATIVE_LIMIT:
                return fit
        shown = ", ".join(f"{f.family} {f.p_negative:.3g}" for f in self._fits)
        raise ValueError(
            "Every fitted family can draw a negative duration (chance of a draw "
            f"below zero: {shown}), and a negative draw stops a run. Use "
            "empirical(data) to resample the observed values, or choose a fit "
            "deliberately with fits.pick(...)."
        )

    def pick(self, choice: str | int) -> Fit:
        """Return a fit by family name or by 1-based rank.

        Unlike ``best``, a picked fit may draw negative values; check its
        ``p_negative`` first.

        Example
        -------
        >>> fits = fit_all([1.0, 2.0, 3.0, 6.0])
        >>> fits.pick("gamma"), fits.pick(1)
        """
        names = [fit.family for fit in self._fits]
        if isinstance(choice, str):
            if choice in names:
                return self._fits[names.index(choice)]
            for family, reason in self._failed:
                if family == choice:
                    raise ValueError(f"{choice} did not fit: {reason}")
            raise ValueError(
                f"No fit named {choice!r}. Fitted families, best first: "
                f"{', '.join(names)}."
            )
        if (
            isinstance(choice, (bool, np.bool_))
            or not isinstance(choice, (int, np.integer))
            or not 1 <= choice <= len(names)
        ):
            ranked = ", ".join(f"{i} {name}" for i, name in enumerate(names, 1))
            raise ValueError(
                f"pick needs a family name or a rank from 1 to {len(names)} "
                f"({ranked}), not {choice!r}."
            )
        return self._fits[int(choice) - 1]

    @property
    def table(self) -> pl.DataFrame:
        """One row per fit in rank order.

        Columns are rank, family, description, mean, sd, p_negative, aic, ks,
        and then one numeric column per fitted parameter, named by its
        constructor keyword (e.g. ``shape`` and ``scale`` for gamma) and empty
        for families without it. A ``mean`` or ``sd`` parameter is the mean
        or sd column itself. For a shifted fit, ``by`` is the shift, so the
        unshifted distribution's mean is ``mean - by``.

        Example
        -------
        >>> fit_all([1.0, 2.0, 3.0, 6.0]).table.select("rank", "family", "aic")
        """
        rows = []
        for rank, fit in enumerate(self._fits, 1):
            params = fit.params
            rows.append(
                {
                    "rank": rank,
                    "family": fit.family,
                    "description": fit.describe(),
                    "mean": fit.mean,
                    "sd": fit.sd,
                    "p_negative": fit.p_negative,
                    "aic": fit.aic,
                    "ks": fit.ks,
                    **{name: params.get(name) for name in _PARAMETER_COLUMNS},
                }
            )
        used = [
            name
            for name in _PARAMETER_COLUMNS
            if any(row[name] is not None for row in rows)
        ]
        schema = {
            "rank": pl.Int64,
            "family": pl.String,
            "description": pl.String,
            "mean": pl.Float64,
            "sd": pl.Float64,
            "p_negative": pl.Float64,
            "aic": pl.Float64,
            "ks": pl.Float64,
            **{name: pl.Float64 for name in used},
        }
        return pl.DataFrame(
            [{name: row[name] for name in schema} for row in rows],
            schema=schema,
        )

    @property
    def failed(self) -> pl.DataFrame:
        """Each attempted family that did not fit, with the reason.

        Example
        -------
        >>> fit_all([0.0, 1.0, 2.0, 3.0]).failed  # lognormal cannot fit a zero
        """
        return pl.DataFrame(
            [{"family": family, "reason": reason} for family, reason in self._failed],
            schema={"family": pl.String, "reason": pl.String},
        )

    def plot(self) -> go.Figure:
        """Plot the data with each fit: densities on the left, CDFs on the right.

        Durations show a histogram scaled to a density; counts show the share
        of each count. The right panel compares the data's empirical CDF with
        each fitted CDF.

        Example
        -------
        >>> fit_all([1.0, 2.0, 3.0, 6.0]).plot().show()
        """
        data = self._data
        values = np.sort(data)
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Share of each count" if self._discrete else "Density",
                "Cumulative share",
            ),
        )
        if self._discrete:
            # Every count from zero to the largest, or at most 1,000 evenly
            # spaced whole numbers when that range is wider, so a few huge
            # counts cannot allocate a huge grid.
            grid = (
                np.arange(0, values[-1] + 1)
                if values[-1] < 1000
                else np.unique(np.linspace(0, values[-1], 1000).round())
            )
            observed, counts = np.unique(
                values,
                return_counts=True,
            )
            figure.add_trace(
                go.Bar(
                    x=observed,
                    y=counts / values.size,
                    name="Data",
                    marker_color=GRAY,
                    opacity=0.5,
                ),
                row=1,
                col=1,
            )
        else:
            grid = np.linspace(0, values[-1] * 1.1, 400)
            figure.add_trace(
                go.Histogram(
                    x=data,
                    histnorm="probability density",
                    name="Data",
                    marker_color=GRAY,
                    opacity=0.5,
                ),
                row=1,
                col=1,
            )
        best = next((f for f in self._fits if f.p_negative <= NEGATIVE_LIMIT), None)
        for rank, fit in enumerate(self._fits, 1):
            with np.errstate(all="ignore"):
                density = (
                    fit._frozen.pmf(grid) if self._discrete else fit._frozen.pdf(grid)
                )
                cumulative = fit._frozen.cdf(grid)
            line = {
                "color": _COLORS[fit.family],
                "width": 3.5 if fit is best else 2,
            }
            name = f"{rank}. {fit.family}" + (" (best)" if fit is best else "")
            figure.add_trace(
                go.Scatter(
                    x=grid,
                    y=np.where(np.isfinite(density), density, np.nan),
                    name=name,
                    legendgroup=fit.family,
                    mode="lines+markers" if self._discrete else "lines",
                    line=line,
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=grid,
                    y=cumulative,
                    name=name,
                    legendgroup=fit.family,
                    showlegend=False,
                    line={**line, "shape": "hv" if self._discrete else "linear"},
                ),
                row=1,
                col=2,
            )
        # Drawn last, so the fitted curves do not hide the data.
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([[0.0], values]),
                y=np.arange(values.size + 1) / values.size,
                name="Data",
                line={"color": GRAY, "width": 2.5, "shape": "hv"},
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        x_label = "Count" if self._discrete else "Duration"
        style(
            figure=figure,
            x_label=x_label,
            y_label="Share" if self._discrete else "Density",
        )
        figure.update_layout(
            width=1000,
            height=450,
            margin={"t": 90},
            legend={"y": 1.25},
            bargap=0.1,
        )
        figure.update_xaxes(
            title_text=x_label,
            rangemode="tozero",
            row=1,
            col=2,
        )
        figure.update_yaxes(
            title_text="Cumulative share",
            range=[0, 1.02],
            row=1,
            col=2,
        )
        return figure

    def _summary(self) -> str:
        data = self._data
        what = "counts" if self._discrete else "durations"
        return (
            f"Fits to {data.size} {what}: mean {data.mean():.4g}, "
            f"SD {data.std(ddof=1):.4g}, min {data.min():.4g}, max {data.max():.4g}"
        )

    def _rows(self) -> list[tuple[str, ...]]:
        rows = []
        for rank, fit in enumerate(self._fits, 1):
            negative = f"{fit.p_negative:.2g}"
            if fit.p_negative > NEGATIVE_LIMIT:
                negative += " (skipped by .best)"
            rows.append(
                (
                    str(rank),
                    fit.family,
                    f"{fit.aic:.1f}",
                    "" if fit.ks is None else f"{fit.ks:.3f}",
                    negative,
                    _short(fit.distribution),
                )
            )
        return rows

    _HEADER = ("rank", "family", "AIC", "KS", "P(X<0)", "distribution")
    _NEXT = (
        'Next: fits.best, fits.pick("name") or fits.pick(rank), fits.plot(), '
        "fits.table, fits.failed"
    )

    def __repr__(self) -> str:
        rows = [self._HEADER, *self._rows()]
        widths = [max(len(row[i]) for row in rows) for i in range(len(self._HEADER))]
        lines = [self._summary()]
        lines += [
            "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows
        ]
        for family, reason in self._failed:
            lines.append(f"Did not fit: {family} ({reason})")
        lines.append(self._NEXT)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        def cells(row, tag):
            return "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in row)

        body = "".join(f"<tr>{cells(row, 'td')}</tr>" for row in self._rows())
        failed = "".join(
            f"<li>{html.escape(family)}: {html.escape(reason)}</li>"
            for family, reason in self._failed
        )
        return (
            f"<p>{html.escape(self._summary())}</p>"
            f"<table><thead><tr>{cells(self._HEADER, 'th')}</tr></thead>"
            f"<tbody>{body}</tbody></table>"
            + (f"<p>Did not fit:</p><ul>{failed}</ul>" if failed else "")
            + f"<p><code>{html.escape(self._NEXT)}</code></p>"
        )


def fit_all(
    data,
    families: Sequence[str] | None = None,
    *,
    shifted: bool = False,
    discrete: bool = False,
) -> Fits:
    """Fit each family to the data by maximum likelihood and rank them by AIC.

    Durations are fitted with location fixed at zero for exponential, gamma,
    lognormal, and Weibull, since a duration cannot start below zero, and each
    fit's AIC is 2k - 2 log-likelihood with k counting only the fitted
    parameters. KS, the largest vertical gap between the empirical and fitted
    CDFs, is shown for comparison but does not set the rank. A family whose fit
    fails is listed in ``.failed`` with the reason.

    Parameters
    ----------
    data : sequence of float
        At least two observed durations, each finite and at least zero, or
        whole-number counts when ``discrete=True``.
    families : sequence of str, optional
        Families to fit. Defaults to ``FAMILIES`` for durations and
        ``COUNT_FAMILIES`` for counts. ``ALL_FAMILIES`` adds uniform,
        triangular, and beta, which are fitted only when named.
    shifted : bool
        Fit a free location (a minimum time) for exponential, gamma,
        lognormal, and Weibull, which adds one parameter to each. The default
        is False, and it cannot combine with discrete=True.
    discrete : bool
        Fit Poisson and negative binomial counts instead of durations. The
        default is False.

    Returns
    -------
    Fits
        The fitted families ranked by AIC, with best, pick, table, failed, and
        plot.

    Example
    -------
    >>> fits = fit_all(np.random.default_rng(7).gamma(4, 1.5, 200))
    >>> fits.best, fits.pick("lognormal"), fits.table
    """
    for name, flag in (("shifted", shifted), ("discrete", discrete)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be True or False, not {flag!r}.")
    if shifted and discrete:
        raise ValueError("shifted=True applies to durations, not discrete=True counts.")
    values = _data(data, discrete=bool(discrete))
    names = _names(families, discrete=bool(discrete))
    if not discrete:
        _warn_if_rounded(values)
    fits, failed = [], []
    for name in names:
        try:
            fits.append(
                _fit_count(values, name)
                if discrete
                else _fit_duration(
                    values=values,
                    family=name,
                    free_location=bool(shifted) or name not in _ZERO_LOCATION,
                )
            )
        except _NoFit as error:
            failed.append((name, str(error)))
        # SciPy's optimizers raise these on a bad start or on data a family
        # cannot fit, and a failed fit means only that this family is not
        # ranked. Any other error is a bug, so it propagates.
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            OverflowError,
            ZeroDivisionError,
            np.linalg.LinAlgError,
        ) as error:
            failed.append((name, f"{type(error).__name__}: {error}"))
    if not fits:
        reasons = "; ".join(f"{family}: {reason}" for family, reason in failed)
        raise ValueError(f"No family produced a valid fit to the data ({reasons}).")
    return Fits(
        data=values,
        fits=fits,
        failed=failed,
        discrete=bool(discrete),
    )


def best_fit(
    data,
    families: Sequence[str] | None = None,
    *,
    shifted: bool = False,
) -> Fit:
    """Return the best fit to a sample of durations, ready for ``service_time=``.

    This is ``fit_all(data, families, shifted=shifted).best``: the
    lowest-AIC fit whose chance of a negative draw is at most 1e-6. Use
    ``fit_all`` to see every candidate and why some failed.

    Parameters
    ----------
    data : sequence of float
        At least two observed durations, each finite and at least zero.
    families : sequence of str, optional
        Families to fit. Defaults to ``FAMILIES``. ``ALL_FAMILIES`` adds
        uniform, triangular, and beta, which are fitted only when named.
    shifted : bool
        Fit a free location (a minimum time) for exponential, gamma,
        lognormal, and Weibull. The default is False.

    Returns
    -------
    Fit
        The best fit, which is callable as a sampler.

    Example
    -------
    >>> fit = best_fit(np.random.default_rng(7).gamma(4, 1.5, 200))
    >>> model.station(name="Machine", capacity=1, service_time=fit)
    """
    return fit_all(
        data,
        families,
        shifted=shifted,
    ).best


def _fit_duration(values: np.ndarray, family: str, free_location: bool) -> Fit:
    scipy_family = _SCIPY[family]
    if not free_location and family != "exponential" and np.any(values == 0):
        raise _NoFit(
            f"the data include zero, where a {family} starting at zero has no "
            "finite log-likelihood; use shifted=True or empirical(data)"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = (
            scipy_family.fit(values)
            if free_location
            else scipy_family.fit(values, floc=0)
        )
        frozen = scipy_family(*params)
        loglik = float(np.sum(frozen.logpdf(values)))
        ks = float(st.kstest(values, frozen.cdf).statistic)
        p_negative = float(frozen.cdf(0))
    if not math.isfinite(loglik):
        raise _NoFit("the fitted density is zero or infinite at some data value")
    *shapes, loc, scale = (float(p) for p in params)
    # With a free location, the gamma, lognormal, and Weibull likelihoods grow
    # without bound as the location approaches the smallest observation, so a
    # fit that ends there is not a maximum (e.g., a lognormal with mean 9e50
    # from 1, 2, 3, 4). Measured fitted gaps are either below 1e-15 of the
    # data's range or above 1e-3, so 1e-6 separates them. The exponential's
    # location equals the minimum at its true maximum, so it is kept.
    if free_location and family in ("gamma", "lognormal", "weibull"):
        if float(values.min()) - loc <= 1e-6 * float(np.ptp(values)):
            raise _NoFit(
                "its fitted minimum time reached the smallest observation, "
                "where the likelihood has no maximum; fit without shifted=True"
            )
    if family == "exponential":
        distribution = exponential(mean=scale)
    elif family == "gamma":
        distribution = gamma(shape=shapes[0], scale=scale)
    elif family == "lognormal":
        # SciPy's lognormal has log-scale sd s and median scale; FlowSim's
        # takes the duration's own mean and sd.
        s = shapes[0]
        mean = scale * math.exp(s**2 / 2)
        distribution = lognormal(mean=mean, sd=mean * math.sqrt(math.expm1(s**2)))
    elif family == "weibull":
        distribution = weibull(shape=shapes[0], scale=scale)
    elif family == "normal":
        distribution = normal(mean=loc, sd=scale)
    else:
        # These families fit free endpoints, which can fall below zero. Their
        # constructors start at zero or above, so a negative fitted start
        # becomes the same distribution from zero, shifted down; p_negative
        # then reports the chance of a negative draw.
        start = max(loc, 0.0)
        if family == "uniform":
            distribution = uniform(low=start, high=start + scale)
        elif family == "triangular":
            distribution = triangular(
                low=start,
                mode=start + shapes[0] * scale,
                high=start + scale,
            )
        else:
            distribution = beta(
                a=shapes[0],
                b=shapes[1],
                low=start,
                high=start + scale,
            )
        if loc < 0:
            distribution = _shifted(distribution, by=loc)
    if family in _ZERO_LOCATION and free_location:
        distribution = _shifted(distribution, by=loc)
    free = len(params) - (0 if free_location else 1)
    return Fit(
        family=family,
        distribution=distribution,
        aic=2 * free - 2 * loglik,
        ks=ks,
        p_negative=p_negative,
        _frozen=frozen,
    )


def _fit_count(values: np.ndarray, family: str) -> Fit:
    # Both maximum likelihood fits set the mean to the sample mean.
    mean, variance = float(values.mean()), float(values.var())
    if mean == 0:
        raise _NoFit("every count is zero, so demand is the constant per_period=0")
    if family == "poisson":
        frozen = st.poisson(mean)
        distribution, free = poisson(mean=mean), 1
    else:
        # The negative binomial with mean m and size r has variance
        # m + m^2 / r. Its likelihood has a finite maximum only when the
        # counts' variance (dividing by n) exceeds their mean.
        if variance <= mean:
            raise _NoFit(
                f"the counts' variance ({variance:.4g}) is not above their mean "
                f"({mean:.4g}), so the fit degenerates to Poisson"
            )

        # SciPy has no fit for discrete families. With the mean at the sample
        # mean, the log-likelihood has one maximum in the size r, so maximize
        # SciPy's log-pmf (computed through log-gamma functions, which keeps
        # its precision for large counts) over log r within wide bounds.
        def negative_loglik(log_size: float) -> float:
            size = math.exp(log_size)
            return -float(np.sum(st.nbinom.logpmf(values, size, size / (size + mean))))

        bounds = (math.log(1e-8), math.log(1e15))
        search = minimize_scalar(
            negative_loglik,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-10},
        )
        # SciPy types the OptimizeResult loosely, so Pyright does not see x.
        best = float(search.x)  # pyright: ignore[reportAttributeAccessIssue]
        if not search.success or best > bounds[1] - 1e-3:
            raise _NoFit("the likelihood has no interior maximum for these counts")
        size = math.exp(best)
        frozen = st.nbinom(size, size / (size + mean))
        distribution = negative_binomial(mean=mean, sd=math.sqrt(mean + mean**2 / size))
        free = 2
    # SciPy types a frozen count distribution as continuous or discrete, so
    # Pyright does not see logpmf.
    loglik = float(np.sum(frozen.logpmf(values)))  # pyright: ignore[reportAttributeAccessIssue]
    return Fit(
        family=family,
        distribution=distribution,
        aic=2 * free - 2 * loglik,
        ks=None,
        p_negative=0.0,
        _frozen=frozen,
    )


def _data(data, discrete: bool) -> np.ndarray:
    try:
        raw = np.asarray(data)
    except (TypeError, ValueError):
        raise ValueError("Fitting data must be a sequence of numbers.") from None
    # float() would read "2.5" as 2.5 and True as 1, so text and booleans are
    # refused by kind. NumPy reads [True, 2] as integers, so a list or tuple is
    # also checked item by item, as is an object array (e.g., one holding None,
    # which still reaches the finite-number check below).
    if raw.dtype.kind == "O":
        items = raw.ravel()
    elif isinstance(data, (list, tuple)):
        items = data
    else:
        items = ()
    if raw.dtype.kind not in "iufO" or any(
        isinstance(item, (str, bytes, bool, np.bool_)) for item in items
    ):
        raise ValueError(
            "Fitting data must be a sequence of numbers, not text or booleans "
            f"(got {raw.dtype} values)."
        )
    try:
        values = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("Fitting data must be a sequence of numbers.") from None
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "Fitting data must be a one-dimensional sequence of at least two values."
        )
    if not np.isfinite(values).all():
        raise ValueError("Fitting data must contain only finite numbers.")
    if np.any(values < 0):
        raise ValueError(
            "Fitting data must be at least zero, since a duration or count "
            f"cannot be negative; the smallest value is {values.min():g}."
        )
    if discrete and np.any(values != np.round(values)):
        raise ValueError("Fitting data for discrete=True must be whole-number counts.")
    values.flags.writeable = False
    return values


def _warn_if_rounded(values: np.ndarray) -> None:
    _, counts = np.unique(
        values,
        return_counts=True,
    )
    # The share of observations that equal another observation.
    tied = float(counts[counts > 1].sum() / values.size)
    zeros = int(np.sum(values == 0))
    if tied > 0.1 or zeros:
        warnings.warn(
            f"Fitting data: {_ROUNDED}. {tied:.0%} of values equal another "
            f"value and {zeros} are exactly zero.",
            UserWarning,
            # Point at the caller's line, whichever fitting function it called.
            skip_file_prefixes=(os.path.dirname(__file__),),
        )


def _names(families: Sequence[str] | None, discrete: bool) -> list[str]:
    allowed = COUNT_FAMILIES if discrete else ALL_FAMILIES
    if families is None:
        return list(allowed if discrete else FAMILIES)
    if isinstance(families, str):
        raise ValueError(
            f'families must be a list of family names, e.g. ["{families}"], '
            "not one string."
        )
    names = list(dict.fromkeys(families))
    if not names:
        raise ValueError("families must name at least one distribution.")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        kind = "counts (discrete=True)" if discrete else "durations"
        hint = ""
        if not discrete and set(unknown) & set(COUNT_FAMILIES):
            hint = " Pass discrete=True to fit counts."
        raise ValueError(
            f"Unknown families {unknown!r} for {kind}; choose from "
            f"{', '.join(allowed)}.{hint}"
        )
    return names
