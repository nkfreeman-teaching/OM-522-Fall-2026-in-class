"""Scenario comparisons with matched replication seeds and paired intervals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from flowsim.catalog import unknown
from flowsim.display import (
    display_html,
    display_text,
    missing_attribute,
    window_line,
)
from flowsim.model import Model
from flowsim.plots import plotly_figure
from flowsim.reporting import (
    comparison_differences,
    comparison_intervals,
    dashboard,
    export_tables,
    interval_figure,
    request_metadata,
)
from flowsim.results import (
    Experiment,
    Settings,
    _run_settings,
    stack,
)
from flowsim.viewer import Viewer


class Comparison:
    """Named scenarios run with matched replication seeds.

    flowsim.compare returns a Comparison. Replication k of every scenario uses
    the same spawned seed, so differences are paired by replication. The first
    scenario is the baseline unless a method names another.

    Attributes
    ----------
    experiments : dict of str to Experiment
        Each scenario's experiment, in the order given.
    baseline : str
        Name of the first scenario, the default reference for differences.
    settings : Settings
        The settings every scenario shares. The seed, end time, and capacities
        are None.
    summary, stations, stock_points : polars.DataFrame
        Each scenario's pooled tables stacked, with a scenario column.

    The common next calls are intervals, differences, plot, viewer, and export.

    Example
    -------
    >>> comparison = flowsim.compare(
    ...     scenarios={"One desk": one, "Two desks": two},
    ...     duration=480,
    ... )
    >>> comparison.differences("mean_total_wait")
    """

    experiments: dict[str, Experiment]
    baseline: str

    def __init__(self, experiments: dict[str, Experiment]):
        if len(experiments) < 2:
            raise ValueError("A comparison needs at least two experiments.")
        self.experiments = experiments
        self.baseline = next(iter(experiments))
        first = experiments[self.baseline].runs[0]
        if any(
            _run_settings(experiment.runs[0]) != _run_settings(first)
            for experiment in experiments.values()
        ):
            raise ValueError(
                "Compared scenarios need the same duration, warm-up, drain rule, "
                "and time unit."
            )

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        raise missing_attribute(self, name, {})

    def __repr__(self) -> str:
        return display_text(*self._display())

    def _repr_html_(self) -> str:
        return display_html(*self._display())

    def _display(self) -> tuple[str, list[tuple[str, str]]]:
        rows = [
            (
                "Scenarios" if index == 0 else "",
                f"{name}{' (baseline)' if name == self.baseline else ''}: "
                f"{len(experiment.runs)} replications",
            )
            for index, (name, experiment) in enumerate(self.experiments.items())
        ]
        rows += [
            ("Settings", window_line(self.settings)),
            ("Next", "comparison.intervals(metric), comparison.differences(metric)"),
            ("", "comparison.plot(), comparison.viewer(), comparison.export(folder)"),
        ]
        return f"Comparison: {len(self.experiments)} scenarios", rows

    @property
    def settings(self) -> Settings:
        """Return the settings every scenario shares.

        The seed and end time differ by run, and capacities differ by scenario,
        so they are None here. Each run's own settings hold them.
        """
        return replace(
            self.experiments[self.baseline].settings,
            capacities=None,
        )

    # The combined tables are built from the experiments on each access, so
    # estimates and exports always read the same replication values.
    @property
    def summary(self) -> pl.DataFrame:
        """Return every scenario's pooled summary table, with a scenario column."""
        return stack(
            parts=(
                (name, experiment.summary)
                for name, experiment in self.experiments.items()
            ),
            column="scenario",
        )

    @property
    def stations(self) -> pl.DataFrame:
        """Return every scenario's pooled stations table, with a scenario column."""
        return stack(
            parts=(
                (name, experiment.stations)
                for name, experiment in self.experiments.items()
            ),
            column="scenario",
        )

    @property
    def stock_points(self) -> pl.DataFrame:
        """Return every scenario's pooled stock_points table, with a scenario column."""
        return stack(
            parts=(
                (name, experiment.stock_points)
                for name, experiment in self.experiments.items()
            ),
            column="scenario",
        )

    def _reference(self, baseline: str | None) -> str:
        reference = self.baseline if baseline is None else baseline
        if reference not in self.experiments:
            raise unknown(
                kind="baseline scenario",
                name=reference,
                choices=list(self.experiments),
                intro=f"Unknown baseline scenario {reference!r}.",
            )
        return reference

    def intervals(
        self,
        metric: str = "mean_total_wait",
        *,
        station: str | None = None,
        stock_point: str | None = None,
        confidence: float = 0.95,
    ) -> pl.DataFrame:
        """Estimate a metric in each scenario with Student t intervals.

        The rows follow Experiment.intervals within each scenario, so a station
        metric without station= gives one row per station per scenario.

        Parameters
        ----------
        metric : str, default 'mean_total_wait'
            A replication metric name from flowsim.metrics().
        station : str, optional
            Station to report, for a station metric.
        stock_point : str, optional
            Stock point to report, for a stock-point metric.
        confidence : float, default 0.95
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        polars.DataFrame
            One row per scenario and location with the mean and interval limits.

        Example
        -------
        >>> comparison.intervals("utilization", station="Desk")
        """
        return comparison_intervals(
            experiments=self.experiments,
            metric=metric,
            station=station,
            stock_point=stock_point,
            confidence=confidence,
        )

    def differences(
        self,
        metric: str = "mean_total_wait",
        *,
        baseline: str | None = None,
        station: str | None = None,
        stock_point: str | None = None,
        confidence: float = 0.95,
    ) -> pl.DataFrame:
        """Estimate each scenario minus the baseline within matched replications.

        A station or stock-point metric without a location gives one row per
        location that the scenario shares with the baseline.

        Parameters
        ----------
        metric : str, default 'mean_total_wait'
            A replication metric name from flowsim.metrics().
        baseline : str, optional
            Name of the reference scenario. Without it, the first scenario.
        station : str, optional
            Station to report, for a station metric.
        stock_point : str, optional
            Stock point to report, for a stock-point metric.
        confidence : float, default 0.95
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        polars.DataFrame
            One row per scenario and location with the mean paired difference
            (scenario minus baseline) and its interval.

        Example
        -------
        >>> comparison.differences("mean_total_wait", baseline="One desk")
        """
        return comparison_differences(
            experiments=self.experiments,
            reference=self._reference(baseline),
            metric=metric,
            station=station,
            stock_point=stock_point,
            confidence=confidence,
        )

    def plot(
        self,
        metric: str = "mean_total_wait",
        *,
        station: str | None = None,
        stock_point: str | None = None,
        confidence: float = 0.95,
        differences: bool = False,
        baseline: str | None = None,
    ) -> go.Figure:
        """Plot each scenario's interval, or its paired difference from the baseline.

        Parameters
        ----------
        metric : str, default 'mean_total_wait'
            A replication metric name from flowsim.metrics().
        station : str, optional
            Station to plot, for a station metric.
        stock_point : str, optional
            Stock point to plot, for a stock-point metric.
        confidence : float, default 0.95
            Confidence level of the interval, strictly between 0 and 1.
        differences : bool, default False
            Whether to plot paired differences from the baseline instead of each
            scenario's own interval.
        baseline : str, optional
            Reference scenario for differences. Without it, the first scenario.

        Returns
        -------
        plotly.graph_objects.Figure
            The interval figure.

        Example
        -------
        >>> comparison.plot("mean_total_wait", differences=True).show()
        """
        table = (
            self.differences(
                metric=metric,
                baseline=baseline,
                station=station,
                stock_point=stock_point,
                confidence=confidence,
            )
            if differences
            else self.intervals(
                metric=metric,
                station=station,
                stock_point=stock_point,
                confidence=confidence,
            )
        )
        return plotly_figure(
            interval_figure(
                table=table,
                metric=metric,
                time_unit=self.settings.time_unit,
                differences=differences,
            )
        )

    def viewer(
        self,
        path: str | Path | None = None,
        *,
        metric: str = "mean_total_wait",
        station: str | None = None,
        stock_point: str | None = None,
        confidence: float = 0.95,
        baseline: str | None = None,
    ) -> Viewer:
        """Build the offline comparison dashboard, saved to path if given.

        A menu chooses among every replication metric that applies to some
        scenario, another chooses the location for a station or stock-point
        metric, and a third chooses the baseline. The page shows each
        scenario's interval, the paired differences from the baseline, and each
        replication's value. When the runs kept their records, selecting a
        scenario's replication opens that run's single-run view on the same
        page, and with keep_records=False the page shows the overview only.
        The arguments set the metric, location, and baseline shown first.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            File to save the page to. Without it, nothing is written.
        metric : str, default 'mean_total_wait'
            The replication metric shown first, a name from flowsim.metrics().
        station : str, optional
            Station shown first, for a station metric. Without it, the first.
        stock_point : str, optional
            Stock point shown first, for a stock-point metric. Without it, the
            first.
        confidence : float, default 0.95
            Confidence level of every interval, strictly between 0 and 1.
        baseline : str, optional
            Reference scenario shown first. Without it, the first scenario.

        Returns
        -------
        Viewer
            The page.

        Example
        -------
        >>> comparison.viewer("output/comparison.html", metric="overtime")
        """
        reference = self._reference(baseline)
        # Check the request as intervals() and differences() do, before
        # building the page.
        self.differences(
            metric=metric,
            baseline=reference,
            station=station,
            stock_point=stock_point,
            confidence=confidence,
        )
        viewer = Viewer(
            document=dashboard(
                experiments=self.experiments,
                comparing=True,
                metric=metric,
                station=station,
                stock_point=stock_point,
                confidence=confidence,
                reference=reference,
            ),
            kind="comparison dashboard",
            name="comparison",
            height=1100,
        )
        if path is not None:
            viewer.save(path)
        return viewer

    def export(
        self,
        directory: str | Path,
        *,
        metric: str = "mean_total_wait",
        station: str | None = None,
        stock_point: str | None = None,
        confidence: float = 0.95,
        baseline: str | None = None,
    ) -> Path:
        """Write pooled run tables, intervals, differences, and metadata.

        Parameters
        ----------
        directory : str or pathlib.Path
            Folder to write to. It is created if needed.
        metric : str, default 'mean_total_wait'
            A replication metric name from flowsim.metrics().
        station : str, optional
            Station to report, for a station metric.
        stock_point : str, optional
            Stock point to report, for a stock-point metric.
        confidence : float, default 0.95
            Confidence level of the interval, strictly between 0 and 1.
        baseline : str, optional
            Reference scenario for differences. Without it, the first scenario.

        Returns
        -------
        pathlib.Path
            The folder.

        Example
        -------
        >>> comparison.export("output/comparison")
        """
        reference = self._reference(baseline)
        request = {
            "metric": metric,
            "station": station,
            "stock_point": stock_point,
            "confidence": confidence,
        }
        supply = any(
            experiment.stock_point_names for experiment in self.experiments.values()
        )
        return export_tables(
            directory,
            summary=self.summary,
            stations=self.stations,
            stock_points=self.stock_points if supply else None,
            intervals=self.intervals(**request),
            differences=self.differences(
                **request,
                baseline=reference,
            ),
            metadata={
                "baseline": reference,
                **request_metadata(
                    experiment=self.experiments[reference],
                    **request,
                ),
                "difference": "Scenario minus baseline within matched replication",
            },
        )


def compare(
    scenarios: Mapping[str, Model],
    duration: float,
    *,
    replications: int = 20,
    seed: int = 42,
    warmup: float = 0,
    drain: bool = False,
    keep_records: bool = True,
) -> Comparison:
    """Run named scenarios using the same independent replication seeds.

    Each scenario runs Model.experiment with the same duration, replications,
    seed, warmup, drain, and keep_records, so replication k uses the same seed
    in every scenario. The scenarios must declare the same time unit, and the
    first is the baseline.

    Parameters
    ----------
    scenarios : mapping of str to Model
        At least two named models. Each is validated.
    duration : float
        Closing time of each run in model time units, greater than 0.
    replications : int, default 20
        Number of replications per scenario, at least 2.
    seed : int, default 42
        Seed from which each replication's seed is spawned, at least 0.
    warmup : float, default 0
        Initial period excluded from measurement in each run.
    drain : bool, default False
        Whether each run continues past duration until no customers remain.
    keep_records : bool, default True
        Whether each run keeps its raw records. False drops them once the run's
        measures are computed, as in Model.experiment.

    Returns
    -------
    Comparison
        The scenarios' experiments, with paired differences.

    Example
    -------
    >>> comparison = flowsim.compare(
    ...     scenarios={"One desk": one, "Two desks": two},
    ...     duration=480,
    ...     replications=30,
    ... )
    """
    if not isinstance(scenarios, Mapping) or len(scenarios) < 2:
        raise ValueError("A comparison needs at least two named models.")
    for name, model in scenarios.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each scenario needs a nonempty name.")
        if not isinstance(model, Model):
            raise ValueError(f"Scenario {name!r} must contain a Model.")
        model.validate()
    if len({model.time_unit for model in scenarios.values()}) != 1:
        raise ValueError("Compared scenarios must declare the same time unit.")
    return Comparison(
        {
            name: model.experiment(
                duration=duration,
                replications=replications,
                seed=seed,
                warmup=warmup,
                drain=drain,
                keep_records=keep_records,
            )
            for name, model in scenarios.items()
        }
    )
