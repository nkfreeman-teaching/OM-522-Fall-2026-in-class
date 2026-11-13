"""Raw records and duration-weighted summaries with explicit censoring."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import polars as pl

from flowsim.catalog import (
    METRICS,
    STATION_METRICS,
    STOCK_POINT_METRICS,
    SYSTEM_METRICS,
    inapplicable,
    metric_units,
)
from flowsim.display import (
    display_html,
    display_text,
    figure,
    missing_attribute,
    window_line,
)
from flowsim.plots import (
    inventory_figure,
    orders_figure,
    plot_customer,
    plotly_figure,
    queue_figure,
    require_stations,
    resources_figure,
    run_page_html,
    utilization_figure,
    wait_figure,
)
from flowsim.records import (
    ACTIVITY_SCHEMA,
    AREA_FIELDS,
    CUSTOMER_SCHEMA,
    FAILURE_SCHEMA,
    INVENTORY_SCHEMA,
    SHIPMENT_SCHEMA,
    STATE_SCHEMA,
    STATION_SCHEMA,
    STOCK_POINT_SCHEMA,
    SUMMARY_HEADLINE,
    VISIT_SCHEMA,
    Status,
    _Activity,
    _Customer,
    _Failure,
    _frame,
    _Inventory,
    _Shipment,
    _State,
    _Visit,
    headline_first,
)
from flowsim.reporting import (
    dashboard,
    experiment_intervals,
    export_tables,
    interval_figure,
    request_metadata,
    write_export,
)
from flowsim.viewer import Viewer

if TYPE_CHECKING:
    import networkx as nx
    from plotly.graph_objects import Figure


def _run_settings(run: Results) -> tuple:
    """Return the settings that pooled or compared runs must share."""
    settings = run.settings
    return (settings.duration, settings.warmup, settings.drain, settings.time_unit)


@dataclass(frozen=True)
class Settings:
    """The settings of a run: its window, stopping rule, seed, and capacities.

    end_time is when the run stopped, which is later than duration in a drained
    run. capacities is a read-only mapping from each station to its servers.
    An experiment's or comparison's settings hold None for the seed and end
    time, which differ by run, and a comparison's hold None for capacities,
    which differ by scenario.

    Attributes
    ----------
    duration : float
        Closing time in model time units.
    warmup : float
        Initial period excluded from measurement.
    drain : bool
        Whether the run continued past duration until no customers remained.
    seed : int or None
        The run's seed.
    time_unit : str
        The model's time unit label.
    end_time : float or None
        When the run stopped.
    capacities : mapping of str to int or None
        Installed servers at each station.

    Example
    -------
    >>> results.settings.duration
    480.0
    """

    duration: float
    warmup: float
    drain: bool
    seed: int | None
    time_unit: str
    end_time: float | None
    capacities: Mapping[str, int] | None


def _frozen(value: object) -> object:
    """Return a read-only copy: mappings become mappingproxies, lists tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    return value


def _plain(value: object) -> object:
    """Undo _frozen for JSON export: mappings become dicts, tuples lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SupplySettings:
    """The supply-chain settings of a run, which its export records.

    Every field is read-only, including the mapping of each stock point's
    settings and the distribution records inside it.

    Attributes
    ----------
    chain : tuple of str
        The demand source, then each stock point in chain order.
    period : float
        The review period.
    stock_points : mapping of str to mapping
        Each stock point's initial stock, lead time (a number, a distribution
        record, "<function>", or "production"), batch size, cover, shortage
        rule, forecast source, forecast window, and safety factor.
    demand : object
        Consumer demand per period: a number, a distribution record, or
        "<function>".
    reconstructible : bool
        Whether the demand and every lead time can be rebuilt from the export.

    Example
    -------
    >>> results.supply.chain
    ('Consumers', 'Retailer', 'Wholesaler', 'Distributor', 'Factory')
    """

    chain: tuple[str, ...]
    period: float
    stock_points: Mapping[str, Mapping[str, object]]
    demand: object
    reconstructible: bool

    def __post_init__(self) -> None:
        # A frozen dataclass sets its own fields only through object.__setattr__.
        for name in ["chain", "stock_points", "demand"]:
            object.__setattr__(self, name, _frozen(getattr(self, name)))


# A run's raw record tables, which model.experiment(..., keep_records=False)
# drops once the run's measures are computed.
RAW_TABLES = (
    "customers",
    "visits",
    "states",
    "activities",
    "failures",
    "inventory",
    "shipments",
)

_DROPPED = (
    "This run's raw records were dropped by model.experiment(..., "
    "keep_records=False). Rerun with keep_records=True to keep them."
)


def stack(
    parts: Iterable[tuple[object, pl.DataFrame]],
    column: str,
) -> pl.DataFrame:
    """Stack labeled tables, with each table's label in a new first column."""
    return pl.concat(
        [table.select(pl.lit(value).alias(column), pl.all()) for value, table in parts]
    )


class Results:
    """The records and measures of one simulation run.

    A run's raw records are the customers, visits, states, activities, failures,
    and (for a supply chain) inventory and shipments tables. Its measures are
    the stations, summary, and stock_points tables. Queue averages,
    utilization, throughput, and the measurement counts use the window
    [warmup, duration). Whole-run counts such as arrivals and completed cover
    the entire run, including a drained continuation, and wait and flow-time
    averages use cohorts of arrivals at or after warmup whose outcomes were
    observed. flowsim.explain(name) gives each measure's window. Model.run
    returns a Results.

    Attributes
    ----------
    settings : Settings
        The duration, warmup, drain rule, seed, time unit, end time, and
        capacities of the run.
    customers, visits, states, activities, failures : polars.DataFrame
        Raw records: each customer, each station visit, station state changes,
        processing and blocked intervals, and server failures.
    inventory, shipments : polars.DataFrame
        Raw records of each stock-point review and each shipment. Both are empty
        for a service network.
    stations, summary, stock_points : polars.DataFrame
        Measures per station, for the whole system, and per stock point. A
        table that does not apply to the model is empty.
    supply : SupplySettings or None
        The chain order, review period, and stock-point settings, or None for
        a service network.
    stock_point_names : list of str
        The stock points in chain order, empty for a service network.
    records_kept : bool
        False after model.experiment(..., keep_records=False) dropped the raw
        records.

    The common next calls are report, plot_queue, plot_wait, plot_utilization,
    plot_inventory, viewer, and export. Call flowsim.explain(name) for a column.

    Example
    -------
    >>> results = model.run(duration=480)
    >>> results.summary.select("arrivals", "completed", "mean_total_wait")
    """

    settings: Settings
    supply: SupplySettings | None
    customers: pl.DataFrame
    visits: pl.DataFrame
    states: pl.DataFrame
    activities: pl.DataFrame
    failures: pl.DataFrame
    inventory: pl.DataFrame
    shipments: pl.DataFrame
    stations: pl.DataFrame
    summary: pl.DataFrame
    stock_points: pl.DataFrame

    def __getattr__(self, name: str) -> object:
        # Called only for a missing attribute. Underscore names, which pickle,
        # copy, and IPython probe, get the plain error with no hint.
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        hints = {table: _DROPPED for table in RAW_TABLES}
        hints["intervals"] = (
            "A single run has no replications. Run model.experiment(...) for intervals."
        )
        hints["differences"] = (
            "A single run has no replications. Run model.experiment(...) for each "
            "scenario, then flowsim.compare(...) for differences."
        )
        hints["plot"] = (
            "A single run has plot_queue(), plot_wait(), plot_utilization(), "
            "plot_resources(), plot_customer(id), plot_inventory(), and "
            "plot_orders(). For intervals across replications, run "
            "model.experiment(...) and call experiment.plot()."
        )
        raise missing_attribute(self, name, hints)

    @property
    def stock_point_names(self) -> list[str]:
        """Return the stock points in chain order, empty for a service network."""
        return [] if self.supply is None else list(self.supply.chain[1:])

    @property
    def records_kept(self) -> bool:
        """Return whether the raw record tables are still held."""
        # _drop_records deletes every raw table, so one stands for all.
        return "customers" in vars(self)

    def _drop_records(self) -> None:
        """Delete the raw record tables and keep the measures."""
        for name in RAW_TABLES:
            delattr(self, name)

    def _require_records(self, call: str) -> None:
        """Raise a clear error when a call needs raw records that were dropped."""
        if not self.records_kept:
            raise ValueError(f"results.{call} reads raw records. {_DROPPED}")

    def __repr__(self) -> str:
        return display_text(*self._display())

    def _repr_html_(self) -> str:
        return display_html(*self._display())

    def _display(self) -> tuple[str, list[tuple[str, str]]]:
        settings = self.settings
        kind = "service network" if self.supply is None else "supply chain"
        rows = [
            ("Run", f"{window_line(settings)}, ends at {settings.end_time:g}"),
            ("Seed", str(settings.seed)),
            (
                "",
                "model.experiment(...) spawns each replication's seed from its seed "
                "argument, so replication 1's seed differs from a single run's seed.",
            ),
        ]
        if self.supply is None:
            first = self.summary.row(0, named=True)
            rows.append(
                (
                    "Summary",
                    ", ".join(
                        f"{name} {figure(first[name])}"
                        for name in [
                            "arrivals",
                            "completed",
                            "mean_total_wait",
                            "mean_flow_time",
                        ]
                    ),
                )
            )
        shown = 6
        stock_rows = list(self.stock_points.iter_rows(named=True))
        for label, frame, name, columns in [
            ("Stock points", stock_rows, "stock_point", ["fill_rate", "mean_on_hand"]),
            (
                "Stations",
                list(self.stations.iter_rows(named=True)),
                "station",
                ["mean_wait", "utilization"],
            ),
        ]:
            for index, row in enumerate(frame[:shown]):
                detail = ", ".join(
                    f"{column} {figure(row[column])}" for column in columns
                )
                rows.append((label if index == 0 else "", f"{row[name]}: {detail}"))
            if len(frame) > shown:
                rows.append(("", f"... and {len(frame) - shown} more"))
        if not self.records_kept:
            kept = [name for name in self._table_names() if name not in RAW_TABLES]
            rows += [
                ("Tables", ", ".join(kept)),
                ("", "Raw records were dropped by keep_records=False."),
                ("Next", "results.report(), flowsim.explain(name)"),
            ]
            return f"Results: {kind}, {settings.duration:g} {settings.time_unit}", rows
        rows.append(("Tables", ", ".join(self._table_names())))
        plots = (
            "results.plot_queue(), results.plot_wait(), results.plot_utilization()"
            if self.supply is None
            else "results.plot_inventory(), results.plot_orders()"
        )
        rows += [
            ("Next", f"results.report(), {plots}"),
            ("", "results.viewer(), results.export(folder), flowsim.explain(name)"),
        ]
        return f"Results: {kind}, {settings.duration:g} {settings.time_unit}", rows

    def _table_names(self) -> list[str]:
        names = [
            "customers",
            "visits",
            "states",
            "stations",
            "summary",
            "activities",
            "failures",
        ]
        if self.supply is not None:
            names += ["inventory", "shipments", "stock_points"]
        return names

    def __init__(
        self,
        customers: list[_Customer],
        visits: list[_Visit],
        states: list[_State],
        activities: list[_Activity],
        failures: list[_Failure],
        inventory: list[_Inventory],
        shipments: list[_Shipment],
        supply: SupplySettings | None,
        definitions: dict,
        resources_vary: bool,
        capacities: dict[str, int],
        duration: float,
        end_time: float,
        drain: bool,
        closing_states: dict[str, _State],
        warmup: float,
        time_unit: str,
        seed: int,
        graph: nx.DiGraph,
        definition: tuple,
    ):
        self.customers = _frame(customers, CUSTOMER_SCHEMA)
        self.visits = _frame(visits, VISIT_SCHEMA)
        self.states = _frame(states, STATE_SCHEMA)
        self.activities = _frame(activities, ACTIVITY_SCHEMA)
        self.failures = _frame(failures, FAILURE_SCHEMA)
        self.inventory = _frame(inventory, INVENTORY_SCHEMA)
        self.shipments = _frame(shipments, SHIPMENT_SCHEMA)
        self.settings = Settings(
            duration=duration,
            warmup=warmup,
            drain=drain,
            seed=seed,
            time_unit=time_unit,
            end_time=end_time,
            capacities=MappingProxyType(dict(capacities)),
        )
        self.supply = supply
        self._definitions = definitions
        # Whether a station has a staffing schedule, a finite buffer, or
        # failures, which decides whether a run view draws server states.
        self._resources_vary = resources_vary
        self._closing_states = closing_states
        self._graph = graph.copy()
        # Experiment pools only runs whose model definitions match.
        self._definition = definition
        self.stations = self._station_metrics()
        self.summary = self._system_metrics()
        self.stock_points = self._stock_point_metrics()

    def _station_areas(self) -> pl.DataFrame:
        """Integrate each station's state counts over [warmup, duration).

        A state row holds until the station's next row, so its width is the
        part of that span inside the window. Areas have units customer*time
        and server*time. The largest queue counts only rows with positive width.
        """
        width = (
            (
                pl.col("time")
                .shift(-1)
                .over("station")
                .clip(upper_bound=self.settings.duration)
                - pl.col("time").clip(lower_bound=self.settings.warmup)
            )
            .clip(lower_bound=0)
            .fill_null(0)
        )
        return (
            self.states.with_columns(width.alias("width"))
            .group_by(
                "station",
                maintain_order=True,
            )
            .agg(
                *[
                    (pl.col(field) * pl.col("width")).sum().alias(f"{field}_area")
                    for field in AREA_FIELDS
                ],
                pl.col("waiting")
                .filter(pl.col("width") > 0)
                .max()
                .fill_null(0)
                .alias("max_queue"),
            )
        )

    def _station_counts(self) -> pl.DataFrame:
        """Count each station's visits in the window and its closing states."""

        def in_window(column: str) -> pl.Expr:
            return pl.col(column).is_between(
                lower_bound=self.settings.warmup,
                upper_bound=self.settings.duration,
                closed="left",
            )

        # Waits belong to station arrivals at or after warm-up, with no upper
        # bound, so drained runs include arrivals after closing.
        cohort = pl.col("arrival_time") >= self.settings.warmup
        waits = pl.col("wait_time").filter(cohort)
        visits = self.visits.group_by("station").agg(
            in_window("arrival_time").sum().alias("arrivals"),
            cohort.sum().alias("wait_arrivals"),
            in_window("service_start").sum().alias("service_starts"),
            in_window("service_end").sum().alias("service_completions"),
            waits.count().alias("wait_observations"),
            waits.mean().alias("mean_wait"),
            waits.quantile(
                quantile=0.95,
                interpolation="linear",
            ).alias("p95_wait"),
        )
        closing = _frame(list(self._closing_states.values()), STATE_SCHEMA).select(
            "station",
            pl.col("waiting").alias("waiting_at_close"),
            pl.col("busy").alias("occupied_at_close"),
            pl.col("blocked").alias("blocked_at_close"),
            pl.col("paused").alias("paused_at_close"),
            pl.col("outside").alias("outside_at_close"),
        )
        final = self.states.group_by(
            "station",
            maintain_order=True,
        ).agg(
            pl.col("waiting").last().alias("waiting_at_end"),
            pl.col("busy").last().alias("occupied_at_end"),
        )
        counts = [
            "arrivals",
            "wait_arrivals",
            "service_starts",
            "service_completions",
            "wait_observations",
        ]
        # A station no customer visited has zero counts and no waits.
        return (
            closing.join(
                other=visits,
                on="station",
                how="left",
                maintain_order="left",
            )
            .join(
                other=final,
                on="station",
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.col(counts).fill_null(0))
        )

    def _station_metrics(self) -> pl.DataFrame:
        window = self.settings.duration - self.settings.warmup
        # Utilization is server*time / (servers*time), dimensionless.
        installed = pl.col("capacity") * window
        # Only a comparison's shared settings hold no capacities.
        capacities = self.settings.capacities
        assert capacities is not None
        stations = (
            pl.DataFrame(
                data={
                    "station": list(capacities),
                    "capacity": list(capacities.values()),
                },
                schema={"station": pl.String, "capacity": pl.Int64},
            )
            .join(
                other=self._station_counts(),
                on="station",
                how="left",
                maintain_order="left",
            )
            .join(
                other=self._station_areas(),
                on="station",
                how="left",
                maintain_order="left",
            )
        )
        return stations.with_columns(
            (pl.col("waiting_area") / window).alias("mean_queue"),
            (pl.col("busy_area") / installed).alias("utilization"),
            (pl.col("processing_area") / installed).alias("processing_utilization"),
            (pl.col("blocked_area") / installed).alias("blocking_fraction"),
            (pl.col("down_area") / installed).alias("downtime_fraction"),
            (pl.col("scheduled_area") / window).alias("mean_scheduled_servers"),
            (pl.col("available_area") / window).alias("mean_available_servers"),
            (pl.col("outside_area") / window).alias("mean_outside_queue"),
            pl.when(pl.col("available_area") == 0)
            .then(None)
            .otherwise(pl.col("processing_area") / pl.col("available_area"))
            .alias("staffed_utilization"),
        ).select(
            [pl.col(column).cast(dtype) for column, dtype in STATION_SCHEMA.items()]
        )

    def _system_metrics(self) -> pl.DataFrame:
        # One query over the customers table gives every count.
        warmup = self.settings.warmup
        duration = self.settings.duration
        status = pl.col("status")
        eligible = status != Status.REJECTED
        cohort = eligible & (pl.col("arrival_time") >= warmup)
        finished = cohort & (status == Status.COMPLETED)
        inside = eligible & pl.col("admission_time").is_not_null()
        abandon = pl.col("abandon_time")
        admission = pl.col("admission_time")
        exit_time = pl.col("exit_time")
        # Each customer's presence clipped to the observation window.
        # Customer*time / time gives the time-average work in process.
        presence = (
            pl.coalesce("exit_time", "abandon_time")
            .fill_null(duration)
            .clip(upper_bound=duration)
            - admission.clip(lower_bound=warmup)
        ).clip(lower_bound=0)
        counts = self.customers.select(
            pl.len().alias("arrivals"),
            pl.col("admitted").sum().alias("admitted"),
            (inside & (admission < duration)).sum().alias("admitted_at_close"),
            (
                eligible
                & (admission.is_null() | (admission >= duration))
                & (abandon.is_null() | (abandon >= duration))
            )
            .sum()
            .alias("outside_at_close"),
            eligible.sum().alias("eligible"),
            exit_time.filter(eligible).count().alias("completed"),
            # Abandoned customers leave without service, so they are neither
            # completed nor remaining, and their waits stay out of the averages.
            abandon.filter(eligible).count().alias("abandoned"),
            (eligible & (abandon < duration)).sum().alias("abandoned_at_close"),
            abandon.filter(cohort).count().alias("cohort_abandoned"),
            (pl.col("arrival_time") >= warmup).sum().alias("measurement_arrivals"),
            ((exit_time >= warmup) & (exit_time < duration)).sum().alias("exits"),
            (exit_time < duration).sum().alias("completed_at_close"),
            # A rejection happens at arrival, so both counts use the event's
            # own time in [warmup, duration), as measurement_completions does.
            ((status == Status.REJECTED) & (pl.col("arrival_time") >= warmup))
            .sum()
            .alias("measurement_rejected"),
            # Abandonments count by their own time, like completions. The
            # rate instead follows the customers who arrived after warm-up.
            (eligible & (abandon >= warmup) & (abandon < duration))
            .sum()
            .alias("measurement_abandoned"),
            (inside & (admission >= warmup) & (admission < duration))
            .sum()
            .alias("measurement_admitted"),
            cohort.sum().alias("cohort"),
            finished.sum().alias("finished"),
        ).row(0, named=True)
        # Float sums and means stay on filtered frames, as before, since a
        # filtered expression can sum in a different order (last-bit changes).
        done = self.customers.filter(finished)
        for column in ["flow_time", "total_wait", "total_blocked", "outside_wait"]:
            counts[f"mean_{column}"] = done.get_column(column).mean()
        counts["presence"] = self.customers.filter(inside).select(presence.sum()).item()
        window = duration - warmup
        rejected = counts["arrivals"] - counts["eligible"]
        # A run's settings always record its end time.
        end_time = self.settings.end_time
        assert end_time is not None
        summary = pl.DataFrame(
            {
                "arrivals": [counts["arrivals"]],
                "admitted": [counts["admitted"]],
                "admitted_at_close": [counts["admitted_at_close"]],
                "outside_at_close": [counts["outside_at_close"]],
                "rejected": [rejected],
                "abandoned": [counts["abandoned"]],
                "completed": [counts["completed"]],
                "remaining": [
                    counts["eligible"] - counts["completed"] - counts["abandoned"]
                ],
                "completed_at_close": [counts["completed_at_close"]],
                "remaining_at_close": [
                    counts["eligible"]
                    - counts["completed_at_close"]
                    - counts["abandoned_at_close"]
                ],
                "after_close_completions": [
                    counts["completed"] - counts["completed_at_close"]
                ],
                "end_time": [end_time],
                "overtime": [end_time - duration],
                "measurement_arrivals": [counts["measurement_arrivals"]],
                "measurement_rejected": [counts["measurement_rejected"]],
                "measurement_abandoned": [counts["measurement_abandoned"]],
                "abandonment_rate": [
                    counts["cohort_abandoned"] / counts["measurement_arrivals"]
                    if counts["measurement_arrivals"]
                    else None
                ],
                "measurement_admitted": [counts["measurement_admitted"]],
                "cohort_arrivals": [counts["cohort"]],
                "measurement_completions": [counts["exits"]],
                "cohort_completions": [counts["finished"]],
                "cohort_remaining": [
                    counts["cohort"] - counts["finished"] - counts["cohort_abandoned"]
                ],
                "mean_flow_time": [counts["mean_flow_time"]],
                "mean_total_wait": [counts["mean_total_wait"]],
                "mean_total_blocked": [counts["mean_total_blocked"]],
                "mean_outside_wait": [counts["mean_outside_wait"]],
                "throughput": [counts["exits"] / window],
                "mean_wip": [counts["presence"] / window],
            },
            schema_overrides={
                "abandonment_rate": pl.Float64,
                "mean_flow_time": pl.Float64,
                "mean_total_wait": pl.Float64,
                "mean_total_blocked": pl.Float64,
                "mean_outside_wait": pl.Float64,
            },
        )
        return summary.select(headline_first(summary.columns, SUMMARY_HEADLINE))

    def _stock_point_metrics(self) -> pl.DataFrame:
        """Summarize each stock point over the review periods in [warmup, duration).

        Means of on-hand stock and backorders average end-of-period values. The
        fill rate is the share of each period's new demand filled from stock in
        that period. The bullwhip ratio divides the variance of orders placed by
        the variance of orders received, both with the n - 1 divisor, and is
        missing with fewer than two periods or constant demand.
        """
        names = self.stock_point_names
        if not names:
            return pl.DataFrame(schema=STOCK_POINT_SCHEMA)
        window = self.inventory.filter(pl.col("time") >= self.settings.warmup)
        measured = window.group_by("stock_point").agg(
            pl.len().alias("periods"),
            pl.col("demand").mean().alias("mean_demand"),
            pl.col("order_quantity").mean().alias("mean_order"),
            pl.col("demand").var(ddof=1).alias("demand_variance"),
            pl.col("order_quantity").var(ddof=1).alias("order_variance"),
            pl.col("demand").sum().alias("demanded"),
            pl.col("filled").sum().alias("filled"),
            pl.col("on_hand").mean().alias("mean_on_hand"),
            pl.col("backorders").mean().alias("mean_backorders"),
            pl.col("lost").sum().alias("lost_units"),
        )
        return (
            pl.DataFrame(
                data={"stock_point": names},
                schema={"stock_point": pl.String},
            )
            .join(
                other=measured,
                on="stock_point",
                how="left",
                maintain_order="left",
            )
            .with_columns(
                pl.when(pl.col("demand_variance") > 0)
                .then(pl.col("order_variance") / pl.col("demand_variance"))
                .alias("bullwhip_ratio"),
                pl.when(pl.col("demanded") > 0)
                .then(pl.col("filled") / pl.col("demanded"))
                .alias("fill_rate"),
                pl.col("periods").fill_null(0),
            )
            .select(
                [
                    pl.col(column).cast(dtype)
                    for column, dtype in STOCK_POINT_SCHEMA.items()
                ]
            )
        )

    def report(self) -> pl.DataFrame:
        """List every replication metric that applies to this run, one per row.

        The columns are scope, location (the station or stock point, null for a
        system metric), metric, label, value, and units in the run's time unit.
        Rows follow the catalog: system metrics, then each station's metrics,
        then each stock point's. A scope that the model lacks has no rows, so a
        service network lists no stock-point metrics, and a supply chain without
        production stations lists only its stock-point metrics and the run's
        end_time and overtime.

        Returns
        -------
        polars.DataFrame
            One row per metric and location, with the columns above.

        Example
        -------
        >>> results.report().filter(pl.col("scope") == "system")
        """
        capacities = self.settings.capacities
        assert capacities is not None
        stations = list(capacities)
        stock_points = self.stock_points.get_column("stock_point").to_list()
        rows = []
        for names, column, table, places in [
            (SYSTEM_METRICS, None, self.summary, [None]),
            (STATION_METRICS, "station", self.stations, stations),
            (STOCK_POINT_METRICS, "stock_point", self.stock_points, stock_points),
        ]:
            names = [
                name
                for name in names
                if inapplicable(
                    name,
                    stations=bool(stations),
                    stock_points=bool(stock_points),
                    supply=self.supply is not None,
                )
                is None
            ]
            if not names:
                continue
            for place in places:
                values = (
                    table if place is None else table.filter(pl.col(column) == place)
                ).row(
                    0,
                    named=True,
                )
                rows += [
                    {
                        "scope": METRICS[name].scope,
                        "location": place,
                        "metric": name,
                        "label": METRICS[name].label,
                        "value": None if values[name] is None else float(values[name]),
                        "units": metric_units(name, self.settings.time_unit),
                    }
                    for name in names
                ]
        return pl.DataFrame(
            data=rows,
            schema={
                "scope": pl.String,
                "location": pl.String,
                "metric": pl.String,
                "label": pl.String,
                "value": pl.Float64,
                "units": pl.String,
            },
        )

    def plot_queue(self, station: str | None = None) -> Figure:
        """Plot waiting customers and occupied servers over time.

        Without a station, a menu chooses among the stations. A named station
        is the only one drawn.

        Parameters
        ----------
        station : str, optional
            Name of one station. Without it, all stations are drawn.

        Returns
        -------
        plotly.graph_objects.Figure
            The queue-length figure.

        Example
        -------
        >>> results.plot_queue("Desk").show()
        """
        self._require_records("plot_queue()")
        require_stations(self)

        return plotly_figure(queue_figure(self, station=station))

    def plot_wait(self, station: str | None = None) -> Figure:
        """Plot the waiting-time histogram of station arrivals after warm-up.

        Parameters
        ----------
        station : str, optional
            Name of one station. Without it, all stations are drawn.

        Returns
        -------
        plotly.graph_objects.Figure
            The waiting-time histogram.

        Example
        -------
        >>> results.plot_wait("Desk").show()
        """
        self._require_records("plot_wait()")
        require_stations(self)

        return plotly_figure(wait_figure(self, station=station))

    def plot_utilization(self, station: str | None = None) -> Figure:
        """Plot each station's occupied share of capacity, or one station's.

        Parameters
        ----------
        station : str, optional
            Name of one station. Without it, every station is drawn.

        Returns
        -------
        plotly.graph_objects.Figure
            The utilization figure.

        Example
        -------
        >>> results.plot_utilization().show()
        """
        self._require_records("plot_utilization()")
        require_stations(self)

        return plotly_figure(utilization_figure(self, station=station))

    def plot_resources(
        self,
        station: str | None = None,
        *,
        stacked: bool = False,
    ) -> Figure:
        """Plot processing, blocked, paused, down, and staffed servers over time.

        The lines are the states table's counts, which overlap: a down server
        holding a paused job counts as both paused and down. With stacked=True,
        each server counts in one layer (processing, blocked, or paused while it
        holds a job, down while it is under repair without one, and idle while
        it is on duty, up, and free), and the layers reach the dashed scheduled
        line except while an off-duty server still holds a job or is under
        repair.

        Parameters
        ----------
        station : str, optional
            Name of one station. Without it, all stations are drawn.
        stacked : bool, default False
            Whether to stack each server's state instead of drawing the
            overlapping counts as lines.

        Returns
        -------
        plotly.graph_objects.Figure
            The server-state figure.

        Example
        -------
        >>> results.plot_resources("Desk", stacked=True).show()
        """
        self._require_records("plot_resources()")
        require_stations(self)

        return plotly_figure(
            resources_figure(
                self,
                station=station,
                stacked=stacked,
            )
        )

    def plot_customer(self, customer_id: int) -> Figure:
        """Plot one customer's waiting, service, paused, and blocked time.

        Parameters
        ----------
        customer_id : int
            The customer_id of a row in the customers table, starting at 1.

        Returns
        -------
        plotly.graph_objects.Figure
            The customer's timeline.

        Example
        -------
        >>> results.plot_customer(1).show()
        """
        self._require_records("plot_customer()")
        return plot_customer(self, customer_id)

    def plot_inventory(self, stock_point: str | None = None) -> Figure:
        """Plot stock on hand, backorders, and orders at each review.

        Without a stock point, a menu chooses among the stock points. A named
        stock point is the only one drawn.

        Parameters
        ----------
        stock_point : str, optional
            Name of one stock point. Without it, all stock points are drawn.

        Returns
        -------
        plotly.graph_objects.Figure
            The inventory figure. It applies to a supply chain.

        Example
        -------
        >>> results.plot_inventory("Store").show()
        """
        self._require_records("plot_inventory()")
        return plotly_figure(inventory_figure(self, stock_point=stock_point))

    def plot_orders(self) -> Figure:
        """Plot consumer demand and every stage's orders, which shows the bullwhip.

        Returns
        -------
        plotly.graph_objects.Figure
            The orders figure. It applies to a supply chain.

        Example
        -------
        >>> results.plot_orders().show()
        """
        self._require_records("plot_orders()")
        return plotly_figure(orders_figure(self))

    def viewer(self, path: str | Path | None = None) -> Viewer:
        """Build the offline results page, saving it when a path is given.

        A notebook displays the returned Viewer. A script calls its save(path)
        or open() method.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            File to save the page to. Without it, nothing is written.

        Returns
        -------
        Viewer
            The page.

        Example
        -------
        >>> results.viewer("output/run.html")
        """
        self._require_records("viewer()")
        supply = self.supply is not None
        page = Viewer(
            document=run_page_html(self),
            kind="supply-chain results page" if supply else "results page",
            name="supply-chain" if supply else "results",
            height=900,
        )
        if path is not None:
            page.save(path)
        return page

    def export(self, directory: str | Path) -> Path:
        """Export raw records, metrics, and their measurement definitions.

        Writes one CSV file per table (customers, visits, states, stations,
        summary, activities, and failures, plus inventory, shipments, and
        stock_points for a supply chain) and metadata.json with the settings,
        routes, and measurement definitions. Missing values stay empty. FlowSim
        export files from an earlier export that this one does not write are
        removed.

        Parameters
        ----------
        directory : str or pathlib.Path
            Folder to write to. It is created if needed.

        Returns
        -------
        pathlib.Path
            The folder.

        Example
        -------
        >>> results.export("output/my-run")
        """
        self._require_records("export()")
        capacities = self.settings.capacities
        assert capacities is not None
        metadata = {
            "time_unit": self.settings.time_unit,
            "duration": self.settings.duration,
            "end_time": self.settings.end_time,
            "drain": self.settings.drain,
            "warmup": self.settings.warmup,
            "seed": self.settings.seed,
            "capacities": dict(capacities),
            "definitions": self._definitions,
            # A typed edge lists each type that may take it, with its probability.
            "routes": [
                {"from": origin, "to": target, "probability": data["probability"]}
                if "probability" in data
                else {"from": origin, "to": target, "types": dict(data["types"])}
                for origin, target, data in self._graph.edges(data=True)
            ],
            "state_window": "[warmup, duration)",
            "wait_cohort": (
                "Visits admitted to the station's line at or after warmup with "
                "observed service starts, including visits admitted after closing "
                "when drained"
            ),
            "flow_cohort": "System arrivals at or after warmup with observed exits",
            "censoring": "Missing starts and exits remain null, never zero",
        }
        if self.supply is not None:
            metadata["supply"] = {
                field.name: _plain(getattr(self.supply, field.name))
                for field in fields(self.supply)
            }
            metadata["inventory_window"] = (
                "Review periods at times in [warmup, duration)"
            )
            metadata["shipment_arrivals"] = (
                "Null for shipments still in transit at the end"
            )
        return write_export(
            directory=directory,
            tables={f"{name}.csv": getattr(self, name) for name in self._table_names()},
            metadata=metadata,
        )


class Experiment:
    """Independent runs and Student t intervals across replication means.

    Model.experiment returns an Experiment. Each replication has its own seed,
    and the intervals treat each run's metric as one independent observation.

    Attributes
    ----------
    runs : list of Results
        The replications, in order.
    settings : Settings
        The settings the runs share, with no seed or end time.
    summary, stations, stock_points : polars.DataFrame
        Each run's tables stacked, with a replication column (starting at 1).
    stock_point_names : list of str
        The stock points in chain order, empty for a service network.

    The common next calls are intervals, plot, viewer, and export. Call
    flowsim.metrics() for the metric names.

    Example
    -------
    >>> experiment = model.experiment(duration=480, replications=30)
    >>> experiment.intervals("mean_total_wait")
    """

    runs: list[Results]

    def __init__(self, runs: list[Results]):
        runs = list(runs)
        if len(runs) < 2:
            raise ValueError("An experiment needs at least two runs.")
        if len({run.settings.seed for run in runs}) != len(runs):
            raise ValueError(
                "Replications need distinct seeds. A repeated run is not an "
                "independent observation."
            )
        # The model check below also covers station names and capacities.
        if any(_run_settings(run) != _run_settings(runs[0]) for run in runs):
            raise ValueError(
                "Replications need the same duration, warm-up, drain rule, "
                "and time unit."
            )
        if any(run._definition != runs[0]._definition for run in runs):
            raise ValueError(
                "Replications need runs of the same model. Separately built "
                "models count as different when their sampling functions are "
                "different objects, so run every replication from one Model."
            )
        self.runs = runs

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        hints = {}
        hints["differences"] = (
            "An experiment is one scenario. Use flowsim.compare({name: model, "
            "...}, duration) to compare scenarios with paired differences."
        )
        raise missing_attribute(self, name, hints)

    def __repr__(self) -> str:
        return display_text(*self._display())

    def _repr_html_(self) -> str:
        return display_html(*self._display())

    def _display(self) -> tuple[str, list[tuple[str, str]]]:
        seeds = [run.settings.seed for run in self.runs]
        listed = ", ".join(str(seed) for seed in seeds[:3])
        if len(seeds) > 3:
            listed += f", ... (first 3 of {len(seeds)}, spawned from the seed argument)"
        names = self.stock_point_names or self.runs[0].stations["station"].to_list()
        rows = [
            ("Settings", window_line(self.settings)),
            ("Seeds", listed),
            (
                "Stock points" if self.stock_point_names else "Stations",
                ", ".join(names),
            ),
            ("Next", "experiment.intervals(metric), experiment.plot()"),
            ("", "experiment.viewer(), experiment.export(folder), flowsim.metrics()"),
        ]
        if not self.runs[0].records_kept:
            rows.insert(
                3,
                ("Records", "Raw records were dropped by keep_records=False."),
            )
        kind = "service network" if self.runs[0].supply is None else "supply chain"
        return f"Experiment: {len(self.runs)} replications of a {kind}", rows

    # The pooled tables are built from the runs on each access, so estimates
    # and exports always read the runs' current values.
    @property
    def stations(self) -> pl.DataFrame:
        """Return every run's stations table, stacked, with a replication column."""
        return stack(
            parts=(
                (index, run.stations) for index, run in enumerate(self.runs, start=1)
            ),
            column="replication",
        )

    @property
    def stock_points(self) -> pl.DataFrame:
        """Return every run's stock_points table, stacked, with a replication column."""
        return stack(
            parts=(
                (index, run.stock_points)
                for index, run in enumerate(self.runs, start=1)
            ),
            column="replication",
        )

    @property
    def summary(self) -> pl.DataFrame:
        """Return every run's summary table, stacked, with a replication column."""
        return stack(
            parts=(
                (index, run.summary) for index, run in enumerate(self.runs, start=1)
            ),
            column="replication",
        )

    @property
    def settings(self) -> Settings:
        """Return the settings the runs share, with no seed or end time."""
        return replace(
            self.runs[0].settings,
            seed=None,
            end_time=None,
        )

    @property
    def stock_point_names(self) -> list[str]:
        """Return the stock points in chain order, empty for a service network."""
        return self.runs[0].stock_point_names

    def intervals(
        self,
        metric: str = "mean_total_wait",
        *,
        station: str | None = None,
        stock_point: str | None = None,
        confidence: float = 0.95,
    ) -> pl.DataFrame:
        """Estimate a metric across runs with Student t intervals.

        A system metric gives one row with a null station. A station metric
        gives one row per station unless station= names one, and a stock-point
        metric gives one row per stock point unless stock_point= names one, in
        a stock_point column that replaces the station column. Missing run
        values are excluded and counted in observed_runs.

        Parameters
        ----------
        metric : str, default 'mean_total_wait'
            A replication metric name from flowsim.metrics().
        station : str, optional
            Station to report, for a station metric. Without it, every station.
        stock_point : str, optional
            Stock point to report, for a stock-point metric. Without it, every
            stock point.
        confidence : float, default 0.95
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        polars.DataFrame
            One row per location with the mean, interval limits, and
            observed_runs.

        Example
        -------
        >>> experiment.intervals("utilization", station="Desk", confidence=0.9)
        """
        return experiment_intervals(
            experiment=self,
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
    ) -> Figure:
        """Plot the intervals for a metric, one point per location it covers.

        Parameters
        ----------
        metric : str, default 'mean_total_wait'
            A replication metric name from flowsim.metrics().
        station : str, optional
            Station to plot, for a station metric. Without it, every station.
        stock_point : str, optional
            Stock point to plot, for a stock-point metric. Without it, every
            stock point.
        confidence : float, default 0.95
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        plotly.graph_objects.Figure
            The interval figure.

        Example
        -------
        >>> experiment.plot("mean_total_wait").show()
        """
        return plotly_figure(
            interval_figure(
                table=self.intervals(
                    metric=metric,
                    station=station,
                    stock_point=stock_point,
                    confidence=confidence,
                ),
                metric=metric,
                time_unit=self.settings.time_unit,
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
    ) -> Viewer:
        """Build the offline experiment dashboard, saved to path if given.

        A menu chooses among every replication metric that applies to the
        model, and for a station or stock-point metric another menu chooses
        the location. The page shows the metric's interval and table and each
        replication's value. When the runs kept their records, selecting a
        replication opens that run's single-run view on the same page, and with
        keep_records=False the page shows the overview only. The arguments set
        the metric and location shown first.

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

        Returns
        -------
        Viewer
            The page.

        Example
        -------
        >>> experiment.viewer("output/experiment.html", metric="utilization")
        """
        # Check the request as intervals() does, before building the page.
        self.intervals(
            metric=metric,
            station=station,
            stock_point=stock_point,
            confidence=confidence,
        )
        viewer = Viewer(
            document=dashboard(
                experiments={"": self},
                comparing=False,
                metric=metric,
                station=station,
                stock_point=stock_point,
                confidence=confidence,
                reference=None,
            ),
            kind="experiment dashboard",
            name="experiment",
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
    ) -> Path:
        """Write the pooled run tables, one metric's intervals, and metadata.

        The folder receives runs.csv (the pooled summary), stations.csv,
        stock_points.csv for a supply chain, intervals.csv, and metadata.json
        with the settings, replication seeds, metric, and confidence level.

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

        Returns
        -------
        pathlib.Path
            The folder.

        Example
        -------
        >>> experiment.export("output/experiment", metric="utilization")
        """
        intervals = self.intervals(
            metric=metric,
            station=station,
            stock_point=stock_point,
            confidence=confidence,
        )
        return export_tables(
            directory,
            summary=self.summary,
            stations=self.stations,
            stock_points=self.stock_points if self.stock_point_names else None,
            intervals=intervals,
            differences=None,
            metadata=request_metadata(
                experiment=self,
                metric=metric,
                station=station,
                stock_point=stock_point,
                confidence=confidence,
            ),
        )


class PageExperiment(Experiment):
    """An experiment whose pooled tables are built once, for one page.

    Experiment rebuilds each pooled table on every access, so estimates always
    read the runs' current values. A dashboard reads one for every metric
    while it is built, and the page is a snapshot of the runs anyway.
    """

    @functools.cached_property
    def stations(self) -> pl.DataFrame:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return every run's stations table, stacked, built once."""
        return super().stations

    @functools.cached_property
    def stock_points(self) -> pl.DataFrame:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return every run's stock_points table, stacked, built once."""
        return super().stock_points

    @functools.cached_property
    def summary(self) -> pl.DataFrame:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return every run's summary table, stacked, built once."""
        return super().summary
