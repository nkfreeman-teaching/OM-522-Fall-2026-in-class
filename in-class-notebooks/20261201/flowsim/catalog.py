"""The replication metrics and table columns: what each means, its units, and its scope."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

import polars as pl

Scope = Literal["system", "station", "stock_point"]


@dataclass(frozen=True)
class ScopeFacts:
    """What a metric's scope sets wherever the metric is listed, requested, or shown.

    table is the per-run table that holds the scope's metrics, and label names
    the scope in messages. needs is the argument that picks one location, and
    example is that argument with a sample location, as explain prints it.
    menu is the dashboard menu heading. column is the interval table's location
    column, which a system metric keeps as a null station column, and heading
    is that column's axis title, which a system metric, with no location axis,
    leaves empty.
    """

    table: str
    label: str
    needs: str
    example: str
    menu: str
    column: str
    heading: str


# The three scopes in catalog order, which is the order of every listing.
SCOPES = {
    "system": ScopeFacts(
        table="summary",
        label="system",
        needs="",
        example="",
        menu="System metrics",
        column="station",
        heading="",
    ),
    "station": ScopeFacts(
        table="stations",
        label="station",
        needs="station=",
        example='station="Service"',
        menu="Station metrics",
        column="station",
        heading="Station",
    ),
    "stock_point": ScopeFacts(
        table="stock_points",
        label="stock-point",
        needs="stock_point=",
        example='stock_point="Retailer"',
        menu="Stock-point metrics",
        column="stock_point",
        heading="Stock point",
    ),
}


@dataclass(frozen=True)
class Metric:
    """A replication metric: where it comes from, how it reads, and what it means.

    The kind sets units and number formats: "time", "rate", "fraction",
    "servers", "customers", "units", "variance", or "ratio". The table is the
    per-run table that holds the metric as a column ("summary", "stations", or
    "stock_points"), and the window names the cohort or period it measures.
    """

    scope: str
    label: str
    kind: str
    meaning: str
    window: str

    @property
    def table(self) -> str:
        """Name the per-run table that holds the metric, which its scope sets."""
        return SCOPES[self.scope].table


_WARMUP_WINDOW = "[warmup, duration)"
_ARRIVAL_COHORT = "arrivals at or after warmup"
_FLOW_COHORT = "arrivals at or after warmup with observed exits"
_WAIT_COHORT = "visits admitted at or after warmup with observed starts"
_PERIODS = "review periods at or after warmup"

# One entry per replication metric, in catalog order. The table follows from
# the scope. Each meaning is one plain sentence.
METRICS = {
    # System metrics, in the summary table.
    "mean_total_wait": Metric(
        scope="system",
        label="Mean total waiting time",
        kind="time",
        window=_FLOW_COHORT,
        meaning="Mean total time that a completed customer waited before service, "
        "summed over its stations and including any wait outside the first station.",
    ),
    "mean_flow_time": Metric(
        scope="system",
        label="Mean flow time",
        kind="time",
        window=_FLOW_COHORT,
        meaning="Mean time from system arrival to exit for completed customers.",
    ),
    "throughput": Metric(
        scope="system",
        label="Throughput during operating hours",
        kind="rate",
        window=_WARMUP_WINDOW,
        meaning="Customers who exit during the operating window, whenever they arrived, "
        "divided by the window's length.",
    ),
    "remaining_at_close": Metric(
        scope="system",
        label="Customers present at closing",
        kind="customers",
        window="at closing",
        meaning="Customers who were not rejected and had neither exited nor abandoned "
        "before closing, including customers still waiting outside the first station.",
    ),
    "completed_at_close": Metric(
        scope="system",
        label="Customers completed before closing",
        kind="customers",
        window="before closing",
        meaning="Customers who exit before closing, whenever they arrived.",
    ),
    "remaining": Metric(
        scope="system",
        label="Customers unfinished at the end",
        kind="customers",
        window="at the end of the run",
        meaning="Customers who were not rejected and had neither completed nor abandoned "
        "when the run ended.",
    ),
    "overtime": Metric(
        scope="system",
        label="Time to finish after closing",
        kind="time",
        window="whole run",
        meaning="Time from closing to the end of the run, i.e., end_time minus duration, "
        "which is zero in a run that is not drained and in a drained run whose "
        "customers all finish by closing.",
    ),
    "mean_wip": Metric(
        scope="system",
        label="Time-average customers in the system",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Time-average number of admitted customers in the system, which excludes "
        "customers still waiting outside the first station.",
    ),
    "rejected": Metric(
        scope="system",
        label="Rejected arrivals",
        kind="customers",
        window="whole run",
        meaning="Arrivals that were rejected on arrival.",
    ),
    "admitted": Metric(
        scope="system",
        label="Admitted customers",
        kind="customers",
        window="whole run",
        meaning="Customers admitted to the first station, including admissions after closing. A customer routed from the source straight to the exit counts as admitted on arrival.",
    ),
    "measurement_rejected": Metric(
        scope="system",
        label="Rejected arrivals after warm-up",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Arrivals at or after warmup that were rejected on arrival.",
    ),
    "measurement_admitted": Metric(
        scope="system",
        label="Admissions after warm-up, before closing",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Customers admitted to the first station during the operating window.",
    ),
    "abandonment_rate": Metric(
        scope="system",
        label="Arrivals after warm-up who abandon",
        kind="fraction",
        window="arrivals at or after warmup, including rejected arrivals",
        meaning="Share of arrivals at or after warmup, counting rejected arrivals in the "
        "denominator, who abandoned at any time in the run, including after "
        "closing in a drained run.",
    ),
    "mean_total_blocked": Metric(
        scope="system",
        label="Mean time blocked between stations",
        kind="time",
        window=_FLOW_COHORT,
        meaning="Mean time that a completed customer was held at a station after service "
        "because the next station could not accept it.",
    ),
    "mean_outside_wait": Metric(
        scope="system",
        label="Mean waiting time outside",
        kind="time",
        window=_FLOW_COHORT,
        meaning="Mean time that a completed customer waited outside the first station "
        "before admission, which is zero when it was admitted on arrival.",
    ),
    "arrivals": Metric(
        scope="system",
        label="Arrivals",
        kind="customers",
        window="whole run",
        meaning="Customers who arrived during the run, including rejected arrivals.",
    ),
    "admitted_at_close": Metric(
        scope="system",
        label="Customers admitted before closing",
        kind="customers",
        window="before closing",
        meaning="Customers admitted to the first station before closing.",
    ),
    "outside_at_close": Metric(
        scope="system",
        label="Customers outside at closing",
        kind="customers",
        window="at closing",
        meaning="Customers who were not rejected and had neither been admitted nor "
        "abandoned before closing, so they were waiting outside the first station.",
    ),
    "abandoned": Metric(
        scope="system",
        label="Customers who abandoned",
        kind="customers",
        window="whole run",
        meaning="Customers who left without service, including abandonments after closing.",
    ),
    "completed": Metric(
        scope="system",
        label="Customers completed",
        kind="customers",
        window="at the end of the run",
        meaning="Customers who completed service, including completions after closing "
        "when the run continues past it.",
    ),
    "after_close_completions": Metric(
        scope="system",
        label="Completions after closing",
        kind="customers",
        window="at or after closing",
        meaning="Customers who exit at or after closing, during the continuation of a "
        "drained run.",
    ),
    "end_time": Metric(
        scope="system",
        label="End of service",
        kind="time",
        window="whole run",
        meaning="Simulation time at which the run stopped, which is the closing time in "
        "a run that is not drained and, in a drained run, the later of closing "
        "and the last exit or abandonment.",
    ),
    "measurement_arrivals": Metric(
        scope="system",
        label="Arrivals after warm-up",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Arrivals at or after warmup, including rejected arrivals, which form the "
        "denominator of the abandonment rate.",
    ),
    "measurement_abandoned": Metric(
        scope="system",
        label="Abandonments after warm-up, before closing",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Customers whose abandonment time falls in the operating window.",
    ),
    "cohort_arrivals": Metric(
        scope="system",
        label="Customers in the arrival cohort",
        kind="customers",
        window=_ARRIVAL_COHORT,
        meaning="Customers who arrived at or after warmup and were not rejected, whose "
        "completed members form the cohort behind the flow-time and total-wait "
        "averages.",
    ),
    "measurement_completions": Metric(
        scope="system",
        label="Completions after warm-up, before closing",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Customers who exit during the operating window, which is the numerator "
        "of throughput.",
    ),
    "cohort_completions": Metric(
        scope="system",
        label="Completed customers in the arrival cohort",
        kind="customers",
        window=_FLOW_COHORT,
        meaning="Members of the arrival cohort whose exits were observed.",
    ),
    "cohort_remaining": Metric(
        scope="system",
        label="Unfinished customers in the arrival cohort",
        kind="customers",
        window=_ARRIVAL_COHORT,
        meaning="Members of the arrival cohort who had neither completed nor abandoned "
        "when the run ended.",
    ),
    # Station metrics, in the stations table.
    "mean_wait": Metric(
        scope="station",
        label="Mean station waiting time",
        kind="time",
        window=_WAIT_COHORT,
        meaning="Mean time from a visit's admission to the station's line until its "
        "service started, which excludes time spent outside a full finite buffer "
        "or blocked upstream and excludes visits that abandoned or had not started.",
    ),
    "mean_queue": Metric(
        scope="station",
        label="Time-average customers waiting",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Time-average number of customers waiting in the station's line, "
        "excluding customers waiting outside a finite buffer.",
    ),
    "utilization": Metric(
        scope="station",
        label="Occupied / installed capacity",
        kind="fraction",
        window=_WARMUP_WINDOW,
        meaning="Occupied-server time as a share of installed capacity times the window "
        "length, where an occupied server holds a job that is processing, "
        "paused, or blocked.",
    ),
    "p95_wait": Metric(
        scope="station",
        label="95th percentile station waiting time",
        kind="time",
        window=_WAIT_COHORT,
        meaning="The 95th percentile of the run's observed visit waits, interpolated "
        "linearly between the two nearest order statistics, so an interval across "
        "replications estimates the expected run-level 95th percentile and not "
        "the 95th percentile of all customers' waits pooled.",
    ),
    "processing_utilization": Metric(
        scope="station",
        label="Processing / installed capacity",
        kind="fraction",
        window=_WARMUP_WINDOW,
        meaning="Server time spent processing as a share of installed capacity times "
        "the window length.",
    ),
    "blocking_fraction": Metric(
        scope="station",
        label="Blocked / installed capacity",
        kind="fraction",
        window=_WARMUP_WINDOW,
        meaning="Server time spent blocked as a share of installed capacity times the "
        "window length.",
    ),
    "downtime_fraction": Metric(
        scope="station",
        label="Down / installed capacity",
        kind="fraction",
        window=_WARMUP_WINDOW,
        meaning="Server time spent down as a share of installed capacity times the "
        "window length, which can overlap paused and blocked time and so must not "
        "be added to the other shares.",
    ),
    "staffed_utilization": Metric(
        scope="station",
        label="Processing / available capacity",
        kind="fraction",
        window=_WARMUP_WINDOW,
        meaning="Server time spent processing as a share of available-server time, "
        "which is missing when no server was available at any time in the window.",
    ),
    "mean_scheduled_servers": Metric(
        scope="station",
        label="Mean scheduled servers",
        kind="servers",
        window=_WARMUP_WINDOW,
        meaning="Time-average number of servers on duty under the staffing schedule, "
        "which equals the capacity at a station without a staffing schedule and "
        "includes servers that are down.",
    ),
    "mean_available_servers": Metric(
        scope="station",
        label="Mean available servers",
        kind="servers",
        window=_WARMUP_WINDOW,
        meaning="Time-average number of servers that are on duty and not down, plus "
        "off-duty servers that are not down and are finishing a job at a station "
        "whose shifts end with shift_end='finish'.",
    ),
    "mean_outside_queue": Metric(
        scope="station",
        label="Time-average customers waiting outside",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Time-average number of arriving customers waiting outside the station's "
        "full finite buffer for admission, which excludes upstream customers "
        "blocked from entering.",
    ),
    "max_queue": Metric(
        scope="station",
        label="Largest queue",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Largest number of customers waiting in the station's line, ignoring "
        "states that last zero time.",
    ),
    "service_starts": Metric(
        scope="station",
        label="Service starts",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Services that start at the station during the operating window.",
    ),
    "service_completions": Metric(
        scope="station",
        label="Service completions",
        kind="customers",
        window=_WARMUP_WINDOW,
        meaning="Services whose processing ends at the station during the operating "
        "window, before any blocked time.",
    ),
    "wait_arrivals": Metric(
        scope="station",
        label="Visits in the wait cohort",
        kind="customers",
        window="visits admitted at or after warmup",
        meaning="Visits admitted to the station's line at or after warmup, which form the "
        "cohort behind the wait averages and include visits after closing when a "
        "drained run continues.",
    ),
    "wait_observations": Metric(
        scope="station",
        label="Visits with an observed wait",
        kind="customers",
        window=_WAIT_COHORT,
        meaning="Visits in the wait cohort whose service start was observed.",
    ),
    "waiting_at_close": Metric(
        scope="station",
        label="Customers waiting at closing",
        kind="customers",
        window="at closing",
        meaning="Customers waiting in the station's line at closing, including customers "
        "who arrived before warmup.",
    ),
    "occupied_at_close": Metric(
        scope="station",
        label="Servers occupied at closing",
        kind="servers",
        window="at closing",
        meaning="Servers holding a job at closing, whether processing, paused, or "
        "blocked, so the count includes blocked_at_close and paused_at_close.",
    ),
    "waiting_at_end": Metric(
        scope="station",
        label="Customers waiting at the end",
        kind="customers",
        window="at the end of the run",
        meaning="Customers waiting in the station's line when the run ended.",
    ),
    "occupied_at_end": Metric(
        scope="station",
        label="Servers occupied at the end",
        kind="servers",
        window="at the end of the run",
        meaning="Servers holding a job when the run ended, whether processing, paused, "
        "or blocked.",
    ),
    "blocked_at_close": Metric(
        scope="station",
        label="Servers blocked at closing",
        kind="servers",
        window="at closing",
        meaning="Servers holding a finished job that the next station could not accept "
        "at closing.",
    ),
    "paused_at_close": Metric(
        scope="station",
        label="Servers paused at closing",
        kind="servers",
        window="at closing",
        meaning="Servers holding a job whose work was paused at closing because the "
        "server was down or off duty.",
    ),
    # Stock-point metrics, in the stock_points table.
    "bullwhip_ratio": Metric(
        scope="stock_point",
        label="Order variance / demand variance",
        kind="ratio",
        window=_PERIODS,
        meaning="Sample variance of the quantities the stock point ordered divided by "
        "the sample variance of its incoming orders, which is missing with fewer "
        "than two periods or constant incoming orders.",
    ),
    "fill_rate": Metric(
        scope="stock_point",
        label="Demand filled from stock in the same period",
        kind="fraction",
        window=_PERIODS,
        meaning="Total units of incoming orders filled from stock in the period they "
        "arrived divided by total incoming orders over the measured periods, so "
        "filled backorders do not count and the rate is missing when no orders "
        "arrived.",
    ),
    "mean_on_hand": Metric(
        scope="stock_point",
        label="Mean end-of-period stock on hand",
        kind="units",
        window=_PERIODS,
        meaning="Mean stock on hand after each review's receipts and shipments, a level "
        "that holds until the next review, over the review periods at or after "
        "warmup.",
    ),
    "mean_backorders": Metric(
        scope="stock_point",
        label="Mean end-of-period backorders",
        kind="units",
        window=_PERIODS,
        meaning="Mean backorders after each review's receipts and shipments, a level "
        "that holds until the next review, over the review periods at or after "
        "warmup.",
    ),
    "lost_units": Metric(
        scope="stock_point",
        label="Lost sales",
        kind="units",
        window=_PERIODS,
        meaning="Total units of incoming orders lost, which is zero unless the stock "
        "point's shortage rule is 'lost'.",
    ),
    "mean_demand": Metric(
        scope="stock_point",
        label="Mean incoming orders",
        kind="units",
        window=_PERIODS,
        meaning="Mean incoming orders per review period, which are consumer demand at "
        "the retailer and the downstream stage's order at every other stage.",
    ),
    "mean_order": Metric(
        scope="stock_point",
        label="Mean orders placed",
        kind="units",
        window=_PERIODS,
        meaning="Mean quantity that the stock point ordered per review period.",
    ),
    "demand_variance": Metric(
        scope="stock_point",
        label="Variance of incoming orders",
        kind="variance",
        window=_PERIODS,
        meaning="Sample variance of incoming orders per review period, with the n - 1 divisor.",
    ),
    "order_variance": Metric(
        scope="stock_point",
        label="Variance of orders placed",
        kind="variance",
        window=_PERIODS,
        meaning="Sample variance of the quantities ordered per review period, with the "
        "n - 1 divisor.",
    ),
}
SYSTEM_METRICS = [name for name, metric in METRICS.items() if metric.scope == "system"]
STATION_METRICS = [
    name for name, metric in METRICS.items() if metric.scope == "station"
]
STOCK_POINT_METRICS = [
    name for name, metric in METRICS.items() if metric.scope == "stock_point"
]


# Plural time units that a rate names in the singular, i.e., "per minute".
# Any other label, such as "ms", stays as the model gives it.
_SINGULAR_UNITS = {
    "seconds": "second",
    "minutes": "minute",
    "hours": "hour",
    "days": "day",
    "weeks": "week",
    "months": "month",
    "years": "year",
    "periods": "period",
    "shifts": "shift",
}


def metric_units(metric: str, time_unit: str) -> str:
    singular = _SINGULAR_UNITS.get(time_unit, time_unit)
    return {
        "time": time_unit,
        "rate": f"customers per {singular}",
        "fraction": "share",
        "servers": "servers",
        "customers": "count",
        "units": "units",
        "ratio": "variance ratio",
        "variance": "squared units",
    }[METRICS[metric].kind]


# Numeric columns of the stations, summary, and stock_points tables that are not
# replication metrics, and identifier columns, each with the reason. A metric
# name has one scope, so the two station columns that share a name with a
# system metric are listed here and read from the stations table.
EXCLUDED = {
    (
        "stations",
        "station",
    ): "It names the row, so it identifies a station and measures nothing.",
    ("stations", "capacity"): "It is a setting, not an outcome.",
    (
        "stations",
        "arrivals",
    ): "It shares its name with the system metric arrivals, which counts customers over the whole run, so this count of visits admitted to the station's line in [warmup, duration) is read from the stations table.",
    (
        "stations",
        "outside_at_close",
    ): "It shares its name with the system metric outside_at_close, which counts customers outside the first station at closing, so this count for one station is read from the stations table.",
    (
        "stock_points",
        "stock_point",
    ): "It names the row, so it identifies a stock point and measures nothing.",
    (
        "stock_points",
        "periods",
    ): "It is fixed by the run length, the warm-up, and the review period, so it does not vary across replications.",
}


@dataclass(frozen=True)
class Column:
    """What one table column holds and its units."""

    meaning: str
    units: str


# Each table's columns as column: (meaning, units).
_SCHEMA_COLUMNS = {
    "customers": {
        "customer_id": ("Identifier of the customer, unique within a run.", "id"),
        "source": (
            "Name of the source that created the customer, or of the stock point that released the production job.",
            "label",
        ),
        "customer_type": (
            "Customer type, empty for a production job or a customer from a source without types.",
            "label",
        ),
        "quantity": (
            "Units in a production job, empty for a customer from a source.",
            "units",
        ),
        "arrival_time": ("Time the customer arrived in the system.", "time unit"),
        "exit_time": (
            "Time the customer left the network, empty if it did not complete.",
            "time unit",
        ),
        "abandon_time": (
            "Time the customer abandoned before service, empty if it did not abandon.",
            "time unit",
        ),
        "flow_time": (
            "Time from arrival to exit, empty if the customer did not exit.",
            "time unit",
        ),
        "total_wait": (
            "Total of the customer's completed waits before service, including any wait outside the first station.",
            "time unit",
        ),
        "total_blocked": (
            "Total time the customer was held at a station after service because the next station could not accept it, counting only blocking that had ended.",
            "time unit",
        ),
        "admitted": (
            "Whether the customer was admitted to the first station. A customer routed from the source straight to the exit counts as admitted on arrival.",
            "true/false",
        ),
        "admission_time": (
            "Time the customer was admitted to the first station, empty if never admitted.",
            "time unit",
        ),
        "outside_wait": (
            "Time from arrival to admission, which is zero for a customer admitted on arrival and empty for a customer never admitted.",
            "time unit",
        ),
        "status": (
            "The customer's status at the end of the run: outside, waiting, in_service, blocked, paused, rejected, abandoned, or completed.",
            "label",
        ),
        "station": (
            "Station where the customer was at the end of the run, or where it was rejected or abandoned, empty for a completed customer.",
            "label",
        ),
    },
    "visits": {
        "customer_id": ("Identifier of the visiting customer.", "id"),
        "source": (
            "Name of the source that created the customer, or of the stock point that released the production job.",
            "label",
        ),
        "customer_type": (
            "Customer type, empty for a production job or a customer from a source without types.",
            "label",
        ),
        "station": ("Station that was visited.", "label"),
        "arrival_time": (
            "Time the customer entered the station's line, which follows its system arrival when it waited outside a full finite buffer and follows its upstream service end when it was blocked upstream.",
            "time unit",
        ),
        "service_start": (
            "Time service started, empty if it never started.",
            "time unit",
        ),
        "service_end": (
            "Time processing finished, so service_end minus service_start includes any paused time, empty if processing had not finished.",
            "time unit",
        ),
        "departure_time": (
            "Time the customer left the station, which can be later than service_end because of downstream blocking, empty if the customer had not left or abandoned the line.",
            "time unit",
        ),
        "abandon_time": (
            "Time the customer abandoned this station's line, empty if it did not.",
            "time unit",
        ),
        "server_id": (
            "Server that served the visit, empty if service never started.",
            "id",
        ),
        "blocked_time": (
            "Time the customer was held after service until departure, empty if not observed.",
            "time unit",
        ),
        "wait_time": (
            "Time from arrival at the station to the start of service, empty if service never started.",
            "time unit",
        ),
        "service_time": (
            "Sampled service requirement, i.e., the per-unit draw times the job's quantity for a production job, which is not elapsed busy time and can extend beyond the run limit.",
            "time unit",
        ),
    },
    "states": {
        "station": ("Station the row describes.", "label"),
        "time": (
            "Time of the row, i.e., the start of the run, a time when the station's state changed, closing, or the end of the run, and the row holds the state after every event at that time.",
            "time unit",
        ),
        "waiting": (
            "Customers waiting in the station's line after the change.",
            "count",
        ),
        "busy": (
            "Servers holding a job that is processing, paused, or blocked.",
            "servers",
        ),
        "processing": ("Servers actively processing a job.", "servers"),
        "blocked": (
            "Servers holding a finished job that the next station cannot accept.",
            "servers",
        ),
        "paused": (
            "Servers holding a job whose work is paused because the server was down or off duty.",
            "servers",
        ),
        "down": ("Servers that are down for repair.", "servers"),
        "scheduled": (
            "Servers on duty under the staffing schedule, which equals the capacity at a station without a staffing schedule and includes servers that are down.",
            "servers",
        ),
        "available": (
            "Servers that are on duty and not down, plus off-duty servers that are not down and are finishing a job when shift_end is 'finish'.",
            "servers",
        ),
        "outside": (
            "Arriving customers waiting outside the station's full finite buffer for admission, which excludes upstream customers blocked from entering.",
            "count",
        ),
    },
    "activities": {
        "customer_id": ("Identifier of the customer served.", "id"),
        "station": ("Station where the interval occurred.", "label"),
        "server_id": ("Server that held the job during the interval.", "id"),
        "state": (
            "Server phase during the interval: processing, blocked, or paused.",
            "label",
        ),
        "start": ("Start of the interval.", "time unit"),
        "end": (
            "End of the interval, empty if it was still open at the end.",
            "time unit",
        ),
    },
    "failures": {
        "station": ("Station of the failed server.", "label"),
        "server_id": ("Server that failed.", "id"),
        "failure_time": ("Time the server failed.", "time unit"),
        "time_to_failure": (
            "Sampled lifetime, counted only while the failure clock runs, so it can be shorter than the time elapsed since the previous repair ended.",
            "time unit",
        ),
        "repair_time": ("Sampled repair requirement.", "time unit"),
        "repair_end": (
            "Time the repair finished, empty if it was not observed.",
            "time unit",
        ),
    },
    "inventory": {
        "stock_point": ("Stock point the row describes.", "label"),
        "period": ("Review period number, counted from zero.", "index"),
        "time": ("Time of the review.", "time unit"),
        "received": (
            "Units added to stock at the review, from shipments due by this period and, at a producing stage, jobs finished since the previous review.",
            "units",
        ),
        "demand": ("Incoming orders in the period.", "units"),
        "filled": (
            "Units of the period's incoming orders filled from stock in the same period.",
            "units",
        ),
        "shipped": (
            "Units shipped downstream in the period, including filled backorders.",
            "units",
        ),
        "lost": ("Units of incoming orders lost in the period.", "units"),
        "on_hand": (
            "Stock on hand at the end of the period, after the review's order.",
            "units",
        ),
        "backorders": ("Backorders at the end of the period.", "units"),
        "on_order": (
            "Units ordered but not yet received, recorded after this stage's order, so an order that the stage above loses later in the same review still counts.",
            "units",
        ),
        "position": (
            "Inventory position after this stage's order, i.e., on hand plus on order minus backorders, with on order as recorded in that column.",
            "units",
        ),
        "forecast": (
            "One-period forecast used for the order, i.e., the moving average of incoming orders, or of consumer demand when forecast_from is 'end_demand', including this period.",
            "units",
        ),
        "target": ("Order-up-to target used for the period's order.", "units"),
        "order_quantity": ("Quantity ordered at the period's review.", "units"),
    },
    "shipments": {
        "sender": (
            "Stage that shipped, or Supplier for orders of a top stage that does not produce.",
            "label",
        ),
        "receiver": ("Stage that receives the shipment.", "label"),
        "quantity": ("Units shipped.", "units"),
        "ship_time": ("Time the shipment was sent.", "time unit"),
        "lead_time": (
            "Lead time of the shipment, constant or sampled, which can be shorter than arrival_time minus ship_time because a shipment arrives only at a review and never before an earlier shipment to the same stage.",
            "time unit",
        ),
        "arrival_time": (
            "Time the shipment arrived, empty if it was still in transit at the end.",
            "time unit",
        ),
    },
    "stations": {
        "station": ("Station the row describes.", "label"),
        "capacity": ("Number of installed servers at the station.", "servers"),
        "arrivals": (
            "Visits admitted to the station's line in [warmup, duration), which excludes rejected arrivals and customers still waiting outside.",
            "count",
        ),
        "outside_at_close": (
            "Arriving customers waiting outside the station's full finite buffer for admission at closing, which excludes upstream customers blocked from entering.",
            "count",
        ),
    },
    "stock_points": {
        "stock_point": ("Stock point the row describes.", "label"),
        "periods": ("Review periods at or after warmup.", "count"),
    },
    "intervals": {
        "scenario": ("Scenario that the interval describes.", "label"),
        "baseline": (
            "Baseline scenario that the difference is measured against, in the differences table only.",
            "label",
        ),
        "station": (
            "Station the metric was measured at, empty for a system metric and absent from the tables of stock-point metrics.",
            "label",
        ),
        "stock_point": (
            "Stock point the metric was measured at, a column that only the tables of stock-point metrics have, in place of station.",
            "label",
        ),
        "metric": ("Name of the replication metric.", "label"),
        "runs": ("Replications run.", "count"),
        "observed_runs": (
            "Replications with a finite value, or, in the differences table, matched pairs whose difference is finite, which are the ones the interval uses.",
            "count",
        ),
        "mean": (
            "Mean of the run-level values, or of the paired differences in the differences table.",
            "metric units",
        ),
        "lower": (
            "Lower limit of the Student t confidence interval, empty when fewer than two runs are observed.",
            "metric units",
        ),
        "upper": (
            "Upper limit of the Student t confidence interval, empty when fewer than two runs are observed.",
            "metric units",
        ),
        "confidence": ("Confidence level of the interval.", "share"),
    },
    "pooled": {
        "replication": (
            "Replication number within an experiment, counted from one.",
            "index",
        ),
        "scenario": ("Scenario name in a comparison's pooled tables.", "label"),
    },
    "fits": {
        "rank": ("Position by AIC, where one is the best fit.", "count"),
        "family": ("Family name.", "label"),
        "description": (
            "Plain-language description of the fitted distribution with its parameters, including any shift and a warning when draws can be negative.",
            "label",
        ),
        "mean": ("Mean of the fitted distribution.", "data unit"),
        "sd": ("Standard deviation of the fitted distribution.", "data unit"),
        "p_negative": (
            "Chance that a draw from the fitted distribution is below zero.",
            "share",
        ),
        "aic": (
            "Akaike information criterion, 2k - 2 log-likelihood for k fitted parameters, where lower is better.",
            "dimensionless",
        ),
        "ks": (
            "Kolmogorov-Smirnov statistic, the largest vertical gap between the data's empirical CDF and the fitted CDF, empty for count fits.",
            "share",
        ),
        "shape": (
            "Shape parameter of a gamma or Weibull fit before any shift.",
            "dimensionless",
        ),
        "scale": (
            "Scale parameter of a gamma or Weibull fit before any shift.",
            "data unit",
        ),
        "low": (
            "Lower end of a uniform, triangular, or beta fit before any shift, so the fitted lower end is low plus by when by is present.",
            "data unit",
        ),
        "mode": (
            "Mode of a triangular fit before any shift, so the fitted mode is mode plus by when by is present.",
            "data unit",
        ),
        "high": (
            "Upper end of a uniform, triangular, or beta fit before any shift, so the fitted upper end is high plus by when by is present.",
            "data unit",
        ),
        "a": ("First shape parameter of a beta fit.", "dimensionless"),
        "b": ("Second shape parameter of a beta fit.", "dimensionless"),
        "by": (
            "Shift added to a shifted fit, so the unshifted mean is mean minus by.",
            "data unit",
        ),
    },
}
# Every documented column as (table, column): Column. The metric columns of
# summary, stations, and stock_points read their entries from METRICS.
COLUMNS = {
    **{
        (table, column): Column(
            meaning=meaning,
            units=units,
        )
        for table, entries in _SCHEMA_COLUMNS.items()
        for column, (meaning, units) in entries.items()
    },
    **{
        (metric.table, name): Column(
            meaning=metric.meaning,
            units=metric_units(name, "time unit"),
        )
        for name, metric in METRICS.items()
    },
}


def _grouped_metrics() -> dict[str, list[str]]:
    return {
        "system metrics": SYSTEM_METRICS,
        "station metrics": STATION_METRICS,
        "stock-point metrics": STOCK_POINT_METRICS,
    }


def unknown(
    kind: str,
    name: str,
    choices: list[str] | dict[str, list[str]],
    *,
    intro: str | None = None,
) -> ValueError:
    """Build the error for a name that is not among the choices.

    The message names what was not found, lists the close matches when there
    are any, and lists the choices, grouped when the choices are a mapping from
    a group label to its names.
    """
    groups = choices if isinstance(choices, dict) else {f"{kind}s": list(choices)}
    pool = [choice for group in groups.values() for choice in group]
    close = difflib.get_close_matches(
        str(name),
        pool,
        n=3,
    )
    lines = [intro or f"{kind.capitalize()} {name!r} is missing in this experiment."]
    if close:
        lines.append(f"Close matches: {', '.join(close)}.")
    if isinstance(choices, dict):
        lines.append("Available:")
        lines += [f"  {label}: {', '.join(group)}" for label, group in groups.items()]
    else:
        lines.append(f"Available {kind}s: {', '.join(pool)}.")
    return ValueError("\n".join(lines))


def metric_error(name: str) -> ValueError:
    """Build the error for a name that is not a registered metric."""
    tables = sorted(table for table, column in COLUMNS if column == name)
    if tables:
        reasons = [
            EXCLUDED[(table, name)] for table in tables if (table, name) in EXCLUDED
        ]
        if len(tables) == 1:
            where = f"a {tables[0]} column"
        else:
            where = f"a column of the {', '.join(tables)} tables"
        return ValueError(
            " ".join(
                [
                    f"{name!r} is {where}, not a replication metric.",
                    *reasons,
                    "Call flowsim.metrics() to list the metrics.",
                ]
            )
        )
    error = unknown(
        kind="metric",
        name=name,
        choices=_grouped_metrics(),
        intro=f"Unknown metric {name!r}.",
    )
    return ValueError(f"{error}\nCall flowsim.metrics() for their meanings.")


def scope_name(metric: str) -> str:
    return SCOPES[METRICS[metric].scope].label


class Inapplicable(ValueError):
    """A metric whose scope the model does not have, e.g., stock points."""


# System metrics about when the run stopped, which every model has, unlike the
# customer counts and averages a supply chain without stations lacks.
RUN_LEVEL_METRICS = frozenset({"end_time", "overtime"})


def inapplicable(
    metric: str,
    *,
    stations: bool,
    stock_points: bool,
    supply: bool,
) -> Inapplicable | None:
    """Return the error for a metric that this kind of model cannot measure.

    A service network has no stock points. A supply chain without production
    stations has no stations and no customers, so its system metrics would all
    be missing or zero. A model without stations has no station metrics.
    """
    scope = METRICS[metric].scope
    # The run's stopping time applies to every model.
    if metric in RUN_LEVEL_METRICS:
        return None
    if scope == "stock_point" and not stock_points:
        other = ", or a station metric, e.g., metric='mean_wait'," if stations else ""
        return Inapplicable(
            f"This model has no stock points, so stock-point metrics such as "
            f"{metric!r} do not apply. Choose a system metric, e.g., "
            f"metric='mean_total_wait'{other}."
        )
    if scope != "stock_point" and supply and not stations:
        return Inapplicable(
            "This supply chain has no production stations, so it has no "
            f"customers or stations, and {scope_name(metric)} metrics such as "
            f"{metric!r} do not apply. Choose a stock-point metric, e.g., "
            "metric='fill_rate'."
        )
    if scope == "station" and not stations:
        return Inapplicable(
            f"This model has no stations, so station metrics such as {metric!r} "
            "do not apply. Choose a system metric, e.g., metric='mean_total_wait'."
        )
    return None


def metrics(scope: Scope | None = None) -> pl.DataFrame:
    """List the replication metrics that ``intervals`` and ``plot`` accept.

    Parameters
    ----------
    scope : {"system", "station", "stock_point"}, optional
        Keep only the metrics of this scope. The default lists every metric.

    Returns
    -------
    polars.DataFrame
        One row per metric with its name, scope, table, label, units, needs
        (the argument that picks one location), window, and meaning. Time
        metrics list "time unit" because no model is known here.

    Example
    -------
    >>> flowsim.metrics("station").select("name", "units", "meaning")
    """
    if scope is not None and scope not in SCOPES:
        raise ValueError(f"Choose a scope from {list(SCOPES)}, not {scope!r}.")
    chosen = {
        name: metric
        for name, metric in METRICS.items()
        if scope is None or metric.scope == scope
    }
    return pl.DataFrame(
        data={
            "name": list(chosen),
            "scope": [metric.scope for metric in chosen.values()],
            "table": [metric.table for metric in chosen.values()],
            "label": [metric.label for metric in chosen.values()],
            "units": [metric_units(name, "time unit") for name in chosen],
            "needs": [SCOPES[metric.scope].needs for metric in chosen.values()],
            "window": [metric.window for metric in chosen.values()],
            "meaning": [metric.meaning for metric in chosen.values()],
        },
        schema={
            "name": pl.String,
            "scope": pl.String,
            "table": pl.String,
            "label": pl.String,
            "units": pl.String,
            "needs": pl.String,
            "window": pl.String,
            "meaning": pl.String,
        },
    )


def explain(name: str) -> None:
    """Print what a metric or table column means and how to request it.

    The printout covers the replication metric called ``name`` (its label,
    scope, table, units, window, meaning, and how to request it), every table
    column with that name, and the reason a column is not a metric. A name that
    is not in the catalog prints its close matches.

    Parameters
    ----------
    name : str
        A metric or column name, e.g. ``"mean_wait"`` or ``"customer_id"``.

    Returns
    -------
    None
        The description is printed, not returned.

    Example
    -------
    >>> flowsim.explain("p95_wait")
    """
    metric = METRICS.get(name)
    columns = sorted((table, column) for table, column in COLUMNS if column == name)
    if metric is None and not columns:
        pool = sorted({*METRICS, *(column for _, column in COLUMNS)})
        close = difflib.get_close_matches(
            name,
            pool,
            n=3,
        )
        print(f"{name!r} is not a metric or column name.")
        print(
            f"Close matches: {', '.join(close)}."
            if close
            else "There are no close matches."
        )
        print("Call flowsim.metrics() to list the replication metrics.")
        return
    lines = []
    if metric is not None:
        where = SCOPES[metric.scope].example
        call = f'"{name}"' if not where else f'"{name}", {where}'
        lines += [
            f"{name}: {metric.label}",
            f"  Replication metric ({SCOPES[metric.scope].label} scope, "
            f"{metric.table} table)",
            f"    Units:   {metric_units(name, 'time unit')}",
            f"    Window:  {metric.window}",
            f"    Meaning: {metric.meaning}",
            f"    Request: experiment.intervals({call})",
            f"             comparison.intervals({call})",
        ]
    else:
        lines.append(name)
    for table, column in columns:
        entry = COLUMNS[(table, column)]
        lines.append(f"  Column in the {table} table (units: {entry.units})")
        if metric is None or entry.meaning != metric.meaning:
            lines.append(f"    Meaning: {entry.meaning}")
        if (table, column) in EXCLUDED:
            lines.append(f"    Not a replication metric: {EXCLUDED[(table, column)]}")
    print("\n".join(lines))
