"""Table schemas, the slotted records a run builds, and the frames made from them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

CUSTOMER_SCHEMA = {
    "customer_id": pl.Int64,
    "source": pl.String,
    "customer_type": pl.String,
    "quantity": pl.Float64,
    "arrival_time": pl.Float64,
    "exit_time": pl.Float64,
    "abandon_time": pl.Float64,
    "flow_time": pl.Float64,
    "total_wait": pl.Float64,
    "total_blocked": pl.Float64,
    "admitted": pl.Boolean,
    "admission_time": pl.Float64,
    "outside_wait": pl.Float64,
    "status": pl.String,
    "station": pl.String,
}
VISIT_SCHEMA = {
    "customer_id": pl.Int64,
    "source": pl.String,
    "customer_type": pl.String,
    "station": pl.String,
    "arrival_time": pl.Float64,
    "service_start": pl.Float64,
    "service_end": pl.Float64,
    "departure_time": pl.Float64,
    "abandon_time": pl.Float64,
    "server_id": pl.Int64,
    "blocked_time": pl.Float64,
    "wait_time": pl.Float64,
    "service_time": pl.Float64,
}
STATE_SCHEMA = {
    "station": pl.String,
    "time": pl.Float64,
    "waiting": pl.Int64,
    "busy": pl.Int64,
    "processing": pl.Int64,
    "blocked": pl.Int64,
    "paused": pl.Int64,
    "down": pl.Int64,
    "scheduled": pl.Int64,
    "available": pl.Int64,
    "outside": pl.Int64,
}
ACTIVITY_SCHEMA = {
    "customer_id": pl.Int64,
    "station": pl.String,
    "server_id": pl.Int64,
    "state": pl.String,
    "start": pl.Float64,
    "end": pl.Float64,
}
FAILURE_SCHEMA = {
    "station": pl.String,
    "server_id": pl.Int64,
    "failure_time": pl.Float64,
    "time_to_failure": pl.Float64,
    "repair_time": pl.Float64,
    "repair_end": pl.Float64,
}
# The stations table's column types. STATION_SCHEMA, near the end of this
# module, puts them in display order.
_STATION_TYPES = {
    "station": pl.String,
    "capacity": pl.Int64,
    "arrivals": pl.Int64,
    "wait_arrivals": pl.Int64,
    "service_starts": pl.Int64,
    "service_completions": pl.Int64,
    "wait_observations": pl.Int64,
    "waiting_at_close": pl.Int64,
    "occupied_at_close": pl.Int64,
    "waiting_at_end": pl.Int64,
    "occupied_at_end": pl.Int64,
    "blocked_at_close": pl.Int64,
    "paused_at_close": pl.Int64,
    "outside_at_close": pl.Int64,
    "mean_wait": pl.Float64,
    "p95_wait": pl.Float64,
    "mean_queue": pl.Float64,
    "max_queue": pl.Int64,
    "utilization": pl.Float64,
    "processing_utilization": pl.Float64,
    "blocking_fraction": pl.Float64,
    "downtime_fraction": pl.Float64,
    "mean_scheduled_servers": pl.Float64,
    "mean_available_servers": pl.Float64,
    "mean_outside_queue": pl.Float64,
    "staffed_utilization": pl.Float64,
}
# One row per stock point and review period. Stock columns are end-of-period
# values after the review's order, and flow columns are totals for the period.
# on_order and position are recorded at the stage's own review, so an order
# that the stage above loses later in the same review still counts in them.
INVENTORY_SCHEMA = {
    "stock_point": pl.String,
    "period": pl.Int64,
    "time": pl.Float64,
    "received": pl.Float64,
    "demand": pl.Float64,
    "filled": pl.Float64,
    "shipped": pl.Float64,
    "lost": pl.Float64,
    "on_hand": pl.Float64,
    "backorders": pl.Float64,
    "on_order": pl.Float64,
    "position": pl.Float64,
    "forecast": pl.Float64,
    "target": pl.Float64,
    "order_quantity": pl.Float64,
}
SHIPMENT_SCHEMA = {
    "sender": pl.String,
    "receiver": pl.String,
    "quantity": pl.Float64,
    "ship_time": pl.Float64,
    "lead_time": pl.Float64,
    "arrival_time": pl.Float64,
}
STOCK_POINT_SCHEMA = {
    "stock_point": pl.String,
    "periods": pl.Int64,
    "mean_demand": pl.Float64,
    "mean_order": pl.Float64,
    "demand_variance": pl.Float64,
    "order_variance": pl.Float64,
    "bullwhip_ratio": pl.Float64,
    "fill_rate": pl.Float64,
    "mean_on_hand": pl.Float64,
    "mean_backorders": pl.Float64,
    "lost_units": pl.Float64,
}
# The top supplier's name in the shipments table. It has unlimited stock.
SUPPLIER = "Supplier"
# State counts integrated over time for the station table.
AREA_FIELDS = [
    "waiting",
    "busy",
    "processing",
    "blocked",
    "down",
    "scheduled",
    "available",
    "outside",
]


class Status(StrEnum):
    """A customer's status, as written to the customers table."""

    OUTSIDE = "outside"
    WAITING = "waiting"
    IN_SERVICE = "in_service"
    BLOCKED = "blocked"
    PAUSED = "paused"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


class Phase(StrEnum):
    """A server's phase. Activity rows record every phase except idle."""

    IDLE = "idle"
    PROCESSING = "processing"
    BLOCKED = "blocked"
    PAUSED = "paused"


# The run builds one record per row, and each record's fields equal its schema's
# columns in order (tests check this). Slots make a misspelled field an error
# rather than a silent extra key.
@dataclass(
    slots=True,
    kw_only=True,
)
class _Customer:
    customer_id: int
    source: str
    # None for a customer from a source without types.
    customer_type: str | None = None
    # Units in a production job, or None for a customer from a source.
    quantity: float | None = None
    arrival_time: float
    exit_time: float | None = None
    abandon_time: float | None = None
    flow_time: float | None = None
    total_wait: float = 0.0
    total_blocked: float = 0.0
    admitted: bool = False
    admission_time: float | None = None
    outside_wait: float | None = None
    # The customer's process sets a status when it starts, at the arrival time.
    status: Status | None = None
    station: str | None = None


@dataclass(
    slots=True,
    kw_only=True,
)
class _Visit:
    customer_id: int
    source: str
    customer_type: str | None = None
    station: str
    arrival_time: float
    service_start: float | None = None
    service_end: float | None = None
    departure_time: float | None = None
    abandon_time: float | None = None
    server_id: int | None = None
    blocked_time: float | None = None
    wait_time: float | None = None
    service_time: float


@dataclass(
    slots=True,
    kw_only=True,
)
class _State:
    station: str
    time: float
    waiting: int
    busy: int
    processing: int
    blocked: int
    paused: int
    down: int
    scheduled: int
    available: int
    outside: int


@dataclass(
    slots=True,
    kw_only=True,
)
class _Activity:
    customer_id: int
    station: str
    server_id: int
    state: Phase
    start: float
    end: float | None = None


@dataclass(
    slots=True,
    kw_only=True,
)
class _Failure:
    station: str
    server_id: int
    failure_time: float
    time_to_failure: float
    repair_time: float
    repair_end: float | None = None


@dataclass(
    slots=True,
    kw_only=True,
)
class _Inventory:
    stock_point: str
    period: int
    time: float
    received: float
    demand: float
    filled: float
    shipped: float
    lost: float
    on_hand: float
    backorders: float
    on_order: float
    position: float
    forecast: float
    target: float
    order_quantity: float


@dataclass(
    slots=True,
    kw_only=True,
)
class _Shipment:
    sender: str
    receiver: str
    quantity: float
    ship_time: float
    lead_time: float
    arrival_time: float | None = None


def _frame(records: list, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Build a table from records, reading each column by its field name.

    Polars would match dataclass fields to the schema by position and cast
    mismatches silently, so the columns are gathered by name instead.
    """
    return pl.DataFrame(
        data={
            column: [getattr(record, column) for record in records] for column in schema
        },
        schema=schema,
    )


# Display order of the stations and summary tables: the headline columns come
# first, then the rest in the order their builders compute them. Order is a
# display choice, applied as each builder's final select, so reordering never
# touches a computation.
STATION_HEADLINE = [
    "station",
    "mean_wait",
    "mean_queue",
    "utilization",
    "p95_wait",
    "max_queue",
]
SUMMARY_HEADLINE = [
    "arrivals",
    "completed",
    "mean_total_wait",
    "mean_flow_time",
    "throughput",
    "overtime",
]


def headline_first(columns: list[str], headline: list[str]) -> list[str]:
    """Order columns with the headline columns first and the rest after them."""
    return [*headline, *(column for column in columns if column not in headline)]


STATION_SCHEMA = {
    column: _STATION_TYPES[column]
    for column in headline_first(list(_STATION_TYPES), STATION_HEADLINE)
}


INTERVAL_SCHEMA = {
    "scenario": pl.String,
    "baseline": pl.String,
    "station": pl.String,
    "stock_point": pl.String,
    "metric": pl.String,
    "runs": pl.Int64,
    "observed_runs": pl.Int64,
    "mean": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
    "confidence": pl.Float64,
}


def interval_frame(rows: list[dict], keys: list[str]) -> pl.DataFrame:
    """Build an interval table whose columns exist even when it has no rows.

    The keys name the identifying columns in display order, and the estimate
    columns from "runs" to "confidence" follow them.
    """
    columns = [*keys, "runs", "observed_runs", "mean", "lower", "upper", "confidence"]
    return pl.DataFrame(
        data=rows,
        schema={column: INTERVAL_SCHEMA[column] for column in columns},
    )
