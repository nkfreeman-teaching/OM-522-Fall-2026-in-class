"""Plotly figures and page data, with no notebook dependency.

A function named ..._figure returns a figure as plain data, a dict of "data"
(the traces) and "layout" as Plotly's JSON holds them. A viewer page embeds
that dict as it is, and the plot methods wrap the same dict with
plotly_figure, so each figure is defined once. Building Plotly objects for the
hundreds of figures a dashboard holds cost most of its build time.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import functools
import html
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl

from flowsim.catalog import METRICS, unknown
from flowsim.records import Phase, Status
from flowsim.validation import integer

if TYPE_CHECKING:
    from flowsim.results import Results

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#667085"
# Marks a diagram node that keeps the model from running.
WARNING = "#C62828"

# Customer timeline activities in legend order, with their bar colors.
TIMELINE_COLORS = {
    "Waiting": BLUE,
    "In service": ORANGE,
    "Paused": PURPLE,
    "Blocked": GREEN,
    "Outside": GRAY,
}


def label(text: str) -> str:
    """Escape model text, since Plotly renders a subset of HTML in labels."""
    return html.escape(text, quote=False)


# True while a page is built, when trace columns become typed arrays.
_BUILDING_PAGE = contextvars.ContextVar("building_page", default=False)


@contextlib.contextmanager
def building_page() -> Iterator[None]:
    """Write the trace columns of figures built inside the block as typed arrays."""
    token = _BUILDING_PAGE.set(True)
    try:
        yield
    finally:
        _BUILDING_PAGE.reset(token)


# plotly.js names the typed arrays it reads by these codes.
TYPED_ARRAY_CODES = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}


def typed_array(values: np.ndarray) -> dict | list:
    """Encode an array as plotly.js typed-array data, as Plotly's own JSON does.

    plotly.js has no 64-bit integers, so an int64 array takes the smallest
    integer type that holds it. An empty array, or one that no typed array
    holds, stays a list.
    """
    if values.size == 0:
        return []
    if values.dtype == np.int64:
        smaller = [
            kind
            for kind in [np.int8, np.int16, np.int32]
            if np.iinfo(kind).min <= values.min() and values.max() <= np.iinfo(kind).max
        ]
        if not smaller:
            return values.tolist()
        values = values.astype(smaller[0])
    code = TYPED_ARRAY_CODES.get(str(values.dtype))
    if code is None:
        return values.tolist()
    return {
        "dtype": code,
        "bdata": base64.b64encode(np.ascontiguousarray(values)).decode("ascii"),
    }


def trace_values(values: pl.Series) -> list | dict:
    """Return a record column for a trace: a typed array on a page, a list otherwise.

    A typed array is compact base64 data, which keeps run pages small and fast
    to write. A column with a null or NaN stays a list on a page, with NaN and
    infinite values as None, as Plotly's own JSON writes them. Figures
    returned to users hold lists.
    """
    if not _BUILDING_PAGE.get():
        return values.to_list()
    if values.null_count() > 0 or (values.dtype.is_float() and values.is_nan().any()):
        return [
            None if value is not None and not math.isfinite(value) else value
            for value in values.to_list()
        ]
    return typed_array(values.to_numpy())


def base_layout(x_label: str, y_label: str) -> dict:
    """Return FlowSim's size, font, axis titles, margins, and legend as layout data."""
    return {
        "width": 800,
        "height": 500,
        "font": {"family": "Georgia, serif", "size": 14},
        "xaxis": {"title": {"text": label(x_label)}, "rangemode": "tozero"},
        "yaxis": {"title": {"text": label(y_label)}, "rangemode": "tozero"},
        "margin": {"l": 85, "r": 30, "t": 35, "b": 70},
        "legend": {"orientation": "h", "y": 1.14, "x": 0},
        "hovermode": "closest",
    }


def plotly_figure(figure: dict) -> go.Figure:
    """Wrap a figure's data in a Plotly figure with the plotly_white template.

    A page holds that template once (see page_template) and adds it to each
    figure it draws, so its figures leave it out.
    """
    return go.Figure(
        data=figure["data"],
        layout={**figure["layout"], "template": "plotly_white"},
    )


def _merged(base: dict, extra: dict) -> dict:
    """Return base updated with extra, as Plotly's update_layout merges them.

    A dict merges into an existing dict, and any other value replaces the old
    one.
    """
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merged(merged[key], value)
        else:
            merged[key] = value
    return merged


def style(figure: go.Figure, x_label: str, y_label: str) -> go.Figure:
    """Give a Plotly figure base_layout's values and the plotly_white template.

    The figure gets one new Layout, its own layout merged with these values,
    because update_layout costs about three times as much per figure. Layout
    set before (e.g., a legend position) is kept, but the new axis titles
    replace the old ones whole. The merge leaves out the old template, since
    validating it again is slow.
    """
    layout = figure.layout.to_plotly_json()
    layout.pop("template", None)
    for axis in ["xaxis", "yaxis"]:
        layout.get(axis, {}).pop("title", None)
    shared = {**base_layout(x_label, y_label), "template": "plotly_white"}
    figure.layout = go.Layout(_merged(layout, shared))
    return figure


def dashed_line(figure: dict, x: float) -> None:
    """Add a dashed gray vertical line at x, as Plotly's add_vline draws it."""
    figure["layout"].setdefault("shapes", []).append(
        {
            "type": "line",
            "x0": x,
            "x1": x,
            "xref": "x",
            "y0": 0,
            "y1": 1,
            "yref": "y domain",
            "line": {"color": GRAY, "dash": "dash"},
        }
    )


def warmup_line(figure: dict, result: Results) -> None:
    if result.settings.warmup > 0:
        dashed_line(figure, result.settings.warmup)


def station_menu(figure: dict, stations: list[str]) -> None:
    """Add a station menu that shows the traces whose meta names that station.

    The viewer's station choice selects traces by the same meta value.
    """
    if not stations:
        return
    buttons = [
        {
            "label": label(station),
            "method": "update",
            "args": [
                {"visible": [trace.get("meta") == station for trace in figure["data"]]}
            ],
        }
        for station in stations
    ]
    figure["layout"]["updatemenus"] = [
        {"buttons": buttons, "x": 1, "y": 1.16, "xanchor": "right"}
    ]


def require_stations(result: Results) -> None:
    """Explain that a run without stations has no station plots."""
    if not result.settings.capacities:
        if result.supply is not None:
            raise ValueError(
                "This run has no stations. plot_inventory() and plot_orders() "
                "show its stock points."
            )
        raise ValueError(
            "This run has no stations, so it has no station plots. Its summary "
            "table holds the customer totals, and plot_customer() shows one "
            "customer's timeline."
        )


def chosen_stations(result: Results, station: str | None) -> list[str]:
    """Return the stations a station plot draws, every one unless one is named.

    A run without stations draws empty station figures, as the viewer needs,
    and the public plot methods call require_stations first.
    """
    # Only a comparison's shared settings hold no capacities.
    capacities = result.settings.capacities
    assert capacities is not None
    names = list(capacities)
    if station is None:
        return names
    require_stations(result)
    if station not in names:
        raise unknown(
            kind="station",
            name=station,
            choices=names,
            intro=f"Station {station!r} is not in this run.",
        )
    return [station]


def chosen_stock_points(result: Results, stock_point: str | None) -> list[str]:
    """Return the stock points a plot draws, every one unless one is named."""
    names = result.stock_point_names
    if not names:
        raise ValueError(
            "This run has no stock points. plot_queue(), plot_wait(), "
            "plot_utilization(), and plot_resources() show its stations."
        )
    if stock_point is None:
        return names
    if stock_point not in names:
        raise unknown(
            kind="stock point",
            name=stock_point,
            choices=names,
            intro=f"Stock point {stock_point!r} is not in this run.",
        )
    return [stock_point]


def queue_figure(result: Results, station: str | None = None) -> dict:
    """Return waiting customers and occupied servers over time at each station.

    Without a station, a menu chooses among the stations, and the first is
    shown. A named station is the only one drawn.
    """
    names = chosen_stations(result, station)
    figure = {
        "data": [],
        "layout": base_layout(
            x_label=f"Simulation time ({result.settings.time_unit})",
            y_label="Customers",
        ),
    }
    for index, place in enumerate(names):
        history = result.states.filter(pl.col("station") == place)
        for field, color, name in [
            ("waiting", BLUE, "Waiting"),
            ("busy", ORANGE, "Occupied servers"),
        ]:
            figure["data"].append(
                {
                    "type": "scatter",
                    "x": trace_values(history.get_column("time")),
                    "y": trace_values(history.get_column(field)),
                    "name": name,
                    "mode": "lines",
                    "line": {"shape": "hv", "color": color, "width": 2},
                    "visible": index == 0,
                    "meta": place,
                    "hovertemplate": f"%{{x:.2f}} {label(result.settings.time_unit)}"
                    "<br>%{y} customers<extra>%{fullData.name}</extra>",
                }
            )
    station_menu(
        figure=figure,
        stations=names if station is None else [],
    )
    figure["layout"]["xaxis"]["rangeslider"] = {"visible": True, "thickness": 0.1}
    if result.settings.drain:
        dashed_line(figure, result.settings.duration)
    warmup_line(figure, result)
    return figure


def wait_figure(result: Results, station: str | None = None) -> dict:
    """Return each station's observed waits for station arrivals after warm-up."""
    names = chosen_stations(result, station)
    figure = {
        "data": [],
        "layout": base_layout(
            x_label=f"Waiting time ({result.settings.time_unit})",
            y_label="Customers",
        ),
    }
    for index, place in enumerate(names):
        waits = trace_values(
            result.visits.filter(
                (pl.col("station") == place)
                & (pl.col("arrival_time") >= result.settings.warmup)
                & pl.col("wait_time").is_not_null()
            ).get_column("wait_time")
        )
        figure["data"].append(
            {
                "type": "histogram",
                "x": waits,
                "name": label(place),
                "visible": index == 0,
                "meta": place,
                "marker": {"color": BLUE, "line": {"color": "black", "width": 1}},
                "hovertemplate": "Wait bin: %{x}<br>Customers: %{y}<extra></extra>",
            }
        )
    station_menu(
        figure=figure,
        stations=names if station is None else [],
    )
    figure["layout"]["showlegend"] = False
    return figure


def utilization_figure(result: Results, station: str | None = None) -> dict:
    """Return each station's occupied share of installed capacity, or one station's."""
    names = chosen_stations(result, station)
    stations = result.stations.filter(pl.col("station").is_in(names))
    layout = base_layout(
        x_label="Occupied server capacity",
        y_label="Station",
    )
    layout["xaxis"].update(range=[0, 1.05], tickformat=".0%")
    layout["height"] = max(220, 150 + 40 * len(names))
    trace = {
        "type": "scatter",
        "x": trace_values(stations.get_column("utilization")),
        "y": [label(name) for name in stations.get_column("station")],
        "mode": "markers",
        "marker": {"size": 12, "color": BLUE, "line": {"color": "black", "width": 1}},
        "hovertemplate": "%{y}<br>Occupied capacity: %{x:.1%}<extra></extra>",
    }
    return {"data": [trace], "layout": layout}


# Server-state lines in legend order, with their colors and dashes.
RESOURCE_LINES = [
    ("processing", ORANGE, "Processing", "solid"),
    ("blocked", GREEN, "Blocked", "solid"),
    ("paused", PURPLE, "Paused", "solid"),
    ("down", "#222222", "Down", "solid"),
    ("available", GRAY, "Available", "dash"),
    ("scheduled", BLUE, "Scheduled", "dash"),
]
# Stacked server states from the bottom up. Each server counts in one layer.
RESOURCE_LAYERS = [
    ("processing", ORANGE, "Processing"),
    ("blocked", GREEN, "Blocked"),
    ("paused", PURPLE, "Paused"),
    ("down", "#222222", "Down"),
    ("idle", "#BFC9D1", "Idle"),
]


def server_layers(result: Results, station: str) -> pl.DataFrame:
    """Count one station's servers in each stacked layer at every states time.

    The states table counts each condition on its own, so a down server that
    holds a paused job is both paused and down there, and the table cannot
    say which down servers hold a job. This function rebuilds each server's
    condition after every event at each states time from the activities
    table (its job and phase), the failures table (its repairs), and the
    states table's scheduled count (server k is on duty while that count is
    at least k, since shifts staff the lowest-numbered servers). A server
    holding a job counts in its phase, a down server without a job counts as
    down, and an on-duty server that is up and has no job is idle. An
    off-duty server with no job and no repair is not counted. Thus, the
    layers sum to the scheduled servers plus any off-duty server that still
    holds a job or is under repair.
    """
    capacities = result.settings.capacities
    assert capacities is not None
    capacity = capacities[station]
    # The recorded count, not the staffing schedule, since a run that stops at
    # closing does not execute a shift change scheduled exactly then.
    times = result.states.filter(pl.col("station") == station).select(
        "time",
        "scheduled",
    )
    # Zero-length intervals hold no state after the events at their time.
    jobs = result.activities.filter(
        (pl.col("station") == station)
        & (pl.col("end").is_null() | (pl.col("end") > pl.col("start")))
    )
    repairs = result.failures.filter(
        (pl.col("station") == station)
        & (
            pl.col("repair_end").is_null()
            | (pl.col("repair_end") > pl.col("failure_time"))
        )
    )
    servers = []
    for server in range(1, capacity + 1):
        # A server's intervals never overlap, so the latest one that starts at
        # or before a time is the only one that can cover it.
        servers.append(
            times.join_asof(
                jobs.filter(pl.col("server_id") == server)
                .select("start", "end", "state")
                .sort("start"),
                left_on="time",
                right_on="start",
                strategy="backward",
            )
            .join_asof(
                repairs.filter(pl.col("server_id") == server)
                .select("failure_time", "repair_end")
                .sort("failure_time"),
                left_on="time",
                right_on="failure_time",
                strategy="backward",
            )
            .select(
                "time",
                pl.when(pl.col("end").is_null() | (pl.col("time") < pl.col("end")))
                .then(pl.col("state"))
                .alias("phase"),
                (
                    pl.col("failure_time").is_not_null()
                    & (
                        pl.col("repair_end").is_null()
                        | (pl.col("time") < pl.col("repair_end"))
                    )
                ).alias("down"),
                (pl.col("scheduled") >= server).alias("on_duty"),
            )
        )
    free = pl.col("phase").is_null()
    return (
        pl.concat(servers)
        .group_by("time", maintain_order=True)
        .agg(
            *[
                (pl.col("phase") == phase).sum().cast(pl.Int64).alias(phase)
                for phase in [Phase.PROCESSING, Phase.BLOCKED, Phase.PAUSED]
            ],
            (free & pl.col("down")).sum().cast(pl.Int64).alias("down"),
            (free & ~pl.col("down") & pl.col("on_duty"))
            .sum()
            .cast(pl.Int64)
            .alias("idle"),
            pl.col("on_duty").sum().cast(pl.Int64).alias("scheduled"),
        )
        .sort("time")
    )


def resources_figure(
    result: Results,
    station: str | None = None,
    stacked: bool = False,
) -> dict:
    """Return each station's server states over time, as lines or stacked.

    The lines are the states table's processing, blocked, paused, down,
    available, and scheduled counts, which overlap (e.g., a down server
    holding a paused job counts as both paused and down). The stacked view
    counts each server in one layer from server_layers and draws the
    scheduled count as a dashed line over the stack.
    """
    names = chosen_stations(result, station)
    figure = {
        "data": [],
        "layout": base_layout(
            x_label=f"Simulation time ({result.settings.time_unit})",
            y_label="Servers",
        ),
    }
    hover = "%{x:.2f} " + label(result.settings.time_unit)
    for index, place in enumerate(names):
        history = result.states.filter(pl.col("station") == place)
        times = trace_values(history.get_column("time"))
        if stacked:
            layers = server_layers(result, place)
            for field, color, name in RESOURCE_LAYERS:
                figure["data"].append(
                    {
                        "type": "scatter",
                        "x": times,
                        "y": trace_values(layers.get_column(field)),
                        "name": name,
                        "mode": "lines",
                        # Each station stacks its own layers.
                        "stackgroup": f"station {index}",
                        "line": {"shape": "hv", "color": color, "width": 0.5},
                        "fillcolor": color,
                        "visible": index == 0,
                        "meta": place,
                        "hovertemplate": hover
                        + "<br>%{y} servers<extra>%{fullData.name}</extra>",
                    }
                )
        lines = (
            [line for line in RESOURCE_LINES if line[0] == "scheduled"]
            if stacked
            else RESOURCE_LINES
        )
        for field, color, name, dash in lines:
            figure["data"].append(
                {
                    "type": "scatter",
                    "x": times,
                    "y": trace_values(history.get_column(field)),
                    "name": name,
                    "mode": "lines",
                    "line": {"shape": "hv", "color": color, "width": 2, "dash": dash},
                    "visible": index == 0,
                    "meta": place,
                    "hovertemplate": hover
                    + "<br>%{y} servers<extra>%{fullData.name}</extra>",
                }
            )
    station_menu(
        figure=figure,
        stations=names if station is None else [],
    )
    figure["layout"].update(
        legend={"orientation": "v", "x": 1.02, "y": 1},
        margin={"l": 85, "r": 175, "t": 60, "b": 70},
    )
    if result.settings.drain:
        dashed_line(figure, result.settings.duration)
    warmup_line(figure, result)
    return figure


def timeline_spans(result: Results) -> pl.DataFrame:
    """Return every customer's timeline bars, grouped in TIMELINE_COLORS order.

    Waiting runs from station arrival to service start or abandonment, and the
    server activity rows give service, paused, and blocked time. A customer who
    waited outside a full buffer gets an "Outside" bar until admission or
    abandonment. An open end runs to the end time. Zero-length service writes no
    activity row, so it draws no bar.
    """
    waiting = result.visits.select(
        "customer_id",
        pl.lit("Waiting").alias("activity"),
        "station",
        pl.col("arrival_time").alias("start"),
        pl.coalesce("service_start", "abandon_time").alias("end"),
    )
    activities = [
        result.activities.filter(pl.col("state") == state).select(
            "customer_id",
            pl.lit(activity).alias("activity"),
            "station",
            "start",
            "end",
        )
        for activity, state in [
            ("In service", Phase.PROCESSING),
            ("Paused", Phase.PAUSED),
            ("Blocked", Phase.BLOCKED),
        ]
    ]
    outside = result.customers.filter(
        (pl.col("status") == Status.OUTSIDE)
        | (pl.col("outside_wait").is_not_null() & (pl.col("outside_wait") != 0))
        | (
            (pl.col("status") == Status.ABANDONED)
            & pl.col("admission_time").is_null()
            & (pl.col("abandon_time") > pl.col("arrival_time"))
        )
    ).select(
        "customer_id",
        pl.lit("Outside").alias("activity"),
        pl.lit("Outside").alias("station"),
        pl.col("arrival_time").alias("start"),
        pl.coalesce("admission_time", "abandon_time").alias("end"),
    )
    return pl.concat([waiting, *activities, outside]).with_columns(
        pl.col("end").fill_null(result.settings.end_time),
    )


def plot_customer(result: Results, customer_id: int) -> go.Figure:
    identifier = integer(customer_id, "Customer ID")
    if result.customers.filter(pl.col("customer_id") == identifier).is_empty():
        raise ValueError(f"Customer {identifier} does not exist in this run.")
    spans = timeline_spans(result).filter(pl.col("customer_id") == identifier)
    figure = go.Figure()
    for activity, color in TIMELINE_COLORS.items():
        bars = spans.filter(pl.col("activity") == activity)
        figure.add_trace(
            go.Bar(
                base=bars.get_column("start").to_list(),
                x=(bars.get_column("end") - bars.get_column("start")).to_list(),
                y=[label(station) for station in bars.get_column("station")],
                orientation="h",
                name=activity,
                marker={
                    "color": color,
                    "line": {"color": "black", "width": 1},
                },
                hovertemplate="%{y}<br>Duration: %{x:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    style(
        figure=figure,
        x_label=f"Simulation time ({result.settings.time_unit})",
        y_label="Station",
    )
    figure.update_layout(barmode="overlay")
    figure.update_yaxes(autorange="reversed")
    return figure


def service_payload(result: Results) -> dict:
    """Return the data the service-network run view draws."""
    figures = {
        "queues": queue_figure(result),
        "waits": wait_figure(result),
        "utilization": utilization_figure(result),
    }
    if result._resources_vary:
        figures["resources"] = resources_figure(result)
        # The page's switch swaps this figure into the resources chart.
        stacked = resources_figure(
            result,
            stacked=True,
        )
    else:
        stacked = None
    return {
        "kind": "service",
        "figures": figures,
        "stacked_resources": stacked,
        # Record tables go column by column, which is about half the size of
        # rows, and the run view rebuilds the rows it needs.
        "customers": result.customers.to_dict(as_series=False),
        "visits": result.visits.to_dict(as_series=False),
        # Table cells for these station metrics display as percentages.
        "percent_columns": [
            name for name, metric in METRICS.items() if metric.kind == "fraction"
        ],
        # Downloads use the export column order, even for empty tables.
        "columns": {
            "customers": result.customers.columns,
            "visits": result.visits.columns,
        },
        # The viewer draws these bars as given, so it matches plot_customer.
        "timeline": timeline_spans(result).to_dict(as_series=False),
        "timeline_colors": TIMELINE_COLORS,
        "failures": result.failures.to_dict(as_series=False),
        "stations": result.stations.to_dict(as_series=False),
        "summary": result.summary.to_dicts()[0],
        "duration": result.settings.duration,
        "end_time": result.settings.end_time,
        "drain": result.settings.drain,
        "warmup": result.settings.warmup,
        "time_unit": result.settings.time_unit,
        "seed": result.settings.seed,
    }


# Stock-point order traces cycle through these colors, after consumer demand.
STAGE_COLORS = [BLUE, ORANGE, GREEN, PURPLE, "#E69F00", "#56B4E9"]


def orders_figure(result: Results) -> dict:
    """Return consumer demand and every stage's orders, which shows the bullwhip."""
    names = chosen_stock_points(result, None)
    # chosen_stock_points raises for a run without a supply chain.
    supply = result.supply
    assert supply is not None
    retailer = result.inventory.filter(pl.col("stock_point") == names[0])
    figure = {
        "data": [
            {
                "type": "scatter",
                "x": trace_values(retailer.get_column("time")),
                "y": trace_values(retailer.get_column("demand")),
                "name": label(supply.chain[0]),
                "mode": "lines",
                "line": {"color": GRAY, "width": 2, "dash": "dot"},
            }
        ],
        "layout": base_layout(
            x_label=f"Review time ({result.settings.time_unit})",
            y_label="Units per period",
        ),
    }
    for index, name in enumerate(names):
        rows = result.inventory.filter(pl.col("stock_point") == name)
        figure["data"].append(
            {
                "type": "scatter",
                "x": trace_values(rows.get_column("time")),
                "y": trace_values(rows.get_column("order_quantity")),
                "name": label(f"{name} orders"),
                "mode": "lines",
                "line": {"color": STAGE_COLORS[index % len(STAGE_COLORS)], "width": 2},
            }
        )
    warmup_line(figure, result)
    # Startup orders can dwarf the measured weeks, so the initial view scales
    # to the periods after warm-up. Autoscaling restores the full range.
    measured = result.inventory.filter(pl.col("time") >= result.settings.warmup)
    if result.settings.warmup > 0 and not measured.is_empty():
        top = measured.select(
            pl.max_horizontal(
                pl.col("order_quantity").max(),
                pl.col("demand").max(),
            )
        ).item()
        figure["layout"]["yaxis"]["range"] = [0, 1.05 * top]
    return figure


def inventory_figure(result: Results, stock_point: str | None = None) -> dict:
    """Return a stock point's stock, backorders, and orders over review times.

    Without a stock point, a menu chooses among the stock points, and the
    first is shown. A named stock point is the only one drawn.
    """
    names = chosen_stock_points(result, stock_point)
    figure = {
        "data": [],
        "layout": base_layout(
            x_label=f"Review time ({result.settings.time_unit})",
            y_label="Units at the end of the period",
        ),
    }
    fields = [
        ("on_hand", BLUE, "On hand", "solid"),
        ("backorders", ORANGE, "Backorders", "solid"),
        ("order_quantity", GREEN, "Orders placed", "solid"),
        ("demand", GRAY, "Orders received", "dot"),
    ]
    for index, name in enumerate(names):
        rows = result.inventory.filter(pl.col("stock_point") == name)
        for field, color, trace_name, dash in fields:
            figure["data"].append(
                {
                    "type": "scatter",
                    "x": trace_values(rows.get_column("time")),
                    "y": trace_values(rows.get_column(field)),
                    "name": trace_name,
                    "mode": "lines",
                    "line": {"color": color, "width": 2, "dash": dash},
                    "visible": index == 0,
                    "meta": name,
                }
            )
    station_menu(
        figure=figure,
        stations=names if stock_point is None else [],
    )
    warmup_line(figure, result)
    return figure


def supply_payload(result: Results) -> dict:
    """Return the data the supply-chain run view draws."""
    figures = {
        "orders": orders_figure(result),
        "inventory": inventory_figure(result),
    }
    # A top stage that produces adds its stations, whose customers are jobs.
    if result.settings.capacities:
        queues = queue_figure(result)
        queues["layout"]["yaxis"]["title"]["text"] = "Jobs"
        for trace in queues["data"]:
            trace["hovertemplate"] = (
                f"%{{x:.2f}} {label(result.settings.time_unit)}"
                "<br>%{y} jobs<extra>%{fullData.name}</extra>"
            )
        figures["queues"] = queues
        figures["utilization"] = utilization_figure(result)
    # orders_figure above raises for a run without a supply chain.
    supply = result.supply
    assert supply is not None
    return {
        "kind": "supply",
        "figures": figures,
        "chain": list(supply.chain),
        "period": supply.period,
        # Record tables go column by column, as in service_payload.
        "stock_points": result.stock_points.to_dict(as_series=False),
        "inventory": result.inventory.to_dict(as_series=False),
        "shipments": result.shipments.to_dict(as_series=False),
        "stations": result.stations.to_dict(as_series=False),
        "jobs": result.customers.to_dict(as_series=False),
        "summary": result.summary.to_dicts()[0],
        # Downloads use the export column order, even for empty tables.
        "columns": {
            "inventory": result.inventory.columns,
            "shipments": result.shipments.columns,
            "jobs": result.customers.columns,
        },
        "percent_columns": [
            name for name, metric in METRICS.items() if metric.kind == "fraction"
        ],
        "duration": result.settings.duration,
        "warmup": result.settings.warmup,
        "time_unit": result.settings.time_unit,
        "seed": result.settings.seed,
    }


def run_payload(result: Results) -> dict:
    """Return one run's data for a run view, the service or supply-chain one."""
    return (
        supply_payload(result) if result.supply is not None else service_payload(result)
    )


@functools.cache
def page_template() -> dict:
    """Return the template plotly_figure gives every figure, as page JSON."""
    text = go.Figure(layout={"template": "plotly_white"}).to_json()
    # Plotly returns None only for a JSON engine it does not know, and it
    # rejects one before that point.
    assert text is not None
    return json.loads(text)["layout"]["template"]


def page_json(payload: object) -> str:
    """Return JSON that is safe inside a page's script element.

    Escaping '<' keeps model labels from ending the script element.
    """
    return json.dumps(payload, allow_nan=False).replace("<", "\\u003c")


def fill_template(name: str, values: dict[str, str]) -> str:
    """Fill a page template's __NAME__ placeholders in one pass.

    One pass means a placeholder-like name inside a value (e.g., in a
    scenario name or in Plotly's code) is never replaced.
    """
    template = Path(__file__).with_name(name).read_text(encoding="utf-8")
    return re.sub(
        pattern=r"__([A-Z_]+)__",
        repl=lambda match: values[match.group(1)],
        string=template,
    )


# Plotly's cartesian bundle draws the scatter, bar, and histogram traces the
# pages use at about a third of the full library's size, which the browser
# parses on every page load. tests/test_dashboard.py checks that its version
# matches the installed plotly package and that it holds every trace type
# FlowSim draws. Replace the file when plotly is upgraded, from
# https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@VERSION/plotly-cartesian.min.js
PLOTLY_BUNDLE = "plotly-cartesian.min.js"


def page_parts() -> dict[str, str]:
    """Return the style, Plotly, template, and run views that every page holds."""
    folder = Path(__file__).parent
    return {
        "STYLE": (folder / "page.css").read_text(encoding="utf-8"),
        "PLOTLY_JS": (folder / PLOTLY_BUNDLE).read_text(encoding="utf-8"),
        "TEMPLATE_JSON": page_json(page_template()),
        "SERVICE_RUN": (folder / "service_run.html").read_text(encoding="utf-8"),
        "SUPPLY_RUN": (folder / "supply_run.html").read_text(encoding="utf-8"),
    }


def run_page_html(result: Results) -> str:
    """Return the offline page of one run."""
    with building_page():
        payload = run_payload(result)
    heading = (
        "FlowSim supply chain" if payload["kind"] == "supply" else "FlowSim results"
    )
    return fill_template(
        "run.html",
        {
            **page_parts(),
            "HEADING": heading,
            "RESULT_JSON": page_json(payload),
        },
    )
