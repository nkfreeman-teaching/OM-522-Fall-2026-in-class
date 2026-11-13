"""Interval tables, plots, pages, and exports shared by experiments and comparisons.

Experiment and Comparison call these functions, so the same request gives the
same rows, figure, page, and files in both. A request names a metric and, for
a station or stock-point metric, optionally one location. Without a location,
a station metric covers every station and a stock-point metric covers every
stock point.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import t

from flowsim import __version__
from flowsim.catalog import (
    METRICS,
    SCOPES,
    Inapplicable,
    inapplicable,
    metric_error,
    metric_units,
    scope_name,
    unknown,
)
from flowsim.plots import (
    BLUE,
    STAGE_COLORS,
    base_layout,
    building_page,
    dashed_line,
    fill_template,
    label,
    page_json,
    page_parts,
    run_payload,
)
from flowsim.records import interval_frame
from flowsim.validation import number

if TYPE_CHECKING:
    from flowsim.results import Experiment


def confidence_level(value: float) -> float:
    level = number(
        value=value,
        label="Confidence level",
        positive=True,
    )
    if level >= 1:
        raise ValueError("Confidence level must be below one.")
    return level


def interval_estimate(values: np.ndarray, confidence: float) -> dict:
    """Estimate uncertainty across independent finite replication values."""
    values = values[np.isfinite(values)]
    count = len(values)
    mean = float(values.mean()) if count else None
    lower = upper = None
    # Estimates have the metric's units. Standard error divides by sqrt(runs).
    if mean is not None and count >= 2:
        half_width = float(
            t.ppf((1 + confidence) / 2, df=count - 1)
            * values.std(ddof=1)
            / np.sqrt(count)
        )
        lower = mean - half_width
        upper = mean + half_width
    return {
        "observed_runs": count,
        "mean": mean,
        "lower": lower,
        "upper": upper,
    }


def location_column(metric: str) -> str:
    """Name the interval table's location column for a metric.

    Stock-point metrics use a stock_point column. Every other metric keeps the
    station column, which is null for a system metric.
    """
    return SCOPES[METRICS[metric].scope].column


def check_request(metric: str, station: str | None, stock_point: str | None) -> None:
    """Reject an unknown metric or a location argument that its scope lacks."""
    if metric not in METRICS:
        raise metric_error(metric)
    scope = METRICS[metric].scope
    kind = scope_name(metric)
    if station is not None and scope != "station":
        raise ValueError(
            f"{metric!r} is a {kind} metric, so it takes no station=. Station "
            "metrics such as 'mean_wait' take station=."
        )
    if stock_point is not None and scope != "stock_point":
        raise ValueError(
            f"{metric!r} is a {kind} metric, so it takes no stock_point=. "
            "Stock-point metrics such as 'fill_rate' take stock_point=."
        )


def locations(
    experiment: Experiment,
    metric: str,
    station: str | None,
    stock_point: str | None,
) -> list[str | None]:
    """Return the locations a checked request covers in one experiment.

    A system metric covers one location, None. A station or stock-point metric
    covers the named location or, when none is named, every one in definition
    order. A metric that the model cannot measure raises Inapplicable.
    """
    # A run records its capacities, and only a comparison's shared settings
    # hold None.
    capacities = experiment.settings.capacities
    assert capacities is not None
    stations = list(capacities)
    stock_points = experiment.stock_point_names
    # The same test Results.report uses, so both agree on what a supply chain is.
    error = inapplicable(
        metric,
        stations=bool(stations),
        stock_points=bool(stock_points),
        supply=experiment.runs[0].supply is not None,
    )
    if error is not None:
        raise error
    scope = METRICS[metric].scope
    if scope == "system":
        return [None]
    if scope == "station":
        names, chosen, kind = stations, station, "station"
    else:
        names, chosen, kind = stock_points, stock_point, "stock point"
    if chosen is None:
        return [*names]
    if chosen not in names:
        raise unknown(
            kind=kind,
            name=chosen,
            choices=names,
        )
    return [chosen]


def scenario_locations(
    experiments: Mapping[str, Experiment],
    metric: str,
    station: str | None,
    stock_point: str | None,
) -> dict[str, list[str | None]]:
    """Return the locations a checked request covers in each scenario.

    A named location must exist in every scenario. Without one, a scenario
    whose model cannot measure the metric adds no station or stock-point rows
    and a row of missing or zero system values, so a comparison of a supply
    chain with and without production stations still reports the jobs of the
    scenarios that have them. The request fails only when no scenario applies.
    """
    found = {}
    skipped = []
    for name, experiment in experiments.items():
        try:
            found[name] = locations(
                experiment=experiment,
                metric=metric,
                station=station,
                stock_point=stock_point,
            )
        except Inapplicable as error:
            if station is not None or stock_point is not None:
                raise ValueError(f"Scenario {name!r}: {error}") from None
            skipped.append(error)
            found[name] = [None] if METRICS[metric].scope == "system" else []
        except ValueError as error:
            raise ValueError(f"Scenario {name!r}: {error}") from None
    if len(skipped) == len(experiments):
        raise skipped[0]
    return found


def run_values(
    experiment: Experiment,
    metric: str,
    names: list[str | None],
) -> dict[str | None, np.ndarray]:
    """Return each location's values, one per replication in replication order."""
    pooled = getattr(experiment, METRICS[metric].table)
    column = location_column(metric)
    expected = list(range(1, len(experiment.runs) + 1))
    values = {}
    for name in names:
        rows = pooled if name is None else pooled.filter(pl.col(column) == name)
        rows = rows.sort("replication")
        if rows.get_column("replication").to_list() != expected:
            raise ValueError("Some replications are missing or duplicated.")
        values[name] = np.asarray(
            rows.get_column(metric).to_numpy(),
            dtype=float,
        )
    return values


def scenario_values(
    experiments: Mapping[str, Experiment],
    metric: str,
    found: Mapping[str, list[str | None]],
) -> dict[str, dict[str | None, np.ndarray]]:
    """Return each scenario's run values at the locations found for it."""
    values = {}
    for name, experiment in experiments.items():
        try:
            values[name] = run_values(
                experiment=experiment,
                metric=metric,
                names=found[name],
            )
        except ValueError as error:
            raise ValueError(f"Scenario {name!r}: {error}") from None
    return values


def estimate_rows(
    values: Mapping[str | None, np.ndarray],
    metric: str,
    level: float,
) -> list[dict]:
    """Estimate one interval per location from its run values."""
    return [
        {
            "metric": metric,
            location_column(metric): name,
            # run_values holds one value per replication.
            "runs": len(runs),
            **interval_estimate(runs, level),
            "confidence": level,
        }
        for name, runs in values.items()
    ]


def experiment_intervals(
    experiment: Experiment,
    metric: str,
    station: str | None,
    stock_point: str | None,
    confidence: float,
) -> pl.DataFrame:
    level = confidence_level(confidence)
    check_request(
        metric=metric,
        station=station,
        stock_point=stock_point,
    )
    names = locations(
        experiment=experiment,
        metric=metric,
        station=station,
        stock_point=stock_point,
    )
    return interval_frame(
        rows=estimate_rows(
            values=run_values(
                experiment=experiment,
                metric=metric,
                names=names,
            ),
            metric=metric,
            level=level,
        ),
        keys=["metric", location_column(metric)],
    )


def scenario_intervals(
    values: Mapping[str, Mapping[str | None, np.ndarray]],
    metric: str,
    level: float,
) -> pl.DataFrame:
    """Estimate each scenario's intervals from scenario_values."""
    return interval_frame(
        rows=[
            {"scenario": name, **row}
            for name, runs in values.items()
            for row in estimate_rows(
                values=runs,
                metric=metric,
                level=level,
            )
        ],
        keys=["scenario", "metric", location_column(metric)],
    )


def comparison_intervals(
    experiments: Mapping[str, Experiment],
    metric: str,
    station: str | None,
    stock_point: str | None,
    confidence: float,
) -> pl.DataFrame:
    level = confidence_level(confidence)
    check_request(
        metric=metric,
        station=station,
        stock_point=stock_point,
    )
    found = scenario_locations(
        experiments=experiments,
        metric=metric,
        station=station,
        stock_point=stock_point,
    )
    return scenario_intervals(
        values=scenario_values(
            experiments=experiments,
            metric=metric,
            found=found,
        ),
        metric=metric,
        level=level,
    )


def paired_differences(
    experiments: Mapping[str, Experiment],
    values: Mapping[str, Mapping[str | None, np.ndarray]],
    reference: str,
    metric: str,
    level: float,
) -> pl.DataFrame:
    """Estimate each scenario minus the baseline within matched replications.

    The values come from scenario_values. Each scenario is paired with the
    baseline at every location they share. A missing value in either run
    excludes that pair, not other observations.
    """
    column = location_column(metric)
    seeds = [run.settings.seed for run in experiments[reference].runs]
    baseline = values[reference]
    rows = []
    for name, experiment in experiments.items():
        if name == reference:
            continue
        if [run.settings.seed for run in experiment.runs] != seeds:
            raise ValueError("Paired scenarios need identical replication seeds.")
        rows += [
            {
                "scenario": name,
                "baseline": reference,
                "metric": metric,
                column: place,
                "runs": len(experiment.runs),
                **interval_estimate(values[name][place] - baseline[place], level),
                "confidence": level,
            }
            for place in baseline
            if place in values[name]
        ]
    return interval_frame(
        rows=rows,
        keys=["scenario", "baseline", "metric", column],
    )


def comparison_differences(
    experiments: Mapping[str, Experiment],
    reference: str,
    metric: str,
    station: str | None,
    stock_point: str | None,
    confidence: float,
) -> pl.DataFrame:
    level = confidence_level(confidence)
    check_request(
        metric=metric,
        station=station,
        stock_point=stock_point,
    )
    found = scenario_locations(
        experiments=experiments,
        metric=metric,
        station=station,
        stock_point=stock_point,
    )
    return paired_differences(
        experiments=experiments,
        values=scenario_values(
            experiments=experiments,
            metric=metric,
            found=found,
        ),
        reference=reference,
        metric=metric,
        level=level,
    )


def metric_axis_parts(
    metric: str,
    place: str | None,
    time_unit: str,
) -> tuple[str, str | None]:
    """Return a metric's axis name with its location, and its units.

    A fraction has no units, so its second part is None.
    """
    text = METRICS[metric].label
    if place is not None:
        text += f" at {place}"
    if METRICS[metric].kind == "fraction":
        return text, None
    return text, f"({metric_units(metric, time_unit)})"


def metric_axis_label(metric: str, place: str | None, time_unit: str) -> str:
    """Name a metric's value axis, with its location and units."""
    name, units = metric_axis_parts(
        metric=metric,
        place=place,
        time_unit=time_unit,
    )
    return name if units is None else f"{name} {units}"


def interval_figure(
    table: pl.DataFrame,
    metric: str,
    time_unit: str,
    *,
    differences: bool = False,
) -> dict:
    """Return each interval-table row as a mean with its confidence limits.

    Rows are scenarios in a comparison and locations in an experiment. When a
    comparison covers several locations, each row names both, and each location
    has its own color. The figure is plain data, as in flowsim.plots.
    """
    column = location_column(metric)
    rows = table.to_dicts()
    places = list(dict.fromkeys(row[column] for row in rows))
    several = len(places) > 1
    scenarios = "scenario" in table.columns

    def row_label(row: dict) -> str:
        place = "System" if row[column] is None else label(row[column])
        if not scenarios:
            return place
        return label(row["scenario"]) + (f"<br>{place}" if several else "")

    data = []
    for index, place in enumerate(places):
        chosen = [row for row in rows if row[column] == place]
        where = "" if place is None else f"<br>{label(place)}"
        color = (
            STAGE_COLORS[index % len(STAGE_COLORS)] if scenarios and several else BLUE
        )
        trace = {
            "type": "scatter",
            "x": [row["mean"] for row in chosen],
            "y": [row_label(row) for row in chosen],
            "name": "System" if place is None else label(place),
            "mode": "markers",
            "marker": {
                "size": 12,
                "color": color,
                "line": {"color": "black", "width": 1},
            },
            "error_x": {
                "type": "data",
                "array": [
                    None if row["upper"] is None else row["upper"] - row["mean"]
                    for row in chosen
                ],
                "thickness": 2.5,
                "width": 5,
            },
            "showlegend": scenarios and several,
            "hovertemplate": (
                f"%{{y}}{'' if scenarios and several else where}"
                "<br>Estimate: %{x:.3f}<extra></extra>"
            ),
        }
        # A system row has no location, and Plotly writes no meta for None.
        if place is not None:
            trace["meta"] = place
        data.append(trace)
    fraction = METRICS[metric].kind == "fraction"
    axis_label = metric_axis_label(
        metric=metric,
        place=places[0] if len(places) == 1 else None,
        time_unit=time_unit,
    )
    if differences:
        axis_label = "Difference in " + axis_label[0].lower() + axis_label[1:]
    if scenarios:
        y_label = "Scenario"
    elif places and places[0] is not None:
        y_label = SCOPES[METRICS[metric].scope].heading
    else:
        y_label = ""
    layout = base_layout(
        x_label=axis_label,
        y_label=y_label,
    )
    layout["height"] = max(240, 150 + 50 * len(rows))
    layout["margin"] = {"l": 155, "r": 35, "t": 35, "b": 80}
    # List rows top to bottom in the order they were defined.
    layout["yaxis"].update(autorange="reversed", automargin=True)
    if fraction:
        layout["xaxis"]["tickformat"] = ".0%"
    figure = {"data": data, "layout": layout}
    if differences:
        dashed_line(figure, 0)
    return figure


def window_text(
    metric: str,
    *,
    duration: float,
    warmup: float,
    drain: bool,
    time_unit: str,
) -> str:
    """Describe the measurement window behind a page's intervals."""
    if METRICS[metric].scope == "stock_point":
        return (
            f"Stock points review each period until {duration:g} {time_unit}. "
            f"Measurement begins at {warmup:g} {time_unit}."
        )
    return (
        f"Operating hours end at {duration:g} {time_unit}. "
        f"Measurement begins at {warmup:g} {time_unit}. "
        + (
            "Arrivals stopped at closing, and the run continued until every "
            "customer who was not rejected had completed service or abandoned."
            if drain
            else "Service stops at closing. Waiting and flow averages include "
            "observed outcomes."
        )
    )


def replication_figure(
    values: Mapping[str | None, list[float | None]],
    scenarios: list[str | None],
    metric: str,
    place: str | None,
    time_unit: str,
) -> dict:
    """Return each replication's value, one marker series per scenario.

    The values map each scenario (None for an experiment) to its runs in
    replication order, and a missing value leaves a gap. A scenario without
    the location has no values. Each point's customdata holds its scenario's
    position in scenarios, the full list, and its replication, both counted
    from 0, which the dashboard uses to open that run. The position also sets
    the color, so a scenario keeps its color at every location.
    """
    data = []
    for scenario, runs in values.items():
        index = scenarios.index(scenario)
        data.append(
            {
                "type": "scatter",
                "x": list(range(1, len(runs) + 1)),
                "y": runs,
                "name": "Runs" if scenario is None else label(scenario),
                "mode": "markers",
                "marker": {
                    "size": 10,
                    "color": STAGE_COLORS[index % len(STAGE_COLORS)],
                    "line": {"color": "black", "width": 1},
                },
                "customdata": [[index, run] for run in range(len(runs))],
                "showlegend": scenario is not None,
                "hovertemplate": (
                    ("" if scenario is None else f"{label(scenario)}<br>")
                    + "Replication %{x}<br>Value: %{y:.3f}<extra></extra>"
                ),
            }
        )
    name, units = metric_axis_parts(
        metric=metric,
        place=place,
        time_unit=time_unit,
    )
    layout = base_layout(
        x_label="Replication",
        y_label=name,
    )
    layout["height"] = 380
    layout["margin"] = {"l": 95, "r": 30, "t": 60, "b": 70}
    if units is not None:
        # The units take a second line, so a long title fits the chart's
        # height. base_layout escapes its labels, so the break is added here.
        layout["yaxis"]["title"]["text"] = f"{label(name)}<br>{label(units)}"
    layout["xaxis"]["rangemode"] = "normal"
    # Plotly labels fractional replications (1.2, 1.4) on a short axis.
    if max((len(runs) for runs in values.values()), default=0) <= 10:
        layout["xaxis"]["dtick"] = 1
    plotted = [value for runs in values.values() for value in runs if value is not None]
    if plotted:
        # A "tozero" range puts a zero value on the plot's lower edge, where
        # Firefox misses a click on its point, so the range pads both ends by
        # 5% of the span from zero to the furthest value (1 if all are zero).
        # Autoscale and double-click return to the same range.
        low = min(0.0, *plotted)
        high = max(0.0, *plotted)
        if high == low:
            high = low + 1.0
        pad = 0.05 * (high - low)
        layout["yaxis"].update(
            range=[low - pad, high + pad],
            autorangeoptions={"minallowed": low - pad, "maxallowed": high + pad},
        )
    if METRICS[metric].kind == "fraction":
        layout["yaxis"]["tickformat"] = ".0%"
    return {"data": data, "layout": layout}


# Metric menu headings, in catalog order.
_GROUPS = [(scope, facts.menu) for scope, facts in SCOPES.items()]


def scenario_runs(
    values: Mapping[str, Mapping[str | None, np.ndarray]],
    place: str | None,
) -> dict[str | None, list[float | None]]:
    """Return each scenario's run values at one location, None where missing.

    The values come from scenario_values. A scenario without the location is
    left out.
    """
    return {
        name: [None if math.isnan(value) else float(value) for value in runs[place]]
        for name, runs in values.items()
        if place in runs
    }


def metric_entry(
    experiments: Mapping[str, Experiment],
    metric: str,
    comparing: bool,
    level: float,
) -> dict | None:
    """Return one metric's dashboard views, one per location, or None.

    None means no scenario's model can measure the metric. Each view holds
    the interval table and figure, the per-replication figure, and, in a
    comparison, the paired differences from each scenario as the baseline.
    The run values are read once and feed all of them.
    """
    try:
        found = scenario_locations(
            experiments=experiments,
            metric=metric,
            station=None,
            stock_point=None,
        )
    except Inapplicable:
        return None
    places = list(dict.fromkeys(place for chosen in found.values() for place in chosen))
    first = next(iter(experiments.values()))
    settings = first.settings
    column = location_column(metric)
    if comparing:
        values = scenario_values(
            experiments=experiments,
            metric=metric,
            found=found,
        )
    else:
        # An experiment's errors name no scenario, as experiment.intervals().
        values = {
            "": run_values(
                experiment=first,
                metric=metric,
                names=found[""],
            )
        }
    table = scenario_intervals(
        values=values,
        metric=metric,
        level=level,
    )
    if comparing:
        changes = {
            baseline: paired_differences(
                experiments=experiments,
                values=values,
                reference=baseline,
                metric=metric,
                level=level,
            )
            for baseline in experiments
        }
    else:
        # An experiment's table has no scenario column, as experiment.intervals().
        table = table.drop("scenario")
        changes = {}
    views = []
    for place in places:
        here = pl.col(column).is_null() if place is None else pl.col(column) == place
        chosen = table.filter(here)
        runs = scenario_runs(
            values=values,
            place=place,
        )
        paired = {baseline: rows.filter(here) for baseline, rows in changes.items()}
        views.append(
            {
                "title": METRICS[metric].label
                + ("" if place is None else f" at {place}"),
                "intervals": chosen.to_dicts(),
                "means": interval_figure(
                    table=chosen,
                    metric=metric,
                    time_unit=settings.time_unit,
                ),
                "runs": replication_figure(
                    values=runs if comparing else {None: runs[""]},
                    scenarios=list(experiments) if comparing else [None],
                    metric=metric,
                    place=place,
                    time_unit=settings.time_unit,
                ),
                "changes": {
                    baseline: None
                    if rows.is_empty()
                    else {
                        "intervals": rows.to_dicts(),
                        "figure": interval_figure(
                            table=rows,
                            metric=metric,
                            time_unit=settings.time_unit,
                            differences=True,
                        ),
                    }
                    for baseline, rows in paired.items()
                },
            }
        )
    kind = METRICS[metric].kind
    return {
        "name": metric,
        "label": METRICS[metric].label,
        "scope": METRICS[metric].scope,
        "location_column": column,
        "fraction": kind == "fraction",
        "units": "percent"
        if kind == "fraction"
        else metric_units(metric, settings.time_unit),
        "meaning": METRICS[metric].meaning,
        "window": window_text(
            metric,
            duration=settings.duration,
            warmup=settings.warmup,
            drain=settings.drain,
            time_unit=settings.time_unit,
        ),
        "locations": places,
        "views": views,
    }


def dashboard(
    experiments: Mapping[str, Experiment],
    *,
    comparing: bool,
    metric: str,
    station: str | None,
    stock_point: str | None,
    confidence: float,
    reference: str | None,
) -> str:
    """Build the offline page of every applicable metric and every kept run.

    For each metric that applies to some scenario, in catalog order, the page
    holds the views from metric_entry. It embeds Plotly once and each run's
    records when every run kept them, in a script element per run that the
    page parses only when the run opens. The caller has checked the request,
    which names the metric, location, and baseline shown first. An
    experiment's single entry has the key "".
    """
    # PageExperiment subclasses Experiment, whose methods call this module,
    # so results.py imports reporting.py and this one class comes in late.
    from flowsim.results import PageExperiment

    level = confidence_level(confidence)
    experiments = {
        name: PageExperiment(experiment.runs)
        for name, experiment in experiments.items()
    }
    # Run views built here hold their record columns as typed arrays.
    with building_page():
        entries = [
            entry
            for name in METRICS
            if (
                entry := metric_entry(
                    experiments=experiments,
                    metric=name,
                    comparing=comparing,
                    level=level,
                )
            )
            is not None
        ]
        records = all(
            run.records_kept
            for experiment in experiments.values()
            for run in experiment.runs
        )
        runs = []
        run_data = []
        for scenario, (name, experiment) in enumerate(experiments.items()):
            prefix = f"{name}, replication" if comparing else "Replication"
            runs.append(
                [
                    {"label": f"{prefix} {index} (seed {run.settings.seed})"}
                    for index, run in enumerate(experiment.runs, start=1)
                ]
            )
            if records:
                run_data += [
                    f'<script id="run-{scenario}-{index}" type="application/json">'
                    f"{page_json(run_payload(run))}</script>"
                    for index, run in enumerate(experiment.runs)
                ]
        payload = {
            "scenarios": list(experiments) if comparing else [None],
            "replications": len(next(iter(experiments.values())).runs),
            "confidence_text": f"{level * 100:g}%",
            "groups": _GROUPS,
            "metrics": entries,
            "initial": {
                "metric": metric,
                "location": {"station": station, "stock_point": stock_point}.get(
                    METRICS[metric].scope
                ),
                "baseline": reference,
            },
            "records": records,
            "runs": runs,
        }
    return fill_template(
        "dashboard.html",
        {
            **page_parts(),
            "HEADING": "Scenario comparison" if comparing else "Replication intervals",
            "DASHBOARD_JSON": page_json(payload),
            "RUN_DATA": "\n".join(run_data),
        },
    )


def write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        data=text,
        encoding="utf-8",
    )


# Every file name a FlowSim export can write. An export removes the ones it did
# not write, so a folder never mixes files from two different exports.
EXPORT_FILE_NAMES = frozenset(
    {
        "customers.csv",
        "visits.csv",
        "states.csv",
        "stations.csv",
        "summary.csv",
        "activities.csv",
        "failures.csv",
        "inventory.csv",
        "shipments.csv",
        "stock_points.csv",
        "metadata.json",
        "runs.csv",
        "intervals.csv",
        "differences.csv",
    }
)


def require_flowsim_folder(destination: Path) -> None:
    """Refuse a folder that holds export-named files FlowSim did not write.

    A folder counts as FlowSim's when its metadata.json parses as a JSON object
    with a "flowsim_version" key. A missing folder or one without export-named
    files is fine.
    """
    if not destination.is_dir():
        return
    clashing = sorted(
        name for name in EXPORT_FILE_NAMES if (destination / name).is_file()
    )
    if not clashing:
        return
    try:
        metadata = json.loads((destination / "metadata.json").read_text())
    except (OSError, ValueError):
        metadata = None
    if isinstance(metadata, dict) and "flowsim_version" in metadata:
        return
    raise ValueError(
        f"{destination} holds {', '.join(clashing)}, but no FlowSim "
        "metadata.json, so FlowSim cannot tell whether it wrote them and will "
        "not replace or delete them. Export into an empty folder or one an "
        "earlier FlowSim export wrote."
    )


def remove_stale_exports(destination: Path, written: set[str]) -> None:
    """Delete FlowSim export files in destination that this export did not write."""
    for name in EXPORT_FILE_NAMES - written:
        stale = destination / name
        if stale.is_file():
            stale.unlink()


def _write_metadata(path: str | Path, data: dict) -> None:
    """Write export metadata as JSON, stamped with the FlowSim version.

    The file is written beside its destination and then renamed over it, so an
    interrupted write keeps the earlier metadata, which marks the folder as
    FlowSim's for require_flowsim_folder.
    """
    destination = Path(path)
    partial = destination.with_name(destination.name + ".partial")
    write_text(partial, json.dumps({"flowsim_version": __version__, **data}, indent=2))
    partial.replace(destination)


def export_tables(
    directory: str | Path,
    *,
    summary: pl.DataFrame,
    stations: pl.DataFrame,
    stock_points: pl.DataFrame | None,
    intervals: pl.DataFrame,
    differences: pl.DataFrame | None,
    metadata: dict,
) -> Path:
    """Write pooled run tables, intervals, and metadata, removing stale files."""
    tables = {
        "runs.csv": summary,
        "stations.csv": stations,
        "stock_points.csv": stock_points,
        "intervals.csv": intervals,
        "differences.csv": differences,
    }
    return write_export(
        directory=directory,
        tables={name: table for name, table in tables.items() if table is not None},
        metadata=metadata,
    )


def write_export(
    directory: str | Path,
    tables: dict[str, pl.DataFrame],
    metadata: dict,
) -> Path:
    """Write metadata.json and each table to its file name, removing stale files."""
    destination = Path(directory)
    require_flowsim_folder(destination)
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Metadata goes first, so a request that cannot be written replaces no
    # tables from an earlier export, and an export interrupted while writing
    # tables leaves a folder that a retry still recognizes as FlowSim's.
    _write_metadata(destination / "metadata.json", metadata)
    for name, table in tables.items():
        table.write_csv(destination / name)
    remove_stale_exports(destination, set(tables) | {"metadata.json"})
    return destination


def request_metadata(
    experiment: Experiment,
    metric: str,
    station: str | None,
    stock_point: str | None,
    confidence: float,
) -> dict:
    """Describe an export's request, settings, and replication seeds."""
    settings = experiment.settings
    return {
        "metric": metric,
        "station": station,
        "stock_point": stock_point,
        # The validated level is a Python float, as JSON needs, for any real input.
        "confidence": confidence_level(confidence),
        "time_unit": settings.time_unit,
        "duration": settings.duration,
        "warmup": settings.warmup,
        "drain": settings.drain,
        "replication_seeds": [run.settings.seed for run in experiment.runs],
    }
