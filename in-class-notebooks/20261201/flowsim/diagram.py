"""Plotly diagrams of a model's service network or supply chain."""

from __future__ import annotations

import html
import math
from collections.abc import Callable
from numbers import Real
from typing import TYPE_CHECKING

import networkx as nx
import plotly.graph_objects as go

from flowsim.distributions import Distribution
from flowsim.plots import BLUE, GRAY, GREEN, ORANGE, PURPLE, WARNING, label, style
from flowsim.records import SUPPLIER

if TYPE_CHECKING:
    from flowsim.model import Model


def layout_layers(model: Model) -> list[list[str]]:
    """Group nodes into left-to-right layers for the network diagram.

    The exit always sits alone in the last layer. An acyclic graph places the
    other nodes by topological generations. Routes that cycle across customer
    types have none, so sources come first and each station sits at its fewest
    steps from a source. In an incomplete model, a node that no source reaches
    sits in the first layer, and an exit without routes still comes last.
    """
    others = nx.DiGraph(model.graph)
    others.remove_node(model.exit.name)
    if nx.is_directed_acyclic_graph(others):
        layers = [list(names) for names in nx.topological_generations(others)]
    else:
        sources = [node.name for node in model.nodes.values() if node.kind == "source"]
        steps = nx.multi_source_dijkstra_path_length(others, sources) if sources else {}
        steps = {name: steps.get(name, 0) for name in others}
        layers = [
            [name for name in others if steps[name] == step]
            # Unweighted path lengths are whole numbers of steps.
            for step in range(int(max(steps.values(), default=-1)) + 1)
        ]
    return [*layers, [model.exit.name]]


def edge_label(data: dict) -> str:
    """Name each type that may take a typed edge, or give an untyped share."""
    if "probability" in data:
        return "" if data["probability"] == 1 else f"{data['probability']:.0%}"
    return ", ".join(
        label(kind) if probability == 1 else f"{label(kind)} {probability:.0%}"
        for kind, probability in data["types"].items()
    )


Point = tuple[float, float]


def node_boxes(
    texts: dict[str, str],
    positions: dict[str, Point],
    scale: Point,
) -> dict[str, tuple[float, float, float, float]]:
    """Return each node's marker and label as (left, right, bottom, top) in pixels.

    A diagram node is a 30-pixel marker with its label below it in 14-point
    Georgia, about 7 pixels per character and 17 per line. Each box adds a
    4-pixel margin. scale gives the pixels per data unit along x and y.
    """
    boxes = {}
    for name, text in texts.items():
        lines = text.split("<br>")
        x = positions[name][0] * scale[0]
        y = positions[name][1] * scale[1]
        half = max(15, 3.5 * max(len(html.unescape(line)) for line in lines)) + 4
        boxes[name] = (x - half, x + half, y - 19 - 17 * len(lines), y + 19)
    return boxes


def curve_point(tail: Point, control: Point, tip: Point, share: float) -> Point:
    """Return the point a share of the way along a quadratic curve."""
    rest = 1 - share
    return (
        rest * rest * tail[0] + 2 * rest * share * control[0] + share * share * tip[0],
        rest * rest * tail[1] + 2 * rest * share * control[1] + share * share * tip[1],
    )


Box = tuple[float, float, float, float]


def crossings(points: list[Point], boxes: list[Box], scale: Point) -> int:
    """Count the boxes (in pixels) that a path through data points enters."""
    pixels = [(x * scale[0], y * scale[1]) for x, y in points]
    return sum(
        any(left < x < right and bottom < y < top for x, y in pixels)
        for left, right, bottom, top in boxes
    )


def arrow_path(
    straight: tuple[Point, Point],
    unshifted: tuple[Point, Point],
    centers: tuple[Point, Point],
    gaps: Callable[[Point, Point], Point],
    boxes: list[Box],
    scale: Point,
    bounds: Box,
) -> tuple[Point, Point, Point | None]:
    """Choose an arrow's tail, tip, and control point (None for a straight arrow).

    The straight arrow wins if it misses every other node's box (see
    node_boxes). Otherwise the arrow without its twin shift wins if it misses
    them, since the twin shifts the other way. Otherwise, like the builder
    page's arrows, it curves 60 to 220 pixels to either side at its middle,
    leaving one node's center and reaching the other's toward the control
    point. gaps maps the pixel directions out of the first node and into the
    second to the pixel distances where the arrow starts and stops. The first
    curve that misses every box wins, or the one that crosses the fewest. A
    curve must stay 10 pixels inside bounds, i.e., the (left, right, bottom,
    top) axis ranges, so that it and its label show, and the straight arrow
    stays when no curve does.
    """
    for tail, tip in (straight, unshifted):
        # A control point halfway along keeps the curve on the straight line.
        halfway = ((tail[0] + tip[0]) / 2, (tail[1] + tip[1]) / 2)
        line = [curve_point(tail, halfway, tip, step / 40) for step in range(41)]
        if not crossings(line, boxes, scale):
            return tail, tip, None

    def unit(start: Point, end: Point) -> Point:
        dx = (end[0] - start[0]) * scale[0]
        dy = (end[1] - start[1]) * scale[1]
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    def inside(point: Point) -> bool:
        left, right, bottom, top = bounds
        x_margin, y_margin = 10 / scale[0], 10 / scale[1]
        return (
            left + x_margin <= point[0] <= right - x_margin
            and bottom + y_margin <= point[1] <= top - y_margin
        )

    middle = (
        (centers[0][0] + centers[1][0]) / 2,
        (centers[0][1] + centers[1][1]) / 2,
    )
    normal = unit(centers[0], centers[1])
    normal = (-normal[1], normal[0])
    best = None
    for offset in [60, -60, 100, -100, 150, -150, 220, -220]:
        # A control point twice the offset away puts the curve's middle at it.
        control = (
            middle[0] + 2 * offset * normal[0] / scale[0],
            middle[1] + 2 * offset * normal[1] / scale[1],
        )
        out = unit(centers[0], control)
        into = unit(control, centers[1])
        leave, reach = gaps(out, into)
        tail = (
            centers[0][0] + leave * out[0] / scale[0],
            centers[0][1] + leave * out[1] / scale[1],
        )
        tip = (
            centers[1][0] - reach * into[0] / scale[0],
            centers[1][1] - reach * into[1] / scale[1],
        )
        points = [curve_point(tail, control, tip, step / 40) for step in range(41)]
        if not all(inside(point) for point in points):
            continue
        hits = crossings(points, boxes, scale)
        if best is None or hits < best[0]:
            best = (hits, tail, tip, control)
        if not hits:
            break
    if best is None:
        return (*straight, None)
    return best[1], best[2], best[3]


def draw_arrow(
    figure: go.Figure,
    tail: Point,
    tip: Point,
    control: Point | None,
) -> None:
    """Draw a straight arrow, or a curved one through a control point."""
    start = tail
    if control is not None:
        figure.add_shape(
            type="path",
            path=(
                f"M {tail[0]},{tail[1]} Q {control[0]},{control[1]} {tip[0]},{tip[1]}"
            ),
            xref="x",
            yref="y",
            line={"color": GRAY, "width": 2},
            layer="below",
        )
        # The arrowhead points along the curve's last stretch.
        start = curve_point(tail, control, tip, 0.9)
    figure.add_annotation(
        x=tip[0],
        y=tip[1],
        ax=start[0],
        ay=start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowwidth=2,
        arrowcolor=GRAY,
        text="",
    )


def node_text(node) -> str:
    """Label a diagram node with its name, and a station with its servers."""
    text = label(node.name)
    if node.kind == "station":
        suffix = "server" if node.capacity == 1 else "servers"
        text += f"<br>{node.capacity} {suffix}"
    return text


def network_diagram(model: Model, problems: dict[str, str]) -> go.Figure:
    """Draw the network's nodes and routes left to right.

    Each node named in problems is drawn in the warning color with its note on
    hover, and a title explains the marks.
    """
    graph = model.graph
    positions = {}
    layers = layout_layers(model)
    for layer, names in enumerate(layers):
        for index, name in enumerate(names):
            positions[name] = (layer, (len(names) - 1) / 2 - index)
    rows = [y for _, y in positions.values()]
    bounds = (-0.45, len(layers) - 0.55, min(rows) - 0.65, max(rows) + 0.35)
    # The figure is 800 by 400 pixels with 50 at each side, 50 above, and 80
    # below.
    scale = (700 / (bounds[1] - bounds[0]), 270 / (bounds[3] - bounds[2]))
    boxes = node_boxes(
        texts={node.name: node_text(node) for node in model.nodes.values()},
        positions=positions,
        scale=scale,
    )
    figure = go.Figure()
    for origin, target, data in graph.edges(data=True):
        x_start, y_start = positions[origin]
        x_end, y_end = positions[target]
        dx, dy = x_end - x_start, y_end - y_start
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        twin = graph.has_edge(target, origin)
        # Other edges stop short along their direction, and an edge whose
        # reverse also exists shifts to its own left, so the two stay apart. A
        # vertical pair shifts farther, past the station labels.
        shift = 0 if not twin else 0.2 if abs(dy) > abs(dx) else 0.08
        unshifted = (
            (x_start + 0.17 * ux, y_start + 0.17 * uy),
            (x_end - 0.17 * ux, y_end - 0.17 * uy),
        )
        if x_end > x_start and not twin:
            # A forward edge stops short of each node horizontally.
            straight = ((x_start + 0.17, y_start), (x_end - 0.17, y_end))
            x_label = (x_start + x_end) / 2
            y_label = (y_start + y_end) / 2 + 0.1
        else:
            straight = (
                (unshifted[0][0] - shift * uy, unshifted[0][1] + shift * ux),
                (unshifted[1][0] - shift * uy, unshifted[1][1] + shift * ux),
            )
            # A label sits just outside its arrow, on the same side.
            away = 0.1 if twin else 0
            x_label = (straight[0][0] + straight[1][0]) / 2 - away * uy
            y_label = (
                (straight[0][1] + straight[1][1]) / 2 + away * ux + (0 if twin else 0.1)
            )

        def gaps(out: Point, into: Point) -> Point:
            """Return a curve's pixel gaps, past a label it leaves or reaches."""
            # Each box's bottom is its label's, below the node's center.
            below = {
                name: positions[name][1] * scale[1] - boxes[name][2]
                for name in (origin, target)
            }
            return (
                below[origin] if out[1] < -0.5 else 0.17 * scale[0],
                below[target] if into[1] > 0.5 else 0.17 * scale[0],
            )

        # An arrow that would cross another node, e.g., one that skips a
        # layer, curves around it, 0.17 of a layer from each node's center.
        tail, tip, control = arrow_path(
            straight=straight,
            unshifted=unshifted,
            centers=(positions[origin], positions[target]),
            gaps=gaps,
            boxes=[box for name, box in boxes.items() if name not in (origin, target)],
            scale=scale,
            bounds=bounds,
        )
        draw_arrow(
            figure=figure,
            tail=tail,
            tip=tip,
            control=control,
        )
        if tail == unshifted[0] and straight != unshifted:
            x_label = (tail[0] + tip[0]) / 2
            y_label = (tail[1] + tip[1]) / 2 + 0.1
        if control is not None:
            # The label sits on the curve's middle.
            x_label, y_label = curve_point(tail, control, tip, 0.5)
        text = edge_label(data)
        if text:
            figure.add_annotation(
                x=x_label,
                y=y_label,
                text=text,
                showarrow=False,
                bgcolor="white",
            )

    for kind, color in [
        ("source", GREEN),
        ("station", BLUE),
        ("exit", ORANGE),
        ("problem", WARNING),
    ]:
        # A node with a problem moves from its kind's trace to the warning trace.
        if kind == "problem":
            nodes = [node for node in model.nodes.values() if node.name in problems]
        else:
            nodes = [
                node
                for node in model.nodes.values()
                if node.kind == kind and node.name not in problems
            ]
        figure.add_trace(
            go.Scatter(
                x=[positions[node.name][0] for node in nodes],
                y=[positions[node.name][1] for node in nodes],
                mode="markers+text",
                text=[node_text(node) for node in nodes],
                customdata=[label(problems.get(node.name, "")) for node in nodes],
                textposition="bottom center",
                marker={
                    "size": 30,
                    "color": color,
                    "line": {"color": "black", "width": 1},
                },
                hovertemplate=(
                    "%{text}<br>%{customdata}<extra></extra>"
                    if kind == "problem"
                    else "%{text}<extra></extra>"
                ),
                meta=kind,
                showlegend=False,
            )
        )
    if problems:
        figure.update_layout(
            title={
                "text": "Red nodes need routes before the model can run. Hover "
                "over one for the reason.",
                "font": {"size": 14},
            },
        )
    style(
        figure=figure,
        x_label="",
        y_label="",
    )
    figure.update_layout(
        height=400,
        xaxis={"visible": False, "range": [bounds[0], bounds[1]]},
        yaxis={"visible": False, "range": [bounds[2], bounds[3]]},
        margin={"l": 50, "r": 50, "t": 50, "b": 80},
    )
    return figure


def chain_positions(model: Model) -> dict[str, tuple[float, float]]:
    """Place a supply chain's stages in one row, with goods moving left to right.

    The outside supplier sits at x = 0 unless the top stage produces. The stock
    points follow from the top of the chain down to the stage nearest
    consumers, and demand comes last, all at y = 0. A producing stage's
    stations sit in the rows below and to the right of it: a station k steps
    along the production route from the stage is k columns to its right, and
    stations at the same step stack downward from y = -1, so the arrows of a
    branch leave the stage at different slopes. A station on no production
    route follows the line in the row y = -1.
    """
    supply = model.supply_chain
    assert supply is not None
    top = supply.stock_points[-1]
    produces = top.lead_time is None
    order = [] if produces else [SUPPLIER]
    order += [node.name for node in reversed(supply.stock_points)]
    order.append(supply.demand.name)
    positions = {name: (float(x), 0.0) for x, name in enumerate(order)}
    if produces:
        line = nx.DiGraph(model.graph)
        line.remove_node(model.exit.name)
        # A stock point enters the graph only with its first route, and an
        # unfinished model may not have one yet.
        line.add_node(top.name)
        # The return to the producing stage closes a loop, so it is left out.
        line.remove_edges_from(list(line.in_edges(top.name)))
        steps = nx.single_source_shortest_path_length(line, top.name)
        column = positions[top.name][0]
        stacked: dict[int, int] = {}
        for name, step in steps.items():
            if name == top.name:
                continue
            row = stacked.get(step, 0)
            stacked[step] = row + 1
            positions[name] = (column + step, -1.0 - row)
    # A station on no production route is drawn after the line, not hidden.
    unrouted = [
        node.name
        for node in model.nodes.values()
        if node.kind == "station" and node.name not in positions
    ]
    start = max((x for x, y in positions.values() if y < 0), default=-1.0) + 1
    for index, name in enumerate(unrouted):
        positions[name] = (start + index, -1.0)
    return positions


def lead_time_label(lead_time: object, time_unit: str) -> str:
    """Show a lead time as a number with its unit, a distribution, or 'random'."""
    if isinstance(lead_time, Distribution):
        return repr(lead_time)
    if callable(lead_time):
        return "random"
    # Stock points accept any real scalar, including NumPy numbers.
    assert isinstance(lead_time, Real)
    return f"{float(lead_time):g} {time_unit}"


def chain_diagram(model: Model) -> go.Figure:
    """Draw a supply chain's stages left to right with lead times on the arrows.

    Each arrow into a stock point carries that stock point's lead time. A
    producing stage's stations sit below it with their production route.
    """
    supply = model.supply_chain
    assert supply is not None
    positions = chain_positions(model)
    names = list(positions)
    # The chain row was placed first, in left-to-right order.
    row = [name for name in names if positions[name][1] == 0]
    shipments = list(zip(row, row[1:]))
    stages = {node.name: node for node in supply.stock_points}
    # Each production edge keeps its branch share, which labels its arrow.
    shares = {
        (origin, target): edge_label(data)
        for origin, target, data in model.graph.edges(data=True)
        if model.exit.name not in (origin, target)
    }
    production = list(shares)

    def label_gap(name: str) -> float:
        """Return the distance below a node that clears its label."""
        # A station's label has a second line for its servers.
        node = model.nodes.get(name)
        return 0.5 if node is not None and node.kind == "station" else 0.35

    texts = {
        name: node_text(model.nodes[name]) if name in model.nodes else label(name)
        for name in names
    }
    lowest = min(y for _, y in positions.values())
    # A short chain keeps the spacing of a five-column one, so arrows and their
    # offsets keep their size instead of stretching across the width.
    right = max(x for x, _ in positions.values())
    span = max(right + 0.9, 5)
    middle = right / 2
    bounds = (middle - span / 2, middle + span / 2, lowest - 0.65, 0.45)
    # The figure is 800 pixels wide, with 50 at each side, and 300 tall plus
    # 100 per row of stations, with 130 in its margins.
    scale = (700 / span, (170 + 100 * int(-lowest)) / (1.1 - lowest))
    boxes = node_boxes(
        texts=texts,
        positions=positions,
        scale=scale,
    )
    figure = go.Figure()
    for origin, target in [*shipments, *production]:
        x_start, y_start = positions[origin]
        x_end, y_end = positions[target]
        dx, dy = x_end - x_start, y_end - y_start
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        # An arrow whose reverse also exists shifts to its own left, so a
        # one-station production route draws two separate arrows.
        twin = (target, origin) in production
        shift = 0 if not twin else 0.2 if abs(dy) > abs(dx) else 0.08
        # Labels sit below the nodes, so an arrow that leaves a node downward
        # or reaches one from below stops farther away, past the label.
        leave = label_gap(origin) if uy < -0.5 else 0.17
        reach = label_gap(target) if uy > 0.5 else 0.17
        unshifted = (
            (x_start + leave * ux, y_start + leave * uy),
            (x_end - reach * ux, y_end - reach * uy),
        )
        straight = (
            (unshifted[0][0] - shift * uy, unshifted[0][1] + shift * ux),
            (unshifted[1][0] - shift * uy, unshifted[1][1] + shift * ux),
        )

        def gaps(out: Point, into: Point) -> Point:
            """Return a curve's pixel gaps, past a label it leaves or reaches."""
            return (
                (label_gap(origin) * scale[1] if out[1] < -0.5 else 0.17 * scale[0]),
                (label_gap(target) * scale[1] if into[1] > 0.5 else 0.17 * scale[0]),
            )

        # An arrow that would cross another station, e.g., a branch arrow
        # passing the station above it, curves around it.
        tail, tip, control = arrow_path(
            straight=straight,
            unshifted=unshifted,
            centers=(positions[origin], positions[target]),
            gaps=gaps,
            boxes=[box for name, box in boxes.items() if name not in (origin, target)],
            scale=scale,
            bounds=bounds,
        )
        draw_arrow(
            figure=figure,
            tail=tail,
            tip=tip,
            control=control,
        )
        if shares.get((origin, target)):
            # A branch share sits on its own arrow, 40% of the way from the
            # branching node, so it cannot be read as the share of a return
            # route or of a neighboring arrow in the fan.
            # A control point at the middle of a straight arrow keeps the
            # curve on it.
            halfway = ((tail[0] + tip[0]) / 2, (tail[1] + tip[1]) / 2)
            x_label, y_label = curve_point(
                tail,
                halfway if control is None else control,
                tip,
                0.4,
            )
            figure.add_annotation(
                x=x_label,
                y=y_label,
                text=shares[(origin, target)],
                showarrow=False,
                bgcolor="white",
            )
        if (origin, target) in shipments and target in stages:
            figure.add_annotation(
                x=(x_start + x_end) / 2,
                y=0.12,
                text=label(lead_time_label(stages[target].lead_time, model.time_unit)),
                showarrow=False,
                bgcolor="white",
            )
    kinds = [
        ("supplier", GREEN, [SUPPLIER] if SUPPLIER in positions else []),
        ("stock_point", BLUE, list(stages)),
        ("demand", ORANGE, [supply.demand.name]),
        ("station", PURPLE, [name for name in names if positions[name][1] < 0]),
    ]
    for kind, color, members in kinds:
        text = [texts[name] for name in members]
        figure.add_trace(
            go.Scatter(
                x=[positions[name][0] for name in members],
                y=[positions[name][1] for name in members],
                mode="markers+text",
                text=text,
                textposition="bottom center",
                marker={
                    "size": 30,
                    "color": color,
                    "line": {"color": "black", "width": 1},
                },
                hovertemplate="%{text}<extra></extra>",
                meta=kind,
                showlegend=False,
            )
        )
    title = "Goods move left to right."
    if any(target in stages for _, target in shipments):
        title += " Each arrow into a stock point shows its lead time."
    if any(y < 0 for _, y in positions.values()):
        title += " The stations below the top stage make its goods."
    figure.update_layout(
        title={
            "text": title,
            "font": {"size": 14},
        },
    )
    style(
        figure=figure,
        x_label="",
        y_label="",
    )
    figure.update_layout(
        height=300 + 100 * int(-lowest),
        xaxis={
            "visible": False,
            "range": [bounds[0], bounds[1]],
        },
        yaxis={"visible": False, "range": [bounds[2], bounds[3]]},
        margin={"l": 50, "r": 50, "t": 50, "b": 80},
    )
    return figure
