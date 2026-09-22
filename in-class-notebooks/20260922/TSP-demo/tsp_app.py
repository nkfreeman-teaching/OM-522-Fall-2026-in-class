"""Interactive classroom dashboard for the traveling-salesman demonstration."""

from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from mpl_toolkits.basemap import Basemap

from tsp_solver import Problem, Progress, Result, load_problem, select_states, solve


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_STATES = ("AL", "TN", "GA", "MS", "LA")
NAVY = "#17324b"
TEAL = "#087e83"
ORANGE = "#d36b3d"


@st.cache_resource
def course_problem() -> Problem:
    return load_problem(DATA_DIR)


@st.cache_data
def geography(bounds: tuple[float, float, float, float]) -> dict:
    """Build local map geometry once per selection, without network tiles."""
    south, north, west, east = bounds
    figure, axes = plt.subplots()
    try:
        basemap = Basemap(
            projection="merc",
            llcrnrlat=south,
            urcrnrlat=north,
            llcrnrlon=west,
            urcrnrlon=east,
            resolution="l",
            ax=axes,
        )
        states = basemap.drawstates(linewidth=0.5)
        coast = basemap.drawcoastlines(linewidth=0.5)
        countries = basemap.drawcountries(linewidth=0.5)
        land = [polygon.get_coords().tolist() for polygon in basemap.landpolygons]
        return {
            "width": basemap.urcrnrx,
            "height": basemap.urcrnry,
            "land": land,
            "states": [segment.tolist() for segment in states.get_segments()],
            "coast": [segment.tolist() for segment in coast.get_segments()],
            "countries": [segment.tolist() for segment in countries.get_segments()],
        }
    finally:
        plt.close(figure)


def map_bounds(problem: Problem) -> tuple[float, float, float, float]:
    south = min(problem.latitudes)
    north = max(problem.latitudes)
    west = min(problem.longitudes)
    east = max(problem.longitudes)
    latitude_padding = max(0.65, (north - south) * 0.08)
    longitude_padding = max(0.65, (east - west) * 0.08)
    return (
        south - latitude_padding,
        north + latitude_padding,
        west - longitude_padding,
        east + longitude_padding,
    )


def map_figure(problem: Problem, tour: tuple[str, ...] | None = None) -> go.Figure:
    bounds = map_bounds(problem)
    geometry = geography(bounds)
    basemap = Basemap(
        projection="merc",
        llcrnrlat=bounds[0],
        urcrnrlat=bounds[1],
        llcrnrlon=bounds[2],
        urcrnrlon=bounds[3],
        resolution=None,
    )
    figure = go.Figure()
    for polygon in geometry["land"]:
        if len(polygon) < 3:
            continue
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in polygon],
                y=[point[1] for point in polygon],
                mode="lines",
                fill="toself",
                fillcolor="#f4f0e7",
                line={"color": "#d7e0df", "width": 0.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for name, color, width in (
        ("states", "#aebbb9", 1.0),
        ("coast", "#809da4", 1.3),
        ("countries", "#809da4", 1.3),
    ):
        x_values = []
        y_values = []
        for segment in geometry[name]:
            x_values.extend([point[0] for point in segment] + [None])
            y_values.extend([point[1] for point in segment] + [None])
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": color, "width": width},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    projected_x, projected_y = basemap(problem.longitudes, problem.latitudes)
    state_centers = {}
    for state in sorted(set(problem.states)):
        members = [i for i, value in enumerate(problem.states) if value == state]
        state_centers[state] = (
            sum(projected_x[i] for i in members) / len(members),
            sum(projected_y[i] for i in members) / len(members),
        )
    figure.add_trace(
        go.Scatter(
            x=[center[0] for center in state_centers.values()],
            y=[center[1] for center in state_centers.values()],
            text=list(state_centers),
            mode="text",
            textfont={"color": "#6a8287", "size": 12, "family": "Arial Black, sans-serif"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    positions = {
        store: (float(projected_x[i]), float(projected_y[i]))
        for i, store in enumerate(problem.stores)
    }
    if tour is not None:
        closed = (*tour, tour[0])
        figure.add_trace(
            go.Scatter(
                x=[positions[store][0] for store in closed],
                y=[positions[store][1] for store in closed],
                mode="lines",
                line={"color": ORANGE, "width": 2.5},
                hoverinfo="skip",
                name="Tour",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=projected_x,
            y=projected_y,
            mode="markers",
            marker={
                "color": NAVY,
                "size": 7 if len(problem.stores) < 250 else 5,
                "line": {"color": "white", "width": 0.8},
            },
            customdata=[
                [store, city, state, latitude, longitude]
                for store, city, state, latitude, longitude in zip(
                    problem.stores,
                    problem.cities,
                    problem.states,
                    problem.latitudes,
                    problem.longitudes,
                )
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}, %{customdata[2]}"
                "<br>%{customdata[3]:.3f}°, %{customdata[4]:.3f}°<extra></extra>"
            ),
            name="Facilities",
        )
    )
    if tour is not None:
        start_x, start_y = positions[tour[0]]
        figure.add_trace(
            go.Scatter(
                x=[start_x],
                y=[start_y],
                mode="markers",
                marker={
                    "color": TEAL,
                    "size": 14,
                    "symbol": "square",
                    "line": {"color": "white", "width": 1.5},
                },
                hovertemplate=f"Start and finish: {tour[0]}<extra></extra>",
                name="Start and finish",
            )
        )
    figure.update_layout(
        height=500,
        margin={"l": 0, "r": 0, "t": 8, "b": 38},
        paper_bgcolor="white",
        plot_bgcolor="#e8f3f4",
        font={"family": "Arial, sans-serif", "color": NAVY},
        hoverlabel={"bgcolor": "white", "font_size": 13},
        legend={"orientation": "h", "y": -0.02, "yanchor": "top", "x": 0.02},
        xaxis={
            "visible": False,
            "range": [0, geometry["width"]],
            "fixedrange": False,
            "constrain": "domain",
        },
        yaxis={
            "visible": False,
            "range": [0, geometry["height"]],
            "scaleanchor": "x",
            "scaleratio": 1,
            "fixedrange": False,
            "constrain": "domain",
        },
    )
    return figure


def history_figure(result: Result) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[attempt for attempt, _ in result.history],
            y=[miles for _, miles in result.history],
            mode="lines+markers",
            line={"color": TEAL, "width": 3, "shape": "hv"},
            marker={"size": 5},
            name="Best distance",
        )
    )
    figure.add_hline(
        y=result.constructed_miles,
        line_dash="dash",
        line_color="#80909a",
        annotation_text="Nearest neighbor",
        annotation_position="top right",
    )
    figure.add_annotation(
        x=result.attempts,
        y=result.final_miles,
        text=f"{result.final_miles:,.0f} mi",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=8,
        bgcolor="white",
        font={"color": TEAL, "size": 13},
    )
    figure.update_layout(
        height=310,
        margin={"l": 18, "r": 18, "t": 12, "b": 18},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font={"family": "Arial, sans-serif", "color": NAVY, "size": 13},
        xaxis_title="Candidate tours evaluated",
        yaxis_title="Best tour distance (road miles)",
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e9eeee", tickfont={"size": 13})
    figure.update_yaxes(showgrid=True, gridcolor="#e9eeee", tickfont={"size": 13})
    return figure


def live_history_figure(history: list[tuple[int, float]]) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=[attempt for attempt, _ in history],
            y=[miles for _, miles in history],
            mode="lines+markers",
            line={"color": TEAL, "width": 2.5, "shape": "hv"},
            marker={"size": 5},
        )
    )
    figure.update_layout(
        height=220,
        margin={"l": 12, "r": 12, "t": 8, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis_title="Candidate tours evaluated",
        yaxis_title="Best distance (road miles)",
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="OM 522 · Route lab",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f7f9f8; color: #17324b; }
        [data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 3.4rem; }
        [data-testid="stAppDeployButton"] { display: none; }
        [data-testid="stSidebar"] { background: #eaf2f1; }
        h1, h2, h3 { color: #17324b; letter-spacing: -0.025em; }
        h1 { margin-bottom: .45rem; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #dde8e6; border-radius: 12px;
            padding: 0.65rem 1rem;
        }
        [data-testid="stPlotlyChart"] {
            background: white; border: 1px solid #dde8e6; border-radius: 12px;
            overflow: hidden;
        }
        @media (max-width: 850px) {
            [data-testid="stMainBlockContainer"] { padding-top: 3.4rem; }
            [data-testid="stExpandSidebarButton"] { width: 132px; }
            [data-testid="stExpandSidebarButton"]::after {
                content: "Route settings";
                white-space: nowrap;
                font-size: .83rem;
                color: #17324b;
                margin-left: .35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    source = course_problem()
    available_states = sorted(set(source.states))

    with st.sidebar:
        st.markdown("### Route settings")
        st.caption("Choose facilities by state, then build a tour.")
        with st.form("route_settings"):
            selected_states = st.multiselect(
                "Include states",
                options=available_states,
                default=sorted(DEFAULT_STATES),
            )
            with st.expander("Advanced search settings"):
                seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=0,
                    step=1,
                    help="The same selection and seed reproduce the same sampled search.",
                )
                stop_limit = st.number_input(
                    "Stop after unsuccessful tries",
                    min_value=100,
                    max_value=100_000,
                    value=10_000,
                    step=1_000,
                    help="The search stops after this many consecutive sampled tours fail to shorten the route.",
                )
            submitted = st.form_submit_button(
                label="Find a route",
                use_container_width=True,
                type="primary",
            )
        st.caption("Map lines show visit order. Tour totals use road miles.")

    st.caption("OM 522  /  TRAVELING SALESMAN PROBLEM")
    st.title("Route lab")
    st.write(
        "Nearest neighbor visits the closest unvisited facility. "
        "The search then reverses segments to shorten the tour."
    )
    if not selected_states:
        st.info("Select at least one state to build an instance.")
        return
    problem = select_states(source, selected_states)
    settings = (tuple(sorted(selected_states)), int(seed), int(stop_limit))
    result: Result | None = None
    if submitted:
        stage = st.empty()
        construction_bar = st.empty()
        streak_bar = st.empty()
        live_values = st.empty()
        live_chart = st.empty()
        last_refresh = 0.0
        live_refresh_count = 0
        construction_miles = float("inf")
        live_history: list[tuple[int, float]] = []

        def update(progress: Progress) -> None:
            nonlocal last_refresh, live_refresh_count, construction_miles
            if progress.stage == "construction":
                construction_miles = progress.best_distance
            elif not live_history:
                live_history.append((0, construction_miles))
            if (
                progress.stage == "improvement"
                and progress.best_distance < live_history[-1][1] - 1e-9
            ):
                live_history.append((progress.attempts, progress.best_distance))
            clock = monotonic()
            if progress.stage != "complete" and clock - last_refresh < 0.2:
                return
            last_refresh = clock
            if progress.stage == "construction":
                stage.info(
                    f"Constructing tours: {progress.starts_done:,} of "
                    f"{progress.total_starts:,} starting facilities tried."
                )
                construction_bar.progress(
                    progress.starts_done / progress.total_starts,
                    text="Construction progress",
                )
            elif progress.stage == "improvement":
                construction_bar.empty()
                stage.info(
                    f"Improving the tour: {progress.attempts:,} candidates evaluated, "
                    f"{progress.accepted_moves:,} improvements accepted."
                )
                streak_bar.progress(
                    progress.non_improving / progress.stop_limit,
                    text="Current streak without improvement toward the stop rule",
                )
            else:
                stage.empty()
                construction_bar.empty()
                streak_bar.empty()
            live_values.metric("Best tour so far", f"{progress.best_distance:,.1f} road miles")
            if live_history:
                live_refresh_count += 1
                live_chart.plotly_chart(
                    live_history_figure(live_history),
                    width="stretch",
                    config={"displaylogo": False},
                    key=f"live_progress_chart_{live_refresh_count}",
                )

        try:
            result = solve(
                problem,
                seed=int(seed),
                non_improving_limit=int(stop_limit),
                on_progress=update,
            )
            st.session_state["route_result"] = (settings, result)
        except Exception as error:
            stage.error(f"The route search stopped: {error}")
            return
        finally:
            construction_bar.empty()
            streak_bar.empty()
            live_values.empty()
            live_chart.empty()
    elif "route_result" in st.session_state:
        previous_settings, previous_result = st.session_state["route_result"]
        if previous_settings == settings:
            result = previous_result

    st.markdown("### Selected facilities" if result is None else "### Tour result")
    st.caption(
        f"{len(problem.stores):,} facilities across {len(selected_states)} "
        f"{'state' if len(selected_states) == 1 else 'states'}: {', '.join(sorted(selected_states))}"
    )
    if result is None:
        st.plotly_chart(
            map_figure(problem),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True},
            key="facility_map",
        )
        st.caption("Use Find a route to see the tour and how its distance changes during the search.")
        return

    distance_saved = result.constructed_miles - result.final_miles
    reduction_percent = 100 * distance_saved / result.constructed_miles
    first, second = st.columns(2)
    first.metric(
        "Improved tour",
        f"{result.final_miles:,.1f} mi",
        delta=f"{reduction_percent:.1f}% shorter",
        delta_color="normal",
    )
    second.metric("Nearest-neighbor tour", f"{result.constructed_miles:,.1f} mi")
    st.caption(
        f"The search saved {distance_saved:,.1f} road miles in "
        f"{result.elapsed_seconds:.1f} seconds, accepting {result.accepted_moves:,} "
        f"improvements from {result.attempts:,} sampled tours."
    )
    st.plotly_chart(
        map_figure(problem, result.final_tour),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
        key="route_map",
    )
    st.caption(
        "The route visits each facility once and returns to its start. "
        "Map lines show visit order, while distances use road miles. "
        "The sampled search does not prove optimality."
    )
    st.caption("Drag to pan, use the map controls to zoom, and hover over a facility for details.")
    st.markdown("### How the distance changed")
    st.plotly_chart(
        history_figure(result),
        width="stretch",
        config={"displaylogo": False},
        key="route_history",
    )
    with st.expander("View the ordered facility visits"):
        details = {
            store: (city, state)
            for store, city, state in zip(
                problem.stores,
                problem.cities,
                problem.states,
            )
        }
        stops = [
            {
                "Visit": number,
                "Facility": store,
                "City": details[store][0],
                "State": details[store][1],
            }
            for number, store in enumerate(result.final_tour, start=1)
        ]
        st.dataframe(stops, hide_index=True)


if __name__ == "__main__":
    main()
