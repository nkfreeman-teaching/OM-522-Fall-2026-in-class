import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import polars as pl

    from tsp_utils import calculate_tour_distance, plot_locations

    return Path, calculate_tour_distance, mo, pl, plot_locations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OM 522 Traveling Salesman Demo

    This notebook uses 439 southeastern facility locations to demonstrate how a
    tour is represented, plotted, and measured. The route lines close back to
    their starting locations automatically.
    """)
    return


@app.cell
def _(Path, pl):
    project_root = Path(__file__).parent
    locations = pl.read_parquet(project_root / "data" / "store_locations.parquet")
    road_distances = pl.read_parquet(
        project_root / "data" / "road_distances.parquet"
    )
    return locations, road_distances


@app.cell(hide_code=True)
def _(locations, mo, road_distances):
    mo.md(f"""
    The location table has **{locations.height:,} rows** and the distance table
    has **{road_distances.height:,} rows**. Each ordered pair is present,
    including a zero-mile diagonal entry for every store.
    """)
    return


@app.cell
def _(locations):
    locations.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All locations

    Calling `plot_locations` without a tour shows the full instance on an
    offline map. The optional `figsize=(width, height)` argument controls the
    figure dimensions in inches.
    """)
    return


@app.cell
def _(locations, plot_locations):
    _all_locations_figure, _all_locations_axes = plot_locations(
        locations=locations,
        figsize=(12, 8),
    )
    _all_locations_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One tour

    A flat list represents one tour. The final return to `L424` is added by both
    utility functions.
    """)
    return


@app.cell
def _():
    single_tour = ["L424", "L263", "L262", "L278", "L277"]
    return (single_tour,)


@app.cell
def _(locations, pl, plot_locations, single_tour):
    single_tour_locations = locations.filter(
        pl.col("store").is_in(single_tour)
    )
    _single_figure, _single_axes = plot_locations(
        locations=single_tour_locations,
        tours=single_tour,
    )
    _single_figure
    return


@app.cell
def _(calculate_tour_distance, mo, road_distances, single_tour):
    single_tour_miles = calculate_tour_distance(
        tour=single_tour,
        road_distances=road_distances,
    )
    mo.md(f"**Closed-tour distance:** {single_tour_miles:,.1f} miles")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple tours

    A nested list represents one tour per vehicle. Each sublist closes back to
    its own starting location.
    """)
    return


@app.cell
def _():
    vehicle_tours = [
        ["L355", "L356", "L337", "L352", "L339"],
        ["L424", "L263", "L262", "L278", "L277"],
    ]
    return (vehicle_tours,)


@app.cell
def _(locations, pl, plot_locations, vehicle_tours):
    vehicle_store_ids = [
        store_id
        for tour in vehicle_tours
        for store_id in tour
    ]
    vehicle_locations = locations.filter(
        pl.col("store").is_in(vehicle_store_ids)
    )
    _vehicle_figure, _vehicle_axes = plot_locations(
        locations=vehicle_locations,
        tours=vehicle_tours,
    )
    _vehicle_figure
    return


@app.cell
def _(calculate_tour_distance, mo, road_distances, vehicle_tours):
    vehicle_distances = [
        calculate_tour_distance(
            tour=tour,
            road_distances=road_distances,
        )
        for tour in vehicle_tours
    ]
    mo.md(
        "\n".join(
            f"- Vehicle {number}: **{distance:,.1f} miles**"
            for number, distance in enumerate(vehicle_distances, start=1)
        )
    )
    return

if __name__ == "__main__":
    app.run()
