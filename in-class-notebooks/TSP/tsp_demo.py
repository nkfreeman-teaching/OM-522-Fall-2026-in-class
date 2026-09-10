import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import numpy as np
    import polars as pl
    from tqdm.auto import tqdm

    from tsp_utils import calculate_tour_distance, plot_locations

    return Path, calculate_tour_distance, mo, np, pl, plot_locations, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OM 522 Traveling Salesman Demo

    This notebook uses 127 facility locations in Alabama, Georgia, and
    Mississippi to demonstrate how a tour is constructed, plotted, and
    measured. The route closes back to its starting location automatically.
    """)
    return


@app.cell
def _(Path, pl):
    project_root = Path(__file__).parent
    locations = pl.read_parquet(project_root / "data" / "store_locations.parquet")
    road_distances = pl.read_parquet(
        project_root / "data" / "road_distances.parquet"
    )
    locations = locations.filter(
        pl.col("state").is_in(["AL", "GA", "MS"]),
    )

    locations.head()

    N = set(locations["store"].to_list())

    # Restrict the distance table before repeatedly searching it in the loop.
    road_distances = road_distances.filter(
        pl.col("store1").is_in(N),
        pl.col("store2").is_in(N),
    )
    return N, locations, road_distances


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
        figsize=(6, 4),
    )
    _all_locations_figure
    return


@app.cell
def _(
    N,
    calculate_tour_distance,
    locations,
    np,
    pl,
    plot_locations,
    road_distances,
    tqdm,
):
    shortest_tour_length = np.inf
    best_tour = None

    # Sorting makes the all-starts search reproducible across Python processes.
    for seed_location in tqdm(sorted(N)):
        tour = []
        U = set(N)  # Copy N because locations are removed from U below.
        origin = seed_location

        tour.append(origin)
        U.remove(origin)

        while len(U) > 0:
            # Greedily select the closest unvisited destination. Store ID
            # provides a deterministic secondary key for equal distances.
            destination = (
                road_distances
                .filter(
                    pl.col("store1") == origin,
                    pl.col("store2").is_in(U),
                )
                .sort(
                    by=["distance_miles", "store2"],
                    descending=False,
                )
                .item(
                    row=0,
                    column="store2",
                )
            )
            tour.append(destination)
            U.remove(destination)
            origin = destination

        # calculate_tour_distance includes the final return to the seed.
        tour_length = calculate_tour_distance(
            tour=tour,
            road_distances=road_distances,
        )
        if tour_length < shortest_tour_length:
            shortest_tour_length = tour_length
            best_tour = list(tour)
            print(
                f" - Happy days!!! Location {seed_location} yields a better tour!!!"
            )

    plot_locations(
        locations=locations,
        tours=best_tour,
        figsize=(6, 4),
    )
    return


if __name__ == "__main__":
    app.run()
