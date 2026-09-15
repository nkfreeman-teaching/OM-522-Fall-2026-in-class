import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from tqdm.auto import tqdm

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"

    from tsp_utils import calculate_tour_distance, plot_locations

    return (
        Path,
        calculate_tour_distance,
        mo,
        np,
        pl,
        plot_locations,
        plt,
        sns,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OM 522 Traveling Salesman Demo

    This notebook uses 86 facility locations in Georgia to demonstrate a
    two-phase heuristic for the traveling salesperson problem. Nearest Neighbor
    constructs a complete tour, then pairwise interchange searches for shorter
    tours. The route closes back to its starting location automatically.
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
        pl.col("state") == "GA",
    )

    locations.head()

    N = set(locations.get_column("store").to_list())

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
        figsize=(8, 5),
    )
    _all_locations_axes.set_title("")
    _all_locations_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 1: Construct a starting tour

    The multistart Nearest Neighbor procedure builds one tour from every
    possible starting location and retains the shortest result. This best
    constructed tour becomes the incumbent for the improvement phase.
    """)
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

        while U:
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
            print(f" - Location {seed_location} yields a shorter tour.")

    _construction_figure, _construction_axes = plot_locations(
        locations=locations,
        tours=best_tour,
        figsize=(8, 5),
    )
    _construction_axes.set_title("")
    _construction_figure
    return best_tour, shortest_tour_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 2: Improve the constructed tour

    Pairwise interchange defines a neighbor by selecting two distinct positions
    and swapping their locations. The move preserves feasibility because it
    changes only the visit order. A seeded random number generator makes the
    sampled sequence of neighbors reproducible.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(seed=0)
    return (rng,)


@app.cell
def _(rng):
    def get_pi_neighbor(solution_list: list[str]) -> list[str]:
        p1, p2 = rng.choice(
            a=len(solution_list),
            size=2,
            replace=False,
        )
        neighbor = list(solution_list)
        neighbor[p1], neighbor[p2] = neighbor[p2], neighbor[p1]
        return neighbor

    return (get_pi_neighbor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The search samples one neighbor at a time and accepts it only when its road
    distance is strictly lower than the incumbent distance. Each accepted move
    resets the non-improving counter. The procedure stops after 2,000 consecutive
    sampled neighbors fail to improve the incumbent.
    """)
    return


@app.cell
def _(
    best_tour,
    calculate_tour_distance,
    get_pi_neighbor,
    pl,
    road_distances,
):
    incumbent = list(best_tour)
    incumbent_value = calculate_tour_distance(
        tour=incumbent,
        road_distances=road_distances,
    )
    construction_solution_value = incumbent_value
    non_improving_limit = 2_000
    non_improving_count = 0
    overall_count = 0
    stats = [
        {
            "iteration": overall_count,
            "incumbent_distance_miles": incumbent_value,
        }
    ]

    while non_improving_count < non_improving_limit:
        neighbor = get_pi_neighbor(solution_list=incumbent)
        neighbor_value = calculate_tour_distance(
            tour=neighbor,
            road_distances=road_distances,
        )
        if neighbor_value < incumbent_value:
            incumbent = list(neighbor)
            incumbent_value = neighbor_value
            non_improving_count = 0
        else:
            non_improving_count += 1

        overall_count += 1
        stats.append(
            {
                "iteration": overall_count,
                "incumbent_distance_miles": incumbent_value,
            }
        )

    stats_df = pl.DataFrame(stats)
    return construction_solution_value, incumbent, incumbent_value, stats_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Improvement results

    The first plot shows the best tour found by pairwise interchange. The second
    plot traces the incumbent distance after each sampled neighbor. The dashed
    line records the Nearest Neighbor starting value.
    """)
    return


@app.cell
def _(incumbent, locations, plot_locations):
    _improved_figure, _improved_axes = plot_locations(
        locations=locations,
        tours=incumbent,
        figsize=(8, 5),
    )
    _improved_axes.set_title("")
    _improved_figure
    return


@app.cell
def _(construction_solution_value, plt, sns, stats_df):
    _plot_data = stats_df.to_dict(as_series=False)
    _figure, _axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(8, 5),
        layout="constrained",
    )
    sns.lineplot(
        data=_plot_data,
        x="iteration",
        y="incumbent_distance_miles",
        color=sns.color_palette("colorblind")[0],
        label="Best distance found",
        ax=_axes,
    )
    _axes.axhline(
        y=construction_solution_value,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label="Nearest Neighbor baseline",
    )
    _axes.set_xlabel("Candidate solutions evaluated")
    _axes.set_ylabel("Incumbent tour distance (road miles)")
    _axes.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )
    sns.despine(ax=_axes)
    _figure
    return


if __name__ == "__main__":
    app.run()
