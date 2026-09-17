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

    return Path, calculate_tour_distance, mo, np, pl, plot_locations, plt, sns, tqdm


@app.cell(hide_code=True)
def _(locations, mo, selected_states):
    mo.md(
        f"""
        # OM 522 TSP Neighborhood Search

        This notebook uses {locations.height} facility locations across
        {", ".join(selected_states)} to demonstrate a two-phase heuristic for the
        traveling salesperson problem. Multistart Nearest Neighbor constructs a
        complete tour. A strict-improvement search then samples either pairwise
        interchange or subsequence-reversal neighbors. Every tour closes back to
        its starting location automatically.
        """
    )
    return


@app.cell
def _(Path, pl):
    project_root = Path(__file__).parent
    locations = pl.read_parquet(project_root / "data" / "store_locations.parquet")
    road_distances = pl.read_parquet(
        project_root / "data" / "road_distances.parquet"
    )
    selected_states = ["AL", "TN", "GA", "MS", "LA"]
    locations = locations.filter(pl.col("state").is_in(selected_states))

    locations.head()

    N = set(locations.get_column("store").to_list())

    # Restrict the distance table before repeatedly searching it in the loop.
    road_distances = road_distances.filter(
        pl.col("store1").is_in(N),
        pl.col("store2").is_in(N),
    )
    return N, locations, road_distances, selected_states


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


@app.cell
def _(road_distances):
    distance_dict = {}
    for _entry in road_distances.to_dicts():
        _store1 = _entry.get("store1")
        _store2 = _entry.get("store2")
        _distance = _entry.get("distance_miles")
        distance_dict[(_store1, _store2)] = _distance

    def calculate_tour_distance_fast(tour: list[str]) -> float:
        _total_distance = []
        for _start, _end in zip(tour[:-1], tour[1:]):
            _total_distance.append(distance_dict[(_start, _end)])
        _total_distance.append(distance_dict[(tour[-1], tour[0])])
        return sum(_total_distance)

    return (calculate_tour_distance_fast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Faster tour evaluation

    The validated utility function uses DataFrame operations and extensive input
    checks. Those checks are useful at a system boundary, but repeating them for
    every sampled neighbor is expensive. The function above converts the filtered
    road-distance table into a dictionary keyed by `(origin, destination)`. Each
    leg then requires one dictionary lookup, and `zip` aligns consecutive stops.
    The final lookup adds the return leg to the starting location.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A dictionary uses hash-based key lookup. The tuple `(origin, destination)` is
    a valid key because tuples of strings are hashable. Sets use the same basic
    lookup idea for their elements, while a list membership check may scan items
    sequentially. The choice of data structure matters when an operation runs
    thousands of times inside an optimization loop.
    """)
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
    calculate_tour_distance_fast,
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

        # calculate_tour_distance_fast includes the final return to the seed.
        tour_length = calculate_tour_distance_fast(
            tour=tour,
        )
        if tour_length < shortest_tour_length:
            shortest_tour_length = tour_length
            best_tour = list(tour)

    _construction_figure, _construction_axes = plot_locations(
        locations=locations,
        tours=best_tour,
        figsize=(8, 5),
    )
    _construction_axes.set_title("")
    _construction_figure
    return best_tour, shortest_tour_length


@app.cell(hide_code=True)
def _(
    best_tour,
    calculate_tour_distance,
    calculate_tour_distance_fast,
    mo,
    np,
    road_distances,
):
    _validated_distance = calculate_tour_distance(
        tour=best_tour,
        road_distances=road_distances,
    )
    _fast_distance = calculate_tour_distance_fast(tour=best_tour)
    assert np.isclose(_fast_distance, _validated_distance)
    mo.md(
        f"""
        The fast evaluator returns **{_fast_distance:,.1f} road miles** for the
        constructed tour, matching the independently validated evaluator.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 2: Improve the constructed tour

    Pairwise interchange selects two distinct positions and swaps their
    locations. Subsequence reversal selects two endpoints and reverses every
    location between them, including both endpoints. For a symmetric TSP, this
    reversal is the route change made by a 2-opt move. Both moves preserve
    feasibility because they change only the visit order. A seeded random number
    generator makes the sampled sequence of neighbors reproducible.
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


@app.cell
def _(rng):
    def get_ssr_neighbor(solution_list: list[str]) -> list[str]:
        position_array = rng.choice(
            a=len(solution_list),
            size=2,
            replace=False,
        )
        position_array.sort()
        p1, p2 = position_array

        neighbor = list(solution_list)
        neighbor[p1 : p2 + 1] = neighbor[p1 : p2 + 1][::-1]
        return neighbor

    return (get_ssr_neighbor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The search samples one neighbor at a time and accepts it only when its road
    distance is strictly lower than the incumbent distance. Each accepted move
    resets the non-improving counter. The procedure stops after 10,000 consecutive
    sampled neighbors fail to improve the incumbent. This sampled stopping rule
    does not prove local or global optimality.
    """)
    return


@app.cell
def _(
    best_tour,
    calculate_tour_distance_fast,
    get_pi_neighbor,
    get_ssr_neighbor,
    pl,
):
    current_neighborhood_function = get_ssr_neighbor
    current_neighborhood_name = "Subsequence reversal (2-opt)"

    incumbent = list(best_tour)
    incumbent_value = calculate_tour_distance_fast(
        tour=incumbent,
    )
    construction_solution_value = incumbent_value
    non_improving_limit = 10_000
    non_improving_count = 0
    overall_count = 0
    stats = [
        {
            "iteration": overall_count,
            "incumbent_distance_miles": incumbent_value,
        }
    ]

    while non_improving_count < non_improving_limit:
        neighbor = current_neighborhood_function(solution_list=incumbent)
        neighbor_value = calculate_tour_distance_fast(
            tour=neighbor,
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
    return (
        construction_solution_value,
        current_neighborhood_name,
        incumbent,
        incumbent_value,
        stats_df,
    )


@app.cell(hide_code=True)
def _(
    calculate_tour_distance,
    construction_solution_value,
    incumbent,
    incumbent_value,
    locations,
    np,
    road_distances,
):
    _expected_stores = set(locations.get_column("store").to_list())
    assert len(incumbent) == locations.height
    assert set(incumbent) == _expected_stores
    assert incumbent_value <= construction_solution_value

    _validated_incumbent_value = calculate_tour_distance(
        tour=incumbent,
        road_distances=road_distances,
    )
    assert np.isclose(incumbent_value, _validated_incumbent_value)
    return


@app.cell(hide_code=True)
def _(
    construction_solution_value,
    current_neighborhood_name,
    incumbent_value,
    mo,
):
    mo.md(f"""
    ## Improvement results

    The first plot shows the best tour found with {current_neighborhood_name}.
    The search reduced the tour from **{construction_solution_value:,.1f}** to
    **{incumbent_value:,.1f} road miles**. The second plot traces the incumbent
    distance after each sampled neighbor. The dashed line records the Nearest
    Neighbor starting value.
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
