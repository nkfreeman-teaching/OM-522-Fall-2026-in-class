import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tsp_utils import calculate_tour_distance, plot_locations


@pytest.fixture
def locations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "store": ["A", "B", "C"],
            "latitude": [33.75, 34.05, 33.52],
            "longitude": [-84.39, -84.31, -86.80],
        }
    )


@pytest.fixture
def road_distances() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "store1": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "store2": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "distance_miles": [0.0, 3.0, 5.0, 3.0, 0.0, 4.0, 5.0, 4.0, 0.0],
        }
    )


def test_calculate_tour_distance_closes_tour(road_distances: pl.DataFrame) -> None:
    assert calculate_tour_distance(["A", "B", "C"], road_distances) == 12.0


def test_calculate_tour_distance_does_not_close_twice(
    road_distances: pl.DataFrame,
) -> None:
    assert calculate_tour_distance(["A", "B", "C", "A"], road_distances) == 12.0


def test_calculate_tour_distance_handles_short_tours(
    road_distances: pl.DataFrame,
) -> None:
    assert calculate_tour_distance([], road_distances) == 0.0
    assert calculate_tour_distance(["A"], road_distances) == 0.0
    assert calculate_tour_distance(["A", "B"], road_distances) == 6.0


def test_calculate_tour_distance_rejects_unknown_store(
    road_distances: pl.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="unknown store IDs: Z"):
        calculate_tour_distance(["A", "Z"], road_distances)


def test_calculate_tour_distance_rejects_missing_edge(
    road_distances: pl.DataFrame,
) -> None:
    incomplete = road_distances.filter(
        ~((pl.col("store1") == "C") & (pl.col("store2") == "A"))
    )
    with pytest.raises(ValueError, match="C->A"):
        calculate_tour_distance(["A", "B", "C"], incomplete)


def test_plot_locations_closes_single_tour(locations: pl.DataFrame) -> None:
    figure, axes = plot_locations(
        locations=locations,
        tours=["A", "B", "C"],
    )
    route = next(line for line in axes.lines if line.get_label() == "Tour 1")
    assert route.get_xdata()[0] == route.get_xdata()[-1]
    assert route.get_ydata()[0] == route.get_ydata()[-1]
    plt.close(figure)


def test_plot_locations_closes_each_vehicle_tour(locations: pl.DataFrame) -> None:
    figure, axes = plot_locations(
        locations=locations,
        tours=[["A", "B"], ["C", "A"]],
    )
    routes = [line for line in axes.lines if line.get_label().startswith("Tour ")]
    assert len(routes) == 2
    for route in routes:
        assert route.get_xdata()[0] == route.get_xdata()[-1]
        assert route.get_ydata()[0] == route.get_ydata()[-1]
    plt.close(figure)


def test_plot_locations_uses_requested_figure_size(locations: pl.DataFrame) -> None:
    figure, _ = plot_locations(
        locations=locations,
        figsize=(12, 6),
    )
    assert figure.get_size_inches().tolist() == pytest.approx([12.0, 6.0])
    plt.close(figure)


def test_plot_locations_rejects_unknown_store(locations: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown store IDs: Z"):
        plot_locations(
            locations=locations,
            tours=["A", "Z"],
        )


def test_clean_course_data() -> None:
    locations = pl.read_parquet("data/store_locations.parquet")
    road_distances = pl.read_parquet("data/road_distances.parquet")

    assert locations.shape == (439, 3)
    assert locations.get_column("store").n_unique() == 439
    assert locations.select(
        pl.struct(["latitude", "longitude"]).n_unique()
    ).item() == 439
    assert road_distances.shape == (192_721, 3)
    assert road_distances.select(
        pl.struct(["store1", "store2"]).n_unique()
    ).item() == 192_721
    assert set(road_distances.get_column("store1")) == set(
        locations.get_column("store")
    )
    assert set(road_distances.get_column("store2")) == set(
        locations.get_column("store")
    )
    assert (
        road_distances
        .filter(pl.col("store1") != pl.col("store2"))
        .get_column("distance_miles")
        .min()
        > 0
    )
    assert (
        road_distances
        .filter(pl.col("store1") == pl.col("store2"))
        .get_column("distance_miles")
        .max()
        == 0
    )

    reverse_distances = road_distances.rename(
        {
            "store1": "store2",
            "store2": "store1",
            "distance_miles": "reverse_miles",
        }
    )
    paired_distances = road_distances.join(
        other=reverse_distances,
        on=["store1", "store2"],
        how="inner",
        validate="1:1",
    )
    assert paired_distances.height == 192_721
    assert paired_distances.select(
        (pl.col("distance_miles") - pl.col("reverse_miles")).abs().max()
    ).item() == 0


def test_course_data_remains_a_nonmetric_tsp() -> None:
    road_distances = pl.read_parquet("data/road_distances.parquet")
    relevant_edges = (
        road_distances
        .filter(
            pl.struct(["store1", "store2"]).is_in(
                [
                    {"store1": "L55", "store2": "L434"},
                    {"store1": "L55", "store2": "L2"},
                    {"store1": "L2", "store2": "L434"},
                ]
            )
        )
        .rows_by_key(["store1", "store2"], unique=True)
    )
    direct = relevant_edges[("L55", "L434")][0]
    via_l2 = (
        relevant_edges[("L55", "L2")][0]
        + relevant_edges[("L2", "L434")][0]
    )
    assert direct > via_l2
