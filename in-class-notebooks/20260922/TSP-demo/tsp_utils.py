"""Small utilities for plotting and measuring traveling-salesman tours."""

from collections.abc import Sequence
from math import ceil, floor

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap


Tour = Sequence[str]
TourInput = Sequence[str] | Sequence[Sequence[str]]


def _require_columns(
    dataframe: pl.DataFrame,
    required: Sequence[str],
    dataframe_name: str,
) -> None:
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {', '.join(missing)}"
        )


def _validate_tour(tour: Tour) -> list[str]:
    if isinstance(tour, (str, bytes)):
        raise TypeError("A tour must be a sequence of store IDs, not a string.")

    values = list(tour)
    if not all(isinstance(store_id, str) for store_id in values):
        raise TypeError("Every tour entry must be a string store ID.")
    return values


def _close_tour(tour: Tour) -> list[str]:
    values = _validate_tour(tour)
    if len(values) > 1 and values[-1] != values[0]:
        values.append(values[0])
    return values


def _normalize_tours(tours: TourInput | None) -> list[list[str]]:
    if tours is None:
        return []
    if isinstance(tours, (str, bytes)):
        raise TypeError("tours must be a list of store IDs or a list of tours.")

    values = list(tours)
    if not values:
        return []
    if all(isinstance(value, str) for value in values):
        return [_validate_tour(values)]
    if any(isinstance(value, (str, bytes)) for value in values):
        raise TypeError("Do not mix store IDs and nested tours in tours.")

    normalized = []
    for value in values:
        if not isinstance(value, Sequence):
            raise TypeError("Each vehicle tour must be a sequence of store IDs.")
        normalized.append(_validate_tour(value))
    return normalized


def _validated_locations(locations: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(locations, pl.DataFrame):
        raise TypeError("locations must be a Polars DataFrame.")
    _require_columns(
        dataframe=locations,
        required=["store", "latitude", "longitude"],
        dataframe_name="locations",
    )
    if locations.is_empty():
        raise ValueError("locations must contain at least one row.")
    if locations.schema["store"] != pl.String:
        raise TypeError("locations['store'] must contain string IDs.")
    if not locations.schema["latitude"].is_numeric():
        raise TypeError("locations['latitude'] must be numeric.")
    if not locations.schema["longitude"].is_numeric():
        raise TypeError("locations['longitude'] must be numeric.")

    selected = locations.select(
        pl.col("store"),
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64),
    )
    if any(selected.null_count().row(0)):
        raise ValueError("locations contains null IDs or coordinates.")
    if selected.get_column("store").is_duplicated().any():
        raise ValueError("locations contains duplicate store IDs.")

    invalid_coordinates = selected.filter(
        ~pl.col("latitude").is_finite()
        | ~pl.col("longitude").is_finite()
        | ~pl.col("latitude").is_between(-90.0, 90.0)
        | ~pl.col("longitude").is_between(-180.0, 180.0)
    )
    if invalid_coordinates.height:
        invalid_ids = ", ".join(invalid_coordinates.get_column("store").to_list())
        raise ValueError(f"locations contains invalid coordinates for: {invalid_ids}")
    return selected


def plot_locations(
    locations: pl.DataFrame,
    tours: TourInput | None = None,
    *,
    figsize: tuple[float, float] = (9, 7),
) -> tuple[Figure, Axes]:
    """Plot locations and optional closed tours over an offline regional map.

    A flat list of store IDs represents one tour. A nested list represents one
    tour per vehicle. Each tour is closed automatically unless its final entry
    already repeats its starting location. ``figsize`` sets the figure width
    and height in inches.
    """
    selected = _validated_locations(locations)
    normalized_tours = _normalize_tours(tours)
    known_ids = set(selected.get_column("store").to_list())
    unknown_ids = sorted(
        {
            store_id
            for tour in normalized_tours
            for store_id in tour
            if store_id not in known_ids
        }
    )
    if unknown_ids:
        raise ValueError(f"Tours contain unknown store IDs: {', '.join(unknown_ids)}")

    minimum_latitude = selected.get_column("latitude").min()
    maximum_latitude = selected.get_column("latitude").max()
    minimum_longitude = selected.get_column("longitude").min()
    maximum_longitude = selected.get_column("longitude").max()
    latitude_padding = max(0.5, (maximum_latitude - minimum_latitude) * 0.05)
    longitude_padding = max(0.5, (maximum_longitude - minimum_longitude) * 0.05)

    with sns.axes_style("white"):
        figure, axes = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=figsize,
            layout="constrained",
        )

    basemap = Basemap(
        projection="merc",
        llcrnrlat=minimum_latitude - latitude_padding,
        urcrnrlat=maximum_latitude + latitude_padding,
        llcrnrlon=minimum_longitude - longitude_padding,
        urcrnrlon=maximum_longitude + longitude_padding,
        resolution="l",
        ax=axes,
    )
    map_boundary = basemap.drawmapboundary(
        fill_color="white",
        linewidth=0.8,
    )
    map_boundary.set_zorder(-1)
    basemap.drawlsmask(
        land_color="#f2efe9",
        ocean_color="#dbe9f6",
        lakes=True,
        resolution="l",
        grid=5,
    )
    basemap.drawcoastlines(
        color="#4d4d4d",
        linewidth=0.8,
    )
    basemap.drawcountries(
        color="#4d4d4d",
        linewidth=0.8,
    )
    basemap.drawstates(
        color="#777777",
        linewidth=0.6,
    )
    parallels = range(
        floor(minimum_latitude),
        ceil(maximum_latitude) + 1,
        2,
    )
    meridians = range(
        floor(minimum_longitude),
        ceil(maximum_longitude) + 1,
        2,
    )
    basemap.drawparallels(
        circles=parallels,
        labels=[True, False, False, False],
        color="#c7c7c7",
        linewidth=0.4,
        fontsize=8,
    )
    basemap.drawmeridians(
        meridians=meridians,
        labels=[False, False, False, True],
        color="#c7c7c7",
        linewidth=0.4,
        fontsize=8,
    )

    longitude_values = selected.get_column("longitude").to_numpy()
    latitude_values = selected.get_column("latitude").to_numpy()
    x_values, y_values = basemap(longitude_values, latitude_values)
    axes.scatter(
        x=x_values,
        y=y_values,
        s=28,
        color="steelblue",
        edgecolor="k",
        linewidth=0.5,
        alpha=0.65,
        label="Locations",
        zorder=4,
    )

    location_lookup = {
        row["store"]: (row["longitude"], row["latitude"])
        for row in selected.iter_rows(named=True)
    }
    colors = sns.color_palette(
        palette="colorblind",
        n_colors=min(max(len(normalized_tours), 1), 8),
    )
    for tour_number, tour in enumerate(normalized_tours, start=1):
        closed_tour = _close_tour(tour)
        if not closed_tour:
            continue

        route_longitudes = [location_lookup[store_id][0] for store_id in closed_tour]
        route_latitudes = [location_lookup[store_id][1] for store_id in closed_tour]
        route_x, route_y = basemap(route_longitudes, route_latitudes)
        color = colors[(tour_number - 1) % len(colors)]
        if len(closed_tour) > 1:
            axes.plot(
                route_x,
                route_y,
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label=f"Tour {tour_number}",
                zorder=5,
            )
        axes.scatter(
            x=route_x[0],
            y=route_y[0],
            s=70,
            marker="s",
            color=color,
            edgecolor="k",
            linewidth=0.8,
            zorder=6,
        )

    axes.set_title("Store locations and tours" if normalized_tours else "Store locations")
    axes.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )
    return figure, axes


def calculate_tour_distance(
    tour: Tour,
    road_distances: pl.DataFrame,
) -> float:
    """Return a tour's road distance in miles, including its closing leg."""
    if not isinstance(road_distances, pl.DataFrame):
        raise TypeError("road_distances must be a Polars DataFrame.")
    _require_columns(
        dataframe=road_distances,
        required=["store1", "store2", "distance_miles"],
        dataframe_name="road_distances",
    )
    if not road_distances.schema["distance_miles"].is_numeric():
        raise TypeError("road_distances['distance_miles'] must be numeric.")

    lookup = road_distances.select(
        pl.col("store1"),
        pl.col("store2"),
        pl.col("distance_miles").cast(pl.Float64),
    )
    if any(lookup.null_count().row(0)):
        raise ValueError("road_distances contains null keys or distances.")
    duplicate_pairs = (
        lookup
        .group_by(["store1", "store2"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_pairs.height:
        raise ValueError("road_distances contains duplicate origin-destination pairs.")
    if lookup.filter(~pl.col("distance_miles").is_finite()).height:
        raise ValueError("road_distances contains non-finite distances.")

    values = _validate_tour(tour)
    if not values:
        return 0.0

    known_ids = set(lookup.get_column("store1").to_list()) | set(
        lookup.get_column("store2").to_list()
    )
    unknown_ids = sorted(set(values) - known_ids)
    if unknown_ids:
        raise ValueError(f"Tour contains unknown store IDs: {', '.join(unknown_ids)}")
    if len(values) == 1:
        return 0.0

    closed_tour = _close_tour(values)
    edges = pl.DataFrame(
        {
            "store1": closed_tour[:-1],
            "store2": closed_tour[1:],
        }
    )
    matched_edges = edges.join(
        other=lookup,
        on=["store1", "store2"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    missing_edges = matched_edges.filter(pl.col("distance_miles").is_null())
    if missing_edges.height:
        missing_pairs = ", ".join(
            f"{source}->{target}"
            for source, target in missing_edges.select(
                pl.col("store1"),
                pl.col("store2"),
            ).iter_rows()
        )
        raise ValueError(f"road_distances is missing tour edges: {missing_pairs}")
    return float(matched_edges.get_column("distance_miles").sum())
