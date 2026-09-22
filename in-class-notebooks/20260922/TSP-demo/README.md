# OM 522 Traveling Salesman Demo

This is the finished September 22 application. The [lesson guide](../README.md)
explains how to recreate the build from the separate starting notebook, then
compare the result with this app. The [review report](process_report.html)
records the original build and browser reviews. Download the lesson folder
and open the report locally to see its screenshots.

This project provides a larger geographic instance for demonstrating the traveling salesman problem in OM 522. It contains 439 anonymized facility locations spanning ten southeastern states and a complete pairwise road-distance table.

## Data

`data/store_locations.parquet` contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `store` | string | Anonymized location ID |
| `latitude` | float | Latitude in decimal degrees |
| `longitude` | float | Longitude in decimal degrees |
| `city` | string | City name as recorded in the source file |
| `state` | string | Two-letter state abbreviation |
| `zip` | integer | Five-digit ZIP code |

`data/road_distances.parquet` contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `store1` | string | Origin location ID |
| `store2` | string | Destination location ID |
| `distance_miles` | float | Pairwise road distance in miles |

The original course files contained 440 rows. `L363` was removed because it had the same coordinates and distance vector as `L355`, including a zero-mile edge between the two IDs, and it remains removed.

The descriptive `city`, `state`, and `zip` fields were dropped in an earlier cleaning pass and have since been restored by joining the 439 retained IDs back to the original course file on `store`. Coordinates were identical across every matched row, so the join changed no geometry. One row carries a manual correction: `L424` has ZIP 31788 and coordinates in south Georgia but was labeled city "White House", state "TN" in the source. Its `state` was set to `GA` so that all 439 rows fall inside the state they are plotted in, while its `city` value was left as recorded. Treat `city` as the least reliable of the three descriptive fields.

The inherited files did not include collection provenance or a unit field. The earlier assignment described a 400-mile road-distance constraint, and comparisons with straight-line distances are consistent with road miles. Thus, `distance_miles` is the best-supported interpretation rather than independently recovered metadata.

The distance table is complete and symmetric, but it does not satisfy the triangle inequality. It is suitable for a general symmetric TSP. Do not claim approximation guarantees that require a metric TSP.

## Run the demo

Install the environment and open the web app:

```bash
pixi install
pixi run app
```

Select one or more states and choose **Find a route**. The app reports progress
while it constructs tours from every starting facility and improves the shortest
one through sampled subsequence reversals. The map can be panned, zoomed, and
hovered without online map tiles. The random seed and stopping limit can be
changed in the sidebar. Map segments connect facilities directly for visual
clarity; distances come from the road-distance table. The search does not prove
optimality.

The original marimo notebook remains available for walking through the code:

```bash
pixi run demo
```

Run the checks with:

```bash
pixi run check
pixi run test
```

The plotting utility uses an offline map with coastlines and state boundaries. Tour lines connect facilities directly on the map for visual clarity. Distance totals use the road-distance table.

The notebook's `plot_locations` utility accepts an optional
`figsize=(width, height)` argument, with
dimensions measured in inches:

```python
figure, axes = plot_locations(
    locations=locations,
    tours=tour,
    figsize=(12, 8),
)
```
