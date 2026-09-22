"""Fast, reproducible construction and improvement for the classroom TSP."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Problem:
    stores: tuple[str, ...]
    cities: tuple[str, ...]
    states: tuple[str, ...]
    latitudes: tuple[float, ...]
    longitudes: tuple[float, ...]
    distances: np.ndarray


@dataclass(frozen=True)
class Progress:
    stage: str
    starts_done: int
    total_starts: int
    attempts: int
    non_improving: int
    stop_limit: int
    best_distance: float
    accepted_moves: int


@dataclass(frozen=True)
class Result:
    constructed_tour: tuple[str, ...]
    final_tour: tuple[str, ...]
    constructed_miles: float
    final_miles: float
    attempts: int
    accepted_moves: int
    elapsed_seconds: float
    history: tuple[tuple[int, float], ...]


def load_problem(data_dir: Path) -> Problem:
    """Load the class inputs once and index the complete distance table."""
    locations = pl.read_parquet(data_dir / "store_locations.parquet").sort("store")
    roads = pl.read_parquet(data_dir / "road_distances.parquet")
    stores = tuple(locations.get_column("store").to_list())
    if len(stores) != len(set(stores)):
        raise ValueError("The facility table contains duplicate store IDs.")
    index = {store: position for position, store in enumerate(stores)}
    distances = np.full(
        shape=(len(stores), len(stores)),
        fill_value=np.nan,
        dtype=np.float64,
    )
    assigned = np.zeros((len(stores), len(stores)), dtype=np.bool_)
    for origin, destination, miles in roads.iter_rows():
        if origin not in index or destination not in index:
            raise ValueError("The road-distance table contains an unknown store ID.")
        origin_index = index[origin]
        destination_index = index[destination]
        if assigned[origin_index, destination_index]:
            raise ValueError("The road-distance table contains duplicate pairs.")
        assigned[origin_index, destination_index] = True
        distances[origin_index, destination_index] = miles
    if not np.isfinite(distances).all():
        raise ValueError("The road-distance table is incomplete or non-finite.")
    if not np.allclose(
        a=distances,
        b=distances.T,
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("The road-distance table must be symmetric for 2-opt.")
    return Problem(
        stores=stores,
        cities=tuple(locations.get_column("city").to_list()),
        states=tuple(locations.get_column("state").to_list()),
        latitudes=tuple(locations.get_column("latitude").to_list()),
        longitudes=tuple(locations.get_column("longitude").to_list()),
        distances=distances,
    )


def select_states(problem: Problem, selected_states: Sequence[str]) -> Problem:
    """Retain facilities in selected states, preserving store-ID order."""
    selected = set(selected_states)
    if not selected:
        raise ValueError("Select at least one state.")
    unknown = selected - set(problem.states)
    if unknown:
        raise ValueError(f"Unknown states: {', '.join(sorted(unknown))}")
    indices = [i for i, state in enumerate(problem.states) if state in selected]
    return Problem(
        stores=tuple(problem.stores[i] for i in indices),
        cities=tuple(problem.cities[i] for i in indices),
        states=tuple(problem.states[i] for i in indices),
        latitudes=tuple(problem.latitudes[i] for i in indices),
        longitudes=tuple(problem.longitudes[i] for i in indices),
        distances=problem.distances[np.ix_(indices, indices)],
    )


def tour_distance(tour: Sequence[int], distances: np.ndarray) -> float:
    """Return road miles for the closed tour represented by integer indices."""
    if len(tour) < 2:
        return 0.0
    return float(sum(distances[tour[i - 1], tour[i]] for i in range(len(tour))))


def solve(
    problem: Problem,
    *,
    seed: int = 0,
    non_improving_limit: int = 10_000,
    on_progress: Callable[[Progress], None] | None = None,
) -> Result:
    """Run all-start nearest neighbor, then sampled strict-improvement 2-opt."""
    if seed < 0:
        raise ValueError("The random seed must be nonnegative.")
    if non_improving_limit < 1:
        raise ValueError("The non-improving limit must be positive.")
    n = len(problem.stores)
    if n == 0:
        raise ValueError("The problem must contain at least one facility.")

    started = perf_counter()
    distances = problem.distances
    # A stable sort of store-ID-ordered columns reproduces the notebook's
    # distance-first, store-ID-second tie break without repeated DataFrame work.
    ranked = np.argsort(
        a=distances,
        axis=1,
        kind="stable",
    )
    best_tour: list[int] = []
    best_miles = float("inf")
    for start in range(n):
        visited = np.zeros(n, dtype=np.bool_)
        visited[start] = True
        tour = [start]
        origin = start
        while len(tour) < n:
            destination = next(i for i in ranked[origin] if not visited[i])
            visited[destination] = True
            tour.append(int(destination))
            origin = int(destination)
        miles = tour_distance(tour, distances)
        if miles < best_miles:
            best_tour = tour
            best_miles = miles
        if on_progress is not None:
            on_progress(
                Progress(
                    stage="construction",
                    starts_done=start + 1,
                    total_starts=n,
                    attempts=0,
                    non_improving=0,
                    stop_limit=non_improving_limit,
                    best_distance=best_miles,
                    accepted_moves=0,
                )
            )

    constructed_tour = tuple(problem.stores[i] for i in best_tour)
    constructed_miles = best_miles
    current = best_tour.copy()
    rng = np.random.default_rng(seed=seed)
    attempts = 0
    accepted_moves = 0
    non_improving = 0
    history = [(0, best_miles)]

    while n > 1 and non_improving < non_improving_limit:
        left, right = sorted(
            rng.choice(
                a=n,
                size=2,
                replace=False,
            )
        )
        left = int(left)
        right = int(right)
        attempts += 1
        improved = False
        if not (left == 0 and right == n - 1):
            before = current[left - 1]
            first = current[left]
            last = current[right]
            after = current[(right + 1) % n]
            # Road miles. Symmetry makes every internal reversed edge cancel.
            delta = (
                distances[before, last]
                + distances[first, after]
                - distances[before, first]
                - distances[last, after]
            )
            if delta < -1e-9:
                candidate = current.copy()
                candidate[left : right + 1] = reversed(candidate[left : right + 1])
                candidate_miles = tour_distance(candidate, distances)
                if candidate_miles < best_miles - 1e-9:
                    current = candidate
                    best_miles = candidate_miles
                    accepted_moves += 1
                    non_improving = 0
                    history.append((attempts, best_miles))
                    improved = True
        if not improved:
            non_improving += 1
        if on_progress is not None and (improved or attempts % 1_000 == 0):
            on_progress(
                Progress(
                    stage="improvement",
                    starts_done=n,
                    total_starts=n,
                    attempts=attempts,
                    non_improving=non_improving,
                    stop_limit=non_improving_limit,
                    best_distance=best_miles,
                    accepted_moves=accepted_moves,
                )
            )

    if on_progress is not None:
        on_progress(
            Progress(
                stage="complete",
                starts_done=n,
                total_starts=n,
                attempts=attempts,
                non_improving=non_improving,
                stop_limit=non_improving_limit,
                best_distance=best_miles,
                accepted_moves=accepted_moves,
            )
        )
    return Result(
        constructed_tour=constructed_tour,
        final_tour=tuple(problem.stores[i] for i in current),
        constructed_miles=constructed_miles,
        final_miles=best_miles,
        attempts=attempts,
        accepted_moves=accepted_moves,
        elapsed_seconds=perf_counter() - started,
        history=tuple(history),
    )
