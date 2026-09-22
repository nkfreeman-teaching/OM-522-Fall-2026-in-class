"""Behavior checks for the web app's indexed TSP solver."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tsp_solver import Problem, load_problem, select_states, solve, tour_distance


@pytest.fixture(scope="module")
def course_problem() -> Problem:
    return load_problem(Path("data"))


def test_state_selection_retains_complete_distance_matrix(
    course_problem: Problem,
) -> None:
    selection = select_states(course_problem, ["AL", "TN", "GA", "MS", "LA"])
    assert len(selection.stores) == 194
    assert selection.distances.shape == (194, 194)
    assert np.isfinite(selection.distances).all()
    assert set(selection.states) == {"AL", "TN", "GA", "MS", "LA"}


def test_state_selection_rejects_empty_and_unknown(course_problem: Problem) -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_states(course_problem, [])
    with pytest.raises(ValueError, match="Unknown states"):
        select_states(course_problem, ["XX"])


def test_two_opt_matches_independent_full_evaluation(
    course_problem: Problem,
) -> None:
    """Compare the optimized search with explicit reversals on a real subset."""
    problem = select_states(course_problem, ["AR"])
    result = solve(
        problem=problem,
        seed=0,
        non_improving_limit=100,
    )
    index = {store: i for i, store in enumerate(problem.stores)}
    constructed = [index[store] for store in result.constructed_tour]
    current = constructed.copy()
    miles = tour_distance(current, problem.distances)
    rng = np.random.default_rng(seed=0)
    non_improving = 0
    attempts = 0
    accepted = 0
    while non_improving < 100:
        left, right = sorted(
            rng.choice(
                a=len(current),
                size=2,
                replace=False,
            )
        )
        candidate = current.copy()
        candidate[left : right + 1] = reversed(candidate[left : right + 1])
        candidate_miles = tour_distance(candidate, problem.distances)
        attempts += 1
        if candidate_miles < miles - 1e-9:
            current = candidate
            miles = candidate_miles
            non_improving = 0
            accepted += 1
        else:
            non_improving += 1
    assert result.attempts == attempts
    assert result.accepted_moves == accepted
    assert result.final_tour == tuple(problem.stores[i] for i in current)
    assert result.final_miles == pytest.approx(miles)


@pytest.mark.parametrize(
    "states",
    [
        ["FL"],
        ["AL", "TN", "GA", "MS", "LA"],
        ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN"],
    ],
)
def test_route_is_complete_closed_and_nonworsening(
    course_problem: Problem,
    states: list[str],
) -> None:
    problem = select_states(course_problem, states)
    result = solve(
        problem=problem,
        seed=0,
        non_improving_limit=100,
    )
    assert len(result.final_tour) == len(problem.stores)
    assert set(result.final_tour) == set(problem.stores)
    assert result.final_miles <= result.constructed_miles
    indices = [problem.stores.index(store) for store in result.final_tour]
    assert result.final_miles == pytest.approx(
        tour_distance(indices, problem.distances)
    )
    assert result.history[0] == (0, result.constructed_miles)
    assert result.history[-1][1] == result.final_miles


def test_progress_reports_both_phases(course_problem: Problem) -> None:
    problem = select_states(course_problem, ["AR"])
    updates = []
    result = solve(
        problem,
        seed=0,
        non_improving_limit=100,
        on_progress=updates.append,
    )
    assert updates[0].stage == "construction"
    assert updates[0].starts_done == 1
    assert updates[-1].stage == "complete"
    assert updates[-1].best_distance == result.final_miles
    assert any(update.stage == "improvement" for update in updates)


def test_solver_rejects_invalid_settings(course_problem: Problem) -> None:
    problem = select_states(course_problem, ["AR"])
    with pytest.raises(ValueError, match="nonnegative"):
        solve(problem, seed=-1)
    with pytest.raises(ValueError, match="positive"):
        solve(problem, non_improving_limit=0)


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([0.0, 3.0, 3.0, 0.0, 4.0], "duplicate pairs"),
        ([0.0, 3.0, 4.0, 0.0], "symmetric"),
    ],
)
def test_loader_rejects_invalid_road_table(
    tmp_path: Path,
    pairs: list[float],
    message: str,
) -> None:
    pl.DataFrame(
        {
            "store": ["A", "B"],
            "city": ["Alpha", "Beta"],
            "state": ["AL", "AL"],
            "latitude": [32.0, 33.0],
            "longitude": [-86.0, -87.0],
        }
    ).write_parquet(tmp_path / "store_locations.parquet")
    pairs_of_stores = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    if len(pairs) == 5:
        pairs_of_stores.append(("A", "B"))
    pl.DataFrame(
        {
            "store1": [origin for origin, _ in pairs_of_stores],
            "store2": [destination for _, destination in pairs_of_stores],
            "distance_miles": pairs,
        }
    ).write_parquet(tmp_path / "road_distances.parquet")
    with pytest.raises(ValueError, match=message):
        load_problem(tmp_path)
