"""Compare dispatch rules followed by single-objective neighborhood descent."""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parent
METRICS = ("completion", "weighted_completion", "lateness", "weighted_tardiness")
RULES = ("EDD", "SPT", "CR", "WSPT")
NEIGHBORHOODS = ("adjacent", "swap", "reversal")


def load_jobs(path: Path) -> list[dict]:
    frame = pl.read_parquet(path)
    required = {"job", "processing_time", "release_time", "due_date", "weight", "fixed_setup_time"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected columns in {path}: {frame.columns}")
    if frame.null_count().sum_horizontal().item() != 0:
        raise ValueError(f"Null job data in {path}")
    jobs = frame.to_dicts()
    if len({row["job"] for row in jobs}) != len(jobs):
        raise ValueError(f"Duplicate job identifiers in {path}")
    if any(row["processing_time"] <= 0 or row["weight"] <= 0 for row in jobs):
        raise ValueError(f"Processing times and weights must be positive in {path}")
    if any(row["release_time"] < 0 or row["fixed_setup_time"] != 0 for row in jobs):
        raise ValueError(f"This study requires nonnegative releases and zero fixed setup in {path}")
    horizon = max(row["release_time"] for row in jobs) + sum(row["processing_time"] for row in jobs)
    total_weight = sum(row["weight"] for row in jobs)
    lateness_bound = horizon + max(abs(row["due_date"]) for row in jobs)
    if max(len(jobs) * horizon, total_weight * horizon, total_weight * lateness_bound) > 2**63 - 1:
        raise ValueError(f"Metrics could exceed signed 64-bit integer range in {path}")
    return jobs


def construct(jobs: list[dict], rule: str) -> list[int]:
    """Choose the best currently released job; idle only if none is ready."""
    remaining = set(range(len(jobs)))
    order = []
    clock = 0
    while remaining:
        available = [index for index in remaining if jobs[index]["release_time"] <= clock]
        if not available:
            clock = min(jobs[index]["release_time"] for index in remaining)
            available = [index for index in remaining if jobs[index]["release_time"] <= clock]

        def priority(index: int) -> tuple:
            job = jobs[index]
            if rule == "EDD":
                value = job["due_date"]
            elif rule == "SPT":
                value = job["processing_time"]
            elif rule == "CR":
                value = (job["due_date"] - clock) / job["processing_time"]
            elif rule == "WSPT":
                value = job["processing_time"] / job["weight"]
            else:
                raise ValueError(rule)
            return (value, job["job"])

        chosen = min(available, key=priority)
        order.append(chosen)
        clock += jobs[chosen]["processing_time"]
        remaining.remove(chosen)
    return order


def evaluate(jobs: list[dict], order: list[int]) -> tuple[int, int, int, int]:
    """Decode a static order, allowing idle time while waiting for its next job."""
    clock = 0
    completion = 0
    weighted_completion = 0
    lateness = -10**18
    weighted_tardiness = 0
    for index in order:
        job = jobs[index]
        clock = max(clock, job["release_time"]) + job["processing_time"]
        late = clock - job["due_date"]
        completion += clock
        weighted_completion += job["weight"] * clock
        lateness = max(lateness, late)
        weighted_tardiness += job["weight"] * max(0, late)
    return completion, weighted_completion, lateness, weighted_tardiness


def prefix_values(jobs: list[dict], order: list[int], objective: int) -> tuple[list[int], list[int]]:
    clocks = [0]
    values = [(-10**18 if objective == 2 else 0)]
    for index in order:
        job = jobs[index]
        finish = max(clocks[-1], job["release_time"]) + job["processing_time"]
        late = finish - job["due_date"]
        if objective == 0:
            value = values[-1] + finish
        elif objective == 1:
            value = values[-1] + job["weight"] * finish
        elif objective == 2:
            value = max(values[-1], late)
        else:
            value = values[-1] + job["weight"] * max(0, late)
        clocks.append(finish)
        values.append(value)
    return clocks, values


def candidate_value(
    jobs: list[dict],
    order: list[int],
    objective: int,
    start: int,
    clocks: list[int],
    values: list[int],
) -> int:
    clock = clocks[start]
    value = values[start]
    for position in range(start, len(order)):
        job = jobs[order[position]]
        clock = max(clock, job["release_time"]) + job["processing_time"]
        late = clock - job["due_date"]
        if objective == 0:
            value += clock
        elif objective == 1:
            value += job["weight"] * clock
        elif objective == 2:
            value = max(value, late)
        else:
            value += job["weight"] * max(0, late)
    return value


def move(order: list[int], first: int, last: int, neighborhood: str) -> None:
    if neighborhood == "reversal":
        order[first:last + 1] = reversed(order[first:last + 1])
    else:
        order[first], order[last] = order[last], order[first]


def search(
    jobs: list[dict],
    initial: list[int],
    objective: int,
    neighborhood: str,
) -> dict:
    order = initial.copy()
    clocks, values = prefix_values(jobs, order, objective)
    initial_value = values[-1]
    evaluations = 0
    accepted = 0
    started = time.perf_counter()
    while True:
        improved = False
        for first in range(len(order) - 1):
            lasts = (first + 1,) if neighborhood == "adjacent" else range(first + 1, len(order))
            for last in lasts:
                move(order, first, last, neighborhood)
                evaluations += 1
                value = candidate_value(
                    jobs=jobs,
                    order=order,
                    objective=objective,
                    start=first,
                    clocks=clocks,
                    values=values,
                )
                if value < values[-1]:
                    clocks, values = prefix_values(jobs, order, objective)
                    accepted += 1
                    improved = True
                    break
                move(order, first, last, neighborhood)
            if improved:
                break
        if not improved:
            break
    seconds = time.perf_counter() - started
    metrics = evaluate(jobs, order)
    if metrics[objective] != values[-1] or metrics[objective] > initial_value:
        raise AssertionError("Incremental objective differs from full evaluation")
    return {
        "order": [jobs[index]["job"] for index in order],
        "metrics": dict(zip(METRICS, metrics)),
        "initial_value": initial_value,
        "evaluations": evaluations,
        "accepted_moves": accepted,
        "seconds": round(seconds, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "study-results.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    results = []
    with tempfile.TemporaryDirectory() as temp:
        executable = Path(temp) / "scheduler"
        subprocess.run(
            args=["g++", "-O3", "-std=c++17", str(ROOT / "scheduler.cpp"), "-o", str(executable)],
            check=True,
        )
        for family, details in manifest["families"].items():
            for instance in details["instances"][:args.limit_instances]:
                jobs = load_jobs(ROOT / "data" / instance["jobs_file"])
                if len(jobs) != instance["job_count"]:
                    raise ValueError(f"Manifest job count differs for {family}/{instance['instance_id']}")
                if details["rush_jobs"]:
                    rush_min = details["rush_jobs"]["weight"][0]
                    rush_count = sum(job["weight"] >= rush_min for job in jobs)
                    if rush_count != instance["rush_jobs"]:
                        raise ValueError(f"Manifest rush count differs for {family}/{instance['instance_id']}")
                input_text = "\n".join(
                    [str(len(jobs))]
                    + [
                        f"{row['job']} {row['processing_time']} {row['release_time']} "
                        f"{row['due_date']} {row['weight']}"
                        for row in jobs
                    ]
                ) + "\n"
                process = subprocess.run(
                    args=[str(executable)],
                    input=input_text,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                for line in process.stdout.splitlines():
                    fields = line.split("\t")
                    if len(fields) != 15:
                        raise ValueError(f"Unexpected scheduler output: {line}")
                    rule, objective, neighborhood = map(int, fields[:3])
                    initial = list(map(int, fields[3:7]))
                    final = list(map(int, fields[7:11]))
                    order = fields[14].split(",")
                    if len(order) != len(jobs) or set(order) != {job["job"] for job in jobs}:
                        raise AssertionError("Invalid final schedule")
                    indices = {job["job"]: index for index, job in enumerate(jobs)}
                    if tuple(final) != evaluate(jobs, [indices[job] for job in order]):
                        raise AssertionError("C++ and Python schedule evaluations differ")
                    if final[objective] > initial[objective]:
                        raise AssertionError("Search worsened its target objective")
                    results.append({
                        "family": family,
                        "instance": instance["instance_id"],
                        "job_count": len(jobs),
                        "rule": RULES[rule],
                        "neighborhood": NEIGHBORHOODS[neighborhood],
                        "objective": METRICS[objective],
                        "initial_metrics": dict(zip(METRICS, initial)),
                        "metrics": dict(zip(METRICS, final)),
                        "evaluations": int(fields[11]),
                        "accepted_moves": int(fields[12]),
                        "seconds": float(fields[13]),
                        "order": order,
                    })
                print(f"{family}/{instance['instance_id']}: {len(jobs)} jobs, 48 searches", flush=True)
    output = {
        "source": "data/manifest.json",
        "design": "static permutation, release-aware evaluation, strict first-improvement descent",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")))
    print(f"Wrote {len(results)} result rows to {args.output}")


if __name__ == "__main__":
    main()
