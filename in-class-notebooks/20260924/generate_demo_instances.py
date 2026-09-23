"""Generate the September 24 demonstration instances in three scenario families."""

import json
import random
from pathlib import Path

import polars as pl


JOB_SCHEMA = {
    "job": pl.String,
    "processing_time": pl.Int64,
    "release_time": pl.Int64,
    "due_date": pl.Int64,
    "weight": pl.Int64,
    "fixed_setup_time": pl.Int64,
}

# Each family stresses a different rule. All ranges are inclusive integer bounds.
# A release window of None spreads releases over five time units per job, which is
# close to the average processing time, so the machine is sometimes idle.
FAMILIES = {
    "backlog": {
        "description": (
            "All jobs arrive in the first 30 time units, so the machine starts with a "
            "large backlog and most jobs finish late."
        ),
        "seed": 9241,
        "release_window": (0, 30),
        "rush_share": 0.0,
        "regular": {"processing_time": (1, 10), "weight": (1, 5), "due_date_slack": (0, 20)},
    },
    "steady": {
        "description": (
            "Arrivals are spread over the horizon, so the machine is sometimes idle and "
            "the order of released jobs matters more."
        ),
        "seed": 9242,
        "release_window": None,
        "rush_share": 0.0,
        "regular": {"processing_time": (1, 10), "weight": (1, 5), "due_date_slack": (0, 20)},
    },
    "rush": {
        "description": (
            "Steady arrivals in which 20 percent of jobs are short, heavily weighted rush "
            "orders with tight due dates."
        ),
        "seed": 9243,
        "release_window": None,
        "rush_share": 0.2,
        "regular": {"processing_time": (3, 10), "weight": (1, 3), "due_date_slack": (5, 30)},
        "rush": {"processing_time": (1, 3), "weight": (8, 10), "due_date_slack": (0, 4)},
    },
}
INSTANCES_PER_FAMILY = 10
JOB_COUNT_RANGE = (40, 80)
RELEASE_TIME_PER_JOB = 5
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data"


def generate_instance(rng: random.Random, family: dict) -> pl.DataFrame:
    """Sample one instance; due date is release + processing time + sampled slack."""
    number_of_jobs = rng.randint(*JOB_COUNT_RANGE)
    if family["release_window"] is None:
        release_low, release_high = 0, RELEASE_TIME_PER_JOB * number_of_jobs
    else:
        release_low, release_high = family["release_window"]

    number_of_rush_jobs = round(family["rush_share"] * number_of_jobs)
    rush_positions = set(rng.sample(range(number_of_jobs), number_of_rush_jobs))

    rows = []
    for index in range(number_of_jobs):
        settings = family["rush"] if index in rush_positions else family["regular"]
        processing_time = rng.randint(*settings["processing_time"])
        release_time = rng.randint(release_low, release_high)
        due_date_slack = rng.randint(*settings["due_date_slack"])
        rows.append(
            {
                "job": f"J{index + 1:03d}",
                "processing_time": processing_time,
                "release_time": release_time,
                "due_date": release_time + processing_time + due_date_slack,
                "weight": rng.randint(*settings["weight"]),
                "fixed_setup_time": 0,
            }
        )
    return pl.DataFrame(rows, schema=JOB_SCHEMA)


def main() -> None:
    if OUTPUT_DIRECTORY.exists() and next(OUTPUT_DIRECTORY.iterdir(), None) is not None:
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIRECTORY}. "
            "Remove it before regenerating the demonstration data."
        )

    manifest = {
        "generator": "generate_demo_instances.py",
        "rng": "random.Random, one seed per family",
        "job_count_range": list(JOB_COUNT_RANGE),
        "due_date_rule": "release_time + processing_time + uniformly sampled slack",
        "families": {},
    }
    for family_name, family in FAMILIES.items():
        rng = random.Random(family["seed"])
        instances = []
        for number in range(1, INSTANCES_PER_FAMILY + 1):
            instance_id = f"instance_{number:03d}"
            instance_directory = OUTPUT_DIRECTORY / family_name / instance_id
            instance_directory.mkdir(parents=True)
            jobs = generate_instance(rng, family)
            jobs.write_parquet(instance_directory / "jobs.parquet", compression="zstd")
            instances.append(
                {
                    "instance_id": instance_id,
                    "job_count": jobs.height,
                    "rush_jobs": round(family["rush_share"] * jobs.height),
                    "jobs_file": f"{family_name}/{instance_id}/jobs.parquet",
                }
            )
        manifest["families"][family_name] = {
            "description": family["description"],
            "seed": family["seed"],
            "release_window": (
                list(family["release_window"])
                if family["release_window"] is not None
                else f"0 to {RELEASE_TIME_PER_JOB} times the job count"
            ),
            "rush_share": family["rush_share"],
            "regular_jobs": {key: list(value) for key, value in family["regular"].items()},
            "rush_jobs": (
                {key: list(value) for key, value in family["rush"].items()}
                if "rush" in family
                else None
            ),
            "instances": instances,
        }
        print(f"{family_name}: {len(instances)} instances")

    with (OUTPUT_DIRECTORY / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


if __name__ == "__main__":
    main()
