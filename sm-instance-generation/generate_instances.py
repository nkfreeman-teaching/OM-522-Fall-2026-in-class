"""Generate reproducible single-machine scheduling instances."""

from dataclasses import dataclass
import json
from pathlib import Path
import random

import polars as pl


IntegerRange = tuple[int, int]
VALID_SETUP_MODES = frozenset({"none", "fixed", "sequence-dependent"})

JOB_SCHEMA = {
    "Job": pl.String,
    "Processing_Time": pl.Int64,
    "Release_Time": pl.Int64,
    "Due_Date": pl.Int64,
    "Weight": pl.Int64,
    "Fixed_Setup_Time": pl.Int64,
}

SETUP_SCHEMA = {
    "From_Job": pl.String,
    "To_Job": pl.String,
    "Setup_Time": pl.Int64,
}


@dataclass(frozen=True)
class GenerationConfig:
    """Settings that define one reproducible batch of instances."""

    seed: int
    number_of_instances: int
    number_of_jobs: int | None
    job_count_range: IntegerRange | None
    processing_time_range: IntegerRange
    release_time_range: IntegerRange
    weight_range: IntegerRange
    due_date_slack_range: IntegerRange
    setup_mode: str
    fixed_setup_time: int
    sequence_setup_time_range: IntegerRange
    output_directory: Path


# Edit this block to define a batch. Exactly one of number_of_jobs and
# job_count_range must be set to a value other than None.
CONFIG = GenerationConfig(
    seed=522,
    number_of_instances=3,
    number_of_jobs=10,
    job_count_range=None,
    processing_time_range=(1, 10),
    release_time_range=(0, 10),
    weight_range=(1, 5),
    due_date_slack_range=(0, 20),
    setup_mode="none",
    fixed_setup_time=1,
    sequence_setup_time_range=(0, 5),
    output_directory=Path(__file__).resolve().parent / "generated-instances",
)


def validate_integer_range(
    name: str,
    values: IntegerRange,
    minimum: int,
) -> None:
    """Validate an inclusive integer range."""
    lower, upper = values
    if not isinstance(lower, int) or not isinstance(upper, int):
        raise TypeError(f"{name} must contain two integers.")
    if lower < minimum:
        raise ValueError(f"{name} must start at or above {minimum}.")
    if lower > upper:
        raise ValueError(f"{name} must have a lower bound no larger than its upper bound.")


def validate_config(config: GenerationConfig) -> None:
    """Validate all settings before creating the output directory."""
    if not isinstance(config.seed, int):
        raise TypeError("seed must be an integer.")
    if not isinstance(config.number_of_instances, int):
        raise TypeError("number_of_instances must be an integer.")
    if config.number_of_instances < 1:
        raise ValueError("number_of_instances must be positive.")

    uses_fixed_job_count = config.number_of_jobs is not None
    uses_job_count_range = config.job_count_range is not None
    if uses_fixed_job_count == uses_job_count_range:
        raise ValueError(
            "Set exactly one of number_of_jobs and job_count_range to a value other than None."
        )

    if config.number_of_jobs is not None:
        if not isinstance(config.number_of_jobs, int):
            raise TypeError("number_of_jobs must be an integer or None.")
        if config.number_of_jobs < 1:
            raise ValueError("number_of_jobs must be positive.")

    if config.job_count_range is not None:
        validate_integer_range(
            name="job_count_range",
            values=config.job_count_range,
            minimum=1,
        )

    validate_integer_range(
        name="processing_time_range",
        values=config.processing_time_range,
        minimum=1,
    )
    validate_integer_range(
        name="release_time_range",
        values=config.release_time_range,
        minimum=0,
    )
    validate_integer_range(
        name="weight_range",
        values=config.weight_range,
        minimum=1,
    )
    validate_integer_range(
        name="due_date_slack_range",
        values=config.due_date_slack_range,
        minimum=0,
    )
    validate_integer_range(
        name="sequence_setup_time_range",
        values=config.sequence_setup_time_range,
        minimum=0,
    )

    if config.setup_mode not in VALID_SETUP_MODES:
        valid_modes = ", ".join(sorted(VALID_SETUP_MODES))
        raise ValueError(f"setup_mode must be one of: {valid_modes}.")
    if not isinstance(config.fixed_setup_time, int):
        raise TypeError("fixed_setup_time must be an integer.")
    if config.fixed_setup_time < 0:
        raise ValueError("fixed_setup_time must be nonnegative.")


def check_output_directory(output_directory: Path) -> None:
    """Reject files and nonempty directories before writing output."""
    if not output_directory.exists():
        return
    if not output_directory.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_directory}")
    if next(output_directory.iterdir(), None) is not None:
        raise FileExistsError(
            f"Output directory is not empty: {output_directory}. "
            "Choose an empty directory before generating another batch."
        )


def sample_integer(
    rng: random.Random,
    values: IntegerRange,
) -> int:
    """Sample from an inclusive integer range."""
    lower, upper = values
    return rng.randint(lower, upper)


def choose_job_count(
    config: GenerationConfig,
    rng: random.Random,
) -> int:
    """Choose the configured fixed or ranged job count."""
    if config.number_of_jobs is not None:
        return config.number_of_jobs
    if config.job_count_range is None:
        raise RuntimeError("Configuration validation did not establish a job count.")
    return sample_integer(rng, config.job_count_range)


def make_job_ids(number_of_jobs: int) -> list[str]:
    """Create stable, naturally sorted job identifiers."""
    width = max(3, len(str(number_of_jobs)))
    return [f"J{job_number:0{width}d}" for job_number in range(1, number_of_jobs + 1)]


def generate_jobs(
    config: GenerationConfig,
    rng: random.Random,
    number_of_jobs: int,
) -> pl.DataFrame:
    """Generate one jobs table."""
    fixed_setup_time = config.fixed_setup_time if config.setup_mode == "fixed" else 0
    rows: list[dict[str, int | str]] = []

    for job_id in make_job_ids(number_of_jobs):
        processing_time = sample_integer(rng, config.processing_time_range)
        release_time = sample_integer(rng, config.release_time_range)
        weight = sample_integer(rng, config.weight_range)
        due_date_slack = sample_integer(rng, config.due_date_slack_range)
        due_date = release_time + processing_time + due_date_slack
        rows.append(
            {
                "Job": job_id,
                "Processing_Time": processing_time,
                "Release_Time": release_time,
                "Due_Date": due_date,
                "Weight": weight,
                "Fixed_Setup_Time": fixed_setup_time,
            }
        )

    return pl.DataFrame(
        data=rows,
        schema=JOB_SCHEMA,
    )


def generate_sequence_dependent_setups(
    config: GenerationConfig,
    rng: random.Random,
    job_ids: list[str],
) -> pl.DataFrame:
    """Generate directed initial and job-to-job setup arcs."""
    rows: list[dict[str, int | str]] = []

    for to_job in job_ids:
        rows.append(
            {
                "From_Job": "START",
                "To_Job": to_job,
                "Setup_Time": sample_integer(rng, config.sequence_setup_time_range),
            }
        )

    for from_job in job_ids:
        for to_job in job_ids:
            if from_job == to_job:
                continue
            rows.append(
                {
                    "From_Job": from_job,
                    "To_Job": to_job,
                    "Setup_Time": sample_integer(rng, config.sequence_setup_time_range),
                }
            )

    return pl.DataFrame(
        data=rows,
        schema=SETUP_SCHEMA,
    )


def build_manifest_settings(config: GenerationConfig) -> dict[str, object]:
    """Convert generation settings into JSON-compatible values."""
    return {
        "number_of_instances": config.number_of_instances,
        "job_count": {
            "fixed": config.number_of_jobs,
            "range": list(config.job_count_range) if config.job_count_range is not None else None,
        },
        "processing_time_range": list(config.processing_time_range),
        "release_time_range": list(config.release_time_range),
        "weight_range": list(config.weight_range),
        "due_date_slack_range": list(config.due_date_slack_range),
        "due_date_rule": "release_time + processing_time + uniformly sampled slack",
        "setup_mode": config.setup_mode,
        "fixed_setup_time": config.fixed_setup_time,
        "sequence_setup_time_range": list(config.sequence_setup_time_range),
    }


def write_manifest(
    output_directory: Path,
    manifest: dict[str, object],
) -> Path:
    """Write a deterministic, human-readable batch manifest."""
    manifest_path = output_directory / "manifest.json"
    with manifest_path.open(
        mode="w",
        encoding="utf-8",
        newline="\n",
    ) as manifest_file:
        json.dump(
            obj=manifest,
            fp=manifest_file,
            indent=2,
            sort_keys=True,
        )
        manifest_file.write("\n")
    return manifest_path


def generate_batch(config: GenerationConfig) -> Path:
    """Generate all instance folders and return the manifest path."""
    validate_config(config)
    output_directory = config.output_directory.resolve()
    check_output_directory(output_directory)

    rng = random.Random(config.seed)
    instance_entries: list[dict[str, object]] = []
    instance_width = max(3, len(str(config.number_of_instances)))
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for instance_number in range(1, config.number_of_instances + 1):
        instance_id = f"instance_{instance_number:0{instance_width}d}"
        number_of_jobs = choose_job_count(config, rng)
        jobs = generate_jobs(
            config=config,
            rng=rng,
            number_of_jobs=number_of_jobs,
        )

        instance_directory = output_directory / instance_id
        instance_directory.mkdir()
        jobs_path = instance_directory / "jobs.parquet"
        jobs.write_parquet(jobs_path, compression="zstd")

        setup_times_file: str | None = None
        if config.setup_mode == "sequence-dependent":
            setup_times = generate_sequence_dependent_setups(
                config=config,
                rng=rng,
                job_ids=jobs["Job"].to_list(),
            )
            setup_path = instance_directory / "setup_times.parquet"
            setup_times.write_parquet(setup_path, compression="zstd")
            setup_times_file = f"{instance_id}/setup_times.parquet"

        instance_entries.append(
            {
                "instance_id": instance_id,
                "job_count": number_of_jobs,
                "jobs_file": f"{instance_id}/jobs.parquet",
                "setup_times_file": setup_times_file,
            }
        )

    manifest = {
        "schema_version": 1,
        "generator": "generate_instances.py",
        "rng": {
            "implementation": "random.Random",
            "seed": config.seed,
        },
        "settings": build_manifest_settings(config),
        "instances": instance_entries,
    }
    manifest_path = write_manifest(output_directory, manifest)

    print(
        f"Generated {config.number_of_instances} instance(s) with seed {config.seed} "
        f"in {output_directory}"
    )
    for entry in instance_entries:
        print(f"  {entry['instance_id']}: {entry['job_count']} jobs")
    return manifest_path


def main() -> None:
    """Generate the batch defined by CONFIG."""
    generate_batch(CONFIG)


if __name__ == "__main__":
    main()
