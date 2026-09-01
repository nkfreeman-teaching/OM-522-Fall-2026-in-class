# Single-Machine Instance Generation

The generator creates reproducible data for single-machine scheduling examples.
Each job has a processing time, release time, due date, and weight. Optional
fixed or sequence-dependent setup times support extensions beyond a basic
dispatching-rule demonstration.

The script generates problem data only. Scheduling rules, schedules, and
tardiness measures are calculated separately.

## Generate a batch

All generation settings appear in the `CONFIG` block near the top of
`generate_instances.py`. After the settings are saved, the following command
runs the generator from the repository root:

```bash
pixi run sm-instance-generate
```

The default configuration generates three instances with 50 jobs each and
seed `522`. Processing times, release times, weights, due-date slack, setup
behavior, and the output location can all be changed in the same block.

## Choose the number of jobs

Exactly one job-count setting must contain a value. A fixed job count uses:

```python
number_of_jobs=10,
job_count_range=None,
```

An inclusive range uses:

```python
number_of_jobs=None,
job_count_range=(8, 15),
```

Each instance receives an independently sampled job count when a range is used.

## Configure job characteristics

All ranges are inclusive and integer-valued.

| Setting | Meaning | Default |
|---|---|---:|
| `processing_time_range` | Minimum and maximum processing time | `(1, 10)` |
| `release_time_range` | Minimum and maximum release time | `(0, 30)` |
| `weight_range` | Minimum and maximum job weight | `(1, 5)` |
| `due_date_slack_range` | Minimum and maximum slack beyond release plus processing | `(0, 20)` |
| `sequence_setup_time_range` | Minimum and maximum directed setup time | `(0, 5)` |

For job \(j\), the due date is generated as

```text
due_date[j] = release_time[j] + processing_time[j] + sampled slack[j]
```

Setup times are generated independently and do not change this due-date
calculation. The resulting data can support maximum tardiness, total tardiness,
or total weighted tardiness. Weights affect weighted objectives only.

## Configure setup times

The `setup_mode` setting accepts three values:

| Mode | Generated data |
|---|---|
| `"none"` | `fixed_setup_time` is zero and no setup table is written. |
| `"fixed"` | Every job receives the value from `fixed_setup_time`. |
| `"sequence-dependent"` | `fixed_setup_time` is zero and a directed setup table is written. |

The sequence-dependent table includes one initial setup arc from `START` to
every job and one arc for every ordered pair of distinct jobs. It excludes
self-arcs and arcs returning to `START`.

## Output files

The default output has this structure:

```text
sm-instance-generation/generated-instances/
├── manifest.json
├── instance_001/
│   └── jobs.parquet
├── instance_002/
│   └── jobs.parquet
└── instance_003/
    └── jobs.parquet
```

An instance also contains `setup_times.parquet` when `setup_mode` is `"sequence-dependent"`.

The jobs table has six columns:

| Column | Type | Meaning |
|---|---|---|
| `job` | String | Stable identifier such as `J001` |
| `processing_time` | Integer | Nonpreemptive processing requirement |
| `release_time` | Integer | Earliest time at which the job is available |
| `due_date` | Integer | Target completion time |
| `weight` | Integer | Relative cost used by weighted objectives |
| `fixed_setup_time` | Integer | Constant setup time, or zero when inactive |

The sequence-dependent setup table contains `from_job`, `to_job`, and
`setup_time`. Its long-form rows can be filtered directly or converted into a
lookup dictionary.

The manifest records the seed, generation settings, job counts, setup mode,
and relative file paths. It contains no timestamp or absolute path.

## Load an instance

Polars reads a generated jobs table with:

```python
from pathlib import Path

import polars as pl


jobs_path = (
    Path("sm-instance-generation")
    / "generated-instances"
    / "instance_001"
    / "jobs.parquet"
)
jobs = pl.read_parquet(jobs_path)
print(jobs)
```

For a sequence-dependent instance, a setup lookup can be constructed with:

```python
setup_times = pl.read_parquet(jobs_path.parent / "setup_times.parquet")
setup_lookup = {
    (row["from_job"], row["to_job"]): row["setup_time"]
    for row in setup_times.iter_rows(named=True)
}
```

## Reproducibility and reruns

The same generation settings and seed reproduce the same logical instance data
when the committed Pixi environment is used. The output directory may differ.
The generator refuses to write into a nonempty output directory, which prevents
stale files from being mixed with a new batch. A rerun requires an empty
directory or a different `output_directory` value.

The default generated-output directory is ignored by Git.
