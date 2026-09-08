import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Single-machine scheduling with the critical-ratio rule

    At each decision time $t$, select the available unscheduled job with
    the smallest critical ratio:

    \[
    CR_j(t) = \frac{d_j - t}{p_j}.
    \]

    Jobs are scheduled without preemption. If no job is available, the
    machine clock advances to the next release time. Job ID breaks ties so
    that repeated runs produce the same schedule.
    """)
    return


@app.cell
def _(Path, mo):
    pseudocode_path = Path(__file__).with_name("pseudocode.png")
    mo.image(
        src=str(pseudocode_path),
        width=800,
    )
    return


@app.cell
def _(Path, pl):
    # Resolve the shared instance relative to this notebook, not the launch directory.
    project_root = Path(__file__).resolve().parents[2]
    data_path = (
        project_root
        / "sm-instance-generation"
        / "generated-instances-20260901"
        / "instance_001"
        / "jobs.parquet"
    )
    jobs = pl.read_parquet(data_path)

    # The ratio divides by processing time, and job IDs identify unscheduled jobs.
    assert jobs["processing_time"].min() > 0
    assert jobs["job"].n_unique() == jobs.height
    return (jobs,)


@app.cell
def _(jobs, pl):
    unscheduled_jobs = set(jobs["job"].to_list())
    schedule_rows = []
    current_time = 0

    while unscheduled_jobs:
        available_jobs = jobs.filter(
            pl.col("job").is_in(unscheduled_jobs),
            pl.col("release_time") <= current_time,
        )

        if available_jobs.is_empty():
            # Jump directly to the next release rather than checking idle periods one by one.
            current_time = (
                jobs
                .filter(pl.col("job").is_in(unscheduled_jobs))
                .select(pl.col("release_time").min())
                .item()
            )
            continue

        ranked_jobs = (
            available_jobs
            .with_columns(
                (
                    (pl.col("due_date") - current_time)
                    / pl.col("processing_time")
                ).alias("critical_ratio"),
            )
            .sort(
                by=["critical_ratio", "job"],
                descending=False,
            )
        )
        selected_job = ranked_jobs.row(0, named=True)

        start_time = current_time
        completion_time = current_time + selected_job["processing_time"]
        schedule_rows.append(
            {
                "job": selected_job["job"],
                "release_time": selected_job["release_time"],
                "processing_time": selected_job["processing_time"],
                "start": start_time,
                "completion": completion_time,
                "due_date": selected_job["due_date"],
                "weight": selected_job["weight"],
                "critical_ratio_at_dispatch": selected_job["critical_ratio"],
            }
        )
        unscheduled_jobs.remove(selected_job["job"])
        current_time = completion_time

    schedule = (
        pl.DataFrame(schedule_rows)
        .with_columns(
            (pl.col("completion") - pl.col("due_date")).alias("lateness"),
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("lateness"),
                pl.lit(0),
            ).alias("tardiness"),
        )
        .with_columns(
            (pl.col("weight") * pl.col("tardiness")).alias(
                "weighted_tardiness"
            ),
        )
    )
    total_weighted_tardiness = schedule["weighted_tardiness"].sum()
    return schedule, total_weighted_tardiness


@app.cell
def _(mo, schedule, total_weighted_tardiness):
    mo.vstack(
        [
            mo.md(
                f"""
    ## Result

    The critical-ratio schedule has **total weighted tardiness
    {total_weighted_tardiness:,.0f}** across {schedule.height} jobs.
    """
            ),
            schedule,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
