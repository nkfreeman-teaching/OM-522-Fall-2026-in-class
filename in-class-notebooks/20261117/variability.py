import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import plotly.express as px
    import polars as pl

    from flowsim import Model

    return Model, mo, pl, px


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Thursday's line: stations of 4, 6, and 5 minutes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Holding exactly N jobs in the line
    """)
    return


@app.cell
def _(pl):
    BOTTLENECK_RATE = 10  # jobs per hour, set by the 6-minute station
    RAW_PROCESS_TIME = 15  # minutes, 4 + 6 + 5

    fixed_wip = (
        pl.DataFrame({"jobs_in_line": [1, 2, 3, 4, 5, 6]})
        .with_columns(
            pl.min_horizontal(
                pl.col("jobs_in_line") * 60 / RAW_PROCESS_TIME,
                pl.lit(BOTTLENECK_RATE),
            ).alias("jobs_per_hour"),
        )
        .with_columns(
            (pl.col("jobs_in_line") / pl.col("jobs_per_hour") * 60).alias("minutes_per_job"),
        )
    )
    fixed_wip
    return (fixed_wip,)


@app.cell
def _(fixed_wip, px):
    px.line(
        fixed_wip.unpivot(
            index="jobs_in_line",
            on=["jobs_per_hour", "minutes_per_job"],
        ),
        x="jobs_in_line",
        y="value",
        facet_col="variable",
        markers=True,
        template="simple_white",
    ).update_yaxes(
        matches=None,
        rangemode="tozero",
    ).add_vline(
        x=2.5,
        line_dash="dash",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Releasing a job every minute, with racks of 0, 1, 2, and 4
    """)
    return


@app.cell
def _(Model):
    def saturated_line(rack):
        model = Model()
        orders = model.source(
            name="Orders",
            interarrival=1,
            first_arrival=0,
        )
        first = model.station(
            name="S1",
            capacity=1,
            service_time=4,
            buffer=0,
            external_overflow="wait",
        )
        second = model.station(
            name="S2",
            capacity=1,
            service_time=6,
            buffer=rack,
            external_overflow="wait",
        )
        third = model.station(
            name="S3",
            capacity=1,
            service_time=5,
            buffer=rack,
            external_overflow="wait",
        )
        model.route([orders, first, second, third, model.exit])
        return model

    return (saturated_line,)


@app.cell
def _(pl, saturated_line):
    rack_rows = []
    for _rack in [0, 1, 2, 4]:
        _summary = (
            saturated_line(_rack)
            .run(
                duration=4800,
                seed=1,
                warmup=480,
            )
            .summary.select(
                pl.lit(_rack).alias("racks"),
                pl.col("mean_wip").alias("jobs_in_line"),
                (pl.col("throughput") * 60).alias("jobs_per_hour"),
                (pl.col("mean_flow_time") - pl.col("mean_outside_wait")).alias("minutes_in_line"),
            )
        )
        rack_rows.append(_summary)
    racks = pl.concat(rack_rows).with_columns(
        (pl.col("jobs_in_line") / pl.col("jobs_per_hour") * 60).alias("littles_law_minutes"),
    )
    racks.with_columns(pl.exclude("racks").round(2))
    return


@app.cell
def _(pl, saturated_line):
    (
        saturated_line(0)
        .run(
            duration=4800,
            seed=1,
            warmup=480,
        )
        .stations.select(
            "station",
            "utilization",
            "processing_utilization",
            "blocking_fraction",
        )
        .with_columns(pl.exclude("station").round(2))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Thirty runs of the variability experiment

    One machine with 4-minute jobs on average, run exactly as in Quiz 4 for seeds 1234 to 1263.
    """)
    return


@app.cell
def _(pl):
    seeds = pl.read_csv("data/variability-seeds.csv")
    return (seeds,)


@app.cell
def _(px, seeds):
    px.line(
        seeds,
        x="utilization",
        y="mean_wait",
        color="case",
        line_group="seed",
        facet_col="case",
        facet_col_wrap=3,
        markers=True,
        template="simple_white",
        labels={"utilization": "Utilization", "mean_wait": "Mean wait (minutes)"},
    ).update_traces(opacity=0.4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Seed 1234, mean wait in minutes
    """)
    return


@app.cell
def _(pl, seeds):
    (
        seeds.filter(pl.col("seed") == 1234)
        .pivot(
            on="case",
            index="utilization",
            values="mean_wait",
        )
        .with_columns(pl.exclude("utilization").round(1))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Wait ≈ (ca2 + ce2)/2 × u/(1 − u) × 4 minutes, for random arrivals and random processing
    """)
    return


@app.cell
def _(pl, seeds):
    (
        seeds.filter(
            pl.col("arrival_scv") == 1,
            pl.col("processing_scv") == 1,
        )
        .group_by("utilization")
        .agg(
            pl.col("mean_wait").mean().alias("thirty_seed_mean"),
            pl.col("mean_wait").min().alias("lowest_seed"),
            pl.col("mean_wait").max().alias("highest_seed"),
        )
        .with_columns(
            (pl.col("utilization") / (1 - pl.col("utilization"))).alias("u_over_1_minus_u"),
            (1 * pl.col("utilization") / (1 - pl.col("utilization")) * 4).alias("formula"),
        )
        .select(
            "utilization",
            "u_over_1_minus_u",
            "formula",
            "thirty_seed_mean",
            "lowest_seed",
            "highest_seed",
        )
        .sort("utilization")
        .with_columns(pl.exclude("utilization").round(1))
    )
    return


if __name__ == "__main__":
    app.run()
