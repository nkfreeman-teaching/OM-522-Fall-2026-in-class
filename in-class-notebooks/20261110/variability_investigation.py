import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import plotly.express as px
    import polars as pl

    from flowsim import Model

    # Replace with the last four digits of your CWID, without leading zeros (0042 becomes 42).
    SEED = 1234
    return Model, SEED, mo, pl, px


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # The single-machine model

    Jobs arrive at one machine, wait in line when it is busy, and take 4 minutes each on average.
    """)
    return


@app.cell
def _(Model):
    MEAN_PROCESSING = 4

    def sampler(mean, scv):
        # A gamma with shape 1/scv has squared coefficient of variation scv. Zero means every
        # value equals the mean.
        if scv == 0:
            return mean
        return lambda rng: rng.gamma(shape=1 / scv, scale=mean * scv)

    def one_station(utilization, arrival_scv, processing_scv):
        model = Model(time_unit="minutes")
        jobs = model.source(
            name="Jobs",
            interarrival=sampler(MEAN_PROCESSING / utilization, arrival_scv),
        )
        machine = model.station(
            name="Machine",
            capacity=1,
            service_time=sampler(MEAN_PROCESSING, processing_scv),
        )
        model.route([jobs, machine, model.exit])
        return model

    return (one_station,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## One day on the machine

    One 8-hour day at 50% utilization, with random arrivals and random processing.
    """)
    return


@app.cell
def _(SEED, one_station):
    day = one_station(
        utilization=0.5,
        arrival_scv=1,
        processing_scv=1,
    ).run(duration=480, seed=SEED)
    day.plot_queue()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Measuring one setting
    """)
    return


@app.cell
def _(one_station, pl):
    def measure(utilization, arrival_scv, processing_scv, seed):
        experiment = one_station(utilization, arrival_scv, processing_scv).experiment(
            duration=4800,
            replications=10,
            seed=seed,
            warmup=480,
        )
        station = experiment.stations.select(
            pl.col("utilization").mean().alias("observed_utilization"),
            pl.col("mean_wait").mean(),
        )
        system = experiment.summary.select(
            pl.col("mean_wip").mean(),
            pl.col("throughput").mean(),
            pl.col("mean_flow_time").mean(),
        )
        return {
            "utilization": utilization,
            "arrival_scv": arrival_scv,
            "processing_scv": processing_scv,
            **station.row(0, named=True),
            **system.row(0, named=True),
        }

    return (measure,)


@app.cell
def _(SEED, measure, pl):
    pl.DataFrame(
        [
            measure(
                utilization=0.5,
                arrival_scv=1,
                processing_scv=1,
                seed=SEED,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The investigation

    Add a pair to `CASES` for each setting in the handout.
    """)
    return


@app.cell
def _(SEED, measure, pl):
    UTILIZATIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Each pair is (arrival_scv, processing_scv).
    CASES = [(1, 1)]

    results = pl.DataFrame(
        [
            measure(utilization, arrival_scv, processing_scv, SEED)
            for arrival_scv, processing_scv in CASES
            for utilization in UTILIZATIONS
        ]
    ).with_columns(
        pl.format("ca2={}, ce2={}", "arrival_scv", "processing_scv").alias("case"),
        pl.lit(SEED).alias("seed"),
    )
    results
    return (results,)


@app.cell
def _(px, results):
    px.line(
        results,
        x="utilization",
        y="mean_wait",
        color="case",
        markers=True,
        labels={"utilization": "Utilization", "mean_wait": "Mean wait (minutes)"},
    )
    return


@app.cell
def _(SEED, results):
    results.write_csv(f"variability-{SEED}.csv")
    f"Saved variability-{SEED}.csv"
    return


if __name__ == "__main__":
    app.run()
