import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    import flowsim
    from flowsim import Model, compare, gamma

    return Model, compare, flowsim, gamma, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Where should the next dollar go on the Cut, Weld, and Paint line?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fit the November 3 shop data
    """)
    return


@app.cell
def _(flowsim, pl):
    releases = pl.read_csv("data/releases.csv", try_parse_dates=True)
    processing = pl.read_csv("data/processing_times.csv")

    release_gaps = (
        releases.sort("released")
        .with_columns(
            (pl.col("released").diff().dt.total_seconds() / 60)
            .over(pl.col("released").dt.date())
            .alias("gap"),
        )
        .drop_nulls("gap")["gap"]
        .to_numpy()
    )
    fits = {"Release gap": flowsim.best_fit(release_gaps)}
    for _station in ["Cut", "Weld", "Paint"]:
        _minutes = processing.filter(pl.col("station") == _station)["minutes"].to_numpy()
        fits[_station] = flowsim.best_fit(_minutes)

    pl.DataFrame(
        {
            "input": list(fits),
            "mean_minutes": [fit.mean for fit in fits.values()],
            "jobs_per_hour": [60 / fit.mean for fit in fits.values()],
            "cv": [fit.sd / fit.mean for fit in fits.values()],
        }
    ).with_columns(pl.exclude("input").round(2))
    return (fits,)


@app.cell
def _(Model, fits):
    def line(cut_machines=1, weld_machines=1, weld_time=None, rack=2):
        model = Model()
        jobs = model.source(
            name="Jobs",
            interarrival=fits["Release gap"],
        )
        cut = model.station(
            name="Cut",
            capacity=cut_machines,
            service_time=fits["Cut"],
        )
        weld = model.station(
            name="Weld",
            capacity=weld_machines,
            service_time=weld_time or fits["Weld"],
            buffer=rack,
            external_overflow="wait",
        )
        paint = model.station(
            name="Paint",
            capacity=1,
            service_time=fits["Paint"],
            buffer=rack,
            external_overflow="wait",
        )
        model.route([jobs, cut, weld, paint, model.exit])
        return model

    return (line,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Five ways to spend the budget, on 100 matched shifts
    """)
    return


@app.cell
def _(compare, fits, gamma, line):
    weld_mean = fits["Weld"].mean
    weld_cv_halved = fits["Weld"].sd / weld_mean / 2
    steadier_weld = gamma(
        shape=1 / weld_cv_halved**2,
        scale=weld_mean * weld_cv_halved**2,
    )

    options = compare(
        scenarios={
            "Today (racks of 2 and 2)": line(),
            "A second Cut machine": line(cut_machines=2),
            "A second Weld machine": line(weld_machines=2),
            "Weld CV halved, same mean": line(weld_time=steadier_weld),
            "Racks of 8 and 8": line(rack=8),
        },
        duration=480,
        replications=100,
        seed=522,
        drain=True,
        keep_records=False,
    )
    return (options,)


@app.cell
def _(options, pl):
    columns = []
    for _metric in ["completed_at_close", "overtime", "mean_flow_time", "mean_wip"]:
        columns.append(
            options.intervals(metric=_metric).select(
                "scenario",
                pl.col("mean").alias(_metric),
            )
        )
    option_table = columns[0]
    for _column in columns[1:]:
        option_table = option_table.join(_column, on="scenario", maintain_order="left")

    option_table.select(
        pl.col("scenario").alias("option"),
        pl.col("completed_at_close").alias("jobs_in_shift"),
        pl.col("overtime").alias("overtime_minutes"),
        pl.col("mean_flow_time").alias("minutes_per_job"),
        pl.col("mean_wip").alias("jobs_in_line"),
    ).with_columns(pl.exclude("option").round(1))
    return


if __name__ == "__main__":
    app.run()
