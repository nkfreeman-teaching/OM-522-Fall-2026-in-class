import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from flowsim import Model, best_fit, compare, gamma

    return Model, best_fit, compare, gamma, mo, pl


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

    The four "values look rounded" warnings are expected, because the shop recorded rounded times.
    """)
    return


@app.cell
def _(best_fit, pl):
    releases = pl.read_csv("data/releases.csv", try_parse_dates=True)
    processing = pl.read_csv("data/processing_times.csv")

    # Minutes between releases on the same day, so the overnight gap is dropped.
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
    fits = {"Release gap": best_fit(release_gaps)}
    for _station in ["Cut", "Weld", "Paint"]:
        _minutes = processing.filter(pl.col("station") == _station)["minutes"].to_numpy()
        fits[_station] = best_fit(_minutes)

    pl.DataFrame(
        {
            "input": list(fits),
            "mean_minutes": [fit.mean for fit in fits.values()],
            "jobs_per_hour": [60 / fit.mean for fit in fits.values()],
            "cv": [fit.sd / fit.mean for fit in fits.values()],
        }
    ).with_columns(
        pl.col("mean_minutes", "cv").round(2),
        pl.col("jobs_per_hour").round(1),
    )
    return (fits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Cut, Weld, and Paint line
    """)
    return


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
            service_time=fits["Weld"] if weld_time is None else weld_time,
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
    ## Today's line and four ways to spend the budget, on 100 matched shifts

    Cut has unlimited space in front of it, and Weld and Paint each have a rack of 2 (8 in the last option).
    """)
    return


@app.cell
def _(compare, fits, gamma, line):
    weld_mean = fits["Weld"].mean
    weld_cv_halved = fits["Weld"].sd / weld_mean / 2
    # A gamma with shape 1/cv² has coefficient of variation cv, and this scale keeps the mean.
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
    (
        options.summary.group_by("scenario", maintain_order=True)
        .agg(pl.col("completed_at_close", "overtime", "mean_flow_time", "mean_wip").mean())
        .select(
            pl.col("scenario").alias("option"),
            pl.col("completed_at_close").alias("jobs_finished_in_shift"),
            pl.col("overtime").alias("overtime_minutes"),
            pl.col("mean_flow_time").alias("minutes_in_line"),
            pl.col("mean_wip").alias("jobs_in_line"),
        )
        .with_columns(pl.exclude("option").round(1))
    )
    return


if __name__ == "__main__":
    app.run()
