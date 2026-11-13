import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from flowsim import Model, compare, exponential

    return Model, compare, exponential, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Push against a limit on work in the shop

    Orders arrive about every 7.5 minutes, and the stations take 4, 6, and 5 minutes on average, so S2 is 80% busy.
    """)
    return


@app.cell
def _(Model, exponential):
    def shop(rack):
        if rack is None:
            first_limit = {}
            later_limit = {}
        else:
            first_limit = {"buffer": 0, "external_overflow": "wait"}
            later_limit = {"buffer": rack, "external_overflow": "wait"}

        model = Model()
        orders = model.source(
            name="Orders",
            interarrival=exponential(mean=7.5),
        )
        first = model.station(
            name="S1",
            capacity=1,
            service_time=exponential(mean=4),
            **first_limit,
        )
        second = model.station(
            name="S2",
            capacity=1,
            service_time=exponential(mean=6),
            **later_limit,
        )
        third = model.station(
            name="S3",
            capacity=1,
            service_time=exponential(mean=5),
            **later_limit,
        )
        model.route([orders, first, second, third, model.exit])
        return model

    return (shop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## One hundred matched 8-hour shifts
    """)
    return


@app.cell
def _(compare, shop):
    policies = compare(
        scenarios={
            "Push": shop(None),
            "Limit, racks of 2": shop(2),
            "Limit, racks of 1": shop(1),
        },
        duration=480,
        replications=100,
        seed=522,
        drain=True,
        keep_records=False,
    )
    return (policies,)


@app.cell
def _(pl, policies):
    columns = []
    for _metric in ["mean_wip", "mean_flow_time", "mean_outside_wait", "completed_at_close", "overtime"]:
        columns.append(
            policies.intervals(metric=_metric).select(
                "scenario",
                pl.col("mean").alias(_metric),
            )
        )
    policy_table = columns[0]
    for _column in columns[1:]:
        policy_table = policy_table.join(_column, on="scenario", maintain_order="left")

    policy_table.select(
        pl.col("scenario").alias("policy"),
        pl.col("mean_wip").alias("jobs_in_shop"),
        (pl.col("mean_flow_time") - pl.col("mean_outside_wait")).alias("minutes_in_shop"),
        pl.col("mean_flow_time").alias("minutes_from_order"),
        pl.col("completed_at_close").alias("jobs_in_shift"),
        pl.col("overtime").alias("overtime_minutes"),
    ).with_columns(pl.exclude("policy").round(1))
    return


if __name__ == "__main__":
    app.run()
