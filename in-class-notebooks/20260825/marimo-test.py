import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    import numpy as np
    import polars as pl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})
    return LinearSegmentedColormap, Patch, Path, math, mo, np, pl, plt, sns


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
          :root { --ink:#17202a; --blue:#1665d8; }
          h1, h2, h3 { color:var(--ink); letter-spacing:-0.02em; }
          .hero { padding:2.2rem 2.4rem; border-radius:18px; color:white;
                  background:linear-gradient(125deg,#102a43 0%,#1665d8 58%,#00a58e 100%);
                  margin-bottom:1.25rem; box-shadow:0 12px 32px #102a4320; }
          .hero h1 { color:white; margin:0 0 .35rem 0; font-size:2.35rem; }
          .hero p { margin:0; opacity:.9; font-size:1.05rem; }
          .note { border-left:4px solid var(--blue); background:#f4f8ff; padding:.8rem 1rem; border-radius:7px; }
          .pill { display:inline-block; padding:.18rem .55rem; border-radius:999px; background:#e9f2ff;
                  color:#0b4da2; font-weight:650; margin-right:.3rem; font-size:.86rem; }
          table { font-size:.92rem; }
        </style>
        <section class="hero">
          <h1>Dispatch Rules Under Pressure</h1>
          <p>A reproducible single-machine comparison of nine common sequencing policies on 30 production jobs</p>
        </section>
        """
    )
    return


@app.cell
def _(Path, pl):
    # Published notebooks are nested under in-class-notebooks/YYYYMMDD/.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "20260825-demo" / "production_jobs_30.csv"
    jobs = (
        pl.read_csv(data_path)
        .with_row_index("Original_Order", offset=1)
        .with_columns((4 - pl.col("Priority")).alias("Weight"))
    )
    return (jobs,)


@app.cell
def _(jobs, mo):
    total_work = jobs["Processing_Time"].sum()
    earliest_due = jobs["Due_Date"].min()
    latest_due = jobs["Due_Date"].max()
    mo.md(
        f"""
        ## Experiment design

        <span class="pill">{jobs.height} jobs</span>
        <span class="pill">{total_work} time units of work</span>
        <span class="pill">due dates {earliest_due}–{latest_due}</span>
        <span class="pill">all jobs available at time 0</span>

        We model one continuously available machine, non-preemptive processing, zero setup time, and no
        inserted idle time. Because all jobs are ready at time zero, **flow time equals completion time**.
        Priority `1` is interpreted as highest and `3` as lowest; for weighted measures this becomes
        weight `3`, `2`, and `1`, respectively.

        <div class="note"><strong>Important:</strong> every rule has the same makespan ({total_work}).
        Sequencing changes when jobs finish—not the total workload—so makespan cannot distinguish these rules.</div>
        """
    )
    return


@app.cell
def _(pl):
    rule_catalog = pl.DataFrame(
        {
            "Rule": ["FCFS", "SPT", "LPT", "EDD", "CR", "MS", "Priority", "WSPT", "ATC"],
            "Full name": [
                "First Come, First Served", "Shortest Processing Time", "Longest Processing Time",
                "Earliest Due Date", "Critical Ratio", "Minimum Slack", "Priority then EDD",
                "Weighted Shortest Processing Time", "Apparent Tardiness Cost",
            ],
            "Decision logic": [
                "Original file order", "Smallest processing time", "Largest processing time",
                "Smallest due date", "Smallest (due − now) / processing", "Smallest due − now − processing",
                "Smallest priority class, then due date", "Smallest processing / priority weight",
                "Largest urgency index combining weight, processing time, and slack",
            ],
            "Typical objective": [
                "Simplicity / perceived fairness", "Mean flow time and WIP", "Stress-test benchmark",
                "Maximum lateness", "Dynamic due-date urgency", "Dynamic due-date urgency",
                "High-priority service", "Weighted mean flow time", "Weighted tardiness compromise",
            ],
        }
    )
    return (rule_catalog,)


@app.cell
def _(mo, rule_catalog):
    mo.vstack([
        mo.md("### Rules compared\n\nStatic rules sort once; **CR, MS, and ATC are recalculated after every completion**."),
        rule_catalog,
    ])
    return


@app.cell
def _(math, pl):
    def choose_job(remaining, rule, now):
        if rule == "FCFS":
            return min(remaining, key=lambda j: j["Original_Order"])
        if rule == "SPT":
            return min(remaining, key=lambda j: (j["Processing_Time"], j["Due_Date"], j["Original_Order"]))
        if rule == "LPT":
            return min(remaining, key=lambda j: (-j["Processing_Time"], j["Due_Date"], j["Original_Order"]))
        if rule == "EDD":
            return min(remaining, key=lambda j: (j["Due_Date"], j["Processing_Time"], j["Original_Order"]))
        if rule == "Priority":
            return min(remaining, key=lambda j: (j["Priority"], j["Due_Date"], j["Processing_Time"], j["Original_Order"]))
        if rule == "WSPT":
            return min(remaining, key=lambda j: (j["Processing_Time"] / j["Weight"], j["Due_Date"], j["Original_Order"]))
        if rule == "CR":
            return min(remaining, key=lambda j: ((j["Due_Date"] - now) / j["Processing_Time"], j["Due_Date"], j["Original_Order"]))
        if rule == "MS":
            return min(remaining, key=lambda j: (j["Due_Date"] - now - j["Processing_Time"], j["Due_Date"], j["Original_Order"]))
        if rule == "ATC":
            mean_p = sum(j["Processing_Time"] for j in remaining) / len(remaining)
            lookahead = 2.0 * mean_p

            def atc_index(j):
                slack = max(j["Due_Date"] - now - j["Processing_Time"], 0)
                return (j["Weight"] / j["Processing_Time"]) * math.exp(-slack / lookahead)

            return max(remaining, key=lambda j: (atc_index(j), -j["Due_Date"], -j["Original_Order"]))
        raise ValueError(f"Unknown rule: {rule}")

    def simulate(source_jobs, rule):
        remaining = source_jobs.to_dicts()
        rows = []
        now = 0
        position = 1
        while remaining:
            job = choose_job(remaining, rule, now)
            start = now
            now += job["Processing_Time"]
            lateness = now - job["Due_Date"]
            tardiness = max(lateness, 0)
            rows.append(
                {
                    "Rule": rule, "Position": position, "Job": job["Job"], "Product": job["Product"],
                    "Priority": job["Priority"], "Weight": job["Weight"],
                    "Processing_Time": job["Processing_Time"], "Due_Date": job["Due_Date"],
                    "Start": start, "Completion": now, "Lateness": lateness,
                    "Tardiness": tardiness, "Weighted_Tardiness": job["Weight"] * tardiness,
                    "On_Time": tardiness == 0,
                }
            )
            remaining.remove(job)
            position += 1
        return pl.DataFrame(rows)

    def summarize(schedule):
        n = schedule.height
        makespan = schedule["Completion"].max()
        return {
            "Rule": schedule["Rule"][0], "Makespan": makespan,
            "Avg_Flow_Time": schedule["Completion"].mean(),
            "Avg_WIP": schedule["Completion"].sum() / makespan,
            "Max_Lateness": schedule["Lateness"].max(),
            "Tardy_Jobs": schedule["Tardiness"].gt(0).sum(),
            "On_Time_Pct": 100 * schedule["On_Time"].sum() / n,
            "Total_Tardiness": schedule["Tardiness"].sum(),
            "Mean_Tardiness": schedule["Tardiness"].mean(),
            "Max_Tardiness": schedule["Tardiness"].max(),
            "Weighted_Tardiness": schedule["Weighted_Tardiness"].sum(),
            "Weighted_Flow_Time": (schedule["Weight"] * schedule["Completion"]).sum(),
        }

    return simulate, summarize


@app.cell
def _(jobs, pl, rule_catalog, simulate, summarize):
    rule_names = rule_catalog["Rule"].to_list()
    schedules = {rule: simulate(jobs, rule) for rule in rule_names}
    metrics = pl.DataFrame([summarize(schedules[rule]) for rule in rule_names])
    score_metrics = {
        "Avg_Flow_Time": 0.25, "Total_Tardiness": 0.25, "Weighted_Tardiness": 0.25,
        "Max_Tardiness": 0.15, "Tardy_Jobs": 0.10,
    }
    scored = metrics
    for metric, weight in score_metrics.items():
        lo, hi = metrics[metric].min(), metrics[metric].max()
        scored = scored.with_columns(
            (weight * (pl.col(metric) - lo) / (hi - lo)).alias(f"Score_{metric}")
        )
    scored = (
        scored.with_columns(pl.sum_horizontal([f"Score_{m}" for m in score_metrics]).alias("Composite_Loss"))
        .with_columns((100 * (1 - pl.col("Composite_Loss"))).alias("Composite_Score"))
        .sort("Composite_Loss")
        .with_row_index("Overall_Rank", offset=1)
    )
    return rule_names, schedules, score_metrics, scored


@app.cell
def _(mo, scored):
    best = scored.row(0, named=True)
    flow_winner = scored.sort("Avg_Flow_Time").row(0, named=True)
    tardy_winner = scored.sort("Total_Tardiness").row(0, named=True)
    weighted_winner = scored.sort("Weighted_Tardiness").row(0, named=True)
    lateness_winner = scored.sort("Max_Lateness").row(0, named=True)
    _best_max_lateness = lateness_winner["Max_Lateness"]
    lateness_ties = ", ".join(
        scored.filter(scored["Max_Lateness"] == _best_max_lateness)["Rule"].to_list()
    )
    mo.md(
        f"""
        ## Headline findings

        The balanced score ranks **{best['Rule']} first** ({best['Composite_Score']:.1f}/100). No single policy
        owns every objective, which is the central scheduling lesson:

        - **{flow_winner['Rule']}** minimizes average flow time ({flow_winner['Avg_Flow_Time']:.1f}) and therefore average WIP ({flow_winner['Avg_WIP']:.1f} jobs).
        - **{tardy_winner['Rule']}** minimizes total tardiness ({tardy_winner['Total_Tardiness']:.0f} time units).
        - **{weighted_winner['Rule']}** best protects priority-weighted due-date performance ({weighted_winner['Weighted_Tardiness']:.0f}).
        - **EDD** achieves the minimum worst lateness ({_best_max_lateness:.0f}); {lateness_ties} tie on this instance. EDD is the rule with the general single-machine maximum-lateness guarantee.

        The composite is a decision aid, not a theorem. It uses 25% each for average flow, total tardiness,
        and weighted tardiness; 15% for maximum tardiness; and 10% for tardy-job count. Change those weights
        when the operating objective changes.
        """
    )
    return


@app.cell
def _(pl, scored):
    leaderboard = scored.select(
        "Overall_Rank", "Rule", pl.col("Composite_Score").round(1).alias("Score / 100"),
        pl.col("Avg_Flow_Time").round(1).alias("Avg flow"), pl.col("Avg_WIP").round(1).alias("Avg WIP"),
        pl.col("Tardy_Jobs").alias("Tardy jobs"), pl.col("On_Time_Pct").round(1).alias("On-time %"),
        pl.col("Total_Tardiness").alias("Total tardiness"), pl.col("Max_Tardiness").alias("Max tardiness"),
        pl.col("Weighted_Tardiness").alias("Weighted tardiness"),
    )
    leaderboard
    return


@app.cell
def _(LinearSegmentedColormap, np, plt, score_metrics, scored, sns):
    heat_metrics = list(score_metrics)
    heat_labels = ["Average flow", "Total tardiness", "Weighted tardiness", "Max tardiness", "Tardy jobs"]
    heat_data = []
    ordered_rules = scored["Rule"].to_list()
    for rule in ordered_rules:
        _heat_row = scored.filter(scored["Rule"] == rule).row(0, named=True)
        heat_data.append([100 * (_heat_row[m] - scored[m].min()) / (scored[m].max() - scored[m].min()) for m in heat_metrics])
    heat_data = np.asarray(heat_data)
    good_bad = LinearSegmentedColormap.from_list("good_bad", ["#d9f3ea", "#fff2c7", "#f6c1c1"])
    fig_heat, ax_heat = plt.subplots(figsize=(10.8, 4.8))
    sns.heatmap(
        heat_data, annot=True, fmt=".0f", cmap=good_bad, vmin=0, vmax=100,
        xticklabels=heat_labels, yticklabels=ordered_rules, linewidths=.8,
        cbar_kws={"label": "Relative loss (0 = best, 100 = worst)"}, ax=ax_heat,
    )
    ax_heat.set_title("Each rule makes a different trade-off", loc="left", weight="bold", fontsize=14)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    fig_heat.tight_layout()
    fig_heat
    return


@app.cell
def _(mo):
    mo.md("""
    The heatmap rescales each column against the nine observed rules. Green means best **within this
    experiment**, not universally optimal. LPT is retained as a useful adverse benchmark: deliberately
    running long jobs first exposes how much flow time can deteriorate under a poor fit to the objective.
    """)
    return


@app.cell
def _(plt, scored):
    fig_trade, ax_trade = plt.subplots(figsize=(10.8, 5.8))
    x = scored["Avg_Flow_Time"].to_numpy()
    y = scored["Weighted_Tardiness"].to_numpy()
    sizes = 45 + 4 * scored["Tardy_Jobs"].to_numpy()
    colors = scored["Composite_Score"].to_numpy()
    scatter = ax_trade.scatter(
        x=x,
        y=y,
        s=sizes,
        c=colors,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.8,
    )
    label_offsets = {
        "SPT": (5, -15),
        "EDD": (5, -2),
        "MS": (5, 11),
        "CR": (5, 24),
        "ATC": (5, -13),
        "WSPT": (5, 7),
    }
    for _trade_row in scored.iter_rows(named=True):
        _offset = label_offsets.get(_trade_row["Rule"], (5, 5))
        ax_trade.annotate(
            text=_trade_row["Rule"],
            xy=(
                _trade_row["Avg_Flow_Time"],
                _trade_row["Weighted_Tardiness"],
            ),
            xytext=_offset,
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )
    ax_trade.set_title("Efficiency versus priority-weighted due-date performance", loc="left", weight="bold", fontsize=14)
    ax_trade.set_xlabel("Average flow time  ← lower is better")
    ax_trade.set_ylabel("Weighted tardiness  ← lower is better")
    fig_trade.colorbar(scatter, ax=ax_trade, label="Composite score")
    fig_trade.tight_layout()
    fig_trade
    return


@app.cell
def _(np, plt, scored):
    rank_rules = scored["Rule"].to_list()
    on_time = (30 - scored["Tardy_Jobs"]).to_numpy()
    tardy = scored["Tardy_Jobs"].to_numpy()
    pos = np.arange(len(rank_rules))
    fig_service, ax_service = plt.subplots(figsize=(10.8, 4.8))
    ax_service.barh(pos, on_time, color="#22a884", label="On time")
    ax_service.barh(pos, tardy, left=on_time, color="#e76f51", label="Tardy")
    ax_service.set_yticks(pos, rank_rules)
    ax_service.invert_yaxis()
    ax_service.set_xlim(0, 30)
    ax_service.set_xlabel("Jobs")
    ax_service.set_title("Service level: number of jobs completed by the due date", loc="left", weight="bold", fontsize=14)
    ax_service.legend(
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
    )
    fig_service.tight_layout()
    fig_service
    return


@app.cell
def _(mo, rule_names):
    selected_rule = mo.ui.dropdown(options=rule_names, value="ATC", label="Inspect a rule's full job sequence")
    mo.vstack([mo.md("## Sequence-level inspection"), selected_rule])
    return (selected_rule,)


@app.cell
def _(mo, pl, schedules, selected_rule):
    chosen_schedule = schedules[selected_rule.value]
    sequence_text = " → ".join(chosen_schedule["Job"].to_list())
    sequence_table = chosen_schedule.select(
        "Position", "Job", "Product", "Priority", "Processing_Time", "Due_Date", "Start", "Completion",
        pl.col("Tardiness"), pl.col("On_Time"),
    )
    mo.vstack([mo.md(f"**{selected_rule.value} sequence:** `{sequence_text}`"), sequence_table])
    return (chosen_schedule,)


@app.cell
def _(Patch, chosen_schedule, plt, selected_rule):
    product_colors = {"A": "#277da1", "B": "#43aa8b", "C": "#f9c74f", "D": "#f9844a"}
    fig_gantt, ax_gantt = plt.subplots(figsize=(11.5, 3.2))
    for _gantt_row in chosen_schedule.iter_rows(named=True):
        ax_gantt.barh(0, _gantt_row["Processing_Time"], left=_gantt_row["Start"], height=.55,
                      color=product_colors[_gantt_row["Product"]], edgecolor="white", linewidth=1)
        ax_gantt.text(_gantt_row["Start"] + _gantt_row["Processing_Time"] / 2, 0, _gantt_row["Job"],
                      ha="center", va="center", fontsize=7, weight="bold")
    ax_gantt.set_yticks([])
    ax_gantt.set_xlabel("Time")
    ax_gantt.set_title(f"{selected_rule.value} machine timeline (color = product)", loc="left", weight="bold", fontsize=14)
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=f"Product {product}",
        )
        for product, color in product_colors.items()
    ]
    ax_gantt.legend(
        handles=legend_handles,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
    )
    fig_gantt.tight_layout()
    fig_gantt
    return


@app.cell
def _(mo):
    mo.md("""
    ## What to use in practice

    - Use **SPT** when fast throughput, low average flow time, and low WIP dominate. It can postpone long jobs.
    - Use **EDD** when controlling the single worst lateness is the contractual objective. It does not explicitly minimize total tardiness.
    - Use **WSPT** when priority-weighted completion time matters; it balances short work with important work.
    - Use **ATC** when weighted tardiness is the operational concern. Its urgency changes as the clock advances.
    - Use **CR or minimum slack** for a transparent dynamic due-date signal, but monitor instability when many jobs become overdue.
    - Keep **FCFS** when simplicity and arrival-order fairness outweigh measurable performance gains.

    ### Limits and next experiment

    These conclusions are specific to one deterministic, single-machine job set. Release dates, uncertain
    processing times, sequence-dependent product changeovers, machine downtime, and precedence constraints
    can change the ranking. The most valuable extension would add a product-change setup matrix and compare
    these policies over many randomly generated demand scenarios, reporting confidence intervals rather than
    one-instance point estimates.
    """)
    return


if __name__ == "__main__":
    app.run()
