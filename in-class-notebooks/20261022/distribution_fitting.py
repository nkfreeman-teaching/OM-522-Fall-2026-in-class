import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pathlib
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import scipy.stats as st
    import seaborn as sns

    sns.set_style("whitegrid")

    # Every random draw below uses this seed, so a rerun gives the same numbers.
    SEED = 522
    return SEED, mo, np, pathlib, pl, plt, sns, st, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # From data to a distribution

    A simulation needs a way to generate realistic values for each uncertain input, e.g.,
    customer interarrival times or machine processing times. In practice those inputs arrive as
    files of historical observations, not as a named distribution with known parameters. This
    notebook, the October 22 class example, takes one such file and finds the distribution that
    best fits it.

    The steps are to (i) look at the data with a histogram, (ii) fit one candidate distribution
    and measure how far it is from the data, and (iii) repeat the fit for every continuous
    distribution in SciPy and rank them by that distance.
    """)
    return


@app.cell
def _(pathlib):
    data_directory = pathlib.Path("data-samples")
    data_filepaths = sorted(list(data_directory.glob("*.csv")))
    data_filepaths
    return (data_filepaths,)


@app.cell
def _(data_filepaths, pl):
    # Change the index to work with another of the eight files.
    data_filepath = data_filepaths[0]

    data = pl.read_csv(data_filepath)
    data_sample = data["x"].to_numpy()
    data
    return data_filepath, data_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Looking at the data

    A histogram splits the range of the data into bins and counts the observations in each one.
    With `stat="density"`, each count is divided by the number of observations times the bin
    width, so the bars have a total area of 1 and approximate the probability density function
    (PDF). With `cumulative=True`, the bars accumulate from left to right and approximate the
    cumulative distribution function (CDF), i.e., the share of observations at or below each
    value.
    """)
    return


@app.cell
def _(data_filepath, data_sample, plt, sns):
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 3.5),
    )
    sns.histplot(
        data_sample,
        bins=20,
        edgecolor="k",
        ax=_axes[0],
    )
    _axes[0].set_title("Frequency")
    sns.histplot(
        data_sample,
        bins=20,
        edgecolor="k",
        stat="density",
        ax=_axes[1],
    )
    _axes[1].set_title("Density (approximates the PDF)")
    sns.histplot(
        data_sample,
        bins=20,
        edgecolor="k",
        stat="density",
        cumulative=True,
        ax=_axes[2],
    )
    _axes[2].set_title("Cumulative (approximates the CDF)")
    _fig.suptitle(data_filepath.name)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Distribution objects in SciPy

    Each distribution in `scipy.stats` is an object. Calling it with parameters returns a
    *frozen* distribution, i.e., one whose parameters are fixed, and its `.rvs(n)` method draws
    `n` random values from it. SciPy gives every continuous distribution a location (`loc`) and a
    scale (`scale`) parameter, and for the normal these are the mean and the standard deviation.
    """)
    return


@app.cell
def _(SEED, plt, sns, st):
    my_dist = st.norm(loc=100, scale=2)

    _fig, _ax = plt.subplots(1, 1, figsize=(6, 3.5))
    sns.histplot(
        my_dist.rvs(1000, random_state=SEED),
        bins=20,
        edgecolor="k",
        ax=_ax,
    )
    _ax.set_title("1,000 draws from a normal with mean 100 and standard deviation 2")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fitting one distribution

    `st.norm.fit(data)` returns the parameters of the normal distribution that best fits the
    data. SciPy estimates them by maximum likelihood. The number of parameters differs from one
    distribution to the next, so we keep them as a tuple and unpack them with `*params` when
    freezing the fitted distribution. We then draw 200 values from it and compare those values
    with the data, using the same bins for both.
    """)
    return


@app.cell
def _(SEED, data_sample, st):
    params = st.norm.fit(data_sample)
    fitted_distribution = st.norm(*params)

    rvs = fitted_distribution.rvs(200, random_state=SEED)
    params
    return (rvs,)


@app.cell
def _(data_sample, np, plt, rvs, sns):
    # Shared bin edges, so each data bar sits directly behind its fitted bar.
    _bins = np.histogram_bin_edges(
        np.concatenate([data_sample, rvs]),
        bins=20,
    )

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11, 3.5),
    )
    for _ax, _cumulative in zip(_axes, [False, True]):
        sns.histplot(
            data_sample,
            bins=_bins,
            edgecolor="k",
            stat="density",
            cumulative=_cumulative,
            alpha=0.6,
            label="data",
            ax=_ax,
        )
        sns.histplot(
            rvs,
            bins=_bins,
            edgecolor="k",
            stat="density",
            cumulative=_cumulative,
            color="orange",
            alpha=0.6,
            label="fitted normal",
            ax=_ax,
        )
        _ax.legend()
    _axes[0].set_title("Density")
    _axes[1].set_title("Cumulative")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Measuring the fit

    Each gap between the two cumulative histograms is an error, much like a residual in a
    regression. Two standard statistics summarize those gaps between two samples, and a smaller
    value means a closer fit for both:

    - the Kolmogorov-Smirnov (KS) statistic is the largest vertical gap between the two empirical
      CDFs; and
    - the Anderson-Darling (AD) statistic combines the squared gaps across the whole range and
      weights the tails more heavily. SciPy reports a standardized version that can be negative.
    """)
    return


@app.cell
def _(data_sample, rvs, st):
    ks_test = st.ks_2samp(data_sample, rvs)
    ks_test.statistic
    return


@app.cell
def _(data_sample, rvs, st, warnings):
    with warnings.catch_warnings():
        # SciPy warns that the p-value is capped, but only the statistic is used here.
        warnings.simplefilter("ignore")
        ad_test = st.anderson_ksamp([data_sample, rvs])
    ad_test.statistic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Searching over distribution families

    The fit above is the best *normal* distribution, but another family may fit better. The
    search is an optimization problem. The decision is the distribution family, and the objective
    is to minimize the AD statistic.

    The candidates come from the module itself. `st.__dict__` maps every name defined in
    `scipy.stats` to its object, and `isinstance(obj, st.rv_continuous)` is `True` only for
    continuous distributions. We keep each name as a string, e.g., `"st.norm"`, so the results
    table can show it, and `eval` turns the string back into the object. `eval` runs any string
    as Python code, so it is safe here only because every string comes from SciPy's own names.
    Never pass text typed by a user to `eval`.
    """)
    return


@app.cell
def _(st):
    all_distributions = []
    for _name in st.__dict__.keys():
        _dist_call = f"st.{_name}"
        try:
            if isinstance(eval(_dist_call), st.rv_continuous):
                all_distributions.append(_dist_call)
        except Exception:
            pass

    # These two take about 6 minutes and 1 minute to fit, while the rest take about 8 seconds together.
    SLOW_TO_FIT = ["st.studentized_range", "st.levy_stable"]
    candidate_distributions = [
        _dist for _dist in all_distributions if _dist not in SLOW_TO_FIT
    ]
    len(all_distributions), len(candidate_distributions)
    return (candidate_distributions,)


@app.cell
def _(SEED, candidate_distributions, data_sample, st, warnings):
    fit_data = []
    for _distribution in candidate_distributions:
        try:
            with warnings.catch_warnings():
                # Many fits warn about convergence; a failed fit raises and is skipped below.
                warnings.simplefilter("ignore")
                _dist = eval(_distribution)
                _params = _dist.fit(data_sample)
                _fit_dist = _dist(*_params)
                _rvs = _fit_dist.rvs(200, random_state=SEED)
                _ad_test = st.anderson_ksamp([data_sample, _rvs])
            fit_data.append(
                {
                    "name": _distribution,
                    "AD": _ad_test.statistic,
                }
            )
        except Exception:
            # Some distributions cannot be fit to this data, so move on to the next one.
            pass
    return (fit_data,)


@app.cell
def _(fit_data, pl):
    fit_results = (
        pl.DataFrame(fit_data)
        .sort("AD")
        .with_row_index("rank", offset=1)
    )
    fit_results
    return (fit_results,)


@app.cell
def _(fit_results, mo, pl):
    _normal_rank = fit_results.filter(pl.col("name") == "st.norm")["rank"].item()
    mo.md(
        f"The normal ranks **{_normal_rank}** of the {fit_results.height} distributions "
        "that could be fit to this file."
    )
    return


if __name__ == "__main__":
    app.run()
