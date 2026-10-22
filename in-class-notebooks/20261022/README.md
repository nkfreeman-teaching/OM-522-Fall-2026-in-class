# OM 522 Distribution Fitting Demo

A simulation needs a way to generate realistic values for each uncertain input,
such as the time between customer arrivals or a machine's processing time. In
practice those inputs arrive as files of historical observations rather than as
a named distribution with known parameters. This project takes such a file and
finds the distribution that best fits it, using SciPy.

## Run the demo

```bash
pixi install
pixi run demo
```

Check the notebook without changing it with:

```bash
pixi run check
```

## The data

`data-samples/` holds eight files, `sample_01.csv` through `sample_08.csv`, with
100 to 300 observations each in a single column named `x`. Each file was drawn
from a different distribution, and the notebook does not say which.

## What the notebook shows

`distribution_fitting.py` reads the first file (change the index in the data
cell to try another) and draws its histogram three ways, as counts, as a density
that approximates the probability density function, and cumulatively, which
approximates the cumulative distribution function. It then fits a normal
distribution with `st.norm.fit`, draws 200 values from the fitted distribution,
and compares them with the data in overlaid histograms and with the
Kolmogorov-Smirnov and Anderson-Darling statistics, where smaller means closer.

The last section repeats the fit for every continuous distribution in
`scipy.stats` and ranks them by the Anderson-Darling statistic. The search skips
`studentized_range` and `levy_stable`, which take about 6 minutes and 1 minute to
fit, while the remaining 106 distributions take about 10 seconds. Every random
draw uses the seed 522, so a rerun gives the same table.
