# OM 522, November 17: variability

This folder has its own environment and a copy of the FlowSim library (version 2.0.2), so it runs
with nothing else installed.

| File | Open it with | What it is |
|---|---|---|
| `variability.py` | `pixi run class` | November 12's 4, 6, and 5 minute line held at 1 to 6 jobs and released every minute, then thirty runs of the variability experiment that Quiz 4 uses, against the waiting-time formula |
| `make_seed_data.py` | `pixi run python make_seed_data.py` | Writes `data/variability-seeds.csv`: the Quiz 4 cases for seeds 1234 to 1263 |

Run `pixi install` once, from this folder, before the first command.

```bash
cd in-class-notebooks/20261117
pixi install
pixi run class
```
