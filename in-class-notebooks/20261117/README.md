# OM 522, November 17: variability

This folder has its own Pixi environment and a copy of the FlowSim library (version 2.0.2), so with
Pixi installed it needs nothing else.

| File | Open it with | What it is |
|---|---|---|
| `variability.py` | `pixi run class` | November 12's line (stations of 4, 6, and 5 minutes) first holding 1 to 6 jobs, then with a job arriving every minute and racks of 0 to 4, and then thirty seeds of the Quiz 4 experiment against the waiting-time formula |
| `make_seed_data.py` | `pixi run python make_seed_data.py` | Already run, and its output is `data/variability-seeds.csv` (the Quiz 4 cases for seeds 1234 to 1263). Rerun it only to regenerate that file. |

Quiz 4 itself runs in `../20261110/` with `pixi run quiz4`, as its handout describes.

Run these commands from the root of the in-class repository. `pixi install` is needed only the first time.

```bash
cd in-class-notebooks/20261117
pixi install
pixi run class
```
