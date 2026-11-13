# OM 522, November 19: push versus a limit on work in the shop

This folder has its own Pixi environment and a copy of the FlowSim library (version 2.0.2), so with
Pixi installed it needs nothing else.

| Notebook | Open it with | What it is |
|---|---|---|
| `push_and_limits.py` | `pixi run class` | A three-station shop run two ways on the same 100 shifts of orders: push (every order enters when it arrives), and a limit (S1 holds one job, S2 and S3 have racks of 2 or 1, and the rest wait outside) |

Run these commands from the root of the in-class repository. `pixi install` is needed only the first time.

```bash
cd in-class-notebooks/20261119
pixi install
pixi run class
```
