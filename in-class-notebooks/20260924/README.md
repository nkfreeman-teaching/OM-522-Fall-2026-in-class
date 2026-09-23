# September 24 materials

This lesson compares single-machine scheduling methods across many instances. The class
directs an AI agent to build a tool that applies dispatching rules and improvement searches
to every instance, checks each schedule independently, and summarizes which approach holds
up across different operating conditions.

## Materials

- [`build-prompt.md`](build-prompt.md) is the request given to the agent.
- [`data/`](data/) holds the 30 demonstration instances described below.
- [`generate_demo_instances.py`](generate_demo_instances.py) regenerates the data.
- The [September 8 notebook](../20260908/critical-ratio.py) implements release-aware Critical
  Ratio dispatching, and the [TSP notebook](../TSP/tsp_demo.py) contains the pairwise-interchange
  and subsequence-reversal search that the tool reuses.

## The demonstration data

The instances fall into three scenario families of 10 instances each. Every instance has
between 40 and 80 jobs, so the comparison also covers problem size.

| Family | Situation | What it tests |
|---|---|---|
| `backlog` | All jobs arrive in the first 30 time units, so most finish late. | Whether the starting rule still matters after improvement when the machine is overloaded. |
| `steady` | Arrivals spread over the horizon, so the machine is sometimes idle. | Whether a method that works under a backlog still works when timing matters. |
| `rush` | Steady arrivals in which 20 percent of jobs are short, heavily weighted rush orders with tight due dates. | Whether a rule that ignores weights can protect the orders that matter most. |

Each instance is `data/<family>/instance_XXX/jobs.parquet` and uses the same columns as the
September 1 instances:

| Column | Meaning |
|---|---|
| `job` | Job identifier, `J001` onward |
| `processing_time` | Processing time $p_j$ |
| `release_time` | Earliest start $r_j$ |
| `due_date` | Due date $d_j$, equal to release plus processing time plus a sampled slack |
| `weight` | Tardiness weight $w_j$; regular jobs have weights of 1 to 5 (1 to 3 in `rush`), and rush orders have 8 to 10 |
| `fixed_setup_time` | Always 0 in this lesson |

`data/manifest.json` records every family's settings, seed, and job counts.

## Checking the tool

Critical Ratio on the published instance
`sm-instance-generation/generated-instances-20260901/instance_001` must reproduce the
September 8 result: total weighted tardiness 11,314, makespan 271, and 47 tardy jobs. Any
tool that disagrees has a defect in its dispatching or evaluation code.

## Regenerate the data

From the repository root, remove `in-class-notebooks/20260924/data/` and run:

```bash
pixi run lecture-20260924-data
```

The generator uses one fixed seed per family, so it recreates identical files. It refuses to
write into a folder that already contains data.
