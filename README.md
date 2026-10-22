# OM 522: Operations Scheduling Problems

This repository contains in-class materials for **OM 522** at the
University of Alabama during Fall 2026.

## Course information

- **Instructor:** Nick Freeman
- **Email:** freem028@ua.edu
- **Meeting time:** Tuesdays and Thursdays, 9:30–10:45 a.m.
- **Location:** Bidgood Hall 121
- **Office hours:** Tuesdays and Thursdays, 11:00 a.m.–noon

## Course overview

The course studies scheduling decisions in production, logistics, and service
settings. We will connect production planning, detailed scheduling, operational
uncertainty, and control while learning to formulate, implement, and evaluate
exact and heuristic solution methods.

Topics include production planning, single-machine scheduling, routing,
parallel-machine scheduling, project scheduling, job-shop scheduling, factory
dynamics, and decision-making under uncertainty.

## Repository contents

The [September 22 agentic AI guide](in-class-notebooks/20260922/README.md)
provides the slides, TSP recreation exercise, finished app, illustrated review
report, and daily AI updates skill creation prompt. Lecture notes are
distributed through Blackboard.

The repository includes dated course materials and reusable utilities:

```text
OM-522-Fall-2026-in-class/
├── data/
│   └── 20260825-demo/
│       └── production_jobs_30.csv
├── in-class-notebooks/
│   ├── 20260825/
│   │   ├── marimo-test.py
│   │   └── scheduling-dispatch-rules-summary.html
│   ├── 20260908/
│   │   ├── critical-ratio.py
│   │   ├── critical-ratio.html
│   │   └── pseudocode.png
│   ├── 20260922/
│   │   ├── README.md
│   │   ├── agentic-ai.html
│   │   ├── transcript.txt
│   │   ├── ai-update-prompt.md
│   │   └── TSP-demo/
│   ├── 20260929/
│   │   ├── README.md
│   │   ├── clarke_wright.py
│   │   └── routing_utils.py
│   ├── 20261006/
│   │   ├── README.md
│   │   ├── parallel_machines.py
│   │   └── parallel_utils.py
│   ├── 20261013/
│   │   ├── README.md
│   │   ├── eligible_machines.py
│   │   └── eligibility_utils.py
│   ├── 20261015/
│   │   ├── README.md
│   │   └── critical_path.py
│   ├── 20261022/
│   │   ├── README.md
│   │   ├── distribution_fitting.py
│   │   └── data-samples/
│   └── TSP/
│       ├── data/
│       ├── tests/
│       ├── tsp_demo.py
│       ├── tsp_utils.py
│       ├── README.md
│       ├── pixi.toml
│       └── pixi.lock
├── sm-instance-generation/
│   ├── generate_instances.py
│   └── README.md
├── pixi.toml
└── pixi.lock
```

The August 25 notebook was created directly inside `in-class-notebooks/`
during class. Its published location adds the dated `20260825/` subdirectory.
The data directory is a separate sibling of `in-class-notebooks/`, not a folder
inside it.

The `.py` file is the editable Marimo notebook. The `.html` file is a static
export that can be opened in a browser without Python or Marimo.

## Set up the course environment

Download and extract the repository, open a terminal, and change into the
repository root, the directory containing `pixi.toml`. If the path contains
spaces, put the complete path in quotation marks.

```bash
pixi install
```

Pixi reads `pixi.toml` and the committed `pixi.lock` file, then creates a local
environment for the operating system in `.pixi/`. The manifest lists the tools
the project needs; the lock file records the resolved package versions.

## August 25 notebook commands

Run these commands from the repository root:

```bash
# Open the notebook in the Marimo editor.
pixi run lecture-20260825-edit

# Check the notebook without changing it.
pixi run lecture-20260825-check

# Rebuild the static HTML summary.
pixi run lecture-20260825-export
```

You can also use the underlying programs directly, for example
`pixi run marimo edit ...` or `pixi run jupyter lab`. The named tasks above
preserve the correct dated paths.

## September 8 notebook commands

The September 8 notebook implements the dynamic critical-ratio dispatching
rule on the first generated single-machine instance. Run these commands from
the repository root:

```bash
# Open the notebook in the Marimo editor.
pixi run lecture-20260908-edit

# Check the notebook without changing it.
pixi run lecture-20260908-check

# Rebuild the static HTML notebook.
pixi run lecture-20260908-export
```

## Traveling salesman demo

The traveling salesman problem (TSP) demo is a self-contained project with 439
anonymized facility locations, a complete road-distance table, plotting
utilities, and a Marimo notebook. Run its commands from the `TSP` directory:

```bash
cd in-class-notebooks/TSP
pixi install
pixi run demo
pixi run check
pixi run test
```

The [TSP guide](in-class-notebooks/TSP/README.md) describes the data and the
general symmetric TSP represented by the instance.

## Clarke–Wright savings demo

The September 29 demo reuses the TSP store locations and road distances, adds
simulated pallet demand and one hub store per state, and sets up a Marimo
notebook for building the Clarke–Wright savings method in class. Run its
commands from the `20260929` directory:

```bash
cd in-class-notebooks/20260929
pixi install
pixi run demo
pixi run test
```

The [Clarke–Wright guide](in-class-notebooks/20260929/README.md) describes the
simulated demand and the hub rule.

## Parallel-machine demo

The October 6 demo provides ten simulated nightly queues of model-training runs
for identical GPU machines, schedule-checking and Gantt-chart helpers, and a
Marimo notebook for building LPT construction and local improvement in class.
Run its commands from the `20261006` directory:

```bash
cd in-class-notebooks/20261006
pixi install
pixi run demo
pixi run test
```

The [parallel-machine guide](in-class-notebooks/20261006/README.md) describes
the schedule format and the generated data.

## Eligible-machines demo

The October 13 demo extends the October 6 notebook to a mixed GPU fleet, in
which a run may use only machines with enough memory and the older cards take
twice as long. It provides ten simulated queues, an eligibility-aware checker,
and a Marimo notebook that adapts the lower bounds, LPT, and the improvement
moves. Its last step turns to a two-stage flow shop (prep, then training) and
checks Johnson's rule against every order of five runs. Run its commands from
the `20261013` directory:

```bash
cd in-class-notebooks/20261013
pixi install
pixi run demo
pixi run test
```

The [eligible-machines guide](in-class-notebooks/20261013/README.md) describes
the processing-time format and the generated data.

## Critical path demo

The October 15 demo schedules the launch of a new production cell with the
critical path method. A Marimo notebook stores the seven-task project as a
networkx graph, runs the forward and backward passes, and draws the network and
a Gantt chart with each task's slack. Sliders change task durations, and a last
section shows how a crew shared by two tasks lengthens the project. Run its
commands from the `20261015` directory:

```bash
cd in-class-notebooks/20261015
pixi install
pixi run demo
```

The [critical path guide](in-class-notebooks/20261015/README.md) describes
what each part of the notebook shows.

## Distribution fitting demo

The October 22 demo starts the simulation unit. It takes a file of historical
observations, fits a normal distribution to it with SciPy, measures the fit
with the Kolmogorov-Smirnov and Anderson-Darling statistics, and then fits and
ranks every continuous distribution in SciPy. Run its commands from the
`20261022` directory:

```bash
cd in-class-notebooks/20261022
pixi install
pixi run demo
```

The [distribution fitting guide](in-class-notebooks/20261022/README.md)
describes the eight data files and what each part of the notebook shows.

## Single-machine instance generation

The reusable single-machine generator creates reproducible Parquet instances
with processing times, release times, due dates, weights, and optional setup
times. Its settings appear in one configuration block inside the script.

```bash
pixi run sm-instance-generate
```

The [instance-generation guide](sm-instance-generation/README.md) describes
the settings, output schemas, setup modes, and reproducibility behavior.

In-class examples, notebooks, data, and supporting files will continue to be
added throughout the semester. Materials may change as the course progresses.

Blackboard is the authoritative source for the syllabus, required materials,
assignments, announcements, grades, and deadlines.
