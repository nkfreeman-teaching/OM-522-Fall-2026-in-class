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

The repository includes dated course materials and reusable utilities:

```text
OM-522-Fall-2026-in-class/
├── data/
│   └── 20260825-demo/
│       └── production_jobs_30.csv
├── in-class-notebooks/
│   └── 20260825/
│       ├── marimo-test.py
│       └── scheduling-dispatch-rules-summary.html
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
