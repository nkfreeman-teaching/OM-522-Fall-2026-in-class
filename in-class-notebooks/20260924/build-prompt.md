# Build prompt

Build a small experiment tool in this folder that compares single-machine scheduling methods
across many instances. The model is one machine, no preemption, job release times, and total
weighted tardiness. Each instance is a jobs.parquet file with job, processing_time,
release_time, due_date, and weight.

Methods: four release-aware nondelay dispatching rules, i.e., EDD (due date), SPT (processing
time), Critical Ratio ((due_date - t) / processing_time), and weighted SPT (processing_time /
weight). Break every tie by job ID. Run each rule alone and followed by each of two
improvement searches that reuse our TSP code in ../TSP/tsp_demo.py: pairwise interchange and
subsequence reversal. Sample one neighbor at a time with rng.choice(a=n, size=2,
replace=False) from np.random.default_rng(0) created fresh for each run, accept only strict
improvements in total weighted tardiness, and stop after 10,000 consecutive non-improving
neighbors.

Instances: the 30 instances in data/, organized as data/<family>/instance_XXX/jobs.parquet
with three families (backlog, steady, and rush). Read data/manifest.json and README.md for
what each family represents. Never modify the data.

Measures for every method and instance: total weighted tardiness, maximum tardiness, total
flow time, number of tardy jobs, makespan, and runtime. Summaries by family: mean and worst
percentage above the best value found on each instance, number of wins, mean rank, and which
rule is best on each measure before improvement.

Checks: an independent validator that confirms every job appears exactly once, no job starts
before its release, jobs do not overlap, and the recomputed objective equals the reported
one. Acceptance test: Critical Ratio on
../../sm-instance-generation/generated-instances-20260901/instance_001/jobs.parquet must give
total weighted tardiness 11314, makespan 271, and 47 tardy jobs.

Outputs: a results CSV, a summary table, one chart comparing methods by family, tests, and a
pixi task that runs the whole study with one command. Keep the evaluator and search loop
readable. Ask me questions before you build.
