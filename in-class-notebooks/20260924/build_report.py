"""Build a self-contained decision report from the verified scheduling results."""

import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAMILIES = ("backlog", "steady", "rush")
METRICS = ("completion", "weighted_completion", "lateness", "weighted_tardiness")
RULES = ("EDD", "SPT", "CR", "WSPT")
NEIGHBORHOODS = ("adjacent", "swap", "reversal")


def summarize(rows: list[dict]) -> dict:
    target_best = {}
    all_best = {}
    for row in rows:
        key = (row["family"], row["instance"], row["objective"])
        target_best[key] = min(target_best.get(key, float("inf")), row["metrics"][row["objective"]])
        for metric in METRICS:
            key = (row["family"], row["instance"], metric)
            all_best[key] = min(all_best.get(key, float("inf")), row["metrics"][metric])

    cells = defaultdict(list)
    compact_rows = []
    for row in rows:
        key = (row["family"], row["instance"], row["objective"])
        best = target_best[key]
        gap = (row["metrics"][row["objective"]] - best) / max(1, abs(best))
        method = f"{row['rule']}|{row['neighborhood']}"
        cells[(method, row["family"], row["objective"])].append(gap)
        compact_rows.append({
            "f": row["family"],
            "i": row["instance"],
            "m": method,
            "o": row["objective"],
            "v": row["metrics"],
            "g": gap,
            "s": row["seconds"],
            "e": row["evaluations"],
            "a": row["accepted_moves"],
        })

    method_cells = {}
    for rule in RULES:
        for neighborhood in NEIGHBORHOODS:
            method = f"{rule}|{neighborhood}"
            method_cells[method] = {
                family: {
                    metric: sum(cells[(method, family, metric)]) / len(cells[(method, family, metric)])
                    for metric in METRICS
                }
                for family in FAMILIES
            }

    # Resample entire paired instance records within each family. Each draw
    # retains all methods and objectives for that instance.
    by_instance = defaultdict(dict)
    for row in compact_rows:
        by_instance[(row["f"], row["i"])][(row["m"], row["o"])] = row["g"]
    family_cases = {
        family: [records for (name, _), records in by_instance.items() if name == family]
        for family in FAMILIES
    }
    methods = list(method_cells)
    rng = random.Random(9242026)
    winner_counts = {method: 0 for method in methods}
    scores_by_method = {method: [] for method in methods}
    for _ in range(2000):
        sampled = {
            family: [rng.choice(family_cases[family]) for _ in family_cases[family]]
            for family in FAMILIES
        }
        scores = {}
        for method in methods:
            score = max(
                sum(instance[(method, metric)] for instance in sampled[family]) / len(sampled[family])
                for family in FAMILIES
                for metric in METRICS
            )
            scores[method] = score
            scores_by_method[method].append(score)
        winner_counts[min(methods, key=lambda method: (scores[method], method))] += 1

    uncertainty = {}
    for method in methods:
        sorted_scores = sorted(scores_by_method[method])
        uncertainty[method] = {
            "winner_frequency": winner_counts[method] / 2000,
            "worst_gap_90_interval": [sorted_scores[100], sorted_scores[1899]],
        }
    return {
        "families": FAMILIES,
        "metrics": METRICS,
        "methods": methods,
        "cells": method_cells,
        "rows": compact_rows,
        "best_all": {"|".join(key): value for key, value in all_best.items()},
        "uncertainty": uncertainty,
        "total_evaluations": sum(row["evaluations"] for row in rows),
        "total_search_seconds": sum(row["seconds"] for row in rows),
    }


def main() -> None:
    source = json.loads((ROOT / "output" / "study-results.json").read_text())
    data = summarize(source["results"])
    template = (ROOT / "report_template.html").read_text()
    html = (
        template
        .replace("__REPORT_CSS__", (ROOT / "report.css").read_text())
        .replace("__REPORT_DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__REPORT_JS__", (ROOT / "report.js").read_text())
    )
    output = ROOT / "scheduling-study.html"
    output.write_text(html)
    print(f"Wrote {output} ({len(html):,} characters)")
    ranked = sorted(
        data["methods"],
        key=lambda method: max(
            data["cells"][method][family][metric]
            for family in FAMILIES
            for metric in METRICS
        ),
    )
    for method in ranked[:3]:
        worst = max(
            data["cells"][method][family][metric]
            for family in FAMILIES
            for metric in METRICS
        )
        print(f"{method}: worst cell {worst:.2%}, bootstrap win {data['uncertainty'][method]['winner_frequency']:.1%}")


if __name__ == "__main__":
    main()
