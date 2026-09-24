"""Small independent checks for release times and accelerated search values."""

import itertools
import unittest

from study import METRICS, NEIGHBORHOODS, candidate_value, evaluate, move, prefix_values, search


class SchedulingChecks(unittest.TestCase):
    def test_waiting_for_rush_order_can_help(self) -> None:
        jobs = [
            {"job": "A", "processing_time": 10, "release_time": 0, "due_date": 100, "weight": 1},
            {"job": "B", "processing_time": 1, "release_time": 1, "due_date": 2, "weight": 100},
        ]
        self.assertEqual(evaluate(jobs, [0, 1]), (21, 1110, 9, 900))
        self.assertEqual(evaluate(jobs, [1, 0]), (14, 212, 0, 0))

    def test_negative_maximum_lateness(self) -> None:
        jobs = [
            {"job": "A", "processing_time": 1, "release_time": 0, "due_date": 10, "weight": 1},
            {"job": "B", "processing_time": 2, "release_time": 0, "due_date": 20, "weight": 1},
        ]
        self.assertEqual(evaluate(jobs, [0, 1])[2], -9)

    def test_incremental_values_and_local_optimum(self) -> None:
        jobs = [
            {"job": "A", "processing_time": 4, "release_time": 0, "due_date": 5, "weight": 1},
            {"job": "B", "processing_time": 2, "release_time": 1, "due_date": 4, "weight": 8},
            {"job": "C", "processing_time": 3, "release_time": 2, "due_date": 8, "weight": 2},
            {"job": "D", "processing_time": 1, "release_time": 5, "due_date": 7, "weight": 3},
        ]
        ids = {job["job"]: index for index, job in enumerate(jobs)}
        for order_tuple in itertools.permutations(range(len(jobs))):
            order = list(order_tuple)
            for objective in range(len(METRICS)):
                clocks, values = prefix_values(jobs, order, objective)
                for first in range(len(order)):
                    self.assertEqual(
                        candidate_value(
                            jobs=jobs,
                            order=order,
                            objective=objective,
                            start=first,
                            clocks=clocks,
                            values=values,
                        ),
                        evaluate(jobs, order)[objective],
                    )
                for neighborhood in NEIGHBORHOODS:
                    result = search(
                        jobs=jobs,
                        initial=order,
                        objective=objective,
                        neighborhood=neighborhood,
                    )
                    final = [ids[job] for job in result["order"]]
                    self.assertEqual(result["metrics"], dict(zip(METRICS, evaluate(jobs, final))))
                    self.assertLessEqual(result["metrics"][METRICS[objective]], evaluate(jobs, order)[objective])
                    for first in range(len(final) - 1):
                        lasts = (first + 1,) if neighborhood == "adjacent" else range(first + 1, len(final))
                        for last in lasts:
                            candidate = final.copy()
                            move(candidate, first, last, neighborhood)
                            self.assertGreaterEqual(
                                evaluate(jobs, candidate)[objective],
                                result["metrics"][METRICS[objective]],
                            )


if __name__ == "__main__":
    unittest.main()
