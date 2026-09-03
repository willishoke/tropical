#!/usr/bin/env python3
"""Focused gates for the oscillator-saturation harness.

These test the harness's own reasoning — fixture construction, the saturation
arithmetic, and the capacity/crossover reporting discipline. They do not
compile or measure anything, so they run in milliseconds and are safe in CI.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run  # noqa: E402


def point(n: int, backend: str, sat_median: float,
          sat_p99: float | None = None, timed_out: bool = False,
          failed: bool = False):
    if timed_out:
        return {"oscillators": n, "backend": backend, "emit_timed_out": True,
                "emit_failed": False}
    if failed:
        return {"oscillators": n, "backend": backend, "emit_timed_out": False,
                "emit_failed": True,
                "emit_failure_detail": "pipeline state failed: Compute function "
                                       "exceeds available stack space"}
    return {
        "oscillators": n,
        "backend": backend,
        "emit_timed_out": False,
        "emit_failed": False,
        "saturation": {
            "median": sat_median,
            "p95": sat_median,
            "p99": sat_p99 if sat_p99 is not None else sat_median,
            "max": sat_p99 if sat_p99 is not None else sat_median,
        },
    }


class GraphFixture(unittest.TestCase):
    def test_node_count_and_shape(self):
        graph = run.build_graph(8, morph=0.0, base_freq=55.0, spread=0.017)
        kinds = [n["kind"] for n in graph["nodes"]]
        self.assertEqual(kinds.count("source"), 8)
        self.assertEqual(kinds.count("mix"), 1)
        self.assertEqual(kinds.count("out"), 1)
        self.assertEqual(graph["out"], "out")

    def test_every_oscillator_reaches_the_mix(self):
        graph = run.build_graph(5, 0.0, 55.0, 0.017)
        mix = next(n for n in graph["nodes"] if n["kind"] == "mix")
        self.assertEqual(mix["in"]["in"], [f"osc{i}" for i in range(5)])

    def test_frequencies_are_distinct_so_cse_cannot_collapse_voices(self):
        """A shared frequency would let the compiler dedupe voices, so the
        sweep would scale node count without scaling real work."""
        graph = run.build_graph(64, 0.0, 55.0, 0.017)
        freqs = [n["params"]["freq"] for n in graph["nodes"]
                 if n["kind"] == "source"]
        self.assertEqual(len(set(freqs)), 64)

    def test_morph_is_carried_to_every_voice(self):
        graph = run.build_graph(3, 0.6, 55.0, 0.017)
        for node in graph["nodes"]:
            if node["kind"] == "source":
                self.assertEqual(node["params"]["morph"], 0.6)


class SaturationArithmetic(unittest.TestCase):
    def test_duty_is_wall_over_deadline(self):
        self.assertAlmostEqual(run.duty(5_805_000, 11_610_000), 0.5)

    def test_duty_is_none_without_a_measurement(self):
        self.assertIsNone(run.duty(None, 11_610_000))

    def test_duty_is_none_without_a_deadline(self):
        self.assertIsNone(run.duty(1000, 0))

    def test_summary_reports_order_statistics(self):
        stats = run.summary([10, 20, 30, 40])
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min"], 10)
        self.assertEqual(stats["max"], 40)
        self.assertEqual(stats["median"], 25)


class CapacityReporting(unittest.TestCase):
    def test_brackets_the_threshold_crossing(self):
        points = [point(64, "jit", 0.02), point(256, "jit", 0.11),
                  point(512, "jit", 0.55)]
        result = run.capacity(points, 0.50, "median")
        self.assertEqual(result["status"], "bracketed")
        self.assertEqual(result["capacity_lower_measured"], 256)
        self.assertEqual(result["capacity_upper_measured"], 512)

    def test_interpolated_estimate_is_labelled_and_inside_the_bracket(self):
        points = [point(256, "jit", 0.11), point(512, "jit", 0.55)]
        result = run.capacity(points, 0.50, "median")
        estimate = result["interpolated_estimate"]
        self.assertGreater(estimate, 256)
        self.assertLess(estimate, 512)
        self.assertIn("not a measured count",
                      result["interpolated_estimate_note"])

    def test_reports_when_the_sweep_never_reaches_the_threshold(self):
        points = [point(1, "jit", 0.01), point(64, "jit", 0.02)]
        result = run.capacity(points, 0.50, "median")
        self.assertEqual(result["status"], "threshold_not_reached_within_sweep")
        self.assertEqual(result["largest_measured"], 64)
        self.assertNotIn("interpolated_estimate", result)

    def test_reports_when_even_the_smallest_count_is_over(self):
        points = [point(1, "metal", 0.7), point(4, "metal", 0.9)]
        result = run.capacity(points, 0.50, "median")
        self.assertEqual(
            result["status"],
            "threshold_exceeded_at_smallest_measured_count")

    def test_timed_out_points_are_excluded_from_capacity(self):
        points = [point(256, "metal", 0.2),
                  point(512, "metal", 0.0, timed_out=True)]
        result = run.capacity(points, 0.50, "median")
        self.assertEqual(result["status"], "threshold_not_reached_within_sweep")
        self.assertEqual(result["largest_measured"], 256)

    def test_backend_refusal_is_excluded_from_capacity(self):
        """Metal refuses a kernel exceeding its per-thread stack. That is the
        backend's structural ceiling, not a measurement of saturation."""
        points = [point(2048, "metal", 0.58),
                  point(3072, "metal", 0.0, failed=True)]
        result = run.capacity(points, 0.50, "median")
        self.assertEqual(result["status"],
                         "threshold_exceeded_at_smallest_measured_count")

    def test_refused_point_does_not_become_a_capacity_bracket(self):
        points = [point(1024, "metal", 0.27), point(2048, "metal", 0.58),
                  point(3072, "metal", 0.0, failed=True)]
        result = run.capacity(points, 0.50, "median")
        self.assertEqual(result["capacity_lower_measured"], 1024)
        self.assertEqual(result["capacity_upper_measured"], 2048)

    def test_no_measurements_is_reported_not_inferred(self):
        result = run.capacity([], 0.50, "median")
        self.assertEqual(result["status"], "no_measurements")


class CrossoverReporting(unittest.TestCase):
    def test_finds_the_first_count_where_metal_is_cheaper(self):
        rows = {
            "jit": [point(64, "jit", 0.02), point(256, "jit", 0.11),
                    point(512, "jit", 0.55)],
            "metal": [point(64, "metal", 0.08), point(256, "metal", 0.20),
                      point(512, "metal", 0.35)],
        }
        result = run.crossover(rows, "median")
        self.assertEqual(result["status"], "metal_wins_from")
        self.assertEqual(result["metal_cheaper_from"], 512)

    def test_reports_jit_cheaper_everywhere_when_it_is(self):
        rows = {
            "jit": [point(64, "jit", 0.02)],
            "metal": [point(64, "metal", 0.08)],
        }
        result = run.crossover(rows, "median")
        self.assertEqual(result["status"],
                         "jit_cheaper_at_every_measured_count")
        self.assertIsNone(result["metal_cheaper_from"])

    def test_compares_only_counts_measured_on_both_backends(self):
        rows = {
            "jit": [point(64, "jit", 0.02), point(512, "jit", 0.55)],
            "metal": [point(64, "metal", 0.08)],
        }
        result = run.crossover(rows, "median")
        self.assertEqual(result["shared_counts"], [64])

    def test_refused_points_are_excluded_from_crossover(self):
        rows = {"jit": [point(3072, "jit", 0.90)],
                "metal": [point(3072, "metal", 0.0, failed=True)]}
        self.assertEqual(run.crossover(rows, "median")["status"],
                         "no_shared_measured_counts")

    def test_no_shared_counts_is_reported(self):
        rows = {"jit": [point(512, "jit", 0.55)],
                "metal": [point(64, "metal", 0.08)]}
        self.assertEqual(run.crossover(rows, "median")["status"],
                         "no_shared_measured_counts")


class EnvironmentPolicy(unittest.TestCase):
    def test_inherited_tropical_variables_are_stripped(self):
        import os
        os.environ["TROPICAL_SHOULD_NOT_SURVIVE"] = "1"
        try:
            env = run.benchmark_env({})
            self.assertNotIn("TROPICAL_SHOULD_NOT_SURVIVE", env)
        finally:
            del os.environ["TROPICAL_SHOULD_NOT_SURVIVE"]

    def test_explicit_controls_are_set(self):
        env = run.benchmark_env({})
        self.assertEqual(env["TROPICAL_STAGE0"], "1")
        self.assertEqual(env["TROPICAL_JIT_OPT_LEVEL"], "O2")

    def test_banks_unroll_stays_absent_so_banked_is_the_default(self):
        self.assertNotIn("TROPICAL_BANKS_UNROLL", run.benchmark_env({}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
