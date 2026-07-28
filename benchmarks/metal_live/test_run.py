#!/usr/bin/env python3
"""Focused acceptance-gate tests for the manual Metal harness."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("metal_live_run", HERE / "run.py")
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def passing_result() -> dict:
    completed = {
        "baseline": True,
        "writes": True,
        "clock_jump": True,
        "hot_swap": True,
        "start_reference": True,
        "post_jump_reference": True,
        "midpoint_reference": True,
        "end_reference": True,
        "callback_progress_stalled": False,
    }
    return {
        "dac_aborted": False,
        "negotiated_buffer_frames": 512,
        "dac_stats": {
            "callback_count": 100,
            "underrun_count": 0,
            "overrun_count": 0,
        },
        "callback_summary_ns": {
            "count": 100,
            "p99_upper_bound_ns": 1_000_000,
        },
        "event_completion": completed,
        "event_blocks": {
            "baseline": 10,
            "clock_jump_progress": 30,
            "hot_swap_progress": 50,
            "start_reference": 11,
            "post_jump_reference": 31,
            "midpoint_reference": 51,
            "end_reference": 90,
            "end_preceding_write": 80,
        },
        "reference_labels": [
            "start",
            "post_2p40_clock_jump",
            "midpoint_after_hot_swap",
            "end",
        ],
        "reference_snr_db": [100.0, 100.0, 100.0, 100.0],
    }


class AcceptanceGateTests(unittest.TestCase):
    def evaluate(self, result: dict) -> tuple[dict[str, bool], list[str]]:
        return RUN.evaluate_acceptance(
            result,
            {"sample_count": 3, "monotonic_growth": False},
            buffer=512,
            rate=44100,
            depth=3,
        )

    def test_passing_row(self) -> None:
        gates, failures = self.evaluate(passing_result())
        self.assertTrue(all(gates.values()))
        self.assertEqual(failures, [])

    def test_low_snr_blocks_qualification(self) -> None:
        result = passing_result()
        result["reference_snr_db"][-1] = 79.999
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["all_reference_snr_at_least_80db"])
        self.assertIn("all_reference_snr_at_least_80db", failures)

    def test_p99_at_half_deadline_blocks_qualification(self) -> None:
        result = passing_result()
        result["callback_summary_ns"]["p99_upper_bound_ns"] = 5_805_000
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["callback_p99_below_half_deadline"])
        self.assertIn("callback_p99_below_half_deadline", failures)

    def test_end_capture_inside_future_queue_blocks_qualification(self) -> None:
        result = passing_result()
        result["event_blocks"]["end_preceding_write"] = 88
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["end_reference_clear_of_future_write_queue"])
        self.assertIn("end_reference_clear_of_future_write_queue", failures)

    def test_too_few_rss_samples_blocks_qualification(self) -> None:
        result = passing_result()
        gates, failures = RUN.evaluate_acceptance(
            result,
            {"sample_count": 2, "monotonic_growth": False},
            buffer=512,
            rate=44100,
            depth=3,
        )
        self.assertFalse(gates["rss_sampling_sufficient"])
        self.assertIn("rss_sampling_sufficient", failures)


if __name__ == "__main__":
    unittest.main()
