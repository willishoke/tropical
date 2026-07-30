#!/usr/bin/env python3
"""Focused acceptance-gate tests for the manual Metal harness."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
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
        "writes_stopped": True,
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
        "device_frames": 512,
        "metal_render_tile_frames": 512,
        "metal_worker_capacity_frames": 2048,
        "negotiated_buffer_frames": 512,
        "measured_loop_ns": 1_160_997_700,
        "measured_loop_callback_count": 100,
        "disconnect_count": 0,
        "reconnect_success_count": 0,
        "reconnect_failure_count": 0,
        "ownership_failure_count": 0,
        "reference_ownership_failure_count": 0,
        "metal_dispatch_failure_count": 0,
        "metal_render_starvation_count": 0,
        "metal_epoch_tag_mismatch_count": 0,
        "metal_activation_retarget_count": 2,
        "metal_activation_failure_count": 0,
        "metal_callback_thread_violation_count": 0,
        "metal_worker_stage_times": {
            "request_received": 100,
            "activation_target_reserved": 110,
            "candidate_render_submitted": 120,
            "candidate_gpu_completion": 130,
            "candidate_window_ready": 140,
            "activation_published": 150,
            "activation_acknowledged": 160,
            "old_epoch_retired": 170,
        },
        "metal_activation_latency": {
            "count": 10,
            "total_ns": 600,
            "min_ns": 40,
            "max_ns": 80,
            "mean_ns": 60,
        },
        "metal_worker_cpu_ns": 10_000_000,
        "metal_worker_wall_ns": 100_000_000,
        "metal_worker_cpu_wall_fraction": 0.1,
        "nonfinite_count": 0,
        "reload_artifacts_distinct": True,
        "dac_stats": {
            "callback_count": 100,
            "underrun_count": 0,
            "overrun_count": 0,
        },
        "callback_summary_ns": {
            "count": 100,
            "p99_upper_bound_ns": 1_000_000,
            "max_exact_ns": 2_000_000,
        },
        "event_completion": completed,
        "event_activation_epochs": {
            "last_write": 80,
            "clock_jump": 30,
            "hot_swap": 50,
            "acknowledged_final": 90,
        },
        "transition_activation_evidence": {
            "clock_jump_requested": 10,
            "clock_jump_acknowledged": 10,
            "hot_swap_requested": 2,
            "hot_swap_acknowledged": 2,
        },
        "event_blocks": {
            "baseline": 10,
            "clock_jump_request": 26,
            "clock_jump_progress": 30,
            "hot_swap_publication": 46,
            "hot_swap_progress": 50,
            "start_reference": 11,
            "post_jump_reference": 31,
            "midpoint_reference": 51,
            "end_reference": 90,
            "duration_elapsed": 89,
            "end_preceding_write": 80,
            "last_write": 80,
            "writes_stopped": 85,
        },
        "reference_labels": [
            "start",
            "post_2p40_clock_jump",
            "midpoint_after_hot_swap",
            "end",
        ],
        "reference_snr_db": [120.0, 120.0, 120.0, 120.0],
        "reference_blocks": [11, 31, 51, 90],
        "reference_start_indices": [0, 1 << 40, 1 << 40, 1 << 40],
        "reference_max_error": [1e-15] * 4,
        "reference_signal_energy": [5.12e-14] * 4,
        "reference_checksum": [1e-7] * 4,
        "rss_blocks": [60, 70, 80],
        "rss_bytes": [100 << 20, 100 << 20, 100 << 20],
        "final_rss_bytes": 100 << 20,
        "dac_snapshots": [],
        "param_events": [
            {"discipline": "raw", "observed_sample_index": 100,
             "effective_sample_index": 1636},
            {"discipline": "glide", "observed_sample_index": 100,
             "effective_sample_index": 1636},
            {"discipline": "anchor", "observed_sample_index": 100,
             "effective_sample_index": 1636},
            {"discipline": "velocity", "observed_sample_index": 100,
             "effective_sample_index": 1636},
        ],
    }


class AcceptanceGateTests(unittest.TestCase):
    def evaluate(self, result: dict) -> tuple[dict[str, bool], list[str]]:
        return RUN.evaluate_acceptance(
            result,
            {
                "sample_count": 3,
                "all_samples_positive": True,
                "material_net_or_level_growth": False,
            },
            buffer=512,
            rate=44100,
            render_frames=512,
        )

    def test_passing_row(self) -> None:
        gates, failures = self.evaluate(passing_result())
        self.assertTrue(all(gates.values()))
        self.assertEqual(failures, [])

    def test_low_snr_blocks_qualification(self) -> None:
        result = passing_result()
        result["reference_snr_db"][-1] = 100.0
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["all_reference_snr_above_100db"])
        self.assertIn("all_reference_snr_above_100db", failures)

    def test_p99_at_half_deadline_blocks_qualification(self) -> None:
        result = passing_result()
        result["callback_summary_ns"]["p99_upper_bound_ns"] = 5_805_000
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["callback_p99_below_half_deadline"])
        self.assertIn("callback_p99_below_half_deadline", failures)

    def test_nonfinite_output_blocks_qualification(self) -> None:
        result = passing_result()
        result["nonfinite_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_nonfinite_output"])
        self.assertIn("zero_nonfinite_output", failures)

    def test_unacknowledged_last_write_blocks_final_reference(self) -> None:
        result = passing_result()
        result["event_activation_epochs"]["acknowledged_final"] = 79
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["final_reference_after_write_stop_and_last_activation"])
        self.assertIn(
            "final_reference_after_write_stop_and_last_activation",
            failures)

    def test_missing_clock_activation_evidence_blocks(self) -> None:
        result = passing_result()
        result["event_activation_epochs"]["clock_jump"] = None
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["clock_and_swap_reach_acknowledged_activations"])
        self.assertIn(
            "clock_and_swap_reach_acknowledged_activations", failures)

    def test_unacknowledged_swap_blocks(self) -> None:
        result = passing_result()
        result["event_activation_epochs"]["acknowledged_final"] = 49
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["clock_and_swap_reach_acknowledged_activations"])
        self.assertIn(
            "clock_and_swap_reach_acknowledged_activations", failures)

    def test_missing_periodic_activation_ack_blocks(self) -> None:
        result = passing_result()
        result["transition_activation_evidence"][
            "clock_jump_acknowledged"] = 9
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["every_requested_jump_and_swap_acknowledged"])
        self.assertIn(
            "every_requested_jump_and_swap_acknowledged", failures)

    def test_too_few_rss_samples_blocks_qualification(self) -> None:
        result = passing_result()
        gates, failures = RUN.evaluate_acceptance(
            result,
            {
                "sample_count": 2,
                "all_samples_positive": True,
                "material_net_or_level_growth": False,
            },
            buffer=512,
            rate=44100,
            render_frames=512,
        )
        self.assertFalse(gates["rss_sampling_sufficient"])
        self.assertIn("rss_sampling_sufficient", failures)

    def test_render_quantum_mismatch_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_render_tile_frames"] = 256
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["device_and_render_quanta_match"])
        self.assertIn("device_and_render_quanta_match", failures)

    def test_low_callback_coverage_blocks_qualification(self) -> None:
        result = passing_result()
        result["measured_loop_callback_count"] = 70
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["callback_rate_covers_measured_wall"])
        self.assertIn("callback_rate_covers_measured_wall", failures)

    def test_callback_coverage_lower_edge_is_accepted(self) -> None:
        result = passing_result()
        block_ns = 512 * 1e9 / 44100
        result["measured_loop_ns"] = 100 * block_ns
        result["measured_loop_callback_count"] = 99
        gates, _ = self.evaluate(result)
        self.assertTrue(gates["callback_rate_covers_measured_wall"])

    def test_callback_coverage_below_lower_edge_blocks(self) -> None:
        result = passing_result()
        block_ns = 512 * 1e9 / 44100
        result["measured_loop_ns"] = 100 * block_ns
        result["measured_loop_callback_count"] = 98
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["callback_rate_covers_measured_wall"])
        self.assertIn("callback_rate_covers_measured_wall", failures)

    def test_callback_coverage_upper_edge_is_accepted(self) -> None:
        result = passing_result()
        block_ns = 512 * 1e9 / 44100
        result["measured_loop_ns"] = 100 * block_ns
        result["measured_loop_callback_count"] = 101
        gates, _ = self.evaluate(result)
        self.assertTrue(gates["callback_rate_covers_measured_wall"])

    def test_callback_coverage_above_upper_edge_blocks(self) -> None:
        result = passing_result()
        block_ns = 512 * 1e9 / 44100
        result["measured_loop_ns"] = 100 * block_ns
        result["measured_loop_callback_count"] = 102
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["callback_rate_covers_measured_wall"])
        self.assertIn("callback_rate_covers_measured_wall", failures)

    def test_sticky_disconnect_blocks_even_after_recovery(self) -> None:
        result = passing_result()
        result["disconnect_count"] = 1
        result["reconnect_success_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_sticky_device_continuity_events"])
        self.assertIn("zero_sticky_device_continuity_events", failures)

    def test_live_ownership_failure_blocks_qualification(self) -> None:
        result = passing_result()
        result["ownership_failure_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_runtime_ownership_failures"])
        self.assertIn("zero_runtime_ownership_failures", failures)

    def test_reference_ownership_failure_blocks_qualification(self) -> None:
        result = passing_result()
        result["reference_ownership_failure_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_runtime_ownership_failures"])
        self.assertIn("zero_runtime_ownership_failures", failures)

    def test_metal_dispatch_failure_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_dispatch_failure_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_metal_dispatch_failures"])
        self.assertIn("zero_metal_dispatch_failures", failures)

    def test_metal_render_starvation_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_render_starvation_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_metal_render_starvations"])
        self.assertIn("zero_metal_render_starvations", failures)

    def test_starvation_phase_classification(self) -> None:
        result = passing_result()
        self.assertEqual(RUN.classify_starvation_phase(result), "none")
        result["metal_render_starvation_count"] = 1
        result["metal_render_starvation_count_before_dac_start"] = 0
        result["metal_render_starvation_count_after_dac_start"] = 1
        result["metal_render_starvation_count_before_stats_reset"] = 1
        result["metal_render_starvation_count_measured_delta"] = 0
        self.assertEqual(
            RUN.classify_starvation_phase(result), "during_dac_start")
        result["metal_render_starvation_count_after_dac_start"] = 0
        self.assertEqual(
            RUN.classify_starvation_phase(result), "startup_warmup")
        result["metal_render_starvation_count_before_stats_reset"] = 0
        result["metal_render_starvation_count_measured_delta"] = 1
        self.assertEqual(
            RUN.classify_starvation_phase(result), "measured_window")

    def test_metal_tag_mismatch_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_epoch_tag_mismatch_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_metal_epoch_tag_mismatches"])
        self.assertIn("zero_metal_epoch_tag_mismatches", failures)

    def test_metal_activation_failure_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_activation_failure_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_metal_activation_failures"])
        self.assertIn("zero_metal_activation_failures", failures)

    def test_callback_thread_metal_violation_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_callback_thread_violation_count"] = 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["zero_callback_thread_metal_violations"])
        self.assertIn("zero_callback_thread_metal_violations", failures)

    def test_callback_max_at_deadline_blocks_qualification(self) -> None:
        result = passing_result()
        result["callback_summary_ns"]["max_exact_ns"] = int(
            512 * 1e9 / 44100) + 1
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["callback_max_below_deadline"])
        self.assertIn("callback_max_below_deadline", failures)

    def test_missing_worker_stage_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_worker_stage_times"]["activation_acknowledged"] = 0
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["worker_stage_timestamps_complete_and_ordered"])
        self.assertIn(
            "worker_stage_timestamps_complete_and_ordered", failures)

    def test_missing_activation_latency_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_activation_latency"]["count"] = 0
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["activation_latency_distribution_reported"])
        self.assertIn("activation_latency_distribution_reported", failures)

    def test_missing_worker_wall_metric_blocks_qualification(self) -> None:
        result = passing_result()
        result["metal_worker_wall_ns"] = 0
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["worker_cpu_wall_fraction_reported"])
        self.assertIn("worker_cpu_wall_fraction_reported", failures)

    def test_missing_production_discipline_blocks_qualification(self) -> None:
        result = passing_result()
        result["param_events"][-1]["discipline"] = "raw"
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["all_production_param_disciplines_exercised"])
        self.assertIn("all_production_param_disciplines_exercised", failures)

    def test_missing_param_dispatch_sample_index_blocks_qualification(
            self) -> None:
        result = passing_result()
        del result["param_events"][2]["effective_sample_index"]
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["all_param_dispatch_indices_recorded"])
        self.assertIn("all_param_dispatch_indices_recorded", failures)

    def test_zero_rss_samples_block_qualification(self) -> None:
        result = passing_result()
        result["rss_bytes"] = [0, 0, 0]
        analysis = RUN.analyze_rss(result, rate=44100, buffer=512)
        gates, failures = RUN.evaluate_acceptance(
            result, analysis, buffer=512, rate=44100, render_frames=512)
        self.assertFalse(gates["rss_samples_are_valid"])
        self.assertIn("rss_samples_are_valid", failures)

    def test_parallel_checkpoint_length_mismatch_blocks(self) -> None:
        result = passing_result()
        result["reference_labels"].append("start")
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["reference_checkpoint_arrays_aligned"])
        self.assertIn("reference_checkpoint_arrays_aligned", failures)

    def test_canary_masking_bank_blocks_qualification(self) -> None:
        result = passing_result()
        result["reference_signal_energy"][-1] = 512 * (1.01e-12 ** 2)
        gates, failures = self.evaluate(result)
        self.assertFalse(gates["bank_dominates_canary_at_every_checkpoint"])
        self.assertIn("bank_dominates_canary_at_every_checkpoint", failures)

    def test_indistinguishable_reload_blocks_qualification(self) -> None:
        result = passing_result()
        result["reload_artifacts_distinct"] = False
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["replacement_ir_msl_distinct_and_midpoint_checked"])
        self.assertIn(
            "replacement_ir_msl_distinct_and_midpoint_checked", failures)

    def test_write_after_stop_contract_blocks_qualification(self) -> None:
        result = passing_result()
        result["event_blocks"]["last_write"] = 86
        result["event_blocks"]["end_preceding_write"] = 86
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["final_reference_after_write_stop_and_last_activation"])
        self.assertIn(
            "final_reference_after_write_stop_and_last_activation",
            failures)

    def test_end_capture_before_stop_queue_clear_blocks_qualification(self) -> None:
        result = passing_result()
        result["event_blocks"]["end_reference"] = 87
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["final_reference_after_write_stop_and_last_activation"])
        self.assertIn(
            "final_reference_after_write_stop_and_last_activation",
            failures)

    def test_end_capture_before_duration_elapsed_blocks_qualification(self) -> None:
        result = passing_result()
        result["event_blocks"]["duration_elapsed"] = 90
        gates, failures = self.evaluate(result)
        self.assertFalse(
            gates["final_reference_after_write_stop_and_last_activation"])
        self.assertIn(
            "final_reference_after_write_stop_and_last_activation",
            failures)

    def test_rss_net_growth_with_one_dip_is_material(self) -> None:
        analysis = RUN.analyze_rss({
            "event_blocks": {"hot_swap_progress": 50},
            "rss_blocks": [222, 300, 400, 500],
            "rss_bytes": [
                100 << 20,
                109 << 20,
                108 << 20,
                107 << 20,
            ],
        }, rate=44100, buffer=512)
        self.assertEqual(analysis["decrease_intervals"], 2)
        self.assertTrue(analysis["material_net_or_level_growth"])
        result = passing_result()
        gates, failures = RUN.evaluate_acceptance(
            result, analysis, buffer=512, rate=44100, render_frames=512)
        self.assertFalse(gates["no_material_post_warmup_rss_growth"])
        self.assertIn("no_material_post_warmup_rss_growth", failures)

    def test_blocked_soak_row_is_written_then_raises(self) -> None:
        result = passing_result()
        artifacts = {
            key: Path(f"/tmp/{key}")
            for key in (
                "ir", "coeff", "msl", "manifest",
                "reload_ir", "reload_coeff", "reload_msl", "reload_manifest",
            )
        }
        slots = {
            "reference_slot": 0,
            "param_events": [
                ("raw", "raw", 0.0, 1.0),
                ("glide", "glide", 0.0, 1.0),
                ("anchor", "anchor", 0.0, 1.0),
                ("velocity", "velocity", 0.0, 1.0),
            ],
            "canary_amplitude_bound": 1e-12,
        }
        rss_analysis = {
            "sample_count": 3,
            "all_samples_positive": True,
            "material_net_or_level_growth": False,
        }
        stream = io.StringIO()
        with (
            mock.patch.object(
                RUN, "prepare_heavy_graph", return_value=(artifacts, slots)),
            mock.patch.object(RUN, "probe", side_effect=[{}, result]),
            mock.patch.object(RUN, "analyze_rss", return_value=rss_analysis),
            mock.patch.object(
                RUN, "evaluate_acceptance",
                return_value=({"forced_failure": False}, ["forced_failure"])),
            self.assertRaises(RUN.QualificationBlocked),
        ):
            RUN.run_soak(stream, Path("/tmp"), 1800, 512, 512)
        row = json.loads(stream.getvalue())
        self.assertEqual(row["qualification_status"], "blocked")
        self.assertEqual(row["failure_reasons"], ["forced_failure"])

    def test_soak_uses_transition_intervals_and_render_quantum(self) -> None:
        result = passing_result()
        artifacts = {
            key: Path(f"/tmp/{key}")
            for key in (
                "ir", "coeff", "msl", "manifest",
                "reload_ir", "reload_coeff", "reload_msl", "reload_manifest",
            )
        }
        slots = {
            "reference_slot": 0,
            "param_events": [
                ("raw", "raw", 0.0, 1.0),
                ("glide", "glide", 0.0, 1.0),
                ("anchor", "anchor", 0.0, 1.0),
                ("velocity", "velocity", 0.0, 1.0),
            ],
            "canary_amplitude_bound": 1e-12,
        }
        rss_analysis = {
            "sample_count": 3,
            "all_samples_positive": True,
            "material_net_or_level_growth": False,
        }
        stream = io.StringIO()
        with (
            mock.patch.object(
                RUN, "prepare_heavy_graph", return_value=(artifacts, slots)),
            mock.patch.object(
                RUN, "probe", side_effect=[{}, result]) as probe,
            mock.patch.object(
                RUN, "analyze_rss", return_value=rss_analysis),
            mock.patch.object(
                RUN, "evaluate_acceptance",
                return_value=({"all": True}, [])),
        ):
            RUN.run_soak(stream, Path("/tmp"), 600, 512, 512)
        live_args, live_env = probe.call_args_list[1].args
        self.assertIn("--jump-every", live_args)
        self.assertIn("--reload-every", live_args)
        self.assertNotIn("1024", live_args)
        self.assertEqual(
            live_env["TROPICAL_METAL_RENDER_TILE_FRAMES"], "512")
        self.assertNotIn("TROPICAL_METAL_PIPELINE_DEPTH", live_env)

    def test_main_returns_nonzero_for_blocked_soak_without_duplicate_row(
            self) -> None:
        def blocked_soak(stream, *_args) -> None:
            stream.write(json.dumps({
                "schema": "tropical_metal_live_soak_2",
                "qualification_status": "blocked",
                "failure_reasons": ["forced_failure"],
            }) + "\n")
            stream.flush()
            raise RUN.QualificationBlocked("forced_failure")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blocked.jsonl"
            with (
                mock.patch.object(
                    sys, "argv",
                    ["run.py", "--mode", "soak", "--skip-build",
                     "--output", str(output)]),
                mock.patch.object(RUN, "RUNTIME_BENCH", Path(__file__)),
                mock.patch.object(RUN, "DIFFCLI", Path(__file__)),
                mock.patch.object(
                    RUN, "manifest",
                    return_value={"schema": "tropical_metal_live_env_2"}),
                mock.patch.object(RUN, "run_soak", side_effect=blocked_soak),
            ):
                self.assertEqual(RUN.main(), 1)
            rows = [
                json.loads(line)
                for line in output.read_text().splitlines()
                if line.strip()
            ]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["qualification_status"], "blocked")
        self.assertNotEqual(rows[1]["schema"], "tropical_metal_live_failure_2")

    def test_classified_failure_row_is_flushed_and_blocked(self) -> None:
        class RecordingStream(io.StringIO):
            flushed = False

            def flush(self) -> None:
                self.flushed = True
                super().flush()

        stream = RecordingStream()
        RUN.write_failure_row(
            stream,
            mode="soak",
            requested_mode="all",
            duration=1800,
            buffer=512,
            render_frames=512,
            error=RuntimeError("device vanished"),
        )
        row = json.loads(stream.getvalue())
        self.assertTrue(stream.flushed)
        self.assertEqual(
            row["schema"], "tropical_metal_qualification_failure_1")
        self.assertEqual(row["mode"], "soak")
        self.assertEqual(row["requested_mode"], "all")
        self.assertEqual(row["device_frames"], 512)
        self.assertEqual(row["render_tile_frames"], 512)
        self.assertEqual(row["duration_requested_seconds"], 1800)
        self.assertEqual(row["exception_class"], "RuntimeError")
        self.assertEqual(row["exception_message"], "device vanished")
        self.assertEqual(row["qualification_status"], "blocked")


if __name__ == "__main__":
    unittest.main()
