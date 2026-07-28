#!/usr/bin/env python3
"""Schema and provenance gates for the current performance baseline."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "current_baseline_run", HERE / "run.py")
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class ExactProductFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        fixtures = RUN.load_matrix()
        self.flagship = next(item for item in fixtures if item.get("flagship"))
        self.graph = RUN.fixture_graph(self.flagship)

    def test_schema_versions_are_explicit(self) -> None:
        self.assertEqual(
            RUN.BASELINE_MATRIX_SCHEMA, "tropical_baseline_matrix_2")
        self.assertEqual(RUN.BASELINE_ROW_SCHEMA, "tropical_baseline_row_3")

    def test_flagship_reads_renderer_graph_directly(self) -> None:
        self.assertNotIn("graph", self.flagship)
        self.assertEqual(self.flagship["product_graph"], {
            "path": "playground/renderer/app.js",
            "contract": RUN.PRODUCT_GRAPH_CONTRACT,
        })
        provenance = RUN.fixture_provenance(self.flagship)
        self.assertEqual(provenance["kind"], "renderer_exact_product_graph")
        self.assertEqual(len(provenance["source_file_sha256"]), 64)
        self.assertEqual(len(provenance["normalized_graph_sha256"]), 64)

    def test_flagship_matches_current_product_circuit(self) -> None:
        nodes = self.graph["nodes"]
        self.assertEqual(
            [node["id"] for node in nodes],
            ["o1", "o2", "adr", "r1", "r2", "r3", "r4",
             "mx", "rv", "flt", "out"])
        by_id = {node["id"]: node for node in nodes}
        self.assertEqual(
            by_id["o1"]["params"], {"freq": 0.11, "morph": 0})
        self.assertEqual(
            by_id["o2"]["params"], {"freq": 2.2, "morph": 0.6})
        self.assertEqual(by_id["adr"]["in"], {"in": ["o1", "o2"]})

        for ring_id, frequency in (
            ("r1", 110), ("r2", 165), ("r3", 220), ("r4", 330),
        ):
            ring = by_id[ring_id]
            self.assertEqual(ring["kind"], "resonator")
            self.assertEqual(
                ring["params"], {"freq": frequency, "decay": 4})
            self.assertNotIn("partials", ring["params"])
            self.assertEqual(ring["in"], {"addr": ["adr"]})

        self.assertEqual(
            by_id["mx"]["in"], {"in": ["r1", "r2", "r3", "r4"]})
        self.assertEqual(by_id["rv"]["params"], {"rt60": 2})
        self.assertEqual(by_id["rv"]["in"], {"in": ["mx"]})
        self.assertEqual(
            by_id["flt"]["params"],
            {"cutoff": 800, "resonance": 0.5})
        self.assertEqual(by_id["flt"]["in"], {"in": ["rv"]})
        self.assertEqual(by_id["out"]["in"], {"in": ["flt"]})
        self.assertEqual(self.graph["out"], "out")
        self.assertIs(self.graph["taps"], True)

    def test_structural_edits_preserve_product_address_path(self) -> None:
        edits = RUN.flagship_edits(self.flagship)
        self.assertEqual(set(edits), {
            "topology_add_addressed_ring",
            "structural_default_partials_6_to_7",
        })

        topology_nodes = edits["topology_add_addressed_ring"]["graph"]["nodes"]
        topology = {node["id"]: node for node in topology_nodes}
        self.assertEqual(topology["r5"], {
            "id": "r5",
            "kind": "resonator",
            "params": {"freq": 440, "decay": 4},
            "sel": {},
            "in": {"addr": ["adr"]},
        })
        self.assertEqual(
            topology["mx"]["in"]["in"], ["r1", "r2", "r3", "r4", "r5"])
        self.assertIs(
            edits["topology_add_addressed_ring"]["graph"]["taps"], True)

        partial_nodes = (
            edits["structural_default_partials_6_to_7"]["graph"]["nodes"])
        partials = {node["id"]: node for node in partial_nodes}
        self.assertEqual(partials["r1"]["params"]["partials"], 7)
        for ring_id in ("r2", "r3", "r4"):
            self.assertNotIn("partials", partials[ring_id]["params"])

        provenance = RUN.structural_edit_provenance(
            self.flagship, edits["topology_add_addressed_ring"])
        self.assertEqual(provenance["kind"], "derived_structural_edit")
        self.assertEqual(provenance["normalized_graph"],
                         edits["topology_add_addressed_ring"]["graph"])
        self.assertEqual(len(provenance["normalized_graph_sha256"]), 64)
        self.assertEqual(
            provenance["source_normalized_graph_sha256"],
            RUN.fixture_provenance(
                self.flagship)["normalized_graph_sha256"])


class ArtifactDigestTests(unittest.TestCase):
    @staticmethod
    def evidence(repeats: int = 3) -> dict:
        artifacts = {
            name: {"sha256": hashlib.sha256(name.encode()).hexdigest(),
                   "bytes": len(name)}
            for name in ("manifest", "audio_ir", "coefficient_ir", "msl")
        }
        samples = [
            {"repeat": repeat, "artifacts": copy.deepcopy(artifacts)}
            for repeat in range(repeats)
        ]
        return {
            "emitted_bytes_stable": True,
            "artifact_digests": {
                "algorithm": "sha256",
                "expected_artifacts":
                    ["manifest", "audio_ir", "coefficient_ir", "msl"],
                "cold": copy.deepcopy(samples),
                "warm": copy.deepcopy(samples),
            },
        }

    def test_digest_evidence_accepts_complete_cold_warm_matrix(self) -> None:
        RUN.validate_generation_evidence(
            self.evidence(), repeats=3, metal=True)

    def test_digest_evidence_rejects_missing_repeat_artifact(self) -> None:
        evidence = self.evidence()
        del evidence["artifact_digests"]["warm"][2]["artifacts"]["msl"]
        with self.assertRaisesRegex(RuntimeError, "artifact digest mismatch"):
            RUN.validate_generation_evidence(
                evidence, repeats=3, metal=True)

    def test_digest_evidence_rejects_false_stability_label(self) -> None:
        evidence = self.evidence()
        evidence["artifact_digests"]["warm"][2]["artifacts"]["audio_ir"][
            "sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            RUN.validate_generation_evidence(
                evidence, repeats=3, metal=True)

    def test_artifact_digest_includes_empty_coefficient_ir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "manifest": b"manifest",
                "ir": b"audio",
                "coeff": b"",
                "msl": b"metal",
            }
            paths = {}
            for key, value in values.items():
                path = root / key
                path.write_bytes(value)
                paths[key] = path
            digests = RUN.artifact_digests(paths)
        self.assertEqual(set(digests), {
            "manifest", "audio_ir", "coefficient_ir", "msl"})
        self.assertEqual(
            digests["coefficient_ir"]["sha256"],
            hashlib.sha256(b"").hexdigest())
        self.assertEqual(digests["coefficient_ir"]["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
