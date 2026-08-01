#!/usr/bin/env python3
"""Deterministic gates for the checked native grouped-room asset."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

import generate_grouped_room_asset as asset


ASSET_DIR = asset.DEFAULT_ASSET_DIR
BINARY_PATH = ASSET_DIR / f"{asset.PROFILE}-44100.tgrm"
MANIFEST_PATH = ASSET_DIR / f"{asset.PROFILE}-44100.json"
CARRIERS_PATH = (
    asset.DEFAULT_NATIVE_DIR / "grouped_fit_current_radii.npz"
)


class NativeGroupedRoomAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = BINARY_PATH.read_bytes()
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_hash_and_shape(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["profile"], asset.PROFILE)
        self.assertEqual(manifest["sample_rate"], asset.NATIVE_SAMPLE_RATE)
        self.assertEqual(manifest["group_count"], 12)
        self.assertEqual(manifest["source_pole_count"], 12)
        self.assertEqual(tuple(manifest["periods"]), asset.NATIVE_PERIODS)
        self.assertEqual(sum(manifest["periods"]), 14_150)
        self.assertEqual(
            hashlib.sha256(self.payload).hexdigest(),
            manifest["hashes"]["binary_sha256"],
        )

    def test_header_metadata_and_tables_round_trip(self) -> None:
        header = asset.HEADER_STRUCT.unpack_from(self.payload)
        self.assertEqual(header[0], asset.MAGIC)
        self.assertEqual(header[1], asset.FORMAT_VERSION)
        self.assertEqual(header[2], asset.HEADER_BYTES)
        self.assertEqual(header[3:7], (44_100, 12, 12, 14_150))
        self.assertEqual(header[7], asset.SCALAR_FORMAT_COMPLEX64)
        self.assertEqual(header[8], asset.ENDIANNESS_LITTLE)
        self.assertEqual(header[18], len(self.payload))

        offsets = self.manifest["binary_format"]["offsets_bytes"]
        periods = np.frombuffer(
            self.payload, dtype="<u4", count=12, offset=offsets["periods"]
        )
        radii = np.frombuffer(
            self.payload, dtype="<f8", count=12, offset=offsets["radii"]
        )
        coordinates = np.frombuffer(
            self.payload,
            dtype="<f8",
            count=24,
            offset=offsets["source_coordinates"],
        ).reshape(12, 2)
        group_offsets = np.frombuffer(
            self.payload,
            dtype="<u4",
            count=13,
            offset=offsets["group_offsets"],
        )
        table_count = 12 * 14_150
        forward = np.frombuffer(
            self.payload,
            dtype="<c8",
            count=table_count,
            offset=offsets["forward"],
        )
        reverse = np.frombuffer(
            self.payload,
            dtype="<c8",
            count=table_count,
            offset=offsets["reverse"],
        )

        np.testing.assert_array_equal(periods, asset.NATIVE_PERIODS)
        np.testing.assert_allclose(radii, self.manifest["radii"], rtol=0, atol=0)
        np.testing.assert_allclose(
            coordinates,
            [
                [row["frequency_hz"], row["sigma"]]
                for row in self.manifest["source_coordinates"]
            ],
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            group_offsets,
            np.concatenate(([0], np.cumsum(asset.NATIVE_PERIODS))),
        )
        self.assertTrue(np.all(np.isfinite(forward)))
        self.assertTrue(np.all(np.isfinite(reverse)))
        self.assertEqual(offsets["forward"] % asset.ALIGNMENT, 0)
        self.assertEqual(offsets["reverse"] % asset.ALIGNMENT, 0)

    def test_generator_reproduces_payload(self) -> None:
        periods, radii, carriers = asset.load_carriers(CARRIERS_PATH)
        coordinates = asset.source_coordinates()
        forward, reverse = asset.prefix_tables(
            periods, radii, carriers, coordinates
        )
        regenerated, offsets = asset.encode_asset(
            periods, radii, coordinates, forward, reverse
        )
        asset.validate_encoded_asset(regenerated, offsets)
        self.assertEqual(regenerated, self.payload)

    def test_oracle_gates_are_recorded_as_passed(self) -> None:
        oracles = self.manifest["oracles"]
        self.assertTrue(oracles["passed"])
        measurements = oracles["measurements"]
        self.assertLess(
            measurements["forward_grouped_vs_convolution_relative_l2"], 1e-9
        )
        for name in (
            "causal_integer_literal_relative_l2",
            "causal_fractional_literal_relative_l2",
            "reverse_integer_literal_relative_l2",
            "reverse_fractional_literal_relative_l2",
        ):
            self.assertLess(measurements[name], 1e-8)
        self.assertEqual(
            measurements["whole_composite_mirror_vs_classic_nrmse"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
