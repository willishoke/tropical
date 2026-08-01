#!/usr/bin/env python3
"""Generate the immutable native-rate grouped-room production asset.

The v1 binary layout is deliberately small and fixed.  Every integer and
floating-point scalar is little-endian.

    0x0000  128-byte header (``<4sHH7I9Q20x``)
    0x0080  uint32 periods[group_count]
            float64 radii[group_count]
            float64 source_coordinates[source_count][frequency, sigma]
            uint32 group_offsets[group_count + 1]
            padding to a 64-byte boundary
            complex64 forward[source][concatenated group prefixes]
            complex64 reverse[source][concatenated group prefixes]

The complex tables are source-major, then group-major, then residue-major.
They contain the handoff's A and B prefix directions; carriers are evidence
inputs and are not needed by the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from run import TWO_PI, snare_modes
from run_grouped_clouds_fit import (
    NATIVE_PERIODS,
    NATIVE_SAMPLE_RATE,
    carrier_source_sha256,
)
from run_reverse_position import cyclic_future_table


AUDITION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = AUDITION_DIR.parents[2]
DEFAULT_NATIVE_DIR = AUDITION_DIR / "native_rate_out"
DEFAULT_ASSET_DIR = REPOSITORY_ROOT / "playground" / "assets" / "grouped-room"
PROFILE = "clouds-current-radii-mono-v1"
PROFILE_VERSION = 1
FORMAT_VERSION = 1
GENERATOR_REVISION = "tgrm-generator-v1"
MAGIC = b"TGRM"
HEADER_STRUCT = struct.Struct("<4sHH7I9Q20x")
HEADER_BYTES = HEADER_STRUCT.size
ALIGNMENT = 64
SCALAR_FORMAT_COMPLEX64 = 1
ENDIANNESS_LITTLE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--carriers",
        type=Path,
        default=DEFAULT_NATIVE_DIR / "grouped_fit_current_radii.npz",
    )
    parser.add_argument(
        "--fit-summary",
        type=Path,
        default=DEFAULT_NATIVE_DIR / "grouped_fit_summary.json",
    )
    parser.add_argument(
        "--reverse-summary",
        type=Path,
        default=DEFAULT_NATIVE_DIR / "reverse_position_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--listening-status",
        choices=("pending", "approved"),
        default="pending",
    )
    parser.add_argument("--listening-approved-by")
    return parser.parse_args()


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def source_coordinates() -> tuple[tuple[float, float], ...]:
    return tuple((row[0], row[1]) for row in snare_modes(0))


def load_carriers(
    path: Path,
) -> tuple[tuple[int, ...], tuple[float, ...], list[np.ndarray]]:
    with np.load(path) as data:
        if "sample_rate" not in data:
            raise RuntimeError("production carrier source has no sample_rate")
        sample_rate = int(data["sample_rate"])
        periods = tuple(int(value) for value in data["periods"])
        radii = tuple(float(value) for value in data["radii"])
        carriers = [
            np.asarray(data[f"carrier_{index}"], dtype=np.float64)
            for index in range(len(periods))
        ]
    if sample_rate != NATIVE_SAMPLE_RATE:
        raise RuntimeError(
            f"production carrier rate {sample_rate} != {NATIVE_SAMPLE_RATE}"
        )
    if periods != NATIVE_PERIODS:
        raise RuntimeError(f"production periods {periods} != {NATIVE_PERIODS}")
    if len(radii) != 12 or len(carriers) != 12:
        raise RuntimeError("production profile must contain exactly twelve groups")
    if any(len(carrier) != period for carrier, period in zip(carriers, periods)):
        raise RuntimeError("carrier length does not match its frozen period")
    if not all(0.0 < radius < 1.0 and math.isfinite(radius) for radius in radii):
        raise RuntimeError("all production radii must be finite and stable")
    if not all(np.all(np.isfinite(carrier)) for carrier in carriers):
        raise RuntimeError("production carrier contains a non-finite value")
    return periods, radii, carriers


def prefix_tables(
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    carriers: list[np.ndarray],
    coordinates: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    forward: list[np.ndarray] = []
    reverse: list[np.ndarray] = []
    for frequency, sigma in coordinates:
        pole = np.exp(complex(-sigma, TWO_PI * frequency) / NATIVE_SAMPLE_RATE)
        for period, radius, carrier in zip(periods, radii, carriers):
            ratio = radius / pole
            forward.append(np.cumsum(
                carrier * np.power(ratio, np.arange(period, dtype=np.int64))
            ))
            reverse.append(cyclic_future_table(carrier, pole * radius))
    forward_table = np.concatenate(forward).astype("<c8")
    reverse_table = np.concatenate(reverse).astype("<c8")
    if not np.all(np.isfinite(forward_table)):
        raise RuntimeError("forward production prefix contains a non-finite value")
    if not np.all(np.isfinite(reverse_table)):
        raise RuntimeError("reverse production prefix contains a non-finite value")
    return forward_table, reverse_table


def encode_asset(
    periods: tuple[int, ...],
    radii: tuple[float, ...],
    coordinates: tuple[tuple[float, float], ...],
    forward: np.ndarray,
    reverse: np.ndarray,
) -> tuple[bytes, dict[str, int]]:
    group_count = len(periods)
    source_count = len(coordinates)
    carrier_count = sum(periods)
    expected_values = source_count * carrier_count
    if len(forward) != expected_values or len(reverse) != expected_values:
        raise RuntimeError("prefix table count does not match source/group shape")

    periods_bytes = np.asarray(periods, dtype="<u4").tobytes()
    radii_bytes = np.asarray(radii, dtype="<f8").tobytes()
    coordinates_bytes = np.asarray(coordinates, dtype="<f8").tobytes()
    group_offsets = np.concatenate((
        np.asarray([0], dtype=np.uint32),
        np.cumsum(np.asarray(periods, dtype=np.uint32)),
    ))
    group_offsets_bytes = np.asarray(group_offsets, dtype="<u4").tobytes()

    metadata_offset = HEADER_BYTES
    periods_offset = metadata_offset
    radii_offset = periods_offset + len(periods_bytes)
    coordinates_offset = radii_offset + len(radii_bytes)
    group_offsets_offset = coordinates_offset + len(coordinates_bytes)
    metadata_end = group_offsets_offset + len(group_offsets_bytes)
    metadata_bytes = metadata_end - metadata_offset
    forward_offset = align(metadata_end)
    forward_bytes = np.asarray(forward, dtype="<c8").tobytes()
    reverse_offset = align(forward_offset + len(forward_bytes))
    reverse_bytes = np.asarray(reverse, dtype="<c8").tobytes()
    file_bytes = reverse_offset + len(reverse_bytes)

    header = HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        NATIVE_SAMPLE_RATE,
        group_count,
        source_count,
        carrier_count,
        SCALAR_FORMAT_COMPLEX64,
        ENDIANNESS_LITTLE,
        0,
        metadata_offset,
        metadata_bytes,
        periods_offset,
        radii_offset,
        coordinates_offset,
        group_offsets_offset,
        forward_offset,
        reverse_offset,
        file_bytes,
    )
    payload = bytearray(file_bytes)
    payload[:HEADER_BYTES] = header
    payload[periods_offset:radii_offset] = periods_bytes
    payload[radii_offset:coordinates_offset] = radii_bytes
    payload[coordinates_offset:group_offsets_offset] = coordinates_bytes
    payload[group_offsets_offset:metadata_end] = group_offsets_bytes
    payload[forward_offset:forward_offset + len(forward_bytes)] = forward_bytes
    payload[reverse_offset:reverse_offset + len(reverse_bytes)] = reverse_bytes
    offsets = {
        "metadata": metadata_offset,
        "periods": periods_offset,
        "radii": radii_offset,
        "source_coordinates": coordinates_offset,
        "group_offsets": group_offsets_offset,
        "forward": forward_offset,
        "reverse": reverse_offset,
        "file_bytes": file_bytes,
        "metadata_bytes": metadata_bytes,
    }
    return bytes(payload), offsets


def validate_encoded_asset(payload: bytes, offsets: dict[str, int]) -> None:
    unpacked = HEADER_STRUCT.unpack_from(payload)
    if unpacked[0] != MAGIC or unpacked[1] != FORMAT_VERSION:
        raise RuntimeError("generated asset failed header round trip")
    if unpacked[2] != HEADER_BYTES or unpacked[3] != NATIVE_SAMPLE_RATE:
        raise RuntimeError("generated asset failed fixed header validation")
    if unpacked[-1] != len(payload) or offsets["file_bytes"] != len(payload):
        raise RuntimeError("generated asset file size does not match its header")
    if offsets["forward"] % ALIGNMENT or offsets["reverse"] % ALIGNMENT:
        raise RuntimeError("generated prefix tables are not 64-byte aligned")


def oracle_manifest(
    fit_summary: dict[str, object], reverse_summary: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    profile = fit_summary["profiles"]["current_radii"]
    fit_error = profile["structured_source_vs_convolution_relative_l2"]
    reverse_validation = reverse_summary["validation"]
    tracked = {
        "forward_grouped_vs_convolution_relative_l2": fit_error,
        "causal_integer_literal_relative_l2": reverse_validation[
            "causal_integer_literal_relative_l2"
        ],
        "causal_fractional_literal_relative_l2": reverse_validation[
            "causal_fractional_literal_relative_l2"
        ],
        "reverse_integer_literal_relative_l2": reverse_validation[
            "reverse_integer_literal_relative_l2"
        ],
        "reverse_fractional_literal_relative_l2": reverse_validation[
            "reverse_fractional_literal_relative_l2"
        ],
        "whole_composite_mirror_vs_classic_nrmse": reverse_validation[
            "whole_composite_mirror_vs_classic_nrmse"
        ],
    }
    passed = (
        fit_error < 1e-9
        and reverse_validation["causal_integer_literal_relative_l2"] < 1e-8
        and reverse_validation["causal_fractional_literal_relative_l2"] < 1e-8
        and reverse_validation["reverse_integer_literal_relative_l2"] < 1e-8
        and reverse_validation["reverse_fractional_literal_relative_l2"] < 1e-8
    )
    if not passed:
        raise RuntimeError(f"production oracle gate failed: {tracked}")

    fit_files = fit_summary["files"]
    reverse_files = reverse_summary["files"]
    auditions = [
        {
            "role": "native wet impulse",
            "filename": "grouped_fit_current_radii_impulse.wav",
            "sha256": fit_files["grouped_fit_current_radii_impulse.wav"]["sha256"],
        },
        {
            "role": "native forward Metal hit",
            "filename": "position_1p0_wet.wav",
            "sha256": reverse_files["position_1p0_wet.wav"]["sha256"],
        },
        {
            "role": "native reverse Metal hit",
            "filename": "position_minus_1p0_wet.wav",
            "sha256": reverse_files["position_minus_1p0_wet.wav"]["sha256"],
        },
        {
            "role": "native POSITION scrub",
            "filename": "position_scrub_forward_to_reverse_wet.wav",
            "sha256": reverse_files[
                "position_scrub_forward_to_reverse_wet.wav"
            ]["sha256"],
        },
    ]
    return {"passed": True, "measurements": tracked}, auditions


def main() -> None:
    args = parse_args()
    if args.listening_status == "approved" and not args.listening_approved_by:
        raise RuntimeError("approved listening status requires --listening-approved-by")
    carriers_path = args.carriers.resolve()
    fit_summary_path = args.fit_summary.resolve()
    reverse_summary_path = args.reverse_summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    periods, radii, carriers = load_carriers(carriers_path)
    coordinates = source_coordinates()
    if len(coordinates) != 12:
        raise RuntimeError("production Metal source must have twelve pole rows")
    forward, reverse = prefix_tables(periods, radii, carriers, coordinates)
    payload, offsets = encode_asset(
        periods, radii, coordinates, forward, reverse
    )
    validate_encoded_asset(payload, offsets)

    binary_path = output_dir / f"{PROFILE}-44100.tgrm"
    binary_path.write_bytes(payload)
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    reverse_summary = json.loads(
        reverse_summary_path.read_text(encoding="utf-8")
    )
    if fit_summary["fit_sample_rate"] != NATIVE_SAMPLE_RATE:
        raise RuntimeError("fit summary is not native 44.1 kHz evidence")
    if reverse_summary["sample_rate"] != NATIVE_SAMPLE_RATE:
        raise RuntimeError("reverse summary is not native 44.1 kHz evidence")
    oracles, auditions = oracle_manifest(fit_summary, reverse_summary)

    source_hash = carrier_source_sha256(
        NATIVE_SAMPLE_RATE, periods, radii, carriers
    )
    fit_source_hash = fit_summary["files"][carriers_path.name][
        "canonical_carrier_source_sha256"
    ]
    if source_hash != fit_source_hash:
        raise RuntimeError("carrier source hash disagrees with fit summary")

    generator_path = Path(__file__).resolve()
    table_values = len(coordinates) * sum(periods)
    manifest = {
        "profile": PROFILE,
        "profile_version": PROFILE_VERSION,
        "binary_format_version": FORMAT_VERSION,
        "sample_rate": NATIVE_SAMPLE_RATE,
        "group_count": len(periods),
        "source_pole_count": len(coordinates),
        "periods": periods,
        "radii": radii,
        "source_coordinates": [
            {"frequency_hz": frequency, "sigma": sigma}
            for frequency, sigma in coordinates
        ],
        "binary_format": {
            "magic_ascii": MAGIC.decode("ascii"),
            "header_struct": HEADER_STRUCT.format,
            "header_bytes": HEADER_BYTES,
            "alignment_bytes": ALIGNMENT,
            "endianness": "little",
            "scalar_format": "IEEE-754 complex64, interleaved real/imag",
            "table_order": "source-major, group-major, residue-major",
            "offsets_bytes": offsets,
            "tables": {
                "periods": {"count": len(periods), "scalar": "uint32"},
                "radii": {"count": len(radii), "scalar": "float64"},
                "source_coordinates": {
                    "count": 2 * len(coordinates),
                    "scalar": "float64",
                    "tuple": ["frequency_hz", "sigma"],
                },
                "group_offsets": {
                    "count": len(periods) + 1,
                    "scalar": "uint32",
                },
                "forward": {"count": table_values, "scalar": "complex64"},
                "reverse": {"count": table_values, "scalar": "complex64"},
            },
        },
        "generator": {
            "revision": GENERATOR_REVISION,
            "source": relative(generator_path),
            "source_sha256": sha256(generator_path),
            "command": (
                "uv run --with numpy python "
                "benchmarks/demo_release/room_audition/"
                "generate_grouped_room_asset.py"
            ),
        },
        "hashes": {
            "clouds_commit": fit_summary["clouds_commit"],
            "frozen_clouds_target_mono_float64_le_sha256": fit_summary[
                "target_mono_float64_le_sha256"
            ],
            "fitted_carrier_source_sha256": source_hash,
            "fitted_carrier_npz_sha256": sha256(carriers_path),
            "binary_sha256": sha256(binary_path),
        },
        "oracles": oracles,
        "auditions": auditions,
        "listening": {
            "status": args.listening_status,
            "approved_by": args.listening_approved_by,
        },
    }
    manifest_path = output_dir / f"{PROFILE}-44100.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
