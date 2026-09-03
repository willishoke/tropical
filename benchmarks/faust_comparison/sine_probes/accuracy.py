#!/usr/bin/env python3
"""Accuracy of the two sine kernels, so the speed table can be read honestly.

`findings.md` reading 1 reports tropical losing ~2.4x to Faust's F1. That
comparison is only meaningful alongside what each kernel costs in accuracy,
which the handoff's hypothesis 5 flagged and nobody had measured.

Faust's `os.osc` compiles to a TRUNCATED lookup in a 65536-entry table -- one
load, no interpolation. Read it in the emitted `compute`:

    ftbl0mydspSIG0[std::max<int>(0, std::min<int>(
        static_cast<int>(65536.0 * fRec1[0]), 65535))]

tropical's `FixedSin` is the Q31 Horner polynomial transcribed below from
`lean/Tropical/EmitArrow/Numerics.lean:fixedSinCycSig`. Python's `>>` on
negative ints is arithmetic (floor), which is what arm64 `asr` does, so the
transcription is faithful.

Both are compared against libm `sin` over a dense phase sweep.
"""
import math

TABLE_BITS = 16                      # Faust os.osc table size, 1 << 16
COEFFS = [1686629713, 693598668, 85569306, 5026995, 172272, 3864, 61]


def fixed_sin_cyc(phase_q: int) -> int:
    """Q32 phase in [0, 2^32) -> Q30 sine in [-2^30, 2^30]."""
    n = (phase_q + (1 << 30)) >> 31
    r = phase_q - (n << 31)
    sign = 1 - 2 * (n & 1)
    z = (r * r) >> 30
    acc = COEFFS[6] - (z >> 30)
    for c in reversed(COEFFS[1:6]):
        acc = c - ((acc * z) >> 30)
    acc = COEFFS[0] - ((acc * z) >> 30)
    return sign * ((r * acc) >> 30)


def faust_table_sin(phase: float, bits: int = TABLE_BITS) -> float:
    n = 1 << bits
    i = min(int(n * phase), n - 1)
    return math.sin(2.0 * math.pi * i / n)


def sweep(samples: int = 400_000) -> None:
    trop = faust = 0.0
    for k in range(samples):
        ph = k / samples
        q = int(ph * 2**32) & 0xFFFFFFFF
        ref = math.sin(2.0 * math.pi * (q / 2**32))
        trop = max(trop, abs(fixed_sin_cyc(q) / 2**30 - ref))
        faust = max(faust, abs(faust_table_sin(ph) - math.sin(2.0 * math.pi * ph)))
    for name, err in (("tropical Q31 polynomial", trop),
                      ("Faust F1 truncated table", faust)):
        print(f"  {name:<26} max |error| {err:.3e}  {20*math.log10(err):7.1f} dB  "
              f"~{-math.log2(err):.0f} bits")
    print(f"  {'ratio':<26}             {faust/trop:8.0f}x  "
          f"{20*math.log10(faust/trop):7.1f} dB")


if __name__ == "__main__":
    print(f"sine kernel accuracy vs libm ({TABLE_BITS}-bit table):")
    sweep()
