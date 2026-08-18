# Absolute-time phaser coefficient staging decision

Date: 2026-08-18

Branch: `perf/phaser-time-staging`

Decision: **revise the endpoint representation before product integration**

## Outcome

The isolated, opt-in linear endpoint prototype is worth retaining, but it is
not a production phaser terminal yet. It establishes the absolute-time
compiler/runtime crossing, makes the 6- and 12-stage audio kernels compact,
and preserves an independently owned exact JIT oracle/fallback. The ordinary
residue image fails the handoff's confluence-safety requirement at the
18-stage/32-partial stress point. The feature therefore remains disabled by
default and is not connected to the factory demo.

Enable the experiment explicitly with:

```text
TROPICAL_PHASER_TIME_STAGING=1
TROPICAL_PHASER_TIME_STAGING_INTERVAL=128
```

The compiler only recognizes one structural degree-zero
`dryWetAllpassCascadeShape?` terminal with unpatched slow controls and exactly
6, 12, or 18 sections. Ineligible graphs preserve the exact terminal and carry
fallback provenance in the manifest. The expanded 12/18-section topology is a
hidden benchmark voicing, not a public musical voicing.

## Evidence

The retained no-DAC smoke run is
`benchmarks/phaser_time_staging/data/smoke-2026-08-18/raw.json`. It contains
three independent repetitions per row, exact/staged f64 audio, error JSON,
operation counts, conservative plan register spans, artifact hashes, and raw
worker telemetry. The script can run the full 162-row matrix with `--full`,
but the matrix was not promoted after the explicit 18-stage stop condition.

| row | audio instructions / divisions | max abs error | SNR | median / max measured deadline load | materializer / exact fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 partials, 6 sections, R=128, 0.2 Hz | 741 / 30 | 4.56e-5 | 92.30 dB | 10.61% / 10.74% | 0 / 0 |
| 6 partials, 6 sections, R=64, 8 Hz, low-center extreme | 741 / 30 | 4.13e-2 | 40.44 dB | 10.78% / 11.01% | 0 / 0 |
| 6 partials, 6 sections, R=32, 8 Hz, high-center extreme | 741 / 30 | 8.25e-3 | 60.54 dB | 10.60% / 10.78% | 0 / 0 |
| 32 partials, 12 sections, R=64, 0.2 Hz | 1989 / 94 | 2.65e-2 | 66.23 dB | 18.59% / 18.61% | 0 / 0 |
| 32 partials, 18 sections, R=32, 8 Hz | 2222 / 106 | not meaningful: exact stress voicing collapsed to zero | n/a | 71.52% / 71.69% | 416 / 416 per repetition |

The first four rows produced finite audio and zero starvation, tag mismatch,
activation, callback-thread, deadline-overrun, materialization, and exact
fallback counters. The 18-stage row produced finite fail-silent output, but
every measured endpoint image exceeded the ordinary-residue safety bound and
correctly selected the exact CPU fallback. Its deliberately expanded exact
voicing also rendered silence, so it cannot support a quality claim.

The focused structural gate reports exact/audio instruction counts of
4768/760, 10534/1006, and 18460/1252 for 6/12/18 sections respectively, with
audio divisions 30/42/54 instead of exact divisions 463/1171/2167. This is the
intended approximately linear audio-side growth. Runtime tests cover absolute
source coordinates, interval subdivisions, rematerialization after a backward
clock jump, immutable program lifetime across publication, and unsafe-image
fallback.

## Why this stops here

Raising the residue limit, clamping, or allowing partially filled images would
hide the numerical failure forbidden by the handoff. Ordinary independent
residue interpolation is not a safe representation through high-order
confluence. The next experiment should publish structural pole identities plus
divided-difference/factored parameters and evaluate the existing stable
carrier on Metal. Admission should then prove the whole interval safe; only
intervals outside the hot/crossing predicate may use ordinary rows.

Before that representation exists, cubic interpolation and factory/reverb
composition would broaden scope without addressing the observed failure. Full
LFO-cycle, shuffled-seek, notch-center/depth, and matched-level listening gates
therefore remain intentionally pending.

## Manual listening checkpoint

Do not enable the factory demo or start a DAC from automation. After a stable
confluence representation passes the full no-DAC matrix:

1. Render exact and staged full-cycle files at 44.1 kHz for the same graph and
   absolute start using `benchmarks/phaser_time_staging/run.py --full`.
2. Import each `.f64le` file into a DAW as mono, little-endian 64-bit float at
   44.1 kHz. Do not normalize or limit either file.
3. RMS-match the two files using one fixed gain, randomize A/B identity, lower
   the monitor to a safe level, and compare notch movement, boundary clicks,
   low-center sweeps, and the 12/18-stage color.
4. Record the listener, hardware, exact artifact hashes, gain, and preference.
   Stop on a preferred exact reference, any click/dead region, or loss of the
   high-stage character; do not merge on CI or benchmark numbers alone.
