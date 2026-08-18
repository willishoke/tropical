# Absolute-time phaser coefficient staging decision

Date: 2026-08-18

Branch: `perf/phaser-time-staging`

Decision: **reject first-order mixed DD staging and pivot to a higher-order
routed/factored endpoint representation before product integration**

## Outcome

The isolated, opt-in linear endpoint prototype is worth retaining, but it is
not a production phaser terminal yet. It establishes the absolute-time
compiler/runtime crossing, makes the 6- and 12-stage audio kernels compact,
and preserves an independently owned exact JIT oracle/fallback. The ordinary
residue image fails the handoff's confluence-safety requirement at the
18-stage/32-partial stress point.

A bounded follow-up tested a mixed ordinary/first-order-divided-difference
image. An interval-wide structural classifier proves that every unordered
all-pass-tail pair is hot: every tail remains a real pole (`omega = 0`) over
the entire control interval. Disjoint first-order pairs cover only a shrinking
fraction of that dense graph, and every smoke row that reached the mixed Metal
kernel emitted non-finite samples. The hybrid is therefore rejected rather
than widened. The next representation must retain the whole tail product or an
equivalent higher-order routed/factored carrier.

Both experiments remain disabled by default and are not connected to the
factory demo.

Enable the experiment explicitly with:

```text
TROPICAL_PHASER_TIME_STAGING=1
TROPICAL_PHASER_TIME_STAGING_INTERVAL=128
```

The rejected mixed image is reproducible only when the additional
falsification flag is present:

```text
TROPICAL_PHASER_TIME_STAGING=1
TROPICAL_PHASER_TIME_STAGING_MIXED_DD=1
TROPICAL_PHASER_TIME_STAGING_INTERVAL=128
```

The compiler only recognizes one structural degree-zero
`dryWetAllpassCascadeShape?` terminal with unpatched slow controls and exactly
6, 12, or 18 sections. Ineligible graphs preserve the exact terminal and carry
fallback provenance in the manifest. The expanded 12/18-section topology is a
hidden benchmark voicing, not a public musical voicing.

## Evidence

The original retained no-DAC smoke run is
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

### First-order mixed DD falsification

The retained follow-up is
`benchmarks/phaser_time_staging/data/mixed-dd-smoke-2026-08-18/raw.json`.
It records the interval-wide classifier, realized pair coverage, artifacts,
audio comparisons, three independent performance repetitions, and explicit
non-finite counts. JSON quality metrics are `null` when any sample is
non-finite; the harness does not turn that failure into an apparently finite
error or SNR.

| row | hot candidates / realized pairs | first-order coverage | audio instructions / divisions | median / max measured deadline load | result |
| --- | ---: | ---: | ---: | ---: | --- |
| 6 partials, 6 sections, R=128, 0.2 Hz | 15 / 3 | 20.0% | 1001 / 42 | 11.41% / 11.45% | 384 non-finite samples per repetition |
| 6 partials, 6 sections, R=64, 8 Hz, low-center extreme | 15 / 3 | 20.0% | 1001 / 42 | 11.44% / 11.45% | 204 non-finite samples per repetition |
| 6 partials, 6 sections, R=32, 8 Hz, high-center extreme | 15 / 3 | 20.0% | 1001 / 42 | 11.39% / 11.49% | 384 non-finite samples per repetition |
| 32 partials, 12 sections, R=64, 0.2 Hz | 66 / 6 | 9.09% | 2231 / 112 | 17.70% / 17.80% | 384 non-finite samples per repetition |
| 32 partials, 18 sections, R=32, 8 Hz | 153 / 9 | 5.88% | 2446 / 130 | 72.87% / 72.89% | 416 materializer/exact fallbacks per repetition |

The first-order rewrite is an exact frozen-endpoint identity: two ordinary
rows become one `PairedMode` plus their summed residue on one pole. That does
not make the higher-order cluster first-order. Large cancellation remains
between the paired atom and its remainder. Trying the incumbent Q4.28 DD
landing merely moves the failure: its intermediate rail saturated at 128.0
absolute error in the focused oracle, so it cannot certify this image either.

The compact cost result is real but unusable. Four rows reach approximately
the desired 11–18% load only by producing invalid audio. The 18-stage row
remains safe solely because the existing endpoint coefficient guard selects
the exact fallback before Metal. No load number is an admission argument in
the presence of either outcome.

## Why this stops here

Raising the residue limit, clamping, or allowing partially filled images would
hide the numerical failure forbidden by the handoff. Ordinary independent
residue interpolation is not a safe representation through high-order
confluence, and disjoint first-order DD atoms do not span the dense all-pass
cluster. The next experiment must publish structural pole identities plus a
higher-order routed/factored image and evaluate that stable carrier on Metal.
Admission must prove the whole interval safe; only intervals outside the
hot/crossing predicate may use ordinary rows.

The successor contract is now narrow:

1. Keep source rows and the authored all-pass tail product structurally
   identified at absolute endpoints; never publish the exploding tail partial
   fractions as independent values.
2. Evaluate a higher-order Newton/Hermite divided-difference or equivalent
   routed factored carrier whose bounded intermediates cover the complete hot
   cluster, including exact pole crossings.
3. Preserve deterministic endpoint materialization, shuffled-seek identity,
   and the existing exact CPU oracle/fallback. A direct `O(P*S)` factored
   cascade is acceptable only as a diagnostic reference; the production goal
   remains approximately `O(P+S)` audio work.
4. Re-run the 6/12/18 matrix and require finite output, the existing oracle and
   boundary gates, zero materializer/exact fallbacks in admitted rows, and
   stable operation growth before any cubic interpolation or factory wiring.

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
