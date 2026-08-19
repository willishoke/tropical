# Absolute-time phaser coefficient staging decision

Date: 2026-08-19

Branch: `perf/phaser-time-staging`

Decision: **the higher-order representation is an isolated core-library
promotion candidate; keep it behind its research flag until matched-level
listening and product integration are completed**

## Outcome

The ordinary endpoint and disjoint first-order-DD representations remain
falsified. The successor whole-tail representation crosses the dense real-pole
cluster without publishing its exploding partial fractions, passes the bounded
automated accuracy/continuity/notch/determinism checks, and meets all three M1
Pro load targets.

The candidate publishes one self-describing absolute-time image per interval.
Support is adaptive: 32-frame images use 8 source / 6 private weight supports,
64-frame images use 8 / 8, and 128-frame images use 10 / 8. It contains:

- eight or ten Chebyshev coefficients for each complex source gain;
- one bounded real whole-tail value for every sample in the interval;
- source primitives, the matching common control supports, and exact response
  times as worker-private inputs.

The off-audio worker computes six or eight internal Newton-weight supports with
residual-corrected source divisions and double-double Newton differences, then
evaluates the complete all-pass tail in a normalized phase-type basis. Fused
multiply-add supplies exact product residuals, and fixed 18-element scratch
arrays avoid materializer heap churn. One entry dispatch selects a compile-time
8/6, 8/8, or 10/8 transform specialization. The published boundary is f32,
finite, and bounded. The Metal audio program performs one indexed source-bank
loop and one tail-table read; it never constructs residues. Section rates in
the admitted topology share one swept scale, so the worker validates one
normalized geometry per tile and reuses it. No previous tile or sample is an
input.

The experiment remains disabled unless both flags are present:

```text
TROPICAL_PHASER_TIME_STAGING=1
TROPICAL_PHASER_TIME_STAGING_HIGHER_ORDER=1
```

Only 6/32 degree-zero source rows, 6/12/18 stable all-pass sections, 32/64/128
frame intervals, one isolated structurally recognized phaser stage, and
unpatched slow controls are admitted. Other shapes retain the incumbent exact
terminal. A dynamic source trip count also stays exact. The exact artifact is
owned independently and remains the runtime fallback.

## Performance evidence

All measurements below are offline Metal at 44.1 kHz with 128-frame device
blocks on the baseline M1 Pro. No DAC was constructed or opened. Each retained
load result is three independent 384-block probes after 32 warmup blocks.

| row | audio instructions / divisions | median load | p99 load | overruns / materializer failures / exact fallbacks |
| --- | ---: | ---: | ---: | ---: |
| 6 partials, 6 sections, R=128, 0.2 Hz | 258 / 4 | 9.29% | 11.43% | 0 / 0 / 0 |
| 32 partials, 12 sections, R=64, 0.2 Hz | 234 / 4 | 19.50% | 19.85% | 0 / 0 / 0 |
| 32 partials, 18 sections, R=32, 8 Hz | 234 / 4 | 32.85% | 32.91% | 0 / 0 / 0 |

The structural compiler test uses the widest 10/8 image and includes an
absolute address source. It reports 350 audio instructions and 8 divisions for
each of 6, 12, and 18 sections. Thus the plan is compact in both emitted
fixtures and its audio instruction count is independent of section count. The
large numeric register span in manifests is the global temporary-ID range, not
a live Metal array.

All retained probes also have zero starvation, dispatch failure, epoch-tag
mismatch, activation failure, callback-thread violation, and non-finite output
counters. Activation completed in approximately 5–9 ms in the focused runs.
A runtime-sized transform experiment regressed the hard row to 35.38–36.14%
with synthetic overruns and was rejected. Compile-time support dispatch first
produced a cold/host-contaminated 43.77% probe with 38 overruns alongside
34.90% and 33.34%; the final warm-cache set retained above is
32.80%/32.85%/32.91%, all clean. This distinction is why the harness retains
raw repetitions instead of reporting only a best number.

The reproducible harness is `benchmarks/phaser_time_staging/run.py`.
`--performance-only` refreshes worker probes from already emitted artifacts so
host scheduling noise can be distinguished from compilation pressure. The
harness never creates a DAC. The retained five-row run is
`benchmarks/phaser_time_staging/data/higher-order-smoke-2026-08-18/raw.json`;
its sibling graph/audio/error files preserve the comparisons. The independent
oracle, distribution, boundary, notch, determinism, and rejected-arithmetic
results are retained in the adjacent `qualification.json`.

## Accuracy and continuity evidence

The six-partial incumbent path remains a valid direct comparison. Over 4,096
samples, the retained 6/6/R=128 row has 1.398e-5 maximum absolute error,
7.576e-6 peak-relative error, and 100.48 dB SNR. This is below the stated 1e-4
promotion tolerance.

The expanded 32-partial incumbent generic bank is not an oracle: its fixed
carrier arithmetic loses the dense cancellation and differs from the staged
output by about 4.9. A separate 100-decimal-digit oracle was therefore used for
the quality claim. Against actual Metal output over the first complete staged
interval, the final representation's focused maxima were:

| row | max absolute | max peak-relative |
| --- | ---: | ---: |
| 32/18/R=32 | 3.252e-5 | 5.047e-6 |
| 6/18/R=64, 8 Hz standard center | 3.716e-5 | 6.898e-6 |
| 32/18/R=64, 8 Hz standard center | 3.679e-5 | 5.709e-6 |
| 6/18/R=128, 8 Hz standard center | 7.332e-5 | 1.361e-5 |
| 32/18/R=128, 8 Hz standard center | 7.475e-5 | 1.160e-5 |
| 32/18/R=128, 8 Hz high/max-sweep | 5.098e-6 | 7.544e-7 |

The oracle applies the fixture's authored 3.7 source gain. It explicitly sets
`mpmath.mp.dps = 100`; lower default precision can manufacture milliscale
errors in this problem.

The final stratified audit covered all 162 discrete shape/rate/control rows at
four active absolute starts: 648 independently materialized segments and
48,384 samples. In authored output units, peak-relative error had p50
9.34e-9, p99 1.09e-6, and maximum 2.50e-5; no normalized sample exceeded
1e-4. Absolute error had p50 2.95e-8 and p99 6.50e-6. The conservative Python
arithmetic predicted 34 samples in eight 18-section startup segments just over
1e-4 (maximum 1.345e-4); every predicted outlier family was then checked on
actual Metal, where the worst observed absolute error was 7.475e-5. The older
96-segment continuous random audit remains useful distribution evidence. A
cheaper Kahan-only alternative was rejected because 19 samples exceeded 1e-4
and its worst absolute error reached 2.60e-4.

The first uniform 8/6 attempt is also retained as a negative control. Its
32/18/R=128, 8 Hz high/max-sweep startup reached 2.646e-4 on actual Metal.
Changing only the wide source support to ten and its private weight support to
eight reduced the same row to 5.098e-6. The 64-frame private 6-to-8 weight
change reduced the checked 32/18 startup from 7.346e-5 to 3.679e-5 without
adding an audio instruction.

Thirty-seven independently materialized boundaries across seven aggressive
trajectories had maximum first-difference error 3.355e-6 of peak and 1.382e-5
of the local exact first-difference envelope. None exceeded 1e-4. The 8 Hz
rows exercise the fastest built-in legal LFO and include low/high center and
maximum-sweep cases.

A dense damped-response audit compared 203 musical notches across all section
counts, intervals, and control extremes. It found no center-bin shift on the
8,193-point logarithmic grid (gate: 1%) and a worst depth delta of 0.000137 dB
(gate: 1 dB).

## Absolute-time and failure evidence

The research cockpit materialized six absolute segment identities in forward,
reverse, shuffled, and repeated order; every published byte image matched.
The native materializer test repeats the transform byte-for-byte and now
checks independently generated absolute starts in forward and reverse order,
plus the 10/8 wide layout against independent source/tail calculations.
The worker test verifies rematerialization after a backward clock jump,
retargeted source coordinates, immutable program lifetime across publication,
and exact fallback on an unsafe image. The native parser rejects invalid
magic/layout, non-finite primitives, and section rates that do not share the
admitted sweep geometry.

The tile split also has a regression test for hash-consed scalars shared by
tile and audio work. Such dependencies are duplicated as fold work rather than
becoming unpublished scalar crossings; the audio slot count must remain
unchanged after the split.

## Remaining gates and scope

This is enough to consider promotion into the core libraries, not enough to
enable a product patch:

- A 100-digit counterpart to all 162 harness rows is retained at four active
  starts, but an exhaustive actual-Metal/full-cycle render is not. Compiling
  the ill-conditioned incumbent 32-partial generic bank for every row is both
  slow and numerically misleading; targeted Metal checks cover each modeled
  outlier family.
- Patched signal-rate controls, dynamic source counts, other source sizes,
  unsupported intervals, and phaser-plus-room terminals deliberately use the
  exact path.
- The 12/18 ratios are deterministic benchmark voicings, not final musical
  voicings.
- No matched-level listening has occurred, and automation must not open a DAC.

The main numerical caveat is not the candidate's measured error but oracle
selection: never qualify the 32-partial stress rows against the incumbent fixed
bank alone. Keep both incumbent residuals and the multiprecision result in any
future report.

## Manual listening checkpoint

1. Render exact and staged full-cycle files at identical absolute coordinates.
   For 32-partial rows, also retain the multiprecision numeric report rather
   than treating the incumbent waveform as ground truth.
2. Import `.f64le` as mono little-endian 64-bit float at 44.1 kHz. Do not
   normalize or limit either file.
3. Apply one documented RMS-match gain, randomize A/B identity, and begin at a
   safe monitor level.
4. Compare boundary clicks, low/high-center motion, notch motion/depth, reverse
   and seek starts, and the 12/18-stage color.
5. Record listener, hardware, hashes, gain, and preference. Stop promotion on a
   preferred exact reference, any click/dead region, or lost high-stage color.

Do not connect the flag to the factory demo or merge on automated evidence
alone.
