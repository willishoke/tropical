# Absolute-time phaser staging qualification

This harness emits the isolated `resonator -> phaser -> out` terminal, enables
the higher-order whole-tail Newton/phase-type image, renders an independently
compiled exact JIT oracle and the staged Metal path at matching absolute
coordinates, retains their f64 audio/error artifacts, and probes the off-line
Metal worker. It never constructs or starts a DAC. The representation remains
behind `TROPICAL_PHASER_TIME_STAGING_HIGHER_ORDER=1` and is not connected to the
factory demo.

The image uses 8 source / 6 private weight supports at 32 frames, 8 / 8 at
64 frames, and 10 / 8 at 128 frames. The wider 128-frame source image and the
refined 64/128-frame private weight field cover the high-rate, high-section
startup corners without changing the performance-critical 32-frame kernel.

```bash
python3 benchmarks/phaser_time_staging/run.py \
  --output-dir benchmarks/phaser_time_staging/data/my-run

# Refresh only the three no-DAC worker probes from those emitted artifacts.
python3 benchmarks/phaser_time_staging/run.py \
  --output-dir benchmarks/phaser_time_staging/data/my-run \
  --performance-only

# Refresh staged audio/error files while retaining the incumbent audio.
python3 benchmarks/phaser_time_staging/run.py \
  --output-dir benchmarks/phaser_time_staging/data/my-run \
  --staged-audio-only
```

The default run is a bounded smoke matrix. Add `--full` for all requested
6/32 partial, 6/12/18 section, 128/64/32 interval, 0.02/0.2/8 Hz, and control
extrema combinations. At 0.02 Hz a full LFO cycle is 50 seconds of rendered
audio, so the full run is intentionally not a presubmit test.

Every row records admission provenance, higher-order image sizes, emitted
operation/register/file summaries, raw worker counters and load measurements,
incumbent-exact/staged audio, absolute and RMS error, SNR, block-boundary first
differences, and five millisecond transient error. The incumbent generic bank
is a useful six-partial oracle but is itself ill-conditioned in the expanded
32-partial stress voicings; use the 100-digit higher-order research cockpit for
those quality claims. `--performance-only` reuses an emitted run and refreshes
only its no-DAC worker probes. Keep raw runs; do not substitute CI success for
the manual listening step in the decision note.

Generated `artifacts/` and JIT `cache/` directories are intentionally ignored:
they are machine-specific and tens of megabytes even for the smoke matrix.
Their hashes, sizes, instruction tags, and plan register span remain in
`raw.json`; rerun the command above to reconstruct the full IR/MSL payloads.
