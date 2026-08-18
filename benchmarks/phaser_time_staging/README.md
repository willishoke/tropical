# Absolute-time phaser staging qualification

This harness emits the isolated `resonator -> phaser -> out` terminal, enables
the falsification-only mixed ordinary/first-order-DD endpoint image, renders
the exact JIT and staged Metal paths at matching absolute coordinates, retains
their f64 audio/error artifacts, and probes the off-line Metal worker. It never
constructs or starts a DAC. The mixed image is intentionally separate from the
original ordinary endpoint experiment and is not a production admission path.

```bash
python3 benchmarks/phaser_time_staging/run.py \
  --output-dir benchmarks/phaser_time_staging/data/my-run
```

The default run is a bounded smoke matrix. Add `--full` for all requested
6/32 partial, 6/12/18 section, 128/64/32 interval, 0.02/0.2/8 Hz, and control
extrema combinations. At 0.02 Hz a full LFO cycle is 50 seconds of rendered
audio, so the full run is intentionally not a presubmit test.

Every row records admission provenance, the interval-wide hot-pair count and
first-order coverage, emitted operation/register/file summaries, raw worker
counters and load measurements, exact/staged audio, absolute and RMS error,
SNR, block-boundary first differences, and five millisecond transient error.
When audio is non-finite, quality metrics are `null` and the runtime's explicit
non-finite count is retained. Keep raw runs; do not substitute CI success for
the manual listening step in the decision note.

Generated `artifacts/` and JIT `cache/` directories are intentionally ignored:
they are machine-specific and tens of megabytes even for the smoke matrix.
Their hashes, sizes, instruction tags, and plan register span remain in
`raw.json`; rerun the command above to reconstruct the full IR/MSL payloads.
