# Metal live qualification

The qualification harness uses the real emitted Metal kernel and mandatory
dual-loaded JIT. It has two modes:

```sh
# B=128/256/512 × D=1/2/3, four write disciplines
benchmarks/metal_live/run.sh --mode latency

# Required live default-device row (30 minutes)
benchmarks/metal_live/run.sh --mode soak \
  --duration-seconds 1800 --buffer 512 --depth 3
```

`TROPICAL_METAL_PIPELINE_DEPTH=1..3` is qualification-only. When both controls
are present it overrides the legacy `TROPICAL_METAL_PIPELINE=1`; the legacy
spelling retains D=3. Missing controls retain synchronous Metal. Invalid depth
values refuse during Metal kernel construction.

The latency fixture changes an output-visible raw slot impulsively. The
`write_count` rows preserve the host disciplines' slot-write shapes:
raw=1, glide=3, anchor=2, velocity=2. This isolates future-block transport
latency from intentionally subjective glide onset. The harness fails if the
first changed block differs from D.

The soak compiles the real 512-partial bank. Its sink passes through a live
parameter-backed zero multiplier: the modal workload remains reachable and
runs, but the default-device output is silent. It records:

- real `tropical_dac_*` callback count/average/max and RtAudio
  underrun/overrun counts;
- snapshots before reset, at a clean post-reset baseline, after periodic
  writes, after a clock jump, after hot-swap, and after stop;
- resident memory every five seconds;
- write and hot-swap walls;
- a separate offline per-block p50/p95/p99 supporting row.

Startup status is preserved but separated from the measured window. Any
post-reset underrun marks the row blocked; the test is never weakened.

The engine boot block length is independently selectable with
`TROPICAL_BUFFER_LENGTH` (16..16384) before runtime/DAC construction. The
default remains 512 and diagnostics already expose the runtime buffer length.
