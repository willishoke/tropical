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

## Current support envelope

On the canonical M1 Pro, B=128/D=3 is a known unsupported configuration: its
reviewed live row recorded a 5.422916 ms callback against the 2.902494 ms hard
callback deadline, plus one underrun. B=256 remains untested. B=512/D=3 is the
supported candidate, subject to the required final long soak; its passing
short validation is not release qualification. The B=128 result blocks that
configuration, not Metal universally. This records the product decision and
does not authorize another DAC run.

`TROPICAL_METAL_PIPELINE_DEPTH=1..3` is qualification-only. When both controls
are present it overrides the legacy `TROPICAL_METAL_PIPELINE=1`; the legacy
spelling retains D=3. Missing controls retain synchronous Metal. Invalid depth
values refuse during Metal kernel construction.

The latency fixture changes an output-visible raw slot impulsively. The
`write_count` rows preserve the host disciplines' slot-write shapes:
raw=1, glide=3, anchor=2, velocity=2. This isolates future-block transport
latency from intentionally subjective glide onset. The harness fails if the
first changed block differs from D.

The soak compiles the real 512-partial bank. The bank stays reachable behind a
live 1e-8 gain (−160 dB), and a continuous 55 Hz correctness canary runs at
1e-12 amplitude (−240 dB). This is effectively silent at the default device but
keeps every reference comparison nonzero, including after the 2^40 clock jump.
It records:

- real callback count/average/max plus p50/p95/p99 upper bounds from a
  preallocated fixed histogram (1 us resolution, >=20 ms overflow bin);
- a callback-boundary stats epoch, requested and negotiated RtAudio frames,
  and RtAudio underrun/overrun counts;
- progress-gated snapshots and actual callback indices for baseline, periodic
  writes, clock jump, publication, and the first post-publication callback;
- nonzero JIT-reference SNR/max error from a separate, control-thread-only JIT
  runtime at start, post-2^40, midpoint-after-swap, and end;
- resident memory every two seconds, with the growth window beginning two
  seconds after observed hot-swap progress;
- process user+system CPU seconds and measured-wall fraction;
- write and hot-swap walls;
- the exact completed sample index read by every production parameter
  dispatch, replayed by the isolated JIT oracle at that same index;
- a separate offline per-block p50/p95/p99 supporting row.

The audio callback performs no allocation, lock, or I/O for telemetry: it adds
one relaxed histogram increment and services at most one preallocated capture
buffer. Startup status is preserved but separated by an epoch applied at an
actual callback boundary. A negotiated-frame mismatch refuses before stream
start. Any post-reset underrun, callback overrun/stall, missing event/reference,
reference checkpoint not strictly above 100 dB, callback p99 at or above 50%
of the block deadline, or monotonic post-warmup RSS growth marks the row
blocked. Ordinary
end captures wait at least pipeline depth plus one callback after the preceding
write; jump and hot-swap checkpoints are tied to their observed re-prime
progress callbacks.

At B=512 and 44.1 kHz, each callback has an 11.610 ms hard processing
deadline. D=3 ordinary-parameter transport is 34.830 ms from capture to the
first output block. Pipeline transport latency does not enlarge the per-callback
processing deadline; these are different measurements and gates.

At least three post-warmup RSS samples are required; an empty or short series
cannot pass the memory gate.

Before every subprocess, inherited `TROPICAL_*` variables are removed and the
benchmark controls are explicitly set and recorded.

The no-DAC large-clock oracle discriminator reproduces the real heavy graph,
post-swap defaults, and final 15-event schedule from the blocked 60-second
smoke:

```sh
python3 benchmarks/metal_live/run_velocity_oracle_discriminator.py
```

It requires true 1↔0.75 velocity toggles and the velocity-no-op control to
remain above 140 dB on synchronous Metal versus JIT. It then forces a callback
boundary between production dispatches: exact-index replay must remain above
140 dB while the obsolete batch-start oracle must fail below 100 dB. This is a
deterministic harness regression, not a DAC qualification row.

The engine boot block length is independently selectable with
`TROPICAL_BUFFER_LENGTH` (16..16384) before runtime/DAC construction. The
default remains 512 and diagnostics already expose the runtime buffer length.
