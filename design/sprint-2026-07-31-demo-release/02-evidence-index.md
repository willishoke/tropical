# Demo release evidence index

This index distinguishes audition material, engineering diagnostics, and
release evidence. A passing diagnostic on the pre-integration room is not a
release qualification result.

## Frozen identity

| Item | Value |
|---|---|
| Baseline commit | `91ecf0c8a590a416b25d5165dffc9d66a78aad68` |
| Required implementation predecessor | `15e5a39bfce9dd242db59aa6ebc2460987aa2199` |
| Handoff ref | `demo/room-position-handoff-2026-08-01` |
| Branch | `demo/modal-pocket-scene` |
| Qualified runtime candidate | `6660f0668b652a8f3ed2df6088dc691790bd8c09` |
| Evidence checkpoint | `8e32e2bd2275517634bfef12fe55e77a6b755b1f` |
| Primary quanta | `Bdev=128`, `Rgpu=512` |
| Release scope | exact modal-pocket graph on the recorded release Mac |

Production room work starts from the commit named by the handoff ref, directly
atop the required implementation predecessor and handoff package. It does not
start from the baseline or a historical lane worktree.

## Room audition

| Evidence | Location |
|---|---|
| Reproducer and method | `benchmarks/demo_release/room_audition/run.py`, `README.md` |
| Frozen candidate profiles | `benchmarks/demo_release/room_audition/out/profiles.json` |
| Objective measurements | `benchmarks/demo_release/room_audition/out/summary.json`, `summary.md` |
| Loudness-matched WAV set | `benchmarks/demo_release/room_audition/out/*.wav` |
| Frozen Clouds host wrapper | `benchmarks/demo_release/room_audition/clouds_reference.cpp`, `run_reference.py` |
| Clouds stereo and mono gold WAVs | `benchmarks/demo_release/room_audition/reference_out/*.wav` |
| Modal capacity/hybrid experiment | `benchmarks/demo_release/room_audition/run_modal_match.py` |
| Reference-distance results and second-pass WAVs | `benchmarks/demo_release/room_audition/match_out/` |
| Structured recovery derivation and verdict | `design/sprint-2026-07-31-demo-release/03-room-recovery-feasibility.md` |
| Exact stateful/modal/grouped reproducer | `benchmarks/demo_release/room_audition/run_structured_recovery.py` |
| Recovery measurements and listening WAVs | `benchmarks/demo_release/room_audition/recovery_out/` |
| Clouds-nearer grouped fit and cost analysis | `design/sprint-2026-07-31-demo-release/04-clouds-nearer-grouped-fit.md` |
| Grouped carrier fit reproducer | `benchmarks/demo_release/room_audition/run_grouped_clouds_fit.py` |
| Fitted carrier assets, measurements, and WAVs | `benchmarks/demo_release/room_audition/grouped_fit_out/` |
| Reverse POSITION proof and implementation scope | `design/sprint-2026-07-31-demo-release/05-reverse-position-scope.md` |
| Classic anti-causal grouped oracle | `benchmarks/demo_release/room_audition/run_reverse_position.py` |
| POSITION measurements and mono/stereo WAVs | `benchmarks/demo_release/room_audition/reverse_position_out/` |
| Authoritative production handoff | `design/sprint-2026-07-31-demo-release/06-room-position-production-handoff.md` |

All v1 room candidates are rejected. The mono `current_radii` grouped room and
classic reverse/forward POSITION are selected for production. Existing 32 kHz
assets remain architecture and listening evidence, not release qualification.
The native 44.1 kHz production asset, selected fixed-scene cache, integrated
hardware evidence, and continuous release WAVs are now committed.

## Implemented gates

| Surface | Reproducer | Current status |
|---|---|---|
| Scope/control independence | `current_module_process`; `playground/phase-lock.test.js`; exact-scene qualification JSONL | Immutable snapshot, hot-swap/name pinning, separate-artifact control projection, and independent phase-lock tests pass. |
| Epoch queue and admission | `current_epoch_tile_queue`; `current_metal_render_worker` | Exact-target, boot-zero, late candidate, admission, and A/B/A regressions pass. |
| Whole-signal morph | `current_metal_render_worker`; `current_metal_kernel` | Offline oracle, hardware exact-E gates, and 10k stress pass. |
| Explicit fixed room decoder | `runExplicitRoomModes` and `runGroupedRoomContract` in `tropicaltest` | Rejected ordinary-room compatibility and both direct/cache Plan-6 production seams pass. |
| Exact-scene harness | `playground/qualification/run.js` | The RCU/projection candidate passes the user-approved 10-minute adversarial soak. The 30-minute normal soak was explicitly waived by the user, not passed. |

## Direct evaluator reserve failure

The required direct-path cost fork is recorded under
`benchmarks/demo_release/data/`:

| Run | Result |
|---|---|
| `2026-08-01_23-12-23-338-smoke-b128-r512` | Pre-optimization direct evaluator: first priming write timed out; one starvation and 443 retargets. |
| `2026-08-01_23-16-23-842-smoke-b128-r512` | Algebraically shared direct evaluator: first priming write still timed out; about 17.0 ms per 512-frame morph render, one starvation, and 185 retargets. |

Both runs retain their JSONL, summary, and manifest. They authorize work on the
bounded fixed-scene cache; they are failed diagnostics, not release evidence.

## Selected fixed-scene cache

| Evidence | Location / result |
|---|---|
| Generator | `benchmarks/demo_release/room_audition/generate_grouped_room_scene_cache.py` |
| Payload and manifest | `playground/assets/grouped-room/clouds-current-radii-mono-v1-scene-44100.{f32le,json}`; 5,644,800 bytes; SHA-256 `22b534e561aa1fef8aa4535ff321ee5df90c9cfb2743c274fc44d183216a615e` |
| Endpoint/FLOW probe | `run_grouped_room_scene_cache_probe.py`; carrier convolution vs infinite grouped equations `5.04e-14` causal / `1.59e-13` reverse; JIT/Metal cache reads vs binary64 generation oracle `<=4.54e-8` across integer/fractional/stopped/reverse FLOW |
| Integrated smoke | `2026-08-02_00-09-48-317-smoke-b128-r512`; clean candidate, pass, zero sticky runtime/Metal/DAC/capture faults and zero retargets |
| Revised pizzicato balance | ROOM default `1.0`, range `0…1.5`; shared section gain `3×`, fixed room compensation `3×`; authored-level dry / POSITION `+1` / `+0.5` peaks `−6.25/−2.43/−3.21` dBFS over the complete 16-second scene |

## 2026-08-02 pizzicato integration qualification

| Gate | Evidence / result |
|---|---|
| Candidate | `4c5f2e86e5a5ba3c2beb6934bda3f6f085d74c60`; graph SHA-256 `bfd717f75fa68ee871a1f0ac11625a8f71094b978eaf470002fa217a4047893f`; exact 16-beat production scene |
| Muted 90-second smoke | `2026-08-02_07-56-20-619-smoke-b128-r512`; 90.09 measured seconds at `Bdev=128`, `Rgpu=512`; output forced to zero after load; all 32 captures intentionally silent |
| Controls | 2,950 dispatched writes; scheduled p95/p99 `11.50/21.43 ms`; audible-activation p95/max `38.02/56.99 ms`; every first/reversal/final delivery gate passed |
| Scopes | 5,420 completed frames, zero preemptions/errors; 59.75 idle fps; RPC p95/p99 `5.36/5.54 ms`; frame-interval p95 `18.89 ms` |
| Runtime/Metal/DAC | Average 512-frame morph render `6.14 ms`; zero starvation, tag mismatch, retarget, dispatch, activation, morph, stale-completion, ownership, callback-thread, underrun, overrun, non-finite, or clamp faults |
| Release WAVs | `room_audition/release_out/`; six continuous 24-bit mono 44.1 kHz files from clean integration commit `8c9f76a`; endpoint parity about `1.1e-16`; final +1/choreography peak `−2.43 dBFS`; no normalization or limiter |

## 2026-08-02 release qualification

| Gate | Evidence / result |
|---|---|
| Five cold boots | `2026-08-02_00-08-50-037-cold-boots`; 5/5 pass from clean `6660f06`, FLOW stress disabled only for boot stability; every runtime, Metal, and DAC fault counter zero |
| 90-second smoke | `2026-08-02_00-09-48-317-smoke-b128-r512`; 2,971 dispatched writes, scheduled p95 `10.99 ms`, audible p95 `37.24 ms`, scope `23.98 fps`, zero faults/retargets |
| 10-minute adversarial | `2026-08-02_00-11-44-755-adversarial-b128-r512`; 19,649 dispatched writes and 843 health records, scheduled p95 `10.99 ms`, audible p95 `37.23 ms`, scope `23.91 fps`; all transient and final fault counters zero |
| 30-minute normal | Explicitly waived by the user after the completed 10-minute soak. The cancelled partial run is not committed and is not called a pass. |
| Full validation | `validation/2026-08-02_8e32e2b_make-validate.log`; trust audit pass, `tropicaltest` 123/123, Bun 137 pass/1 environment duplicate skip/0 fail, CTest 4/4 |
| Release listening set | `room_audition/release_out/`; six continuous 24-bit mono 44.1 kHz WAVs, native JIT/cache parity around `1e-16`, final choreography peak `−5.01 dBFS`, no normalization or limiter |

Every qualification manifest records commit `6660f06`, an empty worktree
status, graph SHA-256 `db884d12d4b87a39fc8b854372ec903f1d6b90d531b89a0e4be154685e8680d1`,
the MacBook Pro speaker device, 44.1 kHz, `Bdev=128`, and `Rgpu=512`.

The qualification `capture.wav` files preserve sample indices from the
qualification-only one-buffer next-callback sampler. They contain explicit
zero-filled holes where control-plane polling skipped a callback and are not
continuous listening files or evidence of DAC silence. The continuous
endpoint, scrub, dry, and final-scene files under `release_out/` are the
listening artifacts.

## 2026-08-02 immutable-scope correction

The interaction freeze was superseded after it proved visibly discontinuous
during knob scrubs. Code candidate `5d004cd` publishes ref-counted immutable
scope snapshots and a separate JIT-only projection artifact; the audio plan no
longer carries inspection outputs. One frontend canvas overlays five active
fundamental modes, each with an independent interpolated positive-going
zero-crossing lock.

| Gate | Evidence / result |
|---|---|
| Focused frontend | `playground/phase-lock.test.js`, `playground/scope-view.test.js`, plus scene/RPC/sender/qualification suites; 27/27 pass |
| Native runtime/backend | CTest 4/4; `current_module_process` 22/22 includes paused-reader publication, hot-swap name pinning, lifetime reuse, and mismatched-layout control projection |
| Lean/JIT trust suite | `tropicaltest` 123/123 |
| Full repository validation | `validation/2026-08-02_scope-rcu_make-validate.log`; `make validate` exits 0 with `tropicaltest` 123/123, web/frontend 136 pass plus 1 intentional skip, and CTest 4/4 including Metal |
| Production socket | cold graph load `960 ms`; 21 published scope entries; all 20 requested projections discovered; five-channel render succeeds |
| 10-minute adversarial | `2026-08-02_02-07-59-280-adversarial-b128-r512`; 600.75 measured seconds, 19,667 dispatched writes, scheduled p95/p99 `11.18/22.15 ms`, audible p95/max `37.40/59.17 ms`, scope RPC p99 `0.738 ms`, `23.94 fps`, zero scope preemptions and zero faults |

The correction soak records graph SHA-256
`f7dab21a7dd2671608d27474a072266cfa7e7e6ebb2b5c7f97ddc54989b01abe`,
MacBook Pro speakers, 44.1 kHz, `Bdev=128`, and `Rgpu=512`. Across 19,675
acknowledged epochs and 207,080 callbacks it reports zero underruns, overruns,
Metal starvation/tag mismatch/retarget/dispatch/activation failure, ownership
failure, callback-thread violation, non-finite sample, clamp, or all-zero
capture block. The manifest's dirty entries are the documentation edits and
the run's own evidence files; no code file differed from candidate `5d004cd`.

### Phase-view visual correction

The long soak above proved the RCU/control/audio path, but it did not detect a
real visual regression: per-frame peak normalization exactly canceled the
projected modal envelope, while the retained 384-point budget decimated each
trace by five before fractional phase locking. Candidate `b706d17` replaces the
normalizer with the analytic paired-envelope maximum, uses the freshest valid
crossing, rejects silent/DC-only windows as unlocked, and raises the projection
request to stride 1.

| Gate | Evidence / result |
|---|---|
| Raw projection content | `playground/qualification/scope-content.js`; 120/120 production-socket observations pass across 20 taps × 6 envelope ages, with at least 1,791/1,792 distinct raw samples, maximum relative lock error `2.47e-15`, visible attack/decay per voice, and RPC p99 `3.907 ms` |
| Focused frontend | 27/27 pass; fixed-scale amplitude, quiet-tail locking, freshest crossing, silent/DC refusal, and stride-1 profile are pinned |
| Full web/JIT/Metal | 143 pass, 1 intentional capability skip, 0 fail across 12 files |
| Integrated corrected profile | `2026-08-02_02-48-59-087-smoke-b128-r512`; 15.38 measured seconds, 521 dispatched writes, scope p99 `4.391 ms`, `24.00 fps`, zero preemptions, and all 15 gates pass |

This short correction run does not replace or extend the user-approved
10-minute duration gate. It specifically qualifies the more expensive
stride-1 visual profile. Subjective confirmation of the corrected phase view
remains a user acceptance item and is not inferred from telemetry.

### Centered 60 Hz log-phase correction

The first fixed-scale correction still chose a physical trigger cycle for each
frame. At 24 fps, the 55–73 Hz fundamentals in the first two chords changed
cycle age in visibly larger steps than the later, higher register. Candidate
`26a3262` separates the two responsibilities: raw scope data supplies the
centered carrier shape, while the exact paired modal envelope at audible-now
supplies its displayed height. Visible cycle count grows logarithmically from
`1.5` at 55 Hz instead of linearly with frequency.

| Gate | Evidence / result |
|---|---|
| Center/content oracle | `playground/qualification/scope-content.js`; 120/120 observations pass, at least 2,047/2,048 distinct raw points, maximum relative center-lock error `5.71e-15`, RPC p99 `5.171 ms` |
| Focused frontend | 31/31 pass; exact audible-now envelope, centered crossings, log-period density, amplitude-jitter rejection, stride 1, and a 60-on-120-Hz scheduler are pinned |
| Full web/JIT/Metal | 147 pass, 1 intentional capability skip, 0 fail across 12 files |
| Integrated 60 Hz profile | `2026-08-02_03-12-06-662-smoke-b128-r512`; 15.54 measured seconds, 527 dispatched writes, scope p99 `4.462 ms`, frame p95 `19.30 ms`, `59.84 fps`, zero preemptions, and all 15 gates pass |

The 60 Hz run specifically replaces the earlier 24 fps visual-profile smoke;
it does not repeat or extend the accepted 10-minute runtime-duration gate.
Subjective confirmation remains pending.

### Cycle-invariant envelope demodulation

Centered locking alone did not make the low A−11 and D7·9 fundamentals a
stationary display: a 55 Hz cycle spans enough time for the paired attack/decay
envelope to change visibly within that cycle. Candidate `2aea011` now divides
the raw projection by the exact envelope at every source sample, locks and
resamples that unit carrier, and restores amplitude once from the exact
audible-now envelope. This is display-only demodulation; the audio graph and
its modal envelope are unchanged.

| Gate | Evidence / result |
|---|---|
| Production carrier oracle | `playground/qualification/scope-content.js`; 360/360 observations pass across 20 taps × 6 ages × 3 equivalent 16-second loop positions, with audio never started |
| Shape and repeatability | Maximum analytic carrier-shape error `3.50e-4`, carrier peak error `1.56e-4`, relative center-lock error `4.80e-15`, and equivalent-loop error `2.87e-7` of displayed amplitude |
| Focused frontend | 35/35 pass across scene, scope, phase-lock, RPC, sender, and qualification suites |
| Full repository | 151 pass, 1 intentional Metal capability skip, 0 fail across the unrestricted web/JIT/Metal run plus the two MCP contracts run against their documented built Lean RPC engine |
| Muted integrated 60 Hz profile | `2026-08-02_05-51-52-039-smoke-b128-r512`; output level exactly zero, 10.07 measured seconds, 337 dispatched writes, scope p99 `5.742 ms`, frame p95 `17.90 ms`, `59.63 fps`, zero preemptions, zero underruns/overruns or other faults, and all 15 gates pass |

The muted hardware mode still runs the actual device clock and Metal worker.
Its 32 all-zero capture blocks are the asserted output condition, not suppressed
data; nonfinite/clamped samples and all runtime, Metal, and DAC faults remain
blocking. It intentionally emits no capture WAV. Subjective confirmation of
the corrected A−11 and D7·9 display remains pending.

### Consecutive-frame coefficient-cache correction

The cycle-invariance oracle above was necessary but insufficient. It wrote
`master.tau_base` before every `render_window`, producing a fresh control
version every time. That accidentally forced the scope coefficient kernel to
run on every observation. The real UI makes many reads of one immutable
version. Its scope workspace reset scalar slots from the pre-coefficient RCU
control image on every read, but reran the coefficient kernel only when the
version changed. Consequently the first frame was correct and later frames
used erased scalar coefficients—the apparent phase/amplitude jitter and
constant traces reported in A−11 and D7·9.

Candidate `9108ab2` preserves a separate fully materialized scalar-slot image
for the pinned version and copies it into per-frame scratch. Candidate
`5e769d0` makes the frontend and probes consume one shared production transform
and adds a continuously advancing, muted hardware probe.

| Gate | Evidence / result |
|---|---|
| Native same-version regression | `current_module_process` 23/23; a separate scope artifact renders the same scalar coefficient twice before and after a control publication |
| Pre-fix discriminator | `playground/qualification/scope-live.js` reproduced A−11 mode 2 with zero positive crossings on live frame 2 while its raw peak remained nonzero |
| Post-fix continuous content | 360/360 consecutive live frames pass: 90 for each chord through the exact shared frontend transform; captured nonzero output samples `0`, scope preemptions `0` |
| A−11 / D7·9 stability | Maximum normalized frame-shape deltas `2.16e-4` / `2.52e-4`; maximum relative lock errors `4.75e-15` / `5.35e-15`; every frame active and locked |
| Focused frontend and silent oracle | 36/36 frontend tests pass; 360/360 static production observations pass with audio never started |
| Full repository/native | Bun 152 pass, 1 intentional capability skip, 0 fail across 12 files; CTest 4/4 including the 23-case module/RCU target and full Metal kernel target |
| Muted integrated profile | `2026-08-02_06-20-19-550-smoke-b128-r512`; 10.05 measured seconds, 336 writes, scope p99 `4.634 ms`, frame p95 `17.93 ms`, `60.10 fps`, zero preemptions, zero underruns/overruns or other faults, and all 15 gates pass |

This section corrects the causal claim attached to DR-22: envelope
demodulation remains valid display processing, but it was not the source of the
large production jitter. Only a repeated read at an unchanged control version
distinguishes the actual defect.

### Uniform 2× time scale

Candidate `91232a6` removes DR-21's logarithmic cycle-density mapping. Every
mode now occupies the same 1,792-sample (`40.63 ms`) centered span—twice the
896-sample base duration in the same width. The request grows to 2,752 stride-1
points so the 55 Hz mode retains a complete crossing-selection period outside
the visible span.

| Gate | Evidence / result |
|---|---|
| Uniform-axis invariant | Cycle count is exactly `frequency × 1792 / 44100`; all traces therefore share one physical time-per-width ratio |
| Silent production oracle | 360/360 observations pass; scope RPC p99 `7.084 ms`, maximum carrier-shape error `3.50e-4`, and maximum equivalent-loop error `6.46e-7` of displayed amplitude |
| Consecutive live content | 360/360 frames pass with every trace active and locked; A−11 / D7·9 maximum frame-shape deltas `2.18e-4` / `2.56e-4`; captured nonzero output samples `0` |
| Focused frontend | 36/36 pass, including the uniform 2× physical-time invariant |
| Muted integrated profile | `2026-08-02_06-40-36-235-smoke-b128-r512`; 10.07 measured seconds, 336 writes, scope p99 `5.567 ms`, frame p95 `17.97 ms`, `59.98 fps`, zero preemptions, zero underruns/overruns or other faults, and all 15 gates pass |

## Release artifact status

The evidence package now contains, without overwriting failed runs:

- five cold-boot records, the clean 90-second smoke, and the completed
  10-minute adversarial soak;
- JSONL, summary, and manifest for every completed run, plus qualification
  capture WAVs for audible runs (the muted visual diagnostic intentionally has
  none);
- final-room wet endpoints, scrub, dry reference, and integrated listening WAV;
- the final full-validation log; and
- qualified commit SHA, clean-worktree state, graph/profile hashes, machine
  data, and raw telemetry.

The original 30-minute normal-soak requirement is covered only by DR-18's
explicit user waiver. Final headphone/monitor approval of
`release_out/final_scene_position_choreography.wav` remains pending, so the
listening gate must not yet be described as approved.

Each qualification manifest records graph hash, commit/worktree state, OS/CPU,
toolchain, selected audio device, sample rate, and negotiated quanta. Missing
measurements fail their gate rather than comparing JavaScript `null` as zero.
