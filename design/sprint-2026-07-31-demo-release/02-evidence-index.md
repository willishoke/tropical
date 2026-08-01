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
assets remain architecture and listening evidence, not release qualification;
the native 44.1 kHz production asset and integrated evidence are pending.

## Implemented gates

| Surface | Reproducer | Current status |
|---|---|---|
| Scope/control priority | `current_module_process`; `playground/scope-arbiter.test.js`; exact-scene qualification JSONL | Focused tests pass. |
| Epoch queue and admission | `current_epoch_tile_queue`; `current_metal_render_worker` | Exact-target, boot-zero, late candidate, admission, and A/B/A regressions pass. |
| Whole-signal morph | `current_metal_render_worker`; `current_metal_kernel` | Offline oracle, hardware exact-E gates, and 10k stress pass. |
| Explicit fixed room decoder | `runExplicitRoomModes` in `tropicaltest` | Lean gate passes for the rejected v1 path; dedicated production `groupedroom` work is pending. |
| Exact-scene harness | `playground/qualification/run.js` | Short pre-room integration smoke passes; not release evidence. |

## Direct evaluator reserve failure

The required direct-path cost fork is recorded under
`benchmarks/demo_release/data/`:

| Run | Result |
|---|---|
| `2026-08-01_23-12-23-338-smoke-b128-r512` | Pre-optimization direct evaluator: first priming write timed out; one starvation and 443 retargets. |
| `2026-08-01_23-16-23-842-smoke-b128-r512` | Algebraically shared direct evaluator: first priming write still timed out; about 17.0 ms per 512-frame morph render, one starvation, and 185 retargets. |

Both runs retain their JSONL, summary, and manifest. They authorize work on the
bounded fixed-scene cache; they are failed diagnostics, not release evidence.

## Required release artifacts (not yet complete)

The final candidate must populate `benchmarks/demo_release/data/` without
overwriting failed runs:

- five cold-boot records;
- per-integrated-change 90-second smoke;
- 10-minute adversarial soak;
- 30-minute normal soak;
- JSONL, summary, manifest, and qualification capture WAV for every run;
- final-room wet/dry/listening WAV and review result;
- focused and full-validation logs; and
- release commit SHA plus clean-worktree status.

Each qualification manifest records graph hash, commit/worktree state, OS/CPU,
toolchain, selected audio device, sample rate, and negotiated quanta. Missing
measurements fail their gate rather than comparing JavaScript `null` as zero.
