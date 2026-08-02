# Modal pocket demo release sprint

- **Baseline:** `91ecf0c` (`demo/modal-pocket-scene`)
- **Required implementation checkpoint:** `15e5a39bfce9dd242db59aa6ebc2460987aa2199`
- **Handoff ref:** `demo/room-position-handoff-2026-08-01`
- **Sprint length:** four engineering days, with an audible candidate by the end
  of Day 2
- **Product target:** the one fixed 16-second modal scene on the release Mac
- **Sprint type:** release recovery, interaction correctness, and perceptual
  finishing
- **Status:** engineering candidate qualified under an explicit user waiver of
  the 30-minute soak; final headphone/monitor listening approval remains open

## 2026-08-01 room-lane production decision

The authoritative production contract is
[`06-room-position-production-handoff.md`](06-room-position-production-handoff.md).
It selects the accepted mono `current_radii` grouped transfer on the four fixed
Metal hits, a bipolar classic reverse/forward POSITION control in place of
LENGTH, the direct dual analytic evaluator with a fixed-scene mono cache only
as its measured fallback, and no live room size or longer-decay work for this
release.

At handoff time, one production gate preceded compiler integration: all
accepted audition and proof assets had been generated at Clouds' native 32 kHz
and only then resampled for listening, while the shipping engine is native
44.1 kHz. The same selected architecture therefore had to be fitted, proved,
and auditioned at 44.1 kHz. The production result below records that this asset
qualification gap is closed without reopening the architecture decision.

The initial Foundry 24, Industrial Cathedral 32, and Tanker 48 auditions are
all rejected. None is eligible for scene integration. Their sparse stationary
late fields read as struck resonant objects, not rooms; the room selection in
the original plan below is therefore superseded rather than awaiting a preset
choice.

Mutable Instruments Clouds is now the gold reference, frozen to official
Eurorack commit `08460a69a7e1f7a81c5a2abcc7189c9a6b7208d4`. The second pass
renders both its native stereo field and a mono fold-down, then compares modal
candidates against echo-density growth, time/frequency energy, and spectral
flatness. The reference remains an audition oracle; its stateful DSP is not
silently imported into the product.

The earlier architecture fork is resolved by the production handoff. The
paragraphs below retain the evidence trail for that decision.

## 2026-08-02 production result

The qualified runtime candidate is
`6660f0668b652a8f3ed2df6088dc691790bd8c09`; the committed hardware evidence
checkpoint is `8e32e2bd2275517634bfef12fe55e77a6b755b1f`. Production selected the
5.38 MiB fixed-scene causal/reverse cache after the direct evaluator's recorded
reserve failure. The native asset remains its generator and oracle.

Five clean transport-stable cold boots, a clean 90-second exact-scene smoke,
and the 10-minute adversarial soak pass at `Bdev=128`, `Rgpu=512`. The
adversarial run dispatched 19,649 writes across 843 health records with zero
runtime, Metal, callback, DAC, non-finite, clamped, or sampled-silent faults;
scheduled-write p95 was `10.99 ms`, audible-activation p95 was `37.23 ms`, and
scope cadence was `23.91 fps`. Full validation against the evidence checkpoint
passes the trust audit, `tropicaltest` 123/123, Bun 137 pass/1 duplicate
environment skip/0 fail, and CTest 4/4.

At the user's direction, the 30-minute normal soak was cancelled and is
explicitly waived in favor of the completed 10-minute adversarial result. It
is not represented as a passing 30-minute run. Continuous cache-backed release
WAVs are committed under `benchmarks/demo_release/room_audition/release_out/`;
the final POSITION choreography peaks at `−5.01 dBFS` without normalization or
a limiter. Final headphone/monitor approval of that capture remains the only
open acceptance decision.

### 2026-08-02 chord-derived percussion revision

Subsequent listening supersedes only the fixed percussion score and its cached
room witness, not the accepted room transfer, POSITION semantics, cache ABI,
scope lock, or mono output contract. The four inharmonic Metal hits are replaced
by audition candidate 02: a chord-derived pizzicato on every beat, with an open
dyad at each chord downbeat and three lighter single-note ghosts. The approved
balance uses a shared `3×` pizzicato-section gain and a further fixed `3×` room
compensation while ROOM retains its `0…1.5×` user range.

The direct grouped evaluator is instrument-specific: its immutable tables and
compiler admission check require the retired twelve Metal pole coordinates.
It therefore remains historical room/ABI evidence rather than the revised
scene-cache generator. The shipping cache ABI is unchanged at 5,644,800 bytes;
its new arms are generated offline from the exact 200-row fixed score and the
accepted native grouped-carrier fit. The finite FFT construction agrees with
the infinite analytic grouped equations at `5.04e-14` causal and `1.59e-13`
reverse relative L2; JIT/Metal reads agree with the binary64 generation oracle
within `4.54e-8` across integer, fractional, stopped, and reverse FLOW cases.
The new immutable cache hash is `22b534e561aa1fef8aa4535ff321ee5df90c9cfb2743c274fc44d183216a615e`.

The complete offline native graph peaks at `−6.25 dBFS` dry, `−2.43 dBFS` at
POSITION `+1`, and `−3.21 dBFS` at `+0.5`; there is no normalization or limiter.
The prior hardware qualification remains evidence for the engine/cache path,
but is not silently relabeled as qualification of this denser revised graph.
The committed revised candidate `4c5f2e8` passed a separate output-muted
90.09-second exact-scene smoke at `Bdev=128`, `Rgpu=512`: 2,950 dispatched
writes, 59.75 scope fps, 5.54 ms scope-RPC p99, zero preemptions, and zero
runtime, Metal, callback, DAC, ownership, non-finite, or clamp faults. All 32
qualification captures were intentionally and verifiably silent.

Exploratory result DR-11 enabled the selected architecture. A 9,133-mode
Freeverb-style LTI teacher was recovered exactly by collecting its complete
delay pole lattices into 12 periodic carriers and analytically convolving those
groups with the 12 authored metal rows. The grouped result matches ordinary
convolution to
`2.72e-13` relative L2 with 144 source/group interactions, a 63.4× reduction
from independent pole evaluation. It does not claim to reproduce Clouds'
time-varying modulation.

Exploratory result DR-12 keeps that 12-group cost and fits the complete lattice
carriers offline to the Clouds impulse. The fitted mono candidate moves late
spectral flatness from the teacher's `0.1882` to `0.1000` against Clouds'
`0.1075`; deterministic residue-phase scrambling also demonstrated a possible
stereo field. One Metal hit remains 144 grouped interactions; four hits remain
576. The stereo experiment remains evidence only; production selects the
accepted mono output.

User listening accepts the mono `current_radii` fitted impulse as the MVP room,
and production is now authorized. The MVP keeps one frozen size and no
`LENGTH` control. Room size and longer-decay variants are deferred rather than
active questions.

Reverse-time addressability is selected production work in DR-13. Reversing a
known source, applying the fixed causal room, and reversing both is equivalent
to applying the room's anti-causal time reverse to the original source.
Tropical's authored modal sources provide the required future values
analytically, so a bipolar wet `POSITION` can continuously place energy before
or after the dry
anchor by mixing anti-causal and causal grouped responses. The exact grouped
formula matches literal reverse-source/reverb/reverse at `1.88e-10` mid and
`2.55e-10` side relative error. The incumbent direction path mirrors the whole
composed output and is not equivalent (`1.0` NRMSE), so it is excluded.

The preferred direct dual evaluator costs 1,152 four-hit interaction
equivalents at an intermediate POSITION. At native 44.1 kHz its mono
forward-plus-reverse prefix payload is 2.59 MiB. If it misses reserve, two
immutable 16-second mono wet bases cost 5.38 MiB. `POSITION` replaces LENGTH
and targets the pre-tail into the six-second Metal hit. The claim does not
extend to unscheduled live input.

## Outcome

Ship one convincing, mouse-playable scene whose cutoff responds promptly and
never clicks or drops to silence, whose two scopes remain legible without
blocking control, and whose wet metal return unmistakably reads as a very large
room.

This sprint qualifies that artifact on the release machine. It does **not**
declare a universal live-Metal operating envelope, solve arbitrary
time-varying filters, or build a general room simulator.

The release claim is deliberately narrow:

> On the qualified Mac, this exact graph, UI workload, and control set can run
> for the release soak with bounded interaction latency, no runtime faults, and
> the frozen audible result.

## Why this is a release blocker

The baseline investigation reproduced three independent failures.

| Failure | Exact-scene evidence | Consequence |
|---|---|---|
| Scope/control priority inversion | With two 24 fps scope requests active, cutoff RPC latency measured 23.8 ms median, 87.1 ms p90, and 171.2 ms maximum; scope calls reached 160.7 ms. | Knob response depends on whether a long scope render already owns the socket and runtime lock. |
| Metal epoch fragility | One repeated 128-frame sweep recorded nine tile starvations while the DAC reported zero underruns; another scope-loaded sweep performed 180 retargets for 60 accepted writes. A sticky tag mismatch also appeared during boot priming. | The callback can emit fail-silent blocks, and the visible DAC counters do not reveal the fault. |
| Insufficient transition semantics | The current handoff matches only `old(E)` and `new(E)`, then removes that scalar correction over one render quantum. | Cutoff pole replacement is value-continuous at one sample but does not preserve waveform slope, phase, or filter trajectory. |

The room complaint is separate and equally real: the current room is a small
log-spaced resonator bank with one uniform RT60. It has no propagation delay,
early reflections, echo-density growth, dimensions, or frequency-dependent
decay. More gain or a longer RT60 cannot make that model sound like a cathedral
or tanker.

## Product and semantic decisions

These decisions are part of the sprint boundary, not questions for individual
implementation agents.

1. **The fixed demo wins over generality.** A containment repair is acceptable
   when it is safe, measured, and explicitly documented as demo-scoped.
2. **Control and visualization are independent readers.** Controls publish a
   small immutable image; a scope pins it without owning audio/control storage.
   Neither lane may wait on the other's work.
3. **Bounded supersession is intentional.** The first value, one direction
   reversal, and the final value are retained. The engine is not required to
   render every pointer event.
4. **A waveform morph is an honest approximation.** The release does not claim
   to solve the exact continuously time-varying filter. It does require a
   finite, reproducible old/new transition with no click or silent block.
5. **One authored grouped room is enough.** Live geometry, decay, and profile
   selection are deferred. The demo ships the frozen mono grouped profile with
   glided ROOM and POSITION controls.
6. **No fault is cosmetic.** Metal starvation, tag mismatch, dispatch failure,
   ownership failure, non-finite output, DAC underrun, or DAC overrun blocks the
   release even when it is hard to hear.
7. **No hidden qualification claim.** Passing this sprint does not overturn the
   repository's existing decision to withhold general live-Metal support.

## Priority tiers

### P0 — required to ship

- Scope work cannot hold up a control write for the duration of a scope window.
- The exact scene completes the gesture smoke and release soak with all runtime
  and DAC fault counters at zero.
- Cutoff changes use a transition whose full waveform is continuous enough to
  pass the deterministic oracle and listening gate; the scalar boundary patch
  alone is not sufficient.
- The wet metal return passes the authored large-room WAV review before it is
  integrated into the scene.
- First, reversal, and final gesture values converge deterministically.
- The final build, focused tests, full gates, and release evidence are produced
  from a clean integration worktree.

### P1 — retain unless it threatens P0

- Idle scopes average at least 23.5 of the authored 24 frames per second and
  resume within 150 ms after a gesture.
- Runtime fault and epoch telemetry is available through the demo
  qualification path rather than inferred from DAC statistics.
- The fixed room has 20 ms glided ROOM and POSITION controls; neither causes a
  room-coefficient epoch.
- The scene keeps both named scopes and all existing mouse/keyboard navigation.

### P2 — cut immediately if P0 slips

- Full-resolution or instrumentation-grade scopes.
- A live room-size/geometry control.
- More than one room profile.
- General multi-client RPC scheduling.
- Universal overlapping parameter morphs.
- General live-Metal support qualification.

## Minimum shippable architecture

```text
Lane A: scope/control isolation ───────────────────────────┐
                                                          │
Lane B: epoch admission + queue safety ─→ Lane C: morph ──┼─→ Lane E: scene integration
                                                          │              │
Lane D: room prototype + fixed implementation ────────────┘              ▼
                                                               Lane F: qualification
                                                                        │
                                                                        ▼
                                                               Lane G: release
```

### Control and scope path

The shipping scope path has four parts:

1. Electron opens independent high-priority control and low-priority scope
   socket connections. The existing server already supports multiple clients;
   connection separation prevents a long scope response from serializing later
   control lines in the same reader thread.
2. Audio and scope programs are separate artifacts published in one generation.
   The audio plan contains no inspection outputs; the JIT-only scope plan holds
   twenty explicit fundamental-mode projections and never enters the Metal
   queue.
3. `render_window` pins one immutable program/control snapshot and evaluates it
   in scope-owned workspace. Control publication swaps a new small control
   image by name and never waits for a scope reader or its mutex. The workspace
   separately caches the fully materialized scalar-coefficient slot image for
   that control version; every frame resets kernel scratch from that image,
   never from the pre-coefficient control projection.
4. One canvas overlays the active chord's five modes. Each trace receives an
   independent interpolated positive-going zero-crossing lock at the center
   graticule and one shared analytic volts/div calibration. Every trace uses
   one uniform 1,792-sample / 40.63 ms time span (2× the 896-sample base view).
   The exact envelope is divided out point-by-point before
   phase locking, then its single audible-now value restores display amplitude;
   envelope slope therefore cannot change one visible cycle relative to the
   next. The stride-1 view is display-synchronized at a 60 Hz cap.

The earlier preempt/freeze containment was removed after it produced visible
gesture pauses. Ref-counted immutable program assets keep an in-flight frame
valid across hot-swap and state-slot reuse.

### Epoch admission and render safety

Control publication must stop speculatively choosing an exact epoch before
expensive epoch-independent work and while an older activation still owns the
worker. The intended shipping rule for the raw modal cutoff is:

1. materialize the raw target's epoch-independent coefficient snapshot;
2. wait off the audio thread until the worker can admit one transition;
3. choose `E` from the then-current published device/source boundary with a
   measured post-render publication margin;
4. compute any genuinely `E`-dependent companions for that `E` once;
5. render and validate the required candidate reserve;
6. publish only when the callback can reach `E` without consuming an
   unprepared tile; and
7. acknowledge accepted, published, and audible/superseded states distinctly
   in qualification telemetry.

`Bdev` and `Rgpu` remain separate. The primary candidate keeps `Bdev=128` and
restores `Rgpu=512`; a measured 256- or 1024-frame render quantum may replace it
only through the qualification/fallback gates. The demo must not force the
128-frame device callback to imply a 128-frame GPU tile. Configuration
selection is evidence-driven, not a source constant presented as generally
supported.

The queue must prefill enough candidate depth to survive activation. Publishing
after one small tile is not accepted merely because the UI receives a fast
acknowledgement.

### Cutoff transition

The P0 transition must blend actual old and new trajectories, not only their
first samples. The primary implementation is an off-thread whole-signal epoch
morph over one 512-sample render tile:

```text
y(k) = (1 - w(k)) old(E + k) + w(k) new(E + k)
```

For each admitted continuous control, the Metal worker:

1. renders the old complete graph at `[E, E + Rgpu)` into fixed worker scratch;
2. renders the new complete graph at the same coordinates;
3. writes their smoothstep mixture into candidate tile 0;
4. renders pure new audio into candidate tile 1; and
5. publishes only when both tiles and the activation deadline are valid.

This costs one extra GPU render per accepted gesture, not per mode or reverb
tap. The callback remains a single prepared-tile reader: bounded,
allocation-free, lock-free, and free of Metal submission. The scalar callback
dezipper is disabled for worker-morphed activations.

An expired activation is never applied at a later device frame: the old epoch
continues while the worker retries. This directly repairs the current protocol
inconsistency where a late descriptor carries a tile tagged at `E` but the
callback begins reading it at `E + Bdev`.

The deterministic oracle checks the complete transition against the same
offline formula. The convex amplitude bound
`|y(k)| <= max(|old(E+k)|, |new(E+k)|)` and exact old/new endpoints are part of
the gate. This is a perceptual parameter transition, not a claim that the
result equals an ideal time-varying low-pass differential equation.

If the worker morph cannot close by the Day-2 decision gate, the
approved fallback is a demo-local dual-filter endpoint morph: update only the
inaudible endpoint's epoch-rate coefficients, then glide an audio-rate mix
between the two complete closed-form filter outputs. Updates are serialized
until the previous finite morph settles. This preserves coefficient hoisting
and avoids putting a sample-rate cutoff expression inside every pole.

### Large-room model

The rejected industrial-cathedral finite-bank design is superseded. The room
is the accepted twelve-group reference-fitted periodic transfer, evaluated
causally and anti-causally around the four Metal hit anchors. It uses the
dedicated `groupedroom` modal-to-signal seam and a versioned immutable native
44.1 kHz asset; it is not encoded as `room_modes` and is never routed through
incumbent modal direction.

The direct midpoint cost is 1,152 source/group interaction equivalents for the
four hits. The native mono forward/reverse prefix payload is 2.59 MiB. ROOM is
the wet VCA, POSITION is the equal-power reverse/forward temporal pan, and the
string lane has no room send. Exact formulas, asset ABI, fallback, work order,
and gates are frozen in `06-room-position-production-handoff.md`.

## Workstream index

| Lane | Mission | Primary owner profile | Must-land artifact |
|---|---|---|---|
| A | Control/scope isolation | C++ runtime + Electron engineer | Preemptible low-priority scopes and bounded high-priority control |
| B | Epoch admission and queue safety | C++/Metal realtime engineer | Zero-fault admitted epochs under the exact scene |
| C | Cutoff waveform morph | DSP/runtime engineer | Offline-oracle-matched finite transition |
| D | Grouped room + POSITION | Modal DSP/Lean/backend engineer | Native-rate asset, immutable binding, exact two-arm lowering, approved WAV |
| E | Scene and UI integration | Demo/audio frontend engineer | One balanced scene using the new contracts |
| F | Reproduction and qualification | Performance/release engineer | Exact-scene harness, telemetry, captures, smoke, and soak |
| G | Staff integration | Integration DRI | Clean release candidate and evidence index |

The staff/integration owner decides contracts and cherry-pick order. Individual
lane agents do not broaden the sprint after discovering a more general design.

## Initial decision log

| ID | Decision | Rejected alternative | Evidence/constraint |
|---|---|---|---|
| DR-01 | Qualify one exact scene on the release Mac. | Reopen universal Metal qualification. | General support is already withheld; the product deadline is the fixed demo. |
| DR-02 | Publish independent immutable scope snapshots and a separate projection artifact. | Preempt/freeze scope work during gestures, or share the audio build lock. | Freeze was visibly discontinuous; snapshot tests prove controls publish while a reader is paused, with coherent old/new frames. |
| DR-03 | Primary cutoff repair is a worker-rendered whole-signal epoch morph. | Retain the scalar callback dezipper or introduce callback-side DSP state. | Complete old/new trajectories give exact endpoints, a convex bound, and no downstream-commutation requirement. |
| DR-04 | **Superseded after listening:** no v1 room profile ships; use the frozen Clouds reference to qualify the replacement architecture. | Select the least objectionable Foundry/Industrial/Tanker preset. | All three sparse stationary candidates were rejected as resonant objects without room-scale diffusion or smear. |
| DR-05 | Start qualification at `Bdev=128`, `Rgpu=512`. | Continue forcing both quanta to 128. | One 128-frame tile gives only 2.9 ms reserve; independent quanta are already supported by the runtime contract. |

## Agent-sized blocks

### Lane A — control/scope isolation

#### A1. Runtime priority seam — 0.5–1 agent-day

- Add a counted control-waiter guard entirely off the callback.
- Let `render_window` terminate cooperatively between coordinates.
- Return an explicit preempted/dropped-frame result.
- Add barrier-driven control-versus-scope, hot-swap, and teardown tests.

Owned surface: `engine/runtime/FlatRuntime.*`, focused runtime tests.

#### A2. Transport separation — 0.5 agent-day

- Split Electron RPC state into control and scope clients.
- Preserve independent request ids, buffering, reconnect, timeout, and engine
  exit rejection.
- Route only `render_window` to the low-priority client.

Owned surface: `playground/main.js`, new transport tests.

#### A3. Scope display profile — completed correction

- Keep the scope request at stride 1 while the projection-only artifact remains
  below the 20 ms RPC gate.
- Calibrate every trace against the exact maximum of its paired modal envelope;
  never normalize each frame by its own peak.
- Select the nearest independently interpolated positive crossing and do not
  draw silent or DC-only windows as locked traces. Center the crossing and use
  one physical time scale for every mode; do not remap spacing by frequency.
- Demodulate each projection sample by that mode's exact paired envelope before
  locking; multiply the resulting unit carrier only by the audible-now
  envelope. Gate analytic cycle shape and equivalent 16-second loop frames.
- Keep rendering through pointer, keyboard-repeat, and sender-busy windows.
- Drive ordinary displays at 60 Hz through `requestAnimationFrame`; cap 120 Hz
  panels at 60 rather than doubling read and JSON work.

Owned surface: `playground/renderer/*` and narrow socket request parsing if
needed. Coordinate the protocol field with A1.

### Lane B — epoch admission and queue safety

#### B1. Fault telemetry contract — 0.5 agent-day

- Expose starvation, tag mismatch, dispatch/activation/ownership failure,
  retargets, stage timestamps, published/acknowledged epochs, first-starvation
  snapshot, and worker CPU/wall time through demo telemetry.
- Add epoch id and device activation frame to a successful Metal `set_param`
  response; source-coordinate `effective_sample_index` alone is not audible
  latency evidence after a clock rebase.
- Include Metal fault counters in `audio_status` so a fail-silent queue cannot
  present as healthy merely because RtAudio returned on time.

Owned surface: `benchmarks/demo_release/` plus read-only telemetry plumbing.

#### B2. Single-transition admission — 0.5–1 agent-day

- Materialize raw cutoff coefficients before reserving its activation epoch.
- Prevent reservation while an earlier activation is awaiting acknowledgement.
- Choose `E` only after admission; remove speculative recomputation loops from
  the ordinary dense-control path.
- Preserve exact companion recomputation when a genuine deadline miss occurs.
- Add rapid A/B/A and multi-parameter scheduling tests.

Owned surface: `engine/metal/MetalRenderWorker.*` and the Metal branch of
`engine/runtime/FlatRuntime.cpp`.

#### B3. Candidate depth and startup tag repair — 0.5–1 agent-day

- Decouple `Bdev=128` from the primary `Rgpu=512` candidate.
- Require at least two exact candidate tiles before publication.
- Reject or expire a descriptor observed after `E`; keep old audio while it is
  retargeted.
- Add the adversarial descriptor-publication-at-`E` race test.
- Repair the reproducible boot-prime tag mismatch rather than resetting its
  counter.
- Exercise independent `Bdev`/`Rgpu` configurations through the harness.
- Fail qualification on the first queue fault with the first-fault snapshot.

Owned surface: worker/queue scheduling and Metal tests. This block lands after
B2 to avoid conflicting worker semantics.

### Lane C — waveform morph

#### C1. Worker morph spike — 0.5 agent-day

- Prototype fixed worker scratch plus old/new whole-graph renders at identical
  coordinates.
- Produce a smoothstep-mixed candidate tile 0 and pure-new candidate tile 1.
- Demonstrate failure leaves the old epoch untouched and callback behavior is
  unchanged.
- Stop and select the demo-local dual-filter fallback if two prepared tiles
  cannot meet the measured admission margin.

#### C2. Chosen morph implementation — 0.5–1 agent-day

- Implement the selected transition and disable the scalar callback dezipper
  for worker-morphed activations.
- Add full-window oracle comparison, convex-bound, repeated-target, reversal,
  silence, high-resonance, and render-failure cases.
- Expose transition counters/timing to the qualification harness.

Owned surface: fixed worker scratch/request data and the smallest queue flag
needed to identify a pre-morphed activation, or demo graph/control files for
the approved fallback. C begins after B2's admission contract freezes.

### Lane D — grouped room + POSITION

Lane D follows the ordered P0–P4 blocks in
`06-room-position-production-handoff.md`:

1. regenerate, prove, audition, and freeze the native 44.1 kHz mono asset;
2. add the Plan-6 immutable asset binding while preserving Plan-5;
3. implement the dedicated `groupedroom` two-arm lowering and backend parity;
4. replace LENGTH with POSITION and route only the four Metal hits; and
5. measure the direct evaluator before considering the documented 5.38 MiB
   fixed-scene mono fallback.

Owned surfaces are the room audition generator and production asset,
`lean/Tropical/Playground/*`, the smallest Patch/IR/emitter additions required
by the new seam, the immutable runtime/Metal binding, focused tests, and then
`playground/scene.js` for integration. Incumbent `reverb`, `room_modes`, and
modal direction semantics remain unchanged.

### Lane E — scene and UI integration

#### E1. Contract integration — 0.5 agent-day

- Consume A, C, and D without adding a new control or screen.
- Keep the 16-second one-scene loop, two scopes, mouse navigation, and current
  chord progression.
- Update labels only when they describe the actual effect.

#### E2. Listening and interaction pass — 0.5 agent-day

- Run slow cutoff, fast sweep, out/back, resonance sweep, ROOM return sweep,
  seek, pause, reverse, and loop-boundary gestures. ROOM is not a
  decay/geometry sweep.
- Produce the release listening WAV and record final control defaults.

Owned surface: `playground/` scene and renderer integration only.

### Lane F — qualification

#### F1. Focused per-commit gate — continuous

- Add `playground/qualification/` as a noninteractive Node harness which
  imports `scene.js`, `LatestValueSender`, and the renderer's selected scope
  profile rather than copying them.
- Reproduce the app's exact load, audio start, prime, scene rebase, listening
  level, tap resolution, scope, and gesture sequence on separate scope/control
  socket connections.
- Wrap the existing preallocated one-buffer DAC capture state machine in a
  qualification-only next-block operation for transition analysis.
- Retain JSONL plus WAV, graph/commit digests, machine/toolchain manifest,
  generated/dispatched inputs, RPC/epoch walls, captures, and diagnostics under
  `benchmarks/demo_release/`.
- Run JS/Lean/C++ focused tests appropriate to each lane.
- Run a 90-second exact-scene hardware smoke after every integrated runtime
  change.
- Reject any result with a nonzero sticky fault even if the audio sounds fine.

#### F2. Release gate — 0.5–1 agent-day

- Run the control/scope stress, output-capture oracle, five cold boots, a
  10-minute adversarial cutoff soak, and a 30-minute normal exact-scene soak on
  the release Mac.
- Run full repository validation from the clean candidate.
- Store machine/configuration metadata and raw results without rewriting prior
  failed evidence.

### Lane G — integration and release

- Create one integration candidate from reviewed lane commits.
- Resolve cross-lane contracts; do not let feature agents merge directly into
  the integration branch.
- Maintain the sprint decision log and evidence index.
- Cut features at the Day-2 gate rather than stretching the sprint.

## Current branch and integration topology

The earlier parallel-lane topology has converged. Production room work begins
on `demo/modal-pocket-scene` from
`demo/room-position-handoff-2026-08-01`, whose required implementation
predecessor is `15e5a39bfce9dd242db59aa6ebc2460987aa2199`:

```text
91ecf0c  original modal-pocket scene
   │
15e5a39  validated control + Metal + morph + qualification checkpoint
   │
handoff  room evidence, frozen decisions, and production contract
   │
P0–P5    native asset → groupedroom → scene → qualification
```

Do not start the room from a fresh `91ecf0c` worktree, the historical Lane D
branch, or `main`; doing so would omit required realtime and qualification
work. Historical lane branches are evidence only unless a specific reviewed
commit is deliberately cherry-picked and the complete checkpoint gates are
rerun. The original four-day schedule below is retained as sprint history;
current production sequencing is P0–P5 in the handoff.

## Current file ownership and collision rules

| Surface | Owner | Coordination rule |
|---|---|---|
| `engine/runtime/FlatRuntime.*`, `engine/metal/*` | implementation checkpoint | Frozen except for the focused immutable-asset binding in P1; preserve admission, morph, and callback contracts. |
| `lean/Tropical/Playground/*`, Patch/IR emitters | P1–P2 | Add only immutable assets and the dedicated `groupedroom` seam; incumbent `reverb` stays unchanged. |
| `playground/scene.js`, renderer controls | P3–P4 | Replace LENGTH with POSITION after the isolated groupedroom gate passes. |
| `benchmarks/demo_release/room_audition/` | P0 | Own native-rate regeneration, oracle, manifest, and production asset generation. |
| `playground/qualification/`, release evidence | P4–P5 | Extend the committed harness; do not create a parallel qualification path. |
| sprint docs and decision log | integration | Record every selected fallback and final asset/commit hash. |

## Four-day schedule

### Day 1 — reproduce, isolate, audition

Parallel wave: A1/A2, B1/B2, D1, and F harness/telemetry.

Exit criteria:

- the exact failure harness is reproducible from one command;
- control preempts or suppresses a scope frame in a focused test;
- epoch admission no longer incurs avoidable reserve-while-pending retries;
- dry/current/room candidate WAVs exist; and
- no lane has expanded into a general framework.

### Day 2 — hard implementation and audible candidate

Parallel wave: A3, B3, C1/C2, D2/D3. E begins integration as lane contracts
land.

End-of-day decision gate:

- choose primary queue morph or demo-local dual-filter fallback;
- choose exact `Bdev`/`Rgpu` candidate for this scene;
- approve or reject the room WAV;
- produce one interactive audible candidate; and
- cut all unfinished P2 work.

If the candidate still starves, tag-mismatches, or clicks, it is not promoted
to release qualification.

### Day 3 — integrate and break it

- E freezes scene wiring/defaults.
- F runs the exact scope-loaded gesture matrix and repeated 90-second smokes.
- F runs the user-approved 10-minute exact-scene soak; no 30-minute run.
- G integrates only commits with focused evidence.
- Fix P0 defects; make no new synthesis or UI feature.

### Day 4 — release candidate

- Perform a clean rebuild and five cold launches.
- Run the final 10-minute exact-scene soak and output-capture gates.
- Run full repository validation.
- Verify clean worktree, release instructions, and evidence index.
- Accept the fixed-demo release or record one explicit blocking failure.

## Release gates

### Interaction latency

On the release Mac, exact scene, the five-trace phase view, and a two-second
log-frequency cutoff sweep at 8–16 ms input cadence:

- scheduled-write latency: p50 no more than 12 ms, p95 no more than 25 ms,
  p99 no more than 35 ms;
- accepted gesture to acknowledged audible activation: p50 no more than 25 ms,
  p95 no more than 50 ms, and maximum no more than 75 ms;
- final authored value scheduled within 35 ms and audible within 75 ms of
  pointer release;
- first and final values always arrive; the bounded reversal case is covered;
- no unbounded per-parameter or cross-parameter backlog; and
- control and scope lanes remain mutually non-blocking.

### Scope behavior

- idle scope RPC p99 no more than 20 ms at the selected point budget;
- average displayed cadence at least 57 fps and p95 frame interval no more
  than 20.83 ms at the 60 Hz profile;
- no blank or paused canvas during a gesture;
- all five traces independently lock to a positive-going zero crossing at the
  center graticule;
- every cycle in a locked trace matches one analytic unit carrier, and
  equivalent frames remain invariant across scene-loop rebases;
- the shared fixed vertical scale exposes each trace's modal attack and decay;
- a newly published control image appears within one displayed scope frame.

### Runtime correctness

Across gesture smoke and the user-approved 10-minute final soak:

- zero Metal starvation;
- zero epoch-tag mismatch, including startup/priming;
- zero Metal dispatch and activation failure;
- zero ownership failure and callback-thread provenance violation;
- zero DAC underrun and overrun;
- zero non-finite or device-clamped samples;
- zero unexpected all-zero captured callback blocks;
- no stale or out-of-order activation; and
- every final control target is acknowledged at the expected audible epoch.

Retargets are expected to be zero on the admitted ordinary-control path. Any
nonzero count requires a diagnosed deadline miss and a repeated passing run;
it is not averaged away.

### Cutoff transition

- the complete captured transition matches the offline chosen-morph oracle at
  `1e-4` or the stricter existing Metal/JIT tolerance where applicable;
- the first sample is old, the final sample is new, and endpoint window
  derivatives are zero by construction;
- every transition sample satisfies the convex old/new amplitude bound within
  float tolerance;
- the maximum adjacent-sample step within 10 ms of activation is no greater
  than `max(0.01 full scale, 4 × surrounding p99 step)`, and 8–20 kHz
  transition energy is no more than 6 dB over the adjacent 50 ms baseline;
- repeated high-resonance sweeps, reversals, and morph completion inside a
  callback remain finite and fault-free; and
- headphone and monitor review finds no click, silent block, or isolated
  transient in wet and dry captures.

### Room

- the native 44.1 kHz mono fit passes the handoff's forward, reverse,
  fractional-address, hash, and listening gates before integration;
- `POSITION=+1/-1` match the causal/classic-reverse references, five positions
  move wet energy monotonically, and the dry Metal source remains causal;
- the six-second pre-tail is unmistakable, seek-to-four-seconds produces
  finite nonzero reverse wet audio, and FLOW agrees with the address oracle;
- the final scene remains below the frozen peak/headroom limit without a
  limiter; and
- fixed complexity assertions prevent a period loop, pole expansion, or static
  asset upload per render tile; and
- cold compile remains no more than 10 seconds and warm load no more than 3
  seconds on the release Mac.

### Repository

- focused JS, C++, Lean, and Metal tests pass;
- `tropicaltest` and the relevant backend-equivalence gates pass;
- full validation passes from the candidate;
- the worktree is clean; and
- docs describe the approximation and fixed-demo qualification honestly.

## Stop-the-line rules

Suspend integration when any of these occurs:

1. a queue fault or silent callback appears;
2. a scope can block a final control value beyond the latency gate;
3. the transition oracle disagrees or a click is audible;
4. a runtime change allocates, locks, submits GPU work, logs, or throws on the
   callback;
5. the room implementation introduces a runtime period loop, a source × room
   pole expansion, or per-tile immutable-asset upload;
6. JIT, Metal, or frozen output changes without an approved semantic reason;
7. a lane changes Plan-5 compatibility, broadens the source language beyond
   the dedicated groupedroom seam, or changes the CF-only boundary; or
8. the candidate passes only after hiding or resetting a sticky diagnostic.

The integration owner either assigns the failure to an existing P0 block or
selects the documented fallback. It is never reclassified as polish.

## Fallback ladder

Fallbacks preserve the release goal while cutting architecture.

1. **Scopes:** reduce point budget only if the stride-1 projection exceeds its
   RPC gate; preserve the immutable control-independent read lane.
2. **Cutoff:** use the serialized demo-local dual-filter morph if the generic
   queue morph cannot satisfy two-bank ownership by Day 2.
3. **Room:** use the 5.38 MiB mono fixed-scene causal/reverse bases only after a
   recorded direct-evaluator reserve failure; retain identical POSITION and
   FLOW semantics and do not restore live decay.
4. **Scene load:** the string wet send is already removed. Do not reduce the
   selected twelve-group Metal witness or weaken runtime gates.
5. **Qualification:** narrow the release to the recorded Mac/configuration;
   never convert a failed soak into a universal-support statement.

There is no fallback that accepts clicks, fail-silent blocks, or a final knob
position that never becomes audible.

## Explicitly deferred

- Ref-counted immutable JIT scope snapshots and a general RPC QoS scheduler.
- An exact theory or implementation of continuously time-varying modal filters.
- Arbitrarily overlapping parameter morphs or preservation of every pointer
  sample.
- General early-reflection authoring, live room dimensions, and multiple room
  presets.
- A universal live-Metal operating envelope or second-machine matrix.
- Drums, additional voices, structural editor work, and new screens.

## Definition of done

The sprint closes only when:

1. the exact interactive scene passes every release gate above;
2. the user approves the room and final listening capture;
3. all sticky runtime and DAC fault counters remain zero for the release soak;
4. cutoff latency and final-value convergence meet the measured bounds with
   scopes active;
5. the chosen transition is named and documented as an approximation;
6. a clean integration commit contains only reviewed sprint work; and
7. remaining durable-engine work is recorded without becoming a prerequisite
   for this demo.

## Evidence package

Lane F and G retain:

- baseline and release candidate SHAs;
- exact machine, OS, audio device, `Bdev`, `Rgpu`, and graph hash;
- raw scope/control latency records;
- pre/post/first-fault runtime telemetry;
- cutoff output captures and oracle comparison;
- dry/incumbent/candidate/final-room WAVs;
- per-commit 90-second smokes, the 10-minute adversarial result, and either the
  30-minute final soak or its explicit user-approved waiver;
- focused and full validation logs; and
- the final decision log, including every selected fallback.

No release claim should require reconstructing evidence from terminal history.
