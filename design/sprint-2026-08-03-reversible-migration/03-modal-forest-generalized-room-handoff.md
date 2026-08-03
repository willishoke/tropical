# Modal gong and generalized live-room architecture handoff

Date: 2026-08-02 (America/Los_Angeles)

Status: **M0/M1 implemented and integration-qualified; M2+ remain follow-on
work and the release candidate is not yet qualified**

This document is a restartable handoff for resolving the two coupled release
gates left by the Reversible migration sprint:

1. keep the served `gong` modal through a downstream room instead of realizing
   it at its pitch-bloom warp; and
2. replace the source-specific/fixed-score grouped-room endpoints with a live,
   source-dependent grouped room for the accepted timed modal scene.

The correction underlying this proposal is important: **a gong does not
mathematically have to expose a signal outlet.** The current served
implementation chooses an early modal-to-signal boundary because its general
clock-warp representation is signal-domain. That is a limitation of today's
intermediate representation, not a property of the instrument.

## 1. Repository and branch context

### 1.1 Current integration state

| Item | Current value |
|---|---|
| Repository | `/Users/willishoke/tropical` |
| Isolated follow-on worktree | `/private/tmp/tropical-modal-forest-grouped-room` |
| Follow-on branch | `sprint/modal-forest-grouped-room` |
| Follow-on head after current-base integration | `5c30142` |
| Modal-forest M1 checkpoint | `86283d57a83393b3856b616438f4ea150b458f5d` |
| Follow-on base / migration head | `2b02a4d3d6ac3e7b58c6012f34f412ac5476c428` |
| Canonical migration branch | `origin/sprint/reversible-migration` |
| Final implementation checkpoint | `4731701f0fc99e0bc55ab59e66aa93f1d5e0803e` |
| Corrected 15-commit replay head | `6a0e0c3765ed1ce263b8f96e8c096863ddd28710` |
| Committed sprint handoff | `3bf81bc47ca52ec1e30b72a5a013e34c1241e4ec` |
| Current demo base | `b8b0576` |
| Demo PR | `#221`: `demo/modal-pocket-scene` -> `main` |
| Migration PR | `#222`: `sprint/reversible-migration` -> `demo/modal-pocket-scene` |
| Follow-on PR | `#223`: `sprint/modal-forest-grouped-room` -> `sprint/reversible-migration` |

The follow-on branch was created and published from the canonical migration
head, implemented M1 in isolation, and then merged the cleaned/current
migration head without rewriting history. The original checkout contains
user-owned work and is not used for sprint edits. Preserve that isolation. The
`/private/tmp` worktree may disappear after a reboot; if it does, recreate it
from the published follow-on branch rather than cleaning or repurposing the
primary checkout.

There is also an older local worktree at
`/private/tmp/tropical-reversible-migration` on local branch
`sprint/reversible-migration`. It is the abandoned first replay and is both
ahead of and behind the corrected remote history. **Do not resume from, merge,
or push that worktree.** The `-corrected` worktree and the canonical remote ref
are authoritative.

The untouched standalone source remains `/Users/willishoke/reversible` at
`dea822ea1062a749ea1d7a76af1e2bd28194dfa1`. Its verified recovery bundle is
`/Users/willishoke/reversible-before-tropical-2026-08-02.bundle`, SHA-256
`b60ee44f9ac18cb9f43be973bff6cd17d530325cc6b93a965fb4fe071bafd5e1`.

The initialized follow-on isolation is equivalent to:

```sh
git -C /Users/willishoke/tropical fetch origin
git -C /Users/willishoke/tropical worktree add \
  -b sprint/modal-forest-grouped-room \
  /private/tmp/tropical-modal-forest-grouped-room \
  origin/sprint/reversible-migration
```

Do not rewrite the migration branch's history. It is the remotely recoverable
integration checkpoint. Push the architectural work in small checkpoints and
merge or retarget only after the semantic and cost gates below pass.

### 1.2 Branch and qualification-artifact hygiene

The branch stack is intentional:

1. demo PR `#221` targets `main`;
2. migration PR `#222` targets the demo branch; and
3. this follow-on targets the migration branch after its first distinct
   checkpoint.

After `#221` merges, retarget `#222` to `main` without rebasing it. The
migration contains a verified replay whose import map, commit identities, and
evidence are made harder to audit by history rewriting.

The demo originally appeared as 157,174 inserted lines because 135,671 lines
(about 86 percent) were qualification evidence under
`benchmarks/demo_release/data`: 96,512 lines of raw JSONL telemetry, 38,296
lines of verbose summary JSON, and 863 manifest lines. That cleanup is now
complete. Raw telemetry and listening/capture WAV files are published in the
immutable GitHub release
`demo-modal-pocket-qualification-2026-08-02`; the 16 MB archive contains 75
entries and has SHA-256
`4a9b0796f46b91df816ee8ce2261c0a5ae2469e1ab261ba402cdc189630cf5e0`.
The release asset was downloaded again and its checksum, gzip integrity, and
entry count were independently reverified before the branch-tip copies were
removed. Compact manifests and aggregate summaries remain in the repository.
PR #221 now reports 25,343 additions, 947 deletions, and 145 changed files.

The normal merge history still retains the historical evidence blobs. That is
an accepted tradeoff: do not rewrite or squash the qualified demo history
merely to expunge those blobs, because doing so would force a migration
transplant and invalidate the current replay provenance. The live tree and PR
diff no longer carry the raw qualification payload.

### 1.3 Existing sprint evidence

- `00-master.md` is the frozen sprint contract.
- `01-implementation-summary.md` records implementation decisions S-01 through
  S-19 and the two open architecture gates.
- `02-evidence-index.md` records the automated results and the still-open
  release qualifications.
- `reversible/IMPORT_PROVENANCE.md`, `reversible/import-map.tsv`, and
  `reversible/scripts/verify-import.sh` preserve the standalone app history.
- `design/seam-atom-contract.md` is the standing correctness contract for
  modal seam algorithms.
- `design/sprint-2026-07-31-demo-release/06-room-position-production-handoff.md`
  defines the accepted grouped carrier and POSITION behavior that this work
  must preserve unless a new listening decision explicitly supersedes it.

### 1.4 Green integrated M1 baseline

The following passed on the integrated M1 head after merging the cleaned demo,
current `main`, and current migration base:

- `make validate`;
- native `tropicaltest`: 125/125;
- Bun: 155 passed, one intentional Metal-capability skip, zero failed, and
  1,145 expectations across 13 files;
- full Lean build and trust audit;
- CTest: 4/4, including the Metal kernel target;
- Swift debug and production builds using the compatible macOS 15.4 SDK; and
- relocated signed-bundle smoke, including an active bundled `groupedroom`,
  expected vocabulary fingerprint `fnv1a64:30da601c40478e7f`, and zero packaged
  fixed wet-score files.

The local Command Line Tools installation does not expose XCTest. Native Swift
tests remain a complete-Xcode/CI gate, not a recorded failure. Qualified-Mac
audio/display lifecycle, ten-minute exact-scene workload, and listening gates
also remain open.

## 2. Current codebase state relevant to this proposal

The migration work already supplies the contracts this follow-on should use:

- the engine owns a versioned, fingerprinted vocabulary;
- Swift renders open node-kind and port IDs instead of copying semantics;
- `load_patch_graph` publishes program/control generation, realized nodes,
  inlet facts, live parameters, disciplines, and taps atomically;
- authored state and last-realized state remain separate;
- live parameter traffic uses one public `set_param` verb;
- graph, control, scope, and telemetry traffic use separate connections;
- v2 documents preserve unknown authored data losslessly;
- requested scope taps and the five-mode frame are generation-coherent; and
- packaging includes the room carrier but excludes the fixed wet-score cache.

Do not bypass these contracts in the architecture follow-on. In particular,
the Swift app must discover a changed `gong` outlet and any context-dependent
live/structural parameter status from the engine response. No Swift semantic
special case is required or permitted.

## 3. Why the served gong is signal today

### 3.1 What remains ordinary modal algebra

The gong's amplitude bloom is already represented strictly in the modal basis:
each blooming partial is a `+a/-a` pair with two decay rates. The default gong
contains two registers:

- a full pitch-glide register; and
- a stiffer half-glide register.

Both begin as ordinary `Array ModalMode` values with one strike anchor. The
instrument remains closed-form and random-access.

### 3.2 The actual early-realization boundary

Pitch bloom applies the analytic clock map

```text
phi(d) = d + B * (1 - exp(-g*d))
```

to each modal register. This is analytic, but after the warp the result is not
a finite fixed pole bank. Today's `gongStrikeNodes` therefore expands each
register as:

```text
modalSource -> warpFx
```

and signal-mixes the two `warpFx` results. `warpFx` is a signal-domain node, so
the modal bank realizes before a downstream patch edge sees the gong. The
served vocabulary then correctly reports the implementation's actual outlet
as signal.

Relevant code:

- `lean/Tropical/EmitArrow/Gong.lean`: `gongStrikeNodes`;
- `lean/Tropical/Playground/Decode.lean`: the served `gong` builder; and
- `lean/Tropical/Playground/Vocabulary.lean`: `outletOf` and the live `beta`
  port.

The accurate statement is therefore:

> The current `gong` is signal because a general `warpFx` is the only served
> representation of its two live pitch-bloomed registers—not because a gong
> must realize before modal composition.

### 3.3 Why `bloomgong` is not the solution by itself

The withheld `bloomgong` proves the intended seam is possible. It stores one
voice bank plus baked `(B, g)` bloom metadata in `ModalBank.bloomed`, folds a
downstream room, and crosses the bloom once at realization via `bloomCompose`.

It is not an acceptable one-line replacement for served `gong`:

- it represents one register, while served `gong` has full- and half-glide
  registers with different bloom scales;
- its `B` and `g` are baked `Float`s, while served `gong.beta` is a live `Sig`
  slot;
- `modalMix` currently refuses bloomed sources rather than preserving them as
  independent branches; and
- its Γ-bridge realizer has a documented conditioning cliff when the composed
  parameter `a` approaches a negative integer at large `|kappa|`. The current
  surface withholds `bloomgong` so that unsafe factor site is unreachable.

Do not expose `bloomgong` or merely add `gong` to the modal outlet table. Either
change would admit a surface topology the lowerer/numerics do not yet serve
totally.

## 4. Proposed modal intermediate: a timed modal forest

### 4.1 Semantic shape

The missing abstraction is a collection of independently timed modal branches,
not a larger pole array with one borrowed anchor.

Conceptually:

```lean
structure ModalBranch where
  bank              : ModalBank
  strikeAnchor      : Sig
  realizationClock  : Clock
  addressNode?      : Option String
  direction?        : Option ModalDir
  modeCount?        : Option Sig

abbrev ModalForest := Array ModalBranch
```

This deliberately resembles today's `LoweredModal`. The smallest semantic
change is to make modal lowering return `ModalForest` rather than exactly one
`LoweredModal`.

The forest is not a new public patch kind. It is the compiler's value for a
modal edge. Each branch retains its own:

- pole bank or specialized bloom representation;
- strike anchor;
- clock/address semantics;
- direction and count metadata; and
- eventual realization route.

### 4.2 Node laws

Define operations by semantics first, then recover sharing as an optimization:

| Node | Forest behavior |
|---|---|
| ordinary modal source | singleton forest |
| served gong | two bloomed branches, same anchor, scale 1.0 and 0.5 |
| `modalmix` | stable forest concatenation |
| modal reverb/filter | map composition over every branch |
| modal gauge | map gauge over every branch |
| grouped room | consume the complete forest in one terminal/batch |
| modal-to-signal edge | realize every branch and sum in stable authored order |

`modalmix` must no longer erase later event timestamps by copying metadata from
its first input. For branches with provably identical anchor/clock/bloom
metadata, an optimizer may union pole arrays after the semantic forest exists.
That optimization must be proved equivalent and must not define correctness.

### 4.3 Served gong lowering

Retain the public kind name `gong`. Change its internal expansion from:

```text
modalSource -> signal warpFx --\
                               signal mix
modalSource -> signal warpFx --/
```

to:

```text
bloomed modalSource(scale=1.0) --\
                                  modalMix/forest
bloomed modalSource(scale=0.5) --/
```

Then:

- `gong -> out` realizes at `out` and must preserve the accepted direct sound;
- `gong -> reverb -> out` composes each register with the room before
  realization; and
- `gong -> modalmix` retains both bloom branches instead of rejecting them.

Once the lowerer and seam are ready, `outletOf "gong"` becomes modal and the
vocabulary fingerprint changes intentionally. Swift should adapt from the
served vocabulary without a source edit.

### 4.4 Live `beta` is a first-class design obligation

Do not silently turn `gong.beta` structural. The current public surface and
tests state that it is a live slot. Unfortunately, the existing deferred bloom
representation stores baked `Float` values because region classification,
series depths, and Γ-bridge constants depend on `B`.

The follow-on must spike and record one of these honest outcomes:

1. **Preferred:** lift the declared `beta` interval into a region-union/
   coefficient-family representation updated through the existing control or
   coefficient epoch, with no graph publication. Retain one public
   `set_param` and use whole-signal transition morphing where coefficient
   images change.
2. **Context-qualified fallback:** keep `beta` live for direct realization but
   have the engine report it structural when the running topology crosses a
   modal room. The authored UI already renders realized live versus structural
   truth. This changes the contextual contract and requires explicit approval.
3. **Rejected default:** bake `beta` everywhere or fake liveness in Swift.

If option 1 cannot be made total and costed, stop for the contextual-contract
decision. Do not hide the tradeoff in the builder.

### 4.5 Bloom seam hardening before surface admission

The current source comments identify a conditioning failure in the fixed-depth
float64 series-M Horner when `a` is close to a negative integer and `|kappa|`
is large. It is not a failure of the underlying Γ identity.

Before a modal served gong reaches arbitrary room poles:

1. extend `SeamRegion` with a named near-negative-integer region;
2. make `classifyBloomPair` total over that region;
3. provide a numerically stable realizer or explicit, measured refusal—not an
   accidental overflow or silent pair drop;
4. land the per-pair exponent/rail scale at the factor site;
5. add boundary and Halton coverage to the seam sweep; and
6. compare the realized result with the independent quadrature oracle under
   the standing seam-atom SNR contract.

The surface may become modal only after this gate passes for the served gong's
declared pole, `beta`, `g`, and downstream-room ranges.

## 5. What a generalized grouped-room profile means

### 5.1 Current profile: room plus one instrument grid

`clouds-current-radii-mono-v1` is source-dependent even though it contains no
wet score. Its immutable prefix tables are laid out source-major for exactly:

- 12 source pole coordinates;
- 12 periodic room groups; and
- forward and reverse POSITION arms.

The approximately 2.7 MiB `.tgrm` file stores precomputed convolution-prefix
data for those exact `(frequency, sigma)` values. Amplitude and phase are free,
but frequency, decay, count, and order are frozen. `validateGroupedRoomModes`
enforces that identity before lowering.

This is an honest live effect for that one coordinate grid. It is not a generic
room for arbitrary authored modal rows.

### 5.2 Proposed profile: acoustic identity plus an admission/capacity contract

A generalized v2 profile should describe the room independently of any one
instrument score:

```text
GroupedRoomProfileV2
  profile/version
  sample rate
  periodic carrier groups
    period
    radius/log-radius
    forward/reverse carrier data
  POSITION law
  accepted source pole domain
  maximum branches and source rows
  evaluator/oracle version
  asset hashes
```

The patch supplies a `ModalForest`:

```text
[(anchor_0, modes_0), (anchor_1, modes_1), ...]
```

Compilation combines the two and produces a graph-specific specialization.
The packaged asset contains room carriers, not source samples and not wet
samples.

“Generalized” does **not** mean arbitrary audio input. The inlet remains modal.
The v2 admission contract may still require:

- finite, build-time-known source capacity;
- constant or declared-range pole coordinates;
- supported damping/frequency bounds;
- a bounded number of timed branches; and
- one qualified native sample rate.

Those are numerical/capacity constraints, not equality with one frozen source
table.

### 5.3 Source-generic and room-parameterized are separate axes

This proposal generalizes the **source** accepted by the room. It does not by
itself make the fitted room radii, periods, geometry, or RT60 arbitrary.

The accepted current room behavior freezes the carrier fit and exposes
POSITION. Room amount is an external live return gain. Any future live decay,
size, or carrier modulation needs its own parameterized carrier model and
oracle. Do not imply those controls merely because the source grid becomes
generic.

### 5.4 The accepted scene's actual scale

The accepted room-send lane currently contains:

- 16 independently timed pizzicato islands;
- 200 source rows;
- 170 distinct `(frequency, sigma)` coordinates; and
- 58 distinct frequencies.

Each downbeat island has 20 rows; each ghost island has 10. Flattening these
into today's `modalmix` would retain only the first anchor and is semantically
wrong.

The v1 evaluator performs a 12-source × 12-group reduction. A literal scene
expansion performs about 200 × 12 source/group pair evaluations per sample,
roughly 16.7 times the pair count before backend effects.

The current source-specific prefix payload scales approximately linearly with
unique source coordinates. Expanding its 2.7 MiB/12-coordinate layout for 170
coordinates would be roughly 38.5 MiB. That remains source-dependent room
coefficient data—not a wet render—but it must pass reserve, load-time, and
runtime cost gates rather than being assumed acceptable.

## 6. Candidate implementations to measure

### 6.1 Exact graph-specialized prefix reference

At graph compilation, derive the existing forward/reverse prefix arrays for
the forest's unique source coordinates. Cache them by:

```text
hash(room-profile, sample-rate, ordered source coordinates,
     evaluator version)
```

This is the closest generalization of the proven v1 formulas and should be the
first executable reference.

Properties:

- source edits regenerate coefficients and change wet output;
- amplitudes, phases, and anchors remain authored graph data;
- no wet samples are stored;
- the existing infinite analytic formula remains the oracle; and
- large generated immutable arrays may fail the release cost envelope.

Generated prefixes must be named and reported as graph specialization, not
mislabelled as the room-only package asset.

### 6.2 Per-island terminals sharing one room asset

Lower each timed island through its own grouped-room evaluator and signal-sum
the results afterward. All evaluators share the same mapped room-carrier asset.

This is a useful intermediate comparison because it preserves anchors with
minimal batch machinery. Measure:

- duplicated instruction/kernel topology;
- immutable-buffer sharing;
- dispatch and activation overhead;
- JIT/Metal compile time; and
- exact sample equivalence with the graph-specialized reference.

It is not automatically the production choice merely because it is easiest.

### 6.3 One batched timed-modal terminal — recommended production target

Flatten the forest into stable arrays of:

- per-branch anchor and row range;
- per-row frequency, sigma, complex residue, and optional bloom metadata; and
- shared room group/carrier descriptors.

Evaluate the event/source/group dimensions in one named constructor/kernel,
sharing carrier loads and POSITION computation. Preserve authored ordering and
each branch's relative coordinate `u = clock - anchor[branch]`.

This is the preferred production architecture because it expresses the actual
semantics while creating one place to optimize carrier reuse. It requires:

- an internal `ModalForest`/batch representation;
- Plan/array capacity and immutable-data ownership decisions;
- JIT and Metal implementations;
- a separate independent oracle; and
- exact cost measurements at `Bdev=128`, `Rgpu=512`.

### 6.4 Existing 32-mode `reverb` as control

The ordinary modal `reverb` already accepts arbitrary modal rows and is the
semantic/listening control. It is not the same acoustic transfer as the
accepted grouped room. Include it in cost and listening comparisons; do not
quietly substitute it and claim the grouped sound survived.

## 7. Recommended implementation sequence

Each checkpoint should be independently buildable, tested, committed, and
pushed.

### M0 — Re-establish baseline and evidence

- create a fresh isolated follow-on worktree;
- verify branch head and import proof;
- run `make validate` and the compatible-SDK Swift build;
- record machine/toolchain differences from the existing manifest; and
- retain `groupedroomcache` only as an oracle/rollback fixture.

Exit: no baseline regression and no user-owned checkout changes.

### M1 — Introduce `ModalForest` without a surface change

- change modal lowering to return a stable forest;
- make existing single sources singleton forests;
- map reverb/filter/gauge over forests;
- realize forest branches in stable order;
- preserve existing graphs and taps; and
- add same-anchor and different-anchor semantic fixtures.

Exit: existing direct modal graphs remain equivalent; two different anchors no
longer collapse to the first input's metadata.

### M2 — Generalize modal mix and timed-island truth

- make `modalmix` concatenate branches;
- add optional same-metadata union only behind an equivalence test;
- serialize 16 timed source islands through one modal path; and
- prove all onset/anchor coordinates survive forward, hold, reverse, and seek.

Exit: the accepted scene's source forest is representable without crossing to
signal or erasing timestamps.

### M3 — Harden the bloom seam

- add the negative-integer conditioning region;
- implement/refuse it explicitly and measure coverage;
- land factor-site rail scaling;
- expand the seam sweep and independent oracle fixtures; and
- decide the live-`beta` implementation using measured cost and totality.

Exit: the complete served gong parameter/range box has an explicit and tested
seam result. If live `beta` cannot remain live through composition, stop for
approval of the context-qualified fallback.

### M4 — Make served `gong` modal

- rebuild the two registers as bloomed modal branches;
- change the served outlet to modal;
- update vocabulary fixtures and expected fingerprint;
- prove direct `gong -> out` parity;
- land literal serialized `gong -> reverb -> out` and room-chain regressions;
- prove source removal silence and structural source mutation; and
- verify Swift requires no semantic case change.

Exit: the public gong composes through modal rooms and realizes only at a
signal consumer.

### M5 — Build generalized grouped-room reference

- define a source-independent carrier/profile v2 manifest;
- extend the existing generator to emit/verify room-only data;
- build graph-specialized prefixes for arbitrary admitted modal rows;
- retain v1 12-coordinate bit/sample parity as a regression; and
- test source edits, zero/removal, fractional coordinates, reverse, and seek.

Exit: one arbitrary timed modal island drives the accepted room with no wet
score data.

### M6 — Compare per-island and batched evaluators

- implement the per-island shared-asset comparison;
- implement the named batched timed-modal constructor;
- compare both against the same independent oracle;
- measure JIT/Metal compile time, immutable bytes, kernel size, dispatch,
  buffers, and fault counters; and
- select the production path from recorded evidence.

Exit: one path meets correctness and the exact-scene release buffer/capacity
envelope. If neither does, the release remains blocked.

### M7 — Re-author the release patch and native performance view

- replace `groupedroomcache` with a modal-input grouped room;
- preserve one document/model/engine generation across Patch and Performance
  views;
- bind the accepted five-mode scope to explicit realized taps;
- expose only realized live macros;
- package only room carriers and graph-authored scene data; and
- run document round-trip and relocated-bundle smoke.

Exit: editing/removing a source changes/nulls the wet output in the packaged
app; there is no parallel hard-coded scene.

### M8 — Qualification and listening

- JIT/Metal/oracle differential over source and POSITION ranges;
- five cold boots;
- muted scope/control cadence and latency capture;
- ten-minute exact-scene adversarial workload;
- lifecycle/orphan matrix;
- dry, wet, full, source-mutation, room sweep, and forward/reverse captures;
- peaks, RMS, asset/patch hashes, buffers, and all fault counters; and
- explicit user listening decision.

Exit: every P0 row passes or the candidate remains explicitly blocked.

## 8. Required validation gates

### 8.1 Gong and modal forest

- served vocabulary reports `gong` as modal;
- direct gong sound is preserved within the declared parity/error contract;
- literal serialized `gong -> reverb -> out` compiles and is audible;
- a two-register gong crosses each downstream room once;
- distinct forest anchors remain distinct;
- direct, hold, reverse, and seek evaluate the same coordinates;
- `beta` behavior matches the chosen explicit contract; and
- no unsafe seam region reaches an unguarded realizer.

### 8.2 Generalized grouped room

- room package asset contains carrier/room data only;
- arbitrary admitted source coordinates compile without equality to the v1
  twelve-coordinate table;
- source removal/zero nulls wet output below `1e-12`;
- source amplitude, phase, coordinate, and anchor edits change wet output;
- all 16 accepted event anchors survive composition;
- POSITION endpoints and fractional coordinates match the existing infinite
  analytic oracle;
- repeated identical generation/coordinate requests are bit-identical;
- JIT and Metal meet recorded tolerance; and
- exact-scene cost/fault gates pass at release buffers.

### 8.3 Product and package

- no production graph references `groupedroomcache`;
- no packaged file matches the fixed wet-score cache role;
- realized graph truth identifies the live room input and controls;
- unknown v2 document data remains lossless;
- relocated bundle uses only embedded engine and room assets; and
- Patch and Performance views share one authored and realized graph.

## 9. Stop-the-line rules

Stop and record the blocker rather than weakening the claim if any of these
occurs:

1. `gong` is marked modal only in vocabulary while its builder still realizes
   through signal `warpFx`;
2. `bloomgong` is un-withheld without closing the factor-site conditioning
   region;
3. live `beta` is silently downgraded or a client guesses its discipline;
4. modal mixing merges branches with different anchors, clocks, or blooms;
5. a “generalized” profile still validates equality with one source table;
6. a room asset or generated specialization contains wet scene samples;
7. source mutation does not change wet output;
8. a cache path is retained as a production fallback;
9. carrier sharing changes POSITION, reverse, fractional-coordinate, or
   random-access semantics;
10. JIT, Metal, and the independent oracle diverge beyond the recorded model;
11. exact-scene cost is inferred from a smaller carrier demo; or
12. listening approval for the old fixed-score cache is reused for the new
    live realization.

## 10. Primary files and ownership hotspots

| Surface | Primary files |
|---|---|
| modal forest and seam placement | `lean/Tropical/EmitArrow/Patch.lean` |
| gong construction | `lean/Tropical/EmitArrow/Gong.lean`, `lean/Tropical/Playground/Decode.lean` |
| vocabulary/outlet/parameter truth | `lean/Tropical/Playground/Vocabulary.lean` |
| bloom classifier and realizer | `lean/Tropical/EmitArrow/Modal/Bloom.lean`, `lean/Tropical/EmitArrow/Modal/Live.lean` |
| standing seam law | `design/seam-atom-contract.md`, `lean/Tropical/Tropicaltest/SeamSweep.lean` |
| grouped room evaluator and asset binding | `lean/Tropical/EmitArrow/Modal/GroupedRoom.lean`, `lean/Tropical/Playground/Compile.lean` |
| room generators/oracles | `benchmarks/demo_release/room_audition/generate_grouped_room_asset.py`, `test_native_grouped_room.py`, runtime probes |
| accepted timed source scene | `playground/scene.js`, `playground/scene.test.js` |
| engine/vocabulary integration gates | `lean/Tropical/Tropicaltest/Vocabulary.lean`, `LiveRoom.lean`, `Modal.lean` |
| native product truth | `reversible/Sources/Reversible/PatchModel.swift`, `TruthInspectorView.swift`, document fixtures/tests |
| packaging | `reversible/scripts/bundle-app`, `reversible/scripts/smoke-bundle.mjs` |

Coordinate edits to `Patch.lean`, `Vocabulary.lean`, and the grouped-room asset
ABI carefully. These are shared semantic/compilation surfaces, not lane-local
implementation details.

## 11. Decision record proposed for adoption

| ID | Proposed decision | Status |
|---|---|---|
| MF-01 | Served `gong` should expose modal, not signal | recommended; implementation gated on seam/live-beta work |
| MF-02 | Use a timed `ModalForest` as the modal edge value | recommended |
| MF-03 | Preserve branches/anchors semantically; merge only as a proved optimization | recommended |
| MF-04 | Do not serve `bloomgong` as a parallel product kind | recommended |
| MF-05 | Preserve live `gong.beta`; stop for approval if only contextual liveness is feasible | recommended with explicit technical spike |
| GR-01 | Generalized profile identity is room/carrier-only plus admission/capacity metadata | recommended |
| GR-02 | Use graph-specialized prefixes as the exact reference | recommended |
| GR-03 | Target one batched timed-modal grouped-room evaluator for production | recommended, contingent on cost |
| GR-04 | Keep per-island shared-asset and ordinary `reverb` paths as measured comparisons | recommended |
| GR-05 | `groupedroomcache` remains oracle/rollback evidence, never a production fallback | frozen by sprint master |

## 12. Handoff completion criterion

This architecture follow-on is complete only when the packaged native patch
contains a real modal edge into the selected room, the served gong can traverse
that edge without early realization, all accepted event anchors survive, source
edits change the wet result, no wet-score asset is shipped, exact-scene cost and
fault gates pass, and the user accepts the new live-room listening captures.

Until then, the existing branch remains an integration candidate rather than a
release candidate.
