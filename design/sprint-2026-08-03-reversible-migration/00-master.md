# Reversible into Tropical — migration sprint master handoff

- **Sprint:** 2026-08-03 through 2026-08-14 (ten engineering days)
- **Integration baseline:** `07a6b2517f8d24c8822d67e0337c0fd99d016bd8`
  (`demo/modal-pocket-scene`, pushed as `origin/demo/modal-pocket-scene`)
- **Standalone source:** `/Users/willishoke/reversible`
- **Standalone source head:**
  `dea822ea1062a749ea1d7a76af1e2bd28194dfa1`
- **Standalone source root:**
  `e7693fa412dfd0082341763a62682ce64a46f8cf`
- **Standalone history:** 15 linear commits, clean `main`, no tags, no remotes
- **Target subtree:** `reversible/`
- **Target branch:** `sprint/reversible-migration`
- **Sprint type:** history-preserving repository migration, native-frontend
  integration, and live-instrument recovery
- **Status:** planned; owners are assigned at kickoff

## Executive mandate

Migrate the standalone Reversible macOS app into Tropical without erasing its
history, then make it the honest native surface of Tropical's self-contained
closed-form instrument.

The product is **not** a generic processor for arbitrary live or external
audio. Its sources, clocks, modal banks, effects, room, transport, and
inspection views live inside one authored, acyclic, random-access graph. That
limitation is intentional and is not a missing feature to work around.

The failure to correct is different: the current release scene presents a
pre-rendered wet score as though it were a room effect. The shipping
`groupedroomcache` has no source inlet; changing the source graph cannot change
its wet output. That is not a live effect inside the self-contained
environment. It is a fixed playback asset. It may remain as historical
evidence and a differential oracle, but it cannot be the room in the migration
candidate.

Reversible is valuable here because its patch canvas forces the implementation
to expose the graph that is actually running. A visible `source → room → out`
edge must correspond to a compiled source-to-room composition. A knob labelled
live must be backed by a live plan parameter. A monitor must be a read-only
consumer. The surface may not imply a capability that the engine has replaced
with an offline imitation.

The sprint is successful only when all of the following are true:

1. all 15 standalone commits exist on a pushed Tropical branch as 15 auditable
   replayed commits under `reversible/`;
2. the native app builds, tests, launches, owns the bundled Tropical engine,
   and saves/loads patches from the monorepo;
3. the engine, not Swift, is the semantic source of truth for node kinds,
   ports, connection colors, parameter ranges, and write disciplines;
4. the authored demo patch contains a real modal source-to-room edge and its
   wet result depends on the live source graph;
5. transport reversal and random access remain coherent across source,
   envelope, room, and scope;
6. the single overlaid modal scope remains phase-locked and amplitude-honest at
   display cadence while controls are scrubbed; and
7. the ten-minute release workload completes on the qualified Mac with no
   runtime, Metal, callback, DAC, ownership, non-finite, or document-integrity
   fault.

If the app is integrated but the live room fails the listening gate, the
sprint result is **integrated but not demoable**. It must not be called a
release candidate and must not quietly fall back to the fixed wet cache.

## Current-state evidence

This plan is based on repository inspection on 2026-08-02, not recollection.

The authoritative repository inputs are
[`design/architecture.md`](../architecture.md),
[`design/cf-only.md`](../cf-only.md),
[`design/host-param-dispatch.md`](../host-param-dispatch.md), the current
[`demo master`](../sprint-2026-07-31-demo-release/00-master.md), its
[`room production handoff`](../sprint-2026-07-31-demo-release/06-room-position-production-handoff.md),
and the qualified
[`scene`](../../playground/scene.js) and
[`scope frontend`](../../playground/renderer/app.js).

### Standalone Reversible history

| Order | Source commit | What it introduced |
|---:|---|---|
| 1 | `e7693fa` | SwiftPM/SwiftUI scaffold, theme, engine-path contract |
| 2 | `2d2f96f` | JSON-RPC engine actor over `frontend --serve` |
| 3 | `8e2c90c` | node vocabulary, patch graph, connection discipline |
| 4 | `8bb24fa` | patch canvas, jacks, wires, drag-to-patch |
| 5 | `ab0808e` | knobs driving the live kernel |
| 6 | `db3746b` | toolbar, transport, and master-clock scrub |
| 7 | `51208e2` | audio load/xrun telemetry and engine-state adoption |
| 8 | `c653234` | normal macOS app activation |
| 9 | `ae47a9d` | knob-pointer rotation correction |
| 10 | `dc1bcc4` | infinite snapped grid plane |
| 11 | `d12d009` | connection color identity and native widgets |
| 12 | `3787192` | topological-rank layout |
| 13 | `ab2a4fe` | four-channel scope over the data plane |
| 14 | `50e7f75` | child-engine crash reaping and orphan sweep |
| 15 | `dea822e` | versioned, human-readable patch save/load |

The history is linear. The working tree is clean. `git remote -v` and
`git tag --list` are empty. Until the import branch is pushed, the only copy of
this commit history is local.

### What the standalone app actually is

Reversible is a native control surface for Tropical, not a second DSP engine.
It spawns Tropical's `frontend`, submits a patch graph, drives plan parameters,
reads telemetry, and renders random-access scope windows. Its product model is
already the right one: topology changes relower and hot-swap; continuous values
write live parameter slots; the global clock scrubs the whole closed-form
graph.

The repository migration therefore imports valuable product work:

- a native infinite patch canvas;
- graph editing with an acyclicity check;
- modal/signal/control connection discipline;
- live knobs and transport;
- engine lifecycle ownership;
- save/load documents;
- telemetry and scope UI.

It does **not** import another compiler, runtime, room algorithm, or audio
backend.

### Contract drift that must be repaired after import

The standalone source was correct against a July engine snapshot. It must be
imported unchanged first for provenance, then modernized in subsequent
commits.

| Area | Standalone assumption | Current Tropical contract | Required action |
|---|---|---|---|
| Node vocabulary | hard-coded Swift `enum` with 14 cases (13 serialized engine kinds plus the scope monitor) | `get_vocabulary` serves 22 kinds from one Lean port-spec table; `bloomgong` is explicitly withheld | replace the semantic enum/spec table with decoded engine data |
| Parameter writes | client chooses `set_param`, `set_param_glide`, or `set_param_freq` | one `set_param`; the loaded plan dispatches raw/glide/anchor/velocity | delete client-side verb selection |
| Graph result | success is treated mostly as `{ok:true}` | `load_patch_graph` returns active/excluded nodes, wired/normalled inlets, live params, disciplines, and taps | render realized state and refuse to guess |
| Scope selection | boolean `taps` and a second `list_scope_taps` request | graph may request an explicit tap-name array; the load report already returns published taps | request exactly the visible taps and adopt the report atomically |
| Scope cadence | one MainActor loop sleeps 33 ms and serially uses one engine actor/socket | audio, control, and low-priority immutable scope snapshots are independent lanes | split connections/actors and render from a display link |
| Scope trigger | generic four-channel level trigger | accepted scene uses one five-mode overlay, independent phase locks, centered positive zero crossing, fixed volts/div, and envelope-aware locking | port the qualified scene scope and its tests |
| Patch documents | `NodeKind` enum; unknown/stale fields are silently dropped toward a “nearest valid” patch | engine rejects broken documents and reports legal exclusions as facts | migrate explicitly; never silently erase topology |
| Engine location | environment override, otherwise `~/tropical/.../frontend` | app and engine will share one repository/bundle | bundle-relative resolution first; no home-checkout dependency |
| Room in demo | fixed `groupedroomcache` wet score, no input edge | Tropical has a genuine modal `reverb`; grouped-room work has source-specific and cache variants | ship a real source-dependent room path; cache is oracle only |

### Tropical capabilities to use, not duplicate

The integration baseline already provides:

- `get_vocabulary`, generated from the authoritative `portSpecs` table;
- a unified public `set_param` dispatched from the loaded plan's
  `param_disciplines` table;
- explicit signal/modal/control port colors and server-side edge validation;
- a realized-state report from `load_patch_graph`;
- separate audio and scope artifacts published in one generation;
- ref-counted immutable program/control snapshots for `render_window`;
- low-priority scope rendering that does not own audio/control storage;
- exact epoch, Metal, morph, starvation, ownership, and latency telemetry;
- a real modal `reverb` node whose `rt60`, `dir`, `sway`, and `rate` are live;
- the modal seam algebra for composing a source bank with a room bank before
  final realization; and
- the qualified 60 Hz, five-mode, independently phase-locked scene scope in
  the existing Electron demo.

The native app should consume those facilities. It must not grow a fourth copy
of their schemas or algorithms.

## Product and semantic contract

These are sprint invariants, not implementation suggestions.

### 1. One self-contained instrument

The supported graph begins with Tropical's analytic, modal, and control
sources. It does not accept a microphone, DAW insert, arbitrary audio stream,
or stateful feedback loop. Such work is out of scope and is not needed to call
the room “live.”

### 2. Live means graph-dependent

For an edge `S → R`, the wet result must be computed from the current output of
`S` and the current parameters of `R`. A structural edit to `S` may relower and
hot-swap; a live parameter edit to `R` must use `set_param`; neither may require
an offline wet render to be substituted into the graph.

An immutable asset is permitted when it describes **only the room**—for
example, room pole/carrier data. An asset containing the wet response of the
authored score is not a room asset and is ineligible for the release graph.

### 3. The graph shown is the graph run

- Every visible engine node serializes to `load_patch_graph`.
- Every visible engine edge appears in that request.
- Every parameter presented as live appears in the realized param report.
- Excluded nodes and normalled inputs are shown as state, not hidden.
- Surface-only monitors are labelled monitors and never presented as audio
  processors.
- The app does not fabricate a missing kind or coerce an invalid connection.

### 4. Engine-owned semantics

The engine owns kind identifiers, port names, port colors, arity, defaults,
ranges, scales, units, owner-port relationships, and write disciplines. Swift
owns presentation: layout, accents, typography, menus, patch positions, and
monitor view state.

The client may cache a vocabulary response for the running engine generation.
It may not hand-maintain a semantic fallback. Failure to obtain or decode the
vocabulary is a visible boot failure.

### 5. Structural and live edits remain distinct

Add/remove/rewire, room-profile changes, authored modal rows, and declared
ranges are topology-grade edits. They compile off the audio thread and
hot-swap. Values named in the realized parameter report are live writes. The
UI labels both classes and never emits a compile loop from a continuous drag.

### 6. Time is global and addressable

The transport maps the device sample index to one scene coordinate. Forward,
hold, reverse, seek, and varispeed apply coherently to sources, envelopes,
room tails, and inspection. Reverse is receding address, not a separately
rendered audio file.

### 7. Scopes are read-only observers

Scope work may pin an immutable program/control image and use scope-owned
workspace. It may not mutate the graph, control image, audio storage, or clock;
hold the control connection; or take a lock needed by an audio or control
operation. Enabling scopes must not change rendered audio samples.

### 8. Honest failure status

A compile error, excluded graph, missing parameter, unknown document kind,
preempted scope read, Metal starvation, DAC xrun, or failed listening gate is
reported in its own category. Silence is not accepted as a generic success
state.

## Sprint scope

### P0 — required to close

- Preserve and remotely back all 15 Reversible commits in Tropical.
- Build and test the imported app under `reversible/` on macOS 14 or newer.
- Replace hard-coded semantic vocabulary and parameter dispatch in Swift.
- Give control, scope, telemetry, and compiler traffic explicit ownership and
  non-blocking interaction rules.
- Migrate v1 `.rvpatch` files without silent node, edge, or value loss.
- Boot the native app into an editable version of the authored scene.
- Replace the scene's fixed wet cache with a source-dependent live room path.
- Port the single five-mode phase scope at display cadence.
- Package an app that finds the engine and production assets without requiring
  `~/tropical` or the launcher's current working directory.
- Pass the focused, full, muted interaction, ten-minute, save/load, lifecycle,
  and listening gates in this handoff.

### P1 — retain unless it threatens P0

- One-action switch between the curated performance workspace and the full
  patch canvas, with the same underlying graph.
- Visual live/structural badges on controls and realized/excluded badges on
  nodes.
- A migration report panel for old documents.
- A reproducible `.app` bundle with embedded engine, assets, document UTI, and
  ad-hoc signing for the qualified Mac.
- Golden fixtures for every currently served engine kind.
- An explicit performance budget shown in the app from runtime telemetry.

### P2 — cut first

- Developer-ID signing and notarization.
- Automatic update machinery.
- iOS, Windows, or Linux UI ports.
- Multiple simultaneous patch documents/windows.
- A general-purpose four-channel analyzer in addition to the release overlay.
- Editable room geometry, multiple room profiles, or an impulse-response
  browser.
- Removal of the Electron playground.

### Non-goals

- No arbitrary external/live audio input.
- No plugin formats, DAW hosting, or audio-unit bridge.
- No stateful delay feedback or cyclic graph support.
- No new surface language.
- No `bloomgong` exposure before its existing admission blocker closes.
- No claim that every Tropical graph meets the release Metal envelope.
- No 30-minute soak. The release workload is ten minutes, matching the
  product decision already made for this demo.

## Target repository and runtime architecture

### Repository layout

The import is prefixed as a self-contained Swift package:

```text
tropical/
├── engine/                    native runtime, DAC, socket, Metal
├── lean/                      compiler, vocabulary, patch lowering
├── playground/                Electron reference and qualification oracle
├── reversible/
│   ├── Package.swift
│   ├── README.md
│   ├── Sources/Reversible/
│   ├── Tests/ReversibleTests/        added after history replay
│   ├── Fixtures/                     patch/vocabulary/realized-state goldens
│   └── scripts/                      dev launch and .app packaging
└── design/sprint-2026-08-03-reversible-migration/
    └── 00-master.md
```

`reversible/` is deliberate: it preserves the product name and makes every
standalone path prefix mechanically. The initial 15 replayed commits contain
only that subtree. Monorepo integration edits follow in new commits.

### Process and concurrency ownership

```text
SwiftUI / MainActor
  │ edits PatchDocument + presentation only
  ├──────────────┬───────────────────────┬────────────────────────┐
  ▼              ▼                       ▼                        ▼
GraphCompiler  ParamSender          ScopeSampler             TelemetryPoller
actor          actor                actor + CVDisplayLink     actor
  │ control RPC  │ control RPC         │ scope RPC               │ data RPC
  └──────────────┴───────────────────────┼────────────────────────┘
                                         ▼
                              one owned frontend process
                                         │
              ┌──────────────────────────┼─────────────────────────┐
              ▼                          ▼                         ▼
       Lean control queue       C++ param/data plane       immutable scope image
              │                          │                         │
              └──────────── publish one generation ───────────────┘
                                         │
                                         ▼
                                audio callback / DAC
```

The supervisor owns the child process and socket namespace. Each RPC lane has
its own connection and actor. A long compile can delay later topology work, but
it cannot serialize a scope response in front of a knob write. The MainActor
does not wait in a display callback and does not perform JSON framing.

The scope sampler requests one frame at a time, drops superseded results, and
publishes only the newest complete immutable frame. `CVDisplayLink` draws at
the display cadence, capped at 60 Hz on faster panels. No queue of stale scope
frames is allowed.

### Boot and package resolution

Engine resolution after migration is:

1. `Reversible.app/Contents/Resources/Tropical/frontend` in a packaged app;
2. `TROPICAL_ENGINE_BIN` for an explicit developer/test override; and
3. the monorepo build path supplied by `reversible/scripts/run-dev`, resolved
   from the script's own repository location.

There is no implicit home-directory checkout fallback. The engine must launch
successfully when the app's current working directory is `/private/tmp`.
Production room assets are embedded beside the engine or compiled into it;
they are not looked up through the source checkout.

### App model

The app has one `EngineVocabulary` loaded at boot, one `PatchDocument`, and one
`RealizedPatch` returned by the last successful compile.

- `EngineVocabulary` answers what can be authored.
- `PatchDocument` records what the user authored, including visual state.
- `RealizedPatch` answers what actually compiled and is running.

The UI never substitutes one for another. A node can be authored but excluded;
that status remains visible. A document can refer to a kind unavailable in the
current vocabulary; it opens in a blocked migration state without discarding
the node.

The release scene is one checked-in `.rvpatch` plus any structural modal data
it references. “Performance” and “Patch” are two views of this same document,
not separate implementations.

## History-preserving import runbook

This lane runs before any Swift modernization. Do not copy files, squash the
repository, or merge an unrelated root as one opaque commit.

### H0. Freeze and back up the only source history

Record the following in `reversible/IMPORT_PROVENANCE.md`:

```text
source_path=/Users/willishoke/reversible
source_head=dea822ea1062a749ea1d7a76af1e2bd28194dfa1
source_root=e7693fa412dfd0082341763a62682ce64a46f8cf
source_commit_count=15
source_branch=main
source_remotes=none
source_tags=none
target_base=07a6b2517f8d24c8822d67e0337c0fd99d016bd8
target_prefix=reversible/
```

Before rewriting anything, create and verify a full bundle at the explicit
path `/Users/willishoke/reversible-before-tropical-2026-08-02.bundle`:

```bash
git -C /Users/willishoke/reversible bundle create \
  /Users/willishoke/reversible-before-tropical-2026-08-02.bundle --all
git bundle verify \
  /Users/willishoke/reversible-before-tropical-2026-08-02.bundle
shasum -a 256 \
  /Users/willishoke/reversible-before-tropical-2026-08-02.bundle
```

The source repository remains untouched and is not deleted after migration.

### H1. Replay the commits under the subtree

Use `format-patch` plus `git am --directory=reversible`. This is the
rebase-equivalent operation for an unrelated root that also needs a path
prefix. It preserves author identity, author date, commit message, ordering,
file modes, and blobs while assigning the new Tropical parent and therefore
new commit IDs.

```bash
REVERSIBLE_IMPORT_TMP=$(mktemp -d /private/tmp/reversible-import.XXXXXX)
mkdir -p "$REVERSIBLE_IMPORT_TMP/patches"
git -C /Users/willishoke/reversible format-patch \
  --root --binary --full-index \
  --output-directory "$REVERSIBLE_IMPORT_TMP/patches"

git -C /Users/willishoke/tropical worktree add \
  -b sprint/reversible-migration \
  /private/tmp/tropical-reversible-migration <handoff-commit>
git -C /private/tmp/tropical-reversible-migration am \
  --directory=reversible "$REVERSIBLE_IMPORT_TMP"/patches/*.patch
```

Do not use `--3way` for the first attempt: a collision is evidence that the
prefix or base is wrong, not something to resolve invisibly. Do not amend the
15 replayed commits.

This command path was dry-run during planning against a temporary local clone:
15 source commits produced 15 prefixed commits, and standalone head tree
`155eb87501c4aa0a530956db4c5bc78ea5fd6893` exactly matched the imported
`HEAD:reversible` tree. That smoke validates the mechanism; it does not replace
the final per-commit metadata/tree proof in H2.

### H2. Prove the replay

Create `reversible/import-map.tsv`, one source SHA and one Tropical SHA per
line in chronological order. A committed verification script must fail unless:

1. the source and target lists both contain exactly 15 commits;
2. each target commit has exactly one predecessor after the integration base;
3. author name, author email, author timestamp, subject, and body match;
4. for every pair, the recursive tree at `<target>:reversible` has the same
   paths, modes, object types, and blob IDs as the full source tree;
5. the final target subtree equals standalone `dea822e`; and
6. no replayed commit changes a path outside `reversible/`.

Committer identity/date and commit IDs are expected to differ because the
parent and path prefix differ. That difference is recorded, not treated as
loss.

### H3. Back the history remotely immediately

After H2 passes:

```bash
git push -u origin sprint/reversible-migration
```

Verify the remote ref resolves to the replayed head before beginning
modernization. The local bundle remains until sprint close.

### History stop conditions

Stop the import if:

- the source working tree is no longer clean;
- source head or count differs from the frozen values;
- a patch conflicts after the directory prefix;
- a commit becomes empty;
- metadata or any subtree tree differs;
- a replayed commit touches another Tropical path; or
- the remote branch cannot be read back after push.

Do not “fix” a failed replay by squashing, skipping, or copying the final tree.

## Workstreams

### Lane A — provenance, import, and monorepo build

**Mission:** land the standalone history losslessly, then make `reversible/` a
first-class macOS subtree without rewriting the imported commits.

**Primary files:** `reversible/**`, `Makefile`, macOS CI configuration,
`reversible/IMPORT_PROVENANCE.md`, `reversible/import-map.tsv`.

#### A1. History safety and replay — 0.5 day

- Execute H0–H3 exactly.
- Commit the verifier, mapping, bundle hash, source facts, and command log in
  the first post-import commit.
- Push the branch before changing Swift source.

**Exit:** all six replay proofs pass and the remote contains the branch.

#### A2. Package/test registration — 0.5–1 day

- Keep the imported executable target.
- Add `ReversibleTests` and fixture resources in a new commit.
- Add `make reversible-build` and `make reversible-test` on macOS.
- Add a macOS CI job or an explicit release-machine gate; Linux validation
  must not pretend to compile SwiftUI.
- Build from a clean worktree with no pre-existing `.build` directory.

**Exit:** `swift build --package-path reversible` and
`swift test --package-path reversible` are reproducible.

#### A3. Bundle and development launcher — 0.5–1 day

- Add a deterministic development launcher that points to the monorepo engine.
- Add a bundle script producing `build/Reversible.app` with `Info.plist`, the
  executable, engine, assets, icon placeholder, and `.rvpatch` document type.
- Launch the bundle from `/private/tmp` and from Finder.
- Record signing state honestly; ad-hoc signing is enough for the sprint Mac.

**Exit:** no engine lookup depends on `/Users/willishoke/tropical` being the
current directory or on an implicit `~/tropical` checkout.

### Lane B — engine capability and host contract

**Mission:** make the existing engine-described surface complete enough that a
native client can render and drive it without semantic copies.

**Primary files:** `lean/Tropical/Playground/Vocabulary.lean`,
`lean/Tropical/Playground/Decode.lean`,
`lean/Tropical/Playground/Report.lean`,
`lean/Tropical/Engine/Front.lean`, contract tests.

#### B1. Version the vocabulary response — 0.5 day

- Add a stable schema identifier/version and deterministic vocabulary
  fingerprint to `get_vocabulary`.
- Keep kinds/ports/defaults/disciplines/display metadata derived from
  `portSpecs`.
- Add only presentation-neutral semantic metadata the client truly needs.
- Gate that served kinds are represented once and withheld kinds stay absent.

**Exit:** the same response drives a compiling minimal graph for every served
kind and decodes in Swift fixtures.

#### B2. Make `load_patch_graph` the compile handshake — 0.5 day

- Treat its realized report as the authoritative generation boundary.
- Ensure it carries vocabulary fingerprint, program/control generation, exact
  tap bindings, node status, inlet state, and live param disciplines.
- Keep legal-but-incomplete states factual; keep malformed documents errors.
- Eliminate the native client's post-compile race through a separate
  `list_scope_taps` request.

**Exit:** one response is enough to publish `RealizedPatch` and bind its scope
taps to the same generation.

#### B3. Unified write conformance — 0.5 day

- Retain one public `set_param` on both socket and Lean control paths.
- Extend the existing host-dispatch differential if new room controls are
  introduced.
- Return a typed missing/inactive-parameter error; never accept a stale slot
  by index.

**Exit:** no Swift source or test calls `set_param_glide`, `set_param_freq`, or
`set_param_velocity`.

### Lane C — Swift vocabulary, graph, and document model

**Mission:** replace the hard-coded July semantic model with an
engine-described, loss-aware authoring model.

**Primary files:** imported `NodeKind.swift`, `PatchModel.swift`,
`PatchDocument.swift`, new vocabulary/realized-state types and tests.

#### C1. Dynamic semantic model — 1 day

- Replace `NodeKind: CaseIterable enum` as the semantic key with a string
  `NodeKindID` plus decoded `NodeDescriptor`/`PortDescriptor` values.
- Derive connection legality from `outlet.color ∈ inlet.accepts`, `multi`, and
  graph acyclicity.
- Derive controls from engine defaults, ranges, log/unit metadata, and
  realized write discipline.
- Keep client-only monitor descriptors in a visibly separate namespace.
- Keep optional visual theme overrides keyed by kind, with a generic fallback;
  they may not change semantics.

**Exit:** adding a served engine kind requires no Swift semantic edit.

#### C2. Authored versus realized state — 0.5–1 day

- Store the authored graph even when the last compile fails.
- Publish a new realized snapshot only after a successful compile handshake.
- Show active/excluded, wired/normalled, live/structural, compiling, failed,
  and superseded states explicitly.
- Reapply live values by realized parameter name after a structural hot-swap;
  do not assume the old slot still exists.

**Exit:** a compile failure cannot make the UI claim the edited graph is
running, and a legal excluded branch is visibly excluded.

#### C3. Patch document v2 and v1 migration — 1 day

- Change document node kinds to strings.
- Store vocabulary schema/fingerprint, scene metadata, client monitor state,
  and structural parameter JSON without reducing it to `[String: Double]`.
- Decode v1 documents through an explicit migrator.
- Preserve unknown kinds, ports, edges, and values in a blocked migration
  representation; do not drop them silently.
- Validate IDs, order, edge targets, arity, colors, cycles, numeric finiteness,
  and the single output contract before compile.
- Present a deterministic migration report and require an explicit save to
  write v2.

**Exit:** every checked-in v1 fixture either round-trips semantically or opens
with a precise, non-destructive migration blocker.

### Lane D — process, RPC lanes, controls, and telemetry

**Mission:** preserve Reversible's lifecycle strengths while making compiler,
control, scope, and telemetry traffic genuinely independent.

**Primary files:** imported `Engine.swift`, `PatchModel.swift`, new RPC and
supervisor types, socket integration tests.

#### D1. Split supervision from connections — 1 day

- One `EngineSupervisor` owns spawn, socket path, readiness, termination,
  crash cleanup, and prior-orphan sweep.
- Independent framed RPC actors own graph/control, scope, and telemetry
  connections.
- Match replies by request ID per connection and bound pending requests.
- Cancel/fail all requests exactly once on child exit.
- Keep app termination and fatal-signal behavior, adding tests around all
  non-fatal paths.

**Exit:** a deliberately delayed compile or scope response cannot prevent a
  concurrent `set_param` request from being issued and answered.

#### D2. Latest-value controls — 0.5 day

- Port the current demo's bounded latest-value sender: first value, one
  direction reversal, and final value are retained; obsolete pointer events
  are superseded.
- Use only unified `set_param`.
- Reconcile UI value with accepted/published/audible/superseded response facts.
- Keep transport clock rebasing from the response's effective sample index.

**Exit:** a deterministic 5-second adversarial gesture converges to its final
value and preserves its one reversal without an unbounded queue.

#### D3. Full fault telemetry — 0.5 day

- Replace the old DAC-only xrun status with `get_telemetry` plus
  `audio_status`.
- Surface runtime ownership, Metal dispatch/starvation/tag/activation/morph,
  callback, DAC, and non-finite failures separately.
- Record program/control generation in diagnostic snapshots.

**Exit:** every release-blocking counter in this handoff appears in captured
evidence and a nonzero synthetic fault produces a visible failure.

### Lane E — native scope and read isolation

**Mission:** port the qualified modal phase view without reintroducing UI,
control, or phase jitter.

**Primary files:** imported `ScopeView.swift`, new scope sampler/frame model,
display-link adapter, shared scope fixtures/tests, current
`playground/scope-*.js` as the oracle.

#### E1. Immutable frame pipeline — 0.5–1 day

- Request explicit visible tap names in the graph.
- Bind returned slots, `program_version`, `control_version`, clock mapping, and
  effective sample index as one frame input.
- Fetch on the scope actor, never the MainActor.
- Keep at most one request in flight and one latest complete frame.
- Draw with `CVDisplayLink`, capped at 60 Hz.
- Treat preemption as a dropped frame, not a graph/control failure.

**Exit:** enabling, disabling, or stalling scope fetch cannot delay control RPC
issuance and cannot alter audio output.

#### E2. Port the accepted five-mode view exactly — 1 day

- One canvas overlays the active chord's five fundamentals.
- Use the same linear 1,792-sample / 40.63 ms window as the accepted scene; no
  logarithmic horizontal scaling.
- Remove the exact paired envelope point-by-point before lock, independently
  lock each mode to an interpolated positive-going zero crossing at the center
  graticule, then restore the audible-now envelope with one shared volts/div.
- Keep twice the base time span in the same width.
- Port JS oracle fixtures before optimizing Swift drawing.

**Exit:** Swift and the existing qualified JS math agree on chosen lock,
display amplitude, cycle count, and sample window for every chord/mode fixture.

#### E3. Muted determinism and cadence gate — 0.5 day

- All automated scope/gesture qualification sets `master.gain=0` before
  starting the DAC and proves the captured output is silent.
- Repeating a frame request at identical program/control generation and clock
  coordinates produces bit-identical values.
- Run four complete 16-second loops; the same chord/time snapshot must not
  choose a different trace on a later loop.

**Exit:** phase/amplitude stability and display cadence meet the scope rows in
the acceptance matrix.

### Lane F — live room and authored scene

**Mission:** make room algebra audible as an actual effect inside the
self-contained graph, then use it in a scene worth demonstrating.

**Primary files:** modal/room builders and lowering, room assets, current scene
data, Reversible release patch, differential/listening fixtures.

#### F1. Establish the semantic truth baseline — 0.5 day

- Author and check in `string → reverb → out` and
  `gong → reverb → out` minimal patches using the existing real modal reverb.
- Prove wet output nulls when its source is removed or zeroed.
- Prove `rt60`, `dir`, `sway`, and `rate` change wet samples through
  `set_param` without `load_patch_graph`.
- Prove a structural source change changes the compiled wet result.
- Exercise forward, hold, reverse, and seek without state reset.

**Exit:** an automated test would fail if the room were replaced by a fixed wet
render. This is the correctness baseline, not automatically the final sound.

#### F2. Select the production live-room realization — 1 day spike, Day-3 gate

The preferred architecture is a generalized `groupedroom` whose immutable
asset contains only the selected room carriers and whose input is the authored
modal source bank. It composes source and room through the modal seam, realizes
last, and uses shared/batched carrier data across timed source islands.

The spike measures and auditions:

1. generalized grouped composition for arbitrary accepted modal rows;
2. per-event modal islands sharing one immutable room-carrier asset; and
3. the existing 32-mode `reverb` as a semantic and listening control.

The decision must account for separate strike anchors. `modalmix` cannot erase
event timestamps merely to reduce cost. If batching timed islands needs a new
named compiler constructor, it follows the repository's cockpit → independent
oracle → named Lean constructor promotion rule.

The selected path must satisfy all of these:

- the room node has a modal source inlet;
- the asset has no authored-score samples;
- source edits change wet output;
- continuous room controls do not trigger relowering;
- JIT and Metal match the independent oracle;
- cost fits the exact scene at release buffers; and
- the user listening gate accepts it as a front-and-center effect.

`groupedroomcache` is not a fallback. If no live candidate passes by the Day-4
risk checkpoint, scope is cut elsewhere and the release remains blocked.

#### F3. Re-author the scene as a patch — 1–2 days

- Recreate the accepted harmonic chord lattice and independently phased modal
  envelopes as authored source data.
- Keep the chord-derived pizzicato/ghost idea only if it survives comparison
  with at least two more forceful transient articulations.
- Route every wet contribution through the selected live room node/path.
- Expose a small performance set: presence, veil, edge, room amount, temporal
  position/direction, flow, and level. Every control must map to a realized
  live parameter or be labelled structural.
- Keep room return and dry impact independently measurable.
- Check in the release `.rvpatch`; do not generate its wet audio into an engine
  cache.

**Exit:** the native app opens the scene, the canvas reveals its real topology,
and the performance view drives that same graph.

#### F4. Sonic evidence — continuous, final decision Day 9

Produce unnormalized dry-only, wet-only, and full-mix captures for at least
three authored gestures. Include forward room, reverse/pre-tail, room-amount
sweep, source mutation, and a complete 16-second cycle. Record peaks and fault
counters. No hidden limiter or post-normalization.

**Exit:** user listening accepts one candidate. Objective gates can reject a
broken render; they cannot declare an uninspiring render compelling.

### Lane G — product integration and native UX

**Mission:** turn the integrated components into one coherent instrument
instead of a developer patcher plus a separate hard-coded demo.

**Primary files:** Reversible app/root/canvas/toolbar views, release patch,
bundle resources.

#### G1. Performance and patch views — 1 day

- Boot the checked-in release patch in a focused performance workspace.
- Put the accepted modal overlay and macro controls in that workspace.
- Make the full patch canvas one action away.
- Keep one document/model/engine generation under both views.
- Preserve keyboard and mouse transport, seek, reverse, and fine control.

**Exit:** changing a node in Patch view changes what Performance view hears;
there is no parallel scene implementation.

#### G2. Compile and truth presentation — 0.5 day

- Display authored-versus-running generation during compiles.
- Render exclusions, normals, live/structural controls, and faults clearly.
- Disable only actions invalid under the engine vocabulary; never hide an
  already-authored unknown node.
- Ensure scope monitors are visibly read-only and surface-owned.

**Exit:** five scripted failure states are distinguishable without reading a
terminal.

#### G3. Lifecycle and document polish — 0.5 day

- New/open/save/save-as, dirty state, recent file, and window title.
- Stop/reap engine on normal quit, last-window close, SIGTERM/SIGINT, failed
  boot, and app relaunch.
- Adopt real audio state after UI recreation.
- Never kill another live Reversible instance's engine.

**Exit:** the lifecycle matrix passes with zero orphan processes/sockets.

### Lane H — qualification, integration, and release

**Mission:** keep the branch reviewable, assemble evidence, and refuse false
completion.

**Primary files:** validation scripts, evidence index, shared build/test
registration. Lane H coordinates shared files but does not rewrite another
lane's implementation.

#### H1. Baseline and rolling integration — continuous

- Record toolchains, machine, OS, display refresh, audio device, device/render
  buffers, and baseline SHAs.
- Require focused tests and `git diff --check` per commit.
- Push after each checkpoint below.
- Run Tropical Tier-1 gates daily and the full gate after cross-lane merges.

#### H2. Release qualification — 1 day

- Five clean cold boots.
- Clean-checkout full validation.
- Document migration/round-trip suite.
- Muted control/scope gestures.
- Ten-minute exact-scene adversarial run.
- Lifecycle/orphan matrix.
- Dry/wet/full listening package.

**Exit:** every P0 acceptance row has a linked artifact, or the candidate is
explicitly blocked.

## Dependency graph

```text
History replay (A1)
   ├─→ package/tests (A2) ──────────────────────────────────────┐
   ├─→ engine contract (B) ─→ Swift graph/docs (C) ────────────┤
   └─→ process/RPC split (D) ─┬─→ native scope (E) ────────────┤
                              └─→ live room/scene (F) ─────────┤
                                                               ▼
                                                    product integration (G)
                                                               │
                                                               ▼
                                                   qualification/release (H)
```

F1 may begin from existing engine fixtures while B/C proceed. F2 must freeze
its room API before C3's v2 release fixture and G1's performance controls are
finalized. E and D coordinate connection ownership before either edits the
imported `Engine.swift`.

## File ownership and collision rules

| Surface | Primary lane | Required reviewer |
|---|---|---|
| import mapping/provenance/replayed commits | A | H |
| `reversible/Package.swift`, build/bundle scripts | A | G |
| vocabulary, decoder, realized report | B | C |
| Swift graph/document types | C | B |
| engine supervisor, RPC framing, param sender | D | runtime reviewer |
| scope runtime contract | D | E |
| Swift scope math/rendering | E | scope-oracle reviewer |
| modal room constructors/lowering/assets | F | DSP/algebra reviewer |
| release `.rvpatch` and macro mapping | F | C + G |
| SwiftUI app/canvas/toolbar | G | C |
| `Makefile`, CI, qualification scripts | H | affected lane |

No lane edits the same imported Swift file concurrently. Shared-file changes
are sequenced through H. The 15 replayed commits are immutable.

## Ten-day schedule

### Day 1 — Monday, August 3: secure history and establish baseline

- A creates/verifies the bundle, replays all 15 commits, proves the mapping,
  and pushes the branch.
- H records clean baseline gates and release-machine manifest.
- B freezes the vocabulary/compile-handshake additions.
- D freezes process and RPC actor ownership.

**Day exit:** standalone history is remotely recoverable; import proof green;
contracts reviewed.

### Day 2 — Tuesday, August 4: build and truthful client skeleton

- A lands Swift test target and monorepo build registration.
- B lands vocabulary version/fingerprint fixtures.
- C decodes the dynamic vocabulary and removes semantic `NodeKind` switching.
- F lands minimal real-reverb truth tests and begins the live-room spike.

**Day exit:** imported app builds; Swift can render a fixture-defined
vocabulary; fixed-cache substitution test exists and fails on a cache.

### Day 3 — Wednesday, August 5: freeze the room and document direction

- F presents cost, oracle, and listening evidence for the three live-room
  realizations and selects one.
- B lands the atomic compile handshake.
- C freezes patch-document v2 and v1 migration policy.
- D lands split RPC connections.

**Day exit:** production room path selected; no unresolved semantic data shape.

### Day 4 — Thursday, August 6: risk checkpoint

Staff/user review answers:

1. Does the selected room consume the current source graph?
2. Does it sound promising enough to continue as the release room?
3. Can it fit the exact graph without a source-specific wet cache?
4. Are controls independent of compile and scope traffic?
5. Does v1 document migration preserve every authored fact?

If 1 or 3 is no, the demo release is blocked and scope is cut to room
correctness. If 2 is no, re-author the room/source interaction immediately;
do not polish the UI around it.

### Day 5 — Friday, August 7: first end-to-end native instrument

- C lands authored/realized graph state and v1 migrator.
- D lands latest-value control and full telemetry.
- E lands immutable frame transport and JS/Swift oracle parity.
- F lands a first full live scene patch.
- G boots it in a provisional performance view.

**Day exit:** native end-to-end candidate exists, even if sound/UI are rough.

### Day 6 — Monday, August 10: phase scope and room hardening

- E lands the five-mode overlay and muted determinism tests.
- F closes JIT/Metal/live-control differentials.
- A produces the first relocatable `.app`.
- H runs the first five-minute integrated stress.

### Day 7 — Tuesday, August 11: sonic comparison and document round trip

- F delivers three transient/room candidates with dry/wet/full captures.
- C/G exercise v1 migration and v2 save/reopen on the real scene.
- D/E run control/scope isolation measurements.
- H runs full Tropical validation.

### Day 8 — Wednesday, August 12: scope freeze

At noon:

- no new node family;
- no new room profile;
- no second scope;
- no document-shape broadening;
- no packaging feature beyond release-machine needs.

Select one sonic candidate. Cut P2 and any P1 that threatens correctness,
sound, or qualification.

### Day 9 — Thursday, August 13: release candidate

- User performs the listening decision on unnormalized captures and the live
  app.
- H runs five cold boots, muted gesture qualification, ten-minute adversarial
  qualification, lifecycle matrix, and clean full gates.
- G produces the candidate bundle and evidence links.

### Day 10 — Friday, August 14: close

Only release blockers and factual corrections.

- Read back the remote history and candidate refs.
- Re-run failed/repaired focused gates and final clean-checkout validation.
- Sign the decision log.
- Label the result release-candidate, integrated-but-sonically-blocked, or
  technically-blocked. Do not collapse those states.

## Validation protocol

### Automated audio policy

All automated scope, control-latency, lifecycle, document, and soak tests are
output-muted unless the test is explicitly named a listening capture.

For a hardware/DAC workload:

1. load the graph with its final level at zero;
2. write `master.gain=0` through unified `set_param`;
3. start audio only after the mute is acknowledged;
4. capture or probe the output; and
5. fail if maximum absolute sample exceeds `1e-12`.

Offline numerical differentials do not open an audio device. Listening renders
are the only audible qualification and are announced as such.

### Gate tiers

#### Tier 0 — each commit

- imported-history verifier when applicable;
- `swift build --package-path reversible`;
- focused `swift test --package-path reversible --filter ...`;
- affected Lean/C++/Bun focused test;
- `git diff --check`;
- no unexpected generated or bundle output tracked.

#### Tier 1 — each checkpoint

- full Reversible Swift tests;
- built `tropicaltest`;
- affected Bun tests against the built frontend;
- affected CTest targets;
- vocabulary and host-dispatch conformance;
- v1/v2 document fixture suite.

#### Tier 2 — daily integration

```bash
make validate
make reversible-build
make reversible-test
```

The Swift gates run on macOS. Existing Linux Tropical validation remains
required and is not weakened by the app.

#### Tier 3 — release candidate

- clean checkout of the candidate SHA;
- Tier 2 from empty build products;
- relocatable app bundle smoke;
- five cold boots;
- muted gesture/scope qualification;
- ten-minute exact-scene adversarial qualification;
- save/quit/reopen/render round trip;
- child-process lifecycle matrix;
- unnormalized listening package and user decision;
- no uncommitted generated files.

## Acceptance matrix

### Provenance and repository

| ID | Gate | Pass criterion | Evidence |
|---|---|---|---|
| PR-01 | Source freeze | source is clean at `dea822e`; root/count/remotes/tags match this handoff | provenance file |
| PR-02 | Backup | bundle verifies and SHA-256 is recorded | verifier output |
| PR-03 | Commit replay | exactly 15 source-to-target rows; metadata and every per-commit subtree match | import verifier + TSV |
| PR-04 | Prefix containment | replayed commits touch only `reversible/` | import verifier |
| PR-05 | Remote recovery | fresh read of `origin/sprint/reversible-migration` reaches replayed head and provenance commit | command log |
| PR-06 | Clean candidate | no untracked build/evidence output in candidate worktree | `git status` |

### Build, package, and lifecycle

| ID | Gate | Pass criterion | Evidence |
|---|---|---|---|
| BP-01 | Swift clean build | succeeds with empty `reversible/.build` | build log |
| BP-02 | Swift tests | all unit/integration tests pass | test log |
| BP-03 | Relocation | app launched from `/private/tmp` finds bundled engine/assets | launch log |
| BP-04 | Cold boot | 5/5 compile, bind taps, and reach ready state | cold-boot report |
| BP-05 | Normal lifecycle | quit, close-last-window, stop, relaunch leave no child/socket | PID/socket report |
| BP-06 | Failure lifecycle | failed boot and killed UI do not leave an audible orphan; prior-orphan sweep does not kill a live sibling | lifecycle fixture |
| BP-07 | Repository independence | no production path contains a hard-coded `/Users/...` or implicit `~/tropical` engine lookup | source audit |

### Vocabulary, graph, and documents

| ID | Gate | Pass criterion | Evidence |
|---|---|---|---|
| VG-01 | Vocabulary parity | every served kind and port decodes; withheld kind absent; fingerprint stable for identical table | cross-language fixture |
| VG-02 | No semantic copy | Swift has no exhaustive served-kind enum, hard-coded port-color rule, or parameter verb table | source audit |
| VG-03 | Connection parity | Swift acceptance/rejection equals engine for every color/arity pair plus cycle/dangling cases | generated differential |
| VG-04 | Realized truth | active/excluded, wired/normalled, params/disciplines, taps and generation match load response | integration test |
| VG-05 | Unified writes | all live knobs/transport use `set_param`; no retired verb remains | source/test audit |
| VG-06 | v1 migration | all v1 fixtures preserve nodes, IDs, order, positions, values, edges, transport, and monitors | migration report |
| VG-07 | Unknown data | unknown kind/port/value opens blocked and survives save/export; nothing silently disappears | fixture |
| VG-08 | v2 round trip | save → quit → open → compile yields identical authored graph and equivalent realized graph | canonical JSON + render |

### Live room and time semantics

| ID | Gate | Pass criterion | Evidence |
|---|---|---|---|
| LR-01 | Input dependency | removing/zeroing the connected source nulls wet output below `1e-12`; substituting a different source changes it | offline differential |
| LR-02 | No wet-score asset | release graph has a modal room input and contains no `groupedroomcache`; production assets contain room data only | graph/asset audit |
| LR-03 | Live room params | every presented continuous room control changes wet samples through `set_param` with zero graph recompiles | RPC/render test |
| LR-04 | Structural source edit | editing authored modes relowers once and changes both dry and wet results | generation/render test |
| LR-05 | Backend agreement | JIT and Metal match independent room oracle within the selected constructor's recorded tolerance | differential sweep |
| LR-06 | Time coherence | forward/hold/reverse/seek samples equal random-access evaluation at the same `τ`; no tail reset | coordinate differential |
| LR-07 | Event anchors | every authored chord/transient retains its intended timestamp through room composition | onset oracle |
| LR-08 | Cost | exact graph meets release buffer/capacity with all starvation/dispatch/tag/activation/morph faults zero | hardware report |
| LR-09 | Honest sound decision | user accepts one live candidate as a front-and-center effect; otherwise release status is blocked | listening decision |

### Scope and control isolation

| ID | Gate | Pass criterion | Evidence |
|---|---|---|---|
| SC-01 | Read-only audio | scopes off/on produce bit-identical offline audio and identical live control image versions | hash/version test |
| SC-02 | Snapshot determinism | identical generation/clock request returns bit-identical values and lock selection | repeated-frame test |
| SC-03 | Center lock | each visible mode's interpolated positive zero crossing is at center within 0.5 display pixel | image/math fixture |
| SC-04 | Amplitude honesty | displayed amplitude matches exact audible-now modal envelope within 1% and uses one fixed volts/div | oracle fixture |
| SC-05 | Cycle stability | same chord and scene time on every loop produces the same phase/amplitude trace | four-loop capture |
| SC-06 | Timescale | linear 1,792-sample / 40.63 ms view; no log horizontal transform | Swift/JS fixture |
| SC-07 | Idle cadence | at least 58 complete trace updates/s over 60 s on the qualified 60 Hz display | cadence log |
| SC-08 | Gesture cadence | at least 55 complete trace updates/s during the adversarial knob gesture, with no UI pause over 100 ms | cadence/event log |
| SC-09 | Control latency | scheduled write p95 ≤ 15 ms and audible activation p95 ≤ 50 ms with scope active | telemetry report |
| SC-10 | Bounded control queue | first, one reversal, and final value retained; final converges; pending writes remain bounded | gesture fixture |
| SC-11 | Muted automation | automated scope/gesture capture max absolute sample ≤ `1e-12` | mute capture |

### Runtime and release workload

| ID | Gate | Pass criterion | Evidence |
|---|---|---|---|
| RT-01 | Exact fault set | runtime ownership, Metal dispatch/starvation/tag/activation/stale/callback/morph, DAC under/overrun, non-finite, and clamp fault counts all zero | telemetry JSON |
| RT-02 | Ten-minute workload | exact release graph, scopes, transport, and gestures complete 10 min at recorded `Bdev`/`Rgpu` | soak report |
| RT-03 | No hidden silence | non-muted listening capture has no unexplained zero block; muted automation proves intentional silence separately | WAV scan |
| RT-04 | Full Tropical gates | `make validate` passes from clean checkout | validation log |
| RT-05 | Swift gates | full Swift build/test and bundle smoke pass from clean checkout | validation log |

### Listening package

The listening gate receives, without normalization or a limiter:

1. dry-only complete cycle;
2. wet-only complete cycle;
3. full mix complete cycle;
4. room amount from zero to maximum on a repeated transient;
5. forward-to-reverse temporal room gesture;
6. source mutation A/B proving the room follows its input; and
7. at least three transient articulations in the same chord context.

Each capture records candidate SHA, patch hash, room-asset hash, backend,
sample rate, buffers, peak/RMS, and all fault counters. The user decision names
the accepted candidate or states why none passed.

## Logical commit and push checkpoints

The 15 history commits are their own first interval. After that, prefer small
independently valid commits in this order:

| Checkpoint | Content | Minimum gate before push |
|---|---|---|
| C0 | this handoff | link audit, `git diff --check` |
| C1 | 15 replayed commits | PR-01 through PR-05 |
| C2 | provenance, import verifier, Swift test target | import verifier + Swift build/test |
| C3 | versioned vocabulary and atomic compile report | vocabulary/Tropical tests |
| C4 | dynamic Swift model and unified writes | Swift contract tests + host differential |
| C5 | patch v2 and v1 migrator | document fixtures |
| C6 | split RPC actors, supervisor, latest-value controls, telemetry | concurrency/lifecycle tests |
| C7 | immutable native scope and phase-lock oracle | muted scope tests |
| C8 | live room constructor/path and backend differential | LR-01 through LR-08 focused gates |
| C9 | authored scene and performance/patch views | end-to-end smoke + document round trip |
| C10 | package, evidence, release candidate | complete Tier 3 |

Do not combine C1 with modernization. Do not combine a room semantic change
with a listening rebalance. Push after each checkpoint so provenance and
reviewable recovery points exist remotely.

## Decision log

| ID | Decision | Status | Reason / evidence |
|---|---|---|---|
| D-01 | Product remains a self-contained closed-form instrument, not a generic live-effects processor | frozen | architecture contract and user correction |
| D-02 | Import target is `reversible/` | frozen | collision-free mechanical prefix; preserves identity and SwiftPM boundary |
| D-03 | Replay 15 commits with `format-patch`/`git am --directory` | frozen | preserves per-commit provenance while changing root parent and prefix |
| D-04 | Integration base is pushed demo head `07a6b25` | frozen | contains the runtime/scope/telemetry work the native app must consume |
| D-05 | Engine vocabulary and realized report are semantic authority | frozen | prevents July contract copies from drifting again |
| D-06 | One public `set_param`; client does not choose discipline | frozen | current host contract and conformance gate |
| D-07 | One five-mode release scope at display cadence | frozen | accepted interaction design and phase-view result |
| D-08 | Fixed wet-score cache is forbidden in release graph | frozen | it has no source dependency and is not a live effect |
| D-09 | Electron remains as oracle/rollback through sprint | frozen | deleting it adds risk and loses qualified comparisons; removal is P2 |
| D-10 | Performance and Patch views share one document/graph | default, Day-5 review | avoids a second hard-coded demo while retaining a focused instrument |
| D-11 | Production live-room realization | Day-3 decision | generalized grouped room preferred; must pass cost, oracle, and audition |
| D-12 | Distribution signing | external decision by Day 8 | ad-hoc local bundle is P0; Developer ID/notarization needs credentials and target distribution decision |

### Ambiguities surfaced

Only two choices require later confirmation; neither blocks history import,
contract modernization, or the live-room truth tests.

1. **Room realization:** the preferred grouped-carrier composition must be
   measured with arbitrary modal rows and multiple event anchors. The existing
   source-specific grouped evaluator cannot simply be relabelled generic. Day 3
   is the bounded technical decision.
2. **External distribution:** a locally runnable `.app` is fully engineering-
   scoped. Handing the binary to someone outside the qualified Mac may require
   Developer ID signing/notarization credentials and a minimum macOS/CPU target.
   The candidate records that status rather than assuming it.

## Stop-the-line rules

Pause integration when any of these occurs:

1. standalone history cannot be reconstructed from the bundle or remote;
2. a replayed commit's metadata/tree differs;
3. Swift contains a new semantic copy of the engine vocabulary or discipline;
4. a document migration silently drops an authored fact;
5. the UI claims an edited graph is running before successful publication;
6. a room candidate does not depend on its connected source;
7. a release graph contains a pre-rendered wet-score cache;
8. scope work delays control, changes audio, or changes the clock mapping;
9. repeated identical scope coordinates produce different traces;
10. automated interaction testing produces audible output;
11. any release fault counter becomes nonzero;
12. JIT, Metal, and the independent room oracle disagree beyond the recorded
    tolerance;
13. an engine child or socket survives a tested exit path; or
14. prose or UI presents an unlanded/failed capability as complete.

The response is diagnosis, scope cut, and an explicit decision. It is never a
fixed cache, hidden warning, relaxed counter, or re-frozen golden without a
semantic explanation.

## Definition of done

The sprint is done only when:

### History

- the standalone repo has a verified bundle backup;
- all 15 commits are replayed, mapped, proved, and pushed;
- the source repo remains intact;
- post-import modernization is separate and reviewable.

### Native app

- Reversible builds/tests inside Tropical;
- it launches one owned bundled/dev engine without checkout-path assumptions;
- patch editing, live controls, transport, scopes, telemetry, and documents
  operate against current contracts;
- authored and realized state cannot be confused;
- lifecycle tests leave no orphan.

### DSP and scene

- the release patch is visible/editable and is the same graph the performance
  view drives;
- its room consumes modal source input live;
- no score-wet cache is in the production graph or asset set;
- room controls are live and source/topology edits are honest relowers;
- transport remains coherent through the room;
- the user accepts the sonic candidate, or the sprint is labelled blocked.

### Scope and realtime

- one overlaid five-mode view meets phase, amplitude, timescale, cadence, and
  determinism gates;
- scopes are read-only and independent of controls/audio;
- automated qualification is verifiably muted;
- the ten-minute workload and exact fault set pass.

### Release

- full Tropical and Swift gates pass from a clean checkout;
- the app bundle smoke passes;
- the evidence index links raw results for every P0 gate;
- branch and candidate refs are pushed;
- remaining P1/P2 work has an owner and does not inflate the release claim.

## Final evidence package

Lane H leaves one committed index containing:

1. source bundle path/hash and verification;
2. source→target commit map and import-verifier output;
3. baseline, checkpoint, and candidate SHAs;
4. toolchain/release-machine manifest;
5. vocabulary and host-dispatch conformance;
6. v1 migration and v2 round-trip reports;
7. live-room input-dependency and backend differentials;
8. room asset/patch hashes and cost report;
9. scope phase/amplitude/cadence/determinism results;
10. muted gesture capture and latency metrics;
11. cold-boot, ten-minute workload, and lifecycle reports;
12. clean Tropical/Swift validation logs;
13. dry/wet/full listening captures and the user decision; and
14. signed decision log, explicit release label, and remaining obligations.

No result may depend on an untracked local note, terminal scrollback, the
standalone repository remaining at one filesystem path, or an unstated waiver.
