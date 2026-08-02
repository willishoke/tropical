# Reversible migration implementation summary

This is the sprint's running record of implementation-level decisions,
deviations, qualification status, and remaining obligations. Frozen product
and architecture decisions remain in `00-master.md`.

## Status

- Sprint branch: `sprint/reversible-migration`
- Integration baseline: `07a6b2517f8d24c8822d67e0337c0fd99d016bd8`
- Committed handoff: `3bf81bc47ca52ec1e30b72a5a013e34c1241e4ec`
- Corrected 15-commit replay head: `6a0e0c3765ed1ce263b8f96e8c096863ddd28710`
- History proof: passing
- Remote recovery: corrected replay read back from `origin/sprint/reversible-migration`

## Small-scale decisions

### S-01 — Keep the sprint in an isolated worktree

The user's primary checkout contains unrelated untracked design material.
All sprint edits, builds, and commits use `/private/tmp/tropical-reversible-migration-corrected`
so those files remain untouched.

### S-02 — Put the committed handoff before the replay

The handoff exists on the remote as commit `3bf81bc`, directly above the
integration baseline. An initial replay began at the integration baseline,
which preserved all 15 source commits but omitted C0 from the sprint branch.
Before modernization, the branch was rebuilt from the handoff commit, the
full proof was rerun, and the remote ref was corrected with a guarded lease.
This gives the intended order `baseline → handoff → replay`.

### S-03 — Make the import verifier restoration-friendly

The verifier accepts a source-repository argument. It can check either the
untouched standalone checkout or a repository restored from the verified
bundle, so the proof procedure is not coupled to one live checkout path.

### S-04 — Keep Swift tests conventional and record the host toolchain gap

The package uses an ordinary XCTest target and processed fixture resources,
which is the stable Swift 5.9/macOS package contract. A clean native build
passes with the locally compatible macOS 15.4 SDK. This execution host has
only Command Line Tools selected and exposes neither a usable XCTest module
nor a usable Swift Testing compatibility module, so `swift test` is recorded
as host-blocked until the qualified Mac has a complete matching Xcode. CI uses
`macos-14` and runs the unmodified build/test commands.

### S-05 — Package only room-owned grouped-carrier data

The app bundle mirrors the relative path compiled into the live
`groupedroom` plan and includes the `.tgrm` carrier asset plus its manifest.
It intentionally excludes the `*-scene-44100.f32le` wet-score cache and its
scene manifest. The packaged engine runs with
`Contents/Resources/Tropical` as its working directory; development runs use
an explicit repository-rooted engine override supplied by `run-dev`.

### S-06 — Bound continuous writes to first, one turn, and final

The Swift sender now has one public verb, `set_param`. While a write is in
flight it retains the first gesture value, the first direction-turning point,
and the newest final value; intermediate events are superseded. The unsent
buffer is therefore bounded at three values before issuance and two after the
first request begins. Non-finite pointer events are rejected client-side, and
the engine remains authoritative for write discipline and acceptance.

### S-07 — Freeze scope math against the qualified JavaScript oracle first

The native scope core uses the accepted geometry and phase algorithm as a
cross-language fixture: 1,792 samples / 40.63 ms on a linear axis, exact
paired-envelope removal before lock, an independent interpolated
positive-going center crossing for each of five modes, and audible-now
envelope restoration under one fixed scale. The five-mode Swift fixture
matches the JavaScript oracle to a maximum observed error of `4.44e-16`.
Display-link transport and canvas wiring consume this math in a later commit.

### S-08 — Return the publication token from inside the atomic load

`load_patch_graph` gets its program/control generation directly from the
runtime publication operation; it does not sample telemetry after loading,
which could observe a later control image. The response carries that token,
the vocabulary fingerprint, and complete tap bindings. The legacy
`list_scope_taps` method remains compatible for older clients, but new clients
need no second request to adopt one realized generation.

### S-09 — Type missing live parameters without hiding runtime faults

The socket data plane reports a missing or inactive name as JSON-RPC code
`-32004` with `data.category = "unknown_param"` and the requested name.
Operational Metal/runtime failures retain the internal-error code and a
distinct `runtime_failure` category instead of being mislabeled as document
or parameter mistakes.

### S-10 — Split traffic by connection under one process supervisor

The existing lifecycle owner remains the single authority for spawn, socket
namespace, crash/quit cleanup, and orphan sweeping. After readiness it opens
four framed connections—graph, control, scope, and telemetry—each with its own
IDs, read buffer, 128-request bound, timeouts, and exactly-once exit failure.
The public engine facade routes methods by traffic class, so delayed compile
or scope response bytes cannot serialize a knob write on the client.

### S-11 — Land the served live-room truth that does not prejudge gong admission

The production serialized-path gate uses `string → reverb → out`, with a
served resonator substitution. It proves exact silence after source removal,
wet changes after source substitution and a structural string-mode edit, live
`rt60`/`dir`/`sway`/`rate` writes under one program version, and bit-exact
forward/hold/reverse/seek coordinate equivalence. It does not expose
`bloomgong` or misrepresent signal-valued `gong` as modal.

### S-12 — Preserve blocked documents as authored data, not approximations

Version 2 stores engine node kinds as open string IDs, structural values as
arbitrary JSON, and client monitors in their own namespace. Unknown fields are
retained at every document layer and are written back unchanged. The v1
migrator operates on raw JSON so an unknown kind, port, edge, value, or vendor
field survives into a precise blocked v2 document; migration is in-memory and
cannot overwrite the source until the user explicitly saves as v2. Compile
validation reports integrity blockers without repairing or pruning authored
topology.

### S-13 — Make the embedded engine loader-local before signing

The checkout engine carries a development rpath back to the repository build
directory. Packaging now copies `libtropical.dylib` beside the embedded
frontend, adds `@executable_path` to the staged frontend, and only then signs
and verifies the app. The package script uses a dedicated Swift scratch path
and disables SwiftPM's nested sandbox so it is reproducible in managed build
environments. A copy of the resulting app under a fresh `/private/tmp`
directory successfully decoded vocabulary fingerprint
`fnv1a64:30da601c40478e7f` and compiled an active live `groupedroom` using only
the bundled room carrier and manifest.

### S-14 — Adopt compile truth once and keep scope traffic off control

Swift decodes the complete `load_patch_graph` handshake into one immutable
realized snapshot and advances it only for a newer published generation.
Failed, superseded, or stale responses leave the prior running graph and tap
bindings intact; the legacy tap-list call is satisfied from that snapshot and
does not issue a second RPC. `playback_position` is routed with
`render_window` on the scope connection, so a display-head read cannot occupy
the live-control connection. Deterministic socket-pair tests pin per-lane IDs
and buffers, out-of-order correlation, the 128-request bound, exactly-once
terminal failure, bounded connection timeout, and graph/control independence.

### S-15 — Keep UI identity open while preserving unrepresentable documents

The canvas now wraps either an engine-owned open `NodeKindID` descriptor or a
client-owned `client.monitor.*` identity; no exhaustive served-kind switch
defines semantics. Generic labels and colors are presentation fallbacks only.
Sink nodes are fixed and omitted from Add, and the boot graph selects its sink
and source-like node from descriptor shape. The legacy four-channel scope and
its window value remain explicitly client-owned.

The current canvas stores nodes in an ID-keyed dictionary and therefore cannot
faithfully edit duplicate IDs or malformed order/position shapes. Such v2
documents still open visibly blocked, but save/export returns the exact decoded
template until the shape is representable; it does not silently choose one
duplicate or normalize the source document. Engine graphs request a stable,
deduplicated array of only the tap node IDs actually observed by monitors.

### S-16 — Publish five scope modes only as one pinned display frame

The native scope requests all five explicit slots in one stride-1 projection,
transforms the response off the main actor, and publishes no partial mode set.
Preemption, malformed/partial data, inexact generations, and older generations
drop the whole frame. The preliminary head read chooses only the request
address; envelope restoration uses the render response's effective sample
index, which is pinned to that response's program/control generation. A
display link capped at 60 Hz is only a wake-up source, with one request in
flight and superseded ticks dropped. The reusable scope accepts an
exactly-five-mode scene binding; until the release scene is resolved, the
legacy patcher scope remains the explicit fallback rather than inventing
scene-specific node IDs.

### S-17 — Let reachability split audio and scope artifacts

Requested tap IDs stay in the complete authored graph for both compilation
artifacts. Each artifact's selected roots perform the exclusion naturally.
Pre-filtering requested IDs as if every tap were a disconnected projection
made an active audible node disappear from the audio graph and left its
downstream wire dangling. The handshake regression now scopes both an active
audible source and a disconnected projection, proving the audio graph remains
valid while the scope artifact publishes both explicit bindings.

## Open architecture gate

The master handoff also asks for a literal serialized `gong → reverb` fixture.
At the frozen baseline, served `gong` lowers to a signal outlet while
`reverb.in` accepts modal data; `bloomgong` is the only patch-surface modal
gong and is explicitly withheld by the sprint. The user has been asked to
choose between recording the string/resonator regression plus lower-level
gong seam evidence as the honest scope, or expanding the sprint to resolve
and admit the modal gong. No release claim treats this conflict as closed.

The release-scene migration has a separate constraint: the production
`groupedroom` implementation accepts only the frozen 12-pole source coordinate
table recorded in its carrier manifest, while the accepted 16-second scene is
an independently phased harmonic chord lattice. The engine rejects those
scene sources at the live-room boundary, so replacing `groupedroomcache` is
not a document-only edit. The release scene remains gated on either broadening
the live-room profile to accept the authored lattice or explicitly approving
a materially different composition; no smaller technical carrier demo is
being relabelled as the accepted scene.

## Qualification notes

No release claim has been made. The listening decision, qualified-Mac
hardware gates, ten-minute workload, and lifecycle/package evidence remain
required even if implementation and automated tests are complete. The local
Swift package build is green; the local Swift test gate is toolchain-blocked as
described in S-04.
