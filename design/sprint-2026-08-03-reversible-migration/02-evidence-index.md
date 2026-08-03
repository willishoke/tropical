# Reversible migration sprint evidence index

This is the committed evidence ledger for
`origin/sprint/reversible-migration`. A row marked **blocked** is not a waiver
or a pass. It names the remaining evidence or architecture decision needed
before a release-candidate label is honest.

## Identity and recovery

| Item | Evidence |
|---|---|
| Integration baseline | `07a6b2517f8d24c8822d67e0337c0fd99d016bd8` |
| Committed sprint handoff | `3bf81bc47ca52ec1e30b72a5a013e34c1241e4ec` |
| Standalone source | `/Users/willishoke/reversible` at `dea822ea1062a749ea1d7a76af1e2bd28194dfa1` |
| Complete source backup | `/Users/willishoke/reversible-before-tropical-2026-08-02.bundle`; SHA-256 `b60ee44f9ac18cb9f43be973bff6cd17d530325cc6b93a965fb4fe071bafd5e1`; 49,958 bytes; `git bundle verify` complete |
| Correct replay head | `6a0e0c3765ed1ce263b8f96e8c096863ddd28710` |
| Final implementation checkpoint | `4731701f0fc99e0bc55ab59e66aa93f1d5e0803e` |
| Source-to-target map | [`reversible/import-map.tsv`](../../reversible/import-map.tsv) |
| Provenance and recovery procedure | [`reversible/IMPORT_PROVENANCE.md`](../../reversible/IMPORT_PROVENANCE.md) |
| Executable import verifier | [`reversible/scripts/verify-import.sh`](../../reversible/scripts/verify-import.sh) |
| Toolchain and machine | [`evidence/machine-manifest.md`](evidence/machine-manifest.md) |

The verifier checks all 15 commits for linear parentage, author identity and
timestamp, full message, recursive tree identity, final subtree identity, and
prefix containment. The standalone checkout remains untouched.

## Engine and client contracts

| Gate | Evidence / result |
|---|---|
| Vocabulary identity | `tropicaltest` vocabulary gates and `reversible/Tests/ReversibleTests/EngineVocabularyTests.swift`; schema version 1, fingerprint `fnv1a64:30da601c40478e7f`, served identifiers unique, withheld `bloomgong` absent |
| Atomic compile truth | `tests/web/compile_handshake.test.ts`; response owns exact program/control versions, vocabulary fingerprint, realized facts, and full tap bindings |
| Unified writes | `tests/web/param_dispatch_conformance.test.ts` and `LatestValueBufferTests.swift`; one `set_param` verb, typed unknown-parameter failure, bounded first/turn/final retention |
| Split RPC lanes | `EngineRPCLaneTests.swift`; one process supervisor with independent graph, control, scope, and telemetry framed connections and bounded pending tables |
| Lossless documents | `PatchDocumentV2LosslessTests.swift` plus v1/v2 fixtures; unknown kinds, ports, edges, structural JSON, and vendor fields survive; explicit v2 save required |
| Live source truth | `lean/Tropical/Tropicaltest/LiveRoom.lean`; served serialized `string → reverb → out`, exact silence on source removal, source/structural mutation differentials, four live controls under one program, and exact forward/hold/reverse/seek coordinates |
| Scope math | `ModalScopeJSOracleParityTests.swift`; 1,792 points / 40.63 ms, paired-envelope removal/restoration, five independent positive crossings, maximum recorded JS/Swift error `4.44e-16` |
| Native truth presentation | `TruthInspectorView.swift`, `CanvasView.swift`, and `KnobView.swift`; authored revision and adopted program/control generation remain distinct, with engine-owned node, inlet, live/structural, failure, blocker, and monitor state projected without semantic name guesses |
| Native document state | Canonical current-versus-opened/saved v2 data drives the AppKit edited indicator; successful open/save records persistent recent URLs; v1 migration remains unsaved until explicit v2 Save As |

Final automated results recorded on the implementation checkpoint:

- `make validate` passed: native `tropicaltest` 124/124; Bun 155 passed,
  one intentional Metal-capability skip, zero failures, and 1,139 assertions
  across 13 files; the full Lean build and trust audit passed; CTest 4/4
  passed, including the Metal-kernel target;
- focused atomic-handshake and parameter-dispatch suites passed (3 tests,
  49 assertions), and the served live-room regression passed;
- Swift debug and production builds passed with the compatible 15.4
  command-line SDK after the final UI/document changes;
- `git diff --check` passed and the implementation worktree was clean;
- the local `swift test` executable gate is environment-blocked because the
  selected Command Line Tools installation does not contain `XCTest`.

## Package and live-room asset

| Item | Evidence / result |
|---|---|
| Carrier | `playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.tgrm`; 2,717,376 bytes; SHA-256 `838019933ddc885cb519ae0ba40233ee5d3e95cce4e8951ca566c8f1f5f65986` |
| Manifest | `playground/assets/grouped-room/clouds-current-radii-mono-v1-44100.json`; 4,716 bytes; SHA-256 `5ab2fd7d9215df2db3b273395c9ff17cd576ac031dd1e7ed40af74d5c2a78804` |
| Wet-score exclusion | `reversible/scripts/bundle-app` copies only the live carrier and manifest; no `*-scene-44100.f32le` cache is packaged |
| Relocation | `reversible/scripts/smoke-bundle.mjs` passed on checkpoint `4731701`; the release app was copied beneath the system temporary directory, then its embedded frontend resolved the adjacent dylib, decoded fingerprint `fnv1a64:30da601c40478e7f`, loaded the bundled carrier, published active `groupedroom` at program/control version 1/1, and found zero fixed wet-score files |

## Release gates still open

| Gate | Status and required evidence |
|---|---|
| Literal `gong → reverb` serialized regression | **Blocked by the current representation, not mathematical necessity.** Served `gong` realizes its two modal registers through signal-domain live pitch-bloom warps; `reverb.in` accepts modal. The proposed timed-modal-forest repair, seam hardening, and live-`beta` obligation are in [`03-modal-forest-generalized-room-handoff.md`](03-modal-forest-generalized-room-handoff.md). |
| Accepted release `.rvpatch` | **Blocked by live-room source contract.** `groupedroom` accepts only its frozen 12-pole source table, while the accepted scene has 16 timed islands and 200 modal rows. A room-only profile plus graph specialization/batched evaluator is proposed in [`03-modal-forest-generalized-room-handoff.md`](03-modal-forest-generalized-room-handoff.md). |
| Sonic decision and captures | **Blocked.** No dry/wet/full release capture or user listening acceptance exists for a live-room native scene. Earlier fixed-cache audition evidence cannot qualify this migration. |
| Qualified hardware | **Blocked.** Audio device, device/render buffers, display refresh, five cold boots, ten-minute exact-scene workload, and lifecycle/orphan matrix remain to be recorded on the qualified Mac. |
| Swift XCTest | **Environment-blocked here.** Run `swift test --package-path reversible` under a complete Xcode installation or CI `macos-14` image. |

## Release label

The branch is an integration candidate, not a release candidate. Automated
implementation gates may continue to pass, but the two architecture choices,
qualified-hardware run, and user listening decision above remain P0. The
implementation decision log is
[`01-implementation-summary.md`](01-implementation-summary.md).
