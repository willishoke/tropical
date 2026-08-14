# Arena-native `Sig` cutover — phase 3 handoff

- **Baseline:** phase-2 checkpoint `7a20103`
- **Date:** 2026-08-13
- **Status:** Phase 3 complete; modal compilation, patch lowering, gong
  authoring, and the production playground compiler now construct stable IDs
  directly in one active arena

## Landed migration

`Tropical.EmitArrow.ArenaModal` ports the retained modal topology without
flattening it into the scalar IR early. `ModalForest`, `ModalKernelExpr`,
`Oriented.Bank`, grouped-room specialization, bloom/live composition, and the
factored two-room terminals keep their existing domain records and authored
order. Their scalar fields are IDs; any callback that allocates expressions is
a `BuildM` action.

The migration covers:

- room profiles, residue algebra, realization, and sigma clamping;
- oriented banks and their terminal realization;
- generic kernel/forest stage retention;
- bloom and live-pole paths with the existing admission regions;
- grouped-room reference specialization;
- all routed images and both factored two-room terminals, including the
  Phaser product.

`ArenaPatch.lowerGraph` is now `PatchGraph → BuildM ArrowTerm`. Kahn ranks and
the mutual-recursion termination argument remain pure, while room-control
freezing, stage resolution, modal realization, effect callbacks, bloom folds,
and term emission share the caller's builder. No expression-tree adapter or
second source arena exists.

`ArenaGong` ports the analytic bloom warp and default register construction.
The production playground uses `ArenaCompiler.decodeGraph` and the native patch
lowerer inside one atomic program assembly. JSON validation, vocabulary
metadata, topology, exact arithmetic, and parameter-table policy remain pure.
Optional taps are attempted transactionally: a refused tap restores its local
builder state and publishes neither an assignment nor an output declaration.

## Arena-aware inspection

The phase introduced the focused queries needed by modal admission:

- `sigConstD?` and `sigConstF?` classify a frozen expression arena in
  child-before-parent ID order;
- `sigIsS0` performs the same one-pass ID classification without reifying
  trees;
- shallow literal/constant classifiers fail closed on dangling IDs;
- same-arena structural equality uses interned `ExprId` equality.

All routed tables, coefficient columns, loop binders, values, routes, and
dynamic counts are constructed once and retained as IDs. `bankSum` and
`routedSum` still receive their arrays in authored order.

## Production assembly boundary

`assembleCompleteWithResult` is the dynamic-output sibling of `assemble`.
It exists for the playground inspection build, where the successful tap set is
known only after lowering. The build, instance declarations, output surface,
assignments, and returned tap names are committed together; any error publishes
no program.

The ordinary stdlib boot path and its 15-program manifest are unchanged.
There is no new backend operation and no backend file changed.

## Qualification evidence

The new `arena-native-phase3` fixture constructs a retained
room → all-pass product → room spine through both representations. After the
ordinary resolved/GC boundary, the native and recursive references have
identical node arrays and identical plan wire.

| Observation | Value |
|---|---:|
| native authored unique nodes | 2,357 |
| native/reference reachable unique nodes | 2,160 / 2,160 |
| routed reductions | 24 |
| native/reference plan wire | 719,467 / 719,467 bytes |

The production Phaser gate retains its existing compact footprint and route
metadata:

| Observation | Value |
|---|---:|
| canonical 6→32→6-section→32 Metal scratch | 22,688 / 24,576 bytes |
| ordinary two-room baseline scratch | 22,320 bytes |
| source routed image | 192 × 4 |
| difference / physical images | 496 / 528 |
| total reciprocal rows | 1,792 |
| confluence rows | 6 |
| observed direct `phasercheck` wall time | about 19 seconds |

The exact phase-3 fixture proves the final reachable node count and plan
footprint did not grow relative to the recursive baseline. It also avoids any
expanded-tree evaluator as a comparator.

## Preserved refusals and order contracts

The native path preserves these named boundaries:

- signal → modal edges are rejected by playground validation;
- graph cycles and dangling nodes retain their existing messages;
- repeated nonterminal room crossings that need a composable DD carrier are
  refused;
- bloomed linear kernels and live direction/sway/gauge crossings without the
  oriented Gamma bridge are refused;
- bloom conditioning, depth, and unsupported coincident live-pole regions
  remain all-or-nothing exclusions;
- malformed or dangling expression IDs fail closed during inspection.

Forest branch order, room order, control order, table/value/route order, binder
identity, declaration order, tap order, and left-associated output sums remain
covered by the parity and production gates.

## Gate results

```text
lake build tropicaltest phasercheck diffcli Tropical.Semantics
  PASS

lean/.lake/build/bin/tropicaltest --arena-native-only
  PASS phase 1
  PASS phase 2
  PASS phase 3

lean/.lake/build/bin/phasercheck
  PASS modal-oriented-patch
  PASS modal-universe-history
  PASS modal-phaser

lean/.lake/build/bin/tropicaltest
  134/135 passed
```

The sole full-suite failure is the unchanged environment-only cooperative
Metal arm: `FlatRuntime: MetalKernel: no Metal device`. Its non-Metal routed-sum
coverage passes, and every other compiler, golden, modal, seam, bank-staging,
playground, and production non-emission gate passes.

## Remaining recursive inventory

No production patch output is authored through recursive `Sig`. The remaining
tree generation is intentionally quarantined for later phases:

- `EmitArrow/Sig.lean`, `Term.lean`, `Numerics.lean`, `Modal/**`, `Patch.lean`,
  and `Gong.lean` are the recursive reference generation used by exact parity
  fixtures and by the proof generation;
- `BankOrder.lean`, `ClockAlgebra.lean`, `Semantics/Sig.lean`, and
  `Ir/EmitBankLaws.lean` still state the recursive-tree theorems owned by phase
  4;
- `Testing/ArenaNative.lean` deliberately retains legacy halves for exact
  old/new comparison;
- modal oracle suites under `Tropicaltest/` continue to use the recursive
  reference compiler where they are testing modal mathematics directly, while
  their `compilePlanPure` production arms exercise the native compiler;
- `Playground/Vocabulary.lean` and `Playground/Decode.lean` still contain the
  legacy scalar builders used by exact-bake probes and classification-drift
  tests. Production compilation calls `ArenaVocabulary`/`ArenaDecode`; phase 5
  should split the pure metadata/probes from those legacy builders before
  deleting them;
- `EmitArrow.lean` imports both generations during coexistence. Promotion and
  recursive-module deletion remain phase-5 work.

There are no temporary tree-to-ID adapters. The next phase can work directly
on arena-native theorem statements and traversal/classifier proofs without
changing the production compiler boundary.
