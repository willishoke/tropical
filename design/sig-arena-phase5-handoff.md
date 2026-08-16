# ID-native `Sig` cutover — phase 5 final checkpoint

- **Baseline:** phase-3 checkpoint `238ef11`
- **Date:** 2026-08-15
- **Status:** Complete on the non-Metal qualification host; Metal hardware
  qualification remains environment-scoped

The execution plan was
`design/sig-arena-phase5-completion-handoff.md`. This checkpoint records the
completed tree after recursive deletion and final API promotion.

## Final production surface

`Tropical.EmitArrow.Sig` is now `ExprId`-native. Smart constructors allocate in
one active `BuildM`/`ExprArena`, and `assemble` publishes expressions plus
ordered declarations atomically. There is no recursive authoring tree or
adapter in production or tests.

The temporary modules and namespaces have been promoted:

- `EmitArrow/{Sig,Inspect,Numerics,Term,ClockAlgebra,BankOrder}.lean`;
- `EmitArrow/Modal.lean` and `EmitArrow/Modal/**`;
- `EmitArrow/{Patch,Gong,Stdlib}.lean`;
- `Playground/{Vocabulary,Decode}.lean` with expression construction in
  `Tropical.Playground.Compiler`; and
- stable test surfaces `Testing/ArrowFixtures.lean`,
  `Testing/ClockLaws.lean`, and `Testing/EmitArrow.lean`.

The production aggregate exports the ID-native API directly from
`Tropical.EmitArrow`; active Lean sources contain no `Tropical.EmitArrow.ArenaNative`
reference or `Arena*` implementation import.

## Test and oracle cutover

All recursive fixture consumers were ported to the shared native construction
boundary. `Testing.ArrowFixtures` accepts only native IDs, `BuildM`, `Arena`,
and `ProgramIdx`. Independent Float/complex references live in
`Testing.ArrowOracles`; the production exact carrier remains the device under
test.

The vocabulary, Phaser, modal, oriented, grouped-room, patch, slide, stress,
clock, exact-bake, and seam suites now construct their DUT through the native
builder. The exact-corpse gate walks 29 promoted production modules, including
the nested modal and playground subtrees, and separately requires the retired
Float-tier definitions to exist in the test oracle module. Its final result is
zero generated-C libm call sites and zero retired Float definitions in
production.

Frozen observations did not change:

- Phase 1: 3 programs, 16 authored nodes, 1/9 reachable nodes, 341/1,945 wire
  bytes;
- Phase 2: 15 stdlib programs, 283 stdlib nodes, 268 numeric nodes/46,451 wire
  bytes, 3 carrier instances, 206 carrier nodes/36,103 wire bytes;
- Phase 3: 2,357 authored modal nodes, 2,160 reachable nodes, 24 routed
  reductions, 719,467 wire bytes;
- Phase 4: 7-node clock-law witness; and
- Phase 5: 20 served vocabulary kinds, 3 reserved parameters, fingerprint
  `fnv1a64:5b536cbc16add425`.

No tolerance or golden was changed.

## Deleted inventory and trust closure

The recursive implementations are absent:

- `EmitArrow/Sig.lean`, `Term.lean`, `Numerics.lean`, `Modal.lean`,
  `Modal/**`, `Patch.lean`, and `Gong.lean` were deleted before their native
  replacements took the stable paths;
- the recursive `Playground/Vocabulary.lean` and `Playground/Decode.lean` were
  deleted before native promotion; and
- the recursive `Testing/ArrowFixtures.lean` was deleted before the native
  fixture module took that path.

The three `lowerSigPtr` trust sites and the
`LOWER_SIG_PTR_REFINES_TREE` obligation were removed with the recursive
implementation. `productionTrustSites` is empty, the generated trust report is
current, and the filesystem audit reports zero sites.

## Qualification

Commands executed from the candidate tree:

```text
lake build Tropical.Semantics Tropical.EmitArrow Tropical.Ir.EmitBankLaws \
  Tropical.Playground tropicaltest trustreport phasercheck diffcli

lean/.lake/build/bin/tropicaltest --emitarrow-only
lean/.lake/build/bin/tropicaltest --arena-native-only  # temporary alias
lean/.lake/build/bin/tropicaltest --routed-only
lean/.lake/build/bin/tropicaltest --oriented-patch-only
lean/.lake/build/bin/tropicaltest --phaser-only
lean/.lake/build/bin/tropicaltest --modal-universe-history-only
lean/.lake/build/bin/tropicaltest --ecdd-only
lean/.lake/build/bin/phasercheck
lean/.lake/build/bin/trustreport --check
python3 tools/audit_trust_sites.py
git diff --check
lean/.lake/build/bin/tropicaltest
```

Results:

```text
required Lean build
  PASS (244 jobs)

EmitArrow phases 1–5, oriented patch, Phaser, modal-universe history, ECDD,
phasercheck, trustreport, trust source audit, and diff check
  PASS

trust source audit
  PASS (0 sites)

exact-corpse
  PASS (29 production modules, 306,961 generated-C lines, 0 libm sites)

full tropicaltest
  136/137 passed
  sole failure: FlatRuntime: MetalKernel: no Metal device
```

The focused routed gate reaches the same device-only refusal. Its non-Metal
routed-sum equivalence, order proofs, generated LLVM/MSL coverage, and every
other suite gate pass. A Metal release host must still run this checkpoint to
137/137 before claiming hardware qualification.
