# Proof initiative 4 — stage-signature soundness and staging refinement

- **Baseline:** `origin/main` at `d7d9e1d4b4d161bf8b1e2800497d79c6381a609f`
- **Baseline date:** 2026-08-21
- **Priority:** P0
- **Difficulty:** Large
- **Suggested worker:** W4
- **Dependencies:** initiative 1 for semantic capstones; initiative 2 for the
  `shiftSampleIndex` construction certificate
- **Can start independently:** yes, through the structural milestones

## Outcome

Prove that intern-time `StageSig` classification is semantically conservative,
then prove that Stage0 and the new TileStage preserve Plan meaning under their
documented execution/publication protocols.

This initiative must correct one misleading phrase in the current code:
`TileStage.lean` says it delegates to “the proven typed Stage0 machinery,” but
`origin/main` contains no Stage0 semantic preservation theorem. Existing
differential and render gates are valuable evidence, not that theorem.

## Ground truth on the pinned baseline

- `Stage` is `fold < s0 < s1`; `StageSig` contains a base stage plus sorted
  input and nested-output dependency arrays.
- `enodeSig` runs at intern time. `inputRef` and `nestedOut` remain symbolic;
  `Staging.resolve` interprets them in a `StageCtx` that reflects child
  execution availability.
- `sampleIndex`, `tileSampleIndex`, and `tilePhase` are all classified `s1`.
  Direct semantics gives `tileSampleIndex` its own `SigEnv.tileSampleIndex`
  rail and maps `tilePhase` to literal zero. Ordinary exact/JIT invocations
  bind the tile rail equal to `sampleIndex`; materializer and substitution
  proofs may bind an independent absolute endpoint coordinate.
- `loopIdx` is deliberately stage-neutral at the value level. Stage0 pins
  delimiters and loop-index readers for individual moves, while permitting a
  whole region to move.
- `Stage0.hoistTyped` consumes emit-order instruction blocks plus optional
  typed stages. It moves `s0` work, duplicates required `fold` support,
  rewrites scalar crossings through `coef:` slots, moves complete reduction
  regions, and publishes coefficient-filled arrays.
- `TileStage.split` runs after Stage0. It finds the dependency slice rooted at
  `tileArraySlots`, retains/duplicates shared scalar support, delegates the
  rewrite to `Stage0.hoistTyped`, and restores the stage-0 array metadata.
- `StagedLoad.emitMetalParts` uses Stage0 followed by TileStage. JIT-only load
  paths do not use TileStage; an exact JIT plan remains loaded for observation
  and fallback.
- The absolute-time phaser path is flag- and shape-admitted, supported by
  multiprecision and hardware evidence, and not a globally promoted semantic
  replacement.

## Proof surface

The sprint resolved one semantic-model fork in favor of independent rails:
`SigEnv.sampleIndex` is the ordinary audio coordinate and
`SigEnv.tileSampleIndex` is the materializer coordinate. This mirrors Plan's
separate `.tick` and `.tileTick` sources and lets the shift theorem preserve
pre-existing tile-coordinate references without a syntactic exclusion.

### 1. Stage lattice and signature invariants

Prove the algebraic facts used implicitly throughout staging:

- `Stage.join` is associative, commutative, idempotent, and least-upper-bound
  for `Stage.le`;
- `StageSig.join` preserves strictly ascending deduplicated dependencies;
- `enodeSig` contains the union of every semantic child dependency;
- `Staging.resolve` is monotone in dependency stages and conservative for
  missing inputs/children;
- `stageOf` is conservative for dangling signatures.

Do not make sortedness a comment-only invariant. Either define a predicate and
prove it for interned signatures or replace the representation with a type
that carries it.

### 2. Semantic noninterference for `StageSig`

Define environment agreement at each binding time. A useful target is:

```lean
def EnvAgreesThrough (stage : Stage) (ctx : StageCtx)
    (a b : SigEnv α) : Prop := ...

theorem stageSig_sound ...
    (hstage : Staging.stageOf arena ctx id |>.le stage = true)
    (henv : EnvAgreesThrough stage ctx env₁ env₂) :
  denoteExpr alg env₁ arena hw id = denoteExpr alg env₂ arena hw id
```

For `s0`, environments may differ in per-sample clocks but must agree on
controls and resolved input/nested dependencies. For `fold`, they must also be
independent of control values. State the exact rules rather than appealing to
“tau-independent” prose.

Because `tilePhase` denotes zero in the unsplit direct semantics yet is marked
`s1`, the classifier is conservative. The proof need not infer the most static
possible stage.

### 3. Stage0 structural correctness

Before semantic preservation, prove invariants about the rewrite:

- `collectBlocks`/rebuild round-trip;
- instruction/stage block lengths align or `hoistTyped` refuses;
- surviving block order is preserved;
- coefficient stream order is preserved;
- only fold support is duplicated;
- scalar boundary slots are fresh, typed, and rewritten at every surviving
  use;
- coefficient/tile array-slot sets are in range and have unique publication
  roles;
- reduce regions move only as balanced units;
- no loop binder becomes unbound;
- an empty selected set produces semantic and structural identity.

### 4. Stage0 semantic refinement

Using initiative 1's `execBlocks`, define the two-step protocol:

1. run the coefficient plan at a control-write epoch into unpublished
   coefficient storage;
2. atomically publish its scalar/array image, then run the audio plan for a
   sample using that image.

Prove that this observation equals the original plan for environments stable
at the classified boundary:

```lean
theorem hoistTyped_refines ...
    (hsplit : Stage0.hoistTyped plan stages = .ok split)
    (hwf : FlatPlanWellFormed plan)
    (hsound : TypedStagesSound plan stages) :
  denoteStaged split controlEnv sampleEnv = denoteFlatPlan plan sampleEnv
```

The theorem is about one consistent published generation. Atomicity of the
C++ runtime's generation flip remains a host/runtime obligation.

### 5. Absolute-coordinate substitution

Using initiative 2's construction certificate, prove the semantic theorem for
`EmitArrow.shiftSampleIndex`:

- evaluating a shifted root with `tileSampleIndex = t` equals evaluating the
  original root with `sampleIndex = t + frames`;
- all other environment components and binder bindings are unchanged;
- frame zero is covered;
- `tilePhase` is preserved, not rewritten;
- arrays, bank sums, routed sums, and shared DAG nodes are covered.

This is an exact/JIT theorem over the direct expression semantics. It does not
claim that an interpolated Metal lane equals the full exact expression.

### 6. TileStage refinement

Define the tile protocol separately from Stage0:

1. the materializer evaluates endpoint-image arrays for the requested absolute
   interval using `tileTick` and materializer sources;
2. the host publishes those immutable arrays for one tile;
3. the audio plan consumes the arrays with lane `tilePhase`;
4. exact fallback remains the reference for unadmitted shapes.

Prove the exact left-endpoint property first: with `tilePhase = 0`, the staged
observation equals the unsplit exact/JIT graph at the interval start. Then
prove dependency-slice and shared-scalar duplication correctness.

Do not claim full-interval equality for the polynomial/interpolated Metal
approximation. Its error bound remains the numeric qualification described in
`absolute-time-phaser-coefficient-staging-decision-2026-08-18.md` unless a
separate approximation theorem is developed.

## Work plan

1. Prove the Stage/StageSig algebra and dependency-array invariants.
2. Prove semantic noninterference for resolved signatures.
3. Prove Stage0 block, order, boundary, and region invariants.
4. Integrate initiative 1 and prove `hoistTyped_refines`.
5. Prove `shiftSampleIndex` semantic substitution after initiative 2's
   construction interface lands.
6. Prove TileStage slice/duplication invariants and exact-left-endpoint
   refinement.
7. Re-run all stage, bank, phaser, and Metal differential evidence without
   changing its stated numeric scope.

## Owned files

- `lean/Tropical/Semantics/Staging.lean`
- `lean/Tropical/Ir/Stage0Laws.lean`
- `lean/Tropical/Ir/TileStageLaws.lean`
- `lean/Tropical/Testing/StagingLaws.lean`
- Minimal proof-enabling edits to `lean/Tropical/Ir/Stage0.lean` and
  `lean/Tropical/Ir/TileStage.lean`

W4 exclusively owns `Stage0.lean` and `TileStage.lean` during parallel work.
It consumes, but does not edit, W1's Plan semantics and W2's builder laws.

## Deliverables

- Stage lattice, signature-shape, resolution-monotonicity, and semantic
  noninterference theorems.
- Stage0 structural invariants and semantic refinement.
- `shiftSampleIndex` semantic substitution.
- TileStage dependency-slice correctness and exact-left-endpoint refinement.
- An explicit statement of what remains tolerance-backed rather than proved.

## Validation

```text
cd lean && lake build Tropical.Semantics.Staging Tropical.Ir.Stage0Laws Tropical.Ir.TileStageLaws
cd lean && lake build Tropical.Testing.StagingLaws tropicaltest phasercheck
./lean/.lake/build/bin/tropicaltest --phaser-only
./lean/.lake/build/bin/tropicaltest --routed-only
./lean/.lake/build/bin/phasercheck
./lean/.lake/build/bin/tropicaltest
```

## Exit criteria

- Interned signatures have machine-checked shape invariants.
- Resolved `fold`/`s0` classifications have a noninterference theorem.
- A successful typed Stage0 split refines the original Plan denotation under a
  stated publication protocol.
- TileStage has a proved exact-left-endpoint contract and shared-scalar safety.
- No theorem upgrades the measured Metal interpolation tolerance to exact
  equality.

## Risks and non-goals

- **Critical risk:** confusing value-stage neutrality of `loopIdx` with
  permission to move loop instructions individually.
- **Critical risk:** proving an in-memory two-plan equation that omits the
  coefficient/tile publication boundary.
- **High risk:** claiming whole-interval phaser equality from endpoint or
  numeric evidence.
- Metal compiler, GPU execution, worker deadlines, and callback atomicity stay
  outside this proof surface.
