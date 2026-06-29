# ArrowWarp cutover — handoff (the strangler endgame)

How to get from "ArrowWarp is six verified slices **beside** the pipeline" to
"ArrowWarp **is** the middle, and the strata passes are deleted." Lets a fresh
session execute the cutover without re-deriving the design. See PR #195 for the
slices; `design/arrow*-*.md` and the `project_arrow_edsl_rearchitecture` memory
for the theory.

## Where it stands (after slices 1–6)

ArrowWarp (`lean/Tropical/ArrowWarp.lean`) is a combinator library that **builds
the resolved IR `Program`** and then **reuses strata + emit verbatim**. It is a
front end sitting beside `parse → elaborate → strata → emit`, not yet replacing
any of it. Surface so far:

- `arr`-level arithmetic: `lit`/`mul`/`add`/`sub`/`toIntE`; `Warp` (`id`/`back`/
  `fwd`/`neg`) + `Warp.apply` (clock-as-value); `Builder.osc` (sources a stdlib
  voice as an instance — strata inlines it); `warpBank (Voice) (Tap[])` generic
  over the voice; the flanger/diagonal/reverse/fixed-point builders.
- The **render bridge**: `compileArrowCarrier` (Tropicaltest.lean) =
  `buildCarrier → Strata.run{upto:=5} → Core.check → compileSession → renderIrBytes`.
- Six arrow laws certified in `tropicaltest` (22/22): inverse, additive, diagonal,
  reverse-involution, reverse-swaps-delay, reverse-commutes-with-flanger (float
  byte-fails / fixed-point byte-passes).

What it does NOT yet do: products/MIMO, sum types, arrays, type-parameter
monomorphization, and it sources oscillators as `FixedSinOsc` *instances* (leaning
on strata's `inlineInstances`) rather than inlining them itself.

## The cutover goal

ArrowWarp emits the **post-strata** form directly, so the five strata passes
(`Strata/{Specialize,SumLower,InlineInstances,ArrayLower,IdentityElim}.lean` +
`Strata.lean`) become dead code and are deleted. `elaborate` (names → pointers,
scope) and the backend (`CompileResolved`/`Emit`/`EmitLlvm`) stay. The surface
(`.md` parser, MCP) targets ArrowWarp.

**The key fact that makes this possible:** each strata pass is an arrow-structure
operation in disguise.

| strata pass | ArrowWarp capability that subsumes it |
|---|---|
| `inlineInstances` (drop nesting) | **composition** — `>>>` *is* inlining; build inlined-by-construction (per-program path) |
| `specialize` (drop type params) | **monomorphic by construction** — instantiate combinators at concrete types |
| `sumLower` (variants → tag+scalars) | **ArrowChoice** — `left`/`right`/`+++`/`|||` (coproducts) |
| `arrayLower` (unroll fold/generate/…) | **array combinators** over an array carrier |
| `identityElim` (categorical id law) | **arrow-law normalization** (`arr id ⋙ f = f`) |

So the cutover is: grow ArrowWarp's combinator surface until those five
capabilities are complete, prove it reproduces the corpus, re-target the front
end, delete strata.

## The strangler invariant (do not violate)

Build beside; **gate on the corpus**; cut over; delete. Nothing is deleted until
*every* stdlib program and the 12 frozen audio goldens are reproduced by the
ArrowWarp path — byte-identical where the construction is structural (slices 1–2),
audio-identical where it differs but denotes the same (the law goldens; note the
float-reassociation finding — some may need the tolerance/fixed-point treatment,
slices 5–6).

## What ArrowWarp must grow (the gap)

1. **Products / MIMO.** Named multi-ports are records/rows; categorically products.
   Add `***`/`first`/`second`/`&&&` (have `&&&` ad hoc) + projections, and the
   named-port ⟷ tuple bridge. (Hughes `first` is the whole of MIMO; the data axis
   is orthogonal to `warp`, which is the clock axis — see the voice-ports plan.)
2. **Self-inlining or residual inline.** Decide: does ArrowWarp build oscillators
   **inlined by construction** (composition, no instance boundary — the per-program
   path, one flat DAG) or keep referencing instances + a residual inline pass?
   Recommend inline-by-construction for the per-program path; the **session path's
   instance boundaries are a separate concern** (the fractal/microkernel realization
   and the cross-instance-CSE / native-DAG Phase B — see below).
3. **Sum types → ArrowChoice**; **arrays → array combinators**; **type params →
   monomorphizing construction**. Each needed only when the corpus gate hits a
   program that uses it — grow on demand.
4. **Canonical/normalizing construction** so `arr id ⋙ f = f` etc. hold without a
   separate `identityElim`.

## Two routes (recommended order)

- **Route 2 first — the `Program → ArrowWarp` bridge (the corpus gate).** Write a
  translator that takes an elaborated `ResolvedProgram` and re-expresses it as
  ArrowWarp terms, which then emit. Run *every* stdlib program through
  `Program → ArrowWarp → emit` and diff against the `strata → emit` golden. This is
  slices 1–2 generalized to the whole corpus, and it is the proof that ArrowWarp
  covers strata's job — **before** anything is deleted. Grow the surface (above) as
  the gate demands.
- **Route 1 then — re-target the front end.** Once the corpus is green, have the
  surface/elaborator produce ArrowWarp terms directly (the combinators *are* the
  elaboration), emitting post-strata. Then delete strata.

## The session / MCP path — WARP-PUSH (the original wire-binding goal, subsumed)

The user-facing UX ("pretend you're writing real effects": `osc → relu → flange`)
desugars by **WARP-PUSH** (proven in the `delay = precompose-time` theorem,
`project_arrow_edsl_rearchitecture` memory): the user writes a downstream insert
chain; the compiler slides each `warp` *up* through the stateless cone to the
generators (R1 slide past clock-agnostic `arr` / R2 fuse `warp∘warp` / R3 fork
multi-tap / R4 land as the generator's clock arg), leaving a pointwise tail and no
buffer. This is the cutover of the **session layer** and it subsumes the original
voice-ports wire-binding (a voice port = a binding-site, dual to a type param;
admission = template, elaborated per-binding — see `voice-admission-handoff.md`).
Honest boundary: WARP-PUSH is exact on the stateless fragment and stops at a
*stateful* effect in the chain (the ceded island) — the UX must say so there.

## Risks / open questions

- **Corpus coverage is the long pole.** The whole stdlib uses features the slices
  haven't grown (arrays, sums, generics). The gate surfaces them one program at a
  time; budget accordingly.
- **Byte- vs audio-identity.** Structural reproductions are byte-identical; where
  ArrowWarp's construction differs from strata's it is audio-identical, and the
  float-reassociation finding (slice 5) means a few may be only audio-identical in
  float and byte-identical in fixed-point. The gate must accept audio-identity with
  a documented reason, not demand byte-identity everywhere.
- **Deleting strata safely.** Confirm nothing outside the per-program/session
  compile depends on `Strata.run` (e.g. `resolveProgramType`, `runStrataChecked`,
  the codec/`tropical_resolved_1` round-trip) before removing the passes.
- **Session-level sharing.** ArrowWarp's fan-out is a pure `let` *within one
  program's emit arena* (slice 4: dedup is at emit-stage `CoreArena` interning, not
  strata). Cross-*instance* sharing still does not happen (the spike) — if the
  session path wants it, that is the deferred native-DAG **Phase B** (carry the DAG
  across the codec), a separate project.
- **Carrier.** The cutover proceeds in **float**. The global fixed-point carrier
  swap (scope A: fixed-point `Sin`, both backends, DAC, re-freeze all goldens) is a
  separate, motivated-but-months-scale project; slice 6 proved its value at the
  carrier-parametric level. Orthogonal to the cutover.

## Code map

- Grow: `lean/Tropical/ArrowWarp.lean` (the library).
- Gate: `lean/Tropicaltest.lean` (the corpus diff lives here; `compileArrowCarrier`
  is the render bridge to reuse/generalize).
- Retire (after corpus-green): `lean/Tropical/Ir/Strata.lean` +
  `Strata/{Specialize,SumLower,InlineInstances,ArrayLower,IdentityElim}.lean`.
- Keep / reuse: `Ir/Elaborator.lean` (elaborate), `Ir/CompileResolved.lean`,
  `Ir/Emit.lean`, `Ir/EmitLlvm.lean` (emit), `Parse/Surface/*` (parser).
- Session/WARP-PUSH: `Engine.lean`, `Compile.lean`, `Lowering.lean`, `Wiring.lean`.

## Phased plan

- **C1.** `Program → ArrowWarp` bridge + corpus gate; grow products/sums/arrays/
  monomorphization as the gate demands. (Done when every stdlib program + the 12
  goldens reproduce.)
- **C2.** Re-target the front end (surface/elaborator → ArrowWarp directly), gated
  by C1's corpus diff.
- **C3.** Delete the strata passes; confirm no residual dependents.
- **C4.** Session/MCP WARP-PUSH desugaring (downstream effect-chains → upstream
  warps); the original wire-binding, subsumed.
- **(Parallel, optional, later.)** Scope-A fixed-point carrier swap.
