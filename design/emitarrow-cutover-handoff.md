# EmitArrow cutover — handoff (the strangler endgame)

How to get from "EmitArrow is a verified front end **beside** the pipeline" to
"EmitArrow **is** the middle, and the strata passes are deleted." Lets a fresh
session execute the cutover without re-deriving the design. See PR #195 for the
slices; `design/arrowwarp-slice-1.md` (historical) and the
`project_arrow_edsl_rearchitecture` memory for the theory.

## Vocabulary — which "emit" (read this first)

"Emit" is overloaded across several stages. The cutover only touches ONE of
them; the rest are reused verbatim. Pin the names before reading on:

```
EmitArrow combinators   build a Program (scalar, post-strata shape)
   │   ← THIS is "emit" in the compiling-to-categories sense:
   │      the arrow's morphisms EMIT IR nodes instead of computing values.
   │      `osc`, `warpBank`, `>>>`, `&&&` are smart constructors over `Expr`.
   ▼
Program (ResolvedProgram)          lean/Tropical/Ir/Nodes.lean
   │   Core.check  — narrows to the Core sub-IR, which is SCALAR BY DEFINITION
   │                 (no fold/generate/map/tag/match/type-params). Today strata
   │                 guarantees this; post-cutover the combinators do, by
   │                 construction.
   ▼
CoreProgram                        lean/Tropical/Ir/Core.lean
   │   Emit.emitResolvedProgram  (lean/Tropical/Ir/Emit.lean) — CoreExpr →
   │     FlatProgram: hash-cons DAG + CSE + register/slot alloc + NInstr stream.
   │     A literal file named Emit.lean. TOTAL over Core only. UNCHANGED by the
   │     cutover.
   ▼
PerInstancePlan                    lean/Tropical/Ir/CompileResolved.lean
   │   pack → tropical_plan_5
   ▼
FlatPlan
   │   emitKernel  (lean/Tropical/Ir/EmitLlvm.lean) — FlatPlan → textual LLVM
   │     IR. The codegen backend. Another "emit." UNCHANGED by the cutover.
   ▼
LLVM IR → JIT / wasm render → audio
```

So three distinct things wear the word: the **combinator layer** (the concept —
realization by emission), `Ir/Emit.lean` (Core→Flat instruction stream), and
`Ir/EmitLlvm.lean` (Flat→LLVM codegen). When this doc says **"emit learns the
combinators,"** it means the **combinator layer** (the top box). The lowering
that strata does today moves UP into the combinators, which then build scalar
Core by construction. `Ir/Emit.lean` and `Ir/EmitLlvm.lean` do not grow new
cases — Core stays scalar, the backend is untouched. This is the design the
slices already follow ("EmitArrow's only job is to build a CoreProgram"); it is
NOT "push fold-unrolling down into `Ir/Emit.lean`."

## Where it stands (after slices 1–6, rename, corpus gate)

EmitArrow (`lean/Tropical/EmitArrow.lean`) is a combinator library that builds
the resolved IR `Program` and reuses elaborate-link + emit verbatim. It is a
front end sitting beside `parse → elaborate → strata → emit`, not yet replacing
any of it. As of `0c17877`:

- `arr`-level arithmetic: `lit`/`mul`/`add`/`sub`/`neg`/`toIntE` (and the C1
  scalar wrappers `div`/`bitAnd`/`rshift`/`lshift`/`gt`/`roundE`/`toFloatE`/
  `clampE`/`selectE`). **There is no `Warp` type** — a warp is a clock
  expression from the same op set (`reverse = neg clk`, `delay δ = sub clk δ`,
  modulated `= sub clk (m clk)`). One algebra.
- `Voice` — the primitive morphism `Clock ⇝ Sig`, generic over the stdlib
  program that realizes it. `Builder.osc` sources a voice instance (strata
  inlines it). `warpBank (Voice) (Tap[])` — the voice-generic flanger.
- The corpus gate (`tropicaltest`): EmitArrow emit ≡ emit-stdlib, byte-identical,
  over `FlangeSin` / `ReversibleComb` / `FixedSinOsc`.
- `buildFixedSinOsc` — the oscillator built **from scratch** (FixedPhasor
  split-multiply + Payne-Hanek + degree-11 Horner `Sin`), not sourced as an
  instance. Byte-identical to the hand-written `FixedSinOsc`. This is the first
  proof that the combinator layer can emit a generator's body itself, not only
  reference one.
- Six arrow laws certified (inverse, additive, diagonal, reverse-involution,
  reverse-swaps-delay, reverse-commutes-with-flanger) — float byte-fails on the
  value-sum reassociation, fixed-point byte-passes. 25/25 tropicaltest.

What it does NOT yet do: products/MIMO, sum types, arrays, type-parameter
monomorphization.

## The cutover goal

EmitArrow's combinator layer emits the **post-strata** (scalar Core) form
directly, so the five strata passes (`Strata/{Specialize,SumLower,
InlineInstances,ArrayLower,IdentityElim}.lean` + `Strata.lean`) become dead code
and are deleted. `elaborate` (names → pointers, scope, cycle-check) and the
backend (`Core.check` / `CompileResolved` / `Emit` / `EmitLlvm`) stay. The
surface (`.md` parser, MCP/session) targets EmitArrow.

**The key fact:** each strata pass is an arrow-structure operation in disguise,
so each is absorbed by a combinator that emits the lowered scalar form.

| strata pass | EmitArrow combinator that subsumes it |
|---|---|
| `inlineInstances` (drop nesting) | **composition** — `>>>` *is* inlining; build inlined-by-construction |
| `specialize` (drop type params) | **monomorphic by construction** — instantiate combinators at concrete types |
| `sumLower` (variants → tag+scalars) | **ArrowChoice** — `left`/`right`/`+++`/`\|\|\|` (coproducts) |
| `arrayLower` (unroll fold/generate/…) | **array combinators** — `fold`/`generate`/`map` as structural ops over an array carrier |
| `identityElim` (categorical id law) | **arrow-law normalization** (`arr id ⋙ f = f` holds at construction) |

## Combinators are first-class STRUCTURAL operations (settled)

The arrow is **cartesian, not cartesian-closed**: it has products, composition,
and the diagonal (so `fold`/`generate`/`map`/`first`/`&&&` are first-class
structural morphisms) but **no exponentials/closures** — there are no runtime
higher-order programs. This resolves the long-standing "combinators: surface
sugar vs first-class HOPs?" fork: **neither.** A `fold` combinator is a smart
constructor that EMITS the unrolled scalar Core DAG (it *is* `arrayLower`,
phrased in the combinator layer). It is first-class (a real morphism you compose)
without needing closures, because emission happens at build time, not run time.
So "emit learns fold/generate" = "the combinator layer gains `fold`/`generate`
constructors that emit the unrolled structure."

## C1 — what EmitArrow must grow, and how it is gated (REVISED)

**Superseded:** the earlier plan's "Route 2 first — a `Program → ArrowWarp`
corpus-reproduction bridge" is **dropped**. Re-expressing every elaborated
stdlib program as combinators and diffing against the strata golden is
per-program hand-lowering with no payoff once the arrow laws are proven — it
tests the translator, not the design. Do not build it.

**The real C1 is per-CONSTRUCT, gated by the EXISTING goldens:**

1. Enumerate which rich constructors the stdlib actually exercises (`fold`/
   `generate`/`map`; sums/choice; products/MIMO; type params). That finite set
   is the worklist.
2. For each, add the combinator that emits its lowered scalar Core — porting the
   corresponding strata pass's logic into the constructor:
   - **Products / MIMO** — named multi-ports are records/rows, categorically
     products. Add `***`/`first`/`second`/`&&&` (have `&&&` ad hoc) +
     projections + the named-port ⟷ tuple bridge. (Hughes `first` is the whole
     of MIMO; the data axis is orthogonal to `warp`, the clock axis.)
   - **Arrays** — `fold`/`generate`/`map` as structural combinators (absorb
     `arrayLower`).
   - **Sums** — `left`/`right`/`+++`/`|||` (absorb `sumLower`).
   - **Type params** — monomorphizing construction (absorb `specialize`).
   - **Identity normalization** — canonical construction so `arr id ⋙ f = f`
     holds without `identityElim`.
3. The gate is the **existing** suite — `tropicaltest` goldens (byte/audio) and
   `wasm≡JIT` — run after re-targeting, NOT a new reproduction harness. C1 is
   done when the surface can be re-targeted (C2) and those goldens stay green.

## The strangler invariant (do not violate)

Build beside; **gate on the existing goldens**; cut over; delete. Nothing is
deleted until the surface re-targets EmitArrow and every frozen audio golden +
`wasm≡JIT` stays green — byte-identical where construction is structural,
audio-identical where it differs but denotes the same (the law goldens; note the
float-reassociation finding — some are only audio-identical in float and
byte-identical in fixed-point).

## The session / MCP / patcher path (settled: lower STRAIGHT to EmitArrow)

The patcher / session graph (wires-as-signals, `osc.out -> flange.in`,
instances + wires + params over MCP) **lowers directly to EmitArrow terms** — it
does NOT round-trip through synthetic `.trop`. Rationale:

- A patch graph and a `.trop` program are **peer notations** over the arrow IR:
  `.trop` is the text notation, the patcher is the graph notation. Wiring IS
  composition (`>>>`); fan-out IS the diagonal (`&&&`). The DAG the user draws
  *is* the EmitArrow DAG.
- Today the session lowers via `sessionToParsed → elaborate → compileSession`:
  a **resolved → named → resolved** round-trip (synthesize names, then re-resolve
  them). That detour is the friction behind the registration-wall episode.
  Lowering straight to EmitArrow deletes it.
- **Bonus:** building the patch as arrow composition FUSES it into one DAG
  (`>>>` = inlining), so cross-instance sharing — the diagonal across the whole
  patch — falls out for free. This is the deferred session-fusion ("Phase B")
  arriving as a consequence of treating wiring as composition, not as a separate
  project.
- `.trop` is **not** demoted: it remains the morphism-DEFINITION language (where
  you author a primitive's body/math/names), and a patch can be RENDERED as
  `.trop` text as a view. Definitions persist as `.trop`; patches persist as
  graphs; both compile to arrow terms. "Source + params is the data" still holds.

This is the C4 cutover. WARP-PUSH desugaring still applies for downstream insert
chains (`osc → relu → flange`): slide each `warp` up through the stateless cone
to the generators (proven in the `delay = precompose-time` theorem, see the
memory). Honest boundary: WARP-PUSH is exact on the stateless fragment and stops
at a stateful effect (the ceded island) — the UX must say so there.

## Risks / open questions

- **Corpus coverage is the long pole.** The stdlib uses features the slices
  haven't grown (arrays, sums, generics). The construct worklist (C1.1) surfaces
  them; budget per-construct, not per-program.
- **Byte- vs audio-identity.** Structural reproductions are byte-identical;
  where construction differs it is audio-identical, and the float-reassociation
  finding means a few are only audio-identical in float and byte-identical in
  fixed-point. The gate must accept audio-identity with a documented reason.
- **Deleting strata safely.** Confirm nothing outside the per-program/session
  compile depends on `Strata.run` (e.g. `resolveProgramType`, `runStrataChecked`,
  the codec/`tropical_resolved_1` round-trip) before removing the passes.
- **Carrier.** ~~The cutover proceeds in **float**. The global fixed-point carrier
  swap (scope A: fixed-point `Sin`, both backends, DAC, re-freeze all goldens) is
  a separate, motivated-but-months-scale project; slice 6 proved its value at the
  carrier-parametric level. Orthogonal to the cutover.~~ **EXECUTED 2026-07
  (`fixed-carrier` branch)** — the "months-scale" estimate was stale: the clock
  axis was already fully integer, so scope A reduced to (A) bounding the modal
  island's one unbounded-float-time site on the integer relative clock, (B1) the
  Q2.30 `FixedSin` datapath (`stdlib/FixedSin.md` ≡ `fixedSinCycSig`,
  corpus-gated byte-identical), (B3) the modal bank in Q (Q4.28 landings,
  associative i64 mode sums), (B2) the voices atop `FixedSin`. Both backends
  inherit via the shared IR text; ZERO goldens re-frozen (the cf goldens pin
  ModalVoice's float `Sin`, deliberately untouched; every fixed-path consumer is
  differential-gated). Remaining float residue is deliberate and enumerated in
  `design/fixed-carrier.md` (envelope `exp`, param landings, DAC scale). The
  motivating consequence for Metal is recorded in
  `design/metal-emitter-evaluation.md`.

## Code map

- Grow: `lean/Tropical/EmitArrow.lean` (the combinator layer).
- Gate: `lean/Tropicaltest.lean` (existing goldens; `compileArrowCarrier` is the
  render bridge, `runEmitCorpusGate` the reusable byte-gate).
- Retire (after re-target + goldens green): `lean/Tropical/Ir/Strata.lean` +
  `Strata/{Specialize,SumLower,InlineInstances,ArrayLower,IdentityElim}.lean`.
- Keep / reuse: `Ir/Elaborator.lean` (elaborate), `Ir/Core.lean` (Core.check),
  `Ir/Emit.lean` (Core→Flat), `Ir/CompileResolved.lean`, `Ir/EmitLlvm.lean`
  (Flat→LLVM), `Parse/Surface/*` (parser).
- Session/patcher (C4): `Engine.lean`, `Compile.lean`, `Lowering.lean`,
  `Wiring.lean` (the `sessionToParsed` round-trip is what C4 replaces).

## Phased plan

- **C1.** Grow the combinator layer per-construct (products/sums/arrays/
  monomorphization/normalization) until the surface can re-target. Gated by the
  existing goldens, not a reproduction harness.
- **C2.** Re-target the front end (surface/elaborator → EmitArrow directly — the
  combinators *are* the elaboration), gated by the existing goldens.
- **C3.** Delete the strata passes; confirm no residual dependents.
- **C4.** Session/MCP/patcher → lower straight to EmitArrow (delete the
  `sessionToParsed` round-trip; wiring = composition = fusion); WARP-PUSH
  desugaring for downstream effect-chains. The original wire-binding, subsumed.
- **(Parallel, optional, later.)** Scope-A fixed-point carrier swap.
