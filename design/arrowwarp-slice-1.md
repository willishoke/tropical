# ArrowWarp — vertical slice 1 (the flanger, gated)

Branch `feat/voice-wire-binding` (the voice-ports sprint has turned into the
ArrowWarp rearchitecture — see `design/arrow-edsl/` thinking and the memory).
This is the first vertebra of the strangler fig: a minimal arrow combinator
library that emits the **post-strata IR the existing backend already eats**, and
is gated byte/audio-exact against a golden the repo already has.

## Architectural principle (why the slice is small)

ArrowWarp's only job is to **build a `ResolvedProgram` / `CoreProgram`** (the
post-strata IR). Everything below that is reused verbatim:

```
ArrowWarp combinators  →  ResolvedProgram (flat, post-strata)
   │  Core.check
   ▼
CoreProgram  ──compileResolved (CompileResolved.lean:65)──▶  PerInstancePlan
   │  pack as a one-instance FlatPlan  (the lift emit-file already uses)
   ▼
FlatPlan  ──emitKernel (EmitLlvm.lean:531)──▶  LLVM IR  ──render──▶  audio
```

We are not building a compiler. We are building a **front end that emits the IR
the backend already consumes**, and racing it against `FlangeSin`.

## The target (M0 — DONE)

`diffcli strata-file stdlib/parsed/FlangeSin.json --upto=5` (default `--mode=inline`)
yields the literal target — a `tropical_resolved_1` `ResolvedProgram`:

- **name** `FlangeSin`; **decls** 0, **registry** 0 — fully inlined, one flat DAG.
- **binderCount** 39 — the shared subexpressions (native-DAG CSE: the fan-out on
  the clock + the shared osc body).
- **inputs**: `clk : int` (default `sampleIndex << 32`), `freq : float`
  (default `220`), `depth : float` (default `0.0007`).
- **outputs**: `out : float`.
- **assigns**: 1 — `out = add( add( mul(0.5, dry), mul(0.25, past) ), mul(0.25, ahead) )`
  where `dry/past/ahead = sin(phasor(clk)) / sin(phasor(clk−δ)) / sin(phasor(clk+δ))`,
  `δ = toInt(depth · sampleRate · 2³²)`.

So the combinators must produce: clock fanned three ways, warped by `{0, −δ, +δ}`,
each through the FixedPhasor→Sin closed form, weighted-summed. The 39 binders are
exactly the `&&&`-sharing the arena interning will recreate.

The emitted plan (`emit-file`, ~27 KB `tropical_plan_5`) is the stretch byte-target.

## Milestones

- **M0 — Pin the target.** DONE. Target shape above; reproduce with the
  `strata-file` invocation (committed input, no scratchpad dependency).
- **M1 — Builder substrate.** Smart constructors over the existing `Expr`.
  **KEY (M0 finding):** `sin`/`phasor` are NOT primitive nodes — the target is
  831 `mul`s + a Q32.32 fixed-point phasor (`bitAnd`/`rshift`/`toInt`) + an
  expanded Horner polynomial × 3. So ArrowWarp's **primitive morphisms are sourced
  from existing stdlib closed-forms, not hand-rolled**: the `osc` primitive *is*
  the elaborated `FixedSinOsc` body (`Clock ⇝ Float`). Constructors: `clock`
  (`sampleIndex << 32`), clock-arith (`clk ± δ`, `δ = toInt(depth·sampleRate·2³²)`),
  `arr`-arithmetic (`mul`/`add`), and `osc` (reuse the FixedSinOsc body, substitute
  its `clk`). Wrap a flat `ResolvedProgram` (inputs clk/freq/depth, output out).
  (Oscillators-as-warps is a later slice — a factoring, not slice 1's problem.)
- **M2 — The four combinators.** `arr`, `>>>`, `&&&`, `warp` as a thin float-only
  typed layer over M1. `warp` = clock-arithmetic wired to the clk input (the
  clock-as-value view; fan-out on the clock *is* the diagonal). The flanger as one
  expression: `(warp 0 &&& warp(−δ) &&& warp(+δ)) >>> oscₓ3 >>> weightedSum`.
- **M3 — Emission bridge.** Built `ResolvedProgram` → `Core.check` →
  `compileResolved` → pack one-instance `FlatPlan` (reuse the `emit-file` lift) →
  `emitKernel`. Same path as production.
- **M4 — The gate.** `tropicaltest` case: render the ArrowWarp flanger, assert
  audio hash `== FlangeSin` golden. **Stretch**: plan byte-identity vs.
  `emit-file FlangeSin` (achievable — the voice-desugar proof hit it — but not a
  blocker; the audio hash is the correctness floor).

## Then grow (slices 2..n) — each arrow law a golden

`warp(neg)⋙warp(neg)=id` · `warp(−δ₁)⋙warp(−δ₂)=warp(−(δ₁+δ₂))` · the diagonal
(shared source ≡ independent copies) · `arr id ⋙ f = f` · assoc of `⋙`. Each a
three-line golden + whatever combinator it needs.

## Cut-over (later, gated)

When the slice covers the stdlib: point the surface parser at ArrowWarp instead
of elaborate+strata, delete the strata passes. Not a line until the corpus
goldens are green.

## Risks

1. **Plan byte-identity may be too strict for M4** (hand vs pipeline node order).
   Mitigation: gate on rendered audio (the golden mechanism); byte-identity is a
   separate later target.
2. **`Expr`/`CNode` constructor surface** — building post-strata IR by hand. But
   the flanger needs ~6 node kinds and M0 gives the exact target; if fiddlier,
   it surfaces in M1 and costs hours, not the slice.

## Laws proven (the safety net these goldens certify)

Denotation `⟦warp φ⟧ s = s∘φ`. Static warp is a lawful arrow (`warp(id)=arr id`;
`warp φ ⋙ warp ψ = warp(φ∘ψ)`; warp slides through pointwise `arr`; warp
distributes over `&&&` — all by ∘-associativity). Modulated warp is lawful iff
the transform is a function of the **clock** (pinned), not the flowing data —
which is also the musical semantics. So these goldens are theorems, not hopes.
