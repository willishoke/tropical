import Tropical.EmitArrow.Modal
import Tropical.EmitArrow.Numerics

/-!
# The clock algebra as theorems (slice 3b) — warp laws over Int, not renders

`Tropicaltest/ArrowLaws.lean` defends the warp arrow laws by rendering both
sides and comparing SHA256 — a sample of a universally quantified statement.
This module states the statement: the integer fragment of `Sig` (the Q32.32
fixed-point clock rail) gets a typing judgment (`OnClockRail`), a total
`Int` denotation dependent on the derivation (`denoteClock` — no `Option`,
no `Float` anywhere), and the warp laws become theorems quantified over ALL
rail expressions, not the four that get rendered.

## The carving (WS-1's finding)

The handoff sketched the fragment as `{num, binary add/sub/mul, select,
sampleIndex, clock leaf, toIntE, ldexpE}`. The rail as actually carved by
the code is different, and the difference is the finding:

* `select` and `ldexpE` are NOT clock-rail ops — they live on the value
  datapath (bounded-type defaults, option E's exponent bump). No clock
  expression in the tree uses either.
* The rail NEEDS `lshift`/`rshift`/`bitAnd` — the root clock is
  `sampleIndex <<< 32`, and the phase reductions (`modePhaseQ`,
  `modePhaseQFromIncr`) are split-multiplies over shifts and masks.
* Every crossing FROM float land enters through exactly one door: `toInt`.
  A `toIntE e` node is an OPAQUE integer leaf (the `boundary` constructor)
  — its value is a free variable of the environment, keyed by the source
  subterm. The dual door `toFloatE` is where the rail ENDS (the phasor's
  phase output, the envelope coordinate `dSec` — value-land, not claimed).

So the rail is: integer literals, `sampleIndex`, `toInt` boundaries, and
`{add, sub, mul, neg, <<<k, >>>k, &&&(2^k−1)}` over rail operands — shift
amounts and masks literal, masks a low-bits pattern (the only masks the
datapath spells). The judgment is syntax-directed: each tree shape admits
one derivation, so the denotation is a function of the tree.

`OnClockRail` is `Type`-valued, not `Prop`-valued: `denoteClock` recurses
on the derivation (Prop can't eliminate into `Int` — the handoff's
`Sig → Prop` sketch and its witness-dependent `denoteClock` were in
tension, and the derivation-as-data form is the one that typechecks).

## The trusted base (one named hypothesis)

**CLOCK_RAIL_IS_EXACT**: *per sample, the runtime computes each rail node
as the two's-complement i64 image of its `denoteClock` value, equal
subterms yielding equal values.*

Discharge notes, per op class (inspection, not proof — the execution is
LLVM/MSL):
* `tick` / `boundary`: the kernel is pure `f(τ, params)` and the emitter
  CSEs by subterm — one i64 value per subterm per sample. The env's
  `boundary : Sig → Int` being a FUNCTION is exactly this assumption.
* `add`/`sub`/`neg`/`mul`/`<<<k`: the quotient `ℤ → ℤ/2⁶⁴` is a ring
  homomorphism and `<<<k` is `·2ᵏ`, so any `denoteClock` equality pushes
  forward to bit-identical i64 — WITHOUT a no-overflow side condition.
  (This is why the laws survive wrap: they are ring identities.)
* `>>>k` (ashr) / `&&&(2ᵏ−1)`: floor-shift and low-bits agree with the
  i64 ops on values whose live bits fit 64 — the Q32.32 headroom
  discipline the datapath already documents (`modePhaseQ`'s docstring).

Everything downstream of the clock — that equal i64 clocks render equal
audio — is the kernel's purity, already the whole system's premise.

## What this does NOT replace

The audio-golden law gates in `Tropicaltest/ArrowLaws.lean` gate the
BACKENDS (LLVM/wasm agree with the frozen render); these theorems gate the
FRONT (the algebra is right for every clock expression). Both survive; the
theorems make the per-law-instance golden multiplication unnecessary
(one-per-backend-path suffices), they do not license deleting the backend
check.

Out-of-fragment laws, documented as scoped:
* **Law 3 (cartesian diagonal)** — a program-graph/CSE property (shared
  vs duplicated dry oscillator), not a clock-expression identity; nothing
  here quantifies over program graphs.
* **Law 6 (reverse commutes with the symmetric flanger)** — a VALUE-
  datapath law: the ±δ tap sum reassociates in float
  (`ArrowLaws.lean`'s slice-5 finding). The proof boundary lands exactly
  on the empirical boundary found by ear; a proof that reached law 6
  through this fragment would be a bug in the fragment.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- The environment a clock expression denotes against: the ambient tick
    (what `sampleIndex` reads this sample) and the value of each `toInt`
    boundary crossing, keyed by the SOURCE subterm. The leaf is a free
    variable, not a model — nothing here claims what the runtime feeds it
    (scope decision 4); `boundary` being a function is the determinism
    half of CLOCK_RAIL_IS_EXACT (equal subterms, equal values). -/
structure ClockEnv where
  tick : Int
  boundary : Sig → Int

/-- The integer (Q32.32) rail as a typing judgment: the derivation that a
    `Sig` stays on the fixed datapath. Syntax-directed — one derivation
    per admissible tree — and `Type`-valued so the denotation can recurse
    on it. Shift amounts and masks are LITERAL (`hm` pins the mantissa),
    masks are the low-bits pattern `2^k − 1` — the only masks the
    datapath spells (`& (2³²−1)`, `& 1`). -/
inductive OnClockRail : Sig → Type where
  | intLit (m : Int) : OnClockRail (.num ⟨m, 0⟩)
  | tick : OnClockRail .sampleIndex
  | boundary (e : Sig) : OnClockRail (.unary .toInt e)
  | neg {a : Sig} : OnClockRail a → OnClockRail (.unary .neg a)
  | add {a b : Sig} : OnClockRail a → OnClockRail b → OnClockRail (.binary .add a b)
  | sub {a b : Sig} : OnClockRail a → OnClockRail b → OnClockRail (.binary .sub a b)
  | mul {a b : Sig} : OnClockRail a → OnClockRail b → OnClockRail (.binary .mul a b)
  | lshift {a : Sig} (m : Int) (k : Nat) (hm : m = (k : Int)) :
      OnClockRail a → OnClockRail (.binary .lshift a (.num ⟨m, 0⟩))
  | rshift {a : Sig} (m : Int) (k : Nat) (hm : m = (k : Int)) :
      OnClockRail a → OnClockRail (.binary .rshift a (.num ⟨m, 0⟩))
  | mask {a : Sig} (m : Int) (k : Nat) (hm : m = 2 ^ k - 1) :
      OnClockRail a → OnClockRail (.binary .bitAnd a (.num ⟨m, 0⟩))

/-- The `Int` denotation of a rail expression — TOTAL on the fragment
    (recursion on the derivation; no `Option`, no `Float`). `>>>` is
    `Int.shiftRight` (arithmetic/floor — i64 `ashr`), the mask is `emod`
    (low `k` bits as a nonnegative value — i64 `and` with a low mask). -/
def denoteClock : {s : Sig} → OnClockRail s → ClockEnv → Int
  | _, .intLit m, _ => m
  | _, .tick, env => env.tick
  | _, .boundary e, env => env.boundary e
  | _, .neg h, env => -(denoteClock h env)
  | _, .add ha hb, env => denoteClock ha env + denoteClock hb env
  | _, .sub ha hb, env => denoteClock ha env - denoteClock hb env
  | _, .mul ha hb, env => denoteClock ha env * denoteClock hb env
  | _, .lshift _ k _ ha, env => denoteClock ha env <<< k
  | _, .rshift _ k _ ha, env => denoteClock ha env >>> k
  | _, .mask _ k _ ha, env => denoteClock ha env % 2 ^ k

-- ─────────────────────────────────────────────────────────────
-- The warp laws — for ALL rail expressions, not four renders
-- ─────────────────────────────────────────────────────────────

/-- **Law 1, universally (warp inverse / cancellation).** `(c+δ)−δ = c`
    for EVERY rail clock `c` and EVERY rail offset `δ` — the τ-scrub moat
    as an equation: a forward warp followed by its inverse feeds the
    oscillator the identical integer clock. "Reverse is exact, no
    rounding" stops being a measurement. -/
theorem warp_inv {c δ : Sig} (hc : OnClockRail c) (hδ : OnClockRail δ)
    (env : ClockEnv) :
    denoteClock (.sub (.add hc hδ) hδ) env = denoteClock hc env := by
  simp [denoteClock]

/-- Law 1 with the warps the other way around: `(c−δ)+δ = c`. -/
theorem warp_inv' {c δ : Sig} (hc : OnClockRail c) (hδ : OnClockRail δ)
    (env : ClockEnv) :
    denoteClock (.add (.sub hc hδ) hδ) env = denoteClock hc env := by
  simp [denoteClock]

/-- **Law 2, universally (additive delay / functoriality).**
    `(c−δ₁)−δ₂ = c−(δ₁+δ₂)`: composing delays is delaying by the sum. -/
theorem warp_assoc {c δ₁ δ₂ : Sig} (hc : OnClockRail c)
    (h₁ : OnClockRail δ₁) (h₂ : OnClockRail δ₂) (env : ClockEnv) :
    denoteClock (.sub (.sub hc h₁) h₂) env
      = denoteClock (.sub hc (.add h₁ h₂)) env := by
  simp [denoteClock]; omega

/-- **Law 4, universally (reverse is an involution).** `−(−c) = c`:
    reversing twice is the identity, for every rail clock. -/
theorem rev_involution {c : Sig} (hc : OnClockRail c) (env : ClockEnv) :
    denoteClock (.neg (.neg hc)) env = denoteClock hc env := by
  simp [denoteClock]

/-- **Law 5, universally (reverse conjugates delay).**
    `−(c−δ) = (−c)+δ`: pulling reverse past a delay flips it to an
    advance — the time-mirror algebra the reverse-cue path leans on. -/
theorem rev_swap {c δ : Sig} (hc : OnClockRail c) (hδ : OnClockRail δ)
    (env : ClockEnv) :
    denoteClock (.neg (.sub hc hδ)) env
      = denoteClock (.add (.neg hc) hδ) env := by
  simp [denoteClock]; omega

-- ─────────────────────────────────────────────────────────────
-- Modal's clock construction typechecks against the predicate
-- ─────────────────────────────────────────────────────────────
-- The done-when criterion: the production clock constructions are ON the
-- rail, witnessed with no escape hatch. Each definition below is the
-- derivation for one constructor in `EmitArrow/Modal.lean` /
-- `EmitArrow/Numerics.lean`; that they typecheck IS the check. Every
-- float ingredient (anchor seconds, ω/2π scaling, the rounded period
-- increment) enters through a `boundary` — the rail is carved exactly at
-- `toInt`.

/-- `relClockQ clk anchor = clk − toInt(anchor·2³²)` — rail, for ANY
    anchor expression (the anchor's float math is behind the boundary). -/
def relClockQ_rail {c : Sig} (hc : OnClockRail c) (anchor : Sig) :
    OnClockRail (relClockQ c anchor) :=
  .sub hc (.boundary _)

/-- The split-multiply phase reduction over a supplied increment
    (`modePhaseQFromIncr`): `(incr·(clk>>>32) + (incr·(clk&&&(2³²−1)))>>>32)
    &&& (2³²−1)` — rail whenever the increment and clock are. -/
def modePhaseQFromIncr_rail {incr clkRel : Sig}
    (hi : OnClockRail incr) (hc : OnClockRail clkRel) :
    OnClockRail (modePhaseQFromIncr incr clkRel) :=
  .mask _ 32 (by decide)
    (.add (.mul hi (.rshift _ 32 rfl hc))
          (.rshift _ 32 rfl (.mul hi (.mask _ 32 (by decide) hc))))

/-- `modePhaseQ ω clkRel` — the increment `⌊(ω/2π)·2³²/SR⌋` is a
    boundary (ω may be float, live, anything); the reduction is rail. -/
def modePhaseQ_rail (omega : Sig) {clkRel : Sig} (hc : OnClockRail clkRel) :
    OnClockRail (modePhaseQ omega clkRel) :=
  modePhaseQFromIncr_rail (.boundary _) hc

/-- The PERIODIC relative clock's INTERIOR phase — the split-multiply
    reduction over the rounded period increment and the one-tick-shifted
    relative clock — is rail whenever the ambient clock is: exactly
    `modePhaseQFromIncr` applied to two boundary-guarded rail operands. -/
def relClockQuotPhase_rail {c : Sig} (hc : OnClockRail c)
    (anchor : Sig) (pSec : Float) :
    OnClockRail (modePhaseQFromIncr
      (toIntE (roundE (div (lit 4294967296) (mul (litF pSec) .sampleRate))))
      (sub (relClockQ c anchor) (litI 1))) :=
  modePhaseQFromIncr_rail (.boundary _) (.sub (relClockQ_rail hc anchor) (.boundary _))

/-- The PERIODIC relative clock `relClockQuot` as a whole is **rail-in,
    boundary-out** — a carving finding, not a defect: the quotient
    re-expands its masked phase to ticks THROUGH FLOAT
    (`toInt(toFloat(phase)·P·SR)`), so the outermost expression is
    `add boundary boundary` and needs NO hypothesis on the ambient clock.
    The exactness story lives in the interior (`relClockQuotPhase_rail`)
    plus one documented float crossing (the ~2e-14 s sub-quantization lag
    in `relClockQuot`'s own docstring). The rail predicate finds the seam
    exactly where the code documented it. -/
def relClockQuot_rail (c anchor : Sig) (pSec : Float) :
    OnClockRail (relClockQuot c anchor pSec) :=
  .add (.boundary _) (.boundary _)

/-- The root clock `clockLit = sampleIndex <<< 32` (the session `clk`
    default, every closed-form carrier's clock) — rail off the tick. -/
def rootClock_rail : OnClockRail clockLit :=
  .lshift _ 32 rfl .tick

/-- The phasor accumulator `inc·(clk>>>32) + (inc·(clk&&&(2³²−1)))>>>32 + off`
    (`phasorPhaseSig`'s integer core) — rail; the phasor's OUTPUT then
    crosses `toFloatE` and leaves the rail (the dual door, where the value
    datapath begins). Witness for the accumulator shape. -/
def phasorAcc_rail {clk : Sig} (hc : OnClockRail clk) (freqE offsetE : Sig) :
    OnClockRail (.binary .add
      (.binary .add
        (.binary .mul (toIntE (div (mul freqE (lit 4294967296)) .sampleRate))
          (.binary .rshift clk (.num ⟨32, 0⟩)))
        (.binary .rshift
          (.binary .mul (toIntE (div (mul freqE (lit 4294967296)) .sampleRate))
            (.binary .bitAnd clk (.num ⟨4294967295, 0⟩)))
          (.num ⟨32, 0⟩)))
      (toIntE (mul offsetE (lit 4294967296)))) :=
  .add (.add (.mul (.boundary _) (.rshift _ 32 rfl hc))
             (.rshift _ 32 rfl (.mul (.boundary _) (.mask _ 32 (by decide) hc))))
       (.boundary _)

/-- The split identity the phase reduction's exactness rests on
    (`modePhaseQ`'s docstring: "`clkRel = (clkRel>>32)·2³² +
    (clkRel & (2³²−1))` holds exactly on NEGATIVE relative clocks") —
    true of the DENOTATION for every `Int`, negative included: floor-shift
    and low-bits recompose exactly. The docstring's claim, as arithmetic. -/
theorem rail_split_identity (x : Int) :
    (x >>> (32 : Nat)) * 2 ^ 32 + x % 2 ^ 32 = x := by
  rw [Int.shiftRight_eq_div_pow]
  omega

end Tropical.EmitArrow
