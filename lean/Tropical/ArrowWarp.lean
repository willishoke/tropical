import Tropical.Ir.Nodes

/-!
# ArrowWarp — vertical slice 1 (the gated flanger)

The first vertebra of the strangler-fig rearchitecture (see
`design/arrowwarp-slice-1.md`). ArrowWarp's *only* job is to **build a
resolved `Program`** — the same pre-strata IR the elaborator hands the
strata pipeline — from a tiny arrow-combinator API. Everything below
(strata → `Core.check` → `compileResolved` → wire) is reused verbatim.

The slice is gated against the hand-written `FlangeSin`: the plan emitted
from the ArrowWarp-built program must be **byte-identical** to
`diffcli emit-file stdlib/parsed/FlangeSin.json`. Byte-identity falls out
for free because the `osc` primitive does NOT hand-roll the phasor +
Horner polynomial — it references the elaborated `FixedSinOsc` body and
lets strata inline it (the M0 finding). So ArrowWarp only ever builds the
*coarse* graph: three oscillators at three warped clocks, weighted-summed.

The combinator surface (float-only, slice 1):

* `lit` / `mul` / `add` / `sub` / `toIntE` — `arr`-level pointwise arithmetic.
* `Warp` (`id` / `back δ` / `fwd δ`) + `Warp.apply` — clock arithmetic, the
  `warp` morphism. Modulated warp (`δ` a function of the clock/params, never
  the flowing data) is lawful — that is also the musical semantics.
* `Builder.osc` — the primitive morphism `Clock ⇝ Sig`, realized as a
  `FixedSinOsc` instance declaration (`freq→port0, clk→port1`) whose `sine`
  output is the returned signal.
* `flangerBody` — the whole flanger as one expression:
  `(warp id &&& warp(−δ) &&& warp(+δ)) >>> oscₓ3 >>> weightedSum`. The shared
  `clkIn` fanned three ways *is* the diagonal `&&&`.
* `buildFlanger` — assemble the `Program` (signature `clk:int / freq:float /
  depth:float → out:float`), merge the registry like the elaborator, and push
  it into a supplied arena (the one `elabChain` filled with `FixedSinOsc`).
-/

namespace Tropical.ArrowWarp

open Lean (JsonNumber)
open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- M1 — builder substrate: smart constructors over the resolved `Expr`
-- ─────────────────────────────────────────────────────────────

/-- A float signal expression (the wires of the float-only arrow layer). -/
abbrev Sig := Expr
/-- A clock-as-value expression (Q32.32 fixed-point sample coordinate). -/
abbrev Clock := Expr

/-- Decimal literal `mantissa · 10^(-exponent)`. The exponent defaults to 0
    (an integer). This mirrors the `JsonNumber` the JSON parser produces, so
    the emitted bytes match: `lit 5 1 = 0.5`, `lit 25 2 = 0.25`, `lit 7 4 =
    0.0007`, `lit 4294967296 = 2³²`. -/
def lit (mantissa : Int) (exponent : Nat := 0) : Expr := .num ⟨mantissa, exponent⟩

/-- `arr`-level pointwise multiply. -/
def mul (a b : Expr) : Expr := .binary .mul a b
/-- `arr`-level pointwise add. -/
def add (a b : Expr) : Expr := .binary .add a b
/-- `arr`-level pointwise subtract. -/
def sub (a b : Expr) : Expr := .binary .sub a b
/-- `arr`-level truncate-to-int (the Q32.32 offset is an integer sample count). -/
def toIntE (a : Expr) : Expr := .unary .toInt a

/-! The flanger's program signature, as port references.
    `clk : int` (input 0), `freq : float` (input 1), `depth : float` (input 2). -/

/-- The clock input (`clk : int`), as a value. -/
def clkIn : Clock := .inputRef ⟨0⟩
/-- The frequency input (`freq : float`). -/
def freqIn : Sig := .inputRef ⟨1⟩
/-- The flange-depth input (`depth : float`). -/
def depthIn : Sig := .inputRef ⟨2⟩

-- ─────────────────────────────────────────────────────────────
-- M2 — `warp`: clock arithmetic
-- ─────────────────────────────────────────────────────────────

/-- A static/clock-modulated time warp — a function on the clock value.
    `id` leaves the clock; `back d` / `fwd d` translate it by `∓d` Q32.32
    samples (the flanger's past/ahead taps). The two directions are kept
    distinct (rather than `fwd (neg d)`) so the built IR matches the source
    flanger's `clk − δ` / `clk + δ` exactly. -/
inductive Warp where
  | id
  | back (d : Expr)
  | fwd (d : Expr)
deriving Repr

/-- Denotation `⟦warp φ⟧ s = s ∘ φ`, here on the clock-as-value. -/
def Warp.apply : Warp → Clock → Clock
  | .id,     c => c
  | .back d, c => sub c d
  | .fwd d,  c => add c d

/-- δ = `toInt(depth · sampleRate · 2³²)` — the flange offset, depth seconds
    expressed in Q32.32 samples. A function of the clock/params, so warping by
    it is a lawful arrow. -/
def deltaSamples : Expr := toIntE (mul (mul depthIn .sampleRate) (lit 4294967296))

-- ─────────────────────────────────────────────────────────────
-- The `osc` primitive — `Clock ⇝ Sig`, sourced from the stdlib closed-form
-- ─────────────────────────────────────────────────────────────

/-- Accumulates the instance declarations the `osc` morphisms emit. The
    `InstanceIdx` of a declared oscillator is its position here, which is what
    `nestedOut` reads back. -/
structure Builder where
  decls : Array BodyDecl := #[]
deriving Inhabited

/-- `osc` — the primitive morphism `Clock ⇝ Sig`. Realized by referencing the
    elaborated `FixedSinOsc` body (NOT a hand-rolled phasor+polynomial): emit a
    `FixedSinOsc` instance wired `freq→port0, clk→port1` (the source-order the
    elaborator records) and return its `sine` output (`OutputIdx 0`). Strata
    inlines the closed form, which is what buys byte-identity. -/
def Builder.osc (b : Builder) (name : String) (freqE : Sig) (clkE : Clock) :
    Sig × Builder :=
  let inst : BodyDecl := .inst name "FixedSinOsc" #[]
    #[ { port := ⟨0⟩, value := freqE }, { port := ⟨1⟩, value := clkE } ]
  let i := b.decls.size
  (.nestedOut ⟨i⟩ ⟨0⟩, { b with decls := b.decls.push inst })

/-- The flanger as one arrow expression:
    `(warp id &&& warp(−δ) &&& warp(+δ)) >>> oscₓ3 >>> weightedSum`.
    The shared `clkIn` fanned three ways is the diagonal `&&&`; the three
    `osc`s are the parallel oscillator bank; the weighted sum
    `0.5·dry + 0.25·past + 0.25·ahead` is the collapsing `arr`. -/
def flangerBody : Builder × Sig :=
  let b0 : Builder := {}
  let (dry,   b1) := b0.osc "dry"   freqIn (Warp.id.apply clkIn)
  let (past,  b2) := b1.osc "past"  freqIn ((Warp.back deltaSamples).apply clkIn)
  let (ahead, b3) := b2.osc "ahead" freqIn ((Warp.fwd  deltaSamples).apply clkIn)
  let out := add (add (mul (lit 5 1) dry) (mul (lit 25 2) past)) (mul (lit 25 2) ahead)
  (b3, out)

-- ─────────────────────────────────────────────────────────────
-- M3 — assemble the resolved `Program` and push it into the arena
-- ─────────────────────────────────────────────────────────────

/-- `clk : int`, default `clock() = sampleIndex << 32`. -/
def clkInputDecl : InputDecl :=
  { name := "clk", type? := some (.scalar .int),
    default? := some (.binary .lshift .sampleIndex (lit 32)) }

/-- `freq : float`, default `select(220 > 0, 220, 0)` (the elaborated default). -/
def freqInputDecl : InputDecl :=
  { name := "freq", type? := some (.scalar .float),
    default? := some (.select (.binary .gt (lit 220) (lit 0)) (lit 220) (lit 0)) }

/-- `depth : float`, default `0.0007`. -/
def depthInputDecl : InputDecl :=
  { name := "depth", type? := some (.scalar .float),
    default? := some (lit 7 4) }

/-- Build the ArrowWarp flanger `Program` into `arena`, returning its
    `ProgramIdx`. `resolved` is the name→idx map `elabChain` produces; the only
    program ArrowWarp links against is `FixedSinOsc` (its registry — and the
    transitive `ClockPhasor`/`Sin` merge — mirrors the elaborator's
    `registerInstanceDecl`). -/
def buildFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some fsIdx := (resolved.find? (·.1 == "FixedSinOsc")).map (·.2)
    | .error "ArrowWarp: FixedSinOsc not found in the elaborated stdlib chain"
  let some fsProg := arena.program? fsIdx
    | .error "ArrowWarp: FixedSinOsc program index out of range"
  -- Transitive registry merge (mirrors `registerInstanceDecl`): the target
  -- under its program name, then the target's own registry entries in order,
  -- skipping keys already present. Insertion order is codec-observable.
  let mut registry : Array (String × ProgramIdx) := #[(fsProg.name, fsIdx)]
  for (k, v) in fsProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, v)
  let (b, outExpr) := flangerBody
  let prog : Program := {
    name := "FlangeSin"
    inputs := #[clkInputDecl, freqInputDecl, depthInputDecl]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := outExpr }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

end Tropical.ArrowWarp
