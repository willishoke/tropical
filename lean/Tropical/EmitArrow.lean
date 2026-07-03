import Tropical.Ir.Nodes

/-!
# EmitArrow — realization-by-emission of the post-strata (scalar) IR

`EmitArrow` (formerly `ArrowWarp`) is the combinator library that **builds the
resolved IR `Program`** directly in the post-strata, scalar shape and reuses the
backend (`elaborate`-linked, then strata/`compileResolved`) to emit. It is named
verb-first: it is a *realization by emission* of the existing scalar IR, not a
new arrow — the "warp" combinators (`warpBank`, `Warp`-as-clock-expression) are
the clock axis of that realization and keep their names.

The post-strata IR is **scalar by definition** (strata's job is to lower arrays,
sums and generics away), so `EmitArrow` stays scalar: `Sig := Expr`. It needs no
richer types — the richness lives in the typed elaborator upstream.

Slice 1 built the `FlangeSin` flanger from a tiny arrow-combinator API and
gated it byte-identical against `diffcli emit-file stdlib/parsed/FlangeSin.json`.
Slice 2 proves the **pointfree-genericity** thesis: the flanger is a *combinator
generic over its voice*. The same `warpBank` morphism, instantiated at two
different voices, reproduces two different hand-written stdlib programs
byte-for-byte:

* `FixedSinOsc` voice → `FlangeSin`     (the slice-1 regression gate)
* `ModalVoice`  voice → `ReversibleComb`

Both targets are the SAME flanger *shape* — three voice taps at
`clk / clk−δ / clk+δ`, weighted-summed — over a different voice. What differs
between them is captured entirely by parameters, never by a second builder:

| differs            | FlangeSin (FixedSinOsc) | ReversibleComb (ModalVoice) |
|--------------------|-------------------------|-----------------------------|
| voice program      | `FixedSinOsc`           | `ModalVoice`                |
| instance wiring    | `freq→0, clk→1`         | `clk→0, f0→1`               |
| pitch input name   | `freq` (default 220)    | `f0` (default 110)          |
| offset input name  | `depth`                 | `delta`                     |

The weights `{0.5, 0.25, 0.25}`, the tap warps `{id, −δ, +δ}`, and the offset
expression `δ = toInt(depth·sampleRate·2³²)` are identical across both — so they
live in the *one* shared `warpBank` combinator, not in either instantiation.

Byte-identity still falls out for free because `osc` does NOT hand-roll the
voice — it references the elaborated voice body (`FixedSinOsc` / `ModalVoice`)
and lets strata inline it (the M0 finding). EmitArrow only ever builds the
*coarse* graph: a bank of voice taps at warped clocks, weighted-summed.

The combinator surface:

* `lit` / `mul` / `add` / `sub` / `neg` / `toIntE` — the operation set, applied
  to values OR to the clock (one algebra). A "warp" is just a clock expression:
  `reverse = neg clk`, `delay δ = sub clk δ`, modulated `= sub clk (m clk)`.
  There is no `Warp` type and no separate clock arithmetic.
* `Voice` — the primitive morphism `Clock ⇝ Sig`, made GENERIC over the stdlib
  program that realizes it (its name, how its instance inputs wire from a
  warped clock, and which output carries the signal).
* `Builder.osc` — emit one `Voice` instance at a warped clock and read its
  signal back.
* `Tap` / `warpBank` — the voice-generic flanger: an array of `(name, warp,
  weight)` taps over a single `Voice`, fanned from the shared `clkIn`
  (the diagonal `&&&`), weighted-summed (the collapsing `arr`).
* `WarpBankProgram` / `buildWarpBank` — wrap the body in a `Program` (signature
  `clk:int / pitch:float / offset:float → out:float`) and push it into the arena
  the elaborator filled, linking the voice by name like `registerInstanceDecl`.
-/

namespace Tropical.EmitArrow

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
/-- Negate — the negate EFFECT (`.unary .neg`, emitted `sub i64 0, c`). The SAME
    operation whether applied to a signal value or to the clock; applied to the
    clock it IS `reverse`. There is one algebra of operations on expressions; the
    clock is one of those expressions. -/
def neg (a : Expr) : Expr := .unary .neg a

/-! More of the universal scalar op set — plain `.binary`/`.unary`/`.clamp`
    wrappers, staying scalar (the post-strata IR is scalar by definition). These
    are what a generator/closed-form voice (phasor arithmetic + polynomial)
    needs beyond the flanger's add/sub/mul: division, the bit ops the fixed-point
    phasor speaks, rounding, the int⇆float casts, and the bounded-type clamp the
    elaborator inserts for `unipolar`/`freq` ports. -/

/-- `arr`-level pointwise divide. -/
def div (a b : Expr) : Expr := .binary .div a b
/-- Bitwise AND on the integer (fixed-point) clock/value (`& (2³²−1)` masks). -/
def bitAnd (a b : Expr) : Expr := .binary .bitAnd a b
/-- Logical/arithmetic right shift (`clk >> 32` etc.; the mask makes the choice
    irrelevant where it follows a `& (2³²−1)`). -/
def rshift (a b : Expr) : Expr := .binary .rshift a b
/-- Left shift (`sampleIndex << 32` — the root clock). -/
def lshift (a b : Expr) : Expr := .binary .lshift a b
/-- `gt` comparison (the bounded-default `select(hz > 0, …)`). -/
def gt (a b : Expr) : Expr := .binary .gt a b
/-- Round to nearest integer-as-float (`Sin`'s half-cycle count `n`). -/
def roundE (a : Expr) : Expr := .unary .round a
/-- Reinterpret int → float (`toFloat(acc & mask)` at the phasor boundary). -/
def toFloatE (a : Expr) : Expr := .unary .toFloat a
/-- Clamp into `[lo, hi]` — the elaborator's lowering of a bounded port type
    (`unipolar` ⇒ `clamp _ 0 1`). -/
def clampE (value lo hi : Expr) : Expr := .clamp value lo hi
/-- Select (`cond ? then : else`) — the bounded-default `select(hz > 0, hz, 0)`. -/
def selectE (cond then_ else_ : Expr) : Expr := .select cond then_ else_

/-! The warp-bank program signature, as port references. The signature is the
    same shape for every voice — only the input NAMES/defaults differ (which is
    a declaration concern, not a wiring one), so the value-level refs are shared:
    `clk : int` (input 0), `pitch : float` (input 1), `offset : float` (input 2). -/

/-- The clock input (`clk : int`), as a value. -/
def clkIn : Clock := .inputRef ⟨0⟩
/-- The pitch input (`freq`/`f0` : float) — input 1 regardless of its name. -/
def pitchIn : Sig := .inputRef ⟨1⟩
/-- The flange-offset input (`depth`/`delta` : float) — input 2. -/
def offsetIn : Sig := .inputRef ⟨2⟩

-- ─────────────────────────────────────────────────────────────
-- M2 — a "warp" is just a clock EXPRESSION (no separate algebra)
-- ─────────────────────────────────────────────────────────────

/-! There is no `Warp` type. The clock is a first-class expression and a warp is
    any operation applied to it, drawn from the SAME operation set used on values
    — one algebra, not a clock algebra distinct from the value algebra:

    * `reverse  = neg clk`          (the same `neg` that negates a signal)
    * `delay δ  = sub clk δ`
    * `advance δ = add clk δ`
    * `timescale k = mul clk k`
    * `modulated δ(t) = sub clk (m clk)`  for any signal `m`  (i.e. PM)

    all just clock expressions. No constructor enumerates them and none can be
    left out; invertibility never enters (you only ever apply operations, never
    invert — `reverse` is `neg`, not an inverse). A combinator generic over
    "which warp" takes a `Clock → Clock` (an operation on the clock), e.g.
    `fun c => sub c δ`. -/

/-- δ = `toInt(offset · sampleRate · 2³²)` — the flange offset, `offset` seconds
    expressed in Q32.32 samples. A function of the clock/params (`offsetIn` is
    input 2), so warping by it is a lawful arrow. Shared across voices: both
    targets carry the identical offset expression (only its input *name* —
    `depth` vs `delta` — differs, and names aren't referenced here). -/
def deltaSamples : Expr := toIntE (mul (mul offsetIn .sampleRate) (lit 4294967296))

-- ─────────────────────────────────────────────────────────────
-- The `Voice` primitive — `Clock ⇝ Sig`, generic over the stdlib closed-form
-- ─────────────────────────────────────────────────────────────

/-- A **voice** is the primitive morphism `Clock ⇝ Sig`, made generic over the
    stdlib program that realizes it. `osc` does NOT hand-roll a phasor +
    polynomial — it references the elaborated voice body and lets strata inline
    it (the M0 finding), which is what buys byte-identity.

    * `programName` — the stdlib program (`FixedSinOsc`, `ModalVoice`, …).
    * `wire` — how to fill the voice instance's inputs from the (warped) clock.
      This captures the per-voice port ORDER and which port is the clock vs the
      pitch (`FixedSinOsc`: `freq→0, clk→1`; `ModalVoice`: `clk→0, f0→1`).
    * `output` — which voice output carries the signal (both voices: index 0). -/
structure Voice where
  programName : String
  wire : Clock → Array InstanceInput
  output : OutputIdx := ⟨0⟩
  /-- The phase-anchor hook (the slide in the phase domain). When set to
      `(phasePort, corr)`, the SLIDE (`emitTermC`) adds `corr shift` to this voice's
      `phasePort` input for each warped copy, where `shift = clk − warpedClk` is the
      clock shift the slide pushed onto that copy. Since phase = inc·clk, a clock
      shift maps to a phase shift via `inc`, so threading the anchor across the
      cartesian copies keeps every warped read (delay/flange tap) phase-continuous
      across a live freq change — not just the un-warped read. `none` = no
      correction (byte-gated voices are untouched). -/
  phaseAnchor : Option (InputIdx × (Clock → Sig)) := none

/-- The `FixedSinOsc` voice — pitch at port 0, clock at port 1 (source order). -/
def fixedSinOscVoice : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, pitchIn⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- The `ModalVoice` voice — clock at port 0, pitch at port 1 (source order). -/
def modalVoice : Voice :=
  { programName := "ModalVoice"
    wire := fun clkE => #[ ⟨⟨0⟩, clkE⟩, ⟨⟨1⟩, pitchIn⟩ ] }

/-- Accumulates the instance declarations the `osc` morphisms emit. The
    `InstanceIdx` of a declared voice is its position here, which is what the
    summed output reads back. -/
structure Builder where
  decls : Array BodyDecl := #[]
deriving Inhabited

/-- `osc` — emit one `Voice` instance named `name`, wired from the warped clock
    `clkE`, and return its signal output (`nestedOut`). Generic over the voice:
    the program name, the input wiring, and the output index all come from `v`. -/
def Builder.osc (b : Builder) (v : Voice) (name : String) (clkE : Clock) :
    Sig × Builder :=
  let inst : BodyDecl := .inst name v.programName #[] (v.wire clkE)
  let i := b.decls.size
  (.nestedOut ⟨i⟩ v.output, { b with decls := b.decls.push inst })

-- ─────────────────────────────────────────────────────────────
-- PRODUCTS / MIMO — the cartesian combinator surface (the DATA axis)
-- ─────────────────────────────────────────────────────────────

/-! `warp` is the CLOCK axis; this is the orthogonal DATA axis. Post-strata, a
    "value" is not one scalar but a *bundle* of scalar wires — a categorical
    PRODUCT. A morphism `A ⇝ B` is then a function from a wire-bundle to a
    wire-bundle that accretes the instance decls it sources along the way:

      `Mor := tuple of input wires → Builder → (tuple of output wires, Builder)`

    The structure is **cartesian, not closed**: it has products (concatenation
    of bundles), the diagonal (duplication), and composition — but no
    exponentials, no closures, no runtime higher-order programs. Every
    combinator is a SMART CONSTRUCTOR that emits the post-strata scalar DAG at
    build time; none survives to run time. Crucially `⋙` (composition) **is**
    `inlineInstances`: `g ⋙ f` feeds f's output wires straight into g, building
    one flattened DAG with no instance boundary between them — the combinator
    layer's image of the strata pass it absorbs. (Hughes' `first` is the whole
    of MIMO; the named-port ⟷ tuple bridge — `instMor` — is its primitive.) -/

/-- A cartesian morphism in the wire category: consumes a product of input
    wires, accretes instance decls, produces a product of output wires. Objects
    are wire-bundle ARITIES (the splitter combinators carry them explicitly,
    since the category is untyped-but-arity-indexed). -/
abbrev Mor := Array Sig → Builder → (Array Sig × Builder)

/-- Identity (`arr id`) — passes its whole bundle through untouched. Absorbed
    by `identityElim` at construction: `idMor ⋙ f = f` holds definitionally. -/
def idMor : Mor := fun xs b => (xs, b)

/-- Sequential composition `g ⋙ f` (read left-to-right): the LEFT runs first and
    its outputs feed the right. THIS is inlining — there is no instance boundary
    between the two, just one threaded DAG (the absorbed `inlineInstances`). -/
def seq (f g : Mor) : Mor := fun xs b => let (ys, b) := f xs b; g ys b

/-- Fan-out / the cartesian diagonal `f &&& g`: run BOTH on the same input,
    concatenating their outputs. The Builder threads f-then-g, so the sourced
    instances get deterministic indices. The shared input bundle is the diagonal
    `Δ` — at the Lean level it is plain value reuse, which is exactly the
    post-strata DAG's shared sub-node. -/
def fan (f g : Mor) : Mor := fun xs b =>
  let (ys, b) := f xs b
  let (zs, b) := g xs b
  (ys ++ zs, b)

/-- Parallel product `f *** g`, split at arity `m`: `f` consumes the first `m`
    wires, `g` the rest; outputs concatenate. The bifunctor on products. -/
def par (m : Nat) (f g : Mor) : Mor := fun xs b =>
  let (ys, b) := f (xs.extract 0 m) b
  let (zs, b) := g (xs.extract m xs.size) b
  (ys ++ zs, b)

/-- `first f` (Hughes): apply `f` to the first `m` wires, pass the remaining
    bundle through unchanged. `first m f = par m f idMor`. The single primitive
    from which all of MIMO is generated. -/
def first (m : Nat) (f : Mor) : Mor := par m f idMor

/-- `second g`: dual of `first` — pass the first `m` wires through, apply `g` to
    the rest. -/
def second (m : Nat) (g : Mor) : Mor := par m idMor g

/-- The diagonal `Δ : A ⇝ A × A` — duplicate the whole bundle. -/
def dup : Mor := fun xs b => (xs ++ xs, b)

/-- Left projection `π₁` — keep the first `m` wires, drop the rest. -/
def exl (m : Nat) : Mor := fun xs b => (xs.extract 0 m, b)

/-- Right projection `π₂` — drop the first `m` wires, keep the rest. -/
def exr (m : Nat) : Mor := fun xs b => (xs.extract m xs.size, b)

/-- Lift a pure wire-bundle function into a morphism (`arr`, RESTRICTED): no
    instances, just structural/arithmetic rewiring of scalar `Expr`s. This is
    the `arr` of a cartesian (not closed) arrow — a fixed structural map, never
    an arbitrary host closure. -/
def arrMor (f : Array Sig → Array Sig) : Mor := fun xs b => (f xs, b)

/-- THE NAMED-PORT ⟷ TUPLE BRIDGE — the MIMO primitive. Instantiate program
    `programName` (a named multi-port morphism `A ⇝ B`) by assigning the input
    wire bundle to ports `portOrder` positionally, reading its `numOut` outputs
    back as a bundle. The categorical content: a named multi-port instance IS a
    morphism between products; this bridge is the iso between its named (record)
    presentation and the positional (tuple) one. Emits a COARSE instance — `⋙`
    and strata's `inlineInstances` flatten it; the combinator surface is what
    the cutover keeps. -/
def instMor (name programName : String) (portOrder : Array InputIdx)
    (numOut : Nat) : Mor := fun args b =>
  let inputs : Array InstanceInput :=
    (portOrder.zip args).map (fun (p, v) => ⟨p, v⟩)
  let i := b.decls.size
  let inst : BodyDecl := .inst name programName #[] inputs
  let outs : Array Sig := (Array.range numOut).map (fun o => .nestedOut ⟨i⟩ ⟨o⟩)
  (outs, { b with decls := b.decls.push inst })

-- ─────────────────────────────────────────────────────────────
-- `warpBank` — the voice-generic flanger combinator
-- ─────────────────────────────────────────────────────────────

/-- One tap of a warp bank: a named voice instance at a warped clock, scaled by
    `weight` in the final sum. -/
structure Tap where
  name : String
  warp : Clock → Clock
  weight : Sig

/-- The voice-generic warp bank as one arrow expression:
    `(warp φ₀ &&& warp φ₁ &&& …) >>> voiceₓn >>> weightedSum`.
    The shared `clkIn` fanned `n` ways is the diagonal `&&&`; each tap is one
    `voice` instance at its warped clock; the weighted sum is the collapsing
    `arr`. Generic over the voice — the SAME combinator builds `FlangeSin` and
    `ReversibleComb`, differing only in the `Voice` and `Tap`s supplied.

    The sum is left-associated (`((w₀·t₀ + w₁·t₁) + w₂·t₂)`) to match the
    source programs' `add(add(_, _), _)` nesting exactly. -/
def warpBank (v : Voice) (taps : Array Tap) : Builder × Sig := Id.run do
  let mut b : Builder := {}
  let mut summands : Array Sig := #[]
  for tap in taps do
    let (sig, b') := b.osc v tap.name (tap.warp clkIn)
    b := b'
    summands := summands.push (mul tap.weight sig)
  let out := match summands[0]? with
    | none => lit 0
    | some s0 => (summands.extract 1 summands.size).foldl add s0
  (b, out)

/-- The flanger taps shared by both targets: dry (`id`, 0.5) plus the two
    delayed taps (`−δ` / `+δ`, 0.25 each). `δ = deltaSamples` references the
    offset input by index, so the same taps serve every voice. -/
def flangerTaps : Array Tap := #[
  { name := "dry",   warp := fun c => c,                 weight := lit 5 1 },
  { name := "past",  warp := fun c => sub c deltaSamples, weight := lit 25 2 },
  { name := "ahead", warp := fun c => add c deltaSamples, weight := lit 25 2 } ]

-- ─────────────────────────────────────────────────────────────
-- M3 — assemble the resolved `Program` and push it into the arena
-- ─────────────────────────────────────────────────────────────

/-- `clk : int`, default `clock() = sampleIndex << 32`. Shared by every voice. -/
def clkInputDecl : InputDecl :=
  { name := "clk", type? := some (.scalar .int),
    default? := some (.binary .lshift .sampleIndex (lit 32)) }

/-- A pitch input (`freq`/`f0` : float), default `select(hz > 0, hz, 0)` — the
    elaborated form of the source's `select(220>0, 220, 0)` / `…110…`. -/
def pitchInputDecl (name : String) (hz : Int) : InputDecl :=
  { name, type? := some (.scalar .float),
    default? := some (.select (.binary .gt (lit hz) (lit 0)) (lit hz) (lit 0)) }

/-- An offset input (`depth`/`delta` : float), default `0.0007`. -/
def offsetInputDecl (name : String) : InputDecl :=
  { name, type? := some (.scalar .float), default? := some (lit 7 4) }

/-- A full warp-bank program: the program name, the voice it is generic over,
    its three input declarations (`clk`, pitch, offset), and the taps. Two
    instantiations of this record — `flangeSinSpec` and `reversibleCombSpec` —
    are the whole of the per-target difference. -/
structure WarpBankProgram where
  name : String
  voice : Voice
  inputs : Array InputDecl
  taps : Array Tap

/-- Build a `WarpBankProgram`'s `Program` into `arena`, returning its
    `ProgramIdx`. `resolved` is the name→idx map `elabChain` produces; the only
    program linked against is `spec.voice.programName` — its registry (and the
    transitive merge of the voice's own entries) mirrors the elaborator's
    `registerInstanceDecl`. Insertion order is codec-observable. -/
def buildWarpBank (spec : WarpBankProgram) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == spec.voice.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{spec.voice.programName}' not found in the \
        elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{spec.voice.programName}' program index out of range"
  -- Transitive registry merge (mirrors `registerInstanceDecl`): the voice under
  -- its program name, then the voice's own registry entries in order, skipping
  -- keys already present.
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, v) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, v)
  let (b, outExpr) := warpBank spec.voice spec.taps
  let prog : Program := {
    name := spec.name
    inputs := spec.inputs
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := outExpr }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

-- ─────────────────────────────────────────────────────────────
-- The two instantiations — one combinator, two voices, two programs
-- ─────────────────────────────────────────────────────────────

/-- `FlangeSin` over the `FixedSinOsc` voice (the slice-1 regression gate). -/
def flangeSinSpec : WarpBankProgram :=
  { name := "FlangeSin", voice := fixedSinOscVoice
    inputs := #[clkInputDecl, pitchInputDecl "freq" 220, offsetInputDecl "depth"]
    taps := flangerTaps }

/-- `ReversibleComb` over the `ModalVoice` voice (the slice-2 gate). -/
def reversibleCombSpec : WarpBankProgram :=
  { name := "ReversibleComb", voice := modalVoice
    inputs := #[clkInputDecl, pitchInputDecl "f0" 110, offsetInputDecl "delta"]
    taps := flangerTaps }

/-- Build the EmitArrow `FlangeSin` (FixedSinOsc voice) — the slice-1 gate.
    Byte-identical to `diffcli emit-file stdlib/parsed/FlangeSin.json`. -/
def buildFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildWarpBank flangeSinSpec arena resolved

/-- Build the EmitArrow `ReversibleComb` (ModalVoice voice) — the slice-2 gate.
    Byte-identical to `diffcli emit-file stdlib/parsed/ReversibleComb.json`. -/
def buildReversibleComb (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildWarpBank reversibleCombSpec arena resolved

-- ─────────────────────────────────────────────────────────────
-- C1 — build the FOUNDATIONAL VOICE directly: `FixedSinOsc` from scratch
-- ─────────────────────────────────────────────────────────────

/-! The flanger family above *sources* `FixedSinOsc` as a `Voice` instance and
    leans on strata's `inlineInstances` to flatten it. The cutover wants
    EmitArrow to build the voice ITSELF — no instance boundary, the per-program
    path's one flat DAG. `buildFixedSinOsc` does exactly that: it reconstructs
    the post-strata (scalar, inlined) `FixedSinOsc` body — the `FixedPhasor`
    fixed-point phase (integer split-multiply on the Q32.32 clock) composed with
    the `Sin` polynomial (Payne–Hanek reduction + degree-11 Horner) — entirely
    from the smart constructors, then emits it.

    Two things make this byte-identical to `diffcli emit-stdlib FixedSinOsc`:

    * The post-strata form is a single inlined `Expr` tree (instances inlined,
      the `Sin` `fold` unrolled, the `let`s flattened). EmitArrow's Lean-level
      value reuse (`phase`, `x`, `n`, `r`, `r2` bound once and shared) yields the
      STRUCTURALLY-identical tree — the shared subterms appear duplicated exactly
      as the unrolled post-strata tree does — so `compileResolved`'s
      value-numbering produces the identical instruction stream.
    * The literals carry the same `JsonNumber` mantissa/exponent the surface
      parser produced (e.g. `6.283185307179586 = 6283185307179586·10⁻¹⁵`,
      `−2.505210838544172e-8 = −2505210838544172·10⁻²³`).

    Notes on the op set this exercises beyond the flanger: `div`, `bitAnd`,
    `rshift`, `lshift`, `round`, `toFloat`, and the `clamp` the elaborator emits
    for the `unipolar` bound (`ClockPhasor.offset` ⇒ `clamp _ 0 1`, and the
    `phase` output bound likewise). All plain scalar ops — no richer types. -/
def buildFixedSinOsc (arena : Arena) (_resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let freqIn : Sig := .inputRef ⟨0⟩
  let clk : Clock := .inputRef ⟨1⟩
  let twoPow32 := lit 4294967296
  let mask := lit 4294967295
  -- FixedPhasor (= ClockPhasor at clk): the integer split-multiply, exact on ℤ/2³².
  let inc := toIntE (div (mul freqIn twoPow32) .sampleRate)
  let thi := rshift clk (lit 32)
  let tlo := bitAnd clk mask
  let off := toIntE (mul (.inputRef ⟨2⟩) twoPow32)   -- offset = the `phase` input (port 2)
  let acc := add (add (mul inc thi) (rshift (mul inc tlo) (lit 32))) off
  let phase := clampE (div (toFloatE (bitAnd acc mask)) twoPow32) (lit 0) (lit 1)
  -- Phase → [0, 2π).
  let x := mul (lit 6283185307179586 15) phase
  -- Sin: Payne–Hanek reduction + degree-11 Horner on r² (fold unrolled).
  let n := roundE (mul x (lit 3183098861837907 16))
  let oddN := bitAnd n (lit 1)
  let sign := sub (lit 1) (mul (lit 2) oddN)
  let r := sub x (mul n (lit 3141592653589793 15))
  let r2 := mul r r
  let poly :=
    add (lit 1)
     (mul (add (lit (-16666666666666666) 17)
       (mul (add (lit 8333333333333333 18)
         (mul (add (lit (-1984126984126984) 19)
           (mul (add (lit 27557319223985893 22)
             (mul (add (lit (-2505210838544172) 23)
               (mul (lit 0) r2)) r2)) r2)) r2)) r2)) r2)
  let sine := mul sign (mul r poly)
  let prog : Program := {
    name := "FixedSinOsc"
    inputs := #[
      { name := "freq", type? := some (.scalar .float),
        default? := some (selectE (gt (lit 440) (lit 0)) (lit 440) (lit 0)) },
      { name := "clk", type? := some (.scalar .int),
        default? := some (lshift .sampleIndex (lit 32)) },
      { name := "phase", type? := some (.scalar .float),
        default? := some (clampE (lit 0) (lit 0) (lit 1)) } ]
    outputs := #[{ name := "sine", type? := some (.scalar .float) }]
    decls := #[]
    assigns := #[{ target := .port ⟨0⟩, expr := sine }]
    binderCount := 0
    registry := #[] }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  .ok ({ arena with programs := arena.programs.push prog }, idx)

-- ─────────────────────────────────────────────────────────────
-- C1 — a real MULTI-PORT program from the cartesian combinators: `MorphOsc`
-- ─────────────────────────────────────────────────────────────

/-! `buildFixedSinOsc` proved the combinators can emit a SISO generator's body
    from scratch (absorbing `inlineInstances` for one voice). `MorphOsc` is the
    DATA-axis step up — a genuine multi-port composition, built point-free from
    the products surface above:

      `ClockPhasor ⋙ (saw &&& Sin) ⋙ crossfade`

    It exercises everything `warpBank` did not: a real multi-INPUT instance
    (`ClockPhasor(clk, freq)` via the named-port bridge), the `ph.phase` diagonal
    fanned into *heterogeneous* consumers (a saw shaper AND a `Sin` instance —
    `&&&`), genuine `⋙` composition between two DIFFERENT sub-programs
    (ClockPhasor's phase feeds Sin's `x`), and the crossfade product
    `(1−morph)·saw + morph·sin`. The byte-gate (`runEmitCorpusGate "MorphOsc"`)
    asserts this reproduces the hand-written `stdlib/MorphOsc.md` exactly. -/

/-- Mirror the elaborator's `registerInstanceDecl` over a sequence of
    instantiated program names: each adds `(name, idx)` then merges that
    program's own registry (skipping keys already present), in declaration
    order. Generalizes `buildWarpBank`'s single-voice registry merge to the
    several distinct programs a multi-instance body references. -/
def buildRegistry (arena : Arena) (resolved : Array (String × ProgramIdx))
    (programNames : Array String) : Except String (Array (String × ProgramIdx)) := do
  let mut registry : Array (String × ProgramIdx) := #[]
  for pn in programNames do
    let some idx := (resolved.find? (·.1 == pn)).map (·.2)
      | .error s!"EmitArrow: program '{pn}' not found in the elaborated stdlib chain"
    let some prog := arena.program? idx
      | .error s!"EmitArrow: program '{pn}' index out of range"
    if !registry.any (·.1 == prog.name) then registry := registry.push (prog.name, idx)
    for (k, v) in prog.registry do
      if !registry.any (·.1 == k) then registry := registry.push (k, v)
  pure registry

/-- ClockPhasor's input ports MorphOsc fills: `clk` (port 0), `freq` (port 1),
    `offset` (port 2 — the phase-anchor hook, wired from MorphOsc's `phase`). -/
def clockPhasorPorts : Array InputIdx := #[⟨0⟩, ⟨1⟩, ⟨2⟩]

/-- The phasor morphism `[clk, freq, offset] ⇝ [phase]` — the named-port bridge
    over `ClockPhasor`. -/
def phasorMor : Mor := instMor "ph" "ClockPhasor" clockPhasorPorts 1

/-- The saw shaper `[phase] ⇝ [2·phase − 1]` (a naive ramp; pure `arr`). -/
def sawMor : Mor := arrMor (fun w => #[sub (mul (lit 2) w[0]!) (lit 1)])

/-- The sine path `[phase] ⇝ [Sin(2π·phase).out]` — scale-by-2π (`arr`) ⋙ the
    `Sin` bridge. The `⋙` here is the cross-program inline: `Sin`'s body is
    fed `2π·phase` with no surviving instance boundary. -/
def sinMor : Mor :=
  seq (arrMor (fun w => #[mul (lit 6283185307179586 15) w[0]!]))
      (instMor "sin" "Sin" #[⟨0⟩] 1)

/-- The crossfade product `[a, b, mix] ⇝ [(1−mix)·a + mix·b]` (pure `arr`) —
    `CrossFade`'s body, inlined. -/
def crossfadeMor : Mor :=
  arrMor (fun w => #[add (mul (sub (lit 1) w[2]!) w[0]!) (mul w[2]! w[1]!)])

/-- `MorphOsc` as one cartesian pipeline over inputs `[freq, morph, clk, phase]`:
    route to `[clk, freq, phase, morph]`, run the phasor on the first three
    (`clk, freq, offset`) while `morph` rides along (`first 3`), fan the phase into
    saw and sine while `morph` rides along (`first 1 (saw &&& sin)`), then
    crossfade. The whole body is `ClockPhasor ⋙ (saw &&& Sin) ⋙ crossfade`; `morph`
    is threaded through the products, never recomputed. -/
def morphOscMor : Mor :=
  seq (arrMor (fun w => #[w[2]!, w[0]!, w[3]!, w[1]!]))   -- [freq,morph,clk,phase] → [clk,freq,phase,morph]
    (seq (first 3 phasorMor)                              -- → [phase, morph]
      (seq (first 1 (fan sawMor sinMor))                  -- → [saw, sin, morph]
           crossfadeMor))                                  -- → [out]

/-- Build the EmitArrow `MorphOsc` (real input ports) into `arena` — the
    products/MIMO corpus gate. Byte-identical to `diffcli emit-stdlib MorphOsc`.
    The input decls reproduce the elaborator's lowering of the source port types
    (`freq` ⇒ `select(hz>0, hz, 0)`, `unipolar` ⇒ `clamp _ 0 1`,
    `clock` ⇒ `sampleIndex << 32`); the registry links `ClockPhasor` and `Sin`. -/
def buildMorphOsc (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "Sin"]
  let (outs, b) := morphOscMor #[.inputRef ⟨0⟩, .inputRef ⟨1⟩, .inputRef ⟨2⟩, .inputRef ⟨3⟩] {}
  let prog : Program := {
    name := "MorphOsc"
    inputs := #[
      { name := "freq", type? := some (.scalar .float),
        default? := some (selectE (gt (lit 220) (lit 0)) (lit 220) (lit 0)) },
      { name := "morph", type? := some (.scalar .float),
        default? := some (clampE (lit 0) (lit 0) (lit 1)) },
      clkInputDecl,
      { name := "phase", type? := some (.scalar .float),
        default? := some (clampE (lit 0) (lit 0) (lit 1)) } ]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := outs[0]! }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  .ok ({ arena with programs := arena.programs.push prog }, idx)

/-- An input-free `MorphOsc` carrier (literal `freqHz`, literal `morph`,
    closed-form `clk = sampleIndex << 32`) for the standard-rep differential —
    same combinator pipeline as `buildMorphOsc`, but renderable directly as a
    session root (no input ports to bind), like the warp-law carriers. -/
def buildMorphOscLit (name : String) (freqHz : Int) (morph : Expr)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["ClockPhasor", "Sin"]
  -- `clk = sampleIndex << 32` inline (the `clkInputDecl` default; `clockLit` is
  -- defined below in the warp-law section, so spell it out here).
  let (outs, b) := morphOscMor #[lit freqHz, morph, .binary .lshift .sampleIndex (lit 32), lit 0] {}
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := outs[0]! }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  .ok ({ arena with programs := arena.programs.push prog }, idx)

-- ─────────────────────────────────────────────────────────────
-- M4 (slice 3) — the warp ARROW LAWS as audio goldens
-- ─────────────────────────────────────────────────────────────

/-! Slices 1-2 certified that one `warpBank` combinator reproduces two
    hand-written stdlib programs byte-for-byte. Slice 3 certifies the warp
    ALGEBRA: the arrow laws of `warp` hold byte-exactly **in rendered audio**.
    Warps are integer add/sub on the Q32.32 fixed-point clock; integer add/sub
    is exact and associative, so the two algebraically-equal sides of a law feed
    the oscillator a *bit-identical* int64 clock and render *bit-identical* audio
    — even though the emitted plans differ (there is no algebraic tree
    normalization; `(clk+δ)−δ` keeps its extra add+sub instructions).

    The carrier is a single closed-form oscillator. To keep both sides of a law
    self-contained (no input ports to bind), the clock and pitch are LITERAL
    closed forms rather than input refs: `clk = sampleIndex << 32` inline, pitch
    a constant. Composing warps needs no new combinator — they are just nested
    clock expressions (`sub (add clk δ) δ`, `neg (neg clk)`, …). -/

/-- The root clock `sampleIndex << 32` as a closed-form value — the inlined
    `clkInputDecl` default, so the carrier needs no `clk` input port. -/
def clockLit : Clock := .binary .lshift .sampleIndex (lit 32)

/-- A δ literal: `toInt(seconds · sampleRate · 2³²)` with `seconds` a concrete
    decimal (`mantissa · 10^(-exponent)`), no input ref — a closed-form Q32.32
    integer sample count, identical across the two sides of every law. -/
def deltaLit (mantissa : Int) (exponent : Nat) : Expr :=
  toIntE (mul (mul (lit mantissa exponent) .sampleRate) (lit 4294967296))

/-- δ₁ = 0.0007 s, as Q32.32 samples. -/
def delta1 : Expr := deltaLit 7 4
/-- δ₂ = 0.0011 s, as Q32.32 samples. -/
def delta2 : Expr := deltaLit 11 4

/-- The `FixedSinOsc` voice with a LITERAL pitch (220 Hz) in place of the
    `pitchIn` input ref — so the warp-law carrier is fully closed-form (no input
    ports for the session-root lowering to bind to). Clock still at port 1. -/
def litPitchSinOscVoice : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, lit 220⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- A single-oscillator carrier clocked at `clkE`: one `FixedSinOsc` voice
    (literal pitch) whose clock is the closed-form expression `clkE`, output =
    its signal. No input ports. The two algebraically-equal `clkE`s of a warp
    law render bit-identical audio though their plans differ.

    Reuses `Builder.osc` (the voice references the elaborated `FixedSinOsc` body;
    strata inlines it) and mirrors `buildWarpBank`'s registry merge. -/
def buildClockCarrier (name : String) (clkE : Clock) (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : Except String (Arena × ProgramIdx) := do
  let v := litPitchSinOscVoice
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{v.programName}' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let (sig, b) := ({} : Builder).osc v "voice" clkE
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := sig }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

/-- Literal-pitch `FixedSinOsc` at 12 kHz — a tone high enough that a small
    lowpass FIR visibly attenuates (for the convolution stress test). -/
def litPitch12kVoice : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, lit 12000⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- A weighted multi-tap carrier over the closed-form `clockLit`: each `Tap` is a
    clock-warp + weight, fanned from the one clock and summed left-assoc. With
    integer-sample-delay warps this IS an FIR convolution. Closed-form (literal
    pitch, no input ports), so it renders directly as a session root — same shape
    as `buildClockCarrier`, generalized from one tap to a bank. -/
def buildTapCarrier (name : String) (v : Voice) (taps : Array Tap)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{v.programName}' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let mut b : Builder := {}
  let mut summands : Array Sig := #[]
  for tap in taps do
    let (sig, b') := b.osc v tap.name (tap.warp clockLit)
    b := b'
    summands := summands.push (mul tap.weight sig)
  let out := match summands[0]? with
    | none => lit 0
    | some s0 => (summands.extract 1 summands.size).foldl add s0
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := out }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

/-- Literal-pitch `FixedSinOsc` at an arbitrary `hz` (pitch at port 0, clock at
    port 1) — closed form, no input refs. -/
def litPitchVoice (hz : Int) : Voice :=
  { programName := "FixedSinOsc"
    wire := fun clkE => #[ ⟨⟨0⟩, lit hz⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- An FM/PM carrier: a `modHz` modulator oscillator (reading the UNwarped
    `clockLit` — the pinning that keeps the warp a closed form in τ) drives a
    *fractional, sub-sample* clock warp `φ(τ) = clk − ⌊depth·mod(τ)·2³²⌋`, fed to
    a `carHz` carrier. `depth` is in samples; the `·2³²` keeps the warp in Q32.32
    so `mod`'s fractional value lands in the clock's sub-sample bits. This is the
    `sub clk (m clk)` modulated warp — a genuinely nonlinear reparametrization,
    not an affine shift. Closed form, no input ports. -/
def buildFmCarrier (name : String) (carHz modHz depthSamples : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == "FixedSinOsc")).map (·.2)
    | .error "EmitArrow: voice 'FixedSinOsc' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error "EmitArrow: voice 'FixedSinOsc' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let (modSig, b0) := ({} : Builder).osc (litPitchVoice modHz) "mod" clockLit
  let warpedClk : Clock :=
    sub clockLit (toIntE (mul (mul (lit depthSamples) modSig) (lit 4294967296)))
  let (carSig, b) := b0.osc (litPitchVoice carHz) "car" warpedClk
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := carSig }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

/-- Two-level phase modulation (operator FM, DX-style): a `mod2Hz` oscillator
    (reading the ambient clock) warps the `modHz` modulator's clock; that
    modulator's output in turn warps the `carHz` carrier's clock. The modulator
    is itself a warped oscillator, so this tests whether the warp/substitution
    composes through NESTING. Closed form, no input ports. -/
def buildPmPmCarrier (name : String) (carHz modHz mod2Hz depth1 depth2 : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let some vIdx := (resolved.find? (·.1 == "FixedSinOsc")).map (·.2)
    | .error "EmitArrow: voice 'FixedSinOsc' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error "EmitArrow: voice 'FixedSinOsc' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  -- innermost: mod2 at the ambient clock; warps the modulator's clock
  let (mod2Sig, b0) := ({} : Builder).osc (litPitchVoice mod2Hz) "mod2" clockLit
  let modClk : Clock :=
    sub clockLit (toIntE (mul (mul (lit depth2) mod2Sig) (lit 4294967296)))
  -- modulator at the mod2-warped clock; its output warps the carrier's clock
  let (modSig, b1) := b0.osc (litPitchVoice modHz) "mod" modClk
  let carClk : Clock :=
    sub clockLit (toIntE (mul (mul (lit depth1) modSig) (lit 4294967296)))
  let (carSig, b) := b1.osc (litPitchVoice carHz) "car" carClk
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := carSig }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

-- Law 1 — INVERSE / CANCELLATION:  warp(back δ) ⋙ warp(fwd δ) = id
-- Both `(clk+δ)−δ` and `(clk−δ)+δ` cancel to `clk` in exact int64; we build
-- the prose form `(clk+δ)−δ` (fwd inner, back outer).
/-- LHS clock `(clk+δ)−δ` — the oscillator after a forward then inverse warp. -/
def invLawLhsClock : Clock := sub (add clockLit delta1) delta1
/-- RHS clock `clk` — the identity. -/
def invLawRhsClock : Clock := clockLit

-- Law 2 — ADDITIVE DELAY / FUNCTORIALITY:
--   warp(back δ₁) ⋙ warp(back δ₂) = warp(back (δ₁+δ₂))
/-- LHS clock `(clk−δ₁)−δ₂` — two delays composed. -/
def addLawLhsClock : Clock := sub (sub clockLit delta1) delta2
/-- RHS clock `clk−(δ₁+δ₂)` — one delay by the summed offset. -/
def addLawRhsClock : Clock := sub clockLit (add delta1 delta2)

-- ─────────────────────────────────────────────────────────────
-- M5 (slice 4) — the cartesian diagonal / fan-out law
-- ─────────────────────────────────────────────────────────────

/-! Slice 3 certified the warp ALGEBRA (compose, cancel) in audio. Slice 4
    certifies the **cartesian diagonal** (`Δ : A ⇝ A × A`, `Δ = id &&& id`): the
    fan-out of one source into two differently-warped flangers is denotationally
    a pure `let` — equal to two independent (source + flanger) pairs. Two carriers,
    same `out`, same denotation:

    * `DiagonalShared` — ONE dry oscillator at `clk`, *fanned* into two flangers
      (offsets δ₁, δ₂); its `nestedOut` is referenced by BOTH flanger sums.
      5 osc instances: `dry@clk · past@clk−δ₁ · ahead@clk+δ₁ · past@clk−δ₂ · ahead@clk+δ₂`.
    * `DiagonalIndependent` — TWO dry oscillators (both at `clk`, same literal
      pitch), each its own flanger. 6 osc instances — the dry tap is declared
      twice.

    `out = flanger(δ₁) + flanger(δ₂)` with the IDENTICAL weighted-sum tree on
    both sides (`add(add(0.5·dry, 0.25·past), 0.25·ahead)`, the source/warpBank
    left-assoc convention). The ONLY structural difference is the duplicated dry
    instance — so the rendered audio is bit-identical (the two dry oscillators are
    the same closed form fed the same clock), and the COST question is whether
    within-program strata CSE collapses the duplicate so both forms reach the same
    minimal DAG. Like slice 3 the carriers are input-free (literal pitch, closed-
    form `clockLit`), so the session-root lowering binds no inputs.

    NB (session boundary): this CSE is a *within-program* property. Across
    SEPARATE top-level session instances cross-instance CSE does NOT happen (an
    earlier spike found two identical oscs cost ~2×) — so the diagonal's
    *efficiency* holds inside one program, not across a session graph. We do not
    build a multi-instance session here; the within-program pair is the slice. -/

/-- The flanger weighted sum, left-associated to match the source/`warpBank`
    convention exactly: `add(add(0.5·dry, 0.25·past), 0.25·ahead)`. Shared by both
    diagonal forms so their output expression trees are identical up to the dry
    instance index. -/
def flangerSum (dry past ahead : Sig) : Sig :=
  add (add (mul (lit 5 1) dry) (mul (lit 25 2) past)) (mul (lit 25 2) ahead)

/-- One INDEPENDENT flanger over voice `v` at base clock `baseClk`, offset `d`:
    three fresh osc instances (`dry`/`past`/`ahead`, suffixed `tag`), weighted-
    summed. The dry tap is its own instance — the duplicated source. -/
def Builder.flanger (b : Builder) (v : Voice) (baseClk : Clock) (d : Expr)
    (tag : String) : Builder × Sig :=
  let (dry,   b) := b.osc v ("dry" ++ tag)   baseClk
  let (past,  b) := b.osc v ("past" ++ tag)  (sub baseClk d)
  let (ahead, b) := b.osc v ("ahead" ++ tag) (add baseClk d)
  (b, flangerSum dry past ahead)

/-- One flanger over voice `v` at `baseClk`, offset `d`, sharing a pre-built dry
    signal `dry` (the fanned source). Only the two delayed taps are fresh — the
    `&&&` diagonal on the shared source. Same `flangerSum` tree as `Builder.flanger`. -/
def Builder.flangerSharedDry (b : Builder) (v : Voice) (baseClk : Clock) (d : Expr)
    (dry : Sig) (tag : String) : Builder × Sig :=
  let (past,  b) := b.osc v ("past" ++ tag)  (sub baseClk d)
  let (ahead, b) := b.osc v ("ahead" ++ tag) (add baseClk d)
  (b, flangerSum dry past ahead)

/-- Push an input-free voice program (`decls` + one `out = expr` assign) into
    `arena`, merging the `litPitchSinOscVoice` registry like `buildClockCarrier`.
    Shared by every input-free EmitArrow carrier (clock carrier + diagonals). -/
def buildVoiceProgram (name : String) (decls : Array BodyDecl) (out : Sig)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let v := litPitchSinOscVoice
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"EmitArrow: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"EmitArrow: voice '{v.programName}' program index out of range"
  let mut registry : Array (String × ProgramIdx) := #[(vProg.name, vIdx)]
  for (k, vi) in vProg.registry do
    if !registry.any (·.1 == k) then registry := registry.push (k, vi)
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls
    assigns := #[{ target := .port ⟨0⟩, expr := out }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with programs := arena.programs.push prog }, idx)

/-- SHARED diagonal: one dry source fanned into two flangers (δ₁, δ₂). 5 osc
    instances; the dry `nestedOut` feeds both flanger sums. -/
def buildSharedDiagonal (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let (dry, b) := ({} : Builder).osc v "dry" clockLit
  let (b, f1) := b.flangerSharedDry v clockLit delta1 dry "1"
  let (b, f2) := b.flangerSharedDry v clockLit delta2 dry "2"
  buildVoiceProgram "DiagonalShared" b.decls (add f1 f2) arena resolved

/-- INDEPENDENT diagonal: two dry sources (same literal pitch + clock), each its
    own flanger (δ₁, δ₂). 6 osc instances — the dry tap declared twice. Same
    denotation as `buildSharedDiagonal`. -/
def buildIndependentDiagonal (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let (b, f1) := ({} : Builder).flanger v clockLit delta1 "1"
  let (b, f2) := b.flanger v clockLit delta2 "2"
  buildVoiceProgram "DiagonalIndependent" b.decls (add f1 f2) arena resolved

-- ─────────────────────────────────────────────────────────────
-- M6 (slice 5) — REVERSE as warp(neg): the moat operation, certified as laws
-- ─────────────────────────────────────────────────────────────

/-! Slice 5 certifies REVERSE — the moat — as warp arrow laws. Reverse precomposes
    the voice's timebase with negation: `reverse = warp(neg)`, the voice reading
    `s(−clk)`. Three laws, all single-literal-pitch input-free carriers:

    * **Involution** `warp(neg) ⋙ warp(neg) = id`: the voice clocked at `−(−clk)`
      vs at `clk`. Byte-IDENTICAL — `−(−x) = x` exactly on the int64 clock (the
      combinator-level twin of the `ClockReverseProbe` golden).
    * **Reverse-swaps-delay** `warp(neg) ⋙ warp(back δ) = warp(fwd δ) ⋙ warp(neg)`:
      both reduce to the voice at `−clk+δ` — reversal turns a delay into an advance.
      Byte-IDENTICAL (pure int64 clock arithmetic, exact); plans differ.
    * **Reverse commutes with the symmetric flanger** `warp(neg) ⋙ flanger ≡
      flanger ⋙ warp(neg)` — denotationally equal, but under reverse the ±δ taps
      SWAP tree slot (see below), so the float value-sum REASSOCIATES. -/

-- Law 1 — INVOLUTION:  warp(neg) ⋙ warp(neg) = id
/-- LHS clock `−(−clk)` — the voice after reverse then reverse. -/
def revInvolutionLhsClock : Clock := neg (neg clockLit)
/-- RHS clock `clk` — the identity. -/
def revInvolutionRhsClock : Clock := clockLit

-- Law 2 — REVERSE-SWAPS-DELAY:  warp(neg) ⋙ warp(back δ) = warp(fwd δ) ⋙ warp(neg)
-- Per the `⋙` convention (left operand = OUTER warp, as in slices 3-4): the LHS
-- applies `neg` outer over `back δ`, the RHS applies `fwd δ` outer over `neg`.
/-- LHS clock `−(clk−δ) = −clk+δ` — reverse of a delayed clock. -/
def revSwapLhsClock : Clock := neg (sub clockLit delta1)
/-- RHS clock `(−clk)+δ = −clk+δ` — advance of a reversed clock. The delay became
    an advance; both denote `−clk+δ` in exact int64, so byte-identical. -/
def revSwapRhsClock : Clock := add (neg clockLit) delta1

-- Law 3 — REVERSE COMMUTES WITH THE SYMMETRIC FLANGER:
--   warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)
-- Both build the SAME left-assoc `flangerSum` tree `add(add(0.5·dry,0.25·past),
-- 0.25·ahead)` over the SAME tap SET {−clk, −clk−δ, −clk+δ}. The difference is
-- WHICH physical tap lands in the `past` vs `ahead` slot, driven by whether `neg`
-- is the OUTER or INNER warp (the `⋙` convention again):
--
--   LHS = warp(neg) ⋙ flanger  (neg OUTER):  tap clock = neg(tapφ(clk))
--     dry   = neg(clk)            = −clk
--     past  = neg(clk−δ)          = −clk+δ      ← +δ tap in the `past` slot
--     ahead = neg(clk+δ)          = −clk−δ      ← −δ tap in the `ahead` slot
--   RHS = flanger ⋙ warp(neg)  (neg INNER):  tap clock = tapφ(neg(clk))
--     dry   = neg(clk)            = −clk
--     past  = neg(clk)−δ          = −clk−δ      ← −δ tap in the `past` slot
--     ahead = neg(clk)+δ          = −clk+δ      ← +δ tap in the `ahead` slot
--
-- So the −δ and +δ summands swap tree position: `(A + B) + C` vs `(A + C) + B`.
-- Float add is NOT associative, so the rendered audio is denotationally equal
-- (max|Δ| at the ULP scale) but NOT byte-identical. With a FIXED-POINT value
-- carrier (integer add is associative AND commutative) it would be byte-exact;
-- the float value sum is the only non-exact link — the (fixed-point) clock side
-- is exact (laws 1-2 prove it). THIS IS THE FINDING.

/-- LHS `warp(neg) ⋙ flanger`: `neg` is the OUTER warp, so `neg(clk−δ)=−clk+δ`
    and `neg(clk+δ)=−clk−δ` — the ±δ taps land swapped vs the RHS. Same
    `flangerSum` tree, input-free literal-pitch carrier. -/
def buildReverseThenFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let tapClk (φ : Clock → Clock) : Clock := neg (φ clockLit)
  let (dry,   b) := ({} : Builder).osc v "dry"   (tapClk (fun c => c))
  let (past,  b) := b.osc v "past"  (tapClk (fun c => sub c delta1))
  let (ahead, b) := b.osc v "ahead" (tapClk (fun c => add c delta1))
  buildVoiceProgram "ReverseThenFlanger" b.decls (flangerSum dry past ahead) arena resolved

/-- RHS `flanger ⋙ warp(neg)`: `neg` is the INNER warp, so `neg(clk)−δ=−clk−δ`
    and `neg(clk)+δ=−clk+δ` — the ±δ taps keep the source flanger's slot order.
    Same `flangerSum` tree as the LHS; only the per-tap clock differs. -/
def buildFlangerThenReverse (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let v := litPitchSinOscVoice
  let negClk : Clock := neg clockLit
  let tapClk (φ : Clock → Clock) : Clock := φ negClk
  let (dry,   b) := ({} : Builder).osc v "dry"   (tapClk (fun c => c))
  let (past,  b) := b.osc v "past"  (tapClk (fun c => sub c delta1))
  let (ahead, b) := b.osc v "ahead" (tapClk (fun c => add c delta1))
  buildVoiceProgram "FlangerThenReverse" b.decls (flangerSum dry past ahead) arena resolved

-- ─────────────────────────────────────────────────────────────
-- M7 (slice 6) — a FIXED-POINT VALUE carrier: the float reassociation
-- failure (slice 5, law 3) becomes BYTE-IDENTICAL
-- ─────────────────────────────────────────────────────────────

/-! Slice 5 found that `reverse ⋙ flanger ≡ flanger ⋙ reverse` is denotationally
    true but NOT byte-identical in FLOAT (1271/4096 samples differ): under reverse
    the ±δ taps SWAP tree slot, so the float weighted-sum reassociates
    `(A+B)+C` vs `(A+C)+B`, and float add is not associative. Slice 6 proves the
    fix — the SAME warp combinators instantiated at a FIXED-POINT value carrier,
    where the mix is INTEGER arithmetic (associative AND commutative), make that
    same law BYTE-IDENTICAL.

    The "fixed-point value" is the existing `int` carrier reinterpreted as a
    Q-fractional saw, mixed with the existing INTEGER IR ops. No type-system /
    `ScalarKind` change, no fixed-point `Sin`, no touching the engine's float
    carrier or the frozen float goldens — just integer `Expr` over the same
    Q32.32 clock the warps already speak, with a single `toFloat` scale at the
    DAC boundary.

    * SOURCE: a raw Q0.32 saw phasor INTEGER value — ClockPhasor's split-multiply
      but stopping BEFORE the `/2³²` toFloat (`fixedPhase`). Integer-only.
    * MIX: the SAME 0.5/0.25/0.25 weights as the float `flangerSum`, but as
      integer right-shifts `((dry>>1)+(past>>2))+(ahead>>2)` (`fixedFlangerSum`).
      Integer add is associative, so the past/ahead slot swap is invisible.
    * OUTPUT: `toFloat(mix)/2³²` at the boundary (`fixedOut`) — the SAME map on
      both law sides, so byte-identical ints render byte-identical floats. -/

/-- `⌊freq·2³²/SR⌋` as a LITERAL int (freq = 220 Hz, SR = 44100):
    `⌊220·4294967296/44100⌋ = 21426140`. Precomputed so the source needs no
    float division — it is integer end to end. -/
def fixedFreqInc : Expr := lit 21426140

/-- A fixed-point Q0.32 saw phasor as a PURE INTEGER function of the clock —
    ClockPhasor's split-multiply (`acc = inc·thi + (inc·tlo)>>32`, masked to
    `[0,2³²)`) but stopping BEFORE the `/2³²` toFloat, so the value stays a raw
    Q0.32 integer the flanger can mix with integer arithmetic. `thi = clk>>32`
    (whole samples), `tlo = clk & (2³²−1)` (the sub-sample fraction). Integer ops
    ONLY: `.rshift`, `.bitAnd`, `.add`, `.mul`. The `& (2³²−1)` reductions make
    the arithmetic/logical-shift distinction irrelevant (masked away), exactly as
    in ClockPhasor. -/
def fixedPhase (clk : Clock) : Sig :=
  let thi := .binary .rshift clk (lit 32)
  let tlo := .binary .bitAnd clk (lit 4294967295)
  let acc := add (mul fixedFreqInc thi)
                 (.binary .rshift (mul fixedFreqInc tlo) (lit 32))
  .binary .bitAnd acc (lit 4294967295)

/-- The fixed-point flanger weighted sum: the SAME 0.5/0.25/0.25 weights as the
    float `flangerSum`, but as INTEGER right-shifts on the Q0.32 source values —
    `((dry>>1) + (past>>2)) + (ahead>>2)`. Integer add is associative AND
    commutative (i64 wraparound is modular), so the past/ahead slot swap that
    `reverse` induces leaves the sum BIT-IDENTICAL — the float reassociation
    `(A+B)+C ≠ (A+C)+B` (slice 5, 1271/4096) cannot occur. Left-assoc to mirror
    `flangerSum`'s tree exactly. -/
def fixedFlangerSum (dry past ahead : Sig) : Sig :=
  add (add (.binary .rshift dry (lit 1)) (.binary .rshift past (lit 2)))
      (.binary .rshift ahead (lit 2))

/-- Scale the final fixed-point mix to the float output at the DAC boundary — a
    single `toFloat · /2³²` map, applied IDENTICALLY on both law sides, so
    byte-identical integer mixes render byte-identical floats. The Q0.32 mix is
    in `[0,2³²)`, so the float output is a unipolar saw in `[0,1)`. -/
def fixedOut (mix : Sig) : Sig :=
  .binary .div (.unary .toFloat mix) (lit 4294967296)

/-- Push an input-free, INSTANCE-FREE expression program into `arena` (no voice,
    no registry — the fixed-point carrier is pure integer arithmetic on
    `sampleIndex`, referencing no stdlib program). Returns its `ProgramIdx`. -/
def buildExprCarrier (name : String) (out : Sig) (arena : Arena) : Arena × ProgramIdx :=
  let prog : Program := {
    name
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := #[]
    assigns := #[{ target := .port ⟨0⟩, expr := out }]
    binderCount := 0
    registry := #[] }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  ({ arena with programs := arena.programs.push prog }, idx)

/-- A single fixed-point source carrier `out = fixedOut(fixedPhase(clkE))` — the
    fixed-point analog of `buildClockCarrier`, for the single-source warp laws
    (involution, reverse-swaps-delay, additive) over the INTEGER source. The two
    algebraically-equal clocks of a law feed `fixedPhase` a bit-identical int64
    clock, so the rendered audio is byte-identical (the clock side is exact). -/
def buildFixedSourceCarrier (name : String) (clkE : Clock) (arena : Arena) :
    Arena × ProgramIdx :=
  buildExprCarrier name (fixedOut (fixedPhase clkE)) arena

-- ── The BOOTSTRAP (part 1): the phasor + sine as pure Expr, over {+, ×, frac} ──
-- The generators (phasor, sine) were the last thing the arrow layer borrowed from
-- `.trop`. These pointwise pieces need no ArrowTerm; the term wrapper that lifts
-- them onto the clock leaf lives below `emitTerm`. Gated bit-exact vs `.trop`
-- `FixedSinOsc` by `bootstrap-sin`.

/-- `ClockPhasor`'s phase as pure arithmetic over a Q32.32 clock signal: the exact
    integer split-multiply `acc = inc·thi + (inc·tlo)>>32 + off`, masked to
    `[0,2³²)`, then `/2³²` to a unipolar float. `inc = ⌊freq·2³²/SR⌋`,
    `off = ⌊offset·2³²⌋`. Structurally `stdlib/ClockPhasor.md`, as a `Sig`. -/
def phasorPhaseSig (freqE offsetE clkSig : Expr) : Expr :=
  let inc := toIntE (div (mul freqE (lit 4294967296)) .sampleRate)
  let off := toIntE (mul offsetE (lit 4294967296))
  let thi := rshift clkSig (lit 32)
  let tlo := bitAnd clkSig (lit 4294967295)
  let acc := add (add (mul inc thi) (rshift (mul inc tlo) (lit 32))) off
  div (toFloatE (bitAnd acc (lit 4294967295))) (lit 4294967296)

/-- `stdlib/Sin`: range-reduce `r = x − round(x/π)·π`, sign from `n & 1`, a Horner
    polynomial in `r²` with the same minimax coefficients. Pointwise `Sig → Sig` —
    every transcendental is a polynomial, so it needs only `+`/`×` (and `round`
    for the range reduction). The `c0` term drops the fold's leading `0·r²` (a
    bit-exact no-op). -/
def sinSig (x : Expr) : Expr :=
  let n := roundE (mul x (lit 3183098861837907 16))
  let r := sub x (mul n (lit 3141592653589793 15))
  let sign := sub (lit 1) (mul (lit 2) (bitAnd n (lit 1)))
  let r2 := mul r r
  let poly :=
    add (lit 1) (mul (
    add (lit (-16666666666666666) 17) (mul (
    add (lit 8333333333333333 18) (mul (
    add (lit (-1984126984126984) 19) (mul (
    add (lit 27557319223985893 22) (mul (
        lit (-2505210838544172) 23) r2)) r2)) r2)) r2)) r2)
  mul sign (mul r poly)

/-- LHS `warp(neg) ⋙ fixed-point flanger` — `neg` is the OUTER warp, so tap clock
    = `neg(tapφ(clk))`: `past = neg(clk−δ) = −clk+δ`, `ahead = neg(clk+δ) = −clk−δ`
    — the ±δ taps land SWAPPED vs the RHS. Same `fixedFlangerSum` tree; the
    integer mix is slot-order-invariant. -/
def buildReverseThenFixedFlanger (arena : Arena) : Arena × ProgramIdx :=
  let tapClk (φ : Clock → Clock) : Clock := neg (φ clockLit)
  let dry   := fixedPhase (tapClk (fun c => c))
  let past  := fixedPhase (tapClk (fun c => sub c delta1))
  let ahead := fixedPhase (tapClk (fun c => add c delta1))
  buildExprCarrier "ReverseThenFixedFlanger"
    (fixedOut (fixedFlangerSum dry past ahead)) arena

/-- RHS `fixed-point flanger ⋙ warp(neg)` — `neg` is the INNER warp, so tap clock
    = `tapφ(neg(clk))`: `past = neg(clk)−δ = −clk−δ`, `ahead = neg(clk)+δ = −clk+δ`
    — the ±δ taps keep the source flanger's slot order. Same `fixedFlangerSum`
    tree; only the per-tap clock differs. The integer sum of the SAME three tap
    values is bit-identical to the LHS. -/
def buildFixedFlangerThenReverse (arena : Arena) : Arena × ProgramIdx :=
  let negClk : Clock := neg clockLit
  let tapClk (φ : Clock → Clock) : Clock := φ negClk
  let dry   := fixedPhase (tapClk (fun c => c))
  let past  := fixedPhase (tapClk (fun c => sub c delta1))
  let ahead := fixedPhase (tapClk (fun c => add c delta1))
  buildExprCarrier "FixedFlangerThenReverse"
    (fixedOut (fixedFlangerSum dry past ahead)) arena

-- ─────────────────────────────────────────────────────────────
-- M8 — the SLIDE (WARP-PUSH): a REIFIED arrow term + the τ-push rewrite
-- ─────────────────────────────────────────────────────────────

/-! Everything above is the SMART-CONSTRUCTOR (deep-by-emission) layer: the
    combinators build the IR directly, with nothing to inspect or rewrite. The
    slide needs the opposite — a REIFIED arrow term it can pattern-match — because
    WARP-PUSH is a *rewrite*: it takes an effect presented DOWNSTREAM (`warp`
    applied to a signal) and pushes the warp UP through the stateless cone until
    it lands as a generator's clock argument. Until now the warped clocks were
    written by hand (`warpBank` builds the already-upstream form); this section
    makes the COMPILER perform downstream→upstream.

    `ArrowTerm` is a tiny inspectable arrow AST. The only non-trivial node is
    `warp φ t` — `warp(φ) ⋙ t`, kept UNREDUCED. `normalize` (the slide) is three
    law-justified rules plus the fork over products:

      * `warp φ ⋙ arr f      ⟶ arr f ⋙ warp φ`     (slide past a pointwise node, R1)
      * `warp φ ⋙ warp ψ     ⟶ warp (φ∘ψ)`          (fuse, R2)
      * `warp φ ⋙ gen[clk:=c] ⟶ gen[clk:=φ c]`       (absorb into the clock, R4)
      * `warp φ ⋙ (a + b)    ⟶ (warp φ ⋙ a) + (warp φ ⋙ b)`, and through `scale`
                                                     (fork over the product, R3)

    Crucially the warp/arr COMMUTATION (R1) is the first thing in this whole file
    that the arrow LAWS make true and that plain `let`/composition does NOT give
    you for free — this is where the EDSL stops being re-spelled `let` and starts
    earning its keep. The denotation is `⟦warp φ t⟧ = ⟦t⟧ ∘ φ`, so the slide is
    just ∘-associativity, exactly the static-warp lawfulness already proven. -/

/-- A reified, inspectable arrow term. `gen` is a voice instance whose clock arg
    is the warp target; `warp φ t` is `warp(φ) ⋙ t` kept unreduced; `scale`/`arrUn`
    are pointwise (clock-agnostic) `arr`s; `sum` is the left-assoc weighted sum
    (the cartesian product collapse). Functions are carried opaquely — the slide
    composes/applies them, never inspects them. -/
inductive ArrowTerm where
  | gen (v : Voice) (name : String) (clk : Clock)
  | warp (φ : Clock → Clock) (t : ArrowTerm)
  | scale (w : Sig) (t : ArrowTerm)
  | arrUn (f : Sig → Sig) (t : ArrowTerm)
  | sum (ts : Array ArrowTerm)
  /-- A SIGNAL-dependent warp — the data-into-clock edge, made first-class. Bends
      the clock of `t` by `mw baseClk modSig`, where `modSig` is the SIGNAL of
      `modulator` (another sub-term, a function of τ, so the warp stays a closed
      form — the PM-of-PM lawfulness). This is the TOTAL generalization of the old
      fused `fmGen`: it is a WRAPPER (like `warp`), so signal-warps compose with
      plain warps and with each other — `swarp ∘ swarp`, `warp ∘ swarp`, all nest.
      `fmGen carrier base mw mod` is just `swarp mw mod (gen carrier base)`. The
      slide (in `emitTermC`) distributes it onto the generators `t` feeds. -/
  | swarp (mw : Clock → Sig → Clock) (modulator : ArrowTerm) (t : ArrowTerm)
  /-- A τ-CONSTANT leaf — a bare signal `s` that does not read the clock (a param
      slot read `paramRef`, or any literal). It has no generator, so the slide's
      clock transform threads STRAIGHT PAST it (warping a constant is the
      constant). This is what a "knob" node lowers to: its value is a module slot,
      wireable into any modulator (`swarp`/`fm`) or generator-pitch position, and
      driven live by `set_param` — no per-sample state, so no relower needed. -/
  | konst (s : Sig)
  /-- The pointwise ring PRODUCT of two sub-terms — `x ⊗ y`, the multiplicative
      dual of `sum`. `scale` multiplies a term by a fixed value `w`; `prod`
      multiplies two TERMS, each with its own generators. The slide law holds
      because pre-composition distributes over pointwise ×: `warp φ (x ⊗ y) =
      (x ∘ φ)·(y ∘ φ) = (warp φ x) ⊗ (warp φ y)`, so `emitTermC` threads the SAME
      clock transform into BOTH factors — a downstream warp reclocks both. This is
      the VCA / amplitude-multiply that `scale` cannot express: a warp on
      `scale w t` leaves `w` un-reclocked, but a warp on `prod x y` reclocks `x`
      AND `y`, so an envelope factored as its own term rides every delay tap. -/
  | prod (x y : ArrowTerm)
  /-- The CLOCK LEAF — the one atom warp actually acts on. Its signal IS the
      (warped) clock: `emitTermC` applies the ambient clock transform to `c` and
      hands the result back as a `Sig`. Everything periodic is built on this — a
      phasor is pointwise arithmetic over `clk`, and because the leaf is warped,
      so is every oscillator built from it (reverse, scrub, future-tap for free).
      This is what lets the stdlib's generators be TERMS over `{clk, +, ×, frac}`
      instead of opaque `.trop` instances — the bootstrap's ground floor. -/
  | clk (c : Clock)

instance : Inhabited ArrowTerm := ⟨.sum #[]⟩

/-- Normalize a term's sub-structure (the modulators, the branches). The SLIDE
    itself — pushing every `warp`/`swarp` onto the generators it feeds — is
    realized in `emitTermC`, which threads the composed clock transform down to
    each generator. Keeping warps as WRAPPERS here (rather than eagerly fusing
    them into generator clocks) is exactly what makes the algebra TOTAL: the slide
    is function composition of clock transforms, so `warp ∘ warp`, `warp ∘ swarp`,
    and `swarp ∘ swarp` all compose — there is no case it cannot reduce. -/
partial def normalize : ArrowTerm → ArrowTerm
  | .gen v name clk => .gen v name clk
  | .scale w t => .scale w (normalize t)
  | .arrUn f t => .arrUn f (normalize t)
  | .sum ts => .sum (ts.map normalize)
  | .warp φ t => .warp φ (normalize t)
  | .swarp mw mod t => .swarp mw (normalize mod) (normalize t)
  | .konst s => .konst s
  | .prod x y => .prod (normalize x) (normalize y)
  | .clk c => .clk c

/-- Emit a (normalized) arrow term to its output signal. Each `gen` sources a
    voice instance in left-to-right order (matching `warpBank`'s instance order);
    `scale`/`arrUn` are the pointwise ops; `sum` is the left-assoc fold
    `((t₀ + t₁) + t₂)`. Instance names are uniquified by position (names are
    inlined away post-strata, so they never reach the emitted bytes). -/
partial def emitTermC (cmod : Clock → Builder → Clock × Builder) :
    ArrowTerm → Builder → Sig × Builder
  | .gen v name clk, b =>
    let (clk', b) := cmod clk b
    -- Thread the phase anchor across this warped copy: the slide shifted the clock
    -- by `clk − clk'`, so add `corr (clk − clk')` to the voice's phase port. Keeps
    -- delay/flange taps phase-continuous under a live freq change (not just the
    -- un-warped read). A `phaseAnchor := none` voice passes through untouched.
    let v := match v.phaseAnchor with
      | some (port, corr) =>
        let c := corr (sub clk clk')
        { v with wire := fun cc => (v.wire cc).map fun ii =>
            if ii.port.idx == port.idx then { ii with value := add ii.value c } else ii }
      | none => v
    b.osc v s!"{name}{b.decls.size}" clk'
  | .scale w t, b => let (s, b) := emitTermC cmod t b; (mul w s, b)
  | .arrUn f t, b => let (s, b) := emitTermC cmod t b; (f s, b)
  -- a plain warp composes into the threaded transform (R1/R2/R4 in one line).
  | .warp φ t, b => emitTermC (fun c b => cmod (φ c) b) t b
  -- a signal warp: source the modulator (pinned through the SAME enclosing `cmod`,
  -- so a downstream warp reclocks it too — the lawful PM-of-PM rule), read its
  -- signal, then bend the clock by `mw` before the enclosing transform.
  | .swarp mw mod t, b =>
    emitTermC (fun c b =>
      let (mSig, b) := emitTermC cmod mod b
      cmod (mw c mSig) b) t b
  -- a τ-constant leaf: no generator to reclock, so the clock transform `cmod` is
  -- discarded and the bare signal is emitted as-is.
  | .konst s, b => (s, b)
  -- the ring product: thread the SAME clock transform into both factors (the slide
  -- distributes over ×), emit both signals, multiply. A downstream warp thus
  -- reclocks x and y together — the amplitude/VCA multiply `scale` can't give.
  | .prod x y, b =>
    let (sx, b) := emitTermC cmod x b
    let (sy, b) := emitTermC cmod y b
    (mul sx sy, b)
  -- the clock leaf: apply the ambient clock transform and hand the warped clock
  -- back AS the signal (Clock and Sig are both `Expr`). This is the one leaf warp
  -- acts on; a phasor over it inherits reverse/scrub/future-tap.
  | .clk c, b => cmod c b
  | .sum ts, b =>
    match ts[0]? with
    | none => (lit 0, b)
    | some t0 =>
      let (s0, b) := emitTermC cmod t0 b
      (ts.extract 1 ts.size).foldl
        (fun (acc : Sig × Builder) ti => let (s, b) := emitTermC cmod ti acc.2; (add acc.1 s, b))
        (s0, b)

/-- Emit a (normalized) term at the identity clock context — the public entry. -/
def emitTerm (t : ArrowTerm) (b : Builder) : Sig × Builder :=
  emitTermC (fun c b => (c, b)) t b

/-- A 3-tap flanger as a DOWNSTREAM-PRESENTED effect (a morphism on terms):
    `s ↦ 0.5·s + 0.25·(warp(−δ) ⋙ s) + 0.25·(warp(+δ) ⋙ s)`. The warps are
    UNREDUCED — they sit on the signal `s`, exactly as "I dropped a flanger after
    this signal" reads. The slide is what turns this into the upstream form. -/
def flangeEffectWith (back fwd : Clock → Clock) (s : ArrowTerm) : ArrowTerm :=
  .sum #[ .scale (lit 5 1) s,
          .scale (lit 25 2) (.warp back s),
          .scale (lit 25 2) (.warp fwd s) ]

/-- The flanger effect with the `FlangeSin` offset `δ = deltaSamples` (the
    offset input), so the slid form reproduces the stdlib `FlangeSin` clocks. -/
def flangeEffect (s : ArrowTerm) : ArrowTerm :=
  flangeEffectWith (fun c => sub c deltaSamples) (fun c => add c deltaSamples) s

/-- A SWEPT delay offset: `δ(τ) = toInt(seconds · m(τ) · sampleRate · 2³²)` — the
    static flanger's δ, but the depth is scaled by a modulator signal `m ∈ [−1,1]`,
    so the comb delay sweeps (through zero, as `m` crosses 0). -/
def sweptDelta (secondsE : Expr) (m : Sig) : Expr :=
  toIntE (mul (mul (mul secondsE m) .sampleRate) (lit 4294967296))

def sflangeBack (secondsE : Expr) : Clock → Sig → Clock := fun c m => sub c (sweptDelta secondsE m)
def sflangeFwd (secondsE : Expr) : Clock → Sig → Clock := fun c m => add c (sweptDelta secondsE m)

/-- A SWEPT (through-zero) flanger as a DOWNSTREAM effect: like `flangeEffect`, but
    the ±δ taps are SIGNAL-modulated (`swarp`) by `mod` — a modulator term (an LFO,
    or any patched signal). The slide distributes the signal-warp onto the input's
    generators (each becomes a modulated carrier); the dry tap is unwarped. Because
    `swarp` is a wrapper, this stacks freely: a swept flange after a plain flange,
    after another swept flange, all compose — totality, not a special case. -/
def sweptFlangeEffect (back fwd : Clock → Sig → Clock) (mod s : ArrowTerm) : ArrowTerm :=
  .sum #[ .scale (lit 5 1) s,
          .scale (lit 25 2) (.swarp back mod s),
          .scale (lit 25 2) (.swarp fwd mod s) ]

/-- THE SLIDE GATE (Test 1). Build `FlangeSin` from the DOWNSTREAM-insert form —
    `osc ⋙ flange`, warps unreduced — then `normalize` (the slide) and emit.
    Byte-identical to stdlib `FlangeSin` ⇒ the compiler turned "flanger dropped
    downstream of the oscillator" into "the oscillator read at warped clocks,"
    reaching the exact hand-written upstream program. The inputs/voice/δ mirror
    `flangeSinSpec` so the only thing under test is the slide. -/
def buildFlangerViaSlide (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let term := normalize (flangeEffect (.gen fixedSinOscVoice "osc" clkIn))
  let (out, b) := emitTerm term {}
  let prog : Program := {
    name := "FlangeSin"
    inputs := #[clkInputDecl, pitchInputDecl "freq" 220, offsetInputDecl "depth"]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := out }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  .ok ({ arena with programs := arena.programs.push prog }, idx)

/-! The slide tests below use closed-form (literal-pitch, no input ports)
    carriers so they render directly as session roots, like the warp-law gates. -/

/-- A literal-δ back/forward warp (`δ = 0.0007 s`, Q32.32). -/
def slideBack : Clock → Clock := fun c => sub c (deltaLit 7 4)
def slideFwd : Clock → Clock := fun c => add c (deltaLit 7 4)
/-- The closed-form base oscillator term (literal pitch, `clk = sampleIndex<<32`). -/
def litOscGen : ArrowTerm := .gen litPitchSinOscVoice "osc" clockLit

/-- DOWNSTREAM `osc ⋙ shaper ⋙ flange` (Test 2): a pointwise `shaper` (square)
    sits BETWEEN the oscillator and the flanger, so the flanger's warps must
    COMMUTE PAST it (R1) to reach the generator's clock. Slid form. -/
def buildSlideShaperDownstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let term := flangeEffectWith slideBack slideFwd (.arrUn (fun s => mul s s) litOscGen)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "SlideShaperDown" b.decls out arena resolved

/-- The hand-written UPSTREAM reference for Test 2: the same shaper applied to the
    oscillator read at each of the three warped clocks — what the slide MUST
    produce if the warp commutes past the shaper. No `warp` nodes (already
    upstream). Byte-equal to `buildSlideShaperDownstream` ⇒ R1 fired correctly. -/
def buildSlideShaperUpstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let shaped (φ : Clock → Clock) : ArrowTerm :=
    .arrUn (fun s => mul s s) (.gen litPitchSinOscVoice "osc" (φ clockLit))
  let term : ArrowTerm := .sum #[
    .scale (lit 5 1) (shaped (fun c => c)),
    .scale (lit 25 2) (shaped slideBack),
    .scale (lit 25 2) (shaped slideFwd) ]
  let (out, b) := emitTerm term {}
  buildVoiceProgram "SlideShaperUp" b.decls out arena resolved

/-- A single closed-form flanger via the slide (Test 3 baseline). -/
def buildSlideSingleFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let (out, b) := emitTerm (normalize (flangeEffectWith slideBack slideFwd litOscGen)) {}
  buildVoiceProgram "SlideSingleFlange" b.decls out arena resolved

/-- CASCADE `osc ⋙ flange ⋙ flange` (Test 3): two downstream flangers in series.
    The slide pushes the outer flanger's warps through the inner flanger's sum
    (R3) and fuses them with the inner warps (R2), producing the oscillator read
    at the NINE convolved offsets {0, ±δ, ±2δ} automatically — the proper
    multiplicity, derived, not hand-written. (Nine instances, not five: with no
    coincident-offset normalization the algebraically-equal taps stay distinct —
    the multiplicative cost discussed for cascades.) -/
def buildSlideDoubleFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let inner := flangeEffectWith slideBack slideFwd litOscGen
  let (out, b) := emitTerm (normalize (flangeEffectWith slideBack slideFwd inner)) {}
  buildVoiceProgram "SlideDoubleFlange" b.decls out arena resolved

/-- The two factors of the product slide-law test: two distinct-pitch oscillators,
    so `x ⊗ y` is a genuine ring-modulation (not `x²`) and reclocking is observable. -/
def prodFactorX : ArrowTerm := .gen (litPitchVoice 220) "a" clockLit
def prodFactorY : ArrowTerm := .gen (litPitchVoice 330) "b" clockLit

/-- PRODUCT slide law (Test 4): `warp φ (x ⊗ y)` DOWNSTREAM — the product formed,
    THEN warped — must byte-equal the hand-written upstream form (φ on each
    factor). Byte-equality ⇒ the slide distributes the warp over `×`, i.e.
    `emitTermC` threads the same clock transform into BOTH factors so each
    generator reclocks. This is the law that makes `prod` (the VCA) lawful under
    the slide, exactly as `slide-past-arr` is the law for a pointwise `arr`. -/
def buildSlideProdDownstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let (out, b) := emitTerm (normalize (.warp slideBack (.prod prodFactorX prodFactorY))) {}
  buildVoiceProgram "SlideProdDown" b.decls out arena resolved

/-- The hand-written UPSTREAM reference for Test 4: the same warp applied to each
    factor before the product. Byte-equal to the downstream form ⇒ the warp
    commuted into both factors of the `prod`. -/
def buildSlideProdUpstream (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  let term : ArrowTerm := .prod (.warp slideBack prodFactorX) (.warp slideBack prodFactorY)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "SlideProdUp" b.decls out arena resolved

-- ── The BOOTSTRAP (part 2): FixedSinOsc as a TERM over the clock leaf ──────────

/-- A `FixedSinOsc` built ENTIRELY as a term over the clock leaf: `Sin(2π·phasor)`,
    no `gen`, no `.trop` instance — pure `{clk, +, ×, round}`. Warps reach it
    through the `clk` leaf, so it reverses/scrubs like any generator. -/
def fixedSinOscTerm (freqE offsetE : Expr) (c : Clock) : ArrowTerm :=
  ArrowTerm.arrUn
    (fun clkSig => sinSig (mul (lit 6283185307179586 15) (phasorPhaseSig freqE offsetE clkSig)))
    (ArrowTerm.clk c)

/-- Emit the bootstrapped `FixedSinOsc` (220 Hz, phase 0, `clk = sampleIndex<<32`)
    as an instance-free carrier — the term side of the `bootstrap-sin` gate. -/
def buildBootstrapSinOsc (name : String) (arena : Arena) : Arena × ProgramIdx :=
  let (out, _) := emitTerm (normalize (fixedSinOscTerm (lit 220) (lit 0) clockLit)) {}
  buildExprCarrier name out arena

-- ─────────────────────────────────────────────────────────────
-- M9 — the PATCHER LOWERING: a downstream-only patch graph → arrow term
-- ─────────────────────────────────────────────────────────────

/-! The MVP front end. A patch is a downstream-only DAG of modules (the "you may
    only patch forward" UX rule — which is just the acyclicity invariant the
    whole language rests on, made visible). Lowering reads the doc's slogan
    literally:

      * a WIRE `A.out → B.in`  ⟶  B's effect morphism APPLIED to A's term  (⋙)
      * a FAN-OUT (one output, many inputs)  ⟶  the shared upstream term  (Δ / &&&)
      * a FAN-IN (a mixer)  ⟶  the sum  (the product collapse)
      * a GENERATOR  ⟶  `gen`;  an EFFECT  ⟶  a `ArrowTerm → ArrowTerm`

    An EFFECT contributes `warp` nodes (a flanger fans its input into warped
    taps; a delay/reverse is a bare warp), so it is authored and wired DOWNSTREAM
    — and then `normalize` (the slide) pushes those warps UP to the generators.
    That is the whole trick: the user patches forward, the compiler reads
    backward in time. Because tropical has no state primitive, the slide is
    always total — every module is stateless, so the warp always reaches the
    generators. `lowerGraph` produces the UNREDUCED (downstream) term; the slide
    is the separate pass that follows. -/

/-- A patch-graph node. Generators carry their clock; effects carry the id of the
    upstream node they consume (`⋙`); the mixer carries the ids it sums (fan-in).
    Generators read the master clock; the flanger/`warp`/shaper are the effect
    morphisms. -/
inductive Node where
  | source (v : Voice) (clk : Clock)
  | flange (input : String) (back fwd : Clock → Clock)
  | shaper (input : String) (f : Sig → Sig)
  | warpFx (input : String) (φ : Clock → Clock)
  | mix (inputs : Array String)
  /-- A MODULATED carrier: `input`'s signal modulates this carrier's clock
      (FM/PM). The signal-into-clock edge — patch `mod.out → carrier.fm`. -/
  | fm (input : String) (carrier : Voice) (baseClk : Clock) (depthE : Expr)
  /-- A SWEPT flanger: `input` is the signal flanged, `modInput` the modulator that
      sweeps the ±δ taps (an LFO, or any patched signal). `depthSec` is the sweep
      depth in seconds. The signal-warp distributes onto `input`'s generators. -/
  | sflange (input modInput : String) (depthSec : Expr)
  /-- A KNOB: a program that is nothing but a param with one output. `idx` is the
      `ParamIdx` of the root's param slot; it lowers to a τ-constant `paramRef`
      leaf, so wiring it into a modulator/pitch position binds that parameter to a
      live module slot (`param:<name>`), driven by `set_param` without a relower. -/
  | knob (idx : Nat)
  /-- A one-sided resonant COMB: `input` read at a bank of clock-warped offsets,
      each weighted — `Σ wₖ·(warpₖ ⋙ input)`. `taps` is `(weight, clockShift)` per
      tap: tap 0 is usually the dry `(1, id)`, the rest a decaying series `(gᵏ,
      c ↦ c + k·D)`. One-sided (all shifts the same direction) makes it
      time-ASYMMETRIC — its impulse tail rings on one side, so under a global clock
      reverse the tail flips (echo ↔ pre-echo). A future shift (`c + kD`, `D > 0`)
      reads AHEAD — an audible pre-echo, impossible on a stream. The slide
      distributes each tap's warp onto the generators, exactly like the flanger. -/
  | comb (input : String) (taps : Array (Sig × (Clock → Clock)))
  /-- The multiplicative fan-in — `⊗` over its inputs, the ring-product twin of
      `mix`'s `sum`. Two inputs is ring modulation; an input × an envelope-generator
      is a VCA; and because the slide distributes over `prod`, a downstream warp
      reclocks every factor, so each factor's own generators stay in step. Empty ⇒
      silence (a graceful half-built patch), like `mix`. -/
  | ring (inputs : Array String)

structure PatchNode where
  id : String
  node : Node

/-- A downstream-only patch DAG: named nodes plus the id wired to the output. -/
structure PatchGraph where
  nodes : Array PatchNode
  output : String

/-- The standard FM modulation law: a modulator signal `m` shifts the carrier's
    Q32.32 clock by `depth · m` samples — `clk − toInt(depth · m · 2³²)`. `depth`
    is an EXPRESSION (a literal, or a live `paramRef` slot), so the depth knob can
    be a live param. Closed form in τ (`m` is a closed-form signal) either way. -/
def fmWarp (depthE : Expr) : Clock → Sig → Clock :=
  fun base m => sub base (toIntE (mul (mul depthE m) (lit 4294967296)))

/-- Lower one node to its arrow term, recursing UP its input wires. A wire is
    `⋙` (the effect applied to the upstream term); fan-out is the shared upstream
    term (the diagonal); a mixer is the sum; an `fm` node routes its input's
    signal into the carrier's clock. The result is the UNREDUCED downstream term
    — effects' warps still sit on their inputs. -/
partial def lowerNode (g : PatchGraph) (id : String) : Except String ArrowTerm := do
  let some pn := g.nodes.find? (·.id == id)
    | .error s!"lower: node '{id}' not found"
  match pn.node with
  | .source v clk => .ok (.gen v id clk)
  | .flange inId back fwd => return flangeEffectWith back fwd (← lowerNode g inId)
  | .shaper inId f => return .arrUn f (← lowerNode g inId)
  | .warpFx inId φ => return .warp φ (← lowerNode g inId)
  | .mix inputs => return .sum (← inputs.mapM (lowerNode g))
  | .fm inId carrier base depth =>
    return .swarp (fmWarp depth) (← lowerNode g inId) (.gen carrier id base)
  | .sflange inId modId depthSec =>
    return sweptFlangeEffect (sflangeBack depthSec) (sflangeFwd depthSec)
      (← lowerNode g modId) (← lowerNode g inId)
  | .knob idx => .ok (.konst (.paramRef ⟨idx⟩))
  | .comb inId taps => do
    -- lower the input ONCE, share it across taps (the diagonal); each tap is a
    -- scaled warp of that shared term, summed. `normalize` then slides every
    -- tap's warp up onto the input's generators.
    let s ← lowerNode g inId
    return .sum (taps.map fun (w, φ) => .scale w (.warp φ s))
  | .ring inputs => do
    -- fold the inputs with `prod` (⊗); empty ⇒ silence, like an empty `mix`.
    let terms ← inputs.mapM (lowerNode g)
    match terms.toList with
    | [] => return .konst (lit 0)
    | t :: ts => return ts.foldl (fun acc u => .prod acc u) t

/-- Lower a whole patch to its (downstream, unreduced) arrow term. Compose with
    `normalize` to run the slide, then `emitTerm` to lower to IR. -/
def lowerGraph (g : PatchGraph) : Except String ArrowTerm := lowerNode g g.output

-- The demo patches ─────────────────────────────────────────────

/-- `osc → flange` as a GRAPH (input-ref `FlangeSin` form): a `FixedSinOsc` source
    and a flanger wired downstream of it, offset `δ = deltaSamples`. -/
def flangeGraph : PatchGraph :=
  { nodes := #[
      { id := "osc", node := .source fixedSinOscVoice clkIn },
      { id := "fl",  node := .flange "osc"
          (fun c => sub c deltaSamples) (fun c => add c deltaSamples) } ],
    output := "fl" }

/-- `osc → flange → flange` as a GRAPH (closed form). -/
def doubleFlangeGraph : PatchGraph :=
  { nodes := #[
      { id := "osc", node := .source litPitchSinOscVoice clockLit },
      { id := "f1",  node := .flange "osc" slideBack slideFwd },
      { id := "f2",  node := .flange "f1" slideBack slideFwd } ],
    output := "f2" }

/-- A FAN-OUT patch: `osc` fanned into two flangers (offsets 0.0007 / 0.0011 s),
    summed by a mixer — the diagonal Δ through the lowering. -/
def fanOutGraph : PatchGraph :=
  { nodes := #[
      { id := "osc", node := .source litPitchSinOscVoice clockLit },
      { id := "fa",  node := .flange "osc" slideBack slideFwd },
      { id := "fb",  node := .flange "osc"
          (fun c => sub c (deltaLit 11 4)) (fun c => add c (deltaLit 11 4)) },
      { id := "mix", node := .mix #["fa", "fb"] } ],
    output := "mix" }

/-- THE LOWERING GATE (L1). Lower the GRAPH `osc → flange`, run the slide, emit,
    and wrap as `FlangeSin` — byte-identical to stdlib `FlangeSin`. The user's
    patch graph, lowered, reaches the exact hand-written program: graph → arrow
    → slide → emit, end to end. -/
def buildFlangeFromGraph (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let registry ← buildRegistry arena resolved #["FixedSinOsc"]
  let term ← lowerGraph flangeGraph
  let (out, b) := emitTerm (normalize term) {}
  let prog : Program := {
    name := "FlangeSin"
    inputs := #[clkInputDecl, pitchInputDecl "freq" 220, offsetInputDecl "depth"]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls
    assigns := #[{ target := .port ⟨0⟩, expr := out }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  .ok ({ arena with programs := arena.programs.push prog }, idx)

/-- Lower the GRAPH `osc → flange → flange` (closed form) — must byte-equal the
    hand-built `buildSlideDoubleFlanger` (L2): the lowering of a chain composes
    effects exactly as the hand-written nested term. -/
def buildDoubleFlangeFromGraph (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph doubleFlangeGraph
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphDoubleFlange" b.decls out arena resolved

/-- Lower the FAN-OUT patch (closed form) — `osc` fanned into two flangers and
    mixed (L3): the diagonal + the product collapse through the lowering. -/
def buildFanOutFromGraph (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph fanOutGraph
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphFanOut" b.decls out arena resolved

/-- `osc(mod) → osc(carrier).fm` as a GRAPH — a single FM voice. Lowers to the
    same term as `buildFmCarrier carHz modHz depth` (the bit-exact modulated-clock
    gate's carrier). -/
def fmGraphOf (carHz modHz depth : Int) : PatchGraph :=
  { nodes := #[
      { id := "mod", node := .source (litPitchVoice modHz) clockLit },
      { id := "car", node := .fm "mod" (litPitchVoice carHz) clockLit (lit depth) } ],
    output := "car" }

/-- Two-level PM (DX-style) as a GRAPH: `mod2 → mod.fm → car.fm`. Lowers to the
    same term as `buildPmPmCarrier carHz modHz mod2Hz depth1 depth2` (the
    bit-exact PM-of-PM gate's carrier). -/
def pmPmGraphOf (carHz modHz mod2Hz depth1 depth2 : Int) : PatchGraph :=
  { nodes := #[
      { id := "mod2", node := .source (litPitchVoice mod2Hz) clockLit },
      { id := "mod",  node := .fm "mod2" (litPitchVoice modHz) clockLit (lit depth2) },
      { id := "car",  node := .fm "mod" (litPitchVoice carHz) clockLit (lit depth1) } ],
    output := "car" }

/-- Lower the single-FM graph (M1: ≡ `buildFmCarrier`, the modulated node). -/
def buildFmFromGraph (carHz modHz depth : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph (fmGraphOf carHz modHz depth)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphFm" b.decls out arena resolved

/-- Lower the PM-of-PM graph (M2: ≡ `buildPmPmCarrier`, the nested modulated node). -/
def buildPmPmFromGraph (carHz modHz mod2Hz depth1 depth2 : Int)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let term ← lowerGraph (pmPmGraphOf carHz modHz mod2Hz depth1 depth2)
  let (out, b) := emitTerm (normalize term) {}
  buildVoiceProgram "GraphPmPm" b.decls out arena resolved

end Tropical.EmitArrow
