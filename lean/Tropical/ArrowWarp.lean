import Tropical.Ir.Nodes

/-!
# ArrowWarp — vertical slice 2 (the voice-generic warp bank)

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
and lets strata inline it (the M0 finding). ArrowWarp only ever builds the
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
/-- Negate — the negate EFFECT (`.unary .neg`, emitted `sub i64 0, c`). The SAME
    operation whether applied to a signal value or to the clock; applied to the
    clock it IS `reverse`. There is one algebra of operations on expressions; the
    clock is one of those expressions. -/
def neg (a : Expr) : Expr := .unary .neg a

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
    | .error s!"ArrowWarp: voice '{spec.voice.programName}' not found in the \
        elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"ArrowWarp: voice '{spec.voice.programName}' program index out of range"
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

/-- Build the ArrowWarp `FlangeSin` (FixedSinOsc voice) — the slice-1 gate.
    Byte-identical to `diffcli emit-file stdlib/parsed/FlangeSin.json`. -/
def buildFlanger (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildWarpBank flangeSinSpec arena resolved

/-- Build the ArrowWarp `ReversibleComb` (ModalVoice voice) — the slice-2 gate.
    Byte-identical to `diffcli emit-file stdlib/parsed/ReversibleComb.json`. -/
def buildReversibleComb (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) :=
  buildWarpBank reversibleCombSpec arena resolved

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
    | .error s!"ArrowWarp: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"ArrowWarp: voice '{v.programName}' program index out of range"
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
    Shared by every input-free ArrowWarp carrier (clock carrier + diagonals). -/
def buildVoiceProgram (name : String) (decls : Array BodyDecl) (out : Sig)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) :
    Except String (Arena × ProgramIdx) := do
  let v := litPitchSinOscVoice
  let some vIdx := (resolved.find? (·.1 == v.programName)).map (·.2)
    | .error s!"ArrowWarp: voice '{v.programName}' not found in the elaborated stdlib chain"
  let some vProg := arena.program? vIdx
    | .error s!"ArrowWarp: voice '{v.programName}' program index out of range"
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

end Tropical.ArrowWarp
