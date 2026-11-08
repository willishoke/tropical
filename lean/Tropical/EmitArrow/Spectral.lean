import Tropical.EmitArrow.Numerics
import Tropical.EmitArrow.Inspect

/-!
# EmitArrow.Spectral — partial families and complex spectral envelopes

A **partial family** is a bundle of partials the algebra can see: partials
`k = 1..K` of one fundamental, each the analytic partial `a_k·e^{iθ_k}` with
`θ_k = k·φ`, where `φ` is ONE phasor on the exact integer phase rail. It
lowers to a single `bankSum` region, so the plan is O(1) in `K` (the weight
column is data, like a modal bank's coefficient columns).

A **spectral envelope** is a closed-form complex gain `G(f, τ)` applied to
each partial at its frequency: the partial becomes `a_k·G(f_k, τ)·e^{iθ_k}`,
amplitude × `|G|` and phase + `arg G`. The real projection the family emits
(`sine`: `Im`, `cosine`: `Re`) is taken last. `G` reads its own control
signals (cutoff, Q, …), which may move at any rate.

The family requires its fundamental to be constant within a kernel call (a
uniform): a partial's frequency `k·f0` is then a constant the envelope can be
evaluated at. A moving fundamental (phase-modulated partials) is a later slice.

Soft edges: a partial whose frequency nears Nyquist or 0 Hz fades with a
smoothstep edge instead of a hard cut, and partials at or above Nyquist are
not evaluated at all (the region's trip count is `⌊Nyquist/|f0|⌋`, clamped to
`K`; a skipped partial's edge weight is exactly 0).

This module holds the data types and the bank lowering; the `ArrowTerm`
constructors that carry them live in `Term`.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- A build-time `Float` as an EXACT decimal literal. A finite double is a
    dyadic `m·2^e`, and `m·2^(−d) = (m·5^d)·10^(−d)`, so the literal carries the
    value with no rounding; `JsonNumber.toFloat` then divides an exact multiple
    of `5^d` and rounds once, back to the same double. (`litF` rounds to 12
    decimals, which is not a round trip.) Non-finite input is an error. -/
def litExact (x : Float) : BuildM Sig := do
  let bits := x.toBits
  let negative := bits >>> 63 == 1
  let biased := ((bits >>> 52) &&& 0x7FF).toNat
  let fraction := (bits &&& 0xFFFFFFFFFFFFF).toNat
  if biased == 0x7FF then
    throw s!"litExact: {x} is not finite"
  let significand := if biased == 0 then fraction else fraction + 2 ^ 52
  if significand == 0 then return ← lit 0
  let exponent : Int := (if biased == 0 then 1 else (biased : Int)) - 1075
  let signed (m : Nat) : Int := if negative then -(m : Int) else m
  if exponent ≥ 0 then
    lit (signed (significand * 2 ^ exponent.toNat))
  else
    -- strip binary trailing zeros so the decimal is as short as it can be
    let rec strip (m d : Nat) (fuel : Nat) : Nat × Nat :=
      match fuel with
      | 0 => (m, d)
      | fuel + 1 => if d > 0 && m % 2 == 0 then strip (m / 2) (d - 1) fuel else (m, d)
    let (m, d) := strip significand exponent.natAbs 1100
    lit (signed (m * 5 ^ d)) d

/-- UNIFORM: constant within a kernel call. Read off the node's stage signature
    (computed at intern, `Ir.StageSig`): a uniform value's base stage is below
    `s1` (no τ) and it depends on no instance output. Program inputs are
    trusted: the rate-inlet rule (a freq inlet accepts only uniforms) is
    enforced where wires are admitted, not here. -/
def sigIsUniform (signal : Sig) : BuildM Bool := do
  let builder ← get
  match builder.exprs.sig? signal with
  | some sig => pure (sig.base != .s1 && sig.nested.isEmpty)
  | none => pure false

/-- Which real projection of the analytic partial `a·G·e^{iθ}` a family emits:
    `sine` is `Im` (`Σ a_k sin kφ` when `G = 1`), `cosine` is `Re`
    (`Σ a_k cos kφ`, zero-phase cosines). -/
inductive PartialWave where
  | sine
  | cosine
deriving Inhabited, BEq, Repr

/-- The partial weights `a_k`: a coefficient table with one entry per partial
    (`k = 1..K`; entries may be live slots), or a closed form in the partial
    number `k` (handed over as a float). -/
inductive PartialWeights where
  | table (weights : Array Sig)
  | law (weight : Sig → BuildM Sig)
  /-- A closed form over structural columns and live inputs. `columns` are
      K-entry tables (constants of `k`, e.g. `1/k` and `ln k`), read at
      partial `k`. `inputs` are per-sample scalars (live controls, any
      signal); they ride the region's loop-invariant columns, so each is
      computed once per sample, before the partial loop. `weight cols ins n
      kf` gets the column reads, the input reads, and the partial number
      `k` as an integer (`n`) and as a float (`kf`). -/
  | closedForm (columns : Array (Array Sig)) (inputs : Array Sig)
      (weight : Array Sig → Array Sig → Sig → Sig → BuildM Sig)

/-- One soft edge. `standard` is 20 Hz at DC and the top 10% below Nyquist;
    `width` is an explicit width in Hz; `off` disables the edge. -/
inductive EdgeSpec where
  | standard
  | off
  | width (hz : Sig)

structure SoftEdges where
  dc : EdgeSpec := .standard
  nyquist : EdgeSpec := .standard

/-- A harmonic partial family: partials `k = 1..count` at `k·freq`, phases
    `k·φ` from one phasor on `clk`, weights `a_k`. `freq` must be uniform.
    `phase` (cycles) offsets the fundamental; partial `k` is offset by
    `k·phase`, so one anchor keeps every partial phase-locked. -/
structure PartialFamily where
  name : String := "family"
  freq : Sig
  phase : Option Sig := none
  clk : Clock
  count : Nat
  /-- An optional live partial count (uniform), clamped to `[0, count]`: the
      region trips `min(liveCount, ⌊Nyquist/|f0|⌋)` partials. `count` stays
      the static capacity (the topology); this is trip-count-as-data. -/
  liveCount : Option Sig := none
  weights : PartialWeights
  wave : PartialWave := .sine
  edges : SoftEdges := {}

/-- A complex spectral envelope `G(f, τ)` as a closed form. `prepare` turns the
    envelope's control signals and the family's (signed) fundamental into the
    scalars the per-partial gain needs; they are computed once per sample,
    before the partial loop. `gain` returns `(Re G, Im G)` from those scalars,
    the partial number `k` (float) and the partial frequency `k·f0` (Hz,
    signed). -/
structure EnvelopeLaw where
  name : String
  prepare : Array Sig → Sig → BuildM (Array Sig)
  gain : Array Sig → Sig → Sig → BuildM (Sig × Sig)
  /-- Where the envelope was authored (a patch node and its input wire), for
      the refusal when its input is not something it can sit on. -/
  site : Option String := none

instance : Inhabited EnvelopeLaw :=
  ⟨{ name := "unity", prepare := fun _ _ => pure #[]
     gain := fun _ _ _ => do pure (← lit 1, ← lit 0) }⟩

/-- The largest partial count the phase rail carries: partial `k`'s phase is
    `k·P` with `P < 2^(32+F)` (the fundamental's Q32 phase with the clock's `F`
    fraction bits kept), and `k·P` must stay below `2^63`. -/
def familyCapacity : Nat := 2 ^ (31 - clockFracBits) - 1

/-- The family region's binder id. Family regions never nest inside each other
    (controls and invariants are computed before the region), and a distinct id
    from the modal banks' 0 keeps a collision impossible should one ever be
    built inside another region. -/
def familyBinderId : Nat := 1

/-- The SVF responses, as named envelope shapes. With `x = f/fc` and
    `D = 1 − x² + j·x/Q`: lowpass `1/D`, bandpass `(j·x/Q)/D` (unity gain at
    `fc`), highpass `−x²/D`. -/
inductive SvfResponse where
  | lowpass
  | bandpass
  | highpass
deriving Inhabited, BEq, Repr

def SvfResponse.name : SvfResponse → String
  | .lowpass => "lowpass" | .bandpass => "bandpass" | .highpass => "highpass"

/-- The SVF envelope law. Controls: `#[cutoff (Hz), Q]`, both any signal.
    Per sample it computes `u = f0/fc` and `1/Q` (so `x_k = k·u`, one multiply
    per partial); per partial one division (`1/|D|²`). Cutoff and Q are floored
    at 1e-9 so a zero or negative control cannot divide by zero. -/
def svfEnvelopeLaw (response : SvfResponse) : EnvelopeLaw where
  name := s!"svf-{response.name}"
  prepare := fun controls f0 => do
    let tiny ← lit 1 9
    let cutoff := controls[0]!
    let q := controls[1]!
    let cutoffSafe ← selectE (← gt cutoff tiny) cutoff tiny
    let qSafe ← selectE (← gt q tiny) q tiny
    let u ← div f0 cutoffSafe
    let invQ ← div (← lit 1) qSafe
    pure #[u, invQ]
  gain := fun inv k _f => do
    let x ← mul k inv[0]!
    let x2 ← mul x x
    let one ← lit 1
    let re ← sub one x2
    let im ← mul x inv[1]!
    let den ← add (← mul re re) (← mul im im)
    let invDen ← div one den
    -- G = N·conj(D)/|D|²,  conj(D) = re − j·im
    match response with
    | .lowpass =>
        pure (← mul re invDen, ← neg (← mul im invDen))
    | .bandpass =>
        pure (← mul (← mul im im) invDen, ← mul (← mul im re) invDen)
    | .highpass =>
        pure (← neg (← mul (← mul x2 re) invDen), ← mul (← mul x2 im) invDen)

private def cmulPair (a b : Sig × Sig) : BuildM (Sig × Sig) := do
  let real ← sub (← mul a.1 b.1) (← mul a.2 b.2)
  let imag ← add (← mul a.1 b.2) (← mul a.2 b.1)
  pure (real, imag)

/-- `t²(3 − 2t)` for `t = clamp(v, 0, 1)`: 1 exactly at `t = 1`. -/
private def smoothEdge (v : Sig) : BuildM Sig := do
  let t ← clampE v (← lit 0) (← lit 1)
  let shaped ← sub (← lit 3) (← mul (← lit 2) t)
  mul (← mul t t) shaped

/-- The family's region, given its (already warped) clock and the envelopes
    applied to it, each with its control signals already emitted. -/
def familySig (f : PartialFamily) (clock : Clock)
    (envelopes : Array (EnvelopeLaw × Array Sig)) : BuildM Sig := do
  if f.count == 0 then return ← lit 0
  if f.count > familyCapacity then
    throw s!"partial family '{f.name}': {f.count} partials exceed the phase rail's capacity of {familyCapacity}"
  unless ← sigIsUniform f.freq do
    throw s!"partial family '{f.name}': the fundamental must be uniform (constant within a kernel call); a moving fundamental is a phase-modulated family, not yet supported"
  if let some phase := f.phase then
    unless ← sigIsUniform phase do
      throw s!"partial family '{f.name}': the phase offset must be uniform"
  let weightTable? ← match f.weights with
    | .table ws =>
        if ws.size != f.count then
          throw s!"partial family '{f.name}': {ws.size} weights for {f.count} partials"
        pure (some (← arr ws))
    | .law _ | .closedForm .. => pure none
  let columnTables ← match f.weights with
    | .closedForm columns _ _ => columns.mapM fun column => do
        if column.size != f.count then
          throw s!"partial family '{f.name}': a weight column of {column.size} entries for {f.count} partials"
        arr column
    | _ => pure #[]
  -- ── uniforms and per-sample scalars (computed before the region) ──
  let twoPow32 ← lit 4294967296
  let sr ← sampleRate
  let inc ← toIntE (← div (← mul f.freq twoPow32) sr)
  let fracBits ← clockFracLit
  let thi ← rshift clock fracBits
  let tlo ← bitAnd clock (← clockFracMaskLit)
  let mask32 ← lit 4294967295
  let turns ← mul inc thi
  let turns ← match f.phase with
    | some phase => add turns (← toIntE (← mul phase twoPow32))
    | none => pure turns
  -- P = the fundamental's phase with the clock's fraction bits kept:
  -- `P ≡ (inc·⌊τ⌋ + off)·2^F + inc·frac(τ)  (mod 2^(32+F))`, so partial k's
  -- Q32 phase `⌊k·P / 2^F⌋ mod 2^32` is `k·inc·τ + k·off` floored ONCE.
  let wrapped ← bitAnd turns mask32
  let p ← bitAnd (← add (← lshift wrapped fracBits) (← mul inc tlo))
    (← lit (2 ^ (32 + clockFracBits) - 1))
  let f0abs ← absE f.freq
  let half ← lit 5 1
  let nyquist ← mul sr half
  let zero ← lit 0
  let nyquistInv? ← match f.edges.nyquist with
    | .off => pure none
    | .standard => pure (some (← div (← lit 10) nyquist))
    | .width hz => pure (some (← div (← lit 1) hz))
  let dcInv? ← match f.edges.dc with
    | .off => pure none
    | .standard => pure (some (← lit 5 2))
    | .width hz => pure (some (← div (← lit 1) hz))
  -- Trip count: partials at or above Nyquist have edge weight 0 and are not
  -- evaluated. `⌊Nyquist/max(|f0|, Nyquist/(K+1))⌋` stays finite and ≤ K+1.
  if let some live := f.liveCount then
    unless ← sigIsUniform live do
      throw s!"partial family '{f.name}': the live partial count must be uniform"
  let nyquistCount? ← match nyquistInv? with
    | none => pure none
    | some _ => do
        let floorF0 ← div nyquist (← lit (f.count + 1))
        let f0floor ← selectE (← gt f0abs floorF0) f0abs floorF0
        pure (some (← unary .floor (← div nyquist f0floor)))
  let tripF? ← match nyquistCount?, f.liveCount with
    | some a, some b => pure (some (← selectE (← gt a b) b a))
    | some a, none => pure (some a)
    | none, some b => pure (some b)
    | none, none => pure none
  let liveCount? ← match tripF? with
    | none => pure none
    | some trip => do
        let clamped ← clampE trip zero (← lit f.count)
        pure (some (← toIntE clamped))
  let mut invariants : Array Sig := #[p, f0abs, f.freq]
  if let some inv := nyquistInv? then invariants := invariants.push nyquist |>.push inv
  if let some inv := dcInv? then invariants := invariants.push inv
  let mut envSlots : Array (EnvelopeLaw × Nat × Nat) := #[]
  for (law, controls) in envelopes do
    let prepared ← law.prepare controls f.freq
    envSlots := envSlots.push (law, invariants.size, prepared.size)
    invariants := invariants ++ prepared
  let weightInputStart := invariants.size
  let weightInputs := match f.weights with
    | .closedForm _ inputs _ => inputs
    | _ => #[]
  invariants := invariants ++ weightInputs
  -- Loop-invariant scalars ride the tables list as 1-element columns, so they
  -- are materialized once before the region (`bankFoldInv`'s reason).
  let invTables ← invariants.mapM fun value => arr #[value]
  let zeroIdx ← lit 0
  let invReads ← invTables.mapM fun table => index table zeroIdx
  -- ── the per-partial body ──
  let k ← loopIdx familyBinderId
  let n ← add k (← litI 1)
  let kf ← toFloatE n
  let pRead ← toIntE invReads[0]!
  let phaseK ← bitAnd (← rshift (← mul n pRead) fracBits) mask32
  let q30 ← lit 1073741824
  let weight ← match f.weights, weightTable? with
    | .table _, some table => index table k
    | .law w, _ => w kf
    | .closedForm _ _ w, _ => do
        let colReads ← columnTables.mapM fun table => index table k
        w colReads (invReads.extract weightInputStart (weightInputStart + weightInputs.size)) n kf
    | .table _, none => throw "partial family: unreachable (table without a column)"
  let fk ← mul kf invReads[2]!
  let fkAbs ← mul kf invReads[1]!
  let mut edge? : Option Sig := none
  let mut slot := 3
  if nyquistInv?.isSome then
    let nyq := invReads[slot]!
    let inv := invReads[slot + 1]!
    edge? := some (← smoothEdge (← mul (← sub nyq fkAbs) inv))
    slot := slot + 2
  if dcInv?.isSome then
    let dcEdge ← smoothEdge (← mul fkAbs invReads[slot]!)
    edge? := some (← match edge? with
      | some e => mul e dcEdge
      | none => pure dcEdge)
  let projected ← if envelopes.isEmpty then
      match f.wave with
      | .sine => div (← toFloatE (← fixedSinCycSig phaseK)) q30
      | .cosine => div (← toFloatE (← fixedCosCycSig phaseK)) q30
    else do
      let mut g? : Option (Sig × Sig) := none
      for (law, start, size) in envSlots do
        let gi ← law.gain (invReads.extract start (start + size)) kf fk
        g? := some (← match g? with
          | some acc => cmulPair acc gi
          | none => pure gi)
      let (gre, gim) := g?.getD (zero, zero)
      let s ← div (← toFloatE (← fixedSinCycSig phaseK)) q30
      let c ← div (← toFloatE (← fixedCosCycSig phaseK)) q30
      match f.wave with
      | .sine => add (← mul gre s) (← mul gim c)
      | .cosine => sub (← mul gre c) (← mul gim s)
  let shaped ← match edge? with
    | some e => mul e projected
    | none => pure projected
  let contribution ← mul weight shaped
  let tables := (weightTable?.map (#[·])).getD #[] ++ columnTables ++ invTables
  bankSum f.count tables contribution liveCount? familyBinderId

/-- The saw's weights `1/k`, `k = 1..count`, as an exact coefficient table. -/
def sawWeights (count : Nat) : BuildM PartialWeights := do
  let ws ← (Array.range count).mapM fun i => litExact (1.0 / (i + 1).toFloat)
  pure (.table ws)

end Tropical.EmitArrow
