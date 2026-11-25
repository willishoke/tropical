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
  /-- A COMPLEX weight `a_k + i·b_k`, so a partial carries its own phase. The
      partial is the family's projection of `(a_k + i·b_k)·G·e^{iθ_k}`.
      `tables` may have any length; each is materialized once per sample,
      before the partial loop (entries may be live signals). `weight tables k
      n kf` gets the tables (to `index` anywhere, e.g. at `k` or at `k mod L`),
      the 0-based partial index `k`, the partial number `n = k + 1` and `n` as
      a float, and returns `(a_k, b_k)`. -/
  | complex (tables : Array (Array Sig))
      (weight : Array Sig → Sig → Sig → Sig → BuildM (Sig × Sig))

/-- One soft edge. `standard` is 20 Hz at DC and the top 10% below Nyquist;
    `width` is an explicit width in Hz; `off` disables the edge. -/
inductive EdgeSpec where
  | standard
  | off
  | width (hz : Sig)

structure SoftEdges where
  dc : EdgeSpec := .standard
  nyquist : EdgeSpec := .standard

/-- Where each partial sits relative to the fundamental: partial `k` runs at
    `r_k · f0`. The column is assemble-time data, like a weight table's shape:
    the ring a rational family runs on (`Λ`) and the order the Nyquist cull
    relies on are both fixed by it. A partial with ratio `r` is still one sine at
    a known frequency, so the envelope `G(r·f0, τ)` and the soft edges apply
    unchanged. -/
inductive PartialRatios where
  /-- `r_k = k`, `k = 1..count`: the harmonic family (the default). -/
  | harmonic
  /-- `r_k = n_k / d_k`, positive naturals, one pair per partial. Every partial
      reads one shared ring position `X` whose wrap is `Λ = lcm d_k`
      fundamental cycles; partial `k`'s phase is `m_k·X` with
      `m_k = n_k·Λ/d_k`, an integer. The partials drift against each other but
      stay locked: the bundle recurs exactly every `Λ` fundamental cycles. An
      integer column (`Λ = 1`) is the harmonic construction, bit for bit. -/
  | rational (ratios : Array (Nat × Nat))
  /-- `r_k` any positive finite real. Partial `k` runs on its own Q0.32
      increment `trunc(r_k · f0·2³²/rate)`. -/
  | free (ratios : Array Float)
deriving Inhabited

/-- A partial family: partials `k = 1..count` at `r_k·freq` (`r_k = k` unless
    `ratios` says otherwise), weights `a_k`. `freq` must be uniform. `phase`
    (cycles) offsets the fundamental; partial `k` is offset by `r_k·phase`, so
    one anchor keeps every harmonic partial phase-locked. -/
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
  /-- The 0 Hz partial's (real) weight `a_0`, added once outside the partial
      loop as `a_0·Re G(0)`. Only a `cosine` family has one (a sine is zero at
      0 Hz), and only with the DC edge `off` (the edge would fade it to 0). -/
  dc : Option Sig := none
  /-- The frequency-ratio column (`ratioFamilySig`). Non-harmonic ratios need
      a `table` of weights; the column is sorted by ratio at assemble time,
      weights with it. -/
  ratios : PartialRatios := .harmonic

/-- A complex spectral envelope `G(f, τ)` as a closed form. `prepare` turns the
    envelope's control signals and the family's (signed) fundamental into the
    scalars the per-partial gain needs; they are computed once per sample,
    before the partial loop. `gain` returns `(Re G, Im G)` from those scalars,
    the partial's frequency ratio `r_k = f_k/f0` (float; `k` in a harmonic
    family) and the partial frequency `r_k·f0` (Hz, signed). -/
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

/-- The 0 Hz partial `a_0·Re G(0)`, given each envelope law with its prepared
    scalars. -/
private def zeroHzTerm (a0 : Sig) (laws : Array (EnvelopeLaw × Array Sig)) :
    BuildM Sig := do
  let zero ← lit 0
  let mut g0? : Option (Sig × Sig) := none
  for (law, prepared) in laws do
    let gi ← law.gain prepared zero zero
    g0? := some (← match g0? with
      | some acc => cmulPair acc gi
      | none => pure gi)
  match g0? with
  | some (gre, _) => mul a0 gre
  | none => pure a0

/-- The extra ring bits `E` of a rational family: 0 when every ratio is an
    integer (`Λ = 1`, the harmonic construction exactly), else 16, the most a
    wrapping 64-bit product holds next to the clock's fraction bits
    (`32 + E + F = 64`). -/
def ratioRingBits (lcm : Nat) : Nat := if lcm == 1 then 0 else 64 - 32 - clockFracBits

/-- One partial of a ratio column, as sorted: its ratio as a double, the
    integer ring multiplier `m = n·Λ/d` (rational only), its weight, and a key
    pair that orders it exactly (`n/d`, or the double itself for free ratios,
    with the authored index as the tiebreak). -/
private structure RatioEntry where
  ratio : Float
  multiplier : Nat
  weight : Sig
  num : Nat
  den : Nat
  authored : Nat
deriving Inhabited

private def RatioEntry.lt (rational : Bool) (a b : RatioEntry) : Bool :=
  let less := if rational then a.num * b.den < b.num * a.den else a.ratio < b.ratio
  let equal := if rational then a.num * b.den == b.num * a.den else a.ratio == b.ratio
  less || (equal && a.authored < b.authored)

/-- `#{k : column[k] ≤ t}` for an ascending column of `count` entries, by a
    binary search of `⌈log₂(count+1)⌉` reads (`lo` stays the count of entries
    known `≤ t`; a probe past the end reads the last entry and is refused by
    the bound). The result is an `int`. -/
def countAtMost (column : Sig) (count : Nat) (t : Sig) : BuildM Sig := do
  let mut step := 1
  while step * 2 ≤ count do step := step * 2
  let mut lo ← litI 0
  let last ← litI (count - 1)
  let countI ← litI count
  let one ← litI 1
  while step > 0 do
    let cand ← add lo (← litI step)
    let over ← gt cand countI
    let idx ← selectE over last (← sub cand one)
    let r ← index column idx
    lo ← selectE over lo (← selectE (← gt r t) lo cand)
    step := step / 2
  pure lo

/-- The Nyquist trip count of a family with an ascending ratio column `column`
    (`count` entries): `#{k : r_k ≤ Nyquist/max(|f0|, Nyquist/(K+1))}`, an
    `int`. The floor on `|f0|` keeps the threshold finite at `f0 = 0`. -/
def ratioTripCount (column : Sig) (count : Nat) (f0abs nyquist : Sig) : BuildM Sig := do
  let floorF0 ← div nyquist (← lit (count + 1))
  let f0floor ← selectE (← gt f0abs floorF0) f0abs floorF0
  countAtMost column count (← div nyquist f0floor)

/-- A family whose ratio column is not the harmonic default
    (`PartialRatios.rational` / `.free`): the same bank, envelope and soft edges
    as `familySig`, with partial `k` at `r_k·f0`.

    * **rational** `n/d`: `Ψ` is the position on a ring of `2^(32+E+F)` units
      per `Λ` fundamental cycles, `X ≡ (inc·⌊τ⌋ + off)·2^F + inc·frac(τ)` with
      `inc = trunc(f0·2^(32+E)/(Λ·rate))`, `off = trunc(phase·2^(32+E)/Λ)`.
      Partial `k`'s Q32 phase is `⌊m_k·X / 2^(E+F)⌋ mod 2³²`: one floor, every
      partial on the same `X`, so the lock `d_k·θ_k/n_k = d_j·θ_j/n_j` holds at
      every τ and the bundle recurs whenever `X` does. The product wraps mod
      2⁶⁴, which keeps bits `E+F..E+F+31` exact (`32+E+F ≤ 64`).
    * **free** `r`: partial `k`'s Q32 phase is `⌊inc_k·τ⌋ + off_k mod 2³²` with
      its own `inc_k = trunc(r_k·(f0·2³²/rate))`, `off_k = trunc(r_k·phase·2³²)`.

    The column is sorted ascending at assemble time (weights with it), so with a
    uniform `f0` the partials below Nyquist are a prefix and the cull is still a
    trip count: `#{k : r_k ≤ Nyquist/|f0|}`, found by a binary search per
    sample. -/
private def ratioFamilySig (f : PartialFamily) (clock : Clock)
    (envelopes : Array (EnvelopeLaw × Array Sig)) : BuildM Sig := do
  let some weightsIn := (match f.weights with
      | .table ws => some ws
      | _ => none)
    | throw s!"partial family '{f.name}': a family with a ratio column takes a weight table (one weight per partial, sorted with its ratio); a closed form in the partial number has no meaning once the column is reordered"
  if weightsIn.size != f.count then
    throw s!"partial family '{f.name}': {weightsIn.size} weights for {f.count} partials"
  unless ← sigIsUniform f.freq do
    throw s!"partial family '{f.name}': the fundamental must be uniform (constant within a kernel call); a moving fundamental is a phase-modulated family, not yet supported"
  if let some phase := f.phase then
    unless ← sigIsUniform phase do
      throw s!"partial family '{f.name}': the phase offset must be uniform"
  if let some live := f.liveCount then
    unless ← sigIsUniform live do
      throw s!"partial family '{f.name}': the live partial count must be uniform"
  -- ── the column: validated, the ring fixed, sorted ──
  let (rational, lcm, entries) ← match f.ratios with
    | .harmonic => throw s!"partial family '{f.name}': unreachable (harmonic column in the ratio lowering)"
    | .rational pairs => do
        if pairs.size != f.count then
          throw s!"partial family '{f.name}': {pairs.size} ratios for {f.count} partials"
        for (n, d) in pairs do
          if n == 0 || d == 0 then
            throw s!"partial family '{f.name}': the ratio {n}/{d} is not positive; ratios are positive n/d"
        let lcm := pairs.foldl (fun acc (_, d) => Nat.lcm acc d) 1
        let entries ← (pairs.zip weightsIn).mapIdxM fun i ((n, d), w) => do
          let m := n * (lcm / d)
          if m ≥ 2 ^ 53 then
            throw s!"partial family '{f.name}': the ratio {n}/{d} over the common period Λ = {lcm} needs the ring multiplier {m}, beyond the 2^53 a table carries exactly"
          pure ({ ratio := n.toFloat / d.toFloat, multiplier := m, weight := w,
                  num := n, den := d, authored := i } : RatioEntry)
        pure (true, lcm, entries)
    | .free rs => do
        if rs.size != f.count then
          throw s!"partial family '{f.name}': {rs.size} ratios for {f.count} partials"
        for r in rs do
          unless r > 0.0 && !r.isNaN && !r.isInf do
            throw s!"partial family '{f.name}': the ratio {r} is not a positive finite number"
        let entries := (rs.zip weightsIn).mapIdx fun i (r, w) =>
          ({ ratio := r, multiplier := 0, weight := w, num := 0, den := 0,
             authored := i } : RatioEntry)
        pure (false, 1, entries)
  let sorted := entries.qsort (RatioEntry.lt rational)
  let ringBits := ratioRingBits lcm
  let ratioTable ← arr (← sorted.mapM fun e => litExact e.ratio)
  let weightTable ← arr (sorted.map (·.weight))
  let multTable? ← if rational then
      pure (some (← arr (← sorted.mapM fun e => lit (e.multiplier : Int))))
    else pure none
  -- ── uniforms and per-sample scalars (computed before the region) ──
  let sr ← sampleRate
  let fracBits ← clockFracLit
  let thi ← rshift clock fracBits
  let tlo ← bitAnd clock (← clockFracMaskLit)
  let mask32 ← lit 4294967295
  -- the regime's loop-invariant phase inputs (ints split to stay exact in the
  -- float columns that carry them)
  let phaseInputs ← if rational then do
      let ringOne ← lit (2 ^ (32 + ringBits))
      let denom ← if lcm == 1 then pure sr else mul (← lit lcm) sr
      let inc ← toIntE (← div (← mul f.freq ringOne) denom)
      let turns ← mul inc thi
      let turns ← match f.phase with
        | some phase => do
            let scaled ← if lcm == 1 then mul phase ringOne
              else div (← mul phase ringOne) (← lit lcm)
            add turns (← toIntE scaled)
        | none => pure turns
      let wrapped ← bitAnd turns (← lit (2 ^ (32 + ringBits) - 1))
      let x ← add (← lshift wrapped fracBits) (← mul inc tlo)
      let x ← if 32 + ringBits + clockFracBits < 64 then
          bitAnd x (← lit (2 ^ (32 + ringBits + clockFracBits) - 1))
        else pure x
      -- X's two 32-bit words (each exact in a double)
      let hi ← bitAnd (← rshift x (← lit 32)) mask32
      let lo ← bitAnd x mask32
      pure #[hi, lo]
    else do
      let incScale ← div (← mul f.freq (← lit 4294967296)) sr
      let offScale ← match f.phase with
        | some phase => mul phase (← lit 4294967296)
        | none => lit 0
      pure #[thi, tlo, incScale, offScale]
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
  -- Trip count: the column is ascending, so the partials at or below Nyquist
  -- are a prefix, `#{k : r_k ≤ Nyquist/max(|f0|, Nyquist/(K+1))}`.
  let nyquistCount? ← match nyquistInv? with
    | none => pure none
    | some _ => do
        pure (some (← toFloatE (← ratioTripCount ratioTable f.count f0abs nyquist)))
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
  let fixedSlots := phaseInputs.size
  let mut invariants : Array Sig := phaseInputs ++ #[f0abs, f.freq]
  if let some inv := nyquistInv? then invariants := invariants.push nyquist |>.push inv
  if let some inv := dcInv? then invariants := invariants.push inv
  let mut envSlots : Array (EnvelopeLaw × Nat × Nat) := #[]
  for (law, controls) in envelopes do
    let prepared ← law.prepare controls f.freq
    envSlots := envSlots.push (law, invariants.size, prepared.size)
    invariants := invariants ++ prepared
  let invTables ← invariants.mapM fun value => arr #[value]
  let zeroIdx ← lit 0
  let invReads ← invTables.mapM fun table => index table zeroIdx
  -- ── the per-partial body ──
  let k ← loopIdx familyBinderId
  let rk ← index ratioTable k
  let phaseK ← match multTable? with
    | some multTable => do
        let m ← toIntE (← index multTable k)
        let x ← add (← lshift (← toIntE invReads[0]!) (← lit 32)) (← toIntE invReads[1]!)
        bitAnd (← rshift (← mul m x) (← lit (ringBits + clockFracBits))) mask32
    | none => do
        let incK ← toIntE (← mul rk invReads[2]!)
        let offK ← toIntE (← mul rk invReads[3]!)
        let turns ← add (← mul incK (← toIntE invReads[0]!)) offK
        let x ← add (← lshift (← bitAnd turns mask32) fracBits)
          (← mul incK (← toIntE invReads[1]!))
        bitAnd (← rshift x fracBits) mask32
  let q30 ← lit 1073741824
  let weight ← index weightTable k
  let fk ← mul rk invReads[fixedSlots + 1]!
  let fkAbs ← mul rk invReads[fixedSlots]!
  let mut edge? : Option Sig := none
  let mut slot := fixedSlots + 2
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
        let gi ← law.gain (invReads.extract start (start + size)) rk fk
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
  let tables := #[weightTable, ratioTable] ++ (multTable?.map (#[·])).getD #[] ++ invTables
  bankSum f.count tables contribution liveCount? familyBinderId

/-- The family's region, given its (already warped) clock and the envelopes
    applied to it, each with its control signals already emitted. -/
def familySig (f : PartialFamily) (clock : Clock)
    (envelopes : Array (EnvelopeLaw × Array Sig)) : BuildM Sig := do
  if f.dc.isSome then
    unless f.wave == .cosine do
      throw s!"partial family '{f.name}': a 0 Hz term needs the cosine projection (a sine is zero at 0 Hz)"
    unless f.edges.dc matches .off do
      throw s!"partial family '{f.name}': a 0 Hz term with a DC edge; the edge would fade it to zero, so set the DC edge off"
  match f.ratios with
  | .harmonic => pure ()
  | _ => return ← ratioFamilySig f clock envelopes
  if f.count == 0 then
    -- No partials; the 0 Hz term, if any, still plays (as it does at a live
    -- count of 0).
    return ← match f.dc with
      | none => lit 0
      | some a0 => do
          let laws ← envelopes.mapM fun (law, controls) => do
            pure (law, ← law.prepare controls f.freq)
          zeroHzTerm a0 laws
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
    | .law _ | .closedForm .. | .complex .. => pure none
  let columnTables ← match f.weights with
    | .closedForm columns _ _ => columns.mapM fun column => do
        if column.size != f.count then
          throw s!"partial family '{f.name}': a weight column of {column.size} entries for {f.count} partials"
        arr column
    | .complex tables _ => tables.mapM fun table => do
        if table.isEmpty then
          throw s!"partial family '{f.name}': an empty weight table"
        arr table
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
  let mut envPrepared : Array (Array Sig) := #[]
  for (law, controls) in envelopes do
    let prepared ← law.prepare controls f.freq
    envSlots := envSlots.push (law, invariants.size, prepared.size)
    envPrepared := envPrepared.push prepared
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
  let (weight, weightIm?) ← match f.weights, weightTable? with
    | .table _, some table => pure (← index table k, none)
    | .law w, _ => pure (← w kf, none)
    | .closedForm _ _ w, _ => do
        let colReads ← columnTables.mapM fun table => index table k
        pure (← w colReads (invReads.extract weightInputStart (weightInputStart + weightInputs.size)) n kf, none)
    | .complex _ w, _ => do
        let (re, im) ← w columnTables k n kf
        pure (re, some im)
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
  let projected ← if envelopes.isEmpty && weightIm?.isNone then
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
      -- A complex weight is one more complex factor of the partial.
      if let some wim := weightIm? then
        g? := some (← match g? with
          | some acc => cmulPair (weight, wim) acc
          | none => pure (weight, wim))
      let (gre, gim) := g?.getD (zero, zero)
      let s ← div (← toFloatE (← fixedSinCycSig phaseK)) q30
      let c ← div (← toFloatE (← fixedCosCycSig phaseK)) q30
      match f.wave with
      | .sine => add (← mul gre s) (← mul gim c)
      | .cosine => sub (← mul gre c) (← mul gim s)
  let shaped ← match edge? with
    | some e => mul e projected
    | none => pure projected
  let contribution ← if weightIm?.isSome then pure shaped else mul weight shaped
  let tables := (weightTable?.map (#[·])).getD #[] ++ columnTables ++ invTables
  let region ← bankSum f.count tables contribution liveCount? familyBinderId
  match f.dc with
  | none => pure region
  | some a0 =>
    -- The 0 Hz partial, per sample, outside the region.
    add region (← zeroHzTerm a0 ((envelopes.zip envPrepared).map fun (e, p) => (e.1, p)))

/-- The saw's weights `1/k`, `k = 1..count`, as an exact coefficient table. -/
def sawWeights (count : Nat) : BuildM PartialWeights := do
  let ws ← (Array.range count).mapM fun i => litExact (1.0 / (i + 1).toFloat)
  pure (.table ws)

end Tropical.EmitArrow
