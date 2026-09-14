import Tropical.EmitArrow.Modal.Block
import Tropical.EmitArrow.Modal.OrientedRealize

/-!
# EmitArrow.Modal.BlockRealize — rendering block rows (slice Phase 2)

A block row `{z₁..z_k; n₁..n_k}` renders as `Σ_m n_m · exp[z_m..z_k](d)`, the
suffix divided differences of `e^{zd}`. Each suffix is an existing row family
or the one new one:

* `exp[z_k]`            — a plain `ModalMode` (the fixed Q datapath, as today);
* `exp[z_{k−1}, z_k]`   — a `PairedMode` (`c·e^{z_k d}·d·cexpm1((z_{k−1}−z_k)d)`,
                          the float terminal lane `pairedSig`);
* `exp[z₁, z₂, z₃]`     — `TripleMode`, the one body this module adds.

A row whose nodes are all ONE expression is a confluent pole of multiplicity
`k`: `exp[z,…,z](d) = d^{k−m}/(k−m)!·e^{zd}`, i.e. polynomial degree — the
`deg` machinery the unrolled bank already renders, at any multiplicity.

A PAST row (slice Phase 4) realizes the same families on the mirrored clock
at the physical poles `ν = −z'` with the reflected coefficients
`(−1)^{k−m+1}·n_m`; the strike sample carries the continuous value `h(0)`.

## The triple body

Anchor at the third node: with `u = (z₁−z₃)·d`, `v = (z₂−z₃)·d`,

    exp[z₁,z₂,z₃](d) = e^{z₃d} · d² · Φ(u, v)

and `Φ` has two lanes, selected per sample by `selectE` (both lanes evaluated,
dead divisors swapped to 1 — the `cexpm1` discipline):

* SERIES, when `max(|u|², |v|², |u−v|²) < 0.01`:
  `Φ = Σ_{n≤10} h_n(u,v)/(n+2)!`, `h_n` the complete homogeneous symmetric
  sums (`h_0 = 1`, `h_n = u·h_{n−1} + vⁿ`). Division-free; exact at `u = v = 0`
  (the triple collision). Truncation < 3e-18 inside the lane.
* DIRECT, otherwise — one level of the divided-difference recurrence over
  size-2 factors, choosing the symmetric form whose divisor is LARGEST so no
  ordering of the nodes can put a near-zero gap in a live divisor:
      A = (e^{v}·cexpm1(u−v) − cexpm1(v)) / u
      B = (cexpm1(u) − cexpm1(v)) / (u − v)
      C = (e^{v}·cexpm1(u−v) − cexpm1(u)) / v
  (all three equal `Φ`; the numerators cancel by at most one digit when the
  chosen divisor is ≥ 0.1).

The `z₃` carrier is the float terminal carrier (`expSig` envelope × the exact
Q0.32 rotator), so the phase argument never leaves the circle; `u`, `v` use
raw pole differences — bounded by the cluster tolerance, so a raw float angle
is fine there (the same convention as `modalBankSigTableDD`'s `z`). Validated
against a 60-digit reference at every gap incl. 0 (`demos/block_carrier_body.py`).
-/

namespace Tropical.EmitArrow.Block

open Tropical.Ir
open Tropical.Exact (DyadicI CplxDI)
open Tropical.EmitArrow.Oriented (natE)

/-- The size-3 row: `c · exp[z₁, z₂, z₃](d)`. Poles in pole form `(−σ, ω)`. -/
structure TripleMode where
  z1 : CplxE
  z2 : CplxE
  z3 : CplxE
  c : CplxE

/-- Coefficient columns of a triple bank: the `z₃` rotator increment and
    damping, the two raw differences `z₁−z₃`, `z₂−z₃`, and the complex coeff. -/
structure TripleBankCols where
  count : Nat
  live? : Option Sig := none
  idxId : Nat := 0
  incr3 : Sig
  sigma3 : Sig
  duRe : Sig
  duIm : Sig
  dvRe : Sig
  dvIm : Sig
  cre : Sig
  cim : Sig

structure TripleModeSym where
  incr3 : Sig
  sigma3 : Sig
  duRe : Sig
  duIm : Sig
  dvRe : Sig
  dvIm : Sig
  cre : Sig
  cim : Sig

def tripleBankCols (rows : Array TripleMode) (live? : Option Sig := none) :
    BuildM TripleBankCols := do
  let twoPi ← twoPiE
  let twoPow32 ← lit 4294967296
  let sr ← sampleRate
  let incr3 ← arr (← rows.mapM fun r => do
    let frequency ← div r.z3.2 twoPi
    let scaled ← mul frequency twoPow32
    div scaled sr)
  let sigma3 ← arr (← rows.mapM fun r => neg r.z3.1)
  let duRe ← arr (← rows.mapM fun r => sub r.z1.1 r.z3.1)
  let duIm ← arr (← rows.mapM fun r => sub r.z1.2 r.z3.2)
  let dvRe ← arr (← rows.mapM fun r => sub r.z2.1 r.z3.1)
  let dvIm ← arr (← rows.mapM fun r => sub r.z2.2 r.z3.2)
  let cre ← arr (rows.map fun r => r.c.1)
  let cim ← arr (rows.map fun r => r.c.2)
  pure { count := rows.size, live?, incr3, sigma3, duRe, duIm, dvRe, dvIm, cre, cim }

def bankFoldTriple (cols : TripleBankCols) (body : TripleModeSym → BuildM Sig) :
    BuildM Sig := do
  let k ← loopIdx cols.idxId
  let incr3 ← index cols.incr3 k
  let sigma3 ← index cols.sigma3 k
  let duRe ← index cols.duRe k
  let duIm ← index cols.duIm k
  let dvRe ← index cols.dvRe k
  let dvIm ← index cols.dvIm k
  let cre ← index cols.cre k
  let cim ← index cols.cim k
  let contribution ← body { incr3, sigma3, duRe, duIm, dvRe, dvIm, cre, cim }
  bankSum cols.count
    #[cols.incr3, cols.sigma3, cols.duRe, cols.duIm, cols.dvRe, cols.dvIm, cols.cre, cols.cim]
    contribution cols.live? cols.idxId

private def cscaleE (s : Sig) (z : CplxE) : BuildM CplxE := do
  pure (← mul s z.1, ← mul s z.2)

private def cselectE (condition : Sig) (a b : CplxE) : BuildM CplxE := do
  pure (← selectE condition a.1 b.1, ← selectE condition a.2 b.2)

private def cnormSqE (z : CplxE) : BuildM Sig := do
  add (← mul z.1 z.1) (← mul z.2 z.2)

/-- `e^z` for a small-argument complex `z` in float (`expSig` × the float
    `cosSig`/`sinSig` polynomials). -/
private def cexpFloatE (z : CplxE) : BuildM CplxE := do
  let env ← expSig z.1
  pure (← mul env (← cosSig z.2), ← mul env (← sinSig z.2))

/-- `(e^z − 1)/z`, lane-safe: the direct quotient when `|z|² ≥ 0.01` (its
    divisor swapped to 1 on the other lane), the Horner series otherwise. -/
private def cexpm1LaneE (z : CplxE) (ez : CplxE) : BuildM CplxE := do
  let threshold ← litF 0.01
  let big ← gt (← cnormSqE z) threshold
  let one ← natE 1
  let safe ← cselectE big z one
  let direct ← cdivE (← csubE ez one) safe
  let series ← cexpm1SeriesE z
  cselectE big direct series

/-- The series lane of `Φ`: `Σ_{n=0}^{10} h_n(u,v)/(n+2)!`. -/
private def secondFactorSeriesE (u v : CplxE) : BuildM CplxE := do
  let mut h ← natE 1
  let mut vPow ← natE 1
  let mut total ← cscaleE (← litF 0.5) h
  for n in [1:11] do
    vPow ← cmulE vPow v
    h ← caddE (← cmulE u h) vPow
    let coefficient ← litF (1.0 / (Oriented.factorial (n + 2)).toFloat)
    total ← caddE total (← cscaleE coefficient h)
  pure total

/-- `Φ(u, v)` with `exp[z₁,z₂,z₃](d) = e^{z₃d}·d²·Φ` — both lanes, per-sample select. -/
private def secondFactorE (u v : CplxE) : BuildM CplxE := do
  let uv ← csubE u v
  let nu ← cnormSqE u
  let nv ← cnormSqE v
  let nuv ← cnormSqE uv
  let threshold ← litF 0.01
  -- series iff every pairwise gap is small
  let uSmall ← binary .lt nu threshold
  let vSmall ← binary .lt nv threshold
  let uvSmall ← binary .lt nuv threshold
  let seriesLane ← binary .and (← binary .and uSmall vSmall) uvSmall
  -- the three symmetric direct forms, divisors swapped to 1 when small;
  -- `e^{u−v} = e^u / e^v` (one complex division instead of a third
  -- exp·cos·sin — the Metal shader compiler's budget is the tighter one)
  let eu ← cexpFloatE u
  let ev ← cexpFloatE v
  let euv ← cdivE eu ev
  let cu ← cexpm1LaneE u eu
  let cv ← cexpm1LaneE v ev
  let cuv ← cexpm1LaneE uv euv
  let one ← natE 1
  let evCuv ← cmulE ev cuv
  let safeU ← cselectE (← gt nu threshold) u one
  let safeV ← cselectE (← gt nv threshold) v one
  let safeUV ← cselectE (← gt nuv threshold) uv one
  let candidateA ← cdivE (← csubE evCuv cv) safeU
  let candidateB ← cdivE (← csubE cu cv) safeUV
  let candidateC ← cdivE (← csubE evCuv cu) safeV
  -- choose the form with the LARGEST divisor
  let aBest ← binary .and (← binary .gte nu nuv) (← binary .gte nu nv)
  let bBest ← binary .gte nuv nv
  let direct ← cselectE aBest candidateA (← cselectE bBest candidateB candidateC)
  let series ← secondFactorSeriesE u v
  cselectE seriesLane series direct

/-- Terminal realization of triple rows: `Re(c · e^{z₃d} · d² · Φ(u,v))` per
    row, one banked reduction, gated causal. With a `landing`, the weight
    `c·d²·Φ·env` lands at `2^(28−k)` and multiplies the exact Q2.30 rotator in
    i64 — the fixed datapath every plain mode uses (slice Phase 6); without
    one, the float carrier. -/
def tripleSig (rows : Array TripleMode) (clkInt anchorSamples : Sig)
    (landing? : Option LandExp := none) : BuildM Sig := do
  if rows.isEmpty then return ← lit 0
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let cols ← tripleBankCols rows
  let weightOf := fun (row : TripleModeSym) => do
    let u ← cscaleE dSec (row.duRe, row.duIm)
    let v ← cscaleE dSec (row.dvRe, row.dvIm)
    let factor ← secondFactorE u v
    let dSq ← mul dSec dSec
    let scaled ← cscaleE dSq factor
    let weight ← cmulE (row.cre, row.cim) scaled
    let env ← expSig (← neg (← mul row.sigma3 dSec))
    let increment ← toIntE row.incr3
    let phaseQ ← modePhaseQFromIncr increment clkRel
    pure (← cscaleE env weight, phaseQ)
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  match landing? with
  | none =>
    let value ← bankFoldTriple cols fun row => do
      let (weight, phaseQ) ← weightOf row
      let q30 ← lit 1073741824
      let carrierCos ← div (← toFloatE (← fixedCosCycSig phaseQ)) q30
      let carrierSin ← div (← toFloatE (← fixedSinCycSig phaseQ)) q30
      let product ← cmulE weight (carrierCos, carrierSin)
      pure product.1
    selectE afterStrike value zero
  | some landing =>
    let landingScale ← landing.scale
    let landingShift ← landing.shift
    let bankQ ← bankFoldTriple cols fun row => do
      let (weight, phaseQ) ← weightOf row
      let wCre ← toIntE (← mul weight.1 landingScale)
      let wCim ← toIntE (← mul weight.2 landingScale)
      let real ← mul wCre (← fixedCosCycSig phaseQ)
      let imag ← mul wCim (← fixedSinCycSig phaseQ)
      rshift (← sub real imag) landingShift
    let output ← fixedOutQ 30 bankQ
    selectE afterStrike output zero

-- ── Landing bounds (option E for the divided-difference families) ─────────────

/-- `|c|` bounded by `|Re c| + |Im c|`, as `modeWeightBoundSig` does. -/
private def ampBoundSig (c : CplxE) : BuildM Sig := do
  add (← absE c.1) (← absE c.2)

private def minSig (a b : Sig) : BuildM Sig := do
  selectE (← gt a b) b a

private def eulerF : Float := 2.718281828459045

/-- The static/dynamic landing exponent of a family from per-row sup bounds:
    `boundD` on the exact carrier when the row's poles and coefficient fold
    (`none` = no finite sup), else `boundSig` as an s0 expression. -/
private def familyLandExp {α : Type} (rows : Array α)
    (boundD : Array (Option DyadicI) → α → Option (Option DyadicI))
    (boundSig : α → BuildM Sig) : BuildM LandExp := do
  let constants := sigConstTable (← get).exprs
  let mut mx := DyadicI.zero
  let mut unbounded := false
  let mut allConst := true
  for row in rows do
    match boundD constants row with
    | some (some b) => mx := DyadicI.max mx b
    | some none => unbounded := true
    | none => allConst := false
  if allConst then return .static (if unbounded then 28 else landK mx)
  let zero ← lit 0
  let maxSig ← rows.foldlM (fun acc row => do
    let b ← boundSig row
    selectE (← gt acc b) acc b) zero
  LandExp.dynamicOf maxSig

/-- `sup_d |c·e^{νd}·d·cexpm1((λ−ν)d)| ≤ |c|/(e·σ_min)`, `σ_min = min(σ_λ, σ_ν)`
    (`|(e^{λd}−e^{νd})/(λ−ν)| ≤ d·e^{−σ_min d}`, the mean-value form). -/
def pairedLandExp (rows : Array PairedMode) : BuildM LandExp :=
  familyLandExp rows
    (fun constants row =>
      match sigConstDFrom? constants row.c.1, sigConstDFrom? constants row.c.2,
            sigConstDFrom? constants row.lam.1, sigConstDFrom? constants row.nu.1 with
      | some cr, some ci, some lr, some nr =>
        let amp := DyadicI.mul (CplxDI.abs (CplxDI.mkI cr ci)) DyadicI.one
        let sigmaMin := DyadicI.min (DyadicI.neg lr) (DyadicI.neg nr)
        if !DyadicI.certGt sigmaMin DyadicI.zero then some none
        else some (some (DyadicI.div amp (DyadicI.mul Tropical.Exact.DyadicI.eulerI sigmaMin)))
      | _, _, _, _ => none)
    (fun row => do
      let amp ← ampBoundSig row.c
      let sigmaMin ← minSig (← neg row.lam.1) (← neg row.nu.1)
      div amp (← mul (← litF eulerF) sigmaMin))

/-- `sup_d |c·exp[z₁,z₂,z₃](d)| ≤ |c|·sup_d d²/2·e^{−σ_min d} = 2|c|/(e·σ_min)²`
    (Hermite–Genocchi: the second divided difference of `e^{zd}` is `d²/2` times
    a mean of `e^{zd}` over the nodes' simplex). -/
def tripleLandExp (rows : Array TripleMode) : BuildM LandExp :=
  familyLandExp rows
    (fun constants row =>
      match sigConstDFrom? constants row.c.1, sigConstDFrom? constants row.c.2,
            sigConstDFrom? constants row.z1.1, sigConstDFrom? constants row.z2.1,
            sigConstDFrom? constants row.z3.1 with
      | some cr, some ci, some r1, some r2, some r3 =>
        let amp := CplxDI.abs (CplxDI.mkI cr ci)
        let sigmaMin := DyadicI.min (DyadicI.min (DyadicI.neg r1) (DyadicI.neg r2)) (DyadicI.neg r3)
        if !DyadicI.certGt sigmaMin DyadicI.zero then some none
        else
          let es := DyadicI.mul Tropical.Exact.DyadicI.eulerI sigmaMin
          some (some (DyadicI.div (DyadicI.mul (DyadicI.ofNat 2) amp) (DyadicI.mul es es)))
      | _, _, _, _, _ => none)
    (fun row => do
      let amp ← ampBoundSig row.c
      let sigmaMin ← minSig (← minSig (← neg row.z1.1) (← neg row.z2.1)) (← neg row.z3.1)
      let es ← mul (← litF eulerF) sigmaMin
      div (← mul (← lit 2) amp) (← mul es es))

-- ── The block terminal ────────────────────────────────────────────────────────

/-- Block rows routed into their realizable families, per arm. `atZero` is the
    value at the strike sample: `0` for a single-sided spine (today's causal
    convention) and, once a past arm exists, the continuous value `h(0) =
    Σ_{future rows} n_k` (every same-side product of two or more factors has
    relative degree ≥ 2, so the bilateral response is continuous at 0). -/
structure BlockTerminal where
  plain : Array ModalMode := #[]
  paired : Array PairedMode := #[]
  triple : Array TripleMode := #[]
  pastPlain : Array ModalMode := #[]
  pastPaired : Array PairedMode := #[]
  pastTriple : Array TripleMode := #[]
  atZero : CplxE

private structure Families where
  plain : Array ModalMode := #[]
  paired : Array PairedMode := #[]
  triple : Array TripleMode := #[]

/-- Route one row's `(nodes, coeffs)` — already in the realizer's frame (physical
    poles, reflection applied) — into its families. -/
private def routeRow (fam : Families) (nodes coeffs : Array CplxE) (confluent : Bool) :
    BuildM Families := do
  let k := nodes.size
  if confluent then
    -- exp[z,…,z] (k−m+1 copies) = d^{k−m}/(k−m)!·e^{zd}
    let some z := nodes[0]? | pure fam
    let mut fam := fam
    for (n, m) in coeffs.zipIdx do
      let degree := k - 1 - m
      let amp ← Oriented.scaledNatQuotient n 1 (Oriented.factorial degree)
      fam := { fam with plain := fam.plain.push (← modeOfE z amp degree) }
    pure fam
  else
    match nodes.toList, coeffs.toList with
    | [z1], [n1] => pure { fam with plain := fam.plain.push (← modeOfE z1 n1) }
    | [z1, z2], [n1, n2] =>
        pure { fam with
          paired := fam.paired.push { lam := z1, nu := z2, c := n1 },
          plain := fam.plain.push (← modeOfE z2 n2) }
    | [z1, z2, z3], [n1, n2, n3] =>
        pure { fam with
          triple := fam.triple.push { z1, z2, z3, c := n1 },
          paired := fam.paired.push { lam := z2, nu := z3, c := n2 },
          plain := fam.plain.push (← modeOfE z3 n3) }
    | _, _ => throw s!"block terminal: a simple row of size {k} has no realizable body"

/-- Route each row. A future row is realized as it stands. A past row is
    realized on the mirrored clock at the PHYSICAL poles `ν = −z'` with
    coefficients `(−1)^{k−m+1}·n_m` (the anti-causal sign of the two-sided
    transform times the divided-difference reflection). A simple row larger
    than the cap cannot reach here (`decompose` splits it) and is a build error. -/
def BlockTerminal.ofRows (rows : Array BlockRow) : BuildM BlockTerminal := do
  let mut future : Families := {}
  let mut past : Families := {}
  let mut atZero ← Oriented.natE 0
  let anyPast := rows.any (·.orientation == .past)
  for row in rows do
    match row.orientation with
    | .future =>
        future ← routeRow future row.nodes row.coeffs row.confluent
        if anyPast then
          if let some nk := row.coeffs.back? then atZero ← caddE atZero nk
    | .past =>
        let k := row.nodes.size
        let nodes ← row.nodes.mapM cnegE
        -- 0-based m: (−1)^{k−m} — negate the last coefficient (m = k−1) always
        let coeffs ← row.coeffs.zipIdx.mapM fun (n, m) =>
          if (k - m) % 2 == 1 then cnegE n else pure n
        past ← routeRow past nodes coeffs row.confluent
  pure { plain := future.plain, paired := future.paired, triple := future.triple,
         pastPlain := past.plain, pastPaired := past.paired, pastTriple := past.triple,
         atZero }

/-- Render the terminal: every family on the fixed i64 datapath. The plain
    families through `Bank.realizeSig` (per-bank option-E landing, the past
    arm on the mirrored clock, `atZero` at the strike); the paired families
    through `modalBankSigTableDD` and the triple families through `tripleSig`,
    each landed at its own per-bank exponent from the family's sup bound
    (`pairedLandExp`, `tripleLandExp`) — no admission cap, the exponent
    absorbs the range (slice Phase 6). -/
def BlockTerminal.realizeSig (terminal : BlockTerminal) (clkInt anchorSamples : Sig)
    (count? : Option Sig := none) : BuildM Sig := do
  let bank : Oriented.Bank :=
    { future := terminal.plain, past := terminal.pastPlain, atZero := terminal.atZero }
  let plainSig ← bank.realizeSig clkInt anchorSamples count?
  let twoPow32 ← lit 4294967296
  let anchorFixed ← mul anchorSamples twoPow32
  let anchorQ ← toIntE anchorFixed
  let two ← lit 2
  let twiceAnchor ← mul two anchorQ
  let mirroredClock ← sub twiceAnchor clkInt
  let futurePaired ← modalBankSigTableDD terminal.paired clkInt anchorSamples none
    (some (← pairedLandExp terminal.paired))
  let pastPaired ← modalBankSigTableDD terminal.pastPaired mirroredClock anchorSamples none
    (some (← pairedLandExp terminal.pastPaired))
  let futureTriple ← tripleSig terminal.triple clkInt anchorSamples
    (some (← tripleLandExp terminal.triple))
  let pastTriple ← tripleSig terminal.pastTriple mirroredClock anchorSamples
    (some (← tripleLandExp terminal.pastTriple))
  let paired ← add futurePaired pastPaired
  let triple ← add futureTriple pastTriple
  add (← add plainSig paired) triple

-- ── Materialization (the gauge seam) ──────────────────────────────────────────

/-- A paired row as two collected modes: `c·exp[λ,ν] = c/(λ−ν)·e^{λd} −
    c/(λ−ν)·e^{νd}`. The `1/Δ` this forms is exactly the collected fold's — the
    status-quo floor, taken only where a consumer needs a plain bank. -/
private def pairedCollected (pair : PairedMode) : BuildM (Array ModalMode) := do
  let gap ← csubE pair.lam pair.nu
  let amp ← cdivE pair.c gap
  pure #[← modeOfE pair.lam amp, ← modeOfE pair.nu (← cnegE amp)]

/-- A triple row as three collected modes (the Lagrange form). -/
private def tripleCollected (row : TripleMode) : BuildM (Array ModalMode) := do
  let nodes := #[row.z1, row.z2, row.z3]
  nodes.zipIdx.mapM fun (z, i) => do
    let mut denominator ← Oriented.natE 1
    for (w, j) in nodes.zipIdx do
      if i != j then denominator ← cmulE denominator (← csubE z w)
    modeOfE z (← cdivE row.c denominator)

/-- The terminal as a plain oriented bank — every divided-difference row
    collected. This is the block carrier's only structure-dropping step, and it
    runs only where the next consumer is nonlinear in the whole bank (a gauge);
    a confluent row is already degree modes and drops nothing. -/
def BlockTerminal.toBank (terminal : BlockTerminal) : BuildM Oriented.Bank := do
  let collect := fun (plain : Array ModalMode) (paired : Array PairedMode)
      (triple : Array TripleMode) => do
    let mut modes := plain
    for pair in paired do modes := modes ++ (← pairedCollected pair)
    for row in triple do modes := modes ++ (← tripleCollected row)
    pure modes
  let future ← collect terminal.plain terminal.paired terminal.triple
  let past ← collect terminal.pastPlain terminal.pastPaired terminal.pastTriple
  pure { future, past, atZero := terminal.atZero }

/-- A plain oriented bank as a retained factor: its future modes as an
    exactly-forward proper kernel, its past modes as an exactly-reversed one,
    in parallel — the input of the segment after a gauge. -/
def bankKernel (bank : Oriented.Bank) : BuildM ModalKernelExpr := do
  let zero ← lit 0
  let one ← lit 1
  let mut branches : Array ModalKernelExpr := #[]
  if !bank.future.isEmpty then branches := branches.push (.proper (.oriented bank.future zero))
  if !bank.past.isEmpty then branches := branches.push (.proper (.oriented bank.past one))
  match branches.toList with
  | [] => pure (.proper (.oriented #[] zero))
  | [single] => pure single
  | _ => pure (.parallel branches)

end Tropical.EmitArrow.Block
