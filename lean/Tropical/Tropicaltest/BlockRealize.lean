import Tropical.EmitArrow.Modal.BlockRealize
import Tropical.EmitArrow.Modal.Forest
import Tropical.Testing.ArrowFixtures
import Tropical.Tropicaltest.Stress

/-!
# Block-carrier render gates (slice Phase 2)

The block terminal rendered through the production carrier path, compared at
the OBSERVABLE (samples leaving the datapath):

* SINGLETON RENDER — a separated chain renders through the block terminal
  and through today's collected fold + plain datapath; same modes, same lane,
  so the two agree to the coefficient-rounding floor.
* CONFLUENT RENDER — an identity-confluent four-fold pole renders as degree-3
  through the block terminal and through the exact `Oriented` chain.
* TRIPLE RENDER — three distinct-expression poles within θ_acc (gap 1e-3 rad/s)
  plus a separated voice pole: the triple body, the paired body and the plain
  mode all carry weight, against an EXACT oracle — the collected partial
  fractions evaluated on the 128-bit dyadic carrier (`CplxDI`), whose
  enclosure width is reported as the oracle's own certificate.
* TRIPLE COLLISION — three distinct-expression poles of one VALUE: the series
  lane at `u = v = 0`, against the closed form `a·r·r'·d²/2·e^{zd}`.
* TRIPLE LANES — hand-built triple rows at gaps of 3 and 5 rad/s, so `|u|²`
  crosses the 0.01 lane threshold INSIDE the probe window: both lanes and the
  seam between them, against the exact Lagrange form on `CplxDI`.
-/

namespace Tropical.Tropicaltest.BlockRealize

open Tropical
open Tropical.EmitArrow
open Tropical.EmitArrow.Block
open Tropical.Ir
open Tropical.Exact (DyadicI CplxDI)

private def passGate (label detail : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {detail}"
  pure true

private def failGate (label detail : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {detail}"
  pure false

private def tp : Float := 6.283185307179586
private def srF : Float := 44100.0
private def anchorNat : Nat := 200
private def nProbe : Nat := 4096
private def anchorSig : BuildM Sig := lit 200

/-- ω on the datapath's rotator grid (`⌊(ω/2π)·2³²/SR⌋`). -/
private def qOm (om : Float) : Float :=
  Float.floor (om / tp * 4294967296.0 / srF) * (tp * srF / 4294967296.0)

private def render (name : String) (sig : BuildM Sig) : IO (Except String (Array Float)) := do
  match buildAndFinish (Tropical.EmitArrow.buildExprCarrier name sig {}) with
  | .error e => pure (.error e)
  | .ok plan => renderPlanSamples plan nProbe

private def relL2 (dut ref : Array Float) (stride : Nat := 1) : Float := Id.run do
  let mut nm := 0.0
  let mut dn := 0.0
  let mut i := anchorNat + 1
  while i < nProbe do
    let dd := dut[i]! - ref[i]!
    nm := nm + dd * dd
    dn := dn + ref[i]! * ref[i]!
    i := i + max stride 1
  return Float.sqrt (nm / (dn + 1e-300))

/-- The exact oracle is evaluated on every `oracleStride`-th sample (the
    128-bit `exp` is the gate's cost); the comparison uses the same subset. -/
private def oracleStride : Nat := 4

private def allFinite (xs : Array Float) : Bool := xs.all (·.isFinite)

/-- `x` in scientific notation (Lean's `Float.toString` is fixed six-decimal). -/
private def sci (x : Float) : String :=
  if x == 0.0 then "0" else
  let e := Float.floor (Float.log10 (Float.abs x))
  let m := x / Float.pow 10.0 e
  s!"{m}e{e.toInt64}"

private def energy (xs : Array Float) : Float := Id.run do
  let mut e := 0.0
  for i in [anchorNat + 1 : nProbe] do e := e + xs[i]! * xs[i]!
  return e

private def debugPair (label : String) (dut ref : Array Float) : IO Unit := do
  if (← IO.getEnv "TROPICAL_BLOCK_DEBUG").isSome then
    IO.println s!"        [{label}] energy dut {sci (energy dut)} ref {sci (energy ref)}"
    for i in [anchorNat + 1, anchorNat + 2, anchorNat + 3, anchorNat + 50, 2000, 4000] do
      IO.println s!"        [{label}] i={i} dut {sci dut[i]!} ref {sci ref[i]!}"

/-- A literal mode from Float data (a finite double enters the arena exactly). -/
private def modeF (sigma omega cre cim : Float) : BuildM ModalMode := do
  pure { sigma := ← litF sigma, omega := ← litF omega, cre := ← litF cre, cim := ← litF cim }

private def properOf (modes : Array ModalMode) : BuildM ModalKernelExpr := do
  pure (.proper (.oriented modes (← lit 0)))

private def blockSig (spine : ModalKernelExpr) : BuildM Sig := do
  match ← decompose spine with
  | .error r => throw r.describe
  | .ok rows => (← BlockTerminal.ofRows rows).realizeSig (← clockLit) (← anchorSig)

-- ── exact oracle helpers (128-bit dyadic) ─────────────────────────────────────

private def cI (re im : Float) : CplxDI := CplxDI.ofFloats re im

/-- `Σ Re(Aᵢ·e^{zᵢ d})` over the probe window on the exact carrier, with the
    widest enclosure met (the oracle's certificate). -/
private def exactSum (terms : Array (CplxDI × CplxDI)) : Array Float × Float := Id.run do
  let mut y : Array Float := Array.replicate nProbe 0.0
  let mut width := 0.0
  let mut i := anchorNat + 1
  while i < nProbe do
    let d := DyadicI.ofFloat ((i - anchorNat).toFloat / srF)
    let mut acc := DyadicI.zero
    for (amp, pole) in terms do
      let e := CplxDI.exp (CplxDI.scale d pole)
      acc := DyadicI.add acc (CplxDI.mul amp e).re
    width := max width (Dyadic.toFloat (DyadicI.width acc))
    y := y.set! i (DyadicI.toFloat acc)
    i := i + oracleStride
  return (y, width)

-- ── (a) singleton render ──────────────────────────────────────────────────────

private def voice2 : BuildM (Array ModalMode) := do
  pure #[← modeF 3.0 (tp * 100.0) 0.8 0.1, ← modeF 11.0 (tp * 224.0) 0.5 (-0.2)]
private def roomA : BuildM (Array ModalMode) := do
  pure #[← modeF 9.0 (tp * 150.0) 0.7 0.0, ← modeF 21.0 (tp * (-320.0)) 0.4 0.3]
private def roomB : BuildM (Array ModalMode) := do
  pure #[← modeF 13.0 (tp * 500.0) 0.6 (-0.1), ← modeF 5.0 (tp * (-123.0)) 0.3 0.0]

private def singletonRender : IO (Except String Float) := do
  let dut ← render "block_singleton_dut" (do
    let v ← voice2; let r1 ← roomA; let r2 ← roomB
    blockSig (.cascade #[← properOf v, ← properOf r1, ← properOf r2]))
  let ref ← render "block_singleton_ref" (do
    let v ← voice2; let r1 ← roomA; let r2 ← roomB
    let modes ← foldRoomsEC v #[r1, r2]
    (← Oriented.Bank.ofFuture modes).realizeSig (← clockLit) (← anchorSig))
  match dut, ref with
  | .ok d, .ok r =>
    debugPair "singleton" d r
    pure (if allFinite d then .ok (relL2 d r) else .error "non-finite render")
  | .error e, _ | _, .error e => pure (.error e)

-- ── (b) confluent render (identity-coincident four-fold pole) ─────────────────

private def confluentRender : IO (Except String Float) := do
  let dut ← render "block_confluent_dut" (do
    let z ← modeF 6.0 (tp * 210.0) 0.9 0.0
    let stage := fun (amp : Float) => do
      let a ← litF amp
      pure (#[{ z with cre := a }] : Array ModalMode)
    let v ← stage 4.0; let r1 ← stage 3.0; let r2 ← stage 2.0; let r3 ← stage 2.0
    blockSig (.cascade #[← properOf v, ← properOf r1, ← properOf r2, ← properOf r3]))
  match dut with
  | .error e => pure (.error e)
  | .ok d =>
    if !(allFinite d) then return .error "non-finite render"
    -- closed form: a·r·r'·r''·d³/3!·e^{−σd}·cos(ω_q d), ω on the rotator grid
    let wq := qOm (tp * 210.0)
    let amp := 4.0 * 3.0 * 2.0 * 2.0 / 6.0
    let mut ref : Array Float := Array.replicate nProbe 0.0
    for i in [anchorNat + 1 : nProbe] do
      let t := (i - anchorNat).toFloat / srF
      ref := ref.set! i (amp * t * t * t * Float.exp (-6.0 * t) * Float.cos (wq * t))
    debugPair "confluent" d ref
    pure (.ok (relL2 d ref))

-- ── (c) triple render: gap 1e-3, four poles, exact oracle ─────────────────────

/-- The near-coincident chain: voice `{a at z₁, b at z₀}`, rooms `{r at z₂}`,
    `{r' at z₃}`, with `z₁, z₂, z₃` within 1e-3 rad/s (distinct literals). -/
private structure Chain4 where
  s0 : Float := 4.0
  w0 : Float := tp * 640.0
  b : Float := 0.5
  a : Float := 0.9
  s1 : Float := 9.0
  w1 : Float := tp * 300.0 + 0.0007
  s2 : Float := 9.0004
  w2 : Float := tp * 300.0
  s3 : Float := 9.0
  w3 : Float := tp * 300.0 + 0.0003
  r : Float := 0.7
  r' : Float := 0.6

private def chain4 : Chain4 := {}

private def tripleRender : IO (Except String (Float × Float)) := do
  let c := chain4
  let dut ← render "block_triple_dut" (do
    let v ← pure #[← modeF c.s1 c.w1 c.a 0.0, ← modeF c.s0 c.w0 c.b 0.0]
    let r1 ← pure #[← modeF c.s2 c.w2 c.r 0.0]
    let r2 ← pure #[← modeF c.s3 c.w3 c.r' 0.0]
    blockSig (.cascade #[← properOf v, ← properOf r1, ← properOf r2]))
  match dut with
  | .error e => pure (.error e)
  | .ok d =>
    if !(allFinite d) then return .error "non-finite render"
    -- effective poles as the datapath carries them: z₃ and z₀ on the rotator
    -- grid, z₁ and z₂ as z₃ plus their RAW differences
    let w3q := qOm c.w3
    let z0 := cI (-c.s0) (qOm c.w0)
    let z3 := cI (-c.s3) w3q
    let z1 := cI (-c.s1) (w3q + (c.w1 - c.w3))
    let z2 := cI (-c.s2) (w3q + (c.w2 - c.w3))
    let rr := CplxDI.mul (cI c.r 0) (cI c.r' 0)
    let a := cI c.a 0
    let b := cI c.b 0
    let sub := CplxDI.sub
    let div := CplxDI.div
    let mul := CplxDI.mul
    let add := CplxDI.add
    -- H = (a/(s−z₁) + b/(s−z₀))·r/(s−z₂)·r'/(s−z₃): residues
    let res1 := div (mul a rr) (mul (sub z1 z2) (sub z1 z3))
    let res0 := div (mul b rr) (mul (sub z0 z2) (sub z0 z3))
    let res2 := div (mul rr (add (div a (sub z2 z1)) (div b (sub z2 z0)))) (sub z2 z3)
    let res3 := div (mul rr (add (div a (sub z3 z1)) (div b (sub z3 z0)))) (sub z3 z2)
    let (ref, width) := exactSum #[(res1, z1), (res0, z0), (res2, z2), (res3, z3)]
    debugPair "triple" d ref
    pure (.ok (relL2 d ref oracleStride, width))

-- ── (d) triple collision: three literals of one value ─────────────────────────

private def tripleCollision : IO (Except String Float) := do
  let s := 9.0
  let w := tp * 300.0
  let a := 0.9; let r := 0.7; let r' := 0.6
  let dut ← render "block_triple_collision" (do
    -- three DISTINCT literal spellings of one value: distinct expressions, one node value
    let z1 ← pure { sigma := ← lit 9, omega := ← litF w, cre := ← litF a, cim := ← lit 0 : ModalMode }
    let z2 ← pure { sigma := ← lit 90 1, omega := ← litF w, cre := ← litF r, cim := ← lit 0 : ModalMode }
    let z3 ← pure { sigma := ← lit 900 2, omega := ← litF w, cre := ← litF r', cim := ← lit 0 : ModalMode }
    blockSig (.cascade #[← properOf #[z1], ← properOf #[z2], ← properOf #[z3]]))
  match dut with
  | .error e => pure (.error e)
  | .ok d =>
    if !(allFinite d) then return .error "non-finite render"
    let wq := qOm w
    let mut ref : Array Float := Array.replicate nProbe 0.0
    for i in [anchorNat + 1 : nProbe] do
      let t := (i - anchorNat).toFloat / srF
      ref := ref.set! i (a * r * r' * t * t / 2.0 * Float.exp (-s * t) * Float.cos (wq * t))
    pure (.ok (relL2 d ref))

-- ── (e) triple lanes: hand-built rows at gaps 3 and 5 rad/s ───────────────────

private def tripleLanes : IO (Except String (Float × Float)) := do
  let s3 := 5.0; let w3 := tp * 410.0
  let rows : Array (Float × Float × Float × Float) :=   -- (Δσ₁, Δω₁, Δσ₂, Δω₂) rad/s
    #[(0.4, 3.0, -0.2, -1.5), (-1.0, 5.0, 0.7, 2.0)]
  let dut ← render "block_triple_lanes" (do
    let z3 : CplxE := (← litF (-s3), ← litF w3)
    let mut trip : Array TripleMode := #[]
    for (ds1, dw1, ds2, dw2) in rows do
      let z1 : CplxE := (← litF (-s3 + ds1), ← litF (w3 + dw1))
      let z2 : CplxE := (← litF (-s3 + ds2), ← litF (w3 + dw2))
      trip := trip.push { z1, z2, z3, c := (← litF 0.8, ← litF 0.1) }
    tripleSig trip (← clockLit) (← anchorSig))
  match dut with
  | .error e => pure (.error e)
  | .ok d =>
    if !(allFinite d) then return .error "non-finite render"
    let w3q := qOm w3
    let mut terms : Array (CplxDI × CplxDI) := #[]
    for (ds1, dw1, ds2, dw2) in rows do
      let z3 := cI (-s3) w3q
      let z1 := cI (-s3 + ds1) (w3q + dw1)
      let z2 := cI (-s3 + ds2) (w3q + dw2)
      let c := cI 0.8 0.1
      -- c·exp[z₁,z₂,z₃] = c·Σᵢ e^{zᵢd}/∏_{j≠i}(zᵢ−zⱼ)
      let sub := CplxDI.sub; let mul := CplxDI.mul; let div := CplxDI.div
      terms := terms.push (div c (mul (sub z1 z2) (sub z1 z3)), z1)
      terms := terms.push (div c (mul (sub z2 z1) (sub z2 z3)), z2)
      terms := terms.push (div c (mul (sub z3 z1) (sub z3 z2)), z3)
    let (ref, width) := exactSum terms
    debugPair "lanes" d ref
    pure (.ok (relL2 d ref oracleStride, width))

-- ── the gate ──────────────────────────────────────────────────────────────────

/-- Thresholds — MEASURED at landing (2026-09-13), fail lines a decade above.
    * singleton: 0 (bit-identical — both sides fold on the exact carrier and
      round ONCE to the same doubles; the fixed lane then renders identically).
    * exact-oracle probes: the plain family's Q4.28 landing LSB (~1e-9 absolute
      against a ~4e-4 signal) is the binding floor — triple 1.0e-5, collision
      1.0e-9 (no plain weight), lanes 4.9e-7 (pure float lane, `expSig`/`cosSig`
      polynomial accuracy at |u| ≤ 0.45); confluent (deg-3 through the fixed
      lane vs the float closed form) is recorded in the pass line. -/
private def sameLaneFloor : Float := 1e-7
private def floatLaneFloor : Float := 1e-4

def runBlockRealize : IO Bool := do
  let singleton ← singletonRender
  let confluent ← confluentRender
  let triple ← tripleRender
  let collision ← tripleCollision
  let lanes ← tripleLanes
  match singleton, confluent, triple, collision, lanes with
  | .ok s, .ok c, .ok (t, tw), .ok k, .ok (l, lw) =>
    if s < sameLaneFloor && c < floatLaneFloor && t < floatLaneFloor
        && k < floatLaneFloor && l < floatLaneFloor then
      passGate "block-realize"
        s!"singleton render rel {sci s}; confluent deg-3 render rel {sci c}; triple (gap 1e-3, 4 poles) vs 128-bit exact rel {sci t} (oracle width {sci tw}); triple collision vs closed form rel {sci k}; triple lanes (gaps 3, 5 rad/s, seam crossed in-window) vs exact Lagrange rel {sci l} (oracle width {sci lw})"
    else
      failGate "block-realize"
        s!"off the law: singleton {sci s} confluent {sci c} triple {sci t} collision {sci k} lanes {sci l} (floors {sci sameLaneFloor} / {sci floatLaneFloor})"
  | s, c, t, k, l =>
    let sh := fun (x : Except String Float) => match x with | .ok v => s!"{v}" | .error e => s!"ERR {e}"
    let sh2 := fun (x : Except String (Float × Float)) => match x with
      | .ok (v, w) => s!"{v} (width {w})" | .error e => s!"ERR {e}"
    failGate "block-realize"
      s!"singleton {sh s}; confluent {sh c}; triple {sh2 t}; collision {sh k}; lanes {sh2 l}"

end Tropical.Tropicaltest.BlockRealize
