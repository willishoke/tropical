import Tropical.Tropicaltest.Modal

/-!
# Tropical.Tropicaltest.SeamSweep — the seam-atom witness harness

The standing witness for the modal seam-crossing atoms, the executable form of
`design/seam-atom-contract.md`. One apparatus, generic over registered
`SeamAtom` instances, that re-checks each atom's stated law over its stated
admission region on every `make validate` — the repair for the artisanal
point-witness chain the atoms shipped with.

The law, at the observable: an atom's realized signal equals the realized
oracle, pointwise, within its error model, over its admission region. The
oracle is ONE thing — direct high-resolution quadrature of the composition
integral `y(d) = Σ a·r·∫₀^d e^{ν(d−s)}·e^{μ·φ(s)} ds` over the voice×room pole
pairs — and every atom is that oracle at a different warp `φ` (`id` for the
residue atoms, the pitch bloom for the Γ-bridge). NOT the Poisson pole ladder
(float64-dead above the fundamental); quadrature is the oracle, the ladder is
the story.

Two instruments, kept both: the frozen goldens (`Tropicaltest/Modal.lean`) pin
HISTORY; this sweep checks the LAW over the region.
-/

open Tropical
open Tropical.Plan
open Tropical.EmitArrow
open Tropical.Ir (Arena ProgramIdx)

namespace Tropical.Tropicaltest.SeamSweep

/-- 2π. -/
private def tp : Float := 6.283185307179586
/-- Sample rate (the rotator grid is `⌊(ω/2π)·2³²/SR⌋`). -/
private def srF : Float := 44100.0
/-- Strike anchor in samples (`clkRel > 0` ⟺ sample index > this). -/
private def anchorNat : Nat := 200
private def anchorSig : Sig := lit 200
/-- Probe-window length (samples). Short (~0.09 s): the law holds pointwise, so
    a short window at the datapath floor is a fair witness and keeps validate
    fast. -/
private def nProbe : Nat := 4096
/-- The carrier output sink gain (`defaultSinkGain`, Plan.lean) — raw-math
    references must scale by it (house precedent: Slide/Stress). -/
private def sinkGain : Float := Tropical.Plan.defaultSinkGain.toFloat

-- ── Low-discrepancy sampling (Halton — deterministic, reproducible) ───────────

/-- Halton radical inverse of `i` in `base` — the low-discrepancy coordinate
    (deterministic and space-filling; `Math.random`/`Date` are banned and would
    break resume anyway). Bounded digit loop (`i < 2³²`). -/
private def haltonF (base i : Nat) : Float := Id.run do
  let mut f : Float := 1.0
  let mut r : Float := 0.0
  let mut n : Nat := i
  for _ in [0:32] do
    if n > 0 then
      f := f / base.toFloat
      r := r + f * (n % base).toFloat
      n := n / base
  return r

-- ── The pole/amp datum (all Float — the sweep bakes every pole) ───────────────

/-- One mode as Float pole/amp data: from it the harness derives BOTH the atom's
    `ModalMode` bank (`litF` fields) and the oracle's `CplxB` pole/amp — the two
    realizations off one datum. -/
structure SeamMode where
  sigma : Float           -- σ = −Re μ (decay ≥ 0)
  omega : Float           -- ω = Im μ (rad/s)
  are   : Float           -- Re A
  aim   : Float := 0.0    -- Im A
deriving Inhabited

def SeamMode.toModal (m : SeamMode) : ModalMode :=
  { sigma := litF m.sigma, omega := litF m.omega, cre := litF m.are, cim := litF m.aim }
def SeamMode.pole (m : SeamMode) : CplxB := ⟨-m.sigma, m.omega⟩
def SeamMode.amp  (m : SeamMode) : CplxB := ⟨m.are, m.aim⟩

/-- A mode from `(freq Hz, σ, real amp)`. -/
private def mkMode (fHz sigma amp : Float) : SeamMode :=
  { sigma, omega := tp * fHz, are := amp }

/-- A Halton-sampled mode over a frequency band (σ ∈ [0.3, 2.0], amp ∈ [0.4,
    1.0], small Im amp) — the low-discrepancy fill of the pole box. -/
private def haltonMode (i : Nat) (fLo fHi : Float) : SeamMode :=
  { sigma := 0.3 + 1.7 * haltonF 3 i
    omega := tp * (fLo + (fHi - fLo) * haltonF 2 i)
    are   := 0.4 + 0.6 * haltonF 5 i
    aim   := (haltonF 7 i - 0.5) * 0.4 }

-- ── The oracle: quadrature of the composition integral, generic over φ ────────

/-- ω quantized to the datapath's rotator grid, so the reference refines toward
    the frequencies the kernel actually carries (else the SR/2³² quantization
    reads as ~2.7e-5 error). -/
private def qOm (om : Float) : Float :=
  Float.floor (om / tp * 4294967296.0 / srF) * (tp * srF / 4294967296.0)

/-- `oracleSeam` — the reference semantics of a seam crossing: for each admitted
    voice×room pair, the prefix quadrature of
    `y(d) = a·r·∫₀^d e^{ν(d−s)}·e^{μ·φ(s)} ds`, `Re` taken, summed, sink-gain
    applied. Generic over `φ` (id → residue atoms; bloom → Γ-bridge) — the
    unification the island never wrote down. `admits` filters to the pairs the
    isolated config admits. Bloom composition itself is all-or-nothing; mixed
    refusal is covered separately by `runBloomSafety`. Composite Simpson per sample-interval (`hdiv`
    EVEN subintervals) — the O(h⁴) rule, needed because `e^{iωs}` at audio ω is
    oscillatory and trapezoid (h²) would undersample it. `hdiv=4` vs `hdiv=8`
    gives the h→h/2 self-distance that certifies the oracle itself. NOT the
    Poisson pole ladder (float64-dead above the fundamental). -/
private def oracleSeam (voice reverb : Array SeamMode) (phi : Float → Float)
    (admits : SeamMode → SeamMode → Bool) (hdiv : Nat) (nLen : Nat := nProbe) :
    Array Float := Id.run do
  let mut y : Array Float := Array.replicate nLen 0.0
  for v in voice do
    let mu : CplxB := ⟨-v.sigma, qOm v.omega⟩
    let a  : CplxB := ⟨v.are, v.aim⟩
    for r in reverb do
      if !(admits v r) then continue
      let nu : CplxB := ⟨-r.sigma, qOm r.omega⟩
      let rr : CplxB := ⟨r.are, r.aim⟩
      let c := (a.mul rr).scale sinkGain
      let h := 1.0 / (srF * hdiv.toFloat)
      let fint := fun (s : Float) => CplxB.exp ((mu.scale (phi s)).sub (nu.scale s))
      let mut jAcc : CplxB := ⟨0, 0⟩
      for i in [anchorNat + 1 : nLen] do
        let dBase := (i - 1 - anchorNat).toFloat / srF
        -- composite Simpson over [dBase, dBase + 1/SR]: weights 1,4,2,…,4,1
        let mut seg : CplxB := ⟨0, 0⟩
        for k in [0:hdiv + 1] do
          let w := if k == 0 || k == hdiv then 1.0 else if k % 2 == 1 then 4.0 else 2.0
          seg := seg.add ((fint (dBase + k.toFloat * h)).scale w)
        jAcc := jAcc.add (seg.scale (h / 3.0))
        let d := (i - anchorNat).toFloat / srF
        let yc := (CplxB.exp (nu.scale d)).mul jAcc
        y := y.set! i (y[i]! + c.re * yc.re - c.im * yc.im)
  return y

-- ── window metrics ────────────────────────────────────────────────────────────

private def relL2Win (xa xb : Array Float) (lo hi : Nat) : Float := Id.run do
  let mut nm := 0.0
  let mut dn := 0.0
  for i in [lo:hi] do
    let dd := xa[i]! - xb[i]!
    nm := nm + dd * dd
    dn := dn + xb[i]! * xb[i]!
  return Float.sqrt (nm / (dn + 1e-300))

private def energyWin (xs : Array Float) (lo hi : Nat) : Float := Id.run do
  let mut e := 0.0
  for i in [lo:hi] do e := e + xs[i]! * xs[i]!
  return e

private def allFinite (xs : Array Float) : Bool := xs.all (fun x => x.isFinite)

-- ── The contract as a structure ───────────────────────────────────────────────

/-- A `SeamAtom`: the contract one composition atom instantiates. The law lives
    at the observable (`realize`d signal vs `oracleSeam φ`), so the atom's
    representation is its own business. -/
structure SeamAtom where
  name : String
  /-- the oracle warp (`id` for residue atoms, the pitch bloom for Γ-bridge). -/
  phi : Float → Float
  /-- the algorithm + realizer, fused: voice×room banks → the atom's realized `Sig`. -/
  realize : Array SeamMode → Array SeamMode → Sig
  /-- the admission region, as an executable predicate (not a comment). -/
  admitsPair : SeamMode → SeamMode → Bool
  /-- `true`: an isolated non-admitted probe refuses the whole atom (the harness
      represents that isolated refusal as silence, and its oracle sums nothing).
      `false`: total/passive (composes every pair; the predicate marks only
      WHERE it is accurate, and the oracle sums all pairs). -/
  activeExclusion : Bool
  /-- the error model: the relative-L2 SNR threshold vs the oracle, above the
      datapath float-vs-fixed floor. -/
  snr : Float
  /-- interior configs, Halton-sampled over the admission region. -/
  interior : Array (Array SeamMode × Array SeamMode)
  /-- boundary probes: candidate isolated pairs at the admission edges; the
      harness tags each LIVE via `admitsPair` (admitted → accuracy; actively
      excluded isolated probe → refusal represented as silence; passively excluded
      → out of contract, skip). -/
  boundary : Array (SeamMode × SeamMode)

/-- Which pairs the isolated-probe oracle sums: admitted-only for an actively
    refusing atom, all pairs otherwise. Mixed all-or-nothing is tested elsewhere. -/
private def SeamAtom.oracleAdmits (atom : SeamAtom) : SeamMode → SeamMode → Bool :=
  if atom.activeExclusion then atom.admitsPair else (fun _ _ => true)

-- ── The three v1 instances (retrofit) ─────────────────────────────────────────

private def idWarp : Float → Float := fun s => s

/-- `residueCompose` — THE one compose seam (fork 3′ erasure). The EC and DD
    atoms MERGED: the realization is the PARTITIONED compose
    (`residueComposePartitioned` — cold couplings collected, hot couplings
    routed to the paired DD body by `couplingHot`, decided at build time), and
    the admission region is the UNION of the two half-regions the predecessors
    certified MINUS the stated cap refusal (`couplingRefused`: a lens fired but
    the paired range cap rejected the coupling back to the collected floor —
    reachable only at extreme Q, and excluded here rather than certified). The
    epistemic win this sprint exists for: the apparatus stops certifying two
    half-regions with a folklore boundary and certifies one whole, with its one
    true edge stated as admission; the old EC exclusion interior (`|a·r/Δ| ≥ 8`)
    is now ordinary served territory with hard probes inside it.

    History (kept because the boundary's meaning has flipped twice): the old EC
    atom's `|a·r/Δ| < 8` admission once guarded the i64 Q4.28 wrap, then —
    after option E's per-bank exponent removed the wrap — the near-coincidence
    accuracy floor. It is now the compiler's own ROUTING boundary
    (`ecddRailCeil`, with margin), so the sweep no longer states it as
    admission at all: both sides are served, each by its correct
    representation, and the probes assert accuracy on both. -/
private def composeAtom : SeamAtom :=
  { name := "residueCompose"
    phi := idWarp
    realize := fun v r =>
      let (plain, paired) :=
        residueComposePartitioned (v.map (·.toModal)) (r.map (·.toModal))
      let base := modalBankSigTable plain clockLit anchorSig
      if paired.isEmpty then base
      else add base (modalBankSigTableDD paired clockLit anchorSig)
    -- the UNION admission minus the STATED cap refusal. The old EC atom
    -- admitted `|a·r/Δ| < 8` and passively excluded the rest; the old DD atom
    -- was total; the merged seam serves everything it routes (per coupling,
    -- `couplingHot`) and REFUSES a lens-fired coupling the paired range cap
    -- rejects (`couplingRefused` — it renders collected at the status-quo
    -- floor, wrong near coincidence, so certifying it would be false). Passive
    -- exclusion: the refused pair still renders; it is out of contract here
    -- and pinned structurally by the ecdd-partition gate.
    admitsPair := fun v r => !couplingRefused v.toModal r.toModal
    activeExclusion := false
    -- one snr: both routes' floors sit under 2e-4 at this window (the trap
    -- note about region-dependent error models is satisfied by construction —
    -- if a route's floor ever rises past the shared snr, the probes on that
    -- side fail loudly rather than silently averaging).
    snr := 2e-4
    interior :=
      -- the old EC interiors (separated pairs — all route collected) …
      (Array.range 4).map (fun c =>
        (#[haltonMode (c*10+1) 220 900, haltonMode (c*10+2) 220 900],
         #[haltonMode (c*10+3) 1500 3300, haltonMode (c*10+4) 1500 3300]))
      -- … plus the old DD interiors (near-coincident offsets — these straddle
      -- θ_acc, so the sweep exercises BOTH sides of the routing boundary).
      ++ (Array.range 4).map (fun c =>
        let vo := #[haltonMode (c*10+1) 400 1200, haltonMode (c*10+2) 400 1200]
        let off := 0.6 * haltonF 2 (c+1)
        (vo, vo.map (fun m => { m with omega := m.omega + tp * off, sigma := m.sigma + 0.1 })))
    boundary :=
      -- the old EC accuracy edge (|Δ| ≈ 0.6 > θ_acc, |a·r/Δ| < 4 — still cold,
      -- the collected route's regression probes) …
      #[ (mkMode 800 0.4 0.8, { sigma := 1.0, omega := tp * 800, are := 0.6 })
       , (mkMode 500 0.5 0.7, { sigma := 1.1, omega := tp * 500, are := 0.7 })
      -- … the old DD coincidence probes (λ = ν exactly; the τ·e resonance) …
       , (mkMode 700 0.8 0.6, mkMode 700 0.8 0.5)
       , (mkMode 1100 0.5 0.7, { sigma := 0.5, omega := tp * 1100 + 0.02, are := 0.4 })
      -- … and the WIDENING (gate iii): hard probes INSIDE the old EC exclusion
      -- — configurations the apparatus previously certified for nobody.
      -- Deep inside the old exclusion (|a·r/Δ| ≈ 14 ≫ 8; Δ = 0.05 < θ_acc, so
      -- the ACCURACY lens routes it — the rail lens's own witness is below):
       , (mkMode 800 0.6 0.9, { sigma := 0.6, omega := tp * 800 + 0.05, are := 0.8 })
      -- the sub-grid detune (Δ = 1e-6 rad/s < the 6.5e-5 rotator quantum —
      -- UNREPRESENTABLE collected; the served-full upgrade's sharpest point):
       , (mkMode 500 0.8 0.7, { sigma := 0.8, omega := tp * 500 + 1e-6, are := 0.7 })
      -- the RAIL lens's own service region (its only discriminating law probe:
      -- Δ = 0.6 > θ_acc so the accuracy lens is OFF; |a·r|/Δ = 4/0.6 ≈ 6.7 > 4
      -- fires the rail lens; σ = 1.5 makes the damping arm the binding sup and
      -- clears the cap, |c|/(e·σ_min) ≈ 0.98 < 8 — routed, law asserted):
       , (mkMode 800 1.5 2.0, { sigma := 1.5, omega := tp * 800 + 0.6, are := 2.0 })
      -- the STATED refusal (sub-grid Δ fires the accuracy lens; σ = 0.02 is
      -- extreme Q, |c|/(e·σ_min) ≈ 18 ≥ 8 — the cap refuses, `admitsPair` is
      -- false, the harness skips it as out-of-contract; the ecdd-partition
      -- gate pins the refusal structurally):
       , (mkMode 400 0.02 1.0, { sigma := 0.02, omega := tp * 400 + 1e-6, are := 1.0 }) ] }

/-- The pitch bloom warp (shipped gong: β=0.05, g=1.8, scale 1 ⇒ B = β/g). -/
private def bloomBg : Float × Float := (0.05 / 1.8, 1.8)
private def bloomWarp : Float → Float := fun s => s + bloomBg.1 * (1.0 - Float.exp (-bloomBg.2 * s))

/-- `bloomCompose` — the Γ-bridge atom (`bloomed voice ⋙ reverb`), total over
    the crossing after atom four (WS-A4): the `|a| < ½` coincidence hole (a room
    pole tuned to a settled partial — the τ·e resonance) is admitted and accurate,
    the coincident divided difference bridging the CF branch (large `z`) to the
    series-DD branch (small `z`). Conditioning/depth/unsupported exclusions are
    typed whole-composition refusals. Baked-pole v1 (the sweep pole set is all literal). NOTE: the 0.09 s
    interior/boundary window only exercises the CF branch (which is accidentally
    stable at coincidence); the series-DD resonance lives in the tail, witnessed by
    the long-render checks in `runSeamSweep` — the `coincidence deep-tail` (a = 0
    EXACT, the `euler` limit branch) AND the `a≠0 coincidentCrossing tail` (the
    general `lgamma(a+1)/a` bridge, WS-AA audit) — plus the `small-κ` subtle-bloom
    direct-sum branch. -/
private def bloomAtom : SeamAtom :=
  let B := bloomBg.1
  let g := bloomBg.2
  { name := "bloomGamma"
    phi := bloomWarp
    realize := fun v r =>
      match bloomCompose (v.map (·.toModal)) (r.map (·.toModal)) B g with
      | some pairs => bloomComposedSig pairs clockLit anchorSig
      | none => litI 0
    admitsPair := fun v r => bloomAdmitsPair v.pole r.pole B g
    activeExclusion := true
    snr := 3e-4
    interior := #[
      -- gong-ish partials × room poles; wide freq gaps ⇒ |a| ≫ ½ (the crossing).
      (#[haltonMode 1 300 1100, haltonMode 2 1400 3200], #[haltonMode 3 150 400, haltonMode 4 500 900]),
      (#[haltonMode 11 300 1100, haltonMode 12 1400 3200], #[haltonMode 13 150 400, haltonMode 14 500 900]),
      (#[haltonMode 21 300 1100, haltonMode 22 1400 3200], #[haltonMode 23 150 400, haltonMode 24 500 900]),
      -- coincidence configs (WS-A4): room poles ON the partials (ω matched; a σ/tiny-ω
      -- offset sets |a| < ½), so each carries near-coincident τ·e pairs. The oracle
      -- sums all admitted pairs; the far cross pairs are ordinary crossings.
      (#[mkMode 520 0.4 0.9, mkMode 880 0.5 0.6],
       #[{ sigma := 0.7, omega := tp * 520, are := 0.5 }, { sigma := 0.9, omega := tp * 880 + 0.4, are := 0.4 }]),
      (#[mkMode 300 0.35 0.8, mkMode 1050 0.6 0.5],
       #[{ sigma := 0.55, omega := tp * 300 + 0.2, are := 0.5 }, { sigma := 0.9, omega := tp * 1050, are := 0.4 }]) ]
    boundary :=
      -- the ½ boundary is now a scheme CROSSOVER, not a cliff: probes straddle it and
      -- all assert accuracy (|a|≈0.53 shipped-side, |a|≈0.44 coincidence-side), plus
      -- a=0 EXACT (ω,σ matched) and a small spiral around it (the removable singularity).
      #[ (mkMode 1000 0.5 0.8, { sigma := 0.5, omega := tp * 1000 + 0.95, are := 0.4 })   -- |a|≈0.53
       , (mkMode 1000 0.5 0.8, { sigma := 0.5, omega := tp * 1000 + 0.80, are := 0.4 })   -- |a|≈0.44
       , (mkMode 700 0.5 0.8, { sigma := 0.5, omega := tp * 700, are := 0.4 })            -- a = 0 EXACT
       , (mkMode 700 0.5 0.8, { sigma := 0.62, omega := tp * 700, are := 0.4 })           -- |a|≈0.067 (real spiral)
       , (mkMode 700 0.5 0.8, { sigma := 0.5, omega := tp * 700 + 0.5, are := 0.4 })      -- |a|≈0.278 (imag spiral)
       , (mkMode 2600 0.9 0.5, mkMode 300 0.8 0.4) ] }

private def allAtoms : Array SeamAtom := #[composeAtom, bloomAtom]

-- ── The gate ──────────────────────────────────────────────────────────────────

/-- Render one config's atom Sig to samples. -/
private def renderConfig (arena : Arena) (name : String) (sig : Sig)
    (nLen : Nat := nProbe) : IO (Except String (Array Float)) := do
  match buildAndFinish (.ok (buildExprCarrier name sig arena)) with
  | .error e => pure (.error e)
  | .ok plan => renderPlanSamples plan nLen

/-- THE SEAM SWEEP: for each registered `SeamAtom`, re-check its law over its
    admission region (Halton interior configs) and at its edges (boundary
    probes: accuracy where admitted, graceful silence where actively excluded).
    Plus the composition laws the quadrature oracle makes nearly free: EC
    commutativity (the two-hats reading) and linear associativity of unbloomed
    composition (re-pinning the PoC differential as a standing witness). Fails
    `make validate` iff an atom drifts off its own stated law. -/
def runSeamSweep (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let lo := anchorNat + 1
  let hi := nProbe
  let mut ok := true
  IO.println "seam-atom sweep (law re-checked over the admission region — not frozen points):"
  for atom in allAtoms do
    let mut worst := 0.0
    let mut worstSelf := 0.0
    let mut nInt := 0
    -- interior: Halton configs over the admission region
    for h in [0:atom.interior.size] do
      let (v, r) := atom.interior[h]!
      match ← renderConfig arena s!"seam_{atom.name}_int{h}" (atom.realize v r) with
      | .error e => IO.println s!"        {atom.name} int{h} BUILD/RENDER: {firstLine e}"; ok := false
      | .ok dut =>
        let ref2 := oracleSeam v r atom.phi atom.oracleAdmits 8
        let ref1 := oracleSeam v r atom.phi atom.oracleAdmits 4
        let e2 := relL2Win dut ref2 lo hi
        let self := relL2Win ref1 ref2 lo hi
        worst := max worst e2; worstSelf := max worstSelf self; nInt := nInt + 1
        if !(allFinite dut) || e2 ≥ atom.snr then
          IO.println s!"        {atom.name} int{h} LAW VIOLATION: rel {e2} (self {self}) ≥ snr {atom.snr}"
          ok := false
    -- boundary: tag each candidate live via the admission predicate
    let mut nAdm := 0
    let mut nExc := 0
    let mut worstBnd := 0.0
    for b in [0:atom.boundary.size] do
      let (v, r) := atom.boundary[b]!
      let vs := #[v]; let rs := #[r]
      match ← renderConfig arena s!"seam_{atom.name}_bnd{b}" (atom.realize vs rs) with
      | .error e => IO.println s!"        {atom.name} bnd{b} BUILD/RENDER: {firstLine e}"; ok := false
      | .ok dut =>
        if atom.admitsPair v r then
          -- admitted edge: assert accuracy (hardest region — near exclusion)
          let ref2 := oracleSeam vs rs atom.phi atom.oracleAdmits 8
          let e2 := relL2Win dut ref2 lo hi
          worstBnd := max worstBnd e2; nAdm := nAdm + 1
          if !(allFinite dut) || e2 ≥ atom.snr then
            IO.println s!"        {atom.name} bnd{b} EDGE VIOLATION: rel {e2} ≥ snr {atom.snr}"
            ok := false
        else if atom.activeExclusion then
          -- actively excluded: graceful silence, never silent wrongness
          let en := energyWin dut lo hi
          nExc := nExc + 1
          if !(allFinite dut) || en > 1e-12 then
            IO.println s!"        {atom.name} bnd{b} NOT GRACEFUL: excluded pair rendered energy {en}"
            ok := false
        -- else: passive exclusion (out of contract) — skip
    IO.println s!"        {atom.name}: interior {nInt} configs worst-rel {worst} (oracle self {worstSelf}); boundary {nAdm} admitted worst-rel {worstBnd}, {nExc} gracefully excluded"
  -- ── composition laws (the quadrature oracle is association-blind) ──
  -- Law 1 — EC commutativity (the two-hats reading: source⋙filter ≡ filter∘filter):
  --   residueComposeEC(b₁,b₂) realized ≡ residueComposeEC(b₂,b₁) realized.
  let b1 : Array SeamMode := #[mkMode 300 0.4 0.8, mkMode 640 0.6 0.6]
  let b2 : Array SeamMode := #[mkMode 1700 0.5 0.7, mkMode 2400 0.8 0.5]
  let ecFwd := fun (x y : Array SeamMode) =>
    modalBankSigTable (residueComposeEC (x.map (·.toModal)) (y.map (·.toModal))) clockLit anchorSig
  match ← renderConfig arena "seam_law_comm_a" (ecFwd b1 b2),
        ← renderConfig arena "seam_law_comm_b" (ecFwd b2 b1) with
  | .ok da, .ok db =>
    let e := relL2Win da db lo hi
    if !(allFinite da) || e ≥ 8e-4 then
      IO.println s!"        law-EC-commute VIOLATION: EC(b₁,b₂) ≢ EC(b₂,b₁) rel {e}"; ok := false
    else IO.println s!"        law-EC-commute: filter∘filter symmetric — EC(b₁,b₂) ≡ EC(b₂,b₁) rel {e}"
  | _, _ => IO.println "        law-EC-commute BUILD/RENDER failed"; ok := false
  -- Law 3 — linear associativity of unbloomed composition (re-pins the PoC diff):
  --   (v∘r₁)∘r₂ ≡ v∘(r₁∘r₂). The right side folds the room chain (filter∘filter),
  --   the exact reassociation Deliverable A's bloom lowering rides.
  let vA : Array SeamMode := #[mkMode 420 0.5 0.9]
  let rA : Array SeamMode := #[mkMode 1300 0.6 0.6]
  let rB : Array SeamMode := #[mkMode 2600 0.7 0.5]
  let ecC := fun (x y : Array ModalMode) => residueComposeEC x y
  let left := ecC (ecC (vA.map (·.toModal)) (rA.map (·.toModal))) (rB.map (·.toModal))
  let right := ecC (vA.map (·.toModal)) (ecC (rA.map (·.toModal)) (rB.map (·.toModal)))
  match ← renderConfig arena "seam_law_assoc_l" (modalBankSigTable left clockLit anchorSig),
        ← renderConfig arena "seam_law_assoc_r" (modalBankSigTable right clockLit anchorSig) with
  | .ok dl, .ok dr =>
    let e := relL2Win dl dr lo hi
    if !(allFinite dl) || e ≥ 8e-4 then
      IO.println s!"        law-assoc VIOLATION: (v∘r₁)∘r₂ ≢ v∘(r₁∘r₂) rel {e}"; ok := false
    else IO.println s!"        law-assoc: (v∘r₁)∘r₂ ≡ v∘(r₁∘r₂) rel {e} — room-fold reassociation holds"
  | _, _ => IO.println "        law-assoc BUILD/RENDER failed"; ok := false
  -- ── coincidence deep-tail (WS-A4, atom four): the τ·e resonance rings PAST the
  -- CF/series-DD switch (z ≈ 1 at d = ln|κ|/g ≈ 1.64 s for the fundamental), which
  -- the 0.09 s interior/boundary window CANNOT reach — there the CF branch is
  -- accidentally stable even for the OLD atom, so a short-window witness is green
  -- on a coincidence the deep tail gets wrong (the point-witness disease this whole
  -- apparatus exists to cure). Render an a = 0 EXACT pair (room pole ON the settled
  -- partial, ν = μ — the removable 0/0) over ~2.2 s and re-check the law where the
  -- resonance actually lives. Without the coincident branch the series lane selects
  -- the singular Γ★/1(ν−μ) constants and the tail goes non-finite.
  let (B, g) := bloomBg
  let admD := fun (x y : SeamMode) => bloomAdmitsPair x.pole y.pole B g
  let nDeep : Nat := 98304                       -- ≈ 2.23 s at 44.1 k
  let deepLo : Nat := 74000                      -- ≈ 1.67 s > dSwitch: the series-DD region
  let vDeep : Array SeamMode := #[mkMode 110 0.2 1.0]                       -- lightly damped, rings long
  let rDeep : Array SeamMode := #[{ sigma := 0.2, omega := tp * 110, are := 0.5 }]  -- ν = μ EXACTLY ⇒ a = 0
  match ← renderConfig arena "seam_bloom_coinc_deep" (bloomAtom.realize vDeep rDeep) nDeep with
  | .error e => IO.println s!"        bloomGamma deep-tail BUILD/RENDER: {firstLine e}"; ok := false
  | .ok dut =>
    let refD := oracleSeam vDeep rDeep bloomWarp admD 8 nDeep
    let eDeep := relL2Win dut refD deepLo nDeep
    let eFull := relL2Win dut refD lo nDeep
    IO.println s!"        bloomGamma coincidence deep-tail (a = 0 EXACT, κ≈19, ~2.2 s): series-DD region rel {eDeep}, full rel {eFull}"
    if !(allFinite dut) || eDeep ≥ 1e-3 || eFull ≥ 1e-4 then
      IO.println s!"        bloomGamma DEEP-TAIL VIOLATION: series-DD rel {eDeep} (full {eFull}) — the τ·e resonance region is off the law (or non-finite)"
      ok := false
  -- ── coincidence SMALL-κ (WS-A4): a SUBTLE bloom (tiny B ⇒ |κ| ≪ |a+1|) makes
  -- dSwitch < 0, so the pair is ALWAYS on the series-DD branch and reads Φ(a,κ)
  -- directly from d = 0 — the region where the lgamma(a+1) bridge catastrophically
  -- cancels (both κ^{−Re a} terms → the tiny Φ(a,κ)≈−κ/(a+1)) and the direct Σdₙκⁿ
  -- sum must take over. A short window suffices (no long tail needed). Its own warp.
  -- Re(a) > 0 (voice MORE damped than the room, σ_μ > σ_ν) is what makes the bridge's
  -- κ^{−Re a} terms blow up and cancel — a pure-imaginary a would not expose it — and
  -- a LOW partial keeps κ tiny (κ ≈ |μ|·B ≈ 3.5e-4) so the garbage is unmistakable.
  let bSmall : Float := 1.67e-6
  let warpSmall : Float → Float := fun s => s + bSmall * (1.0 - Float.exp (-g * s))
  let admS := fun (x y : SeamMode) => bloomAdmitsPair x.pole y.pole bSmall g
  let vSmall : Array SeamMode := #[mkMode 55 0.8 1.0]                                       -- σ_μ = 0.8, κ ≈ 5.8e-4
  let rSmall : Array SeamMode := #[{ sigma := 0.5, omega := tp * 55 + 0.74, are := 0.5 }]  -- Re(a)≈0.17, |a|≈0.44
  let realizeSmall :=
    match bloomCompose (vSmall.map (·.toModal)) (rSmall.map (·.toModal)) bSmall g with
    | some pairs => bloomComposedSig pairs clockLit anchorSig
    | none => litI 0
  match ← renderConfig arena "seam_bloom_coinc_smallk" realizeSmall with
  | .error e => IO.println s!"        bloomGamma small-κ BUILD/RENDER: {firstLine e}"; ok := false
  | .ok dut =>
    let refS := oracleSeam vSmall rSmall warpSmall admS 8
    let eS := relL2Win dut refS lo hi
    IO.println s!"        bloomGamma small-κ coincidence (subtle bloom, |κ|≈6e-4, always series-DD): rel {eS}"
    if !(allFinite dut) || eS ≥ 3e-4 then
      IO.println s!"        bloomGamma SMALL-κ VIOLATION: rel {eS} — Φ(a,κ) bridge cancellation not caught by the direct-sum fallback"
      ok := false
  -- ── coincidence a≠0 series-DD tail (WS-AA audit): the a=0 deep-tail above hits
  -- bloomPhiKappaOverG's `euler` special case (aC.normSq < 1e-12, the removable-
  -- singularity limit); the GENERAL lgamma(a+1)/a divided-difference bridge — baked
  -- into k1SerDD for EVERY a≠0 coincidentCrossing pair (a room pole tuned NEAR but
  -- not ON a settled partial, the generic coincidence, a=0 being measure-zero) — was
  -- witnessed by no oracle. Render an a≠0 coincidentCrossing pair (0<|a|<½, |κ|≥|a+1|)
  -- PAST its dSwitch (≈1.69 s) and re-check the series-DD tail where k1SerDD (via the
  -- lgamma bridge) is the selected e^{νd} constant.
  let vAC : Array SeamMode := #[mkMode 110 0.2 1.0]                                  -- μ, σ small ⇒ rings long
  let rAC : Array SeamMode := #[{ sigma := 0.35, omega := tp * 110, are := 0.5 }]   -- ν≈μ ⇒ a≈−0.083 (real, ≠0)
  let acLo : Nat := 76000                                                            -- ≈1.72 s > dSwitch≈1.69 s
  match ← renderConfig arena "seam_bloom_coinc_aneq0" (bloomAtom.realize vAC rAC) nDeep with
  | .error e => IO.println s!"        bloomGamma a≠0 tail BUILD/RENDER: {firstLine e}"; ok := false
  | .ok dut =>
    let refAC := oracleSeam vAC rAC bloomWarp admD 8 nDeep
    let eAC := relL2Win dut refAC acLo nDeep
    let eFullAC := relL2Win dut refAC lo nDeep
    IO.println s!"        bloomGamma a≠0 coincidentCrossing tail (|a|≈0.083, |κ|≈19, general lgamma(a+1)/a bridge, ~2.2 s): series-DD region rel {eAC}, full rel {eFullAC}"
    if !(allFinite dut) || eAC ≥ 1e-3 || eFullAC ≥ 1e-4 then
      IO.println s!"        bloomGamma a≠0 TAIL VIOLATION: series-DD rel {eAC} (full {eFullAC}) — the general lgamma-bridge k1SerDD is off the law"
      ok := false
  if ok then passGate "seam-sweep" "every registered atom holds its law over its admission region (incl. the τ·e coincidence deep-tail, small-κ direct-sum, AND the a≠0 general lgamma-bridge tail); composition laws (EC-commute, assoc) pinned"
  else failGate "seam-sweep" "a registered seam atom drifted off its stated law — see the per-config lines above"

-- ── Deliverable A: gong⋙reverb(⋙reverb) audible from the patch surface ────────

/-- INDEPENDENT Float fold of two rooms into their filter-product bank (the
    two-hats reading of `room₁ ⋙ room₂`): partial fraction of `H₁·H₂` — each
    `ν₁` keeps residue `r₁·H₂(ν₁)`, each `ν₂` keeps `r₂·H₁(ν₂)` — computed here
    in `CplxB` off the RAW poles, a SEPARATE code path from `residueComposeEC`'s
    symbolic `CplxE` fold. So `chain-patch ≡ oracleSeam(voice, foldRoomsFloat)`
    witnesses the reassociation (fold-then-cross) against an independent fold,
    reusing the accurate single-crossing quadrature rather than a delicate
    difference form (amendment §3, law 2). Separated room poles (the collected
    `1/Δ`; near-coincident chain folds are the DD-fold follow-up). -/
private def foldRoomsFloat (room1 room2 : Array SeamMode) : Array SeamMode := Id.run do
  let poleOf := fun (r : SeamMode) => (⟨-r.sigma, r.omega⟩ : CplxB)   -- RAW (as EC folds)
  let ampOf  := fun (r : SeamMode) => (⟨r.are, r.aim⟩ : CplxB)
  let mut out : Array SeamMode := #[]
  for r1 in room1 do
    let mut h : CplxB := ⟨0, 0⟩
    for r2 in room2 do h := h.add ((ampOf r2).div ((poleOf r1).sub (poleOf r2)))
    let res := (ampOf r1).mul h
    out := out.push { sigma := r1.sigma, omega := r1.omega, are := res.re, aim := res.im }
  for r2 in room2 do
    let mut h : CplxB := ⟨0, 0⟩
    for r1 in room1 do h := h.add ((ampOf r1).div ((poleOf r2).sub (poleOf r1)))
    let res := (ampOf r2).mul h
    out := out.push { sigma := r2.sigma, omega := r2.omega, are := res.re, aim := res.im }
  return out

/-- Lower a `PatchGraph` to a rendered carrier through the production seam
    (`lowerGraph → normalize → emitTerm`). -/
private def renderGraph (arena : Arena) (name : String) (gr : PatchGraph) :
    IO (Except String (Array Float)) := do
  match lowerGraph gr with
  | .error e => pure (.error e)
  | .ok term =>
    let (out, _) := emitTerm (normalize term) {}
    renderConfig arena name out

/-- The verdict half of the gong⋙reverb gate (split out to keep each `do`-block
    shallow — a long single block hits the kernel's recursion limit). -/
private def gongVerdict (voice room1 room2 : Array SeamMode)
    (pS pC dS dC : Array Float) : IO Bool := do
  let lo := anchorNat + 1
  let hi := nProbe
  let (B, g) := bloomBg
  let adm := fun (v r : SeamMode) => bloomAdmitsPair v.pole r.pole B g
  let bitS := bitDiffCount pS dS
  let bitC := bitDiffCount pC dC
  -- audibility: causal (pre-strike energy ~0), nonzero, not blowing up
  let preE := energyWin pS 0 lo
  let eEarly := energyWin pS lo (lo + 1800)
  let eLate := energyWin pS (hi - 1800) hi
  -- law 2: chain ≡ oracle on the INDEPENDENTLY-folded room; single ≡ oracle
  let e2chain := relL2Win pC (oracleSeam voice (foldRoomsFloat room1 room2) bloomWarp adm 8) lo hi
  let e2single := relL2Win pS (oracleSeam voice room1 bloomWarp adm 8) lo hi
  IO.println s!"gong⋙reverb from the patch surface (fold the room chain, cross the bloom once):"
  IO.println s!"        route    patch ≡ direct atom: single bitDiff {bitS}, chain(⋙rev⋙rev) bitDiff {bitC}"
  IO.println s!"        law2     chain ≡ oracle∘(indep fold) {e2chain} · single ≡ oracle {e2single} · pre-E {preE} · E[early] {eEarly} E[late] {eLate}"
  -- within ~0.09 s the reverb tail is still building (attack/bloom); assert only
  -- causal + nonzero + not-blowing-up (decay is the oracle's job to encode).
  let audible := preE < 1e-18 && eEarly > 1e-9 && eLate < eEarly * 4.0
  -- the CHAIN's error model is looser than the single crossing (2e-6): the
  -- COLLECTED folded room's forced/ringing modes partially cancel (the 1/Δ
  -- conditioning the reassociation inherits), so the net signal sits nearer the
  -- datapath floor. An honest tolerance, not a hidden defect — EC's fold and the
  -- independent `foldRoomsFloat` are the same formula (verified), and the bit-
  -- identity route + single-crossing oracle pin the rest.
  let pass := bitS == 0 && bitC == 0 && e2chain < 2e-3 && e2single < 3e-4 && audible
  if pass then
    passGate "gong-reverb" s!"bloomed gong ⋙ reverb (and a reverb CHAIN) from the patch surface — reassociated fold-then-cross (bitC {bitC}), chain ≡ oracle∘(indep fold) {e2chain}, single {e2single}, audible"
  else
    failGate "gong-reverb" s!"bitS={bitS} bitC={bitC} e2chain={e2chain} e2single={e2single} preE={preE} eEarly={eEarly} eLate={eLate}"

/-- THE GONG⋙REVERB GATE (Deliverable A). A bloomed struck resonator reaches a
    reverb — and a reverb chain — from the PATCH SURFACE (`modalSource` carrying
    a pitch bloom → `lowerInput`), realizing through the reassociated lowering:
    the room chain folds (`residueComposeEC` as filter∘filter), the bloom crosses
    ONCE at the tap (`bloomCompose`). Asserts, via `gongVerdict`: (1) the patch
    render is BIT-IDENTICAL to the direct atom `bloomComposedSig(bloomCompose(
    voice, folded-room))` — the surface routes through the crossing and folds
    first; (2) the chain matches the INDEPENDENT double-convolution `oracleChain`
    (raw filter product, not EC's fold) — law 2, the reassociation is correct,
    not just self-consistent; (3) causal + decaying + nonzero — it is audible.
    This fixture uses a baked room; live-rt60 coverage and explicit refusal are
    exercised by `runBloomLivePole` and `runBloomSafety`. -/
def runGongReverb (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let (B, g) := bloomBg
  -- a bloomed gong register (baked partials) and two SEPARATED baked rooms
  -- (close enough that the filter-product coupling is audible, not near-coincident)
  let voice : Array SeamMode := #[mkMode 220 1.0 1.0, mkMode 610 1.3 0.6, mkMode 1050 1.6 0.4]
  let room1 : Array SeamMode := #[mkMode 200 1.2 1.0, mkMode 380 1.6 0.8]
  let room2 : Array SeamMode := #[mkMode 260 1.4 0.9, mkMode 460 1.8 0.7]
  let vM := voice.map (·.toModal)
  let r1M := room1.map (·.toModal)
  let r2M := room2.map (·.toModal)
  let clk : Clock := clockLit
  -- the patch graphs: bloomed source → reverb(s) → out
  let gSingle : PatchGraph :=
    { nodes := #[ { id := "src", node := .modalSource vM anchorSig clk none none (some (B, g)) }
                , { id := "rev", node := .modalReverb "src" r1M none } ]
      output := "rev" }
  let gChain : PatchGraph :=
    { nodes := #[ { id := "src", node := .modalSource vM anchorSig clk none none (some (B, g)) }
                , { id := "rev1", node := .modalReverb "src" r1M none }
                , { id := "rev2", node := .modalReverb "rev1" r2M none } ]
      output := "rev2" }
  -- the direct atoms the lowering should route through (single + folded chain)
  let directSingle := match bloomCompose vM r1M B g with
    | some pairs => bloomComposedSig pairs clk anchorSig | none => litI 0
  let directChain := match bloomCompose vM (residueComposeEC r1M r2M) B g with
    | some pairs => bloomComposedSig pairs clk anchorSig | none => litI 0
  let rS ← renderGraph arena "gr_single" gSingle
  let rC ← renderGraph arena "gr_chain" gChain
  let rdS ← renderConfig arena "gr_dsingle" directSingle
  let rdC ← renderConfig arena "gr_dchain" directChain
  match rS, rC, rdS, rdC with
  | .ok pS, .ok pC, .ok dS, .ok dC => gongVerdict voice room1 room2 pS pC dS dC
  | _, _, _, _ => failGate "gong-reverb" "build/render failed for a gong⋙reverb graph"

-- ── WS-LP Phase 1: the live-pole crossing witness ─────────────────────────────

/-- THE LIVE-POLE CROSSING GATE (WS-LP Phase 1). A bloomed source crosses a
    reverb whose σ is a LIVE rt60 pole — the pole a Sig the folder can NOT
    constant-fold (the surface shape: `6.91/rt60` with rt60 a slot read; here the
    un-foldable stand-in is the same expression over a clamped literal, so the
    carrier path can render it without a param table). Asserts:
    (1) `bloomCompose` LIFTS every supported pair (the checked Patch route would
        explicitly refuse, never substitute a bare bloom, on any exclusion)
        (all voice×room pairs present, and their constants are genuinely live —
        `sigConstF?` fails on them — and genuinely s0 — `sigIsS0` holds, the
        Stage0-hoist precondition);
    (2) the rendered live crossing ≈ the BAKED crossing at the same rt60 value
        (the lift computes the same constants, by kernel arithmetic instead of
        build-time arithmetic — identical up to last-bit float noise);
    (3) the live crossing ≈ `oracleSeam` under the bloom warp (the law, not just
        self-consistency), causal and finite;
    (4) a pair that is unsupported over the whole rt60 range refuses the typed
        composition; no partial room bank is returned. -/
def runBloomLivePole (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let (B, g) := bloomBg
  let rt60Val : Float := 2.0
  let (rtLo, rtHi) : Float × Float := (0.2, 12.0)     -- the reverb knob's declared span
  let sigLive : Float := 6.91 / rt60Val
  let voice : Array SeamMode := #[mkMode 220 1.0 1.0, mkMode 610 1.3 0.6, mkMode 1050 1.6 0.4]
  let room  : Array SeamMode := #[mkMode 200 sigLive 1.0, mkMode 380 sigLive 0.8]
  let vM := voice.map (·.toModal)
  -- the live room: σ = 6.91/rt60 with rt60 NOT a foldable literal (clamp is
  -- outside `sigConstF?`'s vocabulary, exactly like a `paramRef` slot read),
  -- range declared as the knob span mapped through σ = 6.91/rt60
  let liveRoomAt := fun (rv : Float) (rm : Array SeamMode) => rm.map (fun m =>
    { m.toModal with sigma := div (lit 691 2) (clampE (litF rv) (litF rtLo) (litF rtHi))
                     sigmaRange := some (6.91 / rtHi, 6.91 / rtLo) })
  let liveRoom := liveRoomAt rt60Val
  let rMLive  := liveRoom room
  let rMBaked := room.map (·.toModal)
  -- (1) the lift engages: all 6 pairs, live and s0
  let some pairsLive := bloomCompose vM rMLive B g
    | failGate "bloom-live-pole" "bloomCompose explicitly refused the supported live-rt60 room"
  -- NOT `k1Ser`: the lifted Horner shares subterms by REFERENCE, so its DAG is
  -- small but its TREE is exponential in depth — a structural walk (`sigConstF?`
  -- / `sigIsS0`, unmemoized) would never return. Its leaves are exactly the
  -- constituents checked here (`nuSigma`, `fSer`, `invA`, `dSwitch` — plus
  -- literals), so shallow checks carry the same s0/liveness claim.
  let liveConsts := pairsLive.foldl (fun acc p =>
    acc ++ #[p.fSer.1, p.fSer.2, p.nuSigma, p.dSwitch] ++
    p.invA.foldl (fun a ik => a ++ #[ik.1, ik.2]) #[]) #[]
  let genuinelyLive := pairsLive.all (fun p => (sigConstF? p.nuSigma).isNone)
  let allS0 := liveConsts.all sigIsS0
  -- (4) explicit whole-composition refusal: a room mode 0.05 Hz off the 220 Hz partial is
  -- COINCIDENT-dipping (|Im a| ≈ 0.17 < ½, and the rt60 range walks Re a
  -- through the disc) — the remaining unsupported interval after phase 3a (the
  -- τ·e lanes stay baked-only). The drop set shrank twice: phase 1's
  -- on-partial probe (220.5 Hz) lifted with phase 2's crossing arm, and
  -- phase 2's region-CHANGING probe (230.5 Hz) lifted with phase 3a's
  -- union collapse — it is now witness (6) below.
  let nearRoom := liveRoom #[mkMode 220.05 sigLive 0.5]
  let nearComposition := bloomComposeChecked vM nearRoom B g
  let nearRefused := !nearComposition.isComplete && nearComposition.pairs.isEmpty &&
    nearComposition.exclusions.any (fun x =>
      x.reason == .liveRegionUnsupported || x.reason == .excludedConditioning)
  -- (6) phase 3a — the STRADDLING pair (region-changing over the interval):
  -- room 230.5 Hz vs voice 220 (|a+1| straddles |κ| = 38.4: serOnly at low
  -- rt60, crossing at high). Served by the crossing lanes alone (the union
  -- collapse, probed in demos/wslp_union_probe.py). Witnessed on BOTH sides:
  -- at rt60 = 2 the live dSwitch ≈ 0.026 s (seam in-window, both lanes); at
  -- rt60 = 0.25 the pair sits on its serOnly side (dSwitch < 0, series from
  -- d = 0) and the BAKED pair takes the serOnly ARM — so live ≡ baked there
  -- is the bridge identity `Γ★ − CF(κ)/g = mK/(ν−μ)` IN THE EMITTED KERNEL.
  let renderPairs := fun (name : String) (ps : Array BloomPair) =>
    renderConfig arena name (bloomComposedSig ps clockLit anchorSig)
  let straddleCheck : Float → String → IO (Option (Float × Float)) := fun rv tag => do
    let sv := 6.91 / rv
    let vS : Array SeamMode := #[mkMode 220 1.0 1.0]
    let rS : Array SeamMode := #[mkMode 230.5 sv 0.5]
    let some psL := bloomCompose (vS.map (·.toModal)) (liveRoomAt rv rS) B g | return none
    let some psB := bloomCompose (vS.map (·.toModal)) (rS.map (·.toModal)) B g | return none
    if psL.size != 1 || psL.any (fun p => p.cfN.isEmpty) then return none
    match ← renderPairs s!"wslp_str_{tag}_l" psL, ← renderPairs s!"wslp_str_{tag}_b" psB with
    | .ok l, .ok b =>
      let admS := fun (v r : SeamMode) => bloomAdmitsPair v.pole r.pole B g
      let eB := relL2Win l b (anchorNat + 1) nProbe
      let eO := relL2Win l (oracleSeam vS rS bloomWarp admS 8) (anchorNat + 1) nProbe
      if !(allFinite l) then return none
      return some (eB, eO)
    | _, _ => return none
  let some (eBS2, eOS2) ← straddleCheck 2.0 "r2"
    | failGate "bloom-live-pole" "straddle witness (rt60=2, crossing side) failed to lift/render"
  let some (eBS0, eOS0) ← straddleCheck 0.25 "r025"
    | failGate "bloom-live-pole" "straddle witness (rt60=0.25, serOnly side) failed to lift/render"
  -- the full bank with the straddling room mode now lifts ALL THREE pairs
  let straddleBankFull := match bloomCompose vM (liveRoom #[mkMode 230.5 sigLive 0.5]) B g with
    | some ps => ps.size == 3
    | none => false
  -- (5) WS-LP phase 2 — the CROSSING region, live: a room mode close enough to
  -- the 1023 Hz voice partial that `|a+1| < |κ|` over the WHOLE rt60 range
  -- (crossing-throughout: both lanes emitted, `dSwitch` live). The window is
  -- 2·nProbe so the live `dSwitch` (≈ 0.096 s at rt60 = 2 ⇒ sample ≈ 4443)
  -- falls INSIDE it — the render exercises the CF lane, the seam, AND the
  -- Γ★-bridged series lane.
  let nX := 2 * nProbe
  let voiceX : Array SeamMode := #[mkMode 1023 1.13 1.0]
  let roomX  : Array SeamMode := #[mkMode 1066 sigLive 0.6]
  let vMX := voiceX.map (·.toModal)
  let some pairsX := bloomCompose vMX (liveRoom roomX) B g
    | failGate "bloom-live-pole" "bloomCompose returned none for the live crossing-region room"
  let crossingLifted := pairsX.size == 1 && pairsX.all (fun p => !p.cfN.isEmpty)
  let bakedX := match bloomCompose vMX (roomX.map (·.toModal)) B g with
    | some ps => bloomComposedSig ps clockLit anchorSig
    | none => litI 0
  -- (2)+(3) render live vs baked vs oracle (phase 1 graph route + phase 2 direct)
  let clk : Clock := clockLit
  let mkGraph := fun (rm : Array ModalMode) =>
    ({ nodes := #[ { id := "src", node := .modalSource vM anchorSig clk none none (some (B, g)) }
                 , { id := "rev", node := .modalReverb "src" rm none } ]
       output := "rev" } : PatchGraph)
  let rLive ← renderGraph arena "wslp_live" (mkGraph rMLive)
  let rBaked ← renderGraph arena "wslp_baked" (mkGraph rMBaked)
  let rLiveX ← renderConfig arena "wslp_xlive" (bloomComposedSig pairsX clockLit anchorSig) nX
  let rBakedX ← renderConfig arena "wslp_xbaked" bakedX nX
  match rLive, rBaked, rLiveX, rBakedX with
  | .ok pL, .ok pB, .ok pLX, .ok pBX =>
    let lo := anchorNat + 1
    let hi := nProbe
    let adm := fun (v r : SeamMode) => bloomAdmitsPair v.pole r.pole B g
    let eBaked := relL2Win pL pB lo hi
    let eOracle := relL2Win pL (oracleSeam voice room bloomWarp adm 8) lo hi
    let preE := energyWin pL 0 lo
    let e := energyWin pL lo hi
    let eBakedX := relL2Win pLX pBX lo nX
    let eOracleX := relL2Win pLX (oracleSeam voiceX roomX bloomWarp adm 8 nX) lo nX
    IO.println s!"bloom live-pole crossing (WS-LP: serOnly + crossing + straddling, live rt60):"
    IO.println s!"        lift     pairs {pairsLive.size}/6 · live consts (unfoldable) {genuinelyLive} · s0 {allS0} · coincident-dip composition explicitly refused {nearRefused}"
    IO.println s!"        serOnly  live ≡ baked crossing {eBaked * 1e9}e-9 · live ≡ oracle {eOracle} · pre-E {preE} · E {e}"
    IO.println s!"        crossing lifted (1 pair, CF lane) {crossingLifted} · live ≡ baked {eBakedX * 1e9}e-9 · live ≡ oracle {eOracleX} (window {nX}, seam inside)"
    IO.println s!"        straddle rt60=2 (seam in-window) ≡ baked {eBS2 * 1e9}e-9, ≡ oracle {eOS2} · rt60=0.25 (serOnly side) ≡ baked-SER-ARM {eBS0 * 1e9}e-9 (the bridge identity in-kernel), ≡ oracle {eOS0} · full bank 3/3 {straddleBankFull}"
    -- live ≡ baked is a CONSISTENCY check, not the law (the oracle is): the
    -- baked constants come from build-time forward summation (`bloomM1`) and
    -- `lgammaB`, the lifted ones from the emitted Horner/fraction/`lgammaE` —
    -- real reassociations, so the honest bar is ~1e-6, not bit-identity.
    if pairsLive.size == 6 && genuinelyLive && allS0 && nearRefused
        && eBaked < 1e-6 && eOracle < 3e-4 && preE < 1e-18 && e > 1e-9 && allFinite pL
        && crossingLifted && eBakedX < 1e-6 && eOracleX < 3e-4 && allFinite pLX
        && eBS2 < 1e-6 && eOS2 < 3e-4 && eBS0 < 1e-6 && eOS0 < 3e-4 && straddleBankFull then
      passGate "bloom-live-pole" s!"a live-rt60 reverb CROSSES the bloom in every supported non-coincident region: serOnly 6/6 (≡ oracle {eOracle}); crossing incl. live dSwitch seam (≡ oracle {eOracleX}); STRADDLING pair served by the union collapse on both sides (crossing side ≡ oracle {eOS2}; serOnly side ≡ baked ser arm {eBS0 * 1e9}e-9); an unsupported coincident dip explicitly refuses the whole typed composition"
    else
      failGate "bloom-live-pole" s!"pairs={pairsLive.size} live={genuinelyLive} s0={allS0} nearRefused={nearRefused} eBaked={eBaked * 1e9}e-9 eOracle={eOracle} preE={preE} E={e} finite={allFinite pL} xLift={crossingLifted} eBakedX={eBakedX * 1e9}e-9 eOracleX={eOracleX} finiteX={allFinite pLX} eBS2={eBS2 * 1e9}e-9 eOS2={eOS2} eBS0={eBS0 * 1e9}e-9 eOS0={eOS0} bank3={straddleBankFull}"
  | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ | _, _, _, .error e =>
    failGate "bloom-live-pole" s!"build/render: {e}"

-- ── The Γ-bridge coefficient comparator (per-identity, series-shaped) ──────────
-- Truncated power-series (Taylor-in-d) arithmetic over `CplxB`, depth N. The
-- witness compares the ALGEBRA (every coefficient) rather than points of it: the
-- E1 Kummer-series closed form of the bloom composition equals the defining
-- integral term-by-term. In-repo float64 holds only in the representable regime
-- (small |κ|); the full-range comparator is the mpmath cockpit
-- (`demos/modal_bloom_gamma.py`), kept runnable per the role-split rule.

/-- A truncated power series `Σ cₖ dᵏ`, `k = 0..N` (length N+1), over `CplxB`. -/
private abbrev PS := Array CplxB

private def psZero (n : Nat) : PS := Array.replicate (n + 1) (⟨0, 0⟩ : CplxB)
private def psConst (n : Nat) (c : CplxB) : PS := (psZero n).set! 0 c
/-- The monomial `d` as a series. -/
private def psD (n : Nat) : PS := (psZero n).set! 1 ⟨1, 0⟩
private def psAdd (n : Nat) (a b : PS) : PS := (Array.range (n + 1)).map (fun i => a[i]!.add b[i]!)
private def psSub (n : Nat) (a b : PS) : PS := (Array.range (n + 1)).map (fun i => a[i]!.sub b[i]!)
private def psScaleC (s : CplxB) (a : PS) : PS := a.map (fun x => s.mul x)
private def psMul (n : Nat) (a b : PS) : PS := Id.run do
  let mut out := psZero n
  for i in [0:n + 1] do
    for j in [0:n + 1 - i] do
      out := out.set! (i + j) (out[i + j]!.add (a[i]!.mul b[j]!))
  return out
/-- `exp` of a series (`E' = P'·E`): `E₀ = e^{p₀}`, `E_m = (1/m)Σₖ k·pₖ·E_{m−k}`. -/
private def psExp (n : Nat) (p : PS) : PS := Id.run do
  let mut e := psZero n
  e := e.set! 0 (CplxB.exp p[0]!)
  for m in [1:n + 1] do
    let mut s : CplxB := ⟨0, 0⟩
    for k in [1:m + 1] do
      s := s.add ((p[k]!.scale k.toFloat).mul e[m - k]!)
    e := e.set! m (s.scale (1.0 / m.toFloat))
  return e

/-- THE Γ-BRIDGE COEFFICIENT COMPARATOR: in the representable regime (small |κ|),
    the E1 Kummer-series closed form of the bloom composition,
    `F(d) = [M(κ)·e^{νd} − M(1,a+1,z(d))·e^{μφ(d)}]/(ν−μ)`, `z=κe^{−gd}`,
    `φ=d+B(1−e^{−gd})`, equals the defining integral
    `I(d) = ∫₀^d e^{ν(d−s)}e^{μφ(s)}ds` — computed INDEPENDENTLY from the ODE
    `I' = νI + e^{μφ}` — coefficient-for-coefficient in `d` to depth N. This
    witnesses the resummation ALGEBRA (that `M(1,a+1,·)` is the right series),
    not sample points of it. The Γ★ bridge and the large-|κ| range stay in the
    mpmath cockpit (float64 loses ~0.434|κ| digits — the D_bg3 wall). -/
def runGammaCoeff (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let n : Nat := 12
  let g := bloomBg.2
  let B := bloomBg.1
  -- representable regime: low-frequency poles ⇒ |κ| = |μ|·B ≈ 0.5 (< 1 digit
  -- lost), and modest ω keeps the e^{νd} Taylor coefficients well-conditioned.
  let mu : CplxB := ⟨-0.2, tp * 3.0⟩
  let nu : CplxB := ⟨-0.5, tp * 6.0⟩
  let aC : CplxB := (nu.sub mu).scale (1.0 / g)
  let kappa := mu.scale B
  -- series building blocks
  let eNegGd := psExp n ((psZero n).set! 1 ⟨-g, 0⟩)          -- e^{−gd}
  let z := psScaleC kappa eNegGd                             -- z(d) = κe^{−gd}
  let phi := psAdd n (psD n) (psScaleC ⟨B, 0⟩ (psSub n (psConst n ⟨1, 0⟩) eNegGd))
  let eMuPhi := psExp n (psScaleC mu phi)                    -- e^{μφ(d)}
  let eNu := psExp n ((psZero n).set! 1 nu)                  -- e^{νd}
  -- M(1,a+1,z(d)) by Horner over the series z: 1 + z/(a+1)(1 + z/(a+2)(…))
  let mut Mz := psConst n ⟨1, 0⟩
  for jr in [0:n] do
    let j := n - 1 - jr
    let coef := CplxB.div ⟨1, 0⟩ (aC.add ⟨(j + 1).toFloat, 0⟩)
    Mz := psAdd n (psConst n ⟨1, 0⟩) (psMul n (psScaleC coef z) Mz)
  let Mkappa := (bloomM1 aC kappa).1                          -- M(κ), scalar
  -- E1 closed form F(d)
  let num := psSub n (psScaleC Mkappa eNu) (psMul n Mz eMuPhi)
  let fSeries := psScaleC (CplxB.div ⟨1, 0⟩ (nu.sub mu)) num
  -- I(d) from the ODE I' = νI + e^{μφ}, I₀ = 0 (independent route)
  let mut iSeries := psZero n
  for m in [0:n] do
    let val := (nu.mul iSeries[m]!).add eMuPhi[m]!
    iSeries := iSeries.set! (m + 1) (val.scale (1.0 / (m + 1).toFloat))
  -- coefficient-by-coefficient comparison, relative to the PEAK coefficient
  -- magnitude (the coefficients span many orders as ν^k/k! grows; per-coefficient
  -- relative would spuriously spike where a coefficient is ~0, e.g. d⁰).
  let mut peak := 1e-300
  for k in [0:n + 1] do peak := max peak iSeries[k]!.abs
  let mut worst := 0.0
  for k in [0:n + 1] do
    worst := max worst ((fSeries[k]!.sub iSeries[k]!).abs / peak)
  IO.println "Γ-bridge coefficient comparator (representable regime, series-shaped — not points):"
  IO.println s!"        regime   |κ|={kappa.abs} · |a|={aC.abs} · depth N={n}"
  IO.println s!"        witness  E1 Kummer closed form ≡ ∫ composition (ODE route), max coeff rel|Δ| = {worst}"
  if worst < 1e-11 then
    passGate "gamma-coeff" s!"E1 series ≡ the composition integral coefficient-for-coefficient to d^{n} (max |Δ| {worst}) — the resummation algebra, not points"
  else
    failGate "gamma-coeff" s!"E1/integral coefficient mismatch: max |Δ| {worst} at |κ|={kappa.abs}"

-- ── WS-CL: the region-coverage gate (totality as a number) ────────────────────
-- The types prove COVERAGE (the `SeamRegion` match is exhaustive — a config with
-- no handler won't compile); this gate turns "is the partition total over the
-- shipped register" into a reported number, and the depth cap into MEASURED
-- headroom (a sampled max over the Halton batch + the max-|κ| corner scan —
-- empirical, never a typed bound; only the exhaustive match is type-proven).
-- Per-region ACCURACY stays the seam-sweep's job (the contract's division of
-- labor: types for coverage, the sweep + SNR for the law). Depth-cap exclusions
-- are counted and printed — never a silent cap.

/-- Per-region tally + worst envelope depth over a Halton batch. -/
private structure RegionTally where
  serOnly  : Nat := 0
  crossing : Nat := 0
  coincX   : Nat := 0        -- coincidentCrossing
  coincS   : Nat := 0        -- coincidentSubtle
  cond     : Nat := 0        -- excludedConditioning
  excl     : Nat := 0        -- excludedDepth
  maxN     : Nat := 0        -- worst series depth (admitted samples; excluded carry 0)
  maxK     : Nat := 0        -- worst CF depth
  total    : Nat := 0

/-- Classify a Halton batch of legal (voice pole, room pole) pairs and tally the
    region histogram. Voice poles over the shipped full-register box (σ ∈ [0.2,
    3.3], f ∈ [110, 1023] Hz — the register `bloomgong` DEFAULTS to; `freq` is an
    unclamped knob, so higher user registers push |κ| up and legitimately reach
    `excludedDepth` — the coverage below is scoped to the shipped register); room
    poles either uniform over the reverb box (σ ∈ [0.58, 34.5], f ∈ [60, 6000] Hz)
    or, when `near`, placed within `|ν−μ| < 0.9` of the voice pole (a shrinking
    Halton offset) so `|a| < ½` and the coincident regions are actually exercised —
    uniform sampling almost never lands on coincidence. -/
private def tallyBatch (count : Nat) (B g : Float) (near : Bool) : RegionTally := Id.run do
  let mut t : RegionTally := {}
  for i in [1:count + 1] do
    let sV := 0.2 + 3.1 * haltonF 3 i
    let fV := 110.0 + 913.0 * haltonF 2 i
    let mu : CplxB := ⟨-sV, tp * fV⟩
    let nu : CplxB :=
      if near then
        ⟨-(sV + (haltonF 5 i - 0.5) * 0.8), tp * fV + (haltonF 7 i - 0.5) * 1.6⟩
      else
        ⟨-(0.58 + 33.97 * haltonF 5 i), tp * (60.0 + 5940.0 * haltonF 7 i)⟩
    let plan := classifyBloomPair mu nu B g
    t := { t with total := t.total + 1, maxN := max t.maxN plan.nDepth, maxK := max t.maxK plan.kDepth }
    match plan.region with
    | .serOnly            => t := { t with serOnly  := t.serOnly  + 1 }
    | .crossing           => t := { t with crossing := t.crossing + 1 }
    | .coincidentCrossing => t := { t with coincX   := t.coincX   + 1 }
    | .coincidentSubtle   => t := { t with coincS   := t.coincS   + 1 }
    | .excludedConditioning => t := { t with cond   := t.cond     + 1 }
    | .excludedDepth      => t := { t with excl     := t.excl     + 1 }
  return t

/-- The worst envelope depth over the max-|κ| corner (the top of the register,
    where `|zBnd| = |κ|` is largest and `bloomM1`/`bloomCF` converge slowest): voice
    pinned at f = 1023 Hz (σ swept), room swept over the reverb box. The Halton
    interior only approximates this corner, so a random sample UNDERSHOOTS the true
    worst depth (measured ≈204 vs a boundary config's ≈226); this deterministic
    corner scan is folded into the reported max so the headroom is a trustworthy
    sup, not an undershoot. Still empirical (depth is a convergence count, not
    analytic) — the reason the gate says MEASURED, never "provable". -/
private def cornerMaxDepth (B g : Float) : Nat := Id.run do
  let mut d : Nat := 0
  for i in [1:800] do
    let sV := 0.2 + 3.1 * haltonF 3 i
    let mu : CplxB := ⟨-sV, tp * 1023.0⟩                                    -- the top partial ⇒ max |κ|
    let nu : CplxB := ⟨-(0.58 + 33.97 * haltonF 5 i), tp * (60.0 + 5940.0 * haltonF 7 i)⟩
    let plan := classifyBloomPair mu nu B g
    d := max d (max plan.nDepth plan.kDepth)
  return d

/-- THE REGION-COVERAGE GATE (WS-CL). Halton-classify the shipped bloom register
    and report totality as a number: the region histogram, the admitted fraction,
    and the worst envelope depth vs the 300 cap. Asserts (1) the SHIPPED surface
    (β = 0.05, the default register) has zero exclusions, worst SAMPLED
    depth (Halton interior + the max-|κ| corner scan) < 300, so the cap is MEASURED
    headroom (empirical, not proven — a config in a sampling gap could differ, but
    would still become a counted `excludedDepth`, never a silent error); (2) all
    FOUR served regions are reachable — serOnly, crossing, coincidentCrossing,
    coincidentSubtle (the partition claims no region nothing lands in); (3)
    `excludedDepth` IS reachable off-surface (β = 0.5, knob max), is COUNTED, and
    makes the checked composer refuse the whole requested crossing. Accuracy per region is the seam-sweep's job (types for
    coverage, the sweep for the law). -/
def runSeamCoverage (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let g := bloomBg.2
  let Bship := bloomBg.1                         -- shipped bloom advance (β = 0.05)
  let Bover := 0.5 / 1.8                         -- knob-max β = 0.5, scale 1
  let bSmall : Float := 1.67e-6                  -- a subtle bloom (matches the small-κ witness)
  let uni    := tallyBatch 1000 Bship g false    -- shipped, uniform room → serOnly + crossing
  let near   := tallyBatch 1000 Bship g true     -- shipped, near-coincident → coincidentCrossing
  let subtle := tallyBatch 1000 bSmall g true    -- subtle bloom, near-coincident → coincidentSubtle
  let over   := tallyBatch 1000 Bover g false    -- knob-max β → the depth cap bites
  let corner    := cornerMaxDepth Bship g        -- the max-|κ| worst-corner scan
  let shipTotal := uni.total + near.total
  let shipExcl  := uni.excl + near.excl + uni.cond + near.cond
  let admitFrac := (shipTotal - shipExcl).toFloat / shipTotal.toFloat
  let maxDepth  := max (max (max uni.maxN uni.maxK) (max near.maxN near.maxK)) corner
  IO.println "seam region coverage (classify the shipped register — totality as a number):"
  IO.println s!"        shipped register (β=0.05): serOnly {uni.serOnly + near.serOnly}, crossing {uni.crossing + near.crossing}, coincidentCrossing {near.coincX}, excludedConditioning {uni.cond + near.cond}, excludedDepth {uni.excl + near.excl} / {shipTotal}"
  IO.println s!"        admitted fraction {admitFrac} · worst SAMPLED depth {maxDepth} of cap 300 (measured headroom {300 - maxDepth}; Halton interior + max-|κ| corner scan {corner})"
  IO.println s!"        subtle-bloom box (B={bSmall}): coincidentSubtle {subtle.coincS} / {subtle.total} reachable"
  IO.println s!"        knob-max box (β=0.5): excludedDepth {over.excl} / {over.total} — counted explicit refusal, not a silent cap"
  let mut ok := true
  -- (1) the shipped register is depth-total: the ≤300 cap is measured headroom
  --     (sampled max incl. the worst corner; empirical, never a typed bound)
  if shipExcl != 0 || maxDepth ≥ 300 then
    IO.println s!"        COVERAGE VIOLATION: the shipped register hit the depth cap (excl {shipExcl}, maxDepth {maxDepth}) — the ≤300 bound is not headroom for the shipped surface"
    ok := false
  -- (2) all FOUR served regions the box reaches are actually reached (crossing too —
  --     it has its own bloomGammaStar emit lane, so a dead crossing must fail here)
  if uni.serOnly == 0 || uni.crossing + near.crossing == 0 || near.coincX == 0 || subtle.coincS == 0 then
    IO.println s!"        COVERAGE VIOLATION: a served region is unreachable (serOnly {uni.serOnly}, crossing {uni.crossing + near.crossing}, coincidentCrossing {near.coincX}, coincidentSubtle {subtle.coincS}) — the partition claims a region nothing lands in"
    ok := false
  -- (3) excludedDepth is reachable off-surface AND counted. `runBloomSafety`
  --     separately proves the typed composer and Patch refuse it all-or-nothing.
  if over.excl == 0 then
    IO.println s!"        COVERAGE VIOLATION: knob-max β did not exceed the depth cap — excludedDepth is never exercised, so 'counted not silent' is untested"
    ok := false
  if ok then
    passGate "seam-coverage" s!"the shipped register classifies TOTALLY: admitted fraction {admitFrac} (0 exclusions, worst SAMPLED depth {maxDepth} < 300 — measured headroom, Halton + corner scan), all four served regions reachable, excludedDepth counted off-surface ({over.excl}/{over.total})"
  else
    failGate "seam-coverage" "the region partition is not total over the shipped register — see the lines above"

/-- M3-A stop-line gate: the measured negative-integer witness and its exact
    boundary, the conservative live-interval conjunction edge, the real default
    21×32 register/room endpoint census, κ=0 identity classification, and explicit
    Patch refusal with no partial/bare-room fallback. -/
def runBloomSafety (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let g : Float := 1.8
  let B : Float := 0.05 / g
  let mu : CplxB := ⟨-0.5, tp * 413.867122763⟩
  let planAt := fun (reA : Float) (b : Float) =>
    let nu := mu.add ⟨g * reA, 0⟩
    classifyBloomPair mu nu b g
  let hazard := planAt (-0.98) B
  let metric := bloomConditioningMetric hazard.aC
  let boundary := planAt (-1.0 + bloomConditioningRadius) B
  let outside := planAt (-1.0 + bloomConditioningRadius + 0.0009765625) B
  let latticeEnd := planAt (-300.0 + bloomConditioningRadius) 1.0
  let equalityRefused := bloomExcludedConditioningD
    (⟨-0.984375, 0⟩ : CplxB).toExact (⟨0.015625, 0⟩ : CplxB).toExact
  let k0 := planAt (-0.98) 0.0
  let k0ExactPole := planAt (-1.0) 0.0
  let k0Live := !bloomExcludedConditioningLive (-1.0) (-1.0) 0.0 ⟨0, 0⟩
  let tunedOk := hazard.region == .excludedConditioning && metric < bloomConditioningRadius
    && boundary.region == .excludedConditioning
    && outside.region != .excludedConditioning
    && latticeEnd.region == .excludedConditioning
    && equalityRefused
    && k0.region != .excludedConditioning
    && k0ExactPole.region != .excludedConditioning && k0Live
  -- The disc is witnessed at the left/centre end while the crossing is reached
  -- only at the right edge. This is the conjunction the former coupled-point
  -- check missed.
  let liveEdge := bloomExcludedConditioningLive (-2.0) (-1.96875) 0.0 ⟨0.98, 0⟩
  let liveNoCross := !bloomExcludedConditioningLive (-2.0) (-1.96875) 0.0 ⟨0.96, 0⟩
  let liveReversed := bloomExcludedConditioningLive (-1.96875) (-2.0) 0.0 ⟨0.98, 0⟩
  let liveIm := bloomExcludedConditioningLive (-1.0) (-1.0) 0.015625 ⟨0.1, 0⟩
  let liveImOutside := !bloomExcludedConditioningLive (-1.0) (-1.0) 0.04 ⟨0.1, 0⟩

  -- Actual default structures: 13 full + 8 half gong modes against the emitted
  -- 32-mode ordinary room, classified at both rt60 damping endpoints.
  let (full, half) := defaultGongModes 110.0
  let room := Tropical.Playground.bakedReverbProbe 32
  let sLo : Float := 6.91 / 12.0
  let sHi : Float := 6.91 / 0.2
  let mut endpointTotal := 0
  let mut endpointExcluded := 0
  let mut liveExcluded := 0
  for vb in #[(full, B), (half, B * 0.5)] do
    for v in vb.1 do
      let some vs := sigConstF? v.sigma | liveExcluded := liveExcluded + room.size; continue
      let some vo := sigConstF? v.omega | liveExcluded := liveExcluded + room.size; continue
      let vmu : CplxB := ⟨-vs, vo⟩
      for r in room do
        let some ro := sigConstF? r.omega | liveExcluded := liveExcluded + 1; continue
        match classifyBloomPairLiveChecked vmu ro sLo sHi vb.2 g with
        | .ok _ => pure ()
        | .error _ => liveExcluded := liveExcluded + 1
        for rs in #[sLo, sHi] do
          endpointTotal := endpointTotal + 1
          match (classifyBloomPair vmu ⟨-rs, ro⟩ vb.2 g).region with
          | .excludedConditioning | .excludedDepth =>
            endpointExcluded := endpointExcluded + 1
          | _ => pure ()
  let censusOk := full.size == 13 && half.size == 8 && room.size == 32
    && endpointTotal == 1344 && endpointExcluded == 0 && liveExcluded == 0

  -- One safe and one unsafe pair: the typed result must contain no partial bank,
  -- and Patch lowering must name the conditioning refusal. Bare bloom is intact.
  let unsafeNu := mu.add ⟨g * (-0.98), 0⟩
  let mkModePole := fun (p : CplxB) =>
    ({ sigma := litF (-p.re), omega := litF p.im, cre := lit 1 } : ModalMode)
  let voices := #[mkModePole mu, mkMode 610.0 1.0 0.5 |>.toModal]
  let rooms := #[mkModePole unsafeNu]
  let comp := bloomComposeChecked voices rooms B g
  let safeMu : CplxB := ⟨-1.0, tp * 610.0⟩
  let safePlan := classifyBloomPair safeMu unsafeNu B g
  let mixedRefusal := !comp.isComplete && comp.pairs.isEmpty && comp.expectedPairs == 2
    && safePlan.region != .excludedConditioning && safePlan.region != .excludedDepth
    && comp.exclusions.size == 1 && match comp.exclusions[0]? with
      | some x => x.voiceIndex == some 0 && x.roomIndex == some 0 && x.reason == .excludedConditioning
      | none => false
  let crossed : PatchGraph :=
    { nodes := #[{ id := "src", node := .modalSource voices anchorSig clockLit none none (some (B, g)) },
                 { id := "rev", node := .modalReverb "src" rooms none }], output := "rev" }
  let bare : PatchGraph :=
    { nodes := #[{ id := "src", node := .modalSource voices anchorSig clockLit none none (some (B, g)) }],
      output := "src" }
  let patchRefused := match lowerGraph crossed with
    | .error e => (e.splitOn "excludedConditioning").length > 1
    | .ok _ => false
  let bareOk := match lowerGraph bare with | .ok _ => true | .error _ => false

  -- Exact κ=0, a=-1: empty Horner identity, complete composition, finite render;
  -- the live point classifier takes the same zero-depth route.
  let k0Nu := mu.add ⟨-g, 0⟩
  let k0Comp := bloomComposeChecked #[mkModePole mu] #[mkModePole k0Nu] 0.0 g
  let k0LivePlan := classifyBloomPairLiveChecked mu mu.im (-k0Nu.re) (-k0Nu.re) 0.0 g
  let k0Typed := k0Comp.isComplete && k0Comp.pairs.size == 1 && match k0LivePlan with
    | .ok { region := .serOnly, nDepth, kDepth } => nDepth == 0 && kDepth == 0
    | .ok { region := .crossing, .. } => false
    | .error _ => false
  let k0Render ← match k0Comp.toOption with
    | some ps => renderConfig arena "bloom_kappa_zero" (bloomComposedSig ps clockLit anchorSig)
    | none => pure (.error "κ=0 composition refused")
  let k0Ref ← renderConfig arena "bloom_kappa_zero_ref"
    (modalBankSig (residueComposeEC #[mkModePole mu] #[mkModePole k0Nu]) clockLit anchorSig)
  let k0Error := match k0Render, k0Ref with
    | .ok xs, .ok ys => relL2Win xs ys (anchorNat + 1) nProbe
    | _, _ => 1.0
  let k0RenderOk := match k0Render, k0Ref with
    | .ok xs, .ok ys => allFinite xs && allFinite ys && k0Error < 3e-4
    | _, _ => false

  -- Depth exclusion is the same all-or-nothing contract as conditioning.
  let depthB : Float := 0.5 / g
  let depthMu : CplxB := ⟨-0.2, tp * 1023.0⟩
  let depthNu : CplxB := ⟨-0.58, tp * 500.0⟩
  let depthComp := bloomComposeChecked #[mkModePole depthMu] #[mkModePole depthNu] depthB g
  let depthGraph : PatchGraph :=
    { nodes := #[{ id := "src", node := .modalSource #[mkModePole depthMu] anchorSig clockLit none none (some (depthB, g)) },
                 { id := "rev", node := .modalReverb "src" #[mkModePole depthNu] none }], output := "rev" }
  let depthTyped := !depthComp.isComplete && depthComp.pairs.isEmpty
    && depthComp.exclusions.size == 1 && match depthComp.exclusions[0]? with
      | some x => x.voiceIndex == some 0 && x.roomIndex == some 0 && x.reason == .excludedDepth
      | none => false
  let depthPatch := match lowerGraph depthGraph with
    | .error e => (e.splitOn "excludedDepth").length > 1
    | .ok _ => false

  IO.println s!"bloom M3-A safety: metric {metric} (radius {bloomConditioningRadius}) · live conjunction {liveEdge}/{liveNoCross}, reverse {liveReversed}, im {liveIm}/{liveImOutside} · default endpoints {endpointTotal} excluded {endpointExcluded}, live excluded {liveExcluded} · conditioning refusal {mixedRefusal}/{patchRefused}, depth {depthTyped}/{depthPatch}, κ0 {k0Typed}/{k0RenderOk} rel {k0Error}, bare {bareOk}"
  if tunedOk && liveEdge && liveNoCross && liveReversed && liveIm && liveImOutside
      && censusOk && mixedRefusal && patchRefused && depthTyped && depthPatch
      && k0Typed && k0RenderOk && bareOk then
    passGate "bloom-safety" s!"conditioning is named/refused at a≈-0.98, -300, exact disc/equality edges, reversed intervals, and nonzero-im live points; exact κ=0,a=-1 composes with an empty Horner and matches the unwarped residue atom (rel {k0Error}); the actual 21×32 default endpoint box is total; conditioning and depth refuse all-or-nothing with exact pair indices; bare bloom remains intact"
  else
    failGate "bloom-safety" s!"tuned={tunedOk} live={liveEdge}/{liveNoCross}/{liveReversed}/{liveIm}/{liveImOutside} census={censusOk} mixed={mixedRefusal} patch={patchRefused} depth={depthTyped}/{depthPatch} k0={k0Typed}/{k0RenderOk} bare={bareOk}"

/-- Terminal-spike numerical-overlap gate for the four cap-384 capacity
    outliers.  Each moved-seam pair is materialized at beta max and checked
    through the nested timed terminal against the independent quadrature law in
    a window long enough to cross `dSwitch`.  The worst full-register case also
    probes beta=.05 and betaSwitch±one Q16 step, including the coefficient-side
    series/CF transition at onset.  This is representative measured evidence,
    not continuous-box qualification and not a production admission change. -/
def runTimedBloomMovedSeamOracle (arena : Arena) : IO Bool := do
  let g : Float := 1.8
  let betaMax : Float := 0.5
  let sigLo : Float := 6.91 / 12.0
  let sigHi : Float := 6.91 / 0.2
  let sigMid := (sigLo + sigHi) * 0.5
  let (full, half) := defaultGongModes 110.0
  let room := Tropical.Playground.bakedReverbProbe 32
  let specs : Array (String × Bool × Nat × Nat × Float × Float) := #[
    ("f10r20", false, 10, 20, 1.0, sigLo),
    ("f11r21", false, 11, 21, 1.0, sigHi),
    ("h6r25", true, 6, 25, 0.5, sigHi),
    ("h7r25", true, 7, 25, 0.5, sigHi)]
  let nLen : Nat := 8192
  let lo := anchorNat + 1
  let mut ok := true
  let mut probes := 0
  let mut worst := 0.0
  let mut worstTag := ""
  for (tag, isHalf, vi, ri, scale, worstSigma) in specs do
    let bank := if isHalf then half else full
    let some v := bank[vi]? | ok := false; continue
    let some r0 := room[ri]? | ok := false; continue
    let some vs := sigConstF? v.sigma | ok := false; continue
    let some vo := sigConstF? v.omega | ok := false; continue
    let some vaRe := sigConstF? v.cre | ok := false; continue
    let some vaIm := sigConstF? v.cim | ok := false; continue
    let some ro := sigConstF? r0.omega | ok := false; continue
    let some raRe := sigConstF? r0.cre | ok := false; continue
    let some raIm := sigConstF? r0.cim | ok := false; continue
    let mu : CplxB := ⟨-vs, vo⟩
    let planE := planTimedBloomBetaPair mu ro sigLo sigHi betaMax scale g
    let some plan := planE.toOption | ok := false; continue
    if plan.incumbent then ok := false; continue
    let betaSwitch := plan.radialFraction * betaMax
    let betaStep := betaMax / 65536.0
    let localProbes := if tag == "f10r20" then #[
        (worstSigma, betaMax), (sigMid, 0.05),
        (sigMid, max 0.0 (betaSwitch - betaStep)),
        (sigMid, betaSwitch),
        (sigMid, min betaMax (betaSwitch + betaStep))]
      else #[(worstSigma, betaMax)]
    for (sigma, beta) in localProbes do
      let nu : CplxB := ⟨-sigma, ro⟩
      let c := cmulE v.ampE r0.ampE
      let some pair := materializeTimedBloomBetaPair? mu nu c beta betaMax scale g plan
        | ok := false; continue
      let banks : Array TimedBloomBank := #[{ pairs := #[pair], anchor := anchorSig }]
      let some sig := (bloomTimedBatchSig banks clockLit).toOption
        | ok := false; continue
      match ← renderConfig arena s!"moved_seam_{tag}_{probes}" sig nLen with
      | .error _ => ok := false
      | .ok dut =>
        let vS : SeamMode := { sigma := vs, omega := vo, are := vaRe, aim := vaIm }
        let rS : SeamMode := { sigma, omega := ro, are := raRe, aim := raIm }
        let B := beta * scale / g
        let phi := fun s => s + B * (1.0 - Float.exp (-g * s))
        let ref := oracleSeam #[vS] #[rS] phi (fun _ _ => true) 8 nLen
        let e := relL2Win dut ref lo nLen
        probes := probes + 1
        if e > worst then worst := e; worstTag := s!"{tag}@sigma={sigma},beta={beta}"
        if !(allFinite dut) || energyWin dut lo nLen <= 1e-12 || e >= 3e-4 then
          ok := false
  IO.println s!"        moved radial-seam quadrature: probes={probes} worst rel-L2={worst} @ {worstTag}"
  if ok && probes == 8 && worst < 3e-4 then
    passGate "timed-bloom-moved-seam" s!"four cap-384 outliers plus betaSwitch±Q16 agree with the independent composition quadrature (worst rel-L2 {worst}); representative overlap only, production remains unchanged"
  else
    failGate "timed-bloom-moved-seam" s!"ok={ok} probes={probes} worst={worst} @ {worstTag}"

-- ── WS-CL: the lane-tidying bit-clean witness ─────────────────────────────────

/-- Re-bake BOTH dropped lanes onto a `coincidentSubtle` `BloomPair` — reconstructs
    the FULL pre-refactor coincident emit (the `cfB`/`cfN`/`k1Cf` CF lane AND the
    nonempty `invA` E1 lane, both baked at the coincident `nDepth`/`kDepth`) so the
    bit-clean witness compares the region-indexed emit against a faithful old pair,
    not one that happens to share the new pair's empty lanes (no circularity: the
    witness independently tests that BOTH dropped lanes are dead). The re-baked CF
    values are the crossing arm's; the pair is subtle (`dSwitch < 0`) so the
    per-sample `selectE` always discards the CF lane, and the coincident branch
    never reads `invA` — the render must be byte-identical regardless. -/
private def withCfLane (p : BloomPair) : BloomPair := Id.run do
  -- the pair is BAKED (built from literal poles), so its `Sig` fields fold back
  -- to Floats via `sigConstF?` (the WS-LP CplxE lift keeps baked pairs literal)
  let cf := fun (s : Tropical.EmitArrow.Sig) => (Tropical.EmitArrow.sigConstF? s).getD 0.0
  let mu : CplxB := ⟨-(cf p.muSigma), cf p.muOmega⟩
  let nu : CplxB := ⟨-(cf p.nuSigma), cf p.nuOmega⟩
  let aC := (nu.sub mu).scale (1.0 / p.gRate)
  let plan := classifyBloomPair mu nu p.bloomB p.gRate
  let kappaB : CplxB := ⟨cf p.kappa.1, cf p.kappa.2⟩
  let (cfK, _) := bloomCF aC kappaB
  let cfB := (Array.range (plan.kDepth + 1)).map (fun j => (⟨(2 * j + 1).toFloat - aC.re, -aC.im⟩ : CplxB))
  let cfN := (Array.range plan.kDepth).map (fun j =>
    let jf := (j + 1).toFloat
    (⟨jf, 0⟩ : CplxB).mul ((⟨jf, 0⟩ : CplxB).sub aC))
  let invA := (Array.range plan.nDepth).map (fun k => CplxB.div ⟨1, 0⟩ (aC.add ⟨(k + 1).toFloat, 0⟩))
  return { p with k1Cf := cplxLitE ((cfK.scale (1.0 / p.gRate)).neg),
                  cfB := cfB.map cplxLitE, cfN := cfN.map cplxLitE,
                  invA := invA.map cplxLitE }

/-- THE LANE-CLEAN GATE (WS-CL). The region-indexed `coincidentSubtle` emit (the
    whole CF lane DROPPED) is BYTE-IDENTICAL to the pre-refactor emit (which always
    baked the CF lane and let the const-true `selectE` discard it). Builds the same
    subtle pair BOTH ways — the new empty-CF pair and the same pair with the CF lane
    re-baked (`withCfLane`) — renders both, and asserts `bitDiffCount = 0`. The
    lane-tidying's proof: the plan shrank, the samples did not move (the dropped
    lane was truly dead, LLVM `select` ignoring the unselected operand). -/
def runSeamLaneClean (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let g := bloomBg.2
  let bSmall : Float := 1.67e-6
  -- a subtle-bloom coincident pair (dSwitch < 0): |κ| ≪ |a+1|, |a| < ½
  let v : Array SeamMode := #[mkMode 55 0.8 1.0]
  let r : Array SeamMode := #[{ sigma := 0.5, omega := tp * 55 + 0.74, are := 0.5 }]
  let vM := v.map (·.toModal)
  let rM := r.map (·.toModal)
  let newSig := match bloomCompose vM rM bSmall g with
    | some pairs => bloomComposedSig pairs clockLit anchorSig
    | none => litI 0
  let oldSig := match bloomCompose vM rM bSmall g with
    | some pairs => bloomComposedSig (pairs.map withCfLane) clockLit anchorSig
    | none => litI 0
  match ← renderConfig arena "seam_laneclean_new" newSig,
        ← renderConfig arena "seam_laneclean_old" oldSig with
  | .ok dNew, .ok dOld =>
    let bd := bitDiffCount dNew dOld
    IO.println s!"seam lane-clean (WS-CL): coincidentSubtle CF-drop bitDiff {bd} (region-indexed emit vs pre-refactor baked-CF emit)"
    if bd == 0 && allFinite dNew then
      passGate "seam-laneclean" "the region-indexed coincidentSubtle emit is BYTE-IDENTICAL to the pre-refactor always-bake-CF emit — the dropped CF lane was truly dead (const-true selectE)"
    else
      failGate "seam-laneclean" s!"the CF-drop changed {bd} samples — the lane was NOT dead (or the render is non-finite)"
  | _, _ => failGate "seam-laneclean" "build/render failed for the lane-clean witness"

-- ── WS-DDF: the near-coincident room-chain fold ───────────────────────────────

/-- The STABLE oracle for the bloomed room-CHAIN: `c·∫₀^d [(e^{ν1(d−s)}−e^{ν2(d−s)})/Δ]
    ·e^{μφ(s)} ds` per voice mode, summed, Re, sink-gained. The deg-1 folded kernel
    (`room1 ⋙ room2` impulse response) quadratured directly — the `(·−·)/Δ` is
    float64-accurate for the witness's MODERATE Δ (no huge residue). Composite
    Simpson, ω on the rotator grid, matching the DUT carrier. -/
private def oracleFoldChain (voice : Array SeamMode) (nu1p nu2p r1r2 : CplxB)
    (B g : Float) (hdiv : Nat) (nLen : Nat := nProbe) : Array Float := Id.run do
  let mut y : Array Float := Array.replicate nLen 0.0
  -- match the DUT's convention: the ν2 carrier is on the rotator grid (`modePhaseQ`),
  -- but the divided difference uses the RAW Δ (`cexpm1` divisor + slow oscillation),
  -- so the effective ν1 = qOm(ω2) + (ω1−ω2)_raw. Quantizing ω1, ω2 SEPARATELY would
  -- inject a ~res/Δ error into the 1/Δ-sensitive divided difference.
  let om2 := qOm nu2p.im
  let nu2 : CplxB := ⟨nu2p.re, om2⟩
  let nu1 : CplxB := ⟨nu1p.re, om2 + (nu1p.im - nu2p.im)⟩
  let dlt := nu1.sub nu2
  for v in voice do
    let mu : CplxB := ⟨-v.sigma, qOm v.omega⟩
    let a  : CplxB := ⟨v.are, v.aim⟩
    let c := (a.mul r1r2).scale sinkGain
    let h := 1.0 / (srF * hdiv.toFloat)
    let fint := fun (nu : CplxB) (s : Float) =>
      CplxB.exp ((mu.scale (s + B * (1.0 - Float.exp (-g * s)))).sub (nu.scale s))
    let mut j1 : CplxB := ⟨0, 0⟩
    let mut j2 : CplxB := ⟨0, 0⟩
    for i in [anchorNat + 1 : nLen] do
      let dBase := (i - 1 - anchorNat).toFloat / srF
      let mut s1 : CplxB := ⟨0, 0⟩
      let mut s2 : CplxB := ⟨0, 0⟩
      for k in [0:hdiv + 1] do
        let w := if k == 0 || k == hdiv then 1.0 else if k % 2 == 1 then 4.0 else 2.0
        let s := dBase + k.toFloat * h
        s1 := s1.add ((fint nu1 s).scale w)
        s2 := s2.add ((fint nu2 s).scale w)
      j1 := j1.add (s1.scale (h / 3.0))
      j2 := j2.add (s2.scale (h / 3.0))
      let d := (i - anchorNat).toFloat / srF
      let yc := (((CplxB.exp (nu1.scale d)).mul j1).sub ((CplxB.exp (nu2.scale d)).mul j2)).div dlt
      y := y.set! i (y[i]! + (c.mul yc).re)
  return y

/-- Arm B's accuracy floor — MEASURED, not derived (set from the observed value at
    landing, rounded up). Arm B carries the same Q4.28 absolute landing floor as arm
    A against a signal of the same order (`K₁·2/|Δ| ≈ 5e-5` vs arm A's `K₁·d ≈
    7e-5`), so the two floors sit in the same decade — but "same decade" is an
    expectation, and the number below is an observation: arm B renders at **3.22e-3**
    (2026-07-20), and the fail line is set at 5e-3 — ~1.5× margin, tight enough that
    every structural mutation of the Δ-secular moves it by O(1) and trips it. -/
private def ddfoldSeparatedFloor : Float := 5e-3

/-- THE WS-DDF FOLD GATE, in TWO ARMS over one atom.

    **Arm A — near-coincident** (Δ = 0.005 rad/s): the regime the atom is named
    for. The collected fold (`residueComposeEC`) bakes a residue `c/Δ` that is out
    of Q4.28 range as a *value* (`|c/Δ| ≈ 60 > 8`, printed below); the DD fold
    removes it entirely. This arm asserts only that the DD form holds its law and
    never loses to the collected form — it does **not** assert that the collected
    form is broken. See the FINDING in the body, which measures that it is not, at
    reachable Δ.

    **Arm B — separated** (Δ = 30 rad/s, and `σ₁ ≠ σ₂` so `Δ.re ≠ 0`): the arm
    that makes the divided-difference *structure* observable. Arm A alone cannot
    see it: over the render window `|Δ|·d ≤ 4.4e-4`, so `cexpm1(Δd)` departs from
    the constant `1` by only ~2e-4 while arm A's own error floor is ~2.1e-3 of
    Q4.28 landing noise — flipping the Δ sign convention moves arm A by 3.3e-4 and
    deleting the whole `cexpm1` select moves it by 1.7e-4, both invisible. Arm A
    catches term *deletions*; it cannot catch the secular the atom exists to carry.
    Arm B reaches `|Δ|·d ≈ 2.65`, so it selects the `big`/direct lane (dead in arm
    A, which never leaves the series branch) and drives `expSig w.1` off zero for
    the first time in any witness. MUTATION-VERIFIED on arm B: flip the Δ sign
    convention (`Modal.lean` `w`), replace `cxΔ` by `1`, or force the series arm
    (`big := gt wsq (litF 1e9)`) → each turns arm B RED by O(1); the last leaves
    arm A green, which is what proves arm B is the coverage and not incidental.

    Same divided-difference cure as atom four, at the fold site (WS-DDF; cockpit
    `demos/modal_ddfold.py`; audit `design/seam-hardening-remainder-handoff` §0). -/
def runFoldChain (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let lo := anchorNat + 1
  let hi := nProbe
  let (B, g) := bloomBg
  let voice : Array SeamMode := #[mkMode 110 0.2 1.0, mkMode 353 0.56 0.6]
  let r1 : SeamMode := mkMode 300 1.0 0.6
  -- Arm A: the near-coincident pair (Δ = i·0.005 exactly — σ shared, so Δ.re = 0).
  let r2 : SeamMode := { sigma := 1.0, omega := tp * 300 + 0.005, are := 0.5 }
  -- Arm B: separated in BOTH components — Δ = (−0.7) + i·(−30), so |Δ|·d reaches
  -- ≈ 2.65 at the window edge and the direct `cexpm1` lane is genuinely selected.
  let r2s : SeamMode := { sigma := 1.7, omega := tp * 300 + 30.0, are := 0.5 }
  let vM := voice.map (·.toModal)
  let r1M := r1.toModal
  -- One arm: the DD-fold DUT for a room pair against that pair's own stable
  -- oracle. `oracleSelf` is the Simpson self-distance (hdiv 4 vs 8) — a QUADRATURE
  -- convergence check, not a bound on the oracle's `1/Δ` cancellation error.
  let runArm : String → SeamMode → IO (Option (Float × Float × Bool)) := fun tag r2m => do
    let r1r2 := r1.amp.mul r2m.amp
    let dut := match bloomFoldCompose vM r1M r2m.toModal B g with
      | some pairs => bloomFoldComposedSig pairs clockLit anchorSig
      | none => litI 0
    let ref  := oracleFoldChain voice r1.pole r2m.pole r1r2 B g 8
    let ref4 := oracleFoldChain voice r1.pole r2m.pole r1r2 B g 4
    match ← renderConfig arena s!"seam_ddfold_{tag}" dut with
    | .ok d => pure (some (relL2Win d ref lo hi, relL2Win ref4 ref lo hi, allFinite d))
    | .error _ => pure none
  let r1r2 := r1.amp.mul r2.amp
  let coll := match bloomCompose vM (residueComposeEC #[r1M] #[r2.toModal]) B g with
    | some pairs => bloomComposedSig pairs clockLit anchorSig
    | none => litI 0
  let refA := oracleFoldChain voice r1.pole r2.pole r1r2 B g 8
  let resMag := (r1r2.div (r1.pole.sub r2.pole)).abs
  match ← runArm "a" r2, ← runArm "b" r2s, ← renderConfig arena "seam_ddfold_coll" coll with
  | some (eD, selfA, finA), some (eS, selfB, finB), .ok dC =>
    let eC := relL2Win dC refA lo hi
    IO.println s!"WS-DDF room-chain fold, two arms (collected residue |c/Δ|≈{resMag}):"
    IO.println s!"        A near-coincident (Δ=0.005): DD rel {eD} · collected rel {eC} (oracle self {selfA})"
    IO.println s!"        B separated (Δ=30, Δ.re≠0, direct cexpm1 lane): DD rel {eS} (oracle self {selfB})"
    -- The atom's LAW: the DD fold ≡ the stable oracle within the divided-difference
    -- floor (≈2e-3 — the chain signal is ≈ ∂Y0/∂ν, far weaker than the O(1) bloom
    -- cross, so Q4.28's absolute floor costs more relative precision, the chain's
    -- looser model). And it is at least as good as the collected fold. FINDING
    -- (WS-DDF): the collected bloomed chain does NOT catastrophically break at
    -- reachable Δ — the bloom's per-sample K factor (≈ M(κ)/(gα) ≈ 1e-3, rooms far
    -- from the voice) scales the huge residue c/Δ DOWN before landing, so the Q4.28
    -- overflow (modal_divdiff's D_dd2, where the residue lands directly) does not
    -- reach the bloom cross until Δ < ~1e-5 rad/s (~1.6e-6 Hz). The DD fold is the
    -- correct, modestly-tighter alternative; the a-DD machinery TRANSFERS (the datum).
    if !finA || eD ≥ 3e-3 then
      failGate "ddfold" s!"arm A (near-coincident) is off the law (rel {eD}, oracle self {selfA})"
    else if eD > eC * 1.5 + 3e-4 then
      failGate "ddfold" s!"arm A's DD fold ({eD}) is WORSE than the collected ({eC}) — the stable form should never lose"
    else if !finB || eS ≥ ddfoldSeparatedFloor then
      failGate "ddfold" s!"arm B (separated Δ, the direct cexpm1 lane) is off the law (rel {eS} ≥ {ddfoldSeparatedFloor}, oracle self {selfB}) — the Δ-secular's sign/branch is wrong"
    else
      passGate "ddfold" s!"the room-chain fold holds its law in both arms (A near-coincident DD {eD} ≤ collected {eC}; B separated DD {eS} on the direct cexpm1 lane); the a-DD machinery transferred to the fold site (WS-DDF datum). Finding: the bloom's K factor keeps the collected fold from breaking at reachable Δ — the wound is narrower than assumed"
  | _, _, _ => failGate "ddfold" "build/render failed for the WS-DDF fold witness"

-- ── The EC/DD erasure gate (fork 3′ Phase 1) ──────────────────────────────────

/-- Render an arrow TERM through the production emit seam (`normalize` →
    `emitTerm`) — the comparator side of the byte-identity checks and the
    collected counter-renders. -/
private def renderTerm (arena : Arena) (name : String) (t : ArrowTerm)
    (nLen : Nat := nProbe) : IO (Except String (Array Float)) := do
  let (out, _) := emitTerm (normalize t) {}
  renderConfig arena name out nLen

private def renderGraphN (arena : Arena) (name : String) (gr : PatchGraph)
    (nLen : Nat) : IO (Except String (Array Float)) := do
  match lowerGraph gr with
  | .error e => pure (.error e)
  | .ok term => renderTerm arena name term nLen

/-- THE EC/DD ERASURE GATE (fork 3′ Phase 1). The compiler owns the EC-vs-DD
    choice per coupling; this gate pins the three behaviors that make that an
    ERASURE rather than a new mode:
    (1) COLD BYTE-IDENTITY — a separated chain from the patch surface
        (src⋙rev⋙rev, nothing hot) renders BIT-IDENTICAL to the direct
        collected chain `EC(EC(v,r₁),r₂)`: the deferred fold + partition
        emits yesterday's compiler verbatim when θ is false.
    (2) THE MIGRATION WITNESS — a room mode tuned ONTO a voice partial
        (sub-grid detune, the old documented hazard) routes to the paired
        atom: the partitioned render holds the law vs `oracleSeam` while the
        collected counter-render (yesterday's output) is off it by decades —
        served-full where it was served-wrong. Checked at the single compose
        AND through a chain (the coupling forms against the COMPOSED amps at
        the final fold, `sort-hot-last`'s mechanics).
    (3) ROUTING STRUCTURE — the partition routes exactly the expected
        couplings (one paired atom per tuned pair; room-room coincidence in
        a chain routes at the final fold), and a room-room-coincident chain
        renders finite and causal. (Its QUANTITATIVE oracle needs a deg-1
        capable reference — the folded room's τ·e limit — which `oracleSeam`
        can't express; recorded as the v2/deg-1 oracle gap, not silently
        skipped.) -/
def runEcddPartition (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let lo := anchorNat + 1
  let nWin : Nat := 32768        -- ~0.74 s: the τ·e resonance (peak ≈ 1/σ s) is in-window
  let admAll := fun (_ _ : SeamMode) => true
  let voice : Array SeamMode := #[mkMode 220 1.0 1.0, mkMode 610 1.3 0.6]
  let room1 : Array SeamMode := #[mkMode 200 1.2 1.0, mkMode 380 1.6 0.8]
  let vM := voice.map (·.toModal)
  let r1M := room1.map (·.toModal)
  -- (1) cold byte-identity through the production seam
  let roomCold : Array SeamMode := #[mkMode 260 1.4 0.9, mkMode 460 1.8 0.7]
  let rCM := roomCold.map (·.toModal)
  let gCold : PatchGraph :=
    { nodes := #[ { id := "src", node := .modalSource vM anchorSig clockLit none none none }
                , { id := "rev1", node := .modalReverb "src" r1M none }
                , { id := "rev2", node := .modalReverb "rev1" rCM none } ]
      output := "rev2" }
  let directCold := modalBankTerm (residueComposeEC (residueComposeEC vM r1M) rCM) anchorSig clockLit
  -- (2) the tuned-unison mutation: ω on the partial, σ matched ⇒ |Δ| = 1e-6
  -- rad/s (below the 6.5e-5 rotator quantum — the unrepresentable-collected
  -- point), amps const ⇒ `couplingHot` fires on the θ_acc lens.
  let roomHot : Array SeamMode := #[{ sigma := 1.0, omega := tp * 220 + 1e-6, are := 0.7 }]
  let rHM := roomHot.map (·.toModal)
  let gHot1 : PatchGraph :=
    { nodes := #[ { id := "src", node := .modalSource vM anchorSig clockLit none none none }
                , { id := "rev", node := .modalReverb "src" rHM none } ]
      output := "rev" }
  let gHotC : PatchGraph :=
    { nodes := #[ { id := "src", node := .modalSource vM anchorSig clockLit none none none }
                , { id := "rev1", node := .modalReverb "src" r1M none }
                , { id := "rev2", node := .modalReverb "rev1" rHM none } ]
      output := "rev2" }
  -- (3) structural routing: the partition's shape, checked directly
  let (plainS, pairedS) := residueComposePartitioned vM rHM
  let (_plainC, pairedC) := foldRoomsPartitioned vM #[r1M, rHM]
  let r2Coin : Array SeamMode := #[{ sigma := 1.2, omega := tp * 200 + 1e-6, are := 0.7 }]
  let r2CM := r2Coin.map (·.toModal)
  let (_plainR, pairedR) := foldRoomsPartitioned vM #[r1M, r2CM]
  -- (3b) the routing boundary, lens-discriminated. The RAIL lens's service
  -- region (Δ = 0.6 > θ_acc — the accuracy lens is OFF — with |a·r|/Δ ≈ 6.7 >
  -- 4 and σ = 1.5 so the damping arm clears the cap) must route; the SAME Δ at
  -- small amps (|a·r|/Δ ≈ 1.7 < 4) must stay cold — together they prove the
  -- rail lens itself decided, not θ_acc. And the STATED refusal: a lens-fired
  -- coupling at extreme Q (σ = 0.02: |c|/(e·σ_min) ≈ 18 ≥ cap 8) classifies
  -- `refused` and the partition leaves it collected — the admission edge the
  -- merged seam atom now states instead of certifying.
  let railVm := (mkMode 800 1.5 2.0).toModal
  let railRm := ({ sigma := 1.5, omega := tp * 800 + 0.6, are := 2.0 } : SeamMode).toModal
  let railRmCold := ({ sigma := 1.5, omega := tp * 800 + 0.6, are := 0.5 } : SeamMode).toModal
  let refVm := (mkMode 400 0.02 1.0).toModal
  let refRm := ({ sigma := 0.02, omega := tp * 400 + 1e-6, are := 1.0 } : SeamMode).toModal
  let railRouted := (residueComposePartitioned #[railVm] #[railRm]).2.size == 1
  let railDiscr := (residueComposePartitioned #[railVm] #[railRmCold]).2.isEmpty
  let refusedStated := couplingRefused refVm refRm
                    && (residueComposePartitioned #[refVm] #[refRm]).2.isEmpty
  let lensOk := railRouted && railDiscr && refusedStated
  let structOk := pairedS.size == 1 && plainS.size == vM.size + rHM.size
              && pairedC.size == 1 && pairedR.size == 1
  let gCoinC : PatchGraph :=
    { nodes := #[ { id := "src", node := .modalSource vM anchorSig clockLit none none none }
                , { id := "rev1", node := .modalReverb "src" r1M none }
                , { id := "rev2", node := .modalReverb "rev1" r2CM none } ]
      output := "rev2" }
  match ← renderGraph arena "ecdd_cold" gCold,
        ← renderTerm arena "ecdd_cold_direct" directCold,
        ← renderGraphN arena "ecdd_hot1" gHot1 nWin,
        ← renderTerm arena "ecdd_hot1_coll" (modalBankTerm (residueComposeEC vM rHM) anchorSig clockLit) nWin,
        ← renderGraphN arena "ecdd_hotC" gHotC nWin,
        ← renderTerm arena "ecdd_hotC_coll" (modalBankTerm (residueComposeEC (residueComposeEC vM r1M) rHM) anchorSig clockLit) nWin,
        ← renderGraphN arena "ecdd_coinC" gCoinC nWin with
  | .ok cold, .ok coldD, .ok hot1, .ok hot1C, .ok hotC, .ok hotCC, .ok coinC =>
    let bitCold := bitDiffCount cold coldD
    let ref1 := oracleSeam voice roomHot idWarp admAll 8 nWin
    let e1DD := relL2Win hot1 ref1 lo nWin
    let e1Coll := relL2Win hot1C ref1 lo nWin
    let refC := oracleSeam voice (foldRoomsFloat room1 roomHot) idWarp admAll 8 nWin
    let eCDD := relL2Win hotC refC lo nWin
    let eCColl := relL2Win hotCC refC lo nWin
    let coinFinite := allFinite coinC
    let coinPre := energyWin coinC 0 lo
    let coinE := energyWin coinC lo nWin
    IO.println s!"ecdd erasure (per-coupling partition, θ_acc={Tropical.EmitArrow.ecddThetaAcc} rad/s):"
    IO.println s!"        cold chain bitDiff {bitCold} (deferred fold ≡ yesterday's collected chain)"
    IO.println s!"        tuned unison (|Δ|=1e-6, sub-grid): partitioned {e1DD} vs collected {e1Coll} (oracle law)"
    IO.println s!"        tuned unison through a chain:      partitioned {eCDD} vs collected {eCColl}"
    IO.println s!"        routing structure ok={structOk} · lens-discriminated: rail routes {railRouted} / small-amp cold {railDiscr} / extreme-Q refusal stated {refusedStated} · room-room coincident chain finite={coinFinite} preE={coinPre} E={coinE}"
    -- frozen 2026-07-23 off first-landing measurements (e1DD ≈ 5e-7, eCDD ≈
    -- 6e-6, both collected counter-renders ≈ 1.0): the plain-path chain floor
    -- is 1e-4 — deliverable 2's re-freeze, an order tighter than the bloomed
    -- chain gate's 2e-3 (whose 6.7e-4 is floor-relative, not 1/Δ — diagnosed,
    -- out of v1). The collected/partitioned ratio floor is 1e4×.
    let pass := bitCold == 0 && structOk && lensOk
             && e1DD < 5e-5 && e1Coll > 1000.0 * e1DD
             && eCDD < 1e-4 && eCColl > 1000.0 * eCDD
             && coinFinite && coinPre < 1e-18 && coinE > 1e-9
    if pass then
      passGate "ecdd-partition" s!"the EC/DD choice is the compiler's: cold chain byte-identical (bitDiff 0), tuned unison served through the paired route ({e1DD} vs collected {e1Coll}; chain {eCDD} vs {eCColl}), routing structural + lens-discriminated (rail routes, small-amp cold, extreme-Q refusal stated), coincident room chain graceful"
    else
      failGate "ecdd-partition" s!"bitCold={bitCold} structOk={structOk} railRouted={railRouted} railDiscr={railDiscr} refusedStated={refusedStated} e1DD={e1DD} e1Coll={e1Coll} eCDD={eCDD} eCColl={eCColl} coinFinite={coinFinite} coinPre={coinPre} coinE={coinE}"
  | _, _, _, _, _, _, _ => failGate "ecdd-partition" "build/render failed for an erasure-gate config"

-- ── The σ axis of the accuracy lens (the `dSig` repair) ──────────────────────

/-- Per-route tally over a classified register — the repair's measurement. -/
private structure RouteTally where
  cold    : Nat := 0
  paired  : Nat := 0
  refused : Nat := 0

/-- Classify `n` (voice, room) couplings and tally the three verdicts. Each arm
    draws its ω offset from the ±0.8 rad/s band around the voice partial — the
    ONLY band where either lens can fire (θ_acc = 0.4642 rad/s is 0.0739 Hz) —
    so the histogram measures the ROUTER's decision rather than the trivial fact
    that a log-spaced room sits far from every partial. -/
private def tallyRoutes (n : Nat) (mk : Nat → ModalMode × ModalMode) : RouteTally := Id.run do
  let mut t : RouteTally := {}
  for i in [1:n + 1] do
    let (v, r) := mk i
    match classifyCoupling v r with
    | .cold    => t := { t with cold    := t.cold + 1 }
    | .paired  => t := { t with paired  := t.paired + 1 }
    | .refused => t := { t with refused := t.refused + 1 }
  return t

/-- THE σ-AXIS GATE (the `dSig` repair). `classifyCoupling`'s accuracy lens is a
    POLE-distance test — `|Δ| = √((σ_r−σ_v)² + (ω_v−ω_r)²)`, minimized over the
    declared σ spans. Until the repair the σ term was identically zero (the span
    distance took a `max` where it wanted a `min`), so `|Δ|` collapsed to
    `|ω_v−ω_r|` and any two modes sharing a frequency routed to the paired body
    however differently damped. This gate pins the axis as DECIDING, in both
    directions, and prints the census the repair moved:

    1. SEPARATED IN σ ALONE ⇒ cold — ω matched, `|σ_v−σ_r| = 0.6 > θ_acc`. The
       collected `±c/Δ` form is well conditioned there (Δ is a real 0.6, and the
       rotator-grid degeneracy θ_acc was frozen against is an ω-axis mechanism —
       σ never reaches the rotator), so routing it hot only spent plan size.
    2. NEAR IN σ ⇒ still paired, and the ω axis is UNMOVED (matched σ, dω = 0.05
       ⇒ paired exactly as before). Two witnesses, because a gate that only
       showed the new `cold` would also pass on a classifier that had gone cold
       everywhere.
    3. THE DIPPER SURVIVES — a live-rt60 room whose declared σ span CONTAINS the
       partial's σ has span distance 0 and still routes throughout the knob span
       (WS-LP's D2), while a partial damped far OUTSIDE that span separates from
       it and stays collected: the σ axis decides on a live INTERVAL, not only on
       baked points.
    4. THE REFUSAL REGION SHRINKS — an extreme-Q partial (σ = 0.02) sharing a
       frequency with an ordinary room mode was `refused` (a lens fired on the
       collapsed |Δ|, and the paired cap could not clear at that Q), and a
       refusal is OUTSIDE the merged seam atom's admission. With the σ axis live
       it is an ordinary cold coupling — in contract, correctly represented. -/
def runEcddSigmaAxis (_arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- a baked mode at (ω rad/s, σ, real amp); paired ω values are formed from the
  -- SAME Float, so `dw` is the zero-width difference and `dAbs` is the σ term
  let bake := fun (om sigma amp : Float) =>
    ({ sigma := litF sigma, omega := litF om, cre := litF amp, cim := lit 0 } : ModalMode)
  let pt := fun (fHz sigma amp : Float) => bake (tp * fHz) sigma amp
  -- the shipped rt60 knob's span mapped through σ = 6.91/rt60 (`reverbRoom`)
  let rtSpan : Float × Float := (6.91 / 12.0, 6.91 / 0.2)
  -- the WS-LP live-σ stand-in: `6.91/rt60` over a clamped literal — `reverbRoom`'s
  -- own shape, un-foldable by `sigConstD?`, carrying its declared span
  let liveRoom := fun (om amp : Float) =>
    ({ sigma := div (lit 691 2) (clampE (litF 2.0) (litF 0.2) (litF 12.0)),
       omega := litF om, cre := litF amp, cim := lit 0,
       sigmaRange := some rtSpan } : ModalMode)
  let sepCold  := classifyCoupling (pt 800.0 0.4 0.8) (pt 800.0 1.0 0.6) == .cold
  let nearHot  := classifyCoupling (pt 800.0 0.7 0.8) (pt 800.0 1.0 0.6) == .paired
  let omegaHot := classifyCoupling (pt 800.0 0.6 0.9)
                    (bake (tp * 800.0 + 0.05) 0.6 0.8) == .paired
  let dipper   := classifyCoupling (pt 220.0 3.455 1.0)
                    (liveRoom (tp * 220.0) 0.7) == .paired
  let outSpan  := classifyCoupling (pt 220.0 0.05 1.0)
                    (liveRoom (tp * 220.0) 0.7) == .cold
  let exQ      := classifyCoupling (pt 400.0 0.02 1.0) (pt 400.0 1.0 1.0) == .cold
  -- ARM A — the SHIPPED topology: baked partials over the gong register box
  -- against a LIVE-rt60 room (σ declared over [0.576, 34.55])
  let armA := fun (i : Nat) =>
    let sV := 0.2 + 3.1 * haltonF 3 i
    let fV := 110.0 + 913.0 * haltonF 2 i
    let dOm := (haltonF 7 i - 0.5) * 1.6
    ( bake (tp * fV) sV (0.4 + 0.6 * haltonF 5 i)
    , liveRoom (tp * fV + dOm) (0.4 + 0.6 * haltonF 5 (i + 7919)) )
  -- ARM B — the BAKED topology: a const-σ room (a `modes` data table, a fixture,
  -- a folded room chain), where the σ axis actually carries information
  let armB := fun (i : Nat) =>
    let sV := 0.3 + 1.7 * haltonF 3 i
    let fV := 400.0 + 800.0 * haltonF 2 i
    let sR := 0.3 + 1.7 * haltonF 3 (i + 5003)
    let dOm := (haltonF 7 i - 0.5) * 1.6
    ( bake (tp * fV) sV (0.4 + 0.6 * haltonF 5 i)
    , bake (tp * fV + dOm) sR (0.4 + 0.6 * haltonF 5 (i + 7919)) )
  let a := tallyRoutes 1000 armA
  let b := tallyRoutes 1000 armB
  IO.println "ecdd σ-axis (the accuracy lens reads the whole pole distance):"
  IO.println s!"        witnesses: σ-separated cold {sepCold} · σ-near paired {nearHot} · ω-near paired {omegaHot} · live dipper paired {dipper} · out-of-span cold {outSpan} · extreme-Q refusal retired {exQ}"
  IO.println s!"        route census, 1000 couplings drawn in the ±0.8 rad/s lens band:"
  IO.println s!"          live-rt60 room (shipped reverb topology): cold {a.cold} · paired {a.paired} · refused {a.refused}"
  IO.println s!"          baked const-σ room (data rows, fixtures):  cold {b.cold} · paired {b.paired} · refused {b.refused}"
  let ok := sepCold && nearHot && omegaHot && dipper && outSpan && exQ
         && a.paired > 0 && b.paired > 0
  if ok then
    passGate "ecdd-sigma-axis"
      s!"the lens reads BOTH axes: σ-only separation routes cold, σ-near still pairs, the ω axis is unmoved; live spans included (the dipper survives, an out-of-span partial separates, the extreme-Q refusal retires into ordinary cold) — census: live-room {a.paired}/1000 paired, baked-room {b.paired}/1000"
  else
    failGate "ecdd-sigma-axis"
      s!"sepCold={sepCold} nearHot={nearHot} omegaHot={omegaHot} dipper={dipper} outSpan={outSpan} exQ={exQ} armA={a.cold}/{a.paired}/{a.refused} armB={b.cold}/{b.paired}/{b.refused}"

/-- THE EC/DD LIVE-POLE GATE (fork 3′ Phase 2). Interval classification: a
    live-rt60 room mode tuned ONTO a baked partial routes to the paired DD
    body at BUILD time when min |Δ| over the declared σ span dips under θ_acc
    — the dipper takes DD THROUGHOUT the knob range, so no runtime select
    exists and "no click at θ" is a compile-time triviality (D2). Asserts:
    (1) STRUCTURE — the tuned live coupling routes (paired size 1: ω matched,
        σ span ∋ the partial's σ), an UNTUNED live room routes nothing
        (ordinary reverbs stay byte-identical), and a live σ with NO declared
        range stays collected (unclassifiable — graceful);
    (2) THE LAW ACROSS THE KNOB — the live render matches `oracleSeam` at TWO
        rt60 values (same classification, same DD lane, constants computed by
        kernel arithmetic off the live slot instead of at build time) — the
        "confirm, don't assume" check that the paired columns (`ds`, `sigmaNu`
        as s0 expressions of the knob; the per-sample series/direct `cexpm1`
        select) genuinely lift;
    (3) LIVE + CHAIN (the coverage-matrix cell, decided HERE, not by
        accident): the PLAIN arm serves it — a baked room folds first, the
        live tuned room folds last and partitions against the composed
        (const-foldable) amps. The WS-LP bloomed live arm's refusal of
        composed rooms is untouched (`runBloomLivePole` pins it). -/
def runEcddLive (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let lo := anchorNat + 1
  let nWin : Nat := 32768
  let admAll := fun (_ _ : SeamMode) => true
  -- the partial sits at σ = 3.455 = 6.91/2.0 — INSIDE the σ span of an
  -- rt60 ∈ [0.2, 12] room ([0.576, 34.55]) ⇒ with ω matched, min|Δ| = 0
  let voice : Array SeamMode := #[mkMode 220 3.455 1.0, mkMode 610 1.3 0.6]
  let vM := voice.map (·.toModal)
  let room1 : Array SeamMode := #[mkMode 200 1.2 1.0, mkMode 380 1.6 0.8]
  let r1M := room1.map (·.toModal)
  let (rtLo, rtHi) : Float × Float := (0.2, 12.0)
  let liveSigma := fun (rv : Float) =>
    div (lit 691 2) (clampE (litF rv) (litF rtLo) (litF rtHi))
  let liveRoomAt := fun (rv : Float) (fHz amp : Float) (range? : Option (Float × Float)) =>
    #[({ sigma := liveSigma rv, omega := litF (tp * fHz),
         cre := litF amp, cim := lit 0,
         sigmaRange := range? } : ModalMode)]
  let span := some (6.91 / rtHi, 6.91 / rtLo)
  let graphOf := fun (rooms : Array (Array ModalMode)) =>
    let mkRev := fun (i : Nat) (rm : Array ModalMode) =>
      ({ id := s!"rev{i}", node := .modalReverb (if i == 0 then "src" else s!"rev{i-1}") rm none })
    ({ nodes := #[⟨"src", .modalSource vM anchorSig clockLit none none none⟩]
              ++ rooms.zipIdx.map (fun (rm, i) => mkRev i rm)
       output := s!"rev{rooms.size - 1}" } : PatchGraph)
  -- (1) structure: tuned+ranged routes; untuned stays cold; rangeless stays cold
  let tunedAt := fun rv => liveRoomAt rv 220.0 0.7 span
  let (_, pTuned) := residueComposePartitioned vM (tunedAt 2.0)
  let (_, pUntuned) := residueComposePartitioned vM (liveRoomAt 2.0 500.0 0.7 span)
  let (_, pNoRange) := residueComposePartitioned vM (liveRoomAt 2.0 220.0 0.7 none)
  let (_, pChain) := foldRoomsPartitioned vM #[r1M, tunedAt 2.0]
  let structOk := pTuned.size == 1 && pUntuned.isEmpty && pNoRange.isEmpty
               && pChain.size == 1
  -- (2) the law at two knob values (single compose), (3) through a chain
  let oracleAt := fun (rv : Float) => oracleSeam voice
    #[{ sigma := 6.91 / rv, omega := tp * 220, are := 0.7 }] idWarp admAll 8 nWin
  let oracleChainAt := fun (rv : Float) => oracleSeam voice
    (foldRoomsFloat room1 #[{ sigma := 6.91 / rv, omega := tp * 220, are := 0.7 }])
    idWarp admAll 8 nWin
  -- (4) OVERDRIVE — the uniform-bank clamp (`clampSigmas` at graph ingestion):
  -- a σ expression driven OUT of the declared span (rt60 = 0.1 through a
  -- wide value clamp, so the raw σ evaluates to 69.1; the DECLARED span is
  -- still [0.576, 34.55]) saturates in-kernel to the declared edge. The whole
  -- bank — collected modes and the paired atom alike — must behave as
  -- rt60 = 0.2: out-of-span drives can neither desync the two lane families
  -- nor walk a cold-classified coupling's |Δ| under θ_acc.
  let oobRoom : Array ModalMode :=
    #[{ sigma := div (lit 691 2) (clampE (litF 0.1) (litF 0.05) (litF 30.0)),
        omega := litF (tp * 220), cre := litF 0.7, cim := lit 0,
        sigmaRange := span }]
  match ← renderGraphN arena "ecddlive_a" (graphOf #[tunedAt 2.0]) nWin,
        ← renderGraphN arena "ecddlive_b" (graphOf #[tunedAt 4.0]) nWin,
        ← renderGraphN arena "ecddlive_chain" (graphOf #[r1M, tunedAt 2.0]) nWin,
        ← renderGraphN arena "ecddlive_oob" (graphOf #[oobRoom]) nWin with
  | .ok dutA, .ok dutB, .ok dutC, .ok dutO =>
    let eA := relL2Win dutA (oracleAt 2.0) lo nWin
    let eB := relL2Win dutB (oracleAt 4.0) lo nWin
    let eC := relL2Win dutC (oracleChainAt 2.0) lo nWin
    let eO := relL2Win dutO (oracleAt 0.2) lo nWin
    IO.println s!"ecdd live-pole (interval classification over the declared rt60 span):"
    IO.println s!"        structure ok={structOk} (tuned routes, untuned/rangeless stay collected, chain routes at the final fold)"
    IO.println s!"        law: rt60=2.0 rel {eA} · rt60=4.0 rel {eB} (one DD lane across the knob) · live+chain rel {eC}"
    IO.println s!"        overdrive: rt60 driven to 0.1 (out of span) ≡ oracle at the declared edge 0.2 rel {eO} (the whole bank saturates together)"
    -- frozen 2026-07-23 off first-landing measurements (eA/eB ≈ 5e-7, the
    -- single-crossing floor; eC ≈ 1.7e-5): the live lane owes the same floors
    -- as the baked lane — the lift changes WHERE constants are computed, not
    -- their accuracy.
    let pass := structOk && allFinite dutA && allFinite dutO
             && eA < 5e-5 && eB < 5e-5 && eC < 1e-4 && eO < 5e-5
    if pass then
      passGate "ecdd-live" s!"a live-rt60 room tuned onto a partial classifies over its declared interval and takes the DD lane throughout ({eA} / {eB} at two knob values, no runtime select to click); live+chain served on the plain arm ({eC}); untuned/rangeless live rooms stay collected; an out-of-span drive saturates the WHOLE bank to the declared edge ({eO})"
    else
      failGate "ecdd-live" s!"structOk={structOk} eA={eA} eB={eB} eC={eC} eO={eO}"
  | _, _, _, _ => failGate "ecdd-live" "build/render failed for a live-pole erasure config"

/-- THE GAUGE-OVER-HOT-CHAIN GATE (the gauge × partition interaction, measured
    rather than assumed). Gauge is a COLLECTED surface in v1: it forces the
    room fold at the gauge node, so a tuned (sub-grid) coupling under a gauge
    renders collected — the status-quo floor for that pair (grid silence),
    stated in `lowerModal`. The open question was the NORM: `gaugeScale`
    self-measures on the collected amps, which for a tuned unison are the huge
    cancelling `±c/Δ` pair (~7e5 here) — does the measure survive, or does it
    mis-scale the WHOLE bank? It survives, and not by luck: the norm reads the
    bank's transfer function `H(iω)`, and `H` has NO `1/Δ` pole — the `±c/Δ`
    partial fractions recombine into the bounded product form
    `c/((iω−λ)(iω−ν))`, so the f64 build-time cancellation is benign (rel err
    ~eps·|c/Δ|/|H| ≈ 1e-10 at this detune; an f32 coefficient path would owe
    its own witness — recorded, not covered here). Asserts:
    (1) THE SCALE LAW — the gauged render is `scale · (ungauged collected
        render)` with `scale` recomputed from an INDEPENDENT f64 mirror of
        `gaugeScale` on the float-EC amps (the render is linear in residues:
        one shared s0 scale, so any norm damage would show here). Asserted at
        a REPRESENTABLE detune (Δ = 1e-3 > the grid quantum; amps ±700) where
        the datapath resolves the signal;
    (2) NORM SANITY at the sub-grid worst case (|c/Δ| ≈ 7e5) — the mirrored
        scale sits in a sane band: the "scaled to oblivion" failure mode is
        excluded by measurement;
    (3) THE LANDING POISON, recorded as data (this gate's own finding): at
        sub-grid detune the scale-law comparison is dominated by QUANTIZATION,
        not the norm — the cancelling `±c/Δ` amps size the per-bank landing
        exponent (`bankLandExp` k off maxAbs ≈ 7e5 ⇒ LSB ≈ 2⁻¹³), which
        quantizes the surviving cold modes (amps ~1e-4) to ~1 LSB. The
        collected floor under a forced-collected surface is therefore not just
        "the hot pair is silent": the pair's amps degrade the WHOLE bank's
        landing resolution. Where the partition routes, this cannot happen
        (paired amps are bounded, `ecddPairCap`) — one more reason the routed
        path is the served one. Pinned loosely (finite + the poisoned
        comparison stays O(1), i.e. the bank still renders the cold modes at
        the right ORDER, no blowup). -/
def runEcddGauge (arena : Arena)
    (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let lo := anchorNat + 1
  let nWin : Nat := 32768
  let voice : Array SeamMode := #[mkMode 220 1.0 1.0, mkMode 610 1.3 0.6]
  let vM := voice.map (·.toModal)
  -- the f64 norm mirror: float-EC amps → H at each pole ω → S = Σ|H|⁸ →
  -- scale = S^{−g/8} (g = 1) — `gaugeScale`'s formula on a separate code
  -- path (CplxB floats, not symbolic CplxE).
  let scaleOf := fun (room : Array SeamMode) =>
    let ecF := foldRoomsFloat voice room
    let hAt := fun (wk : Float) => ecF.foldl (fun (acc : CplxB) m =>
        acc.add ((⟨m.are, m.aim⟩ : CplxB).div (⟨m.sigma, wk - m.omega⟩ : CplxB))) ⟨0, 0⟩
    let sNorm := ecF.foldl (fun s m =>
        let h := hAt m.omega
        let h2 := h.re * h.re + h.im * h.im
        s + (h2 * h2) * (h2 * h2)) 0.0
    Float.exp (-(1.0 / 8.0) * Float.log sNorm)
  let gaugedGraph := fun (rm : Array ModalMode) =>
    ({ nodes := #[ { id := "src", node := .modalSource vM anchorSig clockLit none none none }
                 , { id := "rev", node := .modalReverb "src" rm none }
                 , { id := "gg",  node := .modalGauge "rev" (litF 1.0) } ]
       output := "gg" } : PatchGraph)
  -- config A: the sub-grid worst case (norm sanity + the landing poison)
  let roomSub : Array SeamMode := #[{ sigma := 1.0, omega := tp * 220 + 1e-6, are := 0.7 }]
  -- config B: representable detune (the clean scale-law witness — amps ±700,
  -- landing exponent k = 6, quantization decades under the signal)
  let roomRep : Array SeamMode := #[{ sigma := 1.0, omega := tp * 220 + 1e-3, are := 0.7 }]
  let rSubM := roomSub.map (·.toModal)
  let rRepM := roomRep.map (·.toModal)
  match ← renderGraphN arena "ecddg_gauged_sub" (gaugedGraph rSubM) nWin,
        ← renderTerm arena "ecddg_bare_sub" (modalBankTerm (residueComposeEC vM rSubM) anchorSig clockLit) nWin,
        ← renderGraphN arena "ecddg_gauged_rep" (gaugedGraph rRepM) nWin,
        ← renderTerm arena "ecddg_bare_rep" (modalBankTerm (residueComposeEC vM rRepM) anchorSig clockLit) nWin with
  | .ok dutGS, .ok dutBS, .ok dutGR, .ok dutBR =>
    let scaleSub := scaleOf roomSub
    let scaleRep := scaleOf roomRep
    let eLawRep := relL2Win dutGR (dutBR.map (· * scaleRep)) lo nWin
    let ePoison := relL2Win dutGS (dutBS.map (· * scaleSub)) lo nWin
    let eGS := energyWin dutGS lo nWin
    let saneSub := 0.1 < scaleSub && scaleSub < 10.0
    IO.println s!"ecdd gauge-over-hot (the norm on cancelling ±c/Δ amps):"
    IO.println s!"        scale law (Δ=1e-3, resolvable): gauged ≡ {scaleRep} × collected rel {eLawRep}"
    IO.println s!"        sub-grid (|c/Δ|≈7e5): norm sane {saneSub} (scale {scaleSub}) · landing-poison comparison {ePoison} (quantization-dominated, recorded) · E {eGS}"
    if allFinite dutGS && allFinite dutGR && eLawRep < 1e-4
        && saneSub && ePoison < 2.0 && eGS > 1e-9 then
      passGate "ecdd-gauge" s!"gauge over a tuned-unison chain: the H-norm survives the ±c/Δ amps (sub-grid scale {scaleSub}, mirrored independently — no oblivion; scale law holds where the datapath resolves, {eLawRep}); the recorded residual is the collected floor PLUS the landing poison (huge amps size the bank's k, LSB over the cold modes)"
    else
      failGate "ecdd-gauge" s!"eLawRep={eLawRep} scaleSub={scaleSub} scaleRep={scaleRep} ePoison={ePoison} finite={allFinite dutGS}/{allFinite dutGR} E={eGS}"
  | _, _, _, _ => failGate "ecdd-gauge" "build/render failed for a gauge-over-hot config"

end Tropical.Tropicaltest.SeamSweep
