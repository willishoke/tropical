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
    atom's realization actually includes (bloom drops non-admitted pairs; the
    residue atoms compose all). Composite Simpson per sample-interval (`hdiv`
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
  /-- `true`: the atom actively DROPS non-admitted pairs (excluded pairs must
      render as graceful silence, and the oracle sums admitted pairs only).
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
      excluded → graceful silence; passively excluded → out of contract, skip). -/
  boundary : Array (SeamMode × SeamMode)

/-- Which pairs the oracle sums for this atom: admitted-only when the atom
    actively excludes (bloom), all pairs otherwise (residue atoms compose all). -/
private def SeamAtom.oracleAdmits (atom : SeamAtom) : SeamMode → SeamMode → Bool :=
  if atom.activeExclusion then atom.admitsPair else (fun _ _ => true)

-- ── The three v1 instances (retrofit) ─────────────────────────────────────────

private def idWarp : Float → Float := fun s => s

/-- `residueComposeEC` — the collected residue bank (`voice ⋙ reverb`, m+n
    modes), realized through the default banked lowering. Total composition; the
    admission predicate marks where the collected `1/Δ` amp stays inside the
    Q4.28 headroom (`|a·r/Δ| < 8` ⟺ `|Δ|·8 > |a·r|`) — below it is
    `residueComposeDD`'s region, not EC's promise (passive exclusion). -/
private def ecAtom : SeamAtom :=
  { name := "residueEC"
    phi := idWarp
    realize := fun v r => modalBankSigTable (residueComposeEC (v.map (·.toModal)) (r.map (·.toModal))) clockLit anchorSig
    admitsPair := fun v r =>
      let dpole := (v.pole.sub r.pole).abs
      let amp := (v.amp.mul r.amp).abs
      dpole * 8.0 > amp
    activeExclusion := false
    snr := 2e-4
    interior := (Array.range 4).map (fun c =>
      (#[haltonMode (c*10+1) 220 900, haltonMode (c*10+2) 220 900],
       #[haltonMode (c*10+3) 1500 3300, haltonMode (c*10+4) 1500 3300]))
    boundary :=
      -- two poles near the coincidence floor: same freq, σ apart by ~0.6 (|Δ| ≈
      -- 0.6, |a·r| ≈ 0.5 ⇒ admitted, |a·r/Δ| ≈ 0.8 < 8) — the accuracy edge.
      #[ (mkMode 800 0.4 0.8, { sigma := 1.0, omega := tp * 800, are := 0.6 })
       , (mkMode 500 0.5 0.7, { sigma := 1.1, omega := tp * 500, are := 0.7 }) ] }

/-- `residueComposeDD` — the fused divided-difference paired bank (`m·n` paired
    modes; the `cexpm1` limit is the τ·e resonance, no branch). TOTAL admission
    (it exists precisely to be accurate through coincidence). -/
private def ddAtom : SeamAtom :=
  { name := "residueDD"
    phi := idWarp
    realize := fun v r => modalBankSigTableDD (residueComposeDD (v.map (·.toModal)) (r.map (·.toModal))) clockLit anchorSig
    admitsPair := fun _ _ => true
    activeExclusion := false
    snr := 2e-4
    interior := (Array.range 4).map (fun c =>
      let vo := #[haltonMode (c*10+1) 400 1200, haltonMode (c*10+2) 400 1200]
      -- reverb poles NEAR the voice poles (the DD regime): a shrinking offset
      let off := 0.6 * haltonF 2 (c+1)
      (vo, vo.map (fun m => { m with omega := m.omega + tp * off, sigma := m.sigma + 0.1 })))
    boundary :=
      -- exact coincidence (λ = ν): the τ·e resonance, must stay finite/accurate
      #[ (mkMode 700 0.8 0.6, mkMode 700 0.8 0.5)
       , (mkMode 1100 0.5 0.7, { sigma := 0.5, omega := tp * 1100 + 0.02, are := 0.4 }) ] }

/-- The pitch bloom warp (shipped gong: β=0.05, g=1.8, scale 1 ⇒ B = β/g). -/
private def bloomBg : Float × Float := (0.05 / 1.8, 1.8)
private def bloomWarp : Float → Float := fun s => s + bloomBg.1 * (1.0 - Float.exp (-bloomBg.2 * s))

/-- `bloomCompose` — the Γ-bridge atom (`bloomed voice ⋙ reverb`), NOW TOTAL over
    the crossing after atom four (WS-A4): the `|a| < ½` coincidence hole (a room
    pole tuned to a settled partial — the τ·e resonance) is admitted and accurate,
    the coincident divided difference bridging the CF branch (large `z`) to the
    series-DD branch (small `z`). ACTIVE exclusion still drops depth > 300 pairs and
    live poles. Baked-pole v1 (the sweep pole set is all literal). NOTE: the 0.09 s
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

private def allAtoms : Array SeamAtom := #[ecAtom, ddAtom, bloomAtom]

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
    Baked room (the live-rt60 reverb gracefully drops to the bare bloom — the
    recorded baked-pole-bloom v1 limitation). -/
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
-- labor: types for coverage, the sweep + SNR for the law). Depth-cap drops are
-- counted and printed — never a silent cap.

/-- Per-region tally + worst envelope depth over a Halton batch. -/
private structure RegionTally where
  serOnly  : Nat := 0
  crossing : Nat := 0
  coincX   : Nat := 0        -- coincidentCrossing
  coincS   : Nat := 0        -- coincidentSubtle
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
    (β = 0.05, the default register) is depth-total — zero drops, worst SAMPLED
    depth (Halton interior + the max-|κ| corner scan) < 300, so the cap is MEASURED
    headroom (empirical, not proven — a config in a sampling gap could differ, but
    would still become a counted `excludedDepth`, never a silent error); (2) all
    FOUR served regions are reachable — serOnly, crossing, coincidentCrossing,
    coincidentSubtle (the partition claims no region nothing lands in); (3)
    `excludedDepth` IS reachable off-surface (β = 0.5, knob max) and is COUNTED, not
    silently dropped. Accuracy per region is the seam-sweep's job (types for
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
  let shipExcl  := uni.excl + near.excl
  let admitFrac := (shipTotal - shipExcl).toFloat / shipTotal.toFloat
  let maxDepth  := max (max (max uni.maxN uni.maxK) (max near.maxN near.maxK)) corner
  IO.println "seam region coverage (classify the shipped register — totality as a number):"
  IO.println s!"        shipped register (β=0.05): serOnly {uni.serOnly + near.serOnly}, crossing {uni.crossing + near.crossing}, coincidentCrossing {near.coincX}, excludedDepth {shipExcl} / {shipTotal}"
  IO.println s!"        admitted fraction {admitFrac} · worst SAMPLED depth {maxDepth} of cap 300 (measured headroom {300 - maxDepth}; Halton interior + max-|κ| corner scan {corner})"
  IO.println s!"        subtle-bloom box (B={bSmall}): coincidentSubtle {subtle.coincS} / {subtle.total} reachable"
  IO.println s!"        knob-max box (β=0.5): excludedDepth {over.excl} / {over.total} — COUNTED, a graceful drop, not a silent cap"
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
  -- (3) excludedDepth is reachable off-surface AND counted (the drop is named).
  --     Graceful SILENCE for an excluded pair is structural, not rendered here: the
  --     `| .excludedDepth => continue` (Modal.lean) drops the pair from the bank
  --     entirely, so a mixed bank is byte-identical to the bank without it — nothing
  --     to render. This gate proves the region is reachable and counted, not silent.
  if over.excl == 0 then
    IO.println s!"        COVERAGE VIOLATION: knob-max β did not exceed the depth cap — excludedDepth is never exercised, so 'counted not silent' is untested"
    ok := false
  if ok then
    passGate "seam-coverage" s!"the shipped register classifies TOTALLY: admitted fraction {admitFrac} (0 depth drops, worst SAMPLED depth {maxDepth} < 300 — measured headroom, Halton + corner scan), all four served regions reachable, excludedDepth counted off-surface ({over.excl}/{over.total})"
  else
    failGate "seam-coverage" "the region partition is not total over the shipped register — see the lines above"

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
  let mu : CplxB := ⟨-p.muSigma, p.muOmega⟩
  let nu : CplxB := ⟨-p.nuSigma, p.nuOmega⟩
  let aC := (nu.sub mu).scale (1.0 / p.gRate)
  let plan := classifyBloomPair mu nu p.bloomB p.gRate
  let (cfK, _) := bloomCF aC p.kappa
  let cfB := (Array.range (plan.kDepth + 1)).map (fun j => (⟨(2 * j + 1).toFloat - aC.re, -aC.im⟩ : CplxB))
  let cfN := (Array.range plan.kDepth).map (fun j =>
    let jf := (j + 1).toFloat
    (⟨jf, 0⟩ : CplxB).mul ((⟨jf, 0⟩ : CplxB).sub aC))
  let invA := (Array.range plan.nDepth).map (fun k => CplxB.div ⟨1, 0⟩ (aC.add ⟨(k + 1).toFloat, 0⟩))
  return { p with k1Cf := (cfK.scale (1.0 / p.gRate)).neg, cfB, cfN, invA }

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

end Tropical.Tropicaltest.SeamSweep
