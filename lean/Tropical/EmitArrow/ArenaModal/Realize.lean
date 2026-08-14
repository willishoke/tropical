import Tropical.EmitArrow.ArenaModal.Live

/-!
# EmitArrow.Modal.Realize

Modal-bank realization, strike trains, and direction-aware signal terms.
-/

namespace Tropical.EmitArrow.ArenaNative

open Tropical.Ir
open Tropical.Exact (DyadicI CplxD CplxDI)

-- ── The MODAL ISLAND (v1): a decaying-resonator bank as a term over the clock ──
-- The pole/modal island's emit path. A bank is a gated sum of decaying sinusoids
-- (`modalBankSig`) — the real part of Σ amp·e^{μd}. It needs NO new ArrowTerm
-- node: as a pure function of the warped clock it rides `arrUn … (.clk c)`, so
-- warps reclock it (affine = exact pole reclocking; nonlinear = the defined
-- varispeed case) and it is random-access by construction. The residue calculus
-- (voice ⋙ reverb) is a BUILD-TIME pass that fills the ModalMode array; the
-- runtime substrate is just this bank. Gated by `modal-bank`.

/-- Banks-as-data is the DEFAULT lowering: uniform (all deg-0) modal banks lower
    through the indexed reduction (`bankFold` — `modalBankSigTable` /
    `modalBankSigDirTable`) instead of unrolling. Sound because the banked
    render is bit-identical to the unroll (order-preserving loop, i64-modular
    sum — the `banks-as-data`/`banks-as-data-dir` gates pin it) and the
    coefficient columns are generation-buffered in FlatRuntime (no cross-column
    tear on live knob moves). `TROPICAL_BANKS_UNROLL` is the escape hatch back
    to the unrolled form (bisection ladder: the naive realization stays
    reachable). Reads the ONE shared flag (`Ir.banksEnabled`), read once at
    load, so the pure lowering may branch on it. -/
def banksTableEnabled : Bool := Tropical.Ir.banksEnabled

/-- A modal bank struck at `anchor` (samples) as a term over the clock leaf: no
    `gen`, no `.trop` instance — `{clk, +, ×, round, clamp, ldexp}` all the way
    down, and warp-reachable like any generator. Rides `arrUn … (.clk c)`, so
    warps reach the (banked or unrolled) body through the clock leaf identically. -/
def modalBankTerm (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (count? : Option Sig := none) : ArrowTerm :=
  -- A dynamic-count bank ALWAYS banks: a runtime count cannot be unrolled, so
  -- when `count?` is present the banked lowering fires regardless of
  -- TROPICAL_BANKS_UNROLL — the escape hatch governs only STATIC banks. (A
  -- non-uniform bank can't take the table lowering at all; it unrolls at
  -- capacity and the live count is dropped — graceful, and unreachable from
  -- the Playground, whose resonator banks are all deg-0.)
  let banked := bankIsUniform modes && (count?.isSome || banksTableEnabled)
  let lower := if banked
    then fun ms clk a => modalBankSigTable ms clk a count?
    else fun ms clk a => modalBankSig ms clk a
  ArrowTerm.arrUn (fun clkSig => lower modes clkSig anchor) (ArrowTerm.clk c)

/-- The PARTITIONED bank: the collected component through `modalBankTerm`'s
    exact lowering plus the paired component (`modalBankSigTableDD`) summed at
    the SAME clock leaf — one `arrUn`, so warps/scrub reach both bodies through
    the shared clock identically. `paired = #[]` falls through to
    `modalBankTerm` VERBATIM (the byte-identity discipline: a cold partition
    emits today's term). The two bank regions are sequential siblings, so the
    `idxId 0` reuse is safe (the `modalBankSigPairTable` precedent). -/
def modalBankTermPartitioned (plain : Array ModalMode) (paired : Array PairedMode)
    (anchor : Sig) (c : Clock) (count? : Option Sig := none) : ArrowTerm :=
  if paired.isEmpty then modalBankTerm plain anchor c count? else
  let banked := bankIsUniform plain && (count?.isSome || banksTableEnabled)
  let lowerP := if banked
    then fun ms clk a => modalBankSigTable ms clk a count?
    else fun ms clk a => modalBankSig ms clk a
  ArrowTerm.arrUn
    (fun clkSig => do
      let plainSig ← lowerP plain clkSig anchor
      let pairedSig ← modalBankSigTableDD paired clkSig anchor
      add plainSig pairedSig)
    (ArrowTerm.clk c)

-- ── The strike train: quotient clock × comb factor (the CF sequencer, tier 0) ──
-- A periodically re-struck bank is NOT an event list: the infinite past of a
-- period-P strike train sums, per pole, to the geometric factor 1/(1 − e^{λP})
-- (a feedback-comb transfer function evaluated at the pole), read on the
-- quotient clock d = (τ − anchor) mod P. The ramp-into-resonator patch is the
-- J = 0 truncation of this (hard retrigger, the tail cut at the wrap — an
-- artifact click); the comb factor is the J = ∞ completion: across the wrap
-- the value steps by EXACTLY one fresh strike's onset
-- (e^{λP}/(1−e^{λP}) + 1 = 1/(1−e^{λP})) — the physical transient — while
-- every previous cycle rings through. Pattern (weights + intra-bar offsets)
-- is authoring-level: each strike is the same combed bank at a shifted
-- quotient anchor. Coefficients, not topology; offsets are free within the
-- bar (microtiming is a phase factor, not a grid).

/-- The comb factor `1/(1 − e^{λP})` for one mode, as s0 `CplxE`. σ may be
    LIVE (the factor becomes an s0 expression of the slot — Stage0 hoists it,
    the same lift discipline as every modal constant); ω must const-fold
    (`none` otherwise — no served surface has a live modal ω). A baked σ
    folds the factor to literals so `bankLandExp` sizes the landing for it —
    which matters at the NEAR-SINGULARITY: a mode ringing at a harmonic of
    the strike rate with small σP (ωP ≈ 2πk) builds resonantly under periodic
    driving, the factor is large, and it lands in the amps where the landing
    exponent sees it. (A live-σ factor leaves the amps unfoldable and the
    landing falls back exactly as gauge output does.) -/
def combFactorE (m : ModalMode) (pSec : Float) : BuildM (Option CplxE) := do
  let some wv ← sigConstF? m.omega | return none
  let pD := DyadicI.ofFloat pSec
  -- ω·P reaches ~1e6 rad for an audio partial at a bar-length period; the
  -- carrier reduces against a 300-bit π, so the reduced argument still lands a
  -- full working mantissa where a double has already spent ~20 of its 53. This
  -- factor sits at a near-singularity (`1/(1−e^{λP})` with ωP ≈ 2πk) whose value
  -- the landing exponent reads, so those bits are not decorative.
  let wpD := DyadicI.mul (DyadicI.ofFloat wv) pD
  let eS ← match ← sigConstF? m.sigma with
    | some sv => litF (DyadicI.toFloat
        (DyadicI.exp (DyadicI.neg (DyadicI.mul (DyadicI.ofFloat sv) pD))))
    | none => do
      let period ← litF pSec
      let sigmaPeriod ← mul m.sigma period
      let negative ← neg sigmaPeriod
      expSig negative
  let cosine ← litF (DyadicI.toFloat (DyadicI.cos wpD))
  let real ← mul eS cosine
  let sine ← litF (DyadicI.toFloat (DyadicI.sin wpD))
  let imag ← mul eS sine
  let eLamP : CplxE := (real, imag)
  let oneReal ← lit 1
  let zeroImag ← lit 0
  let one : CplxE := (oneReal, zeroImag)
  let denominator ← csubE one eLamP
  let factor ← cdivE one denominator
  pure (some factor)

/-- Scale a bank's residues by the period-P comb factor (the strike train's
    steady state). A mode whose ω does not fold keeps its bare amp — the
    retrigger-without-overlap reading, stated (unreachable from served
    surfaces, where modal ω always folds). -/
def combScale (pSec : Float) (modes : Array ModalMode) : BuildM (Array ModalMode) :=
  modes.mapM fun m => do
    match ← combFactorE m pSec with
    | some f =>
      let a ← cmulE m.ampE f
      pure { m with cre := a.1, cim := a.2 }
    | none => pure m

/-- The strike train as a `Sig`: the combed bank read on the quotient clock,
    summed over the bar's strikes `(offset seconds, weight)` — each strike is
    the SAME combed bank at a shifted periodic anchor, so an N-strike bar
    costs N bank reads. Banked/unrolled dispatch mirrors `modalBankTerm`;
    the strike regions are sequential siblings, so the `idxId 0` reuse is
    safe (the `modalBankSigPairTable` precedent). An empty strike list is
    silence (the graceful-silence contract). -/
def strikeTrainSig (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (pSec : Float) (strikes : Array (Float × Float) := #[(0.0, 1.0)])
    (count? : Option Sig := none) : BuildM Sig := do
  let banked := bankIsUniform modes && (count?.isSome || banksTableEnabled)
  let lower := if banked
    then fun ms clk a => modalBankSigTable ms clk a count?
    else fun ms clk a => modalBankSig ms clk a
  let combed ← combScale pSec modes
  let zero ← lit 0
  strikes.foldlM (init := zero) fun acc (off, w) => do
    let weight ← litF w
    let ms ← combed.mapM fun m => do
      let cre ← mul weight m.cre
      let cim ← mul weight m.cim
      pure { m with cre, cim }
    let offset ← litF off
    let sr ← sampleRate
    let offsetSamples ← mul offset sr
    let anchorK ← add anchorSamples offsetSamples
    let quotient ← relClockQuot clkInt anchorK pSec
    let signal ← lower ms quotient zero
    add acc signal

/-- The strike train as a term: one `arrUn` at the clock leaf, so a warped
    master clock warps the WHOLE train — tempo rubato reaches the strikes and
    the tails through the same coordinate (swing later composes UPSTREAM of
    the quotient, never inside it). -/
def strikeTrainTerm (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (pSec : Float) (strikes : Array (Float × Float) := #[(0.0, 1.0)])
    (count? : Option Sig := none) : ArrowTerm :=
  ArrowTerm.arrUn
    (fun clkSig => strikeTrainSig modes clkSig anchor pSec strikes count?)
    (ArrowTerm.clk c)

-- ── The DIRECTION operator: forward↔reverse crossfade ─────────────────────────
-- This low-level primitive crossfades one bank between its CAUSAL tail (energy at
-- d>0) and ANTI-CAUSAL time mirror (energy at d<0). Both keep the mode's own σ and
-- ω; only which side of the strike carries energy changes. Public `reverb.dir`
-- applies this orientation to that room's kernel before convolution with its modal
-- input. Applying it to an already-composed source/room bank would instead reverse
-- the complete output, which is a separate clock/warp operation.

/-- A modal bank read with a forward↔reverse orientation crossfade `dir ∈ [0,1]`.
    Per mode: the CAUSAL ring `e^{−σd}` gated `d>0`, and the ANTI-CAUSAL ring
    `e^{+σd} = e^{−σ|d|}` gated `d<0` reading the time-mirrored oscillator (`cos`
    even, `sin` odd), blended `(1−dir)·forward + dir·reverse`. σ and ω are untouched,
    so no setting can swing the frequency into the damping. `dir=0` reduces
    bit-for-bit to the forward bank (`modalBankSig`); `dir=1` is its exact time-mirror
    as a bank. When the bank is a room kernel, the public room law convolves this
    oriented kernel with the upstream value; it does not orient their complete
    composed output. `dampScale?` bends the decay clock (sway) on both sides. Pure
    `f(clk)`: no state. -/
def modalBankSigDir (modes : Array ModalMode) (clkInt : Sig) (anchorSamples : Sig)
    (dir : Sig) (dampScale? : Option Sig := none) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  -- Q datapath, same ledger as `modalBankSig` (Q(4+k).(28−k) weight landings via
  -- option E's `le`, exact Q2.30 oscillators, i64 mode sums). The causal gates
  -- and the dir crossfade hoist OUT of the per-mode fold. `le` is derived from
  -- the AMP columns only (`bankLandExp` never reads env): the reverse arm's
  -- `envR = e^{+σd}` is > 1 for d > 0, but that landing is DISCARDED by the
  -- reverse `selectE` (active only for clkRel < 0, where envR ≤ 1) — its
  -- non-taken poison must not force k upward, and the audible reverse tail is the
  -- forward tail's time-mirror, equally |A|-bounded. So the same `le` serves both
  -- arms and `dir=0` stays bit-identical to `modalBankSig`.
  let le ← bankLandExp modes
  let landingScale ← le.scale
  let landingShift ← le.shift
  let zeroInt ← litI 0
  let (fwdQ, revQ) ← modes.foldlM
    (fun (acc : Sig × Sig) m => do
      let phQ ← modePhaseQ m.omega clkRel
      -- osc(−d) as the phase of the NEGATED relative clock (not cos-even/
      -- sin-odd), so `dir=1` is the forward tail's exact time-mirror — the
      -- fixed sine isn't bit-symmetric (one signed floor-shift), so the
      -- mirrored phase must be spelled out to match `fwd[2C−i]`. `modePhaseQ`
      -- is exact on the negated i64, so the rev side at `clkRel = −X`
      -- evaluates at bit-equal phase to the fwd side at `+X`.
      let negativeClock ← neg clkRel
      let phQN ← modePhaseQ m.omega negativeClock
      -- σ·d, optionally sway-bent (decay clock only, so pitch is untouched).
      let sigmaTime ← mul m.sigma dSec
      let sd ← match dampScale? with
        | none => pure sigmaTime
        | some scale => mul sigmaTime scale
      let negativeSd ← neg sd
      let forwardBase ← expSig negativeSd
      let envF ← if m.deg == 0 then pure forwardBase else do
        let power ← powE dSec m.deg
        mul power forwardBase
      let reverseBase ← expSig sd
      let envR ← if m.deg == 0 then pure reverseBase else do
        let negativeSec ← neg dSec
        let power ← powE negativeSec m.deg
        mul power reverseBase
      let forwardCre ← mul envF m.cre
      let forwardCreScaled ← mul forwardCre landingScale
      let wCreF ← toIntE forwardCreScaled
      let forwardCim ← mul envF m.cim
      let forwardCimScaled ← mul forwardCim landingScale
      let wCimF ← toIntE forwardCimScaled
      let reverseCre ← mul envR m.cre
      let reverseCreScaled ← mul reverseCre landingScale
      let wCreR ← toIntE reverseCreScaled
      let reverseCim ← mul envR m.cim
      let reverseCimScaled ← mul reverseCim landingScale
      let wCimR ← toIntE reverseCimScaled
      let forwardCos ← fixedCosCycSig phQ
      let forwardReal ← mul wCreF forwardCos
      let forwardSin ← fixedSinCycSig phQ
      let forwardImag ← mul wCimF forwardSin
      let forwardDifference ← sub forwardReal forwardImag
      let fwdM ← rshift forwardDifference landingShift
      let reverseCos ← fixedCosCycSig phQN
      let reverseReal ← mul wCreR reverseCos
      let reverseSin ← fixedSinCycSig phQN
      let reverseImag ← mul wCimR reverseSin
      let reverseDifference ← sub reverseReal reverseImag
      let revM ← rshift reverseDifference landingShift
      let fwdAcc ← add acc.1 fwdM
      let revAcc ← add acc.2 revM
      pure (fwdAcc, revAcc))
    (zeroInt, zeroInt)
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  let forwardOutput ← fixedOutQ 30 fwdQ
  let fwd ← selectE afterStrike forwardOutput zero
  let beforeStrike ← gt zero clkRel
  let reverseOutput ← fixedOutQ 30 revQ
  let rev ← selectE beforeStrike reverseOutput zero
  let one ← lit 1
  let forwardMix ← sub one dir
  let forwardTerm ← mul forwardMix fwd
  let reverseTerm ← mul dir rev
  add forwardTerm reverseTerm

/-- The BANKED direction bank: `modalBankSigDir`'s value with BOTH mode sums as
    indexed reductions — the forward and reverse accumulators are two `bankFold`s
    over the SAME coefficient columns (a pair-valued fold is two scalar folds;
    each visits modes in array order, so each i64 sum agrees bit-for-bit with its
    unrolled half). This is NOT a hand table twin: both bodies are
    `modalBankSigDir`'s per-mode lambda split at the pair, written over the
    symbolic mode. Requires `deg == 0` uniformity (the `bankIsUniform` guard at
    dispatch), so the `powE` degree branches vanish, exactly as in
    `modalBankSigTable`. Same signature as `modalBankSigDir` — the dispatch in
    `modalBankTermDir` picks between them. -/
def modalBankSigDirTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (dir : Sig) (dampScale? : Option Sig := none) (live? : Option Sig := none) : BuildM Sig := do
  let clkRel ← relClockQ clkInt anchorSamples
  let clkFloat ← toFloatE clkRel
  let twoPow32 ← lit 4294967296
  let secondsTimesRate ← div clkFloat twoPow32
  let sr ← sampleRate
  let dSec ← div secondsTimesRate sr
  let cols ← bankCols modes live?
  let le ← bankLandExp modes                 -- option E: amp-only, shared fwd/rev (see modalBankSigDir)
  let landingScale ← le.scale
  let landingShift ← le.shift
  -- σ·d, optionally sway-bent (decay clock only, pitch untouched) — the same
  -- subterm both sides read, per symbolic mode.
  let sdOf := fun (m : ModeSym) => do
    let sd ← mul m.sigma dSec
    match dampScale? with | none => pure sd | some scale => mul sd scale
  let fwdQ ← bankFold cols fun m => do
    let increment ← toIntE m.incr
    let phQ ← modePhaseQFromIncr increment clkRel
    let sd ← sdOf m
    let negativeSd ← neg sd
    let envF ← expSig negativeSd
    let weightedCre ← mul envF m.cre
    let scaledCre ← mul weightedCre landingScale
    let wCreF ← toIntE scaledCre
    let weightedCim ← mul envF m.cim
    let scaledCim ← mul weightedCim landingScale
    let wCimF ← toIntE scaledCim
    let cosine ← fixedCosCycSig phQ
    let realPart ← mul wCreF cosine
    let sine ← fixedSinCycSig phQ
    let imagPart ← mul wCimF sine
    let difference ← sub realPart imagPart
    rshift difference landingShift
  let revQ ← bankFold cols fun m => do
    -- the mirrored phase spelled out on the NEGATED clock, as in the unrolled
    -- path (the fixed sine isn't bit-symmetric; see `modalBankSigDir`).
    let increment ← toIntE m.incr
    let negativeClock ← neg clkRel
    let phQN ← modePhaseQFromIncr increment negativeClock
    let sd ← sdOf m
    let envR ← expSig sd
    let weightedCre ← mul envR m.cre
    let scaledCre ← mul weightedCre landingScale
    let wCreR ← toIntE scaledCre
    let weightedCim ← mul envR m.cim
    let scaledCim ← mul weightedCim landingScale
    let wCimR ← toIntE scaledCim
    let cosine ← fixedCosCycSig phQN
    let realPart ← mul wCreR cosine
    let sine ← fixedSinCycSig phQN
    let imagPart ← mul wCimR sine
    let difference ← sub realPart imagPart
    rshift difference landingShift
  let zero ← lit 0
  let afterStrike ← gt clkRel zero
  let forwardOutput ← fixedOutQ 30 fwdQ
  let fwd ← selectE afterStrike forwardOutput zero
  let beforeStrike ← gt zero clkRel
  let reverseOutput ← fixedOutQ 30 revQ
  let rev ← selectE beforeStrike reverseOutput zero
  let one ← lit 1
  let forwardMix ← sub one dir
  let forwardTerm ← mul forwardMix fwd
  let reverseTerm ← mul dir rev
  add forwardTerm reverseTerm

/-- The direction bank as a term over the clock leaf, with the LOWERING (unrolled
    or banked) supplied explicitly — shared by `modalBankTermDir` (which picks by
    flag) and the `banks-as-data-dir` equivalence gate (which builds both sides
    regardless of the flag). -/
def modalBankTermDirWith
    (lower : Array ModalMode → Sig → Sig → Sig → Option Sig → BuildM Sig)
    (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (dir : Sig) (damp? : Option (Sig × Sig) := none) : ArrowTerm :=
  ArrowTerm.arrUn
    (fun clkSig => do
      -- the sway LFO is a pure function of the SAME clock leaf the bank rides, so a
      -- master scrub reverses it coherently with the tail. Its phase is integer-
      -- reduced (`phasorPhaseSig`), never `rate·t` on unbounded float seconds.
      let dampScale? ← match damp? with
        | none => pure none
        | some (depth, rate) => do
          let zero ← lit 0
          let phase ← phasorPhaseSig rate zero clkSig
          let twoPi ← twoPiE
          let radians ← mul twoPi phase
          let sine ← sinSig radians
          let modulation ← mul depth sine
          let one ← lit 1
          let scale ← add one modulation
          pure (some scale)
      lower modes clkSig anchor dir dampScale?)
    (ArrowTerm.clk c)

/-- `modalBankTerm` with a DIRECTION crossfade. Rides the `.clk` leaf like
    `modalBankTerm`, so master warps still reach it. Dispatches to the banked
    lowering for uniform banks under the banks flag, like `modalBankTerm`. -/
def modalBankTermDir (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (dir : Sig) (damp? : Option (Sig × Sig) := none)
    (count? : Option Sig := none) : ArrowTerm :=
  -- Same dispatch rule as `modalBankTerm`: a dynamic count ALWAYS banks (a
  -- runtime count cannot be unrolled; TROPICAL_BANKS_UNROLL governs only
  -- static banks); non-uniform banks unroll at capacity, dropping the count.
  let lower : Array ModalMode → Sig → Sig → Sig → Option Sig → BuildM Sig :=
    if bankIsUniform modes && (count?.isSome || banksTableEnabled)
    then fun ms clk a d s? => modalBankSigDirTable ms clk a d s? count?
    else fun ms clk a d s? => modalBankSigDir ms clk a d s?
  modalBankTermDirWith lower modes anchor c dir damp?

/-- A room kernel's orientation as data: the forward↔reverse crossfade `dir`
    (0 = forward kernel, 1 = reversed kernel) plus optional decay sway `(depth,
    rateHz)`. Each room owns its value; it is not complete-output reversal data. -/
structure ModalDir where
  dir : Sig
  damp : Option (Sig × Sig) := none
deriving BEq
