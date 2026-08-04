import Tropical.EmitArrow.Modal.BatchPlan

/-!
# EmitArrow.Modal.Batch

Non-public timed bloom-batch terminal spike.  It realizes independently timed
`BloomPair` banks through one nested branch×pair reduction while retaining the
existing Γ-bridge arithmetic and authored branch order.

This module is deliberately not connected to Patch lowering or the served
vocabulary.  It is an executable cost/correctness instrument for deciding
whether a fixed-capacity terminal can carry live bloom and a generalized room.
-/

namespace Tropical.EmitArrow

/-- One independently timed, already-checked bloom crossing.  Pair order is the
    checked composition's stable voice-major/room-minor order. -/
structure TimedBloomBank where
  pairs  : Array BloomPair
  anchor : Sig
deriving Inhabited

/-- Structural measurements of one packed terminal.  These are reported by the
    spike gates; they are not a served capacity claim. -/
structure TimedBloomBatchShape where
  branches         : Nat
  pairCapacity     : Nat
  totalPairs       : Nat
  maxSeriesDepth   : Nat
  maxCfDepth       : Nat
  coefficientCells : Nat
deriving Inhabited, Repr

private structure TimedBloomBatchCols where
  shape       : TimedBloomBatchShape
  branchCount : Sig
  anchor0     : Sig
  anchor1     : Sig
  anchor2     : Sig
  anchor3     : Sig
  muSigma     : Sig
  muOmega     : Sig
  nuSigma     : Sig
  nuOmega     : Sig
  bloomB      : Sig
  gRate       : Sig
  cRe         : Sig
  cIm         : Sig
  kappaRe     : Sig
  kappaIm     : Sig
  k1SerRe     : Sig
  k1SerIm     : Sig
  k1CfRe      : Sig
  k1CfIm      : Sig
  fSerRe      : Sig
  fSerIm      : Sig
  dSwitch     : Sig
  hasCf       : Sig
  invARe      : Sig
  invAIm      : Sig
  cfBRe       : Sig
  cfBIm       : Sig
  cfNRe       : Sig
  cfNIm       : Sig

private def TimedBloomBatchCols.tables (c : TimedBloomBatchCols) : Array Sig :=
  #[c.branchCount, c.anchor0, c.anchor1, c.anchor2, c.anchor3,
    c.muSigma, c.muOmega, c.nuSigma, c.nuOmega, c.bloomB, c.gRate,
    c.cRe, c.cIm, c.kappaRe, c.kappaIm,
    c.k1SerRe, c.k1SerIm, c.k1CfRe, c.k1CfIm,
    c.fSerRe, c.fSerIm, c.dSwitch, c.hasCf,
    c.invARe, c.invAIm, c.cfBRe, c.cfBIm, c.cfNRe, c.cfNIm]

private def padRows (banks : Array TimedBloomBank) (capacity : Nat) : Array BloomPair :=
  banks.foldl (init := #[]) fun out b =>
    out ++ b.pairs ++ Array.replicate (capacity - b.pairs.size) default

private def flatCoefficients (rows : Array BloomPair) (depth : Nat)
    (get : BloomPair → Array CplxE) (pad : CplxE) : Array CplxE :=
  rows.foldl (init := #[]) fun out p =>
    out ++ (Array.range depth).map fun j => (get p)[j]?.getD pad

private def packTimedBloomBatch (banks : Array TimedBloomBank) : TimedBloomBatchCols :=
  let pairCapacity := banks.foldl (fun n b => max n b.pairs.size) 0
  let rows := padRows banks pairCapacity
  let maxSeriesDepth := rows.foldl (fun n p => max n p.invA.size) 0
  let maxCfDepth := rows.foldl (fun n p => max n p.cfN.size) 0
  let totalPairs := banks.foldl (fun n b => n + b.pairs.size) 0
  let coefficientCells := rows.size * (2 * maxSeriesDepth + 4 * maxCfDepth + 2)
  let shape : TimedBloomBatchShape :=
    { branches := banks.size, pairCapacity, totalPairs,
      maxSeriesDepth, maxCfDepth, coefficientCells }
  let col (f : BloomPair → Sig) := Sig.arr (rows.map f)
  let invA := flatCoefficients rows maxSeriesDepth (·.invA) (lit 0, lit 0)
  -- Extending a CF with b=1/n=0 levels is identity padding: at the first
  -- padded numerator the bottom-up recurrence resets to the original terminal
  -- `z + b_k`, after which the original levels run unchanged.
  let cfB := flatCoefficients rows (maxCfDepth + 1) (·.cfB) cOneE
  let cfN := flatCoefficients rows maxCfDepth (·.cfN) (lit 0, lit 0)
  -- Coefficient arrays cross Metal as f32.  A Q32.32 anchor does not fit in one
  -- f32 cell, but each unsigned 16-bit limb does exactly.  Split the s0 anchor
  -- after landing it to i64 and reconstruct the identical two's-complement bit
  -- pattern in the audio body (negative anchors included).
  let anchorQ := banks.map fun b => toIntE (mul b.anchor (lit 4294967296))
  let anchorLimb (shift : Nat) := Sig.arr (anchorQ.map fun q =>
    toFloatE (bitAnd (if shift == 0 then q else rshift q (litI shift)) (litI 65535)))
  { shape
    branchCount := Sig.arr (banks.map fun b => litF b.pairs.size.toFloat)
    anchor0 := anchorLimb 0, anchor1 := anchorLimb 16
    anchor2 := anchorLimb 32, anchor3 := anchorLimb 48
    muSigma := col (·.muSigma), muOmega := col (·.muOmega)
    nuSigma := col (·.nuSigma), nuOmega := col (·.nuOmega)
    bloomB := col (fun p => litF p.bloomB), gRate := col (fun p => litF p.gRate)
    cRe := col (·.c.1), cIm := col (·.c.2)
    kappaRe := col (·.kappa.1), kappaIm := col (·.kappa.2)
    k1SerRe := col (·.k1Ser.1), k1SerIm := col (·.k1Ser.2)
    k1CfRe := col (·.k1Cf.1), k1CfIm := col (·.k1Cf.2)
    fSerRe := col (·.fSer.1), fSerIm := col (·.fSer.2)
    dSwitch := col (·.dSwitch)
    hasCf := col (fun p => if p.cfN.isEmpty then lit 0 else lit 1)
    invARe := Sig.arr (invA.map (·.1)), invAIm := Sig.arr (invA.map (·.2))
    cfBRe := Sig.arr (cfB.map (·.1)), cfBIm := Sig.arr (cfB.map (·.2))
    cfNRe := Sig.arr (cfN.map (·.1)), cfNIm := Sig.arr (cfN.map (·.2)) }

/-- Pack-shape inspection without lowering a program. -/
def timedBloomBatchShape (banks : Array TimedBloomBank) : TimedBloomBatchShape :=
  (packTimedBloomBatch banks).shape

private def batchPairBody (c : TimedBloomBatchCols) (clkRel dPos : Sig)
    (row : Sig) : Sig :=
  let read (a : Sig) := Sig.index a row
  let coeffAt (re im : Sig) (depth j : Nat) : CplxE :=
    let idx := add (mul row (litI depth)) (litI j)
    (Sig.index re idx, Sig.index im idx)
  let muSigma := read c.muSigma
  let muOmega := read c.muOmega
  let nuSigma := read c.nuSigma
  let nuOmega := read c.nuOmega
  let B := read c.bloomB
  let g := read c.gRate
  let pairC : CplxE := (read c.cRe, read c.cIm)
  let kappa : CplxE := (read c.kappaRe, read c.kappaIm)
  let k1Ser : CplxE := (read c.k1SerRe, read c.k1SerIm)
  let k1Cf : CplxE := (read c.k1CfRe, read c.k1CfIm)
  let fSer : CplxE := (read c.fSerRe, read c.fSerIm)
  let invA := (Array.range c.shape.maxSeriesDepth).map fun j =>
    coeffAt c.invARe c.invAIm c.shape.maxSeriesDepth j
  let cfB := (Array.range (c.shape.maxCfDepth + 1)).map fun j =>
    coeffAt c.cfBRe c.cfBIm (c.shape.maxCfDepth + 1) j
  let cfN := (Array.range c.shape.maxCfDepth).map fun j =>
    coeffAt c.cfNRe c.cfNIm c.shape.maxCfDepth j
  let eg := expSig (neg (mul g dPos))
  let z : CplxE := (mul kappa.1 eg, mul kappa.2 eg)
  let off := mul B (sub (lit 1) eg)
  let clkW := add clkRel (toIntE (mul (mul off .sampleRate) (lit 4294967296)))
  let phNu := modePhaseQ nuOmega clkRel
  let phMu := modePhaseQ muOmega clkW
  let envNu := expSig (neg (mul nuSigma dPos))
  let envMu := expSig (neg (mul muSigma (add dPos off)))
  let mser := bloomM1E invA z
  let k2ser := cmulE mser fSer
  let cf := bloomCFE cfB cfN z
  let k2cf : CplxE := (div cf.1 g, div cf.2 g)
  let onSer := gt dPos (read c.dSwitch)
  let hasCf := gt (read c.hasCf) (lit 0)
  let k1Cross : CplxE :=
    (selectE onSer k1Ser.1 k1Cf.1, selectE onSer k1Ser.2 k1Cf.2)
  let k2Cross : CplxE :=
    (selectE onSer k2ser.1 k2cf.1, selectE onSer k2ser.2 k2cf.2)
  let k1 : CplxE :=
    (selectE hasCf k1Cross.1 k1Ser.1, selectE hasCf k1Cross.2 k1Ser.2)
  let k2 : CplxE :=
    (selectE hasCf k2Cross.1 k2ser.1, selectE hasCf k2Cross.2 k2ser.2)
  let w1 := cmulE pairC k1
  let w2 := cmulE pairC k2
  let land := fun (env : Sig) (w : CplxE) (ph : Sig) =>
    rshift (sub (mul (toIntE (mul (mul env w.1) (lit 268435456))) (fixedCosCycSig ph))
                (mul (toIntE (mul (mul env w.2) (lit 268435456))) (fixedSinCycSig ph)))
           (lit 28)
  add (land envNu w1 phNu) (land envMu w2 phMu)

/-- Admission for the first executable slice.  Keeping this separate from the
    renderer prevents a term wrapper from turning a refusal into silence. -/
private def validateTimedBloomBanks (banks : Array TimedBloomBank) : Except String Unit := do
  let pairCapacity := banks.foldl (fun n b => max n b.pairs.size) 0
  -- The dynamic trip count crosses the coefficient boundary as f32.  Integers
  -- through 2^24 are exact there; larger counts need their own limb transport.
  for bi in [0:banks.size] do
    let some bank := banks[bi]? | continue
    if bank.pairs.size > 16777216 then
      throw s!"timed bloom batch: branch[{bi}] pair count {bank.pairs.size} exceeds the exact f32 integer range (2^24)"
  -- FlatRuntime and the Metal column ABI use uint32 lengths.  Refuse before
  -- constructing padded rows or flattened recurrence columns.
  let maxColumnLength : Nat := 4294967295
  if pairCapacity > 0 && banks.size > maxColumnLength / pairCapacity then
    throw "timed bloom batch: padded branch×pair row count exceeds the uint32 coefficient-column ABI"
  let rows := banks.size * pairCapacity
  let maxSeriesDepth := banks.foldl (fun n b =>
    b.pairs.foldl (fun m p => max m p.invA.size) n) 0
  let maxCfWidth := banks.foldl (fun n b =>
    b.pairs.foldl (fun m p => max m p.cfB.size) n) 0
  let maxCfDepth := banks.foldl (fun n b =>
    b.pairs.foldl (fun m p => max m p.cfN.size) n) 0
  if maxSeriesDepth > 0 && rows > maxColumnLength / maxSeriesDepth then
    throw "timed bloom batch: flattened series column exceeds the uint32 coefficient-column ABI"
  if maxCfWidth > 0 && rows > maxColumnLength / maxCfWidth then
    throw "timed bloom batch: flattened continued-fraction column exceeds the uint32 coefficient-column ABI"
  -- Metal receives all hoisted columns as one packed f32 buffer, whose total
  -- float count is narrowed to uint32 by the runtime. Bound the aggregate as
  -- well as every constituent array. The 18 row columns and five per-branch
  -- count/anchor columns are easy to miss when reasoning only from recurrence
  -- storage.
  let packedCells := 5 * banks.size + 18 * rows
    + 2 * rows * maxSeriesDepth
    + 2 * rows * maxCfWidth
    + 2 * rows * maxCfDepth
  if packedCells > maxColumnLength then
    throw "timed bloom batch: aggregate packed coefficient columns exceed the uint32 Metal ABI"
  for bi in [0:banks.size] do
    let some bank := banks[bi]? | continue
    if !sigIsS0 bank.anchor then
      throw s!"timed bloom batch: branch[{bi}] anchor is not s0; branch-local audio-rate clocks require a separate terminal group"
    for pi in [0:bank.pairs.size] do
      let some pair := bank.pairs[pi]? | continue
      if pair.coincident then
        throw s!"timed bloom batch: branch[{bi}]×pair[{pi}] is coincident; the spike admits only the default gong's non-coincident profile"

private def bloomTimedBatchSigUnchecked (banks : Array TimedBloomBank) (clkInt : Sig) : Sig := Id.run do
  if banks.isEmpty then return lit 0
  let c := packTimedBloomBatch banks
  if c.shape.pairCapacity == 0 then return lit 0
  let branch := Sig.loopIdx 0
  let pair := Sig.loopIdx 1
  let limb (a : Sig) := toIntE (Sig.index a branch)
  let l0 := limb c.anchor0
  let l1 := limb c.anchor1
  let l2 := limb c.anchor2
  let l3 := limb c.anchor3
  -- Reconstruct the signed top limb arithmetically.  This covers the full i64
  -- domain, including -2^63, without depending on signed `<< 48` behavior in
  -- an emitted backend.
  let low48 := bitOr l0 (bitOr (lshift l1 (litI 16)) (lshift l2 (litI 32)))
  let highSigned := sub l3 (selectE (gt l3 (litI 32767)) (litI 65536) (litI 0))
  let anchorQ := add low48 (mul highSigned (litI 281474976710656))
  let clkRel := sub clkInt anchorQ
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let dPos := clampE dSec (lit 0) (lit 1000000)
  let row := add (mul branch (litI c.shape.pairCapacity)) pair
  let pairQ := Sig.bankSum c.shape.pairCapacity #[]
    (selectE (gt clkRel (lit 0)) (batchPairBody c clkRel dPos row) (litI 0))
    (some (toIntE (Sig.index c.branchCount branch))) 1
  let branchOut := fixedOutQ 30 pairQ
  return Sig.bankSum c.shape.branches c.tables branchOut none 0

/-- The executable terminal spike.  Branches reduce in authored order; each
    branch's pair bank reduces in the checked composition's order and converts
    from Q only at that branch boundary, matching the existing forest signal
    sum.  Coincident pairs remain out of scope for this first fixed-shape probe
    and are refused with exact branch/pair indices. -/
def bloomTimedBatchSig (banks : Array TimedBloomBank) (clkInt : Sig) : Except String Sig := do
  validateTimedBloomBanks banks
  return bloomTimedBatchSigUnchecked banks clkInt

/-- Term wrapper for the terminal spike. -/
def bloomTimedBatchTerm (banks : Array TimedBloomBank) (c : Clock) : Except String ArrowTerm := do
  validateTimedBloomBanks banks
  return ArrowTerm.arrUn (fun clk => bloomTimedBatchSigUnchecked banks clk) (ArrowTerm.clk c)

end Tropical.EmitArrow
