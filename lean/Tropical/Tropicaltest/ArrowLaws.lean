import Tropical.Tropicaltest.Reversibility

/-!
# Tropical.Tropicaltest.ArrowLaws

The warp arrow laws as audio goldens (algebraically-equal warps ⇒ bit-identical audio), the EmitArrow corpus gate (EmitArrow's emit ≡ strata's emit, per program), and the cartesian-diagonal / reverse / fixed-point-value laws.
-/

open Tropical
open Tropical.Plan

-- ── (h) EmitArrow arrow laws: algebraically-equal warps ⇒ bit-identical audio ─
-- Slice 3. The warp arrow laws are certified as AUDIO goldens: build the law's
-- LHS and RHS as two EmitArrow carrier programs (a single FixedSinOsc clocked at
-- two algebraically-equal clock expressions), render both, and assert the
-- rendered audio is byte-identical (SHA256). The laws hold byte-exactly because
-- warps are integer add/sub on the Q32.32 fixed-point clock — exact and
-- associative — so the two sides feed the oscillator a bit-identical int64 clock
-- and render bit-identical audio, EVEN THOUGH the emitted plans differ (no
-- algebraic tree normalization). The render bridge reuses the production session
-- path: buildClockCarrier (EmitArrow) → Strata.run → Core.check → compileSession
-- (the carrier as a one-instance root wired to the dac) → FlatPlan → renderIrBytes.

open Tropical.Ir (Arena ProgramIdx)

/-- Elaborate the whole stdlib bridge chain in manifest order (the shared
    `StdlibChain` driver), so `FixedSinOsc` (and its transitive voice deps)
    are in the arena for `buildClockCarrier` to link against. Done once and
    reused. -/
def arrowElabStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) :=
  Tropical.StdlibChain.elabStdlib

/-- Finish a built carrier program into a runnable `FlatPlan` via the
    production session path: full strata, then the carrier as the synthetic
    session root wired straight to the dac at its `out` port
    (`__root__.out`). No session wires, no params, no inputs. -/
private def finishCarrier (arena : Arena) (idx : ProgramIdx) :
    Except String Tropical.Plan.FlatPlan := do
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved { upto := 5 } arena idx).mapError (·.message)
  Tropical.Compile.compileSession (.forRoot core coreArena)

def buildAndFinish (built : Except String (Arena × ProgramIdx)) :
    Except String Tropical.Plan.FlatPlan := do
  let (a, i) ← built
  finishCarrier a i

/-- Build one EmitArrow clock carrier (named `name`, clocked at `clkE`) into a
    runnable `FlatPlan` via the production session path. -/
def compileArrowCarrier (arena : Arena) (resolved : Array (String × ProgramIdx))
    (name : String) (clkE : Tropical.EmitArrow.Clock) :
    Except String Tropical.Plan.FlatPlan :=
  buildAndFinish (Tropical.EmitArrow.buildClockCarrier name clkE arena resolved)

-- ── (h′) EmitArrow corpus gate: EmitArrow's emit ≡ strata's emit, per program ─
-- The cutover (phase C1) reproduces the corpus one program at a time. This is
-- the reusable instrument: given a resolved program (built by an EmitArrow
-- constructor) and a stdlib program NAME, emit BOTH through the production
-- per-program recipe — strata (full) → Core.check → compileResolved → wire —
-- and compare byte-for-byte. The stdlib side is exactly what `diffcli
-- emit-stdlib <Name>` produces (the target strata produces TODAY); the
-- EmitArrow side is what the cutover would emit instead. Byte-identity proves
-- EmitArrow covers strata's job for that program — the slices-1/2 byte-gate,
-- generalized to any program, run as a tropicaltest assertion rather than an
-- external `diff` of two diffcli verbs.

/-- The production per-program emit recipe (the `diffcli emit-*` body): strata
    (all ported passes, inline) → Core.check → compileResolved → wire JSON. The
    canonical `tropical_plan_5`-per-instance bytes a program emits today. -/
def emitResolvedWire (arena : Arena) (idx : ProgramIdx) : Except String String := do
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved
      { upto := Tropical.Ir.Strata.portedPasses, inlineNested := true } arena idx).mapError (·.message)
  let plan ← Tropical.Ir.CompileResolved.compileResolved core coreArena
  let wire ← plan.toWire
  pure wire.compress

/-- THE CORPUS GATE (reusable). Build a program with `builder` (an EmitArrow
    constructor over the elaborated stdlib arena), then assert its emit is
    byte-identical to stdlib program `stdName`'s emit — the form strata produces
    today (`diffcli emit-stdlib {stdName}`). `arena`/`resolved` come from
    `arrowElabStdlib`; `stdName` is emitted from the ORIGINAL arena (the builder
    only appends), so the comparison is EmitArrow-emit vs strata-emit of the same
    target. This covers the corpus one program at a time. -/
def runEmitCorpusGate (label stdName : String)
    (arena : Arena) (resolved : Array (String × ProgramIdx))
    (builder : Arena → Array (String × ProgramIdx) → Except String (Arena × ProgramIdx)) :
    IO Bool := do
  let some (_, stdIdx) := resolved.find? (·.1 == stdName)
    | IO.println s!"  FAIL  corpus-gate/{label}  stdlib '{stdName}' not in elaborated chain"
      pure false
  match builder arena resolved with
  | .error e => failGate s!"corpus-gate/{label}" s!"build: {firstLine e}"
  | .ok (arena', idx) =>
    match emitResolvedWire arena' idx, emitResolvedWire arena stdIdx with
    | .error e, _ => failGate s!"corpus-gate/{label}" s!"emit EmitArrow: {firstLine e}"
    | _, .error e => failGate s!"corpus-gate/{label}" s!"emit stdlib {stdName}: {firstLine e}"
    | .ok got, .ok want =>
      if got == want then
        passGate s!"corpus-gate/{label}" s!"EmitArrow ≡ emit-stdlib {stdName} ({got.length}B)"
      else
        failGate s!"corpus-gate/{label}" s!"EmitArrow ≠ emit-stdlib {stdName} (EmitArrow {got.length}B, stdlib {want.length}B)"

/-- Certify one warp law: build LHS and RHS carriers, render both, assert the
    rendered audio is byte-identical (SHA256). Also reports whether the emitted
    plans are byte-identical (EXPECTED NO — the algebra is exact in the clock,
    not normalized in the tree). PASS iff the audio hashes match. -/
def runArrowLaw (name : String)
    (lhsClk rhsClk : Tropical.EmitArrow.Clock)
    (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match compileArrowCarrier arena resolved s!"{name}_lhs" lhsClk,
        compileArrowCarrier arena resolved s!"{name}_rhs" rhsClk with
  | .error e, _ => failGate s!"arrow-law/{name}" s!"build lhs: {firstLine e}"
  | _, .error e => failGate s!"arrow-law/{name}" s!"build rhs: {firstLine e}"
  | .ok lhsPlan, .ok rhsPlan =>
    match lhsPlan.toWire, rhsPlan.toWire with
    | .error e, _ | _, .error e =>
      failGate s!"arrow-law/{name}" s!"toWire: {firstLine e}"
    | .ok lhsWire, .ok rhsWire =>
      let plansIdentical := lhsWire.compress == rhsWire.compress
      match ← renderIrBytes lhsPlan, ← renderIrBytes rhsPlan with
      | .error e, _ | _, .error e =>
        failGate s!"arrow-law/{name}" s!"render: {firstLine e}"
      | .ok lhsBytes, .ok rhsBytes =>
        let planNote := if plansIdentical then "plans identical" else "plans differ (expected)"
        hashGate s!"arrow-law/{name}" lhsBytes rhsBytes
          (fun h => s!"audio ≡ {h.take 16} ({planNote})")
          (fun hl hr => s!"audio differs: lhs {hl.take 16} rhs {hr.take 16} ({planNote})")

-- ── (h) slice 4 — the cartesian diagonal / fan-out law + its COST story ───────
-- The diagonal `Δ = id &&& id`: one source fanned into two differently-warped
-- flangers ≡ two independent (source+flanger) pairs. AUDIO: byte-identical (the
-- duplicated source is the same closed form fed the same clock, so sharing is
-- invisible to the output). COST (the real content): whether within-program
-- strata CSE collapses the duplicated osc(clk) so both forms reach the SAME
-- minimal DAG — i.e. "fan-out is a pure let". We report eval/register/instruction
-- counts for both forms and whether the emitted plans are byte-identical (the
-- program name does NOT flow into the plan — the root instance is `__root__` —
-- so plan byte-identity is a pure structural test of the CSE collapse).

/-- Count instructions across an InstanceFunction tree (body + nested children). -/
private partial def countFnInstrs (f : Tropical.Plan.InstanceFunction) : Nat :=
  f.instructions.size + f.children.foldl (fun acc c => acc + countFnInstrs c) 0

/-- Total instruction count of a FlatPlan (all instance functions, recursive). -/
def planInstrCount (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countFnInstrs f) 0

/-- Count instructions with a given tag across an InstanceFunction tree. -/
private partial def countFnTagged (t : String) (f : Tropical.Plan.InstanceFunction) : Nat :=
  (f.instructions.filter (·.tag == t)).size
    + f.children.foldl (fun acc c => acc + countFnTagged t c) 0

/-- Count instructions with a given tag across a FlatPlan (e.g. "ReduceBegin"
    to count reduce regions, "Pack" to count materialized columns). -/
def planTagCount (t : String) (p : Tropical.Plan.FlatPlan) : Nat :=
  p.instanceFunctions.foldl (fun acc f => acc + countFnTagged t f) 0

/-- Strata-process an already-built carrier program, then compile it via the
    production session path. Returns (post-strata binderCount, post-strata
    declCount, FlatPlan) — the post-strata program is the CSE'd DAG, so its
    binderCount is the count of distinct shared subexpressions. -/
private def diagStrataCompile (arena : Arena) (idx : ProgramIdx) :
    Except String (Nat × Nat × Tropical.Plan.FlatPlan) := do
  let (arena'', root') ← (Tropical.Ir.Strata.run { upto := 5 } arena idx).mapError (·.message)
  let some prog := arena''.program? root'
    | .error "diagonal: post-strata root program index out of range"
  let (coreArena, core) ← Tropical.Ir.checkResolvedArena arena'' root'
  let plan ← Tropical.Compile.compileSession (.forRoot core coreArena)
  pure (prog.binderCount, prog.decls.size, plan)

/-- Certify the diagonal law: build SHARED (one fanned source) and INDEPENDENT
    (two sources) carriers, assert their rendered audio is byte-identical (the
    law), and report the cost story — per-form eval/register/instruction counts
    and whether the emitted plans are byte-identical (CSE collapse). PASS iff the
    audio hashes match. -/
def runDiagonalLaw (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSharedDiagonal arena resolved,
        Tropical.EmitArrow.buildIndependentDiagonal arena resolved with
  | .error e, _ => failGate "arrow-law/diagonal" s!"build shared: {firstLine e}"
  | _, .error e => failGate "arrow-law/diagonal" s!"build independent: {firstLine e}"
  | .ok (arenaS, idxS), .ok (arenaI, idxI) =>
    -- Pre-strata instance counts: the visible structural difference (5 vs 6).
    let preShared := (arenaS.program? idxS).map (·.decls.size) |>.getD 0
    let preIndep := (arenaI.program? idxI).map (·.decls.size) |>.getD 0
    match diagStrataCompile arenaS idxS, diagStrataCompile arenaI idxI with
    | .error e, _ => failGate "arrow-law/diagonal" s!"compile shared: {firstLine e}"
    | _, .error e => failGate "arrow-law/diagonal" s!"compile independent: {firstLine e}"
    | .ok (bcS, dcS, planS), .ok (bcI, dcI, planI) =>
      let instrS := planInstrCount planS
      let instrI := planInstrCount planI
      match planS.toWire, planI.toWire with
      | .error e, _ | _, .error e =>
        failGate "arrow-law/diagonal" s!"toWire: {firstLine e}"
      | .ok wireS, .ok wireI =>
        let plansIdentical := wireS.compress == wireI.compress
        match ← renderIrBytes planS, ← renderIrBytes planI with
        | .error e, _ | _, .error e =>
          failGate "arrow-law/diagonal" s!"render: {firstLine e}"
        | .ok bytesS, .ok bytesI =>
          -- The plans being byte-identical IS the definitive collapse (the
          -- carrier program name never reaches the plan — the root instance is
          -- `__root__`). NB the post-strata binder DAGs DIFFER (independent
          -- carries the extra inlined osc(clk) body's binders); the duplicate
          -- source is deduped at EMIT (compileResolved value-numbering), not in
          -- the strata binder DAG — yet both reach the same minimal kernel.
          IO.println s!"        cost  shared:      pre-strata insts={preShared} post-strata binders={bcS} decls={dcS} plan-instrs={instrS} regs={planS.registerCount}"
          IO.println s!"        cost  independent: pre-strata insts={preIndep} post-strata binders={bcI} decls={dcI} plan-instrs={instrI} regs={planI.registerCount}"
          IO.println s!"        cost  plans byte-identical: {plansIdentical}  ·  CSE collapsed both to same {instrS}-instr/{planS.registerCount}-reg DAG: {plansIdentical && instrS == instrI}  (post-strata binders {bcS} vs {bcI} differ — dedup is at emit)"
          hashGate "arrow-law/diagonal" bytesS bytesI
            (fun h => s!"audio ≡ {h.take 16} (shared ≡ independent)")
            (fun hs hi => s!"audio differs: shared {hs.take 16} independent {hi.take 16}")

-- ── (h) slice 5 — REVERSE (the moat) as warp(neg): involution, reverse-swaps- ──
-- delay, and reverse-equivariance of the symmetric flanger. Laws 1-2 reuse
-- `runArrowLaw` (byte-identical: the clock side is exact int64). Law 3 cannot be
-- byte-identical in float — under reverse the ±δ taps swap tree slot, so the
-- value weighted-sum reassociates `(A+B)+C` vs `(A+C)+B`, and float add is not
-- associative — so we assert the DENOTATIONAL form (max|Δ| < ε) and REPORT the
-- byte-identity + max|Δ| as the finding. It would be byte-exact with a fixed-
-- point value carrier (the clock side, laws 1-2, is already exact).

/-- Certify reverse-equivariance of the symmetric flanger:
    `warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)`. Builds both flanger carriers
    (neg OUTER vs INNER, swapping the ±δ tap slots), renders both, asserts the
    DENOTATIONAL law (max|Δ| < ε), and reports byte-identity + max|Δ|. PASS iff
    max|Δ| < ε and the signal is non-silent. -/
def runReverseFlangerCommute
    (arena : Arena) (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let eps := 1e-6
  match Tropical.EmitArrow.buildReverseThenFlanger arena resolved,
        Tropical.EmitArrow.buildFlangerThenReverse arena resolved with
  | .error e, _ => failGate "arrow-law/reverse-flanger-commute" s!"build lhs: {firstLine e}"
  | _, .error e => failGate "arrow-law/reverse-flanger-commute" s!"build rhs: {firstLine e}"
  | .ok (arenaL, idxL), .ok (arenaR, idxR) =>
    match diagStrataCompile arenaL idxL, diagStrataCompile arenaR idxR with
    | .error e, _ => failGate "arrow-law/reverse-flanger-commute" s!"compile lhs: {firstLine e}"
    | _, .error e => failGate "arrow-law/reverse-flanger-commute" s!"compile rhs: {firstLine e}"
    | .ok (_, _, planL), .ok (_, _, planR) =>
      match ← renderIrBytes planL, ← renderIrBytes planR with
      | .error e, _ | _, .error e =>
        failGate "arrow-law/reverse-flanger-commute" s!"render: {firstLine e}"
      | .ok bytesL, .ok bytesR =>
        let hashL ← sha256Hex bytesL
        let hashR ← sha256Hex bytesR
        let byteIdentical := hashL == hashR
        let samplesL := decodeF64LE bytesL
        let samplesR := decodeF64LE bytesR
        let n := min samplesL.size samplesR.size
        let mut maxAbs := 0.0
        let mut energy := 0.0
        let mut bitDiff := 0
        for k in [0:n] do
          let d := (samplesL[k]! - samplesR[k]!).abs
          if d > maxAbs then maxAbs := d
          if samplesL[k]!.toBits != samplesR[k]!.toBits then bitDiff := bitDiff + 1
          energy := energy + samplesL[k]! * samplesL[k]!
        -- max|Δ| is sub-ULP-scale; Float.toString rounds it to 0.000000. Show the
        -- scaled value (×10¹⁵, i.e. femto-units) so the ULP magnitude is legible.
        IO.println s!"        finding  byte-identical: {byteIdentical}  ·  bit-differing samples: {bitDiff}/{n}  ·  max|Δ|={maxAbs} (={maxAbs * 1e15}e-15, ε={eps})  ·  lhs energy={energy}"
        IO.println s!"        finding  ±δ taps swap slot ⇒ float value-sum reassociates ((A+B)+C vs (A+C)+B); exact on the (fixed-point) clock (laws 1-2), float-tolerance on the (float) value sum — byte-exact with a fixed-point value carrier"
        if energy <= 1e-6 then
          failGate "arrow-law/reverse-flanger-commute" s!"signal silent (energy={energy})"
        else if maxAbs < eps then
          passGate "arrow-law/reverse-flanger-commute" s!"denotational ≡ (max|Δ|={maxAbs} < ε; byte-identical={byteIdentical})"
        else
          failGate "arrow-law/reverse-flanger-commute" s!"max|Δ|={maxAbs} ≥ ε={eps}"

-- ── (h) slice 6 — a FIXED-POINT VALUE carrier: slice-5's reassociation ────────
-- failure becomes BYTE-IDENTICAL. The SAME warp combinators, but instantiated at
-- an integer Q0.32 saw source (`fixedPhase`) mixed with INTEGER right-shifts
-- (`fixedFlangerSum`) instead of a float oscillator + float weighted-sum. Integer
-- add is associative AND commutative, so the past/ahead slot swap that reverse
-- induces (the float reassociation `(A+B)+C ≠ (A+C)+B`, slice 5's 1271/4096) is
-- invisible — the law is now byte-exact. No type-system / engine-float change;
-- the carrier is integer `Expr` with one `toFloat` scale at the DAC boundary.

/-- Certify one single-source fixed-point warp law: build LHS and RHS fixed-point
    source carriers (`fixedOut(fixedPhase(clkE))`) at two algebraically-equal
    clocks, render both, assert the audio is BYTE-IDENTICAL (SHA256). The clock
    side is exact int64, so these hold byte-exactly — the fixed-point analog of
    `runArrowLaw`'s float laws (which also pass) over the integer source. -/
def runFixedSourceLaw (name : String)
    (lhsClk rhsClk : Tropical.EmitArrow.Clock) (arena : Arena) : IO Bool := do
  let (arenaL, idxL) := Tropical.EmitArrow.buildFixedSourceCarrier s!"{name}_lhs" lhsClk arena
  let (arenaR, idxR) := Tropical.EmitArrow.buildFixedSourceCarrier s!"{name}_rhs" rhsClk arena
  match diagStrataCompile arenaL idxL, diagStrataCompile arenaR idxR with
  | .error e, _ => failGate s!"arrow-law/{name}" s!"compile lhs: {firstLine e}"
  | _, .error e => failGate s!"arrow-law/{name}" s!"compile rhs: {firstLine e}"
  | .ok (_, _, planL), .ok (_, _, planR) =>
    match ← renderIrBytes planL, ← renderIrBytes planR with
    | .error e, _ | _, .error e =>
      failGate s!"arrow-law/{name}" s!"render: {firstLine e}"
    | .ok bytesL, .ok bytesR =>
      hashGate s!"arrow-law/{name}" bytesL bytesR
        (fun h => s!"audio ≡ {h.take 16} (byte-identical)")
        (fun hl hr => s!"audio differs: lhs {hl.take 16} rhs {hr.take 16}")

/-- THE GATE: certify reverse-equivariance of the FIXED-POINT flanger
    `warp(neg) ⋙ flanger ≡ flanger ⋙ warp(neg)` — the exact law slice 5 found
    byte-DIFFERENT in float (1271/4096). Builds both carriers (neg OUTER vs INNER,
    swapping the ±δ tap slots) over the INTEGER source + integer shift-add mix,
    renders both, and asserts the audio is BYTE-IDENTICAL (SHA256). Reports the
    differing-sample count (must be 0/N vs slice 5's 1271/4096). PASS iff
    byte-identical and non-silent. -/
def runReverseFlangerCommuteFixedpoint (arena : Arena) : IO Bool := do
  let (arenaL, idxL) := Tropical.EmitArrow.buildReverseThenFixedFlanger arena
  let (arenaR, idxR) := Tropical.EmitArrow.buildFixedFlangerThenReverse arena
  match diagStrataCompile arenaL idxL, diagStrataCompile arenaR idxR with
  | .error e, _ => failGate "arrow-law/reverse-flanger-commute-fixedpoint" s!"compile lhs: {firstLine e}"
  | _, .error e => failGate "arrow-law/reverse-flanger-commute-fixedpoint" s!"compile rhs: {firstLine e}"
  | .ok (_, _, planL), .ok (_, _, planR) =>
    match ← renderIrBytes planL, ← renderIrBytes planR with
    | .error e, _ | _, .error e =>
      failGate "arrow-law/reverse-flanger-commute-fixedpoint" s!"render: {firstLine e}"
    | .ok bytesL, .ok bytesR =>
      let hashL ← sha256Hex bytesL
      let hashR ← sha256Hex bytesR
      let byteIdentical := hashL == hashR
      let samplesL := decodeF64LE bytesL
      let samplesR := decodeF64LE bytesR
      let n := min samplesL.size samplesR.size
      let mut bitDiff := 0
      let mut energy := 0.0
      for k in [0:n] do
        if samplesL[k]!.toBits != samplesR[k]!.toBits then bitDiff := bitDiff + 1
        energy := energy + samplesL[k]! * samplesL[k]!
      IO.println s!"        gate  byte-identical: {byteIdentical}  ·  bit-differing samples: {bitDiff}/{n} (slice-5 float was 1271/4096)  ·  lhs energy={energy}"
      IO.println s!"        gate  integer add is associative — ±δ tap slot swap leaves the Q0.32 mix bit-identical; one toFloat·/2³² scale at the boundary"
      if energy <= 1e-6 then
        failGate "arrow-law/reverse-flanger-commute-fixedpoint" s!"signal silent (energy={energy})"
      else if byteIdentical then
        passGate "arrow-law/reverse-flanger-commute-fixedpoint" s!"audio ≡ {hashL.take 16} ({bitDiff}/{n} differing — byte-exact)"
      else
        failGate "arrow-law/reverse-flanger-commute-fixedpoint" s!"audio differs ({bitDiff}/{n}): lhs {hashL.take 16} rhs {hashR.take 16}"
