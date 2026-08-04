import Tropical.Tropicaltest.Modal

/-!
# Tropical.Tropicaltest.TimedBloomMetal

Focused backend probe for the non-public timed bloom-batch terminal. This
fixture never enters Patch lowering or the served vocabulary: it builds one of
the measured moved-radial-seam pairs directly and places it at a far negative
fractional Q32.32 anchor. Ordinary validation proves the typed split and MSL
column crossing structurally. `TROPICAL_TIMED_BLOOM_METAL=1` opts into the real
JIT↔Metal render; callers must impose an external timeout because the first
host-GPU qualification attempt exceeded five minutes in Metal compile/render.
-/

open Tropical
open Tropical.Ir (Arena ProgramIdx)

namespace Tropical.Tropicaltest.TimedBloomMetal

open Tropical.EmitArrow

private def gateName : String := "timed-bloom-batch-backend-shape"

private def farAnchorNumber : Lean.JsonNumber := ⟨-150000000075, 2⟩
private def farAnchor : Float := -1500000000.75
private def farAnchorQ : Int := -6442450947221225472

private structure ParityMetrics where
  samples : Nat
  finite : Bool
  signalEnergy : Float
  errorEnergy : Float
  relativeL2 : Float
  snrDb : Float

private structure MetalRender where
  samples : Array Float
  loadNanos : Nat
  firstBlockNanos : Nat
  remainingBlocksNanos : Nat

private def parityMetrics (reference candidate : Array Float) : ParityMetrics := Id.run do
  let n := min reference.size candidate.size
  let mut finite := reference.size == candidate.size
  let mut signalEnergy := 0.0
  let mut errorEnergy := 0.0
  for i in [0:n] do
    let a := reference[i]!
    let b := candidate[i]!
    if !a.isFinite || !b.isFinite then finite := false
    signalEnergy := signalEnergy + a * a
    let d := a - b
    errorEnergy := errorEnergy + d * d
  let relativeL2 := Float.sqrt (errorEnergy / max signalEnergy 1.0e-300)
  let snrDb := if errorEnergy == 0.0 then 999.0
    else 10.0 * (Float.log signalEnergy - Float.log errorEnergy) / Float.log 10.0
  return { samples := n, finite, signalEnergy, errorEnergy, relativeL2, snrDb }

private def finishTimedPlan (arena : Arena) (out : Sig) :
    Except String (Tropical.Plan.FlatPlan
      × Array (Array (Option Tropical.Ir.Stage))) := do
  let (builtArena, idx) := assemble arena "timed_bloom_metal"
    #[] #[{ name := "out", type? := some (.scalar .float) }]
    #[] #[(.port ⟨0⟩, out)] #[]
    (extraDecls := #[.param "anchor" (some farAnchorNumber)])
  let (coreArena, core) ←
    (Tropical.Ir.Strata.runResolved {} builtArena idx).mapError (·.message)
  let params : Array (String × Lean.Json) :=
    #[("anchor", .num farAnchorNumber)]
  let alloc := Tropical.Lowering.allocate #["anchor"] #[]
  Tropical.Compile.compileSessionStaged
    (.forRoot core coreArena (params := params) (alloc := alloc))

private def renderJit (plan : Tropical.Plan.FlatPlan)
    (blocks : Array (Array (Option Tropical.Ir.Stage)))
    (frames buffer : Nat) : IO (Except String (Array Float)) := do
  try
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    Tropical.StagedLoad.loadTyped rt plan blocks
    let some anchorSlot ← rt.slotIndex? "param:anchor"
      | return .error "JIT plan omitted param:anchor"
    rt.setSlot anchorSlot farAnchor
    rt.setSampleIndex 0
    let mut out : Array Float := #[]
    for _ in [0:frames] do
      rt.process
      out := out ++ decodeF64LE (← rt.outputBytes)
    return .ok out
  catch e => return .error e.toString

private def renderMetal (plan : Tropical.Plan.FlatPlan)
    (blocks : Array (Array (Option Tropical.Ir.Stage)))
    (frames buffer : Nat) : IO (Except String MetalRender) := do
  try
    let rt ← Tropical.Ffi.Runtime.new buffer.toUInt32
    let loadStart ← IO.monoNanosNow
    Tropical.StagedLoad.loadMslTyped rt plan blocks
    let loadEnd ← IO.monoNanosNow
    let some anchorSlot ← rt.slotIndex? "param:anchor"
      | return .error "Metal plan omitted param:anchor"
    rt.setSlot anchorSlot farAnchor
    rt.setSampleIndex 0
    let mut out : Array Float := #[]
    let firstStart ← IO.monoNanosNow
    if frames > 0 then
      rt.processOffline
      out := out ++ decodeF64LE (← rt.outputBytes)
    let firstEnd ← IO.monoNanosNow
    let remainingStart ← IO.monoNanosNow
    for _ in [1:frames] do
      rt.processOffline
      out := out ++ decodeF64LE (← rt.outputBytes)
    let remainingEnd ← IO.monoNanosNow
    return .ok {
      samples := out
      loadNanos := loadEnd - loadStart
      firstBlockNanos := firstEnd - firstStart
      remainingBlocksNanos := remainingEnd - remainingStart }
  catch e => return .error e.toString

private def metalUnavailable (e : String) : Bool :=
  e.contains "without TROPICAL_METAL" || e.contains "no Metal device"

/-- One moved-seam pair through the typed JIT and Metal runtime paths. The far
    anchor is a live s0 parameter so its four 16-bit limbs must
    cross through `coeff_columns`; the logical clock is translated by the exact
    matching Q32.32 value, making the rendered window start at the pair's onset.

    The runtime half is intentionally opt-in: this terminal is a rejected cost
    spike, and an unbounded driver compile may not stall ordinary validation.
    When opted in, Metal is still an optional build/device capability. A build explicitly lacking
    `TROPICAL_METAL`, or a host with no Metal device, records a portable skip;
    every emitter, loader, pipeline, dispatch, or numerical failure on a Metal-
    capable build remains a gate failure. -/
def runTimedBloomMetalParity (arena : Arena) : IO Bool := do
  let g : Float := 1.8
  let betaMax : Float := 0.5
  let sigLo : Float := 6.91 / 12.0
  let sigHi : Float := 6.91 / 0.2
  let (full, _) := defaultGongModes 110.0
  let room := Tropical.Playground.bakedReverbProbe 32
  let some voice := full[10]?
    | return ← failGate gateName "default full register omitted moved-seam voice[10]"
  let some roomMode := room[20]?
    | return ← failGate gateName "32-mode room omitted moved-seam room[20]"
  let some voiceSigma := sigConstF? voice.sigma
    | return ← failGate gateName "moved-seam voice sigma did not settle"
  let some voiceOmega := sigConstF? voice.omega
    | return ← failGate gateName "moved-seam voice omega did not settle"
  let some roomOmega := sigConstF? roomMode.omega
    | return ← failGate gateName "moved-seam room omega did not settle"
  let mu : CplxB := ⟨-voiceSigma, voiceOmega⟩
  let movedPlan ← match planTimedBloomBetaPair
      mu roomOmega sigLo sigHi betaMax 1.0 g with
    | .ok p => pure p
    | .error e =>
      return ← failGate gateName s!"moved-seam planner refused f10×r20: {e.label}"
  if movedPlan.incumbent then
    return ← failGate gateName "f10×r20 unexpectedly returned to the incumbent depth plan"
  let nu : CplxB := ⟨-sigLo, roomOmega⟩
  let pairC := cmulE voice.ampE roomMode.ampE
  let some pair := materializeTimedBloomBetaPair?
      mu nu pairC betaMax betaMax 1.0 g movedPlan
    | return ← failGate gateName "moved f10×r20 plan did not materialize at beta max"

  let anchor := (Sig.paramRef ⟨0⟩ : Sig)
  let banks : Array TimedBloomBank := #[{ pairs := #[pair], anchor }]
  -- Runtime sample zero maps to the far negative authored anchor.  The offset
  -- is exact i64 Q32.32; subsequent samples therefore exercise ordinary small
  -- positive branch age without weakening the far signed/fractional transport.
  let evalClock := add clockLit (litI farAnchorQ)
  let batch ← match bloomTimedBatchSig banks evalClock with
    | .ok s => pure s
    | .error e => return ← failGate gateName e
  let (plan, stageBlocks) ← match finishTimedPlan arena batch with
    | .ok built => pure built
    | .error e => return ← failGate gateName s!"build: {firstLine e}"
  let split ← match Tropical.Ir.Stage0.hoistTyped plan stageBlocks with
    | .ok s => pure s
    | .error e => return ← failGate gateName s!"typed split: {firstLine e}"
  let anchorColumns := split.audio.coeffArraySlots.size
  if anchorColumns != 4 then
    return ← failGate gateName
      s!"expected four hoisted Q32.32 anchor-limb columns; got {anchorColumns}"

  let runDevice := (← IO.getEnv "TROPICAL_TIMED_BLOOM_METAL") == some "1"
  if !runDevice then
    match Tropical.Ir.EmitMsl.emitKernel split.audio with
    | .error e => return ← failGate gateName s!"MSL emit: {firstLine e}"
    | .ok msl =>
      let hasColumns := msl.contains "coeff_columns [[buffer(3)]]"
      if !hasColumns then
        return ← failGate gateName "typed MSL omitted coeff_columns buffer(3)"
      return ← passGate gateName
        s!"typed moved-pair MSL emits four far-anchor columns ({msl.length} bytes); real JIT↔Metal render is an explicit timeout-bounded diagnostic because the first host attempt exceeded five minutes"

  let buffer : Nat := 512
  let frames : Nat := 16
  let jit ← renderJit plan stageBlocks frames buffer
  let reference ← match jit with
    | .ok samples => pure samples
    | .error e => return ← failGate gateName s!"JIT render: {firstLine e}"
  let metal ← renderMetal plan stageBlocks frames buffer
  match metal with
  | .error e =>
    if metalUnavailable e then
      passGate gateName
        s!"SKIP: Metal unavailable on this build/host ({firstLine e}); JIT and four-column typed transport built and rendered"
    else
      failGate gateName s!"Metal render: {firstLine e}"
  | .ok timed =>
    let m := parityMetrics reference timed.samples
    IO.println "        moved timed-bloom batch, JIT ↔ Metal (f10×r20 @ beta max):"
    IO.println s!"        anchor={farAnchor} samples ({farAnchorQ} Q32.32) · hoisted anchor columns={anchorColumns} · moved depth={movedPlan.nDepth}/{movedPlan.kDepth}"
    IO.println s!"        samples={m.samples} finite={m.finite} · JIT energy={m.signalEnergy} · error energy={m.errorEnergy} · rel-L2={m.relativeL2} · SNR={m.snrDb} dB"
    IO.println s!"        non-gating lower-bound timing (one moved pair): Metal typed load={timed.loadNanos / 1000} us · first 512-frame offline block={timed.firstBlockNanos / 1000} us · remaining 15 blocks={timed.remainingBlocksNanos / 1000} us"
    if m.finite && m.samples == frames * buffer && m.signalEnergy > 1.0e-12
        && m.relativeL2 < 3.0e-4 then
      passGate gateName
        s!"the moved non-public batch terminal crosses four far-anchor columns and agrees across typed JIT/Metal loads (rel-L2 {m.relativeL2}, SNR {m.snrDb} dB)"
    else
      failGate gateName
        s!"finite={m.finite} samples={m.samples} energy={m.signalEnergy} rel-L2={m.relativeL2} SNR={m.snrDb} dB"

end Tropical.Tropicaltest.TimedBloomMetal
