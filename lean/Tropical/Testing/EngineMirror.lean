import Tropical.Engine

/-!
# EngineMirror — harness-only session recompiles (test support, not production)

The `compileMirror*` family: rebuild a `tropical_plan_5` from the engine's
current session mirror at an arbitrary compilation mode, WITHOUT loading it
or touching the runtime. The production path compiles AND loads through
`syncCompile`; these entry points exist so a harness can hold the compiled
artifact in hand — `diffcli compile`/`emit-ir` print it, the tropicaltest
golden and equivalence gates hash and cross-compare it (elaborate-path vs
direct-root, plan vs typed stage blocks). Consumed ONLY by `tropicaltest`
and the `diffcli` verbs — nothing on the production compile path imports
this module.

Everything here stays in `namespace Tropical.Engine` so call sites read
identically to the production entry points they mirror.
-/

namespace Tropical.Engine

/-- Harness-only (diffcli `compile`): rebuild the plan from the current
    mirror at an arbitrary compilation mode, WITHOUT loading it or
    touching the service. Mirrors the `compile_patch.ts` contract's
    final `compileSession(session, options)`: by load time the wires
    are lifted and every type is registered, so the mode only reaches
    the plan's `compilation_mode` field (the TS auto-flip for
    `microkernel-deep` lands after every registration — the recorded
    known limitation — so plan structure is mode-independent there
    too). Collapses into `syncCompile` at 6f when the sync payload
    retires. -/
def buildSessionInput (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM Tropical.Compile.SessionInput := do
  let st ← env.state.get
  let alloc := Tropical.Lowering.allocate (st.params.map (·.1)) st.instances
  let wiresPost := st.wires
  Tropical.Lowering.assertSessionAcyclic st.instances wiresPost
  let storedProgName (i : InstanceInfo) : Option String :=
    (i.resolvedIdx.bind st.arena.program?).map (·.name)
  let lowerInstances := st.instances.map fun (n, i) =>
    match storedProgName i with
    | some pname => (n, { i with progMeta := { i.progMeta with programName := pname } })
    | none => (n, i)
  let parsed ← Tropical.Lowering.sessionToParsed lowerInstances wiresPost
  let typed ← match Tropical.Parse.JsonV.parse parsed.compress with
    | .error e => internalError s!"session root: ParsedProgram JSON re-parse failed: {e}"
    | .ok jv =>
      match Tropical.Parse.decodeProgram jv with
      | .error e => internalError s!"session root: {e}"
      | .ok p => pure p
  let mut resolverTbl : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for (_, i) in st.instances do
    if let some idx := i.resolvedIdx then
      if let some p := st.arena.program? idx then
        if !resolverTbl.any (·.1 == p.name) then
          resolverTbl := resolverTbl.push (p.name, idx)
  let tbl := resolverTbl
  let (arena', rootIdx) ← match Tropical.Ir.elaborateInto st.arena typed
      (some fun n => (tbl.find? (·.1 == n)).map (·.2)) with
    | .error e => internalError e.message
    | .ok r => pure r
  let (rootArena, rootCore) ← match Tropical.Ir.checkResolvedArena arena' rootIdx with
    | .error e => internalError s!"compileMirrorPlan: post-elaboration Core check failed (engine bug): {e}"
    | .ok r => pure r
  let mut coreInstances : Array (String × Tropical.Ir.Core.CoreProgram) := #[]
  for (n, i) in st.instances do
    let some pname := storedProgName i
      | internalError s!"compileMirrorPlan: instance '{n}' has no resolved snapshot (engine bug)"
    let some core := rootCore.registryGet? pname
      | internalError s!"compileMirrorPlan: instance '{n}' program '{pname}' missing from root registry (engine bug)"
    coreInstances := coreInstances.push (n, core)
  pure {
    instances := coreInstances
    wiresPost
    graphOutputs := st.graphOutputs
    params := st.params
    alloc
    root := rootCore
    arena := rootArena
    mode }

def compileMirrorFlatPlan (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM Tropical.Plan.FlatPlan := do
  let input ← buildSessionInput env mode
  match Tropical.Compile.compileSession input with
  | .error msg => internalError msg
  | .ok p => pure p

/-- The stage-differential entry: the session input, the compiled plan,
    and the typed per-instruction stages in emit order. -/
def compileMirrorStaged (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM (Tropical.Compile.SessionInput × Tropical.Plan.FlatPlan
      × Array (Array (Option Tropical.Ir.Stage))) := do
  let input ← buildSessionInput env mode
  match Tropical.Compile.compileSessionStaged input with
  | .error msg => internalError msg
  | .ok (p, blocks) => pure (input, p, blocks)

def compileMirrorPlan (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM String := do
  let plan ← compileMirrorFlatPlan env mode
  match plan.toWire with
  | .error msg => internalError msg
  | .ok j => pure j.compress

/-- The C4 path: identical to `compileMirrorFlatPlan` but builds the session root
    via `sessionToResolvedRoot` (direct, no `sessionToParsed → reparse →
    elaborate`). Gated `tropical_resolved_1`/plan-identical against
    `compileMirrorFlatPlan` on the golden corpus before it becomes the default. -/
def compileMirrorFlatPlanViaArrow (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM Tropical.Plan.FlatPlan := do
  let st ← env.state.get
  let alloc := Tropical.Lowering.allocate (st.params.map (·.1)) st.instances
  let wiresPost := st.wires
  Tropical.Lowering.assertSessionAcyclic st.instances wiresPost
  let storedProgName (i : InstanceInfo) : Option String :=
    (i.resolvedIdx.bind st.arena.program?).map (·.name)
  let lowerInstances := st.instances.map fun (n, i) =>
    match storedProgName i with
    | some pname => (n, { i with progMeta := { i.progMeta with programName := pname } })
    | none => (n, i)
  let mut resolverTbl : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for (_, i) in st.instances do
    if let some idx := i.resolvedIdx then
      if let some p := st.arena.program? idx then
        if !resolverTbl.any (·.1 == p.name) then
          resolverTbl := resolverTbl.push (p.name, idx)
  let tbl := resolverTbl
  -- THE DELETION: build the resolved root directly (no parsed round-trip).
  let (arena', rootIdx) ← match sessionToResolvedRoot st.arena lowerInstances wiresPost tbl with
    | .error e => internalError e
    | .ok r => pure r
  let (rootArena, rootCore) ← match Tropical.Ir.checkResolvedArena arena' rootIdx with
    | .error e => internalError s!"compileMirrorPlanViaArrow: post-construction Core check failed: {e}"
    | .ok r => pure r
  let mut coreInstances : Array (String × Tropical.Ir.Core.CoreProgram) := #[]
  for (n, i) in st.instances do
    let some pname := storedProgName i
      | internalError s!"compileMirrorPlanViaArrow: instance '{n}' has no resolved snapshot (engine bug)"
    let some core := rootCore.registryGet? pname
      | internalError s!"compileMirrorPlanViaArrow: instance '{n}' program '{pname}' missing from root registry (engine bug)"
    coreInstances := coreInstances.push (n, core)
  match Tropical.Compile.compileSession {
      instances := coreInstances
      wiresPost
      graphOutputs := st.graphOutputs
      params := st.params
      alloc
      root := rootCore
      arena := rootArena
      mode } with
  | .error msg => internalError msg
  | .ok p => pure p

def compileMirrorPlanViaArrow (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM String := do
  let plan ← compileMirrorFlatPlanViaArrow env mode
  match plan.toWire with
  | .error msg => internalError msg
  | .ok j => pure j.compress

end Tropical.Engine
