import Tropical.Session.Registry
import Tropical.Session.Ingest
import Tropical.Ir.WireProgram
import Tropical.Lowering
import Tropical.Compile

/-!
# Session.Compile — the session graph → `tropical_plan_7`, purely

The compile path over `SessionSt`: lift free wire expressions into synthetic
programs, build the resolved session root directly from the graph, check its
Core shape, partition, and assemble the plan with its typed stage blocks.
Nothing here touches a runtime: the engine's `syncCompile` runs
`compileStaged` and publishes the result through its host port; the offline
compiler runs `compilePatch` and writes the artifact.
-/

namespace Tropical.Session

open Lean (Json toJson)

-- ── Snapshot compile ─────────────────────────────────────────────────────────

/-- If any wire needs lifting (array literals), lift each to an
    anonymous `__wire_N` program: build the wire program
    (`Tropical.Ir.WireProgram.lift`), run strata, register the raw form
    under its name (what `export_program`'s resolver reads) and the
    post-strata entry as the instance's resolved type, add the instance,
    wire each free ref back to its source, and replace the original wire
    with a ref to the lifted instance. Lift/strata failures are
    `internal_error` with the verbatim message. -/
def liftIfNeeded : SessionM Unit := do
  let st ← get
  -- Capture before mutation — never re-lift a wire just inserted.
  let toLift := st.wires.filter (·.expr.needsLift)
  for w in toLift do
    let st ← get
    let counter := (st.nameCounters.get? "__wire").getD 0 + 1
    let synthName := s!"__wire_{counter}"
    let lifted ← match Tropical.Ir.WireProgram.lift w.expr synthName st.arena.exprs with
      | .error msg => internalError msg
      | .ok r => pure r
    -- The raw lifted program joins the store under its name — the RAW
    -- form, which is what a later registration's relink byName sees. The
    -- lifted body's ids intern into the store's shared expression DAG.
    let arenaRaw := {
      st.arena with
      programs := st.arena.programs.push lifted.program
      exprs := lifted.exprs }
    let rawIdx : Tropical.Ir.ProgramIdx := ⟨st.arena.programs.size⟩
    -- Only the post-strata form is encoded into an entry below. Strata
    -- runs with no relink: a lifted body has no InstanceDecls.
    let (arenaPost, postIdx) ← match runStrataChecked arenaRaw rawIdx with
      | .error msg => internalError msg
      | .ok r => pure r
    -- Persist the raw program + counter; the post-strata arena growth
    -- is transient — the store adopts the entry's round trip, one copy,
    -- like the registration path.
    modify fun s =>
      { s with arena := arenaRaw,
               nameCounters := s.nameCounters.insert "__wire" counter,
               templateByName := s.templateByName.insert synthName rawIdx }
    let entry ← match Tropical.Entries.concreteEntry arenaPost synthName postIdx with
      | .error e => internalError e
      | .ok j => pure j
    let pm := ProgMeta.fromEntry entry
    let resolvedIdx ← adoptResolved entry
    modify fun s =>
      let s := s.addProgram pm
      if (s.findInstance? synthName).isSome then s
      else s.addInstance synthName
        { baseTypeName := synthName, typeArgs := none, progMeta := pm, resolvedIdx }
    -- Wire each free ref to its corresponding input on the lifted instance,
    -- then replace the original typed wire in place without changing its
    -- deterministic position.
    for ref in lifted.freeRefs do
      modify (·.setWireRaw synthName ref.inputName
        (.ref ref.instanceName (.name ref.outputName)))
    modify (·.setWireRaw w.instName w.portName (.ref synthName (.name "out")))

-- ─────────────────────────────────────────────────────────────
-- Session → resolved root DIRECTLY (no parsed round-trip)
-- ─────────────────────────────────────────────────────────────

/-! The session graph already has resolved identity — instances carry resolved
    type snapshots and wires are graph edges — so `sessionToResolvedRoot` builds
    the resolved root `Program` DIRECTLY. This replaced the historical
    parsed-program/reparse/elaborate round-trip and was gated
    `tropical_resolved_1`-identical before that path was deleted. The current
    construction is:
    instance decls in topo order then params alphabetical; each `InstanceInput`
    `port` = the target program's input position; wires resolved to `Ir.Expr`
    (`nestedOut ⟨topoIdx⟩ ⟨outputIdx⟩`, `paramRef ⟨alphaIdx⟩`, builtins —
    `clock()` ⇒ `tau`, `sampleIndex` ⇒ `tau >> clockFracBits`, `sampleRate` direct); registry
    accumulated per-instance with the transitive merge. -/

/-- Resolution context for a session wire expression. Not `private`: the
    export path (`ProgramIO.handleExportProgram`) drives the same conversion
    with its own ref/name resolution (siblings → `nestedOut` by decl position,
    exposed ports → `inputRef`, no params). -/
structure WireCtx where
  /-- `(instanceName, outputName)` → the `nestedOut` leaf node. -/
  instOut : String → String → Except String Tropical.Ir.ENode
  /-- param name → `ParamIdx` (alphabetical position). -/
  paramIdx : String → Option Nat
  /-- name → `InputIdx` — the fallback category after params, mirroring the
      resolution order names always had (params, then inputs). The session
      root has no input ports (`fun _ => none`); export's program body does. -/
  inputIdx : String → Option Nat := fun _ => none

abbrev WireM := StateT Tropical.Ir.ExprArena (Except String)

private def internWE (n : Tropical.Ir.ENode) : WireM Tropical.Ir.ExprId :=
  fun a => .ok ((Tropical.Ir.eintern n).run a)

/-- Resolve a session wire expression directly to a resolved arena `ExprId`,
    mirroring the elaborator's `resolveExpr` over a session-root scope,
    interning into the shared DAG. Structural over `WireExpr` — the
    grammar checks happened at decode; only the storable-but-uncompilable
    forms are refused here. -/
def wireExprToResolved (ctx : WireCtx) (expr : Tropical.WireExpr) :
    WireM Tropical.Ir.ExprId :=
  match expr with
  | .num n => internWE (.num n)
  | .bool b => internWE (.bool b)
  | .arr items => do
    internWE (.arr (← items.attach.mapM fun ⟨x, _⟩ => wireExprToResolved ctx x))
  | .ref inst output => do
    let outName := match output with
      | .name s => s
      | .index n => n.toString
    internWE (← ctx.instOut inst outName)
  | .param name =>
    match ctx.paramIdx name with
    | some i => internWE (.paramRef ⟨i⟩)
    | none =>
      match ctx.inputIdx name with
      | some i => internWE (.inputRef ⟨i⟩)
      | none => throwThe String s!"unknown name '{name}'"
  | .sampleRate => internWE .sampleRate
  | .sampleIndex => do
    -- The integer sample index is `⌊τ⌋`; the IR's time leaf is τ itself.
    internWE (.binary .rshift (← internWE .tau) (← internWE (.num ⟨Tropical.Ir.clockFracBits, 0⟩)))
  | .clock => internWE .tau
  | .clamp a b c => do
    internWE (.clamp (← wireExprToResolved ctx a) (← wireExprToResolved ctx b)
      (← wireExprToResolved ctx c))
  | .select a b c => do
    internWE (.select (← wireExprToResolved ctx a) (← wireExprToResolved ctx b)
      (← wireExprToResolved ctx c))
  | .arraySet a b c => do
    internWE (.arraySet (← wireExprToResolved ctx a) (← wireExprToResolved ctx b)
      (← wireExprToResolved ctx c))
  | .index a b => do
    internWE (.index (← wireExprToResolved ctx a) (← wireExprToResolved ctx b))
  | .binary tag a b => do
    internWE (.binary tag (← wireExprToResolved ctx a) (← wireExprToResolved ctx b))
  | .unary tag a => do
    internWE (.unary tag (← wireExprToResolved ctx a))
  | .broadcastTo .. | .input _ | .nestedOut .. =>
    throwThe String s!"session wire: unsupported op '{expr.opName}'"
termination_by sizeOf expr
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

/-- Build the resolved session root `Program` directly from the graph (instances
    already carry resolved type snapshots via `resolverTbl`), deleting the
    `sessionToParsed → reparse → elaborate` round-trip. Byte-identical to the
    elaborated root (gated). `lowerInstances` carries each instance's STORED
    program name (the resolver key). Not `private`: the harness mirror
    (`Testing/EngineMirror`) drives the same construction cross-module. -/
def sessionToResolvedRoot (arena : Tropical.Ir.Arena)
    (lowerInstances : Array (String × InstanceInfo)) (wiresPost : Array Tropical.Wire)
    (resolverTbl : Array (String × Tropical.Ir.ProgramIdx)) :
    Except String (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let order := Tropical.Lowering.computeInstanceTopoOrder lowerInstances wiresPost
  -- instName → topo InstanceIdx
  let mut instIdxOf : Array (String × Nat) := #[]
  for k in [0:order.size] do
    instIdxOf := instIdxOf.push (order[k]!, k)
  -- params: every param referenced in any wire, alphabetical.
  let mut pnames : Array String := #[]
  for w in wiresPost do
    for nm in w.expr.paramNames do
      if !pnames.contains nm then pnames := pnames.push nm
  let sortedParams := pnames.qsort (· < ·)
  -- the wire-resolution context (closes over the maps above).
  let instOut := fun (rn on : String) =>
    match (instIdxOf.find? (·.1 == rn)).map (·.2) with
    | none => .error s!"session wire ref: instance '{rn}' not declared"
    | some idx =>
      match (lowerInstances.find? (·.1 == rn)).map (·.2.progMeta.programName) with
      | none => .error s!"session wire ref: instance '{rn}' not found"
      | some sn =>
        match (resolverTbl.find? (·.1 == sn)).map (·.2) with
        | none => .error s!"session wire ref: program '{sn}' not resolved"
        | some ti =>
          match arena.program? ti with
          | none => .error "session wire ref: target out of range"
          | some tgt =>
            match tgt.outputs.findIdx? (·.name == on) with
            | none => .error s!"session wire ref: '{rn}' ({tgt.name}) has no output '{on}'"
            | some o => .ok (Tropical.Ir.ENode.nestedOut ⟨idx⟩ ⟨o⟩)
  let ctx : WireCtx := { instOut, paramIdx := fun nm => sortedParams.findIdx? (· == nm) }
  -- instance decls (topo order), then param decls (alphabetical). Wire
  -- expressions intern into the shared DAG (seeded from the arena's exprs).
  let mut exprs := arena.exprs
  let mut decls : Array Tropical.Ir.BodyDecl := #[]
  for name in order do
    let some info := (lowerInstances.find? (·.1 == name)).map (·.2)
      | .error s!"sessionToResolvedRoot: instance '{name}' missing (topo bug)"
    let sn := info.progMeta.programName
    let some ti := (resolverTbl.find? (·.1 == sn)).map (·.2)
      | .error s!"sessionToResolvedRoot: instance '{name}' program '{sn}' not resolved"
    let some tgt := arena.program? ti
      | .error s!"sessionToResolvedRoot: target idx out of range for '{sn}'"
    let wires := (wiresPost.filter (·.instName == name)).qsort (fun a b => a.portName < b.portName)
    let mut inputs : Array Tropical.Ir.InstanceInput := #[]
    for w in wires do
      let some pos := tgt.inputs.findIdx? (·.name == w.portName)
        | .error s!"sessionToResolvedRoot: '{name}' ({tgt.name}) has no input '{w.portName}'"
      let (value, exprs') ← (wireExprToResolved ctx w.expr).run exprs
      exprs := exprs'
      inputs := inputs.push { port := ⟨pos⟩, value }
    decls := decls.push (.inst name tgt.name inputs)
  for pname in sortedParams do
    decls := decls.push (.param pname none)
  -- registry: per-instance (topo order) target name → idx, transitive merge.
  let mut registry : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for name in order do
    let some info := (lowerInstances.find? (·.1 == name)).map (·.2)
      | .error s!"sessionToResolvedRoot: instance '{name}' missing (registry)"
    let sn := info.progMeta.programName
    let some ti := (resolverTbl.find? (·.1 == sn)).map (·.2)
      | .error s!"sessionToResolvedRoot: program '{sn}' not resolved (registry)"
    let some tgt := arena.program? ti
      | .error s!"sessionToResolvedRoot: target out of range (registry)"
    registry := match registry.findIdx? (·.1 == tgt.name) with
      | some i => registry.set! i (tgt.name, ti)
      | none => registry.push (tgt.name, ti)
    for (k, v) in tgt.registry do
      if !registry.any (·.1 == k) then registry := registry.push (k, v)
  let prog : Tropical.Ir.Program := { name := "__session__", decls, registry }
  let idx : Tropical.Ir.ProgramIdx := ⟨arena.programs.size⟩
  .ok ({ arena with programs := arena.programs.push prog, exprs }, idx)

/-- Build the `SessionInput` from the engine's current session mirror: allocate,
    assert acyclic, rewrite each instance's `programName` to its stored program's
    name, build the first-instance-wins name→idx resolver, construct the resolved
    session root directly via `sessionToResolvedRoot`, Core-check it, and
    materialize the session instances against the root registry.

    THE one session-compile prologue, shared by `compileStaged` (the engine's
    `syncCompile` and the offline compiler) and the `compileMirror*` harness
    entry points. `ctx` labels the engine-bug internal
    errors for the caller. `mode` reaches only the plan's `compilation_mode`
    field. -/
def buildSessionInput (ctx : String)
    (mode : Tropical.Plan.CompilationMode := .fused) :
    SessionM Tropical.Compile.SessionInput := do
  let st ← get
  let alloc := Tropical.Lowering.allocate (st.params.map (·.1)) st.instances
  let wiresPost := st.wires
  Tropical.Lowering.assertSessionAcyclic st.instances wiresPost

  -- Preserved historical parity: the retired parsed-session path emitted the
  -- stored program name here.
  -- the root references each instance's program by the *stored*
  -- program's name — for specialized generics that is the base name
  -- (`Delay`), not the display key the catalog entry carries
  -- (`Delay<N=8>`). Falls back to the entry name when an instance has
  -- no snapshot (engine bug — elaboration will report it).
  let storedProgName (i : InstanceInfo) : Option String :=
    (i.resolvedIdx.bind st.arena.program?).map (·.name)
  let lowerInstances := st.instances.map fun (n, i) =>
    match storedProgName i with
    | some pname => (n, { i with progMeta := { i.progMeta with programName := pname } })
    | none => (n, i)
  -- Root resolver = `sessionTypeResolver` parity: keyed by the stored
  -- program's name, instance order, first instance wins per name. Uses
  -- the instances' snapshots, never a late name lookup.
  let mut resolverTbl : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for (_, i) in st.instances do
    if let some idx := i.resolvedIdx then
      if let some p := st.arena.program? idx then
        if !resolverTbl.any (·.1 == p.name) then
          resolverTbl := resolverTbl.push (p.name, idx)
  let tbl := resolverTbl

  -- Build the resolved session root directly (no parsed round-trip): the session
  -- graph is already post-elaborate-shaped. Failure maps onto the recoverable
  -- envelope so the previous kernel keeps playing.
  let (arena', rootIdx) ←
    match sessionToResolvedRoot st.arena lowerInstances wiresPost tbl with
    | .error e => internalError e
    | .ok r => pure r
  let (rootArena, rootCore) ← match Tropical.Ir.checkResolvedArena arena' rootIdx with
    | .error e => internalError s!"{ctx}: post-construction Core check failed (engine bug): {e}"
    | .ok r => pure r

  -- Session instances in registry order, each materialized as the Core
  -- form the root's registry linked (first-instance-wins per stored
  -- program name — the sessionTypeResolver contract).
  let mut coreInstances : Array (String × Tropical.Ir.Core.CoreProgram) := #[]
  for (n, i) in st.instances do
    let some pname := storedProgName i
      | internalError s!"{ctx}: instance '{n}' has no resolved snapshot (engine bug)"
    let some core := rootCore.registryGet? pname
      | internalError s!"{ctx}: instance '{n}' program '{pname}' missing from root registry (engine bug)"
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

/-- The session compile: lift, build the session input, compile to a plan
    with typed stage blocks. `ctx` labels engine-bug internal errors. -/
def compileStaged (ctx : String) (mode : Tropical.Plan.CompilationMode := .fused) :
    SessionM (Tropical.Plan.FlatPlan × Array (Array (Option Tropical.Ir.Stage))) := do
  liftIfNeeded
  let input ← buildSessionInput ctx mode
  match Tropical.Compile.compileSessionStaged input with
  | .error msg => internalError msg
  | .ok p => pure p

/-- A document's text → its plan, over a booted registry: normalize, load,
    compile. The whole front door, with no runtime in sight. -/
def compilePatch (text : String) (mode : Tropical.Plan.CompilationMode := .fused) :
    SessionM (Tropical.Plan.FlatPlan × Array (Array (Option Tropical.Ir.Stage))) := do
  let node ← normalizePatch text
  loadPatch node
  compileStaged "compilePatch" mode

end Tropical.Session
