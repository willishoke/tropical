import Tropical.Engine.Core

/-!
# Engine.Compile — the compile path: mirror, elaborate, partition, hot-swap

Every graph mutation ends here. `syncCompile` lowers the session mirror,
elaborates it, downcasts to Core, partitions, assembles the plan, and hot-swaps
it into the Lean-owned runtime; a failed compile leaves the mutated graph in
place and the previous kernel playing. `sessionToResolvedRoot` is the direct
session→resolved-root lowering (no parsed round-trip); `liftIfNeeded` lifts
free wire expressions into synthetic programs; `buildKernelIr`/`loadKernel`
bridge to the FFI. `adoptResolved` re-adopts a serialized resolved entry into
the typed store.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── Typed-store adoption ─────────────────────────────────────────────────────

/-- Adopt a catalog entry's `resolved` IR into the typed store (EngineM
    shell over `SessionSt.adoptResolved`). Returns the store index, or
    `none` for entries without resolved IR (generic templates). A decode
    failure is an engine bug — the service encoded it — so it maps to
    `internal_error`. -/
def adoptResolved (env : Env) (entry : Json) : EngineM (Option Tropical.Ir.ProgramIdx) := do
  let st ← env.state.get
  match st.adoptResolved entry with
  | .error msg => internalError msg
  | .ok (st', idx?) =>
    env.state.set st'
    pure idx?

-- ── Engine-side strata ───────────────────────────────────────────────────────

/-- The service session's `inlineNested` (makeSession defaults it to
    `true`; nothing on the MCP path overrides it). Threaded into every
    engine-side strata run so the two sides can never disagree about
    which realization the registered catalog holds. -/
private def sessionInlineNested : Bool := true

/-- The direct lowering + the Core downcast (inline path), as a pure
    `Except` so call sites choose the envelope: registration failures
    map to `internal_error`. (Retired constructs are refused at the JSON
    front doors — by the time IR exists here it is trunk-only.) -/
def runStrataChecked (arena : Tropical.Ir.Arena)
    (rootIdx : Tropical.Ir.ProgramIdx) :
    Except String (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let (arena, rootIdx) ←
    (Tropical.Ir.Strata.run
      { inlineNested := sessionInlineNested } arena rootIdx).mapError (·.message)
  if sessionInlineNested then
    if let .error e := Tropical.Ir.checkResolvedArena arena rootIdx then
      throw s!"core check failed: {e}"
  pure (arena, rootIdx)

/-- Emit the (LLVM IR text, manifest JSON) for a compiled session plan
    without loading it. Shared by the audio compile (`syncCompile`) and the
    lazy inspection compile (Part II): both realizations go through the same
    `toWire`/`emitKernel` path; only the originating `FlatPlan` differs. -/
def buildKernelIr (plan : Tropical.Plan.FlatPlan) : EngineM (String × String) := do
  let planJson ← match plan.toWire with
    | .error msg => internalError msg
    | .ok j => pure j.compress
  let ir ← match Tropical.Ir.EmitLlvm.emitKernel plan with
    | .error msg => internalError s!"EmitLlvm: {msg}"
    | .ok s => pure s
  pure (ir, planJson)

/-- Emit the plan's kernel artifacts and load them into the runtime: the
    stage-0 split first (Stage0.hoist, gated by `TROPICAL_STAGE0`), then
    LLVM IR for the audio kernel — and the coefficient kernel when
    anything hoisted — always; on the metal backend also the MSL kernel,
    dual-loaded (audio on the GPU, render_window/the scope on the JIT;
    coefficients run on the CPU JIT in f64 and cross to the GPU as
    host-written slots, coefficient COLUMNS as the packed `coeff_columns`
    buffer). Any emit failure errors BEFORE the load, so the
    previous kernel keeps playing — the same recoverable contract as the
    IR path. -/
def loadKernel (env : Env) (plan : Tropical.Plan.FlatPlan)
    (stages? : Option (Array (Array (Option Tropical.Ir.Stage))) := none) :
    EngineM Unit := do
  let split ← match stages? with
    | some blocks => Tropical.StagedLoad.splitTyped plan blocks
    | none => Tropical.StagedLoad.split plan
  let (ir, planJson) ← buildKernelIr split.audio
  let coeffIr ← match Tropical.StagedLoad.coeffIr split with
    | .error msg => internalError s!"EmitLlvm (coeff): {msg}"
    | .ok s => pure s
  if env.metalBackend then
    -- Hoisted coefficient columns (banks-as-data) cross to the GPU as
    -- a packed `coeff_columns` device buffer (`buffer(3)`): EmitMsl
    -- emits column reads for the slots the split advertises, the
    -- stage-0 coefficient kernel fills the generation-buffered storage
    -- host-side in f64, and process() uploads the captured generation
    -- per dispatch (f64→f32, like slots). So the metal backend runs
    -- the SAME typed split as the JIT — coefficient math at knob rate
    -- on CPU, the audio loop on GPU reading real columns.
    let msl ← match Tropical.Ir.EmitMsl.emitKernel split.audio with
      | .error msg => internalError s!"EmitMsl: {msg}"
      | .ok s => pure s
    env.runtime.loadIrStaged ir msl coeffIr planJson
  else
    env.runtime.loadIrStaged ir "" coeffIr planJson

-- ── Snapshot compile (`wire()` in TS) ────────────────────────────────────────

/-- If any wire needs lifting (array literals), lift each to an
    anonymous `__wire_N` program — engine-side as of Phase 5 stage 6b
    (port of `liftWiresToInstances`): build the wire program
    (`Tropical.Ir.WireProgram.lift`), run strata, ship the raw AND
    post-strata forms to the service's `register_lifted` (registry
    residue: typeRegistry Compiled + the raw form in
    `session.programs`, which export_program's resolver reads), adopt
    the entry, add the instance, wire each free ref back to its
    source, and replace the original wire with a ref to the lifted
    instance — same observable behavior the TS engine's in-session
    lift had. Lift/strata failures were plain TS Errors → internal_error
    with the verbatim message. -/
def liftIfNeeded (env : Env) : EngineM Unit := do
  let st ← env.state.get
  -- Capture before mutation — never re-lift a wire just inserted.
  let toLift := st.wires.filter (fun w => Tropical.Lowering.needsWireLift w.expr)
  for w in toLift do
    let st ← env.state.get
    let counter := (st.nameCounters.get? "__wire").getD 0 + 1
    let synthName := s!"__wire_{counter}"
    let (prog, sortedRefs, exprs) ← match Tropical.Ir.WireProgram.lift w.expr synthName st.arena.exprs with
      | .error msg => internalError msg
      | .ok r => pure r
    -- The raw lifted program joins the store: templateByName mirrors
    -- TS `session.programs.set(name, lifted)` — the RAW form, which
    -- is what a later registration's relink byName would see. The lifted
    -- body's ids intern into the store's shared expression DAG.
    let arenaRaw := { st.arena with programs := st.arena.programs.push prog, exprs }
    let rawIdx : Tropical.Ir.ProgramIdx := ⟨st.arena.programs.size⟩
    -- (the raw form itself is never encoded — it has no consumer since the
    -- service residue left; only the post-strata encode below is used)
    -- Strata, no relink (lifted bodies have no InstanceDecls — the
    -- registry is empty, exactly the TS lift's `programRegistry: new
    -- Map()`).
    let (arenaPost, postIdx) ← match runStrataChecked arenaRaw rawIdx with
      | .error msg => internalError msg
      | .ok r => pure r
    -- Persist the raw program + counter; the post-strata arena growth
    -- is transient — the store adopts the service entry's round trip,
    -- one copy, like the registration path.
    env.state.modify fun s =>
      { s with arena := arenaRaw,
               nameCounters := s.nameCounters.insert "__wire" counter,
               templateByName := s.templateByName.insert synthName rawIdx }
    let entry ← match Tropical.Entries.concreteEntry arenaPost synthName postIdx with
      | .error e => internalError e
      | .ok j => pure j
    let pm := ProgMeta.fromEntry entry
    let resolvedIdx ← adoptResolved env entry
    env.state.modify fun s =>
      let s := s.addProgram pm
      if (s.findInstance? synthName).isSome then s
      else s.addInstance synthName
        { baseTypeName := synthName, typeArgs := none, progMeta := pm, resolvedIdx }
    -- Wire each free ref to its corresponding input on the lifted
    -- instance (raw refs, NO delay wrap — `liftOneWire` set
    -- inputExprNodes directly), then replace the original wire in
    -- place (TS-Map position semantics).
    for (inst, port) in sortedRefs do
      let inputName := s!"{inst.replace "." "_"}__{port}"
      env.state.modify (·.setWireRaw synthName inputName <| Json.mkObj
        [("op", Json.str "ref"), ("instance", Json.str inst), ("output", Json.str port)])
    env.state.modify (·.setWireRaw w.instName w.portName <| Json.mkObj
      [("op", Json.str "ref"), ("instance", Json.str synthName), ("output", Json.str "out")])

-- ─────────────────────────────────────────────────────────────
-- Session → resolved root DIRECTLY (no parsed round-trip)
-- ─────────────────────────────────────────────────────────────

/-! The session graph is already post-elaborate-shaped — instances carry resolved
    type snapshots and wires are graph edges — so `sessionToResolvedRoot` builds
    the resolved root `Program` DIRECTLY, reproducing what the elaborator would
    have produced byte-for-byte. This replaced the former `sessionToParsed →
    reparse → elaborate` round-trip (serialize the instances into a NAMED
    `__session__` ParsedProgram, re-elaborate the names back to pointers), against
    which it was gated `tropical_resolved_1`-identical on every golden before that
    path was deleted. `elaborate` stays the reifier
    for the morphism-definition (`.trop`) language; the patcher skips it by BEING
    a graph. The construction (verified against `Elaborator.lean`):
    instance decls in topo order then params alphabetical; each `InstanceInput`
    `port` = the target program's input position; wires resolved to `Ir.Expr`
    (`nestedOut ⟨topoIdx⟩ ⟨outputIdx⟩`, `paramRef ⟨alphaIdx⟩`, builtins —
    `clock()` ⇒ `sampleIndex << 32`, `sampleRate`/`sampleIndex` direct); registry
    accumulated per-instance with the transitive merge. -/

/-- Collect the param/trigger names a wire expression references (for the root's
    alphabetical param table). -/
private partial def collectWireParams (expr : Json) : Array String :=
  match expr with
  | .arr items => items.foldl (fun acc e => acc ++ collectWireParams e) #[]
  | .obj _ =>
    let op := (opOf? expr).getD ""
    if op == "param" || op == "trigger" then #[(getStrField? expr "name").getD ""]
    else
      let args := match getField? expr "args" with | some (.arr a) => a | _ => #[]
      let items := match getField? expr "items" with | some (.arr a) => a | _ => #[]
      (args ++ items).foldl (fun acc e => acc ++ collectWireParams e) #[]
  | _ => #[]

/-- Resolution context for a session wire expression. -/
private structure WireCtx where
  /-- `(instanceName, outputName)` → the `nestedOut` leaf node. -/
  instOut : String → String → Except String Tropical.Ir.ENode
  /-- param/trigger name → `ParamIdx` (alphabetical position). -/
  paramIdx : String → Option Nat

private abbrev WireM := StateT Tropical.Ir.ExprArena (Except String)

private def internWE (n : Tropical.Ir.ENode) : WireM Tropical.Ir.ExprId :=
  fun a => .ok ((Tropical.Ir.eintern n).run a)

/-- Resolve a raw session wire expression directly to a resolved arena `ExprId`,
    mirroring the elaborator's `resolveExpr` over a session-root scope, interning
    into the shared DAG. Same op set as `wireExprToParsed`; no parsed intermediate. -/
private partial def wireExprToResolved (ctx : WireCtx) (expr : Json) :
    WireM Tropical.Ir.ExprId :=
  match expr with
  | .num n => internWE (.num n)
  | .bool b => internWE (.bool b)
  | .arr items => do internWE (.arr (← items.mapM (wireExprToResolved ctx)))
  | .obj _ => do
    let some op := opOf? expr
      | throwThe String s!"session wire node missing op: {expr.compress}"
    let rawArgs := match getField? expr "args" with | some (.arr a) => a | _ => #[]
    let args ← rawArgs.mapM (wireExprToResolved ctx)
    if op == "ref" then
      internWE (← ctx.instOut ((getStrField? expr "instance").getD "") ((getStrField? expr "output").getD ""))
    else if op == "param" || op == "trigger" then
      let name := (getStrField? expr "name").getD ""
      match ctx.paramIdx name with
      | some i => internWE (.paramRef ⟨i⟩)
      | none => throwThe String s!"session wire: param '{name}' not in root param table"
    else if op == "array" then
      match getField? expr "items" with
      | some (.arr items) => do internWE (.arr (← items.mapM (wireExprToResolved ctx)))
      | _ => throwThe String "session wire: 'array' missing items"
    else if op == "sampleRate" then internWE .sampleRate
    else if op == "sampleIndex" then internWE .sampleIndex
    else if op == "clock" || op == "sampleClock" || op == "sample_clock" then
      internWE (.binary .lshift (← internWE .sampleIndex) (← internWE (.num ⟨32, 0⟩)))
    else if op == "clamp" then
      if args.size == 3 then internWE (.clamp args[0]! args[1]! args[2]!)
      else throwThe String s!"session wire: clamp expects 3 args, got {args.size}"
    else if op == "select" then
      if args.size == 3 then internWE (.select args[0]! args[1]! args[2]!)
      else throwThe String s!"session wire: select expects 3 args, got {args.size}"
    else if op == "arraySet" || op == "array_set" then
      if args.size == 3 then internWE (.arraySet args[0]! args[1]! args[2]!)
      else throwThe String s!"session wire: arraySet expects 3 args, got {args.size}"
    else if op == "index" then
      if args.size == 2 then internWE (.index args[0]! args[1]!)
      else throwThe String s!"session wire: index expects 2 args, got {args.size}"
    else if let some tag := Tropical.Ir.BinaryOpTag.ofWire? op then
      if args.size == 2 then internWE (.binary tag args[0]! args[1]!)
      else throwThe String s!"session wire: binary '{op}' expects 2 args, got {args.size}"
    else if let some tag := Tropical.Ir.UnaryOpTag.ofWire? op then
      if args.size == 1 then internWE (.unary tag args[0]!)
      else throwThe String s!"session wire: unary '{op}' expects 1 arg, got {args.size}"
    else
      throwThe String s!"session wire: unsupported op '{op}'"
  | _ => throwThe String s!"session wire: invalid value {expr.compress}"

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
  -- params: every param/trigger referenced in any wire, alphabetical.
  let mut pnames : Array String := #[]
  for w in wiresPost do
    for nm in collectWireParams w.expr do
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

    THE one session-compile prologue, shared by `syncCompile` (production) and the
    `compileMirror*` harness entry points. `ctx` labels the engine-bug internal
    errors for the caller. `mode` reaches only the plan's `compilation_mode`
    field. -/
def buildSessionInputVia (env : Env) (ctx : String)
    (mode : Tropical.Plan.CompilationMode := .fused) :
    EngineM Tropical.Compile.SessionInput := do
  let st ← env.state.get
  let alloc := Tropical.Lowering.allocate (st.params.map (·.1)) st.instances
  let wiresPost := st.wires
  Tropical.Lowering.assertSessionAcyclic st.instances wiresPost

  -- TS parity (`sessionToParsedProgram` emits `inst.compiled.prog.name`):
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

/-- The session lowering + compile: lift free wires if needed, build the session
    input from the mirror (the shared prologue), compile it to a plan with typed
    stage blocks, and hot-swap the result into the Lean-owned runtime. A failed
    compile leaves the mutated graph in place and the previous kernel playing. -/
def syncCompile (env : Env) : EngineM Unit := do
  liftIfNeeded env
  -- Build the resolved session root directly (the only session path).
  let input ← buildSessionInputVia env "syncCompile"
  let (plan, stageBlocks) ← match Tropical.Compile.compileSessionStaged input with
    | .error msg => internalError msg
    | .ok p => pure p
  -- Lean owns codegen: emit the kernel artifacts from the in-memory plan
  -- and hand them to the engine (planJson is the metadata manifest). The
  -- split is TYPED here — the session compile is in hand.
  loadKernel env plan (some stageBlocks)

end Tropical.Engine
