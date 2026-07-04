import Std.Data.HashMap
import Tropical.Errors
import Tropical.Ffi
import Tropical.Expr
import Tropical.Wiring
import Tropical.Session
import Tropical.Lowering
import Tropical.Parse.Nodes
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Ir.WireProgram
import Tropical.Ir.EmitLlvm
import Tropical.TypeArgs
import Tropical.Compile
import Tropical.Entries
import Tropical.Playground

/-!
# The tropical IR engine — tool semantics, in Lean

Port of `mcp/engine.ts`. As of Phase 6 the engine is the whole stack:
the session, the native runtime (FFI), registration (raise + elaborate
+ strata + entry rendering), the compiler (Core downcast + partition +
plan assembly), the v2 ingest (load/merge), and save/export. There is
no compiler-service subprocess.

Every graph mutation ends in `syncCompile`: the mirror lowers,
elaborates, downcasts, partitions, and the plan hot-swaps into the
Lean-owned runtime. State mutations precede the compile (matching TS:
a failed compile leaves the mutated graph in place; the previous
kernel keeps playing and the error is recoverable).
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

structure Env where
  state   : IO.Ref SessionSt
  /-- The native FlatRuntime this engine owns: plans load here; the DAC
      reads from it; param slots are driven on it. -/
  runtime : Ffi.Runtime
  dac     : IO.Ref (Option Ffi.Dac)

-- Reserved audio-output boundary leaf.
private def dacName : String := "dac"
private def dacOut : String := "out"
private def scopeName : String := "scope"

-- ── Json access helpers ──────────────────────────────────────────────────────

/-- Field access with `??` semantics: absent and `null` are both `none`. -/
private def arg? (args : Json) (k : String) : Option Json :=
  match getField? args k with
  | some .null => none
  | v => v

private def argStr? (args : Json) (k : String) : Option String :=
  match arg? args k with
  | some (.str s) => some s
  | _ => none

private def argArr (args : Json) (k : String) : Array Json :=
  match arg? args k with
  | some (.arr a) => a
  | _ => #[]

private def jsonNull : Json := Json.null

/-- Render a Json value the way a TS template literal renders it
    (`${x}`): bare strings unquoted, everything else as JSON. -/
private def tsInterp : Json → String
  | .str s => s
  | j => j.compress

-- ── Lookup helpers (ported failure shapes) ───────────────────────────────────

private def requireInstance (st : SessionSt) (name : String) (param : String) : EngineM InstanceInfo :=
  match st.findInstance? name with
  | some info => pure info
  | none => throwEnum .unknownInstance param (Json.str name) st.instanceNames

/-- Output name-or-index → index. Numbers (and digit strings) pass
    through unchecked, mirroring TS. -/
private def resolveOutputIdx (pm : ProgMeta) (nameOrIdx : Json) : EngineM Nat :=
  match nameOrIdx with
  | .num n => pure n.toFloat.toUInt64.toNat
  | .str s =>
    if s.toList.all Char.isDigit && !s.isEmpty then
      pure s.toNat!
    else
      match pm.outputNames.idxOf? s with
      | some i => pure i
      | none => throwEnum .unknownOutput "output" (Json.str s) pm.outputNames
  | j => throwEnum .unknownOutput "output" j pm.outputNames

private def resolveInputIdx (pm : ProgMeta) (nameOrIdx : Json) : EngineM Nat :=
  match nameOrIdx with
  | .num n => pure n.toFloat.toUInt64.toNat
  | .str s =>
    if s.toList.all Char.isDigit && !s.isEmpty then
      pure s.toNat!
    else
      match pm.inputNames.idxOf? s with
      | some i => pure i
      | none => throwEnum .unknownInput "input" (Json.str s) pm.inputNames
  | j => throwEnum .unknownInput "input" j pm.inputNames

private def resolveOutputName (pm : ProgMeta) (nameOrIdx : Json) : EngineM String := do
  let idx ← resolveOutputIdx pm nameOrIdx
  pure <| (pm.outputNames[idx]?).getD (toString idx)

private def resolveInputName (pm : ProgMeta) (nameOrIdx : Json) : EngineM String := do
  let idx ← resolveInputIdx pm nameOrIdx
  pure <| (pm.inputNames[idx]?).getD (toString idx)

private def inputTypeObj (pm : ProgMeta) (idx : Nat) : Option Json :=
  (pm.inputs[idx]?).bind (·.typeObj)

-- ── adaptInputExpr (type-check + auto-broadcast) ─────────────────────────────

private def scalarTypeJson (k : String) : Json :=
  Json.mkObj [("kind", Json.str "scalar"), ("scalar", Json.str k)]

/-- Infer a source port type. Returns both the parsed form (for the
    connection check) and the structured PortType Json (echoed verbatim
    as the `got` field of `type_mismatch` envelopes — TS passes the
    PortType object itself). -/
private def srcTypeOf (st : SessionSt) (node : Json) : Option (PortType × Json) :=
  match node with
  | .num _  => some (.scalar .float, scalarTypeJson "float")
  | .bool _ => some (.scalar .bool, scalarTypeJson "bool")
  | .arr items =>
    some (.array { display := "float", kind := some .float } #[items.size],
          Json.mkObj [("kind", Json.str "array"), ("element", Json.str "float"),
                      ("shape", Json.arr #[Lean.toJson items.size])])
  | .obj _ =>
    if opOf? node == some "ref" then do
      let instName ← getStrField? node "instance"
      let info ← st.findInstance? instName
      let outIdx ← match getField? node "output" with
        | some (.num n) => some n.toFloat.toUInt64.toNat
        | some (.str s) => info.progMeta.outputNames.idxOf? s
        | _ => none
      let port ← info.progMeta.outputs[outIdx]?
      let typeObj ← port.typeObj
      let parsed ← parsePortType? typeObj
      pure (parsed, typeObj)
    else none
  | _ => none

private def adaptInputExpr (st : SessionSt) (node : Json) (dstTypeObj : Option Json)
    (instanceName inputName : String) : EngineM Json := do
  match srcTypeOf st node with
  | none => pure node
  | some (srcType, srcTypeJson) =>
    let dstType := dstTypeObj.bind parsePortType?
    let check := checkArrayConnection (some srcType) dstType node
    if !check.compatible then
      throwPredicate .typeMismatch "expr" node "type_compatible"
        dstTypeObj (some srcTypeJson)
        (some s!"Type mismatch on '{instanceName}'.{inputName}: {check.error.getD ""}")
    pure (check.broadcastExpr.getD node)

-- ── Typed-store adoption (Phase 4 stage 4a) ──────────────────────────────────

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

-- ── Engine-side strata (Phase 5 stage 6b) ────────────────────────────────────

/-- The service session's `inlineNested` (makeSession defaults it to
    `true`; nothing on the MCP path overrides it). Threaded into every
    engine-side strata run so the two sides can never disagree about
    which realization the registered catalog holds. -/
private def sessionInlineNested : Bool := true

/-- The strata pipeline + the post-strata Core downcast (inline path),
    as a pure `Except` so call sites choose the envelope: registration
    failures map to `internal_error` (TS strata throws were plain
    Errors), specialization failures to `invalid_type_args` (the
    engine mapped any service `resolve_type` failure that way before
    the move). A Core-check failure is a port bug, surfaced loudly. -/
def runStrataChecked (typeArgs : Array (String × Lean.JsonNumber))
    (arena : Tropical.Ir.Arena) (rootIdx : Tropical.Ir.ProgramIdx) :
    Except String (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let (arena, rootIdx) ←
    (Tropical.Ir.Strata.run
      { upto := Tropical.Ir.Strata.portedPasses,
        inlineNested := sessionInlineNested, typeArgs } arena rootIdx).mapError (·.message)
  if sessionInlineNested then
    if let .error e := Tropical.Ir.checkResolvedArena arena rootIdx then
      throw s!"post-strata Core check failed (port bug): {e}"
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
    let (prog, sortedRefs) ← match Tropical.Ir.WireProgram.lift w.expr synthName with
      | .error msg => internalError msg
      | .ok r => pure r
    -- The raw lifted program joins the store: templateByName mirrors
    -- TS `session.programs.set(name, lifted)` — the RAW form, which
    -- is what a later registration's relink byName would see.
    let arenaRaw := { st.arena with programs := st.arena.programs.push prog }
    let rawIdx : Tropical.Ir.ProgramIdx := ⟨st.arena.programs.size⟩
    let rawJson ← match Tropical.Ir.Codec.encodeResolved arenaRaw rawIdx with
      | .error e => internalError e
      | .ok j => pure j
    -- Strata, no relink (lifted bodies have no InstanceDecls — the
    -- registry is empty, exactly the TS lift's `programRegistry: new
    -- Map()`).
    let (arenaPost, postIdx) ← match runStrataChecked #[] arenaRaw rawIdx with
      | .error msg => internalError msg
      | .ok r => pure r
    let postJson ← match Tropical.Ir.Codec.encodeResolved arenaPost postIdx with
      | .error e => internalError e
      | .ok j => pure j
    -- Persist the raw program + counter; the post-strata arena growth
    -- is transient — the store adopts the service entry's round trip,
    -- one copy, like the registration path.
    env.state.modify fun s =>
      { s with arena := arenaRaw,
               nameCounters := s.nameCounters.insert "__wire" counter,
               templateByName := s.templateByName.insert synthName rawIdx }
    let _ := rawJson  -- the raw form's only consumer was the service residue
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
-- Session → resolved root DIRECTLY — the C4 cutover (no parsed round-trip)
-- ─────────────────────────────────────────────────────────────

/-! `sessionToParsed → reparse → elaborate` takes the already-resolved session
    instances, serializes them into a NAMED `__session__` ParsedProgram, and
    re-elaborates the names back to pointers — the "resolved → named → resolved"
    round-trip. `sessionToResolvedRoot` deletes it: the session graph is already
    post-elaborate-shaped (instances carry resolved type snapshots; wires are
    graph edges), so it builds the resolved root `Program` DIRECTLY, reproducing
    the elaborator's output byte-for-byte (gated `tropical_resolved_1`-identical
    against the elaborate path on every golden). `elaborate` stays the reifier
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
  /-- `(instanceName, outputName)` → `nestedOut ⟨instIdx⟩ ⟨outputIdx⟩`. -/
  instOut : String → String → Except String Tropical.Ir.Expr
  /-- param/trigger name → `ParamIdx` (alphabetical position). -/
  paramIdx : String → Option Nat

/-- Resolve a raw session wire expression directly to a resolved `Ir.Expr`,
    mirroring the elaborator's `resolveExpr` over a session-root scope. Same op
    set as `wireExprToParsed`; no parsed intermediate. -/
private partial def wireExprToResolved (ctx : WireCtx) (expr : Json) :
    Except String Tropical.Ir.Expr :=
  match expr with
  | .num n => .ok (.num n)
  | .bool b => .ok (.bool b)
  | .arr items => do pure (.arr (← items.mapM (wireExprToResolved ctx)))
  | .obj _ => do
    let some op := opOf? expr
      | .error s!"session wire node missing op: {expr.compress}"
    let rawArgs := match getField? expr "args" with | some (.arr a) => a | _ => #[]
    let args ← rawArgs.mapM (wireExprToResolved ctx)
    if op == "ref" then
      ctx.instOut ((getStrField? expr "instance").getD "") ((getStrField? expr "output").getD "")
    else if op == "param" || op == "trigger" then
      let name := (getStrField? expr "name").getD ""
      match ctx.paramIdx name with
      | some i => .ok (.paramRef ⟨i⟩)
      | none => .error s!"session wire: param '{name}' not in root param table"
    else if op == "array" then
      match getField? expr "items" with
      | some (.arr items) => do pure (.arr (← items.mapM (wireExprToResolved ctx)))
      | _ => .error "session wire: 'array' missing items"
    else if op == "sampleRate" then .ok .sampleRate
    else if op == "sampleIndex" then .ok .sampleIndex
    else if op == "clock" || op == "sampleClock" || op == "sample_clock" then
      .ok (.binary .lshift .sampleIndex (.num ⟨32, 0⟩))
    else if op == "clamp" then
      if args.size == 3 then .ok (.clamp args[0]! args[1]! args[2]!)
      else .error s!"session wire: clamp expects 3 args, got {args.size}"
    else if op == "select" then
      if args.size == 3 then .ok (.select args[0]! args[1]! args[2]!)
      else .error s!"session wire: select expects 3 args, got {args.size}"
    else if op == "arraySet" || op == "array_set" then
      if args.size == 3 then .ok (.arraySet args[0]! args[1]! args[2]!)
      else .error s!"session wire: arraySet expects 3 args, got {args.size}"
    else if op == "index" then
      if args.size == 2 then .ok (.index args[0]! args[1]!)
      else .error s!"session wire: index expects 2 args, got {args.size}"
    else if op == "zeros" then
      if args.size == 1 then .ok (.zeros args[0]!)
      else .error s!"session wire: zeros expects 1 arg, got {args.size}"
    else if let some tag := Tropical.Ir.BinaryOpTag.ofWire? op then
      if args.size == 2 then .ok (.binary tag args[0]! args[1]!)
      else .error s!"session wire: binary '{op}' expects 2 args, got {args.size}"
    else if let some tag := Tropical.Ir.UnaryOpTag.ofWire? op then
      if args.size == 1 then .ok (.unary tag args[0]!)
      else .error s!"session wire: unary '{op}' expects 1 arg, got {args.size}"
    else
      .error s!"session wire: unsupported op '{op}'"
  | _ => .error s!"session wire: invalid value {expr.compress}"

/-- Build the resolved session root `Program` directly from the graph (instances
    already carry resolved type snapshots via `resolverTbl`), deleting the
    `sessionToParsed → reparse → elaborate` round-trip. Byte-identical to the
    elaborated root (gated). `lowerInstances` carries each instance's STORED
    program name (the resolver key). -/
private def sessionToResolvedRoot (arena : Tropical.Ir.Arena)
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
            | some o => .ok (Tropical.Ir.Expr.nestedOut ⟨idx⟩ ⟨o⟩)
  let ctx : WireCtx := { instOut, paramIdx := fun nm => sortedParams.findIdx? (· == nm) }
  -- instance decls (topo order), then param decls (alphabetical).
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
      let value ← wireExprToResolved ctx w.expr
      inputs := inputs.push { port := ⟨pos⟩, value }
    decls := decls.push (.inst name tgt.name #[] inputs)
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
  .ok ({ arena with programs := arena.programs.push prog }, idx)

/-- The session lowering + compile (Phase 3 lowering, Phase 4 stage 4a
    elaboration): lift if needed, then run the Lean lowering — slot
    allocation, delay extraction, acyclicity, serialization to a
    ParsedProgram — elaborate the root over the typed store (LINKing
    each instance's `resolvedIdx` snapshot through the resolver), ship
    the encoded `tropical_resolved_1` root with the slot bookkeeping to
    the service for decode + partition, and hot-swap the returned plan
    into the Lean-owned runtime. -/
def syncCompile (env : Env) : EngineM Unit := do
  liftIfNeeded env
  let st ← env.state.get

  -- Lowering (pure over the mirror; the mirror itself stays canonical
  -- pre-extraction).
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

  -- EXPERIMENT (TROPICAL_ARROW, uncommitted): build the resolved session
  -- root directly via `sessionToResolvedRoot` — the session → arrow path,
  -- deleting the "resolved → named → resolved" parsed round-trip. Plan is
  -- byte-identical to the elaborate path (gated by `runSessionViaArrowEquiv`
  -- in Tropicaltest). With the var unset the legacy elaborate path runs
  -- unchanged, so default behavior is untouched.
  let (arena', rootIdx) ← if (← IO.getEnv "TROPICAL_ARROW").isSome then
    match sessionToResolvedRoot st.arena lowerInstances wiresPost tbl with
    | .error e => internalError e
    | .ok r => pure r
  else do
    -- Legacy: session → parsed → reparse (Json → ordered JsonV, lossless)
    -- → elaborate over the store arena. The appended root is transient.
    let parsed ← Tropical.Lowering.sessionToParsed lowerInstances wiresPost
    let typed ← match Tropical.Parse.JsonV.parse parsed.compress with
      | .error e => internalError s!"session root: ParsedProgram JSON re-parse failed: {e}"
      | .ok jv =>
        match Tropical.Parse.decodeProgram jv with
        | .error e => internalError s!"session root: {e}"
        | .ok p => pure p
    -- Failure maps onto the recoverable envelope (ElaborationError /
    -- CycleViolation): the previous kernel keeps playing.
    match Tropical.Ir.elaborateInto st.arena typed
        (some fun n => (tbl.find? (·.1 == n)).map (·.2)) with
    | .error e => internalError e.message
    | .ok r => pure r
  let (rootArena, rootCore) ← match Tropical.Ir.checkResolvedArena arena' rootIdx with
    | .error e => internalError s!"syncCompile: post-elaboration Core check failed (engine bug): {e}"
    | .ok r => pure r

  -- Session instances in registry order, each materialized as the Core
  -- form the root's registry linked (first-instance-wins per stored
  -- program name — the sessionTypeResolver contract).
  let mut coreInstances : Array (String × Tropical.Ir.Core.CoreProgram) := #[]
  for (n, i) in st.instances do
    let some pname := storedProgName i
      | internalError s!"syncCompile: instance '{n}' has no resolved snapshot (engine bug)"
    let some core := rootCore.registryGet? pname
      | internalError s!"syncCompile: instance '{n}' program '{pname}' missing from root registry (engine bug)"
    coreInstances := coreInstances.push (n, core)

  let plan ← match Tropical.Compile.compileSession {
      instances := coreInstances
      wiresPost
      graphOutputs := st.graphOutputs
      params := st.params
      alloc
      root := rootCore
      arena := rootArena } with
    | .error msg => internalError msg
    | .ok p => pure p
  -- Lean owns codegen: emit LLVM IR from the in-memory plan and hand it to
  -- the engine (planJson is the metadata manifest). No C++ plan compiler.
  let (ir, planJson) ← buildKernelIr plan
  env.runtime.loadIr ir planJson

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
def compileMirrorFlatPlan (env : Env) (mode : Tropical.Plan.CompilationMode) :
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
  let plan ← match Tropical.Compile.compileSession {
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
  pure plan

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

-- ── Catalog adoption ─────────────────────────────────────────────────────────

def adoptEntries (env : Env) (entries : Json) : EngineM Unit := do
  match entries with
  | .arr es =>
    for e in es do
      env.state.modify (·.addProgram (ProgMeta.fromEntry e))
      let _ ← adoptResolved env e
  | _ => pure ()

-- ── Program registration (Phase 4 stage 4b; strata engine-side at 6b) ────────
-- The engine runs the stage-2 raise, the elaboration, AND (Phase 5
-- stage 6b) the strata pipeline itself; the service's
-- `register_program` shrinks to typeDef registration + decode +
-- `makeCompiled`, one call per batch item.

/-- Run the strata pipeline on an elaborated concrete program — the
    engine-side image of the service residue stage 6b retired:
    relink sub-program registry entries to the canonical post-strata
    registrations (`concreteByName` over TS `session.programs` =
    `templateByName` restricted to concrete entries; load-bearing on
    the boot path, where elaboration resolved against the raw stdlib
    map — a structural no-op on the define path), then run the full
    pipeline. -/
def strataConcrete (st : SessionSt) (arena : Tropical.Ir.Arena)
    (rootIdx : Tropical.Ir.ProgramIdx) :
    EngineM (Tropical.Ir.Arena × Tropical.Ir.ProgramIdx) := do
  let byName : String → Option Tropical.Ir.ProgramIdx := fun n =>
    (st.templateByName.get? n).filter fun idx =>
      match arena.program? idx with
      | some prog => prog.typeParams.isEmpty
      | none => false
  let (arena, rootIdx) ←
    match Tropical.Ir.Strata.relinkProgramRegistry arena rootIdx byName with
    | .error e => internalError e.message
    | .ok r => pure r
  match runStrataChecked #[] arena rootIdx with
  | .error msg => internalError msg
  | .ok r => pure r

private def renameProgram (p : Tropical.Parse.Program) (name : String) :
    Tropical.Parse.Program :=
  .mk name p.typeParams p.ports p.body p.breaksCycles

private def portNames (ps : Option (Array Tropical.Parse.ProgramPort)) : Array String :=
  (ps.getD #[]).map fun
    | .bare n => n
    | .spec s => s.name

/-- The registration batch for a def: nested programDecls depth-first
    in post-order (children before parents, source order), each renamed
    to its decl name; the root last. This is exactly the order
    `loadProgramAsType` registered them (it recursed into
    `{...sub.program, name: sub.name}` BEFORE registering the parent).
    Items keep their nested programDecls inline — the elaborator
    re-elaborates them in scope, as TS does. -/
private partial def registrationBatch (name : String) (p : Tropical.Parse.Program) :
    Array (String × Tropical.Parse.Program) := Id.run do
  let mut out : Array (String × Tropical.Parse.Program) := #[]
  for d in p.body.decls do
    if let .prog subName inner := d then
      out := out ++ registrationBatch subName inner
  return out.push (name, renameProgram p name)

/-- Register one program: elaborate it over the typed store, run the
    strata pipeline on it (concrete programs only — generics ship the
    raw template, which the service stores unstrata'd and never
    relinks), ship `{name, parsed, resolved}` to the service (typeDef
    registration + decode + `makeCompiled` + registry insert), and
    adopt the returned entry.

    Resolver: `templateByName` — the mirror of TS `session.programs`
    (post-strata for concrete, raw template for generics) — unless the
    caller supplies one (boot passes the raw stdlib map, mirroring
    `loadStdlibFromResolved`'s `localResolved`).

    Store discipline: for concrete programs the store adopts ONLY the
    service's post-strata round trip (the engine's raw elaboration is
    transient arena growth); for generics the engine's raw template IS
    the stored form, since the service ships `resolved: null` for
    generic entries. Returns the entry and the raw elaborated index. -/
def registerOne (env : Env) (name : String) (p : Tropical.Parse.Program)
    (resolver : Option Tropical.Ir.Resolver := none) :
    EngineM (Json × Tropical.Ir.ProgramIdx) := do
  let st ← env.state.get
  let res : Tropical.Ir.Resolver := match resolver with
    | some r => r
    | none => fun n => st.templateByName.get? n
  -- Elaboration failure → internal_error with the verbatim message
  -- (ElaborationError / CycleViolation are plain Errors in TS; the
  -- service's toEnvelope made them internal_error). Items registered
  -- before a mid-batch failure STAY registered — oracle behavior.
  let (arena', rawIdx) ← match Tropical.Ir.elaborateInto st.arena p (some res) with
    | .error e => internalError e.message
    | .ok r => pure r
  -- Phase 5 stage 6b: concrete programs ship POST-STRATA resolved.
  let isGeneric := !(p.typeParams.getD #[]).isEmpty
  let (arenaShip, shipIdx) ←
    if isGeneric then pure (arena', rawIdx)
    else strataConcrete st arena' rawIdx
  env.state.modify fun st => { st with arena := arenaShip }
  let entry ← if isGeneric then
      pure (Tropical.Entries.genericEntry arenaShip name shipIdx)
    else
      match Tropical.Entries.concreteEntry arenaShip name shipIdx with
      | .error e => internalError e
      | .ok j => pure j
  env.state.modify (·.addProgram (ProgMeta.fromEntry entry))
  let idx? ← adoptResolved env entry
  -- templateByName mirrors what TS `session.programs.set(name, ...)`
  -- stored: the decided-by-this-item form, NOT whatever entryFor
  -- echoed (a generic redefining a concrete name gets a STALE concrete
  -- entry from the service's typeRegistry — TS still stores the new
  -- generic template in session.programs).
  env.state.modify fun st => { st with
    templateByName := st.templateByName.insert name
      (if isGeneric then rawIdx else idx?.getD rawIdx) }
  pure (entry, rawIdx)

-- ── Per-tool handlers ────────────────────────────────────────────────────────

private def instanceSummary (st : SessionSt) (name : String) : Json :=
  match st.findInstance? name with
  | none => jsonNull
  | some info => Json.mkObj [
      ("name", Json.str name),
      ("type_name", Json.str info.baseTypeName),
      ("type_args", info.typeArgs.getD jsonNull),
      ("inputs", toJson info.progMeta.inputNames),
      ("outputs", toJson info.progMeta.outputNames)]

/-- Resolve a program name (+ optional type args) to instance metadata
    plus the typed-store snapshot for the resolved program, with the TS
    failure shapes. Generic programs specialize through the service's
    `resolve_type`, whose entry is adopted into the store; concrete
    programs take the store's current mapping for the name.

    `toolEnvelopes := false` selects the LOAD/MERGE ingest path's
    failure shapes — `resolveProgramType`'s plain TS Errors
    (`Unknown program type '…'. Known: …` etc.), which the service
    relay surfaced as `internal_error` with the verbatim message. -/
private def resolveInstanceMeta (env : Env) (programName : String)
    (typeArgs : Option Json) (programParam : String)
    (toolEnvelopes : Bool := true) :
    EngineM (Option Json × ProgMeta × Option Tropical.Ir.ProgramIdx) := do
  let st ← env.state.get
  match st.programs.get? programName with
  | none =>
    -- TS options: [...typeRegistry.keys(), ...programs.keys()] — concrete
    -- names first, then every program name (concrete ones repeat).
    let concrete := st.catalogOrder.filter fun n =>
      match st.programs.get? n with | some m => !m.generic | none => false
    if !toolEnvelopes then
      let known := String.intercalate ", " (concrete ++ st.catalogOrder).toList
      internalError s!"Unknown program type '{programName}'. Known: {if known.isEmpty then "(none)" else known}"
    throwEnum .unknownProgram programParam (Json.str programName)
      (concrete ++ st.catalogOrder)
  | some pm =>
    if pm.generic then
      -- Phase 5 stage 6b: type-arg resolution + the specialization
      -- (strata over the raw template, NO relink — TS
      -- `specializeFromResolvedTemplate` never relinked) run
      -- engine-side. Every failure — arg validation and strata alike —
      -- maps to invalid_type_args with the raw message, exactly how
      -- the engine mapped any service resolve_type failure before the
      -- move.
      let some templateIdx := st.templateByName.get? programName
        | internalError s!"resolve_type: generic '{programName}' has no stored template (engine bug)"
      let some template := st.arena.program? templateIdx
        | internalError s!"resolve_type: template pool index {templateIdx.idx} out of range (engine bug)"
      let declared ← template.typeParams.mapM fun i =>
        match st.arena.typeParam? i with
        | some tp => pure (tp.name, tp.default?)
        | none => internalError s!"resolve_type: typeParam pool index out of range (engine bug)"
      let resolvedArgs ←
        match Tropical.TypeArgs.resolve typeArgs declared s!"instance of '{programName}'" with
        | .error msg =>
          if !toolEnvelopes then internalError msg
          throwBare .invalidTypeArgs msg (param := some "type_args") (value := typeArgs)
        | .ok r => pure r
      let key := Tropical.TypeArgs.cacheKey programName resolvedArgs
      let echo := Json.mkObj (resolvedArgs.toList.map fun (n, v) => (n, Json.num v))
      match st.specializationCache.get? key with
      | some (pmCached, idx?) => pure (some echo, pmCached, idx?)
      | none =>
        let (arena', specIdx) ←
          match runStrataChecked resolvedArgs st.arena templateIdx with
          | .error msg =>
            if !toolEnvelopes then internalError msg
            throwBare .invalidTypeArgs msg (param := some "type_args") (value := typeArgs)
          | .ok r => pure r
        -- arena' (the pre-round-trip specialization) is deliberately
        -- NOT persisted: the store adopts the entry's codec round trip
        -- below, exactly one arena copy — the same growth the
        -- service-relay path had.
        let entry ← match Tropical.Entries.concreteEntry arena' key specIdx with
          | .error e => internalError e
          | .ok j => pure j
        let pmNew := ProgMeta.fromEntry entry
        let idx? ← adoptResolved env entry
        env.state.modify fun s =>
          { s with specializationCache := s.specializationCache.insert key (pmNew, idx?) }
        pure (some echo, pmNew, idx?)
    else
      match typeArgs with
      | some ta =>
        let keys := match ta with
          | .obj m => String.intercalate ", " (m.toList.map Prod.fst)
          | _ => ""
        if keys.isEmpty then pure (none, pm, st.resolvedByName.get? programName)
        else if !toolEnvelopes then
          internalError s!"Program '{programName}' does not declare type_params; got type_args: {keys}"
        else
          throwBare .invalidTypeArgs
            (s!"Program '{programName}' does not declare type_params; got type_args: {keys}")
            (param := some "type_args") (value := some ta)
      | none => pure (none, pm, st.resolvedByName.get? programName)

def handleDefineProgram (env : Env) (args : Json) : EngineM Json := do
  let def_ := (arg? args "def").getD jsonNull
  -- Bridge to the ordered decoder by re-parsing the compressed form.
  -- Lean Json objects are key-sorted, which is exactly what the relay
  -- shipped to the TS service before this stage — observable raise
  -- behavior is unchanged.
  let jv ← match Tropical.Parse.JsonV.parse def_.compress with
    | .error e => internalError s!"define_program: def JSON re-parse failed: {e}"
    | .ok v => pure v
  -- Stage-2 raise: normalizeProgramFile (schema-tag check + Zod-strip)
  -- + raiseProgram. Failures map to internal_error with the verbatim
  -- message — the envelope the TS path produced when its
  -- normalizeProgramFile / loadProgramAsType threw.
  let (prog, _top) ← match Tropical.Parse.Raise.raiseFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  -- One service call per batch item, adoption between items: a later
  -- item (the parent) thereby elaborates against the earlier item's
  -- POST-STRATA form, matching TS (the sub's registration round-trips
  -- through strata before the parent is processed).
  let batch := registrationBatch prog.name prog
  let mut rootEntry := jsonNull
  for (name, p) in batch do
    let (entry, _) ← registerOne env name p
    rootEntry := entry
  -- The TS handler's two result shapes.
  if (prog.typeParams.getD #[]).isEmpty then
    let names := fun (k : String) => Json.arr <|
      (match getField? rootEntry k with
       | some (.arr ps) => ps
       | _ => #[]).filterMap fun pj => (getStrField? pj "name").map Json.str
    pure <| Json.mkObj [
      ("program_name", (getField? rootEntry "program_name").getD (Json.str prog.name)),
      ("inputs", names "inputs"),
      ("outputs", names "outputs")]
  else
    pure <| Json.mkObj [
      ("program_name", Json.str prog.name),
      ("inputs", toJson (portNames (prog.ports.bind (·.inputs)))),
      ("outputs", toJson (portNames (prog.ports.bind (·.outputs)))),
      ("type_params", Tropical.Parse.Encode.typeParams (prog.typeParams.getD #[]))]

def handleAddInstance (env : Env) (args : Json) : EngineM Json := do
  let programName := (argStr? args "program").getD ""
  let instanceName := (argStr? args "instance_name").getD ""
  if instanceName == dacName || instanceName == scopeName then
    throwBare .invalidValue
      s!"'{instanceName}' is a reserved instance name ({dacName} = audio output, {scopeName} = inspection taps). Choose a different name."
      (param := some "instance_name") (value := some (Json.str instanceName))
  let st ← env.state.get
  if (st.findInstance? instanceName).isSome then
    throwBare .instanceExists s!"Instance '{instanceName}' already exists."
      (param := some "instance_name") (value := some (Json.str instanceName))
  let (typeArgs, pm, resolvedIdx) ← resolveInstanceMeta env programName (arg? args "type_args") "program"
  env.state.modify (·.addInstance instanceName
    { baseTypeName := programName, typeArgs, progMeta := pm, resolvedIdx })
  pure (instanceSummary (← env.state.get) instanceName)

def handleReplicate (env : Env) (args : Json) : EngineM Json := do
  let programName := (argStr? args "program").getD ""
  let countJ := (arg? args "count").getD jsonNull
  let count? : Option Nat := match countJ with
    | .num n => if n.toFloat == n.toFloat.floor && n.toFloat ≥ 1 then some n.toFloat.toUInt64.toNat else none
    | _ => none
  let some count := count?
    | throwRecord .invalidValue "count" countJ
        [("count", { type := "int", required := true, min := some 1.0 })]
        (some s!"count must be a positive integer, got {tsInterp countJ}")
  let namePrefix := argStr? args "name_prefix"
  let prefix' := namePrefix.getD programName.toLower
  if prefix' == dacName || prefix' == scopeName then
    throwBare .invalidValue
      s!"'{prefix'}' is a reserved instance name ({dacName} = audio output, {scopeName} = inspection taps). Choose a different name_prefix."
      (param := some "name_prefix") (value := some (Json.str prefix'))
  let mut created : Array Json := #[]
  for _ in [0:count] do
    let st ← env.state.get
    let (st', name) := st.nextName prefix'
    env.state.set st'
    if (st'.findInstance? name).isSome then
      throwBare .instanceExists
        s!"Instance '{name}' already exists — pick a different name_prefix"
        (param := some "name_prefix") (value := namePrefix.map Json.str)
    let (typeArgs, pm, resolvedIdx) ← resolveInstanceMeta env programName (arg? args "type_args") "program"
    env.state.modify (·.addInstance name { baseTypeName := programName, typeArgs, progMeta := pm, resolvedIdx })
    created := created.push (instanceSummary (← env.state.get) name)
  pure <| Json.mkObj [("created", Json.arr created)]

def handleRemoveInstance (env : Env) (args : Json) : EngineM Json := do
  let instanceName := (argStr? args "instance_name").getD ""
  let st ← env.state.get
  let _ ← requireInstance st instanceName "instance_name"
  env.state.modify fun st =>
    let st := st.removeInstance instanceName
    { st with
      wires := st.wires.filter fun w =>
        !(w.instName == instanceName || (exprDependencies w.expr).contains instanceName)
      graphOutputs := st.graphOutputs.filter (·.1 != instanceName)
      scopeTaps := st.scopeTaps.filter (·.2.1 != instanceName) }
  syncCompile env
  pure <| Json.mkObj [("removed", Json.str instanceName)]

def handleListPrograms (env : Env) : EngineM Json := do
  let st ← env.state.get
  let portJson (withDefault : Bool) (p : PortInfo) : Json :=
    Json.mkObj <|
      [("name", Json.str p.name),
       ("type", match p.typeStr with | some s => Json.str s | none => jsonNull)]
      ++ (if withDefault then [("default", p.default.getD jsonNull)] else [])
  let render (m : ProgMeta) : Json :=
    Json.mkObj [
      ("program_name", Json.str m.programName),
      ("inputs", Json.arr (m.inputs.map (portJson true))),
      ("outputs", Json.arr (m.outputs.map (portJson false))),
      ("registers", Json.arr (m.registers.map (portJson false))),
      ("type_params", if m.generic then m.typeParams.getD jsonNull else jsonNull)]
  let metas := st.catalogOrder.filterMap st.programs.get?
  let concrete := (metas.filter (!·.generic)).map render
  let generic := (metas.filter (·.generic)).map render
  pure <| Json.arr (concrete ++ generic)

def handleListInstances (env : Env) : EngineM Json := do
  let st ← env.state.get
  pure <| Json.arr (st.instances.map fun (n, _) => instanceSummary st n)

def handleGetInfo (env : Env) (args : Json) : EngineM Json := do
  let instanceName := (argStr? args "instance_name").getD ""
  let st ← env.state.get
  let info ← requireInstance st instanceName "instance_name"
  let lookupOutputs := fun n => (st.findInstance? n).map (·.progMeta.outputNames)
  let inputs := info.progMeta.inputs.mapIdx fun i p =>
    let wire := st.findWire? instanceName p.name
    Json.mkObj [
      ("name", Json.str p.name), ("index", toJson i),
      ("type", p.typeObj.getD jsonNull),
      ("expr", match wire with | some w => w.expr | none => jsonNull),
      ("pretty", match wire with
        | some w => Json.str (prettyExpr w.expr lookupOutputs)
        | none => jsonNull)]
  let outputs := info.progMeta.outputs.mapIdx fun i p =>
    Json.mkObj [("name", Json.str p.name), ("index", toJson i),
                ("type", p.typeObj.getD jsonNull)]
  let registers := info.progMeta.registers.mapIdx fun i p =>
    Json.mkObj [("name", Json.str p.name), ("index", toJson i),
                ("type", p.typeObj.getD jsonNull)]
  pure <| Json.mkObj [
    ("name", Json.str instanceName),
    ("program", Json.str info.baseTypeName),
    ("type_args", info.typeArgs.getD jsonNull),
    ("inputs", Json.arr inputs),
    ("outputs", Json.arr outputs),
    ("registers", Json.arr registers)]

-- ── wire (the unified mutation tool) ─────────────────────────────────────────

private def resolveDacSource (st : SessionSt) (expr : Json) : EngineM (String × String) := do
  let isRefObj := match expr with
    | .obj _ => true
    | _ => false
  if !isRefObj then
    throwBare .invalidValue
      s!"dac.{dacOut} requires a ref-shaped expression (use refExpr or \{op:'ref',instance,output}). Got literal/array."
      (param := some "expr") (value := some expr)
  if opOf? expr != some "ref" then
    throwBare .invalidValue
      s!"dac.{dacOut} requires expr.op === 'ref'. Got op='{(opOf? expr).getD "undefined"}'."
      (param := some "expr") (value := some expr)
  let instJ := getField? expr "instance"
  let some (Json.str instName) := instJ
    | throwBare .invalidValue s!"dac.{dacOut}: ref.instance must be a string"
        (param := some "instance") (value := instJ)
  let info ← requireInstance st instName "instance"
  let outNames := info.progMeta.outputNames
  match getField? expr "output" with
  | some (.num n) =>
    let idx := n.toFloat.toUInt64.toNat
    if n.toFloat < 0 || idx ≥ outNames.size then
      throwEnum .unknownOutput "output" (Json.num n) outNames
    pure (instName, outNames[idx]!)
  | some (.str s) =>
    if !outNames.contains s then
      throwEnum .unknownOutput "output" (Json.str s) outNames
    pure (instName, s)
  | other =>
    throwBare .invalidValue s!"dac.{dacOut}: ref.output must be a number or string"
      (param := some "output") (value := other)

private def validateOrInternal (expr : Json) (path : String) : EngineM Unit :=
  match validateExpr expr path with
  | .ok () => pure ()
  | .error msg => internalError msg

def handleWire (env : Env) (args : Json) : EngineM Json := do
  let setOps := argArr args "set"
  let removeOps := argArr args "remove"

  -- Removes first
  let mut dacRemoved := 0
  for r in removeOps do
    let rInst := (argStr? r "instance").getD ""
    let rInput := (getField? r "input").getD jsonNull
    if rInst == dacName then
      if rInput != Json.str dacOut then
        throwBare .unknownOutput
          s!"dac has only one output port: '{dacOut}'. Got '{tsInterp rInput}'."
          (param := some "remove[].input") (value := some rInput)
      let st ← env.state.get
      dacRemoved := dacRemoved + st.graphOutputs.size
      env.state.modify fun st => { st with graphOutputs := #[] }
    else
      let st ← env.state.get
      let info ← requireInstance st rInst "remove[].instance"
      let inputId ← resolveInputIdx info.progMeta rInput
      let resolvedName := (info.progMeta.inputNames[inputId]?).getD (toString inputId)
      env.state.modify (·.removeWire rInst resolvedName)

  -- Sets
  let mut results : Array Json := #[]
  let mut dacWires : Array Json := #[]
  let mut scopeWires : Array Json := #[]
  for s in setOps do
    let sInst := (argStr? s "instance").getD ""
    let sInput := (getField? s "input").getD jsonNull
    let sExpr := (getField? s "expr").getD jsonNull
    if sInst == dacName then
      if sInput != Json.str dacOut then
        throwBare .unknownOutput
          s!"dac has only one output port: '{dacOut}'. Got '{tsInterp sInput}'."
          (param := some "set[].input") (value := some sInput)
      validateOrInternal sExpr s!"{dacName}.{dacOut}"
      let st ← env.state.get
      let (srcInst, srcOut) ← resolveDacSource st sExpr
      env.state.modify fun st =>
        { st with graphOutputs := st.graphOutputs.push (srcInst, srcOut) }
      dacWires := dacWires.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", sInput), ("expr", sExpr)]
    else if sInst == scopeName then
      let tapName := (argStr? s "input").getD ""
      if tapName.isEmpty then
        throwBare .invalidValue
          s!"scope tap requires a string input name ({scopeName}.<name>)."
          (param := some "set[].input") (value := some sInput)
      validateOrInternal sExpr s!"{scopeName}.{tapName}"
      let st ← env.state.get
      let (srcInst, srcOut) ← resolveDacSource st sExpr
      env.state.modify fun st =>
        { st with scopeTaps := (st.scopeTaps.filter (·.1 != tapName)).push (tapName, srcInst, srcOut) }
      scopeWires := scopeWires.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", sInput), ("expr", sExpr)]
    else
      let st ← env.state.get
      let info ← requireInstance st sInst "set[].instance"
      let inputId ← resolveInputIdx info.progMeta sInput
      let resolvedName := (info.progMeta.inputNames[inputId]?).getD (toString inputId)
      validateOrInternal sExpr s!"{sInst}.{resolvedName}"
      let adapted ← adaptInputExpr st sExpr (inputTypeObj info.progMeta inputId) sInst resolvedName
      let existing := st.findWire? sInst resolvedName
      let toStore := match existing, argStr? s "combine" with
        | some w, some combine =>
          Json.mkObj [("op", Json.str combine),
                      ("args", Json.arr #[w.expr, adapted])]
        | _, _ => adapted
      env.state.modify (·.setWireRaw sInst resolvedName toStore)
      results := results.push <| Json.mkObj
        [("instance", Json.str sInst), ("input", Json.str resolvedName), ("expr", toStore)]

  syncCompile env
  pure <| Json.mkObj <|
    [("set", Json.arr results)]
    ++ (if dacWires.isEmpty then [] else [("dac", Json.arr dacWires)])
    ++ (if scopeWires.isEmpty then [] else [("scope", Json.arr scopeWires)])
    ++ [("removed", toJson removeOps.size)]
    ++ (if dacRemoved > 0 then [("dacRemoved", toJson dacRemoved)] else [])

-- ── Wiring conveniences ──────────────────────────────────────────────────────

def handleWireChain (env : Env) (args : Json) : EngineM Json := do
  let instanceNames := (argArr args "instances").filterMap fun j =>
    match j with | .str s => some s | _ => none
  let outputPort := (getField? args "output").getD jsonNull
  let inputPort := (getField? args "input").getD jsonNull
  let initialExpr := arg? args "initial_expr"

  if instanceNames.size < 2 && initialExpr.isNone then
    throwBare .arityError
      "wire_chain needs at least 2 instances, or 1 instance with initial_expr"
      (param := some "instances") (value := some (toJson instanceNames))

  let st ← env.state.get
  let mut insts : Array InstanceInfo := #[]
  for n in instanceNames do
    insts := insts.push (← requireInstance st n "instances")

  if let some initial := initialExpr then
    let firstName := instanceNames[0]!
    let firstInst := insts[0]!
    let inputName ← resolveInputName firstInst.progMeta inputPort
    let idx := (firstInst.progMeta.inputNames.idxOf? inputName).getD 0
    let expr ← adaptInputExpr st initial (inputTypeObj firstInst.progMeta idx) firstName inputName
    env.state.modify (·.setWireRaw firstName inputName expr)

  let mut linked : Array Json := #[]
  for i in [0:instanceNames.size - 1] do
    let srcInst := insts[i]!
    let dstInst := insts[i+1]!
    let srcName := instanceNames[i]!
    let dstName := instanceNames[i+1]!
    let outName ← resolveOutputName srcInst.progMeta outputPort
    let inName ← resolveInputName dstInst.progMeta inputPort
    let refExpr := Json.mkObj [("op", Json.str "ref"),
      ("instance", Json.str srcName), ("output", Json.str outName)]
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let st' ← env.state.get
    let expr ← adaptInputExpr st' refExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWireRaw dstName inName expr)
    linked := linked.push (Json.str s!"{srcName}.{outName} → {dstName}.{inName}")

  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleWireZip (env : Env) (args : Json) : EngineM Json := do
  let sources := argArr args "sources"
  let targets := argArr args "targets"
  if sources.size != targets.size then
    throwBare .lengthMismatch
      s!"sources and targets must be the same length (got {sources.size} vs {targets.size})"
      (param := some "sources")
      (value := some (Json.mkObj [("sources", toJson sources.size), ("targets", toJson targets.size)]))
  let mut linked : Array Json := #[]
  for i in [0:sources.size] do
    let src := sources[i]!
    let dst := targets[i]!
    let srcName := (argStr? src "instance").getD ""
    let dstName := (argStr? dst "instance").getD ""
    let st ← env.state.get
    let srcInst ← requireInstance st srcName "sources[].instance"
    let dstInst ← requireInstance st dstName "targets[].instance"
    let outName ← resolveOutputName srcInst.progMeta ((getField? src "output").getD jsonNull)
    let inName ← resolveInputName dstInst.progMeta ((getField? dst "input").getD jsonNull)
    let refExpr := Json.mkObj [("op", Json.str "ref"),
      ("instance", Json.str srcName), ("output", Json.str outName)]
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st refExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWireRaw dstName inName expr)
    linked := linked.push (Json.str s!"{srcName}.{outName} → {dstName}.{inName}")
  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleFanOut (env : Env) (args : Json) : EngineM Json := do
  let rawSource := (getField? args "source").getD jsonNull
  let targets := argArr args "targets"
  let st ← env.state.get

  let isPortRef := match rawSource with
    | .obj _ => (getStrField? rawSource "instance").isSome
                && (getField? rawSource "output").isSome
                && (opOf? rawSource).isNone
    | _ => false
  -- TS checks `typeof rawSource.instance === 'string' && rawSource.output !== undefined`
  -- regardless of an `op` field; a `ref` ExprNode matches the port-ref arm too.
  let isPortRefTS := match rawSource with
    | .obj _ => (getStrField? rawSource "instance").isSome && (getField? rawSource "output").isSome
    | _ => false

  let (sourceExpr, sourceLabel) ←
    if isPortRefTS then do
      let sName := (getStrField? rawSource "instance").getD ""
      let srcInst ← requireInstance st sName "source.instance"
      let outName ← resolveOutputName srcInst.progMeta ((getField? rawSource "output").getD jsonNull)
      pure (Json.mkObj [("op", Json.str "ref"),
              ("instance", Json.str sName), ("output", Json.str outName)],
            s!"{sName}.{outName}")
    else
      pure (rawSource, rawSource.compress)
  let _ := isPortRef

  let mut linked : Array Json := #[]
  for dst in targets do
    let dstName := (argStr? dst "instance").getD ""
    let st' ← env.state.get
    let dstInst ← requireInstance st' dstName "targets[].instance"
    let inName ← resolveInputName dstInst.progMeta ((getField? dst "input").getD jsonNull)
    let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
    let expr ← adaptInputExpr st' sourceExpr (inputTypeObj dstInst.progMeta idx) dstName inName
    env.state.modify (·.setWireRaw dstName inName expr)
    linked := linked.push (Json.str s!"{sourceLabel} → {dstName}.{inName}")
  syncCompile env
  pure <| Json.mkObj [("linked", Json.arr linked)]

def handleFanIn (env : Env) (args : Json) : EngineM Json := do
  let sources := argArr args "sources"
  let target := (getField? args "target").getD jsonNull
  if sources.isEmpty then
    throwBare .arityError "sources must be non-empty"
      (param := some "sources") (value := some (Json.arr sources))
  let st ← env.state.get
  let targetName := (argStr? target "instance").getD ""
  let dstInst ← requireInstance st targetName "target.instance"

  let mut terms : Array Json := #[]
  for src in sources do
    let srcName := (argStr? src "instance").getD ""
    let srcInst ← requireInstance st srcName "sources[].instance"
    let outName ← resolveOutputName srcInst.progMeta ((getField? src "output").getD jsonNull)
    let ref := Json.mkObj [("op", Json.str "ref"),
      ("instance", Json.str srcName), ("output", Json.str outName)]
    terms := terms.push <| match getField? src "gain" with
      | some g@(.num _) => Json.mkObj [("op", Json.str "mul"), ("args", Json.arr #[ref, g])]
      | _ => ref

  let sumExpr := terms[1:].foldl
    (fun acc t => Json.mkObj [("op", Json.str "add"), ("args", Json.arr #[acc, t])])
    terms[0]!

  let inName ← resolveInputName dstInst.progMeta ((getField? target "input").getD jsonNull)
  let idx := (dstInst.progMeta.inputNames.idxOf? inName).getD 0
  let expr ← adaptInputExpr st sumExpr (inputTypeObj dstInst.progMeta idx) targetName inName
  env.state.modify (·.setWireRaw targetName inName expr)
  syncCompile env
  pure <| Json.mkObj [("mixed", toJson sources.size),
                      ("target", Json.str s!"{targetName}.{inName}")]

def handleListWiring (env : Env) (args : Json) : EngineM Json := do
  let filter := argStr? args "instance"
  let st ← env.state.get
  let lookupOutputs := fun n => (st.findInstance? n).map (·.progMeta.outputNames)
  let results := st.wires.filterMap fun w =>
    if let some f := filter then
      if w.instName != f then none else
      some (Json.mkObj [("instance", Json.str w.instName), ("input", Json.str w.portName),
                        ("expr", Json.str (prettyExpr w.expr lookupOutputs))])
    else
      some (Json.mkObj [("instance", Json.str w.instName), ("input", Json.str w.portName),
                        ("expr", Json.str (prettyExpr w.expr lookupOutputs))])
  pure (Json.arr results)

-- ── Program I/O (engine-side as of Phase 6 stage 6e) ────────────────────────

def handleSave (env : Env) : EngineM Json := do
  let st ← env.state.get
  -- CF-only: wires are stored raw (no session-level unit delay wrap), so
  -- save echoes the canonical wires directly.
  let wiresPost := st.wires
  let mut decls : Array Json := #[]
  -- paramDecls first, so they're declared before referencing instances.
  for (name, value) in st.params do
    decls := decls.push <| Json.mkObj
      [("op", Json.str "paramDecl"), ("name", Json.str name), ("value", value)]
  for (name, info) in st.instances do
    let mut inputs : Array (String × Json) := #[]
    for portName in info.progMeta.inputNames do
      if let some w := wiresPost.find? fun w => w.instName == name && w.portName == portName then
        inputs := inputs.push (portName, w.expr)
    decls := decls.push <| Json.mkObj <|
      [("op", Json.str "instanceDecl"), ("name", Json.str name),
       ("program", Json.str info.baseTypeName)]
      ++ (match info.typeArgs with | some ta => [("type_args", ta)] | none => [])
      ++ (if inputs.isEmpty then [] else [("inputs", Json.mkObj inputs.toList)])
  let mut assigns : Array Json := #[]
  for (inst, output) in st.graphOutputs do
    if let some info := st.findInstance? inst then
      if let some idx := info.progMeta.outputNames.idxOf? output then
        assigns := assigns.push <| Json.mkObj
          [("op", Json.str "outputAssign"), ("name", Json.str "dac.out"),
           ("expr", Json.mkObj [("op", Json.str "ref"), ("instance", Json.str inst),
             ("output", toJson idx)])]
  let body := Json.mkObj <|
    [("op", Json.str "block"), ("decls", Json.arr decls)]
    ++ (if assigns.isEmpty then [] else [("assigns", Json.arr assigns)])
  pure <| Json.mkObj [("program", Json.mkObj
    [("schema", Json.str "tropical_program_2"), ("name", Json.str "patch"),
     ("body", body)])]

/-- Port of `rewriteRefs`: session refs to internal instances become
    `nestedOut`; recursion follows `args` only (the TS shape). -/
private partial def rewriteRefs (reachable : Array String) (node : Json) : Json :=
  match node with
  | .arr items => .arr (items.map (rewriteRefs reachable))
  | .obj fields =>
    let isInternalRef :=
      opOf? node == some "ref" &&
      (match getStrField? node "instance" with
       | some i => reachable.contains i
       | none => false)
    if isInternalRef then
      Json.mkObj [("op", Json.str "nestedOut"),
        ("ref", Json.str ((getStrField? node "instance").getD "")),
        ("output", (getField? node "output").getD jsonNull)]
    else
      match getField? node "args" with
      | some (.arr items) => Id.run do
        let mut out : List (String × Json) := []
        for (k, v) in fields.toArray do
          if k == "args" then
            out := out ++ [(k, Json.arr (items.map (rewriteRefs reachable)))]
          else out := out ++ [(k, v)]
        return Json.mkObj out
      | _ => node
  | _ => node

/-- Port of `reachableInstances` (stack-pop walk; discovery order is the
    TS Set's insertion order). -/
private def reachableFrom (rootExprs : Array Json) (wires : Array Wire)
    (allInstances : Array String) : Array String := Id.run do
  let mut reachable : Array String := #[]
  let mut queue : Array String := #[]
  for expr in rootExprs do
    for dep in exprDependencies expr do
      if allInstances.contains dep && !reachable.contains dep then
        reachable := reachable.push dep
        queue := queue.push dep
  while !queue.isEmpty do
    let name := queue.back!
    queue := queue.pop
    for w in wires do
      if w.instName == name then
        for dep in exprDependencies w.expr do
          if allInstances.contains dep && !reachable.contains dep then
            reachable := reachable.push dep
            queue := queue.push dep
  return reachable

/-- The TS `portTypeToDecl` over an entry's structured `type_obj`. -/
private def portTypeToDecl (t : Json) : EngineM Json := do
  match getStrField? t "kind" with
  | some "scalar" => pure ((getField? t "scalar").getD jsonNull)
  | some "alias" =>
    pure <| Json.str (((getField? t "alias").bind (getStrField? · "name")).getD "")
  | some "array" =>
    let element : Json := match getField? t "element" with
      | some (.str s) => Json.str s
      | some el => Json.str ((getStrField? el "name").getD "")
      | none => jsonNull
    let shape ← match getField? t "shape" with
      | some (.arr dims) => dims.mapM fun d => do
        match d with
        | .num n => pure (Json.num n)
        | _ => internalError s!"export: array shape carries unresolved type-param '{(getStrField? d "name").getD ""}'"
      | _ => pure #[]
    pure <| Json.mkObj [("kind", Json.str "array"), ("element", element),
      ("shape", Json.arr shape)]
  | _ => pure jsonNull

private def isDefaultPortType (t? : Option Json) : Bool :=
  match t? with
  | none => true
  | some t =>
    getStrField? t "kind" == some "scalar" && getStrField? t "scalar" == some "float"

def handleExportProgram (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  if name.isEmpty then
    throwBare .missingArgument "name is required" (param := some "name")
  let outputsArg := arg? args "outputs"
  let outputPairs : Array (String × Json) := match outputsArg with
    | some (.obj m) => m.toArray
    | _ => #[]
  if outputPairs.isEmpty then
    throwBare .missingArgument "outputs is required (at least one)" (param := some "outputs")

  let st ← env.state.get
  -- CF-only: wires are stored raw; export reads the canonical wires.
  let wiresPost := st.wires
  let allInstances := st.instanceNames

  -- Validate output mappings; build the root ref expressions.
  let mut rootExprs : Array Json := #[]
  for (outName, ref) in outputPairs do
    let instName := (getStrField? ref "instance").getD ""
    let some info := st.findInstance? instName
      | internalError s!"export: output '{outName}' references unknown instance '{instName}'."
    let portName := (getStrField? ref "output").getD ""
    if !info.progMeta.outputNames.contains portName then
      internalError s!"export: instance '{instName}' has no output '{portName}'. Available: {String.intercalate ", " info.progMeta.outputNames.toList}"
    rootExprs := rootExprs.push <| Json.mkObj
      [("op", Json.str "ref"), ("instance", Json.str instName),
       ("output", Json.str portName)]

  -- Validate input mappings ("instance:port" targets).
  let inputPairs : Array (String × Json) := match arg? args "inputs" with
    | some (.obj m) => m.toArray
    | _ => #[]
  let mut exposed : Array (String × String × String) := #[]  -- (inputName, inst, port)
  for (inputName, targetJ) in inputPairs do
    let parsed? : Option (String × String) := match targetJ with
      | .str target =>
        match target.splitOn ":" with
        | instPart :: rest =>
          let portPart := String.intercalate ":" rest
          if rest.isEmpty || instPart.isEmpty || portPart.isEmpty then none
          else some (instPart, portPart)
        | [] => none
      | _ => none
    let some (instName, portName) := parsed?
      | internalError s!"export: input '{inputName}' target must be \"instance:port\", got '{tsInterp targetJ}'."
    let some info := st.findInstance? instName
      | internalError s!"export: input '{inputName}' references unknown instance '{instName}'."
    if !info.progMeta.inputNames.contains portName then
      internalError s!"export: instance '{instName}' has no input '{portName}'. Available: {String.intercalate ", " info.progMeta.inputNames.toList}"
    exposed := exposed.push (inputName, instName, portName)
  let exposedKeys := exposed.map fun (_, i, p) => s!"{i}:{p}"
  let findWire := fun (inst port : String) =>
    (wiresPost.find? fun w => w.instName == inst && w.portName == port).map (·.expr)

  -- Reverse reachability from the outputs, extended by exposed-input
  -- wiring defaults.
  let mut reachable := reachableFrom rootExprs wiresPost allInstances
  for (_, instName, portName) in exposed do
    if let some currentExpr := findWire instName portName then
      for dep in exprDependencies currentExpr do
        if allInstances.contains dep && !reachable.contains dep then
          let extra := reachableFrom #[currentExpr] wiresPost allInstances
          for e in extra do
            if !reachable.contains e then reachable := reachable.push e

  -- Dangling-reference check.
  for instName in reachable do
    for w in wiresPost do
      if w.instName == instName && !exposedKeys.contains w.key then
        for dep in exprDependencies w.expr do
          if !reachable.contains dep && allInstances.contains dep then
            internalError (s!"export: instance '{instName}' wiring '{w.key}' references '{dep}' which is outside the exported subgraph. "
              ++ s!"Either expose it as an input or include '{dep}' in the output dependency chain.")

  -- Port entries (type metadata + folded wiring defaults).
  let portInfoOf := fun (inst port : String) (isInput : Bool) =>
    (st.findInstance? inst).bind fun info =>
      (if isInput then info.progMeta.inputs else info.progMeta.outputs).find? (·.name == port)
  let mut inputEntries : Array Json := #[]
  for (inputName, instName, portName) in exposed do
    let typeObj? := (portInfoOf instName portName true).bind (·.typeObj)
    let default? := (findWire instName portName).map (rewriteRefs reachable)
    let typeDecl? ← if isDefaultPortType typeObj? then pure (none : Option Json)
      else some <$> portTypeToDecl (typeObj?.getD jsonNull)
    inputEntries := inputEntries.push <|
      match typeDecl?, default? with
      | none, none => Json.str inputName
      | t?, d? => Json.mkObj <|
        [("name", Json.str inputName)]
        ++ (match t? with | some t => [("type", t)] | none => [])
        ++ (match d? with | some d => [("default", d)] | none => [])
  let mut outputEntries : Array Json := #[]
  for (outName, ref) in outputPairs do
    let instName := (getStrField? ref "instance").getD ""
    let portName := (getStrField? ref "output").getD ""
    let typeObj? := (portInfoOf instName portName false).bind (·.typeObj)
    if isDefaultPortType typeObj? then
      outputEntries := outputEntries.push (Json.str outName)
    else
      let t ← portTypeToDecl (typeObj?.getD jsonNull)
      outputEntries := outputEntries.push <| Json.mkObj
        [("name", Json.str outName), ("type", t)]

  -- Topological order over the exported subgraph (Kahn, sorted ready
  -- queues; cycle members append in reachable order).
  let topo := Tropical.Lowering.computeInstanceTopoOrder
    (reachable.filterMap fun n => (st.findInstance? n).map (n, ·))
    (wiresPost.filter fun w => reachable.contains w.instName)
  let order := Id.run do
    let mut out := topo
    for n in reachable do
      if !out.contains n then out := out.push n
    return out

  -- Instance decls: exposed ports become {op:'input'}, sibling refs
  -- become nestedOut.
  let mut decls : Array Json := #[]
  for instName in order do
    let some info := st.findInstance? instName
      | internalError s!"export: internal: '{instName}' missing from registry"
    let mut instInputs : Array (String × Json) := #[]
    for portName in info.progMeta.inputNames do
      let key := s!"{instName}:{portName}"
      match exposed.find? fun (_, i, p) => s!"{i}:{p}" == key with
      | some (inputName, _, _) =>
        instInputs := instInputs.push (portName,
          Json.mkObj [("op", Json.str "input"), ("name", Json.str inputName)])
      | none =>
        if let some expr := findWire instName portName then
          instInputs := instInputs.push (portName, rewriteRefs reachable expr)
    decls := decls.push <| Json.mkObj <|
      [("op", Json.str "instanceDecl"), ("name", Json.str instName),
       ("program", Json.str info.baseTypeName)]
      ++ (match info.typeArgs with | some ta => [("type_args", ta)] | none => [])
      ++ (if instInputs.isEmpty then [] else [("inputs", Json.mkObj instInputs.toList)])
  let mut assigns : Array Json := #[]
  for (outName, ref) in outputPairs do
    assigns := assigns.push <| Json.mkObj
      [("op", Json.str "outputAssign"), ("name", Json.str outName),
       ("expr", Json.mkObj [("op", Json.str "nestedOut"),
         ("ref", (getField? ref "instance").getD jsonNull),
         ("output", (getField? ref "output").getD jsonNull)])]

  let node := Json.mkObj [
    ("op", Json.str "program"), ("name", Json.str name),
    ("ports", Json.mkObj [("inputs", Json.arr inputEntries),
      ("outputs", Json.arr outputEntries)]),
    ("body", Json.mkObj [("op", Json.str "block"), ("decls", Json.arr decls),
      ("assigns", Json.arr assigns)])]

  -- Register the exported program through the engine batch (raise +
  -- elaborate + strata + service registry residue + adoption) — the
  -- loadProgramAsType image; raise/elaborate failures surface as
  -- internal_error with the verbatim message.
  let jv ← match Tropical.Parse.JsonV.parse node.compress with
    | .error e => internalError s!"export_program: node JSON re-parse failed: {e}"
    | .ok v => pure v
  let parsed ← match Tropical.Parse.Raise.raiseProgram jv with
    | .error msg => internalError msg
    | .ok p => pure p
  let mut rootEntry := jsonNull
  for (n, p) in registrationBatch name (renameProgram parsed name) do
    let (entry, _) ← registerOne env n p
    rootEntry := entry
  let entryNames := fun (k : String) => Json.arr <|
    (match getField? rootEntry k with
     | some (.arr ps) => ps
     | _ => #[]).filterMap fun pj => (getStrField? pj "name").map Json.str

  let exported := order
  let exportedJ := Json.arr (order.map Json.str)

  let removeExported := match arg? args "remove_exported" with
    | some (.bool b) => b
    | _ => false
  if removeExported then
    env.state.modify fun st =>
      { st with
        instances := st.instances.filter fun (n, _) => !exported.contains n
        wires := st.wires.filter fun w =>
          !(exported.contains w.instName
            || (exprDependencies w.expr).any exported.contains)
        graphOutputs := st.graphOutputs.filter fun (i, _) => !exported.contains i }
    let st ← env.state.get
    if !st.instances.isEmpty || !st.graphOutputs.isEmpty then
      syncCompile env

  pure <| Json.mkObj [
    ("program_name", Json.str name),
    ("inputs", entryNames "inputs"),
    ("outputs", entryNames "outputs"),
    ("instances_included", exportedJ),
    ("program", Json.mkObj
      [("schema", Json.str "tropical_program_2"), ("name", Json.str name),
       ("ports", (getField? node "ports").getD jsonNull),
       ("body", (getField? node "body").getD jsonNull)])]

-- ── v2 ingest (load/merge — Phase 6 stage 6d) ────────────────────────────────
-- Port of `loadProgramAsSession` / `mergeProgramIntoSession` over the
-- engine's mirror: the normalized v2 node (Zod-stripped, key order
-- preserved by JsonV) is walked directly; inline programDecls register
-- through the engine's own `registerOne` batches; instances resolve
-- through the engine specialization path with the load-path failure
-- shapes; wires store RAW (loadProgramAsSession sets inputExprNodes
-- directly — no auto-delay wrap). All failures are plain TS Errors on
-- the oracle → `internal_error` with the verbatim message.

open Tropical.Parse (JsonV) in
private def jvBodyEntries (node : JsonV) (k : String) : Array JsonV :=
  match (node.getField? "body").bind (·.getField? k) with
  | some (.arr items) => items
  | _ => #[]

open Tropical.Parse (JsonV) in
private def jvStr? (j : JsonV) (k : String) : Option String :=
  match j.getField? k with
  | some (.str s) => some s
  | _ => none

open Tropical.Parse (JsonV) in
private def jvOp? (j : JsonV) : Option String := jvStr? j "op"

/-- Resolve a body dac-wire expression (port of
    `resolveDacWireExprToGraphOutput`; exact messages). -/
private def resolveDacWire (st : SessionSt) (expr : Tropical.Parse.JsonV)
    (context : String) : EngineM (String × String) := do
  match expr with
  | .obj _ =>
    if jvOp? expr != some "ref" then
      internalError s!"{context}: dac.out wire requires expr.op === 'ref'; got '{(jvOp? expr).getD "undefined"}'."
    let some instName := jvStr? expr "instance"
      | internalError s!"{context}: dac.out wire ref.instance must be a string"
    let some info := st.findInstance? instName
      | internalError s!"{context}: dac.out wire references unknown instance '{instName}'."
    let outNames := info.progMeta.outputNames
    match expr.getField? "output" with
    | some (.num n) =>
      let i := n.toFloat.toUInt64.toNat
      if n.toFloat < 0 || i ≥ outNames.size then
        internalError s!"{context}: dac.out wire output index {n} out of range for '{instName}' ({outNames.size} outputs)."
      pure (instName, outNames[i]!)
    | some (.str s) =>
      if !outNames.contains s then
        internalError s!"{context}: dac.out wire references unknown output '{s}' on '{instName}'. Valid: {String.intercalate ", " outNames.toList}"
      pure (instName, s)
    | _ => internalError s!"{context}: dac.out wire ref.output must be a number or string"
  | _ =>
    internalError s!"{context}: dac.out wire requires a ref-shaped expression (use \{op:'ref',instance,output}); got literal/array."

/-- Walk a normalized v2 node + topLevel into the engine session
    (additive; the caller cleared state for load). Mirrors the TS
    ingest order: type_defs → inline programDecls → params → instances
    (+ wires) → defaults → graph outputs. -/
private def ingestProgram (env : Env) (node : Tropical.Parse.JsonV)
    (top : Tropical.Parse.Raise.TopLevel) (merge : Bool) : EngineM Unit := do
  let context := if merge then "mergeProgramIntoSession" else "loadProgramAsSession"

  -- Merged param specs: body paramDecls canonical, topLevel fallback
  -- (dedup by name, body wins).
  let bodyParams := (jvBodyEntries node "decls").filterMap fun d =>
    if jvOp? d == some "paramDecl" then do
      let name ← jvStr? d "name"
      let value : Json := match d.getField? "value" with
        | some (.num n) => Json.num n
        | _ => Lean.toJson (0 : Nat)
      pure (name, value)
    else none
  let mut paramSpecs := bodyParams
  for p in top.params.getD #[] do
    if !paramSpecs.any (·.1 == p.name) then
      paramSpecs := paramSpecs.push (p.name,
        match p.value with | some v => Json.num v | none => Lean.toJson (0 : Nat))

  -- Merge collision checks (fail fast, TS order: instances then params).
  if merge then
    let st ← env.state.get
    for d in jvBodyEntries node "decls" do
      if jvOp? d == some "instanceDecl" then
        if let some name := jvStr? d "name" then
          if (st.findInstance? name).isSome then
            internalError s!"merge collision: instance '{name}' already exists."
    for (name, _) in paramSpecs do
      if st.params.any (·.1 == name) then
        internalError s!"merge collision: param '{name}' already exists."

  -- Inline program definitions, each through the engine registration
  -- batch (nested programDecls depth-first, exactly loadProgramAsType's
  -- recursion).
  for d in jvBodyEntries node "decls" do
    if jvOp? d == some "programDecl" then
      let some subName := jvStr? d "name"
        | internalError s!"{context}: programDecl missing name"
      let some subNode := d.getField? "program"
        | internalError s!"{context}: programDecl '{subName}' missing program"
      let parsed ← match Tropical.Parse.Raise.raiseProgram subNode with
        | .error msg => internalError msg
        | .ok p => pure p
      for (n, p) in registrationBatch subName (renameProgram parsed subName) do
        let _ ← registerOne env n p

  -- Params before instances (instances may reference them). Idempotent
  -- per name.
  for (name, value) in paramSpecs do
    let st ← env.state.get
    if !st.params.any (·.1 == name) then
      env.state.modify (·.setParamValue name value)

  -- Instances + their wires (raw — no auto-delay wrap on this path).
  for d in jvBodyEntries node "decls" do
    if jvOp? d != some "instanceDecl" then
      continue
    let some instName := jvStr? d "name"
      | internalError s!"{context}: instanceDecl missing name"
    let some programName := jvStr? d "program"
      | internalError s!"{context}: instanceDecl '{instName}' missing program"
    let typeArgs : Option Json := match d.getField? "type_args" with
      | some ta => some ta.toJson
      | none => none
    let (typeArgsEcho, pm, resolvedIdx) ←
      resolveInstanceMeta env programName typeArgs "program" (toolEnvelopes := false)
    env.state.modify (·.addInstance instName
      { baseTypeName := programName, typeArgs := typeArgsEcho, progMeta := pm, resolvedIdx })
    -- Wires in declared input-port order (the canonical order; JS's
    -- stable sort leaves unknown ports trailing in JSON-key order,
    -- which JsonV preserves).
    if let some (.obj inputFields) := d.getField? "inputs" then
      let declared := pm.inputNames
      let orderOf := fun (k : String) =>
        match declared.idxOf? k with
        | some i => i
        | none => declared.size
      let sorted := inputFields.zipIdx.qsort fun p q =>
        let oa := orderOf p.1.1
        let ob := orderOf q.1.1
        if oa == ob then Nat.blt p.2 q.2 else Nat.blt oa ob
      for ((input, exprV), _) in sorted do
        let expr := exprV.toJson
        match validateExpr expr s!"{instName}.{input}" with
        | .error msg => internalError msg
        | .ok _ => pure ()
        env.state.modify (·.setWireRaw instName input expr)

  -- Input defaults — every instance in registry order (TS loops the
  -- whole registry, pre-existing instances included on merge).
  let st ← env.state.get
  for (name, info) in st.instances do
    for port in info.progMeta.inputs do
      if let some defaultExpr := port.default then
        let st ← env.state.get
        if (st.findWire? name port.name).isNone then
          env.state.modify (·.setWireRaw name port.name defaultExpr)

  -- Graph outputs: body dac.out wires canonical, file-root
  -- audio_outputs deprecated fallback (appended after).
  let st ← env.state.get
  let mut outs : Array (String × String) := #[]
  for a in jvBodyEntries node "assigns" do
    if jvOp? a == some "outputAssign" && jvStr? a "name" == some "dac.out" then
      let some expr := a.getField? "expr"
        | internalError s!"{context}: dac.out wire requires a ref-shaped expression (use \{op:'ref',instance,output}); got literal/array."
      outs := outs.push (← resolveDacWire st expr context)
  for o in top.audioOutputs.getD #[] do
    match o with
    | .expr _ =>
      internalError s!"{context}: file-root audio_outputs[].expr form not supported. Use \{instance, output} or migrate to body dac.out wires."
    | .ref instName output =>
      if (st.findInstance? instName).isNone then
        internalError s!"{context}: audio_outputs references unknown instance '{instName}'."
      let outName := match output with
        | .str s => s
        | .num n => toString n
        | other => other.toJson.compress
      outs := outs.push (instName, outName)
  env.state.modify fun st => { st with graphOutputs := st.graphOutputs ++ outs }

def handleLoad (env : Env) (args : Json) : EngineM Json := do
  let path := arg? args "path"
  let program := arg? args "program"
  if path.isNone && program.isNone then
    throwBare .missingArgument "Provide either path (file) or program (inline JSON)."
  -- Stop audio before replacing the session (TS handleLoad semantics).
  if let some dac := ← env.dac.get then
    if ← dac.isRunning then dac.stop
  let rawText ← match path with
    | some (.str p) =>
      match ← (IO.FS.readFile p).toBaseIO with
      | .ok text => pure text
      | .error e => internalError (toString e)
    | _ => pure (program.getD jsonNull).compress
  let t0 ← IO.monoMsNow
  let jv ← match Tropical.Parse.JsonV.parse rawText with
    | .error e => internalError s!"JSON Parse error: {e}"
    | .ok v => pure v
  let (node, top) ← match Tropical.Parse.Raise.normalizeProgramFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  -- Clear the session (typeRegistry / programs / specializationCache
  -- survive — they hold the stdlib + session-defined types).
  env.state.modify fun st =>
    { st with instances := #[], wires := #[], graphOutputs := #[],
              params := #[], nameCounters := {} }
  ingestProgram env node top (merge := false)
  syncCompile env
  let t1 ← IO.monoMsNow
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1))),
    ("timing", Json.mkObj [("wall_ms", toJson (t1 - t0))])]

def handleMerge (env : Env) (args : Json) : EngineM Json := do
  let raw := match arg? args "program" with
    | some p => some p
    | none => arg? args "patch"
  let some program := raw
    | throwBare .missingArgument "Provide a program or patch object."
  let jv ← match Tropical.Parse.JsonV.parse program.compress with
    | .error e => internalError s!"JSON Parse error: {e}"
    | .ok v => pure v
  let (node, top) ← match Tropical.Parse.Raise.normalizeProgramFile jv with
    | .error msg => internalError msg
    | .ok r => pure r
  ingestProgram env node top (merge := true)
  syncCompile env
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1)))]

-- ── Audio / params (native — the engine owns the runtime and DAC) ───────────

private def validatePositiveInt (v : Option Json) (param : String)
    (default : Nat) : EngineM Nat :=
  match v with
  | none => pure default
  | some j@(.num n) =>
    let f := n.toFloat
    if f == f.floor && f > 0 then pure f.toUInt64.toNat
    else
      throwBare .invalidValue
        s!"{param} must be a positive integer; got {j.compress}"
        (param := some param) (value := some j)
  | some j =>
    throwBare .invalidValue
      s!"{param} must be a positive integer; got {j.compress}"
      (param := some param) (value := some j)

private def containsSub (hay needle : String) : Bool :=
  needle.isEmpty || (hay.splitOn needle).length > 1

private def isRunningJson (dac : Ffi.Dac) : EngineM Json := do
  pure <| Json.mkObj [("is_running", Json.bool (← dac.isRunning))]

def handleStartAudio (env : Env) (args : Json) : EngineM Json := do
  let dac ← match ← env.dac.get with
    | some d => pure d
    | none =>
      let sampleRate ← validatePositiveInt (arg? args "sample_rate") "sample_rate" 44100
      let channels   ← validatePositiveInt (arg? args "channels") "channels" 2
      let d ← Ffi.Dac.fromRuntime env.runtime sampleRate.toUInt32 channels.toUInt32
      env.dac.set (some d)
      pure d
  match argStr? args "device_name" with
  | some deviceName =>
    let devices ← Ffi.listDevices
    let lower := deviceName.toLower
    match devices.find? (fun d => containsSub d.name.toLower lower) with
    | none =>
      throwEnum .unknownDevice "device_name" (Json.str deviceName)
        (devices.map (·.name))
    | some m =>
      if ← dac.isRunning then
        let _ ← dac.switchDevice m.id
      else
        dac.start
        let _ ← dac.switchDevice m.id
      pure <| Json.mkObj [("is_running", Json.bool (← dac.isRunning)),
                          ("device", Json.str m.name)]
  | none =>
    if !(← dac.isRunning) then dac.start
    isRunningJson dac

def handleStopAudio (env : Env) : EngineM Json := do
  match ← env.dac.get with
  | none => throwBare .invalidState "DAC has not been created yet."
  | some dac =>
    dac.stop
    isRunningJson dac

def handleAudioStatus (env : Env) : EngineM Json := do
  match ← env.dac.get with
  | none => pure <| Json.mkObj [("is_running", Json.bool false)]
  | some dac =>
    let stats ← dac.stats
    pure <| Json.mkObj [
      ("is_running", Json.bool (← dac.isRunning)),
      ("is_reconnecting", Json.bool (← dac.isReconnecting)),
      ("stats", Json.mkObj [
        ("callbackCount", toJson stats.callbackCount.toNat),
        ("avgCallbackMs", toJson stats.avgCallbackMs),
        ("maxCallbackMs", toJson stats.maxCallbackMs),
        ("underrunCount", toJson stats.underrunCount.toNat),
        ("overrunCount",  toJson stats.overrunCount.toNat)])]

/-- `set_param`: update the mirror AND drive the live `param:<name>`
    module slot. (The TS engine only wrote the detached Param handle,
    which the session-path kernel never reads — set_param was audibly
    inert until the next recompile.) -/
def handleSetParam (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let valueJ := (getField? args "value").getD jsonNull
  let st ← env.state.get
  if (st.params.find? (·.1 == name)).isNone then
    throwEnum .unknownParam "name" (Json.str name) (st.params.map (·.1))
  let value ← match valueJ with
    | .num n => pure n.toFloat
    | _ => internalError s!"set_param: value must be a number, got {valueJ.compress}"
  env.state.modify (·.setParamValue name valueJ)
  if let some idx := ← env.runtime.slotIndex? s!"param:{name}" then
    env.runtime.setSlot idx value
  pure <| Json.mkObj [("name", Json.str name), ("value", valueJ)]

/-- `set_param_glide`: a CLOSED-FORM parameter ramp (no per-sample state). The
    kernel evaluates `f(τ) = v0 + (v1−v0)·smoothstep(clamp((τ−t0)/dur, 0, 1))` from
    three slots (`param:<name>#v0/#v1/#t0`); this op RE-ANCHORS the ramp: read the
    current sample index `now`, evaluate the ramp at `now` (so the new ramp starts
    exactly where we are — no jump), then set `v0 = current, v1 = target, t0 = now`.
    Stateless and click-free — the "state" is the anchor slots, navigable like τ.
    `dur = 0.02·SR` samples (20 ms) matches the kernel's ramp, at any sample rate. -/
def handleSetParamGlide (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let target ← match getField? args "value" with
    | some (.num n) => pure n.toFloat
    | _ => internalError "set_param_glide: value must be a number"
  let slotOf (sfx : String) : EngineM (Option UInt32) :=
    env.runtime.slotIndex? s!"param:{name}#{sfx}"
  let some _ := ← slotOf "v0"
    | internalError s!"set_param_glide: no glide slots for '{name}'"
  let read (sfx : String) : EngineM Float := do
    match ← slotOf sfx with | some i => env.runtime.getSlot i | none => pure 0.0
  let write (sfx : String) (v : Float) : EngineM Unit := do
    match ← slotOf sfx with | some i => env.runtime.setSlot i v | none => pure ()
  let now ← env.runtime.currentSampleIndex
  let dur := (← env.runtime.sampleRate) * 0.02   -- 20 ms, matching the kernel's ramp
  let v0 ← read "v0"; let v1 ← read "v1"; let t0 ← read "t0"
  let raw := (now - t0) / dur
  let s := if raw < 0.0 then 0.0 else if raw > 1.0 then 1.0 else raw
  let curr := v0 + (v1 - v0) * (s * s * (3.0 - 2.0 * s))
  write "v0" curr
  write "v1" target
  write "t0" now
  pure <| Json.mkObj [("name", Json.str name), ("value", toJson target)]

/-- `set_param_freq`: a PHASE-ANCHORED frequency change. `freq = f·τ + φ_off` — so
    changing `f` alone jumps the phase by `Δf·τ` (a click that grows with τ). If the
    source carries a `#phase` offset slot, this bumps it by the phase the frequency
    change would have jumped — `Δφ = ((inc₀ − inc₁)·T) / 2³²` cycles, `inc =
    ⌊freq·2³²/SR⌋` (the phasor's own quantized increment), `T` = now — so the phase
    stays CONTINUOUS across the change (a soft freq corner, not a hard click). This
    is the stateless anchor: the accumulated phase lives in a param, navigable like
    τ. No `#phase` slot (saw/morph, or knob-driven freq) ⇒ a raw freq write. -/
def handleSetParamFreq (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let target ← match getField? args "value" with
    | some (.num n) => pure n.toFloat
    | _ => internalError "set_param_freq: value must be a number"
  let some freqIdx := ← env.runtime.slotIndex? s!"param:{name}"
    | internalError s!"set_param_freq: no slot '{name}'"
  match ← env.runtime.slotIndex? s!"param:{name}#phase" with
  | some phaseIdx =>
    let now ← env.runtime.currentSampleIndex
    let f0 ← env.runtime.getSlot freqIdx
    let sr ← env.runtime.sampleRate
    let inc0 := Float.floor (f0 * 4294967296.0 / sr)
    let inc1 := Float.floor (target * 4294967296.0 / sr)
    let dcyc := ((inc0 - inc1) * now) / 4294967296.0
    let off0 ← env.runtime.getSlot phaseIdx
    let raw := off0 + dcyc
    env.runtime.setSlot phaseIdx (raw - Float.floor raw)   -- frac → [0, 1)
    env.runtime.setSlot freqIdx target
  | none => env.runtime.setSlot freqIdx target
  pure <| Json.mkObj [("name", Json.str name), ("value", toJson target)]

/-- `set_param_velocity`: the GLOBAL TIME-WARP scrub. The master clock is
    `M(n) = tau_base·SR·2³² + velocity·2³²·n`; changing `velocity` alone would jump
    `M` by `Δv·n` (a click growing with n). This re-bases the host-held origin so
    `M` stays value-continuous across the change: read `now`, set
    `tau_base += (v_old − v_new)·now/SR`, then write the new velocity. Exactly the
    stateless `ScrubClock` host-split (and the clock-domain twin of
    `set_param_freq`): the accumulator lives in a param, navigable like τ, so the
    kernel stays `f(τ)`. `velocity = 1` forward · `0` freeze · `−1` reverse · `>1`
    varispeed. `name` is the velocity slot (`master.velocity`); the origin slot is
    the sibling `master.tau_base`. -/
def handleSetParamVelocity (env : Env) (args : Json) : EngineM Json := do
  let name := (argStr? args "name").getD ""
  let target ← match getField? args "value" with
    | some (.num n) => pure n.toFloat
    | _ => internalError "set_param_velocity: value must be a number"
  let some velIdx := ← env.runtime.slotIndex? s!"param:{name}"
    | internalError s!"set_param_velocity: no slot '{name}'"
  let tauName := name.replace "velocity" "tau_base"
  let some tauIdx := ← env.runtime.slotIndex? s!"param:{tauName}"
    | internalError s!"set_param_velocity: no origin slot '{tauName}'"
  let now ← env.runtime.currentSampleIndex
  let sr ← env.runtime.sampleRate
  let v0 ← env.runtime.getSlot velIdx
  let tb0 ← env.runtime.getSlot tauIdx
  env.runtime.setSlot tauIdx (tb0 + (v0 - target) * now / sr)
  env.runtime.setSlot velIdx target
  env.state.modify (·.setParamValue name (toJson target))
  pure <| Json.mkObj [("name", Json.str name), ("value", toJson target)]

def handleListParams (env : Env) : EngineM Json := do
  let st ← env.state.get
  pure <| Json.arr <| st.params.map fun (n, v) =>
    Json.mkObj [("name", Json.str n), ("value", v)]

/-- Differential-harness probe (rpc-only, not an MCP tool): render N
    buffers through the Lean-owned runtime, return the raw samples as
    hex — lets the recorded scripts assert audio equivalence between
    engines. -/
def handleDebugRender (env : Env) (args : Json) : EngineM Json := do
  let frames := match arg? args "frames" with
    | some (.num n) => n.toFloat.toUInt64.toNat
    | _ => 4
  let hexDigit (n : UInt8) : Char :=
    if n < 10 then Char.ofNat ('0'.toNat + n.toNat)
    else Char.ofNat ('a'.toNat + n.toNat - 10)
  let mut hex := ""
  for _ in [0:frames] do
    env.runtime.process
    let bytes ← env.runtime.outputBytes
    let mut chunk := ""
    for b in bytes.toList do
      chunk := chunk.push (hexDigit (b >>> 4)) |>.push (hexDigit (b &&& 0xf))
    hex := hex ++ chunk
  pure <| Json.mkObj [("frames", toJson frames), ("hex", Json.str hex)]

-- ── Dispatcher ───────────────────────────────────────────────────────────────

def handleListScopeTaps (env : Env) : EngineM Json := do
  let st ← env.state.get
  let taps := st.scopeTaps.map fun (name, inst, out) =>
    Json.mkObj [("name", Json.str name), ("instance", Json.str inst),
                ("output", Json.str out), ("slot", Json.str s!"{inst}.{out}")]
  pure <| Json.mkObj [("taps", Json.arr taps)]

/-- EXPERIMENT (`load_patch_graph`): compile a downstream-only patch graph (the
    playground GUI) through the EmitArrow arrow lowering — `lowerGraph → normalize
    (the slide) → emitTerm` — to a session root, then the production
    `compileSession → buildKernelIr → loadIr` tail. A compile failure errors
    BEFORE `loadIr`, so the previous kernel keeps playing. -/
def handleLoadPatchGraph (env : Env) (args : Json) : EngineM Json := do
  let (plan, taps) ← match ← Tropical.Playground.compilePlan args with
    | .error e => internalError e
    | .ok p => pure p
  let (ir, planJson) ← buildKernelIr plan
  env.runtime.loadIr ir planJson
  -- Seed the session param mirror with the graph's knobs so `set_param` — which
  -- guards on the mirror, then drives the live `param:<name>` slot — reaches them
  -- without a relower. Replaces (not appends): the mirror tracks the current graph.
  -- Also publish the arrow taps as `scopeTaps` (each already routed to a
  -- `render_window`-readable root output slot), so an attached scope discovers
  -- this graph's inspection points via `list_scope_taps` with no session wiring.
  env.state.modify (fun st => { st with
    params := Tropical.Playground.knobParams args
    scopeTaps := taps })
  pure <| Json.mkObj [("ok", Json.bool true)]

def handleTool (env : Env) (name : String) (args : Json) : IO Json :=
  wrap <| match name with
  | "load_patch_graph" => handleLoadPatchGraph env args
  | "define_program"  => handleDefineProgram env args
  | "add_instance"    => handleAddInstance env args
  | "remove_instance" => handleRemoveInstance env args
  | "replicate"       => handleReplicate env args
  | "wire_chain"      => handleWireChain env args
  | "wire_zip"        => handleWireZip env args
  | "fan_out"         => handleFanOut env args
  | "fan_in"          => handleFanIn env args
  | "export_program"  => handleExportProgram env args
  | "list_programs"   => handleListPrograms env
  | "list_instances"  => handleListInstances env
  | "get_info"        => handleGetInfo env args
  | "wire"            => handleWire env args
  | "list_wiring"     => handleListWiring env args
  | "list_scope_taps" => handleListScopeTaps env
  | "load"            => handleLoad env args
  | "save"            => handleSave env
  | "merge"           => handleMerge env args
  | "start_audio"     => handleStartAudio env args
  | "stop_audio"      => handleStopAudio env
  | "audio_status"    => handleAudioStatus env
  | "set_param"       => handleSetParam env args
  | "set_param_glide" => handleSetParamGlide env args
  | "set_param_freq"  => handleSetParamFreq env args
  | "set_param_velocity" => handleSetParamVelocity env args
  | "list_params"     => handleListParams env
  | "debug_render"    => handleDebugRender env args
  | _ => internalError s!"Unknown tool: '{name}'"

-- ── Boot ─────────────────────────────────────────────────────────────────────

/-- Spawn the compiler service, boot the stdlib from the pre-parsed
    bridge, build the Env.

    The service no longer loads the stdlib (Phase 4 stage 4b): the
    engine reads `stdlib/parsed/manifest.json` (the registration order
    `loadStdlib` produced) and each `stdlib/parsed/<Name>.json` from
    the repo root, and registers each through the SAME
    elaborate→register→adopt flow `define_program` uses. Mirroring
    `loadStdlibFromResolved`: stdlib elaboration resolves siblings
    through the RAW elaborated map (TS `localResolved`), and the
    service relinks concrete registrations to the post-strata canon
    before strata (its `processedByName` step) — so the registered
    catalog is byte-faithful to the old TS `loadStdlib`. Any failure
    here is fatal: the engine cannot compile without its store. -/
def boot : IO Env := do
  let state ← IO.mkRef ({} : SessionSt)
  let runtime ← Ffi.Runtime.new 512
  let dac ← IO.mkRef (none : Option Ffi.Dac)
  let env : Env := { state, runtime, dac }
  let manifestText ← IO.FS.readFile "stdlib/parsed/manifest.json"
  let names ← match Json.parse manifestText with
    | .error e => throw <| IO.userError s!"stdlib/parsed/manifest.json: {e}"
    | .ok j =>
      match j.getObjVal? "programs" with
      | .ok (.arr ns) => pure <| ns.filterMap fun n =>
          match n with | .str s => some s | _ => none
      | _ => throw <| IO.userError "stdlib/parsed/manifest.json: missing programs[]"
  let registerAll : EngineM Unit := do
    let mut raw : Std.HashMap String Tropical.Ir.ProgramIdx := {}
    for name in names do
      let path := s!"stdlib/parsed/{name}.json"
      let text ← IO.FS.readFile path
      let prog ← match Tropical.Parse.JsonV.parse text with
        | .error e => internalError s!"{path}: JSON parse failed: {e}"
        | .ok jv =>
          match Tropical.Parse.decodeProgram jv with
          | .error e => internalError s!"{path}: {e}"
          | .ok p => pure p
      let rawMap := raw
      let (_, rawIdx) ← registerOne env name prog (some fun n => rawMap.get? n)
      raw := raw.insert name rawIdx
  match ← registerAll.run with
  | .ok () => pure ()
  | .error f => throw <| IO.userError s!"stdlib boot failed: {f.toJson.compress}"
  pure env

end Tropical.Engine
