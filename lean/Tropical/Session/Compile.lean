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
    -- an `int32` source output keeps its kind on the lifted input
    let refKind (inst out : String) : Option Tropical.Parse.ScalarKind := do
      let info ← st.findInstance? inst
      let port ← info.progMeta.outputs.find? (·.name == out)
      let k ← port.typeStr.bind Tropical.Parse.ScalarKind.ofWire?
      if k == .int32 then some k else none
    let lifted ← match Tropical.Ir.WireProgram.lift w.expr synthName st.arena.exprs refKind with
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
    -- Only the post-strata form becomes the instance's type below. Strata
    -- runs with no relink: a lifted body has no InstanceDecls.
    let (arenaPost, postIdx) ← match runStrataChecked arenaRaw rawIdx with
      | .error msg => internalError msg
      | .ok r => pure r
    -- Render the entry first: state survives a failure in `SessionM`, so
    -- nothing is written to the store until every fallible step is done.
    -- The store adopts the post-strata arena (the raw program plus what
    -- strata appended), like the registration path; then the counter and
    -- the raw program's name are persisted.
    let entry ← match Tropical.Entries.concreteEntry arenaPost synthName postIdx with
      | .error e => internalError e
      | .ok j => pure j
    adoptResolved arenaPost postIdx
    modify fun s =>
      { s with nameCounters := s.nameCounters.insert "__wire" counter,
               templateByName := s.templateByName.insert synthName rawIdx }
    let pm := ProgMeta.fromEntry entry
    let resolvedIdx := some postIdx
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

/-- A bare literal crossing into a fixed-point input is the REAL value, as a
    `paramDecl.value` is: `440` into an `int32` Q16.16 input is 440, the word
    `440 · 2^16` (`PortFormat.encodeLiteral?`). The rule is the bare-crossing
    rule of the format check (`checkWireFormats`): it applies to a wire that is
    exactly a literal; a literal inside arithmetic stays a word, like every
    other value the IR computes with. A literal whose word is not an integer,
    or lies outside the format, is refused. An input with no fraction bits
    keeps its literal as written (the word is the value), and float inputs
    are untouched. A literal outside the input's declared interval
    (`InputDecl.interval?`, the real value before any format) is refused
    too. -/
def bareLiteralAtPort (what : String) (decl : Tropical.Ir.InputDecl)
    (expr : Tropical.WireExpr) : Except String Tropical.WireExpr := do
  if let (.num n, some i) := (expr, decl.interval?) then
    unless i.contains n do
      throw s!"input {what}: literal {n} is outside the input's interval {i.label}"
  match expr, decl.type? with
  | .num n, some (.scalar k) =>
    if k.isInteger && decl.fracBits != 0 then
      let f : Tropical.PortFormat := { kind := k, fracBits := decl.fracBits }
      match f.encodeLiteral? n with
      | .ok w => .ok (.num ⟨w, 0⟩)
      | .error e => .error s!"input {what}: {e}"
    else .ok expr
  | _, _ => .ok expr

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
      let wexpr ← bareLiteralAtPort s!"'{name}.{w.portName}'" tgt.inputs[pos]! w.expr
      let (value, exprs') ← (wireExprToResolved ctx wexpr).run exprs
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

/-- The declared format of a port from its catalog `type_obj` (`float` when
    it has none, or is an array). -/
private def portFormatOf (p? : Option PortInfo) : Tropical.PortFormat :=
  match p?.bind (·.typeObj) with
  | some t =>
    if (t.getObjVal? "kind").toOption == some (Json.str "scalar") then
      (Tropical.PortFormat.ofJson? t).toOption.getD {}
    else {}
  | none => {}

/-- Formats are checked where a value crosses a port bare: a wire that is
    exactly a param, or exactly another instance's output, into an input of
    the same integer kind must agree on `frac_bits`. (A kind mismatch is the
    compiler's crossing check, with the cast it needs.) A wire that computes
    — `rshift(param(f), 8)` — yields a plain word, and binds to any input of
    its kind: arithmetic on words is how a patch converts between formats, and
    the IR never sees a format. -/
private def checkWireFormats (st : SessionSt) : SessionM Unit := do
  for w in st.wires do
    let some info := st.findInstance? w.instName | continue
    let dst := portFormatOf (info.progMeta.inputs.find? (·.name == w.portName))
    let src? : Option (String × Tropical.PortFormat) := match w.expr with
      | .param nm =>
        if st.params.any (·.1 == nm) then some (s!"param '{nm}'", st.paramFormat nm) else none
      | .ref inst out =>
        (st.findInstance? inst).map fun srcInfo =>
          let port := match out with
            | .name nm => srcInfo.progMeta.outputs.find? (·.name == nm)
            | .index n => srcInfo.progMeta.outputs[n.toFloat.toUInt64.toNat]?
          let outName := match out with | .name nm => nm | .index n => toString n
          (s!"output '{inst}.{outName}'", portFormatOf port)
      | _ => none
    if let some (what, src) := src? then
      if src.kind == dst.kind && src.kind.isInteger && src.fracBits != dst.fracBits then
        internalError s!"Format mismatch: input '{w.instName}.{w.portName}' is {dst.describe} but {what} is {src.describe}; a value crosses a port bare only at its own format — convert it with arithmetic in the wire (a shift by the difference in fraction bits)"

-- ── Anchored params (design/host-param-dispatch.md, "Patch-bay params") ─────

/-- The companion slot of an anchored param: the phase offset the host moves
    when it retunes the param. -/
def anchorCompanion (name : String) : String := s!"{name}#phase"

/-- What the compile adds for the session's anchored params: each companion
    param (seeded 0) wired into the phase input its consumers declare, and the
    plan's discipline table. Session state is untouched — the companions are
    the discipline's implementation, not the patch, so `save` writes only the
    `paramDecl`'s `discipline`. -/
structure AnchorLowering where
  wires : Array Wire
  params : Array (String × Json)
  disciplines : Array Tropical.Plan.ParamDiscipline

/-- The session instances whose own program reads param `p` in its body: the
    `__wire_N` programs `liftIfNeeded` makes from a wire with an array
    literal, which declare the params the wire read. A wire that refs one of
    them reads `p` through an expression. -/
private def liftedReaders (st : SessionSt) (p : String) : Array String :=
  st.instances.filterMap fun (n, info) =>
    match info.resolvedIdx.bind st.arena.program? with
    | some prog =>
      if prog.params.any (fun d => match d with | .param m _ => m == p | _ => false)
      then some n else none
    | none => none

/-- Does wire `w` read param `p` other than bare — through an expression, or
    through a lifted wire program (`liftedReaders`)? -/
private def readsThroughExpression (w : Wire) (p : String) (lifted : Array String) : Bool :=
  w.expr != .param p &&
    (w.expr.paramNames.contains p || w.expr.deps.any lifted.contains)

/-- Check every param the session declares `anchor` against its wiring, and
    lower it. The host's bump `Φ(inc₀, C) − Φ(inc₁, C)` is computed from the
    param's own value and the sample clock `C = now·2¹⁶`, and it moves only
    the phase offset an anchor-declaring input names. A reader the check
    cannot see into — an input with no anchor, which may integrate the value
    as a frequency inside its type, or an expression whose result may reach
    one through other instances — would click on every retune unannounced.
    So the rule is strict, and everything else is refused, naming the wire:

    - the param is a float (an integer param is written raw);
    - it is read only bare, and only by inputs that declare a phase anchor
      (`InputDecl.anchor?`): no expression, no lifted wire (`liftedReaders`),
      no input without an anchor;
    - each consumer's clock input is wired to τ itself (`{op:'clock'}`): the
      `anchor` identity is the sample clock's, and a phasor reading a warped
      or scrubbed clock steps by a different amount;
    - each consumer's phase input is free (unwired, or on its declared
      default) and unclaimed: the companion takes it;
    - nothing reads the companion's name (`anchorCompanion`), declared or not.

    One companion serves every consumer: they share the clock, so the bump
    depends on the param's value alone. A param declared `anchor` that nothing
    reads gets no companion and no table entry (it is raw until wired), so an
    edit session is not refused while its phasor is being rewired. Run on the
    session as it stands, lifted wires included. -/
def anchorParams (st : SessionSt) : Except String AnchorLowering := do
  let mut wires := st.wires
  let mut params := st.params
  let mut disciplines : Array Tropical.Plan.ParamDiscipline := #[]
  let mut claimed : Array (String × String) := #[]
  let declOf := fun (inst port : String) =>
    (st.findInstance? inst).bind fun info =>
      (info.resolvedIdx.bind st.arena.program?).bind fun prog =>
        (prog.inputs.find? (·.name == port)).map (prog.name, ·)
  for (p, disc) in st.paramDisciplines do
    unless disc == .anchor do continue
    let what := s!"param '{p}' is declared anchor"
    let f := st.paramFormat p
    unless f.isDefault do
      throw s!"{what} but is {f.describe}; a phase anchor is a float frequency, and an integer param is written raw"
    let companion := anchorCompanion p
    if st.params.any (·.1 == companion) then
      throw s!"{what}, and its phase companion '{companion}' is also declared as a param; rename that param"
    let lifted := liftedReaders st p
    let companionReaders := liftedReaders st companion
    if let some w := st.wires.find? fun w =>
        w.expr.paramNames.contains companion || w.expr.deps.any companionReaders.contains then
      throw s!"{what} but input '{w.instName}.{w.portName}' reads its phase companion '{companion}'; the companion is the host's, not the patch's"
    let anchoredInput := fun (w : Wire) => ((declOf w.instName w.portName).bind (·.2.anchor?)).isSome
    if let some w := st.wires.find? fun w => readsThroughExpression w p lifted then
      throw s!"{what} but input '{w.instName}.{w.portName}' reads it through an expression; an anchored param is read only bare, by the frequency inputs it retunes (the host computes the phase step from the param's own value) — drive anything else from a second, raw param"
    if let some w := st.wires.find? fun w => w.expr == .param p && !anchoredInput w then
      let ty := ((declOf w.instName w.portName).map (·.1)).getD "?"
      throw s!"{what} but input '{w.instName}.{w.portName}' ({ty}) declares no phase anchor; an anchored param is read only by inputs that declare one (ClockPhasor.freq) — drive anything else from a second, raw param"
    let consumers := st.wires.filter fun w => w.expr == .param p
    if consumers.isEmpty then continue
    for w in consumers do
      let some info := st.findInstance? w.instName
        | throw s!"{what} but its wire into '{w.instName}' names no instance"
      let some (_, decl) := declOf w.instName w.portName
        | throw s!"{what}: input '{w.instName}.{w.portName}' has no resolved declaration (engine bug)"
      let some a := decl.anchor?
        | throw s!"{what}: input '{w.instName}.{w.portName}' lost its anchor (engine bug)"
      match st.findWire? w.instName a.clock with
      | some cw =>
        unless cw.expr == .clock do
          throw s!"{what} but '{w.instName}.{w.portName}' reads the clock '{w.instName}.{a.clock}', which is wired from '{cw.expr.opName}'; the anchor discipline holds on the sample clock only — wire '{w.instName}.{a.clock}' to \{op:'clock'}, or write '{p}' raw"
      | none =>
        throw s!"{what} but the clock input '{w.instName}.{a.clock}' of '{w.instName}.{w.portName}' is unwired and has no default; wire it to \{op:'clock'}"
      let portDefault := (info.progMeta.inputs.find? (·.name == a.phase)).bind (·.default)
      if let some pw := st.findWire? w.instName a.phase then
        unless some pw.expr == portDefault do
          throw s!"{what} but the phase input '{w.instName}.{a.phase}' of '{w.instName}.{w.portName}' is wired from '{pw.expr.opName}'; an anchored frequency owns its phase input (its companion '{companion}' is wired there)"
      if let some (_, other) := claimed.find? (·.1 == s!"{w.instName}.{a.phase}") then
        throw s!"{what} but the phase input '{w.instName}.{a.phase}' is already the companion of anchored param '{other}'"
      claimed := claimed.push (s!"{w.instName}.{a.phase}", p)
      wires := match wires.findIdx? (fun x => x.instName == w.instName && x.portName == a.phase) with
        | some i => wires.set! i { instName := w.instName, portName := a.phase, expr := .param companion }
        | none => wires.push { instName := w.instName, portName := a.phase, expr := .param companion }
    params := params.push (companion, toJson (0 : Nat))
    disciplines := disciplines.push
      { name := p, discipline := "anchor", companions := #[companion] }
  return { wires, params, disciplines }

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
  Tropical.Lowering.assertSessionAcyclic st.instances st.wires
  checkWireFormats st
  -- Anchored params: checked against their wiring, then each companion is
  -- wired into its consumers' phase inputs (`anchorParams`).
  let anchored ← match anchorParams st with
    | .error e => internalError s!"{ctx}: {e}"
    | .ok r => pure r
  let wiresPost := anchored.wires
  let alloc := Tropical.Lowering.allocate (anchored.params.map (·.1)) st.instances

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
    params := anchored.params
    paramTypes := st.paramTypes
    paramDisciplines := anchored.disciplines
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
