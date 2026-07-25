import Tropical.Ir.Nodes

/-!
# Elaborator — line-faithful port of compiler/ir/elaborator.ts

`ParsedProgram → ResolvedProgram` over the arena. Single top-down pass:
each declaration is constructed once, registered in scope, and re-used
by (typed-index) reference at every site that names it.

Observable behaviors preserved from the original port:

- **Decl reordering.** `registerBodyDecls` pushes ALL nested
  programDecls first (source order), then non-program decls in source
  order: resolved `body.decls` ≠ parsed order.
- **Two-phase decl elaboration.** Shells are registered first (forward
  refs work), expressions resolved in a second pass iterating in
  pairing-insertion order (= non-program source order).
- **Transitive registry merge.** `registerInstanceDecl` inserts the
  target under its program name, then the target's own registry entries
  in their order, skipping keys already present. Insertion order is
  observable through the codec.
- **findInstanceCycles**: dep-edge collection order, Tarjan visit
  order, SCC member order (stack-pop order — the SCC root is last),
  and nontrivial filter (size > 1 or self-loop).

The binder/combinator/sum-type machinery (scope binders, mintBinder,
variant registries, tag/match/let elaboration) was deleted with those
constructors — `ParsedExpr` cannot spell them, so no arm exists here.
-/

namespace Tropical.Ir

open Lean (JsonNumber)
open Tropical.Parse (ParsedExpr)

-- ─────────────────────────────────────────────────────────────
-- Errors
-- ─────────────────────────────────────────────────────────────

/-- Port of `ElaborationError` (plain message) and `CycleViolation`
    (message prebuilt by `throwOnCycles`). -/
inductive ElabError where
  | elaboration (msg : String)
  | cycle (msg : String)
deriving Repr

def ElabError.message : ElabError → String
  | .elaboration m => m
  | .cycle m => m

-- ─────────────────────────────────────────────────────────────
-- Monad + scope
-- ─────────────────────────────────────────────────────────────

/-- Port of `ExternalProgramResolver`: already-resolved programs by
    name, as arena indices. -/
abbrev Resolver := String → Option ProgramIdx

structure ElabSt where
  arena : Arena := {}

abbrev ElabM := StateT ElabSt (Except ElabError)

private def throwElab {α} (msg : String) : ElabM α :=
  throw (.elaboration msg)

/-- The chain-visible scope family: nested program types. (Value decls
    never leak to nested programs — `lookupValueRef` is local-only.) -/
structure Frame where
  programs : Array (String × ProgramIdx) := #[]

structure Scope where
  frame : Frame
  parents : List Frame
  resolver : Option Resolver := none
  /-- Position = InputIdx. -/
  inputNames : Array String := #[]
  params : Array (String × Nat) := #[]
  /-- name → (InstanceIdx, typeKey). -/
  instances : Array (String × Nat × String) := #[]
  registry : Array (String × ProgramIdx) := #[]

private def Scope.frames (s : Scope) : List Frame :=
  s.frame :: s.parents

private def lookupProgramChain (s : Scope) (name : String) : Option ProgramIdx :=
  s.frames.findSome? fun f => (f.programs.find? (·.1 == name)).map (·.2)

private def getProgram (i : ProgramIdx) : ElabM Program := do
  match (← get).arena.program? i with
  | some p => pure p
  | none => throwElab s!"internal: arena program index {i.idx} out of range"

/-- Intern a resolved-expression node into the arena's shared DAG, returning
    its (possibly shared) id. The elaborator builds the id-form directly — there
    is no tree `Expr`. -/
private def internE (n : ENode) : ElabM ExprId := do
  let st ← get
  let (id, ex) := (eintern n).run st.arena.exprs
  set { st with arena := { st.arena with exprs := ex } }
  pure id

-- ─────────────────────────────────────────────────────────────
-- Builtin tables (elaborator.ts constants, verbatim)
-- ─────────────────────────────────────────────────────────────

/-- BUILTIN_TYPE_TO_SCALAR (includes `phase`). -/
private def builtinTypeToScalar? : String → Option ScalarKind
  | "float" => some .float
  | "int" => some .int
  | "bool" => some .bool
  | "signal" | "freq" | "unipolar" | "bipolar" | "phase" => some .float
  -- `clock`/`time`: a fixed-point time coordinate (Q32.32 samples), carried as
  -- the `int` (i64) carrier. A semantic alias — no separate runtime kind.
  | "clock" | "time" => some .int
  | _ => none

private def nullaryCalls : List String :=
  ["sample_rate", "sample_index", "sampleRate", "sampleIndex",
   "clock", "sample_clock", "sampleClock"]

private def unaryCall? : String → Option UnaryOpTag
  | "sqrt" => some .sqrt
  | "abs" => some .abs
  | "neg" => some .neg
  | "floor" => some .floor
  | "ceil" => some .ceil
  | "round" => some .round
  | "not" => some .not
  | "bit_not" | "bitNot" => some .bitNot
  | "to_int" | "toInt" => some .toInt
  | "to_bool" | "toBool" => some .toBool
  | "to_float" | "toFloat" => some .toFloat
  | "float_exponent" | "floatExponent" => some .floatExponent
  | _ => none

private def binaryCall? : String → Option BinaryOpTag
  | "floor_div" | "floorDiv" => some .floorDiv
  | "ldexp" => some .ldexp
  | _ => none

-- ─────────────────────────────────────────────────────────────
-- Port types (builtin scalar kinds only — user type defs are retired)
-- ─────────────────────────────────────────────────────────────

private def resolveElement (name : String) : ElabM ScalarKind := do
  match builtinTypeToScalar? name with
  | some k => pure k
  | none => throwElab s!"unknown type name '{name}'"

private def resolvePortType : Tropical.Parse.PortTypeDecl → ElabM PortType
  | .scalar name => do
    match builtinTypeToScalar? name with
    | some k => pure (.scalar k)
    | none => throwElab s!"unknown port type '{name}'"
  | .array element shape => do
    pure (.array (← resolveElement element) shape)

-- ─────────────────────────────────────────────────────────────
-- Expressions
-- ─────────────────────────────────────────────────────────────

/-- Port of `lookupValueRef` — local scope only, fixed category order. Returns
    the leaf `ENode` (the caller interns it). -/
private def lookupValueRef (s : Scope) (name : String) : Option ENode :=
  match s.params.find? (·.1 == name) with
  | some (_, i) => some (.paramRef ⟨i⟩)
  | none =>
  match s.inputNames.idxOf? name with
  | some i => some (.inputRef ⟨i⟩)
  | none => none

mutual

partial def resolveExpr (scope : Scope) : ParsedExpr → ElabM ExprId
  | .num n => internE (.num n)
  | .bool b => internE (.bool b)
  | .arr items => do
    let mut out : Array ExprId := #[]
    for e in items do
      out := out.push (← resolveExpr scope e)
    internE (.arr out)
  | .binary tag lhs rhs => do
    internE (.binary (BinaryOpTag.ofParse tag) (← resolveExpr scope lhs) (← resolveExpr scope rhs))
  | .unary tag arg => do
    internE (.unary (UnaryOpTag.ofParse tag) (← resolveExpr scope arg))
  | .nameRef name =>
    match lookupValueRef scope name with
    | some n => internE n
    | none => throwElab s!"unknown name '{name}'"
  | .nestedOut refName outputName => resolveNestedOut scope refName outputName
  | .index arr idx => do
    internE (.index (← resolveExpr scope arr) (← resolveExpr scope idx))
  | .call callee args => resolveCall scope callee args

partial def resolveNestedOut (scope : Scope) (refName outputName : String) : ElabM ExprId := do
  let some (_, instIdx, typeKey) := scope.instances.find? (·.1 == refName)
    | throwElab s!"instance '{refName}' is not declared in this scope"
  let some targetIdx := (scope.registry.find? (·.1 == typeKey)).map (·.2)
    | throwElab s!"internal: instance '{refName}' typeKey '{typeKey}' not in scope registry"
  let target ← getProgram targetIdx
  match target.outputs.findIdx? (·.name == outputName) with
  | some o => internE (.nestedOut ⟨instIdx⟩ ⟨o⟩)
  | none =>
    let portList := String.intercalate ", " (target.outputs.map (·.name)).toList
    throwElab s!"instance '{refName}': program '{target.name}' has no output '{outputName}' (have: {portList})"

partial def resolveCall (scope : Scope) (callee : ParsedExpr)
    (args : Array ParsedExpr) : ElabM ExprId := do
  let .nameRef fname := callee
    | throwElab "unsupported call form: callee must be an identifier (no first-class function values yet)"
  if nullaryCalls.contains fname then
    if args.size != 0 then
      throwElab s!"'{fname}()' takes no arguments"
    if fname == "sample_rate" || fname == "sampleRate" then
      internE .sampleRate
    else if fname == "sample_index" || fname == "sampleIndex" then
      internE .sampleIndex
    else
      -- clock(): the root fixed-point time coordinate θ = sampleIndex << 32,
      -- i.e. the current sample expressed in Q32.32 samples (zero fraction).
      internE (.binary .lshift (← internE .sampleIndex) (← internE (.num { mantissa := 32, exponent := 0 })))
  else if let some tag := unaryCall? fname then
    if args.size != 1 then
      throwElab s!"'{fname}' takes 1 argument; got {args.size}"
    internE (.unary tag (← resolveExpr scope args[0]!))
  else if let some tag := binaryCall? fname then
    if args.size != 2 then
      throwElab s!"'{fname}' takes 2 arguments; got {args.size}"
    internE (.binary tag (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!))
  else if fname == "arraySet" || fname == "array_set" then
    if args.size != 3 then
      throwElab s!"'{fname}' takes 3 arguments (arr, idx, value); got {args.size}"
    internE (.arraySet (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!)
                       (← resolveExpr scope args[2]!))
  else if fname == "clamp" then
    if args.size != 3 then
      throwElab s!"'clamp' takes 3 arguments (value, lo, hi); got {args.size}"
    internE (.clamp (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!)
                    (← resolveExpr scope args[2]!))
  else if fname == "select" then
    if args.size != 3 then
      throwElab s!"'select' takes 3 arguments (cond, then, else); got {args.size}"
    internE (.select (← resolveExpr scope args[0]!) (← resolveExpr scope args[1]!)
                     (← resolveExpr scope args[2]!))
  else
    throwElab <|
      s!"unknown function '{fname}'. The resolved IR has no escape hatch for unknown calls — " ++
      "add the builtin to the elaborator's registry, or use an instance declaration if it's a program type."

end

-- ─────────────────────────────────────────────────────────────
-- Cycle detection (port of cycle_break.ts findInstanceCycles)
-- ─────────────────────────────────────────────────────────────

/-- Dep-edge collection (collectNestedOutInstances): exact traversal
    order; first-add wins position (Set insertion parity). Walks the id-form
    expression by derefing through the arena. Total by descent on `id.idx`
    (`hw` is checked once by `findInstanceCycles`). -/
private def collectNestedOutDeps (ea : ExprArena) (hw : ea.wf = true)
    (numInstances : Nat) (acc : Array Nat) (id : ExprId) : Array Nat :=
  match _hd : ea.deref id with
  | none => acc
  | some (.num _) | some (.bool _) => acc
  | some (.arr items) =>
    items.attach.foldl (fun a ⟨x, _⟩ => collectNestedOutDeps ea hw numInstances a x) acc
  | some (.nestedOut inst _) =>
    if inst.idx < numInstances && !acc.contains inst.idx then acc.push inst.idx else acc
  | some (.binary _ lhs rhs) =>
    collectNestedOutDeps ea hw numInstances
      (collectNestedOutDeps ea hw numInstances acc lhs) rhs
  | some (.unary _ arg) => collectNestedOutDeps ea hw numInstances acc arg
  | some (.clamp a b c) | some (.select a b c) | some (.arraySet a b c) =>
    collectNestedOutDeps ea hw numInstances
      (collectNestedOutDeps ea hw numInstances
        (collectNestedOutDeps ea hw numInstances acc a) b) c
  | some (.index a b) =>
    collectNestedOutDeps ea hw numInstances
      (collectNestedOutDeps ea hw numInstances acc a) b
  | some (.bankSum _ ts b dc _) =>
    let acc := ts.attach.foldl
      (fun a ⟨t, _⟩ => collectNestedOutDeps ea hw numInstances a t) acc
    collectNestedOutDeps ea hw numInstances
      (match _hdc : dc with
        | some d => collectNestedOutDeps ea hw numInstances acc d
        | none => acc) b
  | some (.inputRef _) | some (.paramRef _)
  | some (.sampleRate) | some (.sampleIndex)
  | some (.loopIdx _) => acc
termination_by id.idx
decreasing_by
  all_goals
    apply Tropical.Ir.ExprArena.forall_children_lt hw ‹_ = some _›
    simp_all [ENode.children] <;>
      first
        | exact Or.inl ⟨_, by assumption, rfl⟩
        | exact Or.inr ⟨_, by assumption, rfl⟩
        | exact ⟨_, by assumption, rfl⟩

private structure TarjanSt where
  indexOf : Array (Option Nat)
  lowlink : Array Nat
  onStack : Array Bool
  stack : Array Nat := #[]
  sccs : Array (Array Nat) := #[]
  next : Nat := 0

/-- Tarjan's SCC, recursion + orders matching cycle_break.ts: nodes
    visited in instance order, successors in dep insertion order, SCC
    members in stack-pop order (the SCC root is last).

    Deliberately `partial`: the discharging measure is "unvisited nodes
    strictly decrease" (each recursive call happens under a
    `visited[w] = false` check that its own entry immediately flips),
    a fact threaded through mutable state inside a `for` loop — carrying
    it as data is the classic well-founded-Tarjan exercise and buys
    nothing here. -/
private partial def strongConnect (deps : Array (Array Nat)) (v : Nat)
    (st0 : TarjanSt) : TarjanSt := Id.run do
  let mut st := st0
  st := { st with
    indexOf := st.indexOf.set! v (some st.next)
    lowlink := st.lowlink.set! v st.next
    next := st.next + 1
    stack := st.stack.push v
    onStack := st.onStack.set! v true }
  for w in deps[v]! do
    if st.indexOf[w]!.isNone then
      st := strongConnect deps w st
      st := { st with lowlink := st.lowlink.set! v (Nat.min st.lowlink[v]! st.lowlink[w]!) }
    else if st.onStack[w]! then
      st := { st with lowlink := st.lowlink.set! v (Nat.min st.lowlink[v]! (st.indexOf[w]!.getD 0)) }
  if some st.lowlink[v]! == st.indexOf[v]! then
    let mut scc : Array Nat := #[]
    let mut go := true
    while go do
      match st.stack.back? with
      | none => go := false
      | some w =>
        st := { st with stack := st.stack.pop, onStack := st.onStack.set! w false }
        scc := scc.push w
        if w == v then go := false
    st := { st with sccs := st.sccs.push scc }
  return st

/-- Port of `findInstanceCycles`: non-trivial SCCs of the inter-instance
    dep graph, as instance-name lists in SCC member order. -/
def findInstanceCycles (ea : ExprArena) (prog : Program) : Array (Array String) := Id.run do
  let insts : Array (String × Array InstanceInput) := prog.decls.filterMap fun d =>
    match d with
    | .inst name _ inputs => some (name, inputs)
    | _ => none
  if insts.isEmpty then return #[]
  -- One O(edges) sweep buys the dep walk's termination measure; every
  -- arena built through `eintern` is child-descending by construction,
  -- so the panic is an interning-order bug, never a user error.
  if hw : ea.wf then
    let n := insts.size
    let deps : Array (Array Nat) := insts.map fun (_, inputs) =>
      inputs.foldl (fun acc w => collectNestedOutDeps ea hw n acc w.value) #[]
    let mut st : TarjanSt := {
      indexOf := Array.replicate n none
      lowlink := Array.replicate n 0
      onStack := Array.replicate n false }
    for v in [0:n] do
      if st.indexOf[v]!.isNone then
        st := strongConnect deps v st
    let nontrivial := st.sccs.filter fun scc =>
      scc.size > 1 || (scc.size == 1 && (deps[scc[0]!]!).contains scc[0]!)
    return nontrivial.map (·.map fun i => insts[i]!.1)
  else
    panic! "findInstanceCycles: arena is not child-descending (internal interning-order bug)"

/-- The acyclic-source contract's error: name the cycle and say the
    true thing — there is no state primitive to break it through. -/
private def throwOnCycles (prog : Program) : ElabM Unit := do
  let cycles := findInstanceCycles (← get).arena.exprs prog
  if cycles.isEmpty then return
  let diagnostics := cycles.map fun scc =>
    let memberPath := String.intercalate " → " scc.toList
    s!"tropical: cycle in program '{prog.name}'\n" ++
    s!"  Instances in cycle: {memberPath}\n" ++
    "  There is no state primitive to break a cycle through — kernels are " ++
    "closed-form f(τ, params), so instance graphs must feed forward. " ++
    "Restructure the graph; recursive feedback on live input is outside this language."
  throw (.cycle ("tropical: strict cycle policy violated:\n" ++
    String.intercalate "\n\n" diagnostics.toList))

-- ─────────────────────────────────────────────────────────────
-- Instance inputs (second pass)
-- ─────────────────────────────────────────────────────────────

private def resolveInstanceInputs (scope : Scope) (instName : String) (typeKey : String)
    (inputs : Option (Array (String × ParsedExpr))) :
    ElabM (Array InstanceInput) := do
  let some targetIdx := (scope.registry.find? (·.1 == typeKey)).map (·.2)
    | throwElab s!"internal: instance '{instName}' typeKey '{typeKey}' not in scope registry"
  let target ← getProgram targetIdx
  -- Inputs: port NameRef → position in the target's ports.inputs.
  let mut ins : Array InstanceInput := #[]
  for (portName, valueExpr) in inputs.getD #[] do
    let some pos := target.inputs.findIdx? (·.name == portName)
      | let expected := if target.inputs.isEmpty then "(none)"
          else String.intercalate ", " (target.inputs.map (·.name)).toList
        throwElab s!"instance '{instName}': input '{portName}' is not a declared port of '{target.name}' (have: {expected})"
    if ins.any (·.port.idx == pos) then
      throwElab s!"instance '{instName}': duplicate input '{portName}'"
    let value ← resolveExpr scope valueExpr
    ins := ins.push { port := ⟨pos⟩, value }
  pure ins

-- ─────────────────────────────────────────────────────────────
-- Program elaboration
-- ─────────────────────────────────────────────────────────────

private def assocSet (a : Array (String × ProgramIdx)) (k : String) (v : ProgramIdx) :
    Array (String × ProgramIdx) :=
  match a.findIdx? (·.1 == k) with
  | some i => a.set! i (k, v)
  | none => a.push (k, v)

partial def elaborateProgram (p : Tropical.Parse.Program) (parents : List Frame)
    (resolver : Option Resolver) : ElabM ProgramIdx := do
  let mut frame : Frame := {}

  -- 1. Input + output ports. Defaults are resolved incrementally —
  --    a default sees earlier inputs, nothing else.
  let mut inputs : Array InputDecl := #[]
  for portSpec in (p.ports.bind (·.inputs)).getD #[] do
    let scope : Scope := { frame, parents, resolver, inputNames := inputs.map (·.name) }
    let decl ← match portSpec with
      | .bare name => pure { name : InputDecl }
      | .spec s => do
        let type? ← match s.type? with
          | none => pure none
          | some t => pure (some (← resolvePortType t))
        let default? ← match s.default? with
          | none => pure none
          | some d => pure (some (← resolveExpr scope d))
        pure { name := s.name, type?, default? : InputDecl }
    if inputs.any (·.name == decl.name) then
      throwElab s!"duplicate input port '{decl.name}'"
    inputs := inputs.push decl
  let mut outputs : Array OutputDecl := #[]
  for portSpec in (p.ports.bind (·.outputs)).getD #[] do
    let decl ← match portSpec with
      | .bare name => pure { name : OutputDecl }
      | .spec s => do
        let type? ← match s.type? with
          | none => pure none
          | some t => pure (some (← resolvePortType t))
        pure { name := s.name, type? : OutputDecl }
    if outputs.any (·.name == decl.name) then
      throwElab s!"duplicate output port '{decl.name}'"
    outputs := outputs.push decl

  -- 4. Register body decls. Pre-pass: ALL nested programDecls first
  --    (source order) — sibling instances may reference them.
  let mut decls : Array BodyDecl := #[]
  let mut pairing : Array (Nat × Tropical.Parse.BodyDecl) := #[]
  let mut paramsTbl : Array (String × Nat) := #[]
  let mut instTbl : Array (String × Nat × String) := #[]
  let mut registry : Array (String × ProgramIdx) := #[]
  for d in p.body.decls do
    if let .prog name inner := d then
      let innerIdx ← elaborateProgram inner (frame :: parents) resolver
      if frame.programs.any (·.1 == name) then
        throwElab s!"duplicate nested program '{name}'"
      frame := { frame with programs := frame.programs.push (name, innerIdx) }
      decls := decls.push (.prog name innerIdx)
  -- Then the rest, in source order: decl shells with placeholder
  -- expressions; per-kind de Bruijn levels recorded as we go.
  let mut paramCount := 0
  let mut instCount := 0
  for d in p.body.decls do
    match d with
    | .prog .. => pure ()  -- already handled
    | .param name value? => do
      if paramsTbl.any (·.1 == name) then
        throwElab s!"duplicate param '{name}'"
      let cell := decls.size
      decls := decls.push (.param name value?)
      pairing := pairing.push (cell, d)
      paramsTbl := paramsTbl.push (name, paramCount)
      paramCount := paramCount + 1
    | .inst name progName _ => do
      if instTbl.any (·.1 == name) then
        throwElab s!"duplicate instance '{name}'"
      let scope : Scope := { frame, parents, resolver }
      let targetIdx? : Option ProgramIdx :=
        match lookupProgramChain scope progName with
        | some t => some t
        | none => resolver.bind (· progName)
      let some targetIdx := targetIdx?
        | throwElab <|
            s!"instance '{name}': program type '{progName}' is not a nested program in scope " ++
            "and no external resolver provided it. Pass an ExternalProgramResolver to elaborate() " ++
            "to resolve cross-program references (e.g. stdlib types)."
      let target ← getProgram targetIdx
      let tk := target.name
      registry := assocSet registry tk targetIdx
      -- Transitive merge: the target's own registry entries in their
      -- order, skipping keys already present.
      for (k, v) in target.registry do
        if !registry.any (·.1 == k) then
          registry := registry.push (k, v)
      let cell := decls.size
      decls := decls.push (.inst name tk #[])
      pairing := pairing.push (cell, d)
      instTbl := instTbl.push (name, instCount, tk)
      instCount := instCount + 1

  -- 5. Resolve expressions inside body decls, in pairing order.
  let scope : Scope := {
    frame, parents, resolver
    inputNames := inputs.map (·.name)
    params := paramsTbl, instances := instTbl, registry }
  for (cell, parsed) in pairing do
    match parsed, decls[cell]! with
    | .param .., _ => pure ()  -- no expressions on paramDecl
    | .inst _ _ instInputs, .inst name tk _ => do
      let ins ← resolveInstanceInputs scope name tk instInputs
      decls := decls.set! cell (.inst name tk ins)
    | parsedD, resolvedD =>
      throwElab s!"internal: paired {parsedDeclOp parsedD} with {resolvedDeclOp resolvedD}"

  -- 6. Body assigns. (CF-only: the only assign is an output wire.)
  let mut assigns : Array OutputAssign := #[]
  for a in p.body.assigns do
    match a with
    | .output name exprParsed => do
      let target ← if name == "dac.out" then
          pure OutputTarget.dac
        else
          match outputs.findIdx? (·.name == name) with
          | some i => pure (OutputTarget.port ⟨i⟩)
          | none => throwElab s!"outputAssign references unknown output port '{name}'"
      assigns := assigns.push { target, expr := ← resolveExpr scope exprParsed }

  -- Build the program (mkProgram), validating registry coverage
  -- (validateProgramRegistry — same error text) before the cycle check.
  for d in decls do
    if let .inst iname tk _ := d then
      if !registry.any (·.1 == tk) then
        let keys := String.intercalate ", " (registry.map (·.1)).toList
        let keysShown := if keys.isEmpty then "(empty)" else keys
        throwElab <|
          s!"validateProgramRegistry: instance '{iname}' typeKey '{tk}' " ++
          s!"is not in the supplied program registry " ++
          s!"(keys present: {keysShown}). " ++
          "Construction site must add the target program to the registry before mkProgram/withDeclTables."
  let prog : Program := {
    name := p.name
    inputs, outputs
    decls, assigns
    registry }

  -- Strict cycle policy (Phase 4b): inter-instance cycles that don't
  -- pass through an explicit user register throw CycleViolation.
  throwOnCycles prog

  let idx : ProgramIdx := ⟨(← get).arena.programs.size⟩
  modify fun st => { st with
    arena := { st.arena with programs := st.arena.programs.push prog } }
  pure idx
where
  parsedDeclOp : Tropical.Parse.BodyDecl → String
    | .param .. => "paramDecl"
    | .inst .. => "instanceDecl" | .prog .. => "programDecl"
  resolvedDeclOp : BodyDecl → String
    | .param .. => "paramDecl"
    | .inst .. => "instanceDecl" | .prog .. => "programDecl"

-- ─────────────────────────────────────────────────────────────
-- Public entry
-- ─────────────────────────────────────────────────────────────

/-- Elaborate into an existing arena (the chain form: stdlib programs
    elaborated in dependency order share one arena, the resolver maps
    names to earlier results). -/
def elaborateInto (arena : Arena) (p : Tropical.Parse.Program)
    (resolver : Option Resolver := none) : Except ElabError (Arena × ProgramIdx) := do
  let (idx, st) ← (elaborateProgram p [] resolver).run { arena }
  pure (st.arena, idx)

/-- Port of `elaborate(prog, resolveExternalProgram?)`. -/
def elaborate (p : Tropical.Parse.Program) (resolver : Option Resolver := none) :
    Except ElabError (Arena × ProgramIdx) :=
  elaborateInto {} p resolver

end Tropical.Ir
