import Tropical.Ir.Nodes

/-!
# Instance-graph cycle detection

`findInstanceCycles`: the non-trivial SCCs of a program's
inter-instance dependency graph (edges = `nestedOut` references inside
instance input wiring), as instance-name lists in SCC member order.
Shared by the elaborator's acyclic-source contract (`CycleViolation`)
and the lowering's entry tripwire (`Strata.assertAcyclic`) — the IR is
acyclic by construction, and both boundaries state it.
-/

namespace Tropical.Ir

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

/-- The acyclic-source contract's message: name the cycle and say the true
    thing — there is no state primitive to break it through. Shared by every
    boundary that enforces the contract on a constructed `Program`
    (`export_program`'s direct registration; formerly the elaborator's
    `CycleViolation`). -/
def cycleViolationMessage (progName : String) (cycles : Array (Array String)) : String :=
  let diagnostics := cycles.map fun scc =>
    let memberPath := String.intercalate " → " scc.toList
    s!"tropical: cycle in program '{progName}'\n" ++
    s!"  Instances in cycle: {memberPath}\n" ++
    "  There is no state primitive to break a cycle through — kernels are " ++
    "closed-form f(τ, params), so instance graphs must feed forward. " ++
    "Restructure the graph; recursive feedback on live input is outside this language."
  "tropical: strict cycle policy violated:\n" ++
    String.intercalate "\n\n" diagnostics.toList

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


end Tropical.Ir
