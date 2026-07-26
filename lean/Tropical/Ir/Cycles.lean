import Tropical.Ir.Nodes

/-!
# Instance-graph cycle detection

`findCycle` / `findInstanceCycle?`: one cycle of the inter-instance
dependency graph (edges = `nestedOut` references inside instance input
wiring), if any exists. Cycles are always fatal here — the IR is
acyclic by construction and every boundary states it — so the
tripwires need *detection with a nameable loop*, not SCC computation:
a Kahn peel leaves exactly the nodes involved in or fed by cycles, and
a successor walk inside that remainder must close a loop within n+1
steps. Every loop is bounded by the node count, so the whole check is
total by construction (the Tarjan pair this replaced was the last
`partial` graph algorithm). Shared by the lowering's entry tripwire
(`Strata.assertAcyclic`), direct program registration (`ProgramIO`),
and the session tripwire (`Lowering.assertSessionAcyclic`).
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

/-- One cycle of a finite digraph, if any. `deps[v]` lists v's
    successors; out-of-range successors are ignored (they cannot sit on
    a cycle). Returns the loop as node indices in successor order, open
    (the caller closes it for rendering); a self-loop is `#[v]`.

    Kahn peel: repeatedly remove nodes whose live successors are all
    removed. Afterward every remaining node keeps at least one live
    successor, so a successor walk from any remaining node must revisit
    within n+1 steps — the revisited suffix is the cycle. Each node
    enqueues at most once and the walk is bounded, so every loop is a
    plain `for` over a range: total by construction, no measure to
    prove. -/
def findCycle (deps : Array (Array Nat)) : Option (Array Nat) := Id.run do
  let n := deps.size
  let live := fun (w : Nat) => w < n
  -- liveSucc[v] = v's not-yet-peeled in-range successors, as a count.
  let mut liveSucc : Array Nat := deps.map fun ss => (ss.filter live).size
  -- Reverse adjacency: preds[w] = the nodes listing w as a successor.
  let mut preds : Array (Array Nat) := Array.replicate n #[]
  for v in [0:n] do
    for w in deps[v]! do
      if live w then preds := preds.set! w (preds[w]!.push v)
  let mut peeled : Array Bool := Array.replicate n false
  let mut queue : Array Nat := #[]
  for v in [0:n] do
    if liveSucc[v]! == 0 then queue := queue.push v
  let mut qi := 0
  for _ in [0:n] do
    if qi < queue.size then
      let w := queue[qi]!
      qi := qi + 1
      peeled := peeled.set! w true
      for v in preds[w]! do
        if !peeled[v]! then
          let c := liveSucc[v]! - 1
          liveSucc := liveSucc.set! v c
          if c == 0 then queue := queue.push v
  -- Remainder walk: follow the first live successor until a revisit.
  match (Array.range n).find? (fun v => !peeled[v]!) with
  | none => return none
  | some s =>
    let mut posOf : Array (Option Nat) := Array.replicate n none
    let mut path : Array Nat := #[]
    let mut cur := s
    let mut result : Option (Array Nat) := none
    for _ in [0:n+1] do
      if result.isNone then
        match posOf[cur]! with
        | some p => result := some (path.extract p path.size)
        | none =>
          posOf := posOf.set! cur (some path.size)
          path := path.push cur
          match deps[cur]!.find? (fun w => live w && !peeled[w]!) with
          | some w => cur := w
          | none =>
            -- Unreachable: the peel invariant guarantees a live
            -- successor. Loud if the invariant ever breaks.
            result := panic! "findCycle: remainder node has no live successor (peel invariant broken)"
    return result

/-- Close an open loop for rendering: `[a, b] → "a → b → a"`. -/
def renderLoop (cycle : Array String) : String :=
  let loop := cycle ++ ((cycle[0]?).map (#[·])).getD #[]
  String.intercalate " → " loop.toList

/-- The acyclic-source contract's message: name the cycle and say the true
    thing — there is no state primitive to break it through. Shared by every
    boundary that enforces the contract on a constructed `Program`
    (`export_program`'s direct registration; formerly the elaborator's
    `CycleViolation`). -/
def cycleViolationMessage (progName : String) (cycle : Array String) : String :=
  "tropical: strict cycle policy violated:\n" ++
    s!"tropical: cycle in program '{progName}'\n" ++
    s!"  Instances in cycle: {renderLoop cycle}\n" ++
    "  There is no state primitive to break a cycle through — kernels are " ++
    "closed-form f(τ, params), so instance graphs must feed forward. " ++
    "Restructure the graph; recursive feedback on live input is outside this language."

/-- One cycle of the inter-instance dep graph, if any, as instance
    names in signal-flow order (producer → consumer, open — callers
    close the loop via `renderLoop`). -/
def findInstanceCycle? (ea : ExprArena) (prog : Program) : Option (Array String) := Id.run do
  let insts : Array (String × Array InstanceInput) := prog.decls.filterMap fun d =>
    match d with
    | .inst name _ inputs => some (name, inputs)
    | _ => none
  if insts.isEmpty then return none
  -- One O(edges) sweep buys the dep walk's termination measure; every
  -- arena built through `eintern` is child-descending by construction,
  -- so the panic is an interning-order bug, never a user error.
  if hw : ea.wf then
    let n := insts.size
    let deps : Array (Array Nat) := insts.map fun (_, inputs) =>
      inputs.foldl (fun acc w => collectNestedOutDeps ea hw n acc w.value) #[]
    -- `deps` edges point consumer → producer; reverse for flow order.
    return (findCycle deps).map fun cyc => cyc.reverse.map fun i => insts[i]!.1
  else
    panic! "findInstanceCycle?: arena is not child-descending (internal interning-order bug)"


end Tropical.Ir
