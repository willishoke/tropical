/-!
# Parallel-universe semantics for deferred modal forests

This module states the temporal law independently of a numeric carrier or a
particular room implementation.  It is a canonical semantic obligation, not a
refinement theorem about the current production representation.  A live
parameter family selects one complete static universe at the terminal
observation coordinate under the enclosing clock context.  Sources and an
authored-order, heterogeneous stage spine then use that one frozen parameter
vector while evaluating each branch at its own response coordinate.

The model deliberately distinguishes two coordinates:

* `terminal` selects the static universe;
* `response` drives the modal source and room-local clocked behavior.

An absolute branch address changes only `response`.  A downstream graph warp
composes with the enclosing clock context before controls, authored source
clocks, or address terms are evaluated.  There is no parameter-history or
state constructor in this model.
-/

namespace Tropical.Semantics.ModalUniverse

universe uParam uCoord uOutput uSource uStage uMetadata uValue

-- ── Static universes and their live diagonal ────────────────────────────────

/-- The clock transformation accumulated by enclosing ordinary graph warps. -/
abbrev ClockContext (Coord : Type uCoord) := Coord → Coord

/-- The unwarped enclosing clock context. -/
def ClockContext.identity {Coord : Type uCoord} : ClockContext Coord := id

/-- Push an ordinary graph warp into the enclosing context.  The order is the
one used by `ArrowTerm.emitTermC`: the existing context is applied after the
newly encountered warp. -/
def ClockContext.pushWarp {Coord : Type uCoord} (context : ClockContext Coord)
    (warp : Coord → Coord) : ClockContext Coord :=
  fun coordinate => context (warp coordinate)

@[simp] theorem ClockContext.identity_apply
    {Coord : Type uCoord} (coordinate : Coord) :
    ClockContext.identity coordinate = coordinate := rfl

@[simp] theorem ClockContext.pushWarp_apply
    {Coord : Type uCoord} (context : ClockContext Coord)
    (warp : Coord → Coord) (coordinate : Coord) :
    (context.pushWarp warp) coordinate = context (warp coordinate) := rfl

/-- Every live control is evaluated under the enclosing context and through
its own authored clock/warp path, abstracted by this family. -/
abbrev LiveParameters (Params : Type uParam) (Coord : Type uCoord) :=
  ClockContext Coord → Coord → Params

/-- Sample the complete live parameter family at one terminal/source
coordinate under the enclosing context. -/
def freeze {Params : Type uParam} {Coord : Type uCoord}
    (parameters : LiveParameters Params Coord) (context : ClockContext Coord)
    (terminal : Coord) : Params :=
  parameters context terminal

/-- A family of static renderers, one for each complete parameter vector. -/
abbrev StaticRenderer (Params : Type uParam) (Coord : Type uCoord)
    (Output : Type uOutput) :=
  Params → ClockContext Coord → Coord → Output

/-- The live renderer is the terminal-coordinate diagonal of static universes. -/
def diagonal {Params : Type uParam} {Coord : Type uCoord}
    {Output : Type uOutput}
    (renderStatic : StaticRenderer Params Coord Output)
    (parameters : LiveParameters Params Coord) (context : ClockContext Coord)
    (terminal : Coord) : Output :=
  renderStatic (freeze parameters context terminal) context terminal

/-- Universal equality of static universes lifts pointwise to arbitrary live
parameter signals on their diagonal. -/
theorem pointwise_lifts_to_diagonal
    {Params : Type uParam} {Coord : Type uCoord} {Output : Type uOutput}
    {lhs rhs : StaticRenderer Params Coord Output}
    (hStatic : ∀ parameters, lhs parameters = rhs parameters)
    (live : LiveParameters Params Coord) (context : ClockContext Coord)
    (terminal : Coord) :
    diagonal lhs live context terminal = diagonal rhs live context terminal := by
  exact congrFun (congrFun (hStatic (freeze live context terminal)) context)
    terminal

-- ── Coordinate law ──────────────────────────────────────────────────────────

/-- The two coordinates carried through terminal realization. -/
structure Coordinates (Coord : Type uCoord) where
  terminal : Coord
  response : Coord

/-- An address term produces an absolute response clock after being evaluated
through its own authored path under the enclosing context.  The enclosing
context is subsequently applied once more by `swarp`. -/
abbrev AddressSignal (Coord : Type uCoord) :=
  ClockContext Coord → Coord → Coord

-- ── Canonical deferred forest ────────────────────────────────────────────────

/-- Everything about a branch that appending a modal stage must preserve.
`metadata` abstracts the strike anchor, capacity/count facts, and any further
source truth owned by a production representation. -/
structure BranchSpine (Source : Type uSource) (Coord : Type uCoord)
    (Metadata : Type uMetadata) where
  source : Source
  metadata : Metadata
  realizationClock : Coord → Coord
  address? : Option (AddressSignal Coord) := none

/-- One independently timed source plus its authored-order modal stage spine. -/
structure Branch (Source : Type uSource) (Stage : Type uStage)
    (Coord : Type uCoord) (Metadata : Type uMetadata) where
  spine : BranchSpine Source Coord Metadata
  stages : Array Stage := #[]

/-- Authored branch order is semantic and is retained as an array. -/
abbrev Forest (Source : Type uSource) (Stage : Type uStage)
    (Coord : Type uCoord) (Metadata : Type uMetadata) :=
  Array (Branch Source Stage Coord Metadata)

/-- Append one stage without touching source truth or realization metadata. -/
def Branch.appendStage
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata}
    (branch : Branch Source Stage Coord Metadata) (stage : Stage) :
    Branch Source Stage Coord Metadata :=
  { branch with stages := branch.stages.push stage }

/-- The complete source/metadata spine survives a stage append definitionally. -/
@[simp] theorem appendStage_preserves_metadata
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata}
    (branch : Branch Source Stage Coord Metadata) (stage : Stage) :
    (branch.appendStage stage).spine = branch.spine := rfl

/-- Stage append is authored-order array append by one. -/
@[simp] theorem appendStage_preserves_order
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata}
    (branch : Branch Source Stage Coord Metadata) (stage : Stage) :
    (branch.appendStage stage).stages = branch.stages.push stage := rfl

/-- Resolve one branch's response coordinate under the enclosing clock
context.  Without an address, the context is applied to the authored branch
clock.  With an address, the address term first evaluates under the same
context and produces an absolute clock; `swarp` then applies the context to
that clock.  In either case the terminal/source coordinate is unchanged. -/
def BranchSpine.coordinatesAt
    {Source : Type uSource} {Coord : Type uCoord}
    {Metadata : Type uMetadata}
    (spine : BranchSpine Source Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) :
    Coordinates Coord :=
  { terminal
    response :=
      match spine.address? with
      | none => context (spine.realizationClock terminal)
      | some address => context (address context terminal) }

@[simp] theorem BranchSpine.coordinatesAt_terminal
    {Source : Type uSource} {Coord : Type uCoord}
    {Metadata : Type uMetadata}
    (spine : BranchSpine Source Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) :
    (spine.coordinatesAt context terminal).terminal = terminal := rfl

theorem BranchSpine.coordinatesAt_response
    {Source : Type uSource} {Coord : Type uCoord}
    {Metadata : Type uMetadata}
    (spine : BranchSpine Source Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) :
    (spine.coordinatesAt context terminal).response =
      match spine.address? with
      | none => context (spine.realizationClock terminal)
      | some address => context (address context terminal) := rfl

/-- Branch clocks and addresses never retime independent control sources. -/
theorem BranchSpine.coordinatesAt_preserves_freeze
    {Params : Type uParam} {Source : Type uSource} {Coord : Type uCoord}
    {Metadata : Type uMetadata}
    (parameters : LiveParameters Params Coord)
    (spine : BranchSpine Source Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) :
    freeze parameters context (spine.coordinatesAt context terminal).terminal =
      freeze parameters context terminal := by
  rw [spine.coordinatesAt_terminal]

/-- The graph fragment relevant to deferred modal lowering.  `Stage` is wholly
generic: it may be an inductive containing ordinary rooms, grouped rooms,
gauges, and filter-as-room stages, so the laws below cover heterogeneous and
repeated sequences without a commutativity premise. -/
inductive Graph (Source : Type uSource) (Stage : Type uStage)
    (Coord : Type uCoord) (Metadata : Type uMetadata) where
  | source (branch : Branch Source Stage Coord Metadata)
  | stage (input : Graph Source Stage Coord Metadata) (stage : Stage)
  | mix (inputs : Array (Graph Source Stage Coord Metadata))

/-- Canonical structural lowering: stages map over every branch and modalMix
concatenates forests in authored input order. -/
def lowerGraph
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata} :
    Graph Source Stage Coord Metadata → Forest Source Stage Coord Metadata
  | .source branch => #[branch]
  | .stage input stage =>
      (lowerGraph input).map fun branch => branch.appendStage stage
  | .mix inputs =>
      inputs.attach.foldl
        (fun forest input => forest ++ lowerGraph input.1) #[]
termination_by graph => sizeOf graph
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem input.2; simp_all; omega)

@[simp] theorem lower_source
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata}
    (branch : Branch Source Stage Coord Metadata) :
    lowerGraph (.source branch) = #[branch] := by
  rw [lowerGraph]

@[simp] theorem lower_stage_maps_every_branch
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata}
    (input : Graph Source Stage Coord Metadata) (stage : Stage) :
    lowerGraph (.stage input stage) =
      (lowerGraph input).map fun branch => branch.appendStage stage := by
  rw [lowerGraph]

/-- `foldl (++)` fixes modalMix's authored input and branch order; no flattening
through, fusion of, or commutation with the stage spine is assumed. -/
theorem lower_mix_preserves_authored_order
    {Source : Type uSource} {Stage : Type uStage}
    {Coord : Type uCoord} {Metadata : Type uMetadata}
    (inputs : Array (Graph Source Stage Coord Metadata)) :
    lowerGraph (.mix inputs) =
      inputs.attach.foldl
        (fun forest input => forest ++ lowerGraph input.1) #[] := by
  rw [lowerGraph]

-- ── Static and live realization ──────────────────────────────────────────────

/-- Carrier-generic semantics for sources, heterogeneous stages, and the final
authored-order branch sum.  Stages see only the response coordinate; the frozen
parameter vector is their sole access to live numeric controls. -/
structure StaticSemantics
    (Params : Type uParam) (Coord : Type uCoord)
    (Source : Type uSource) (Stage : Type uStage) (Value : Type uValue) where
  source : Source → Params → Coord → Value
  stage : Stage → Params → Coord → Value → Value
  zero : Value
  mix : Value → Value → Value

/-- Apply a static universe's stages in authored order. -/
def renderStagesStatic
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : Params) (response : Coord) : List Stage → Value → Value
  | [], value => value
  | stage :: stages, value =>
      renderStagesStatic semantics parameters response stages
        (semantics.stage stage parameters response value)

/-- Compositional live reading: every stage consults the same terminal
coordinate.  The theorem below collapses these reads to one whole-graph freeze. -/
def renderStagesLive
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (coordinates : Coordinates Coord) :
    List Stage → Value → Value
  | [], value => value
  | stage :: stages, value =>
      renderStagesLive semantics parameters context coordinates stages
        (semantics.stage stage
          (freeze parameters context coordinates.terminal)
          coordinates.response value)

/-- Ordered heterogeneous/repeated stage realization commutes with one freeze. -/
theorem renderStages_commutes_freeze
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (coordinates : Coordinates Coord)
    (stages : List Stage) (initial : Value) :
    renderStagesLive semantics parameters context coordinates stages initial =
      renderStagesStatic semantics
        (freeze parameters context coordinates.terminal)
        coordinates.response stages initial := by
  induction stages generalizing initial with
  | nil => rfl
  | cons stage stages ih =>
      exact ih _

/-- Render one branch in a selected static universe. -/
def renderBranchStatic
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : Params) (branch : Branch Source Stage Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) : Value :=
  let coordinates := branch.spine.coordinatesAt context terminal
  renderStagesStatic semantics parameters coordinates.response
    branch.stages.toList
    (semantics.source branch.spine.source parameters coordinates.response)

/-- Render one branch by compositional live parameter reads. -/
def renderBranchLive
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (branch : Branch Source Stage Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) : Value :=
  let coordinates := branch.spine.coordinatesAt context terminal
  renderStagesLive semantics parameters context coordinates branch.stages.toList
    (semantics.source branch.spine.source
      (freeze parameters context coordinates.terminal) coordinates.response)

/-- A branch clock/address changes response evaluation but every source and
stage still sees the one parameter vector frozen at the terminal coordinate. -/
theorem renderBranch_commutes_freeze
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (branch : Branch Source Stage Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) :
    renderBranchLive semantics parameters branch context terminal =
      renderBranchStatic semantics (freeze parameters context terminal) branch
        context terminal := by
  unfold renderBranchLive renderBranchStatic
  rw [renderStages_commutes_freeze]
  rw [branch.spine.coordinatesAt_preserves_freeze]

/-- Left-fold a static forest in authored branch order. -/
def renderBranchesStaticFrom
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : Params) (context : ClockContext Coord) (terminal : Coord) :
    List (Branch Source Stage Coord Metadata) → Value → Value
  | [], accumulated => accumulated
  | branch :: branches, accumulated =>
      renderBranchesStaticFrom semantics parameters context terminal branches
        (semantics.mix accumulated
          (renderBranchStatic semantics parameters branch context terminal))

/-- Left-fold a live forest in the same authored branch order. -/
def renderBranchesLiveFrom
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (terminal : Coord) :
    List (Branch Source Stage Coord Metadata) → Value → Value
  | [], accumulated => accumulated
  | branch :: branches, accumulated =>
      renderBranchesLiveFrom semantics parameters context terminal branches
        (semantics.mix accumulated
          (renderBranchLive semantics parameters branch context terminal))

theorem renderBranchesFrom_commutes_freeze
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (terminal : Coord)
    (branches : List (Branch Source Stage Coord Metadata))
    (accumulated : Value) :
    renderBranchesLiveFrom semantics parameters context terminal branches
        accumulated =
      renderBranchesStaticFrom semantics (freeze parameters context terminal)
        context terminal branches accumulated := by
  induction branches generalizing accumulated with
  | nil => rfl
  | cons branch branches ih =>
      rw [renderBranchesLiveFrom, renderBranchesStaticFrom,
        renderBranch_commutes_freeze]
      exact ih _

/-- Static terminal realization of an authored-order forest. -/
def renderForestStatic
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : Params) (forest : Forest Source Stage Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) : Value :=
  renderBranchesStaticFrom semantics parameters context terminal forest.toList
    semantics.zero

/-- Live terminal realization of an authored-order forest. -/
def renderForestLive
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (forest : Forest Source Stage Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) : Value :=
  renderBranchesLiveFrom semantics parameters context terminal forest.toList
    semantics.zero

/-- Source, stage, modalMix, branch-coordinate, and terminal composition all
commute with one whole-forest freeze. -/
theorem renderForest_commutes_freeze
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (parameters : LiveParameters Params Coord)
    (forest : Forest Source Stage Coord Metadata)
    (context : ClockContext Coord) (terminal : Coord) :
    renderForestLive semantics parameters forest context terminal =
      renderForestStatic semantics (freeze parameters context terminal) forest
        context terminal :=
  renderBranchesFrom_commutes_freeze semantics parameters context terminal
    forest.toList semantics.zero

/-- Compile and render a canonical graph in one selected static universe. -/
def compileStatic
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (graph : Graph Source Stage Coord Metadata) (parameters : Params)
    (context : ClockContext Coord) (terminal : Coord) : Value :=
  renderForestStatic semantics parameters (lowerGraph graph) context terminal

/-- Compile and render the same graph from live parameter signals. -/
def compileLive
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (graph : Graph Source Stage Coord Metadata)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (terminal : Coord) : Value :=
  renderForestLive semantics parameters (lowerGraph graph) context terminal

/-- Compiler-commutation capstone: compositional live lowering equals lowering
the complete graph once in the static universe frozen at the terminal. -/
theorem compileLive_eq_compileStatic_freeze
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (graph : Graph Source Stage Coord Metadata)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (terminal : Coord) :
    compileLive semantics graph parameters context terminal =
      compileStatic semantics graph (freeze parameters context terminal) context
        terminal :=
  renderForest_commutes_freeze semantics parameters (lowerGraph graph) context
    terminal

/-- Context-parametric corollary for a downstream ordinary graph warp.  The
warp composes into the context before the graph is interpreted; it is not a
substitution of `warp terminal` for `terminal`.  Consequently authored branch
clocks, address terms, and independent controls all receive the same pushed
context while retaining their distinct authored paths. -/
theorem compileLive_pushWarp_eq_compileStatic_freeze
    {Params : Type uParam} {Coord : Type uCoord}
    {Source : Type uSource} {Stage : Type uStage}
    {Metadata : Type uMetadata} {Value : Type uValue}
    (semantics : StaticSemantics Params Coord Source Stage Value)
    (graph : Graph Source Stage Coord Metadata)
    (parameters : LiveParameters Params Coord)
    (context : ClockContext Coord) (warp : Coord → Coord)
    (terminal : Coord) :
    compileLive semantics graph parameters (context.pushWarp warp) terminal =
      compileStatic semantics graph
        (freeze parameters (context.pushWarp warp) terminal)
        (context.pushWarp warp) terminal :=
  compileLive_eq_compileStatic_freeze semantics graph parameters
    (context.pushWarp warp) terminal

end Tropical.Semantics.ModalUniverse
