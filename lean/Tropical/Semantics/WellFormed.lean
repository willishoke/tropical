import Tropical.EmitArrow.Sig
import Tropical.Ir.Core
import Tropical.Semantics.Arena

/-!
# Builder and program well-formedness

This module makes the ownership convention on expression ids explicit.  The
predicates are deliberately layered: addressability, local index validity,
registry shape, and arena invariants remain separately reusable facts.

`BuildM` remains a public state monad, so arbitrary actions are not certified.
`PreservesBuilderWF` is instead a contract proved for the production actions.
-/

namespace Tropical.Semantics

open Tropical.Ir
open Tropical.Ir.Core
open Tropical.EmitArrow

/-- An expression id is addressable in an expression arena. -/
def ExprIdIn (arena : ExprArena) (id : ExprId) : Prop :=
  id.idx < arena.nodes.size

/-- An authored signal belongs to the active builder arena. -/
def SigIn (builder : Builder) (id : Sig) : Prop :=
  ExprIdIn builder.exprs id

/-- Every expression child of `node` belongs to `arena`. -/
def ENodeChildrenIn (arena : ExprArena) (node : ENode) : Prop :=
  ∀ child ∈ node.children, ExprIdIn arena child

/-- Every instance input stored by the builder belongs to its expression arena. -/
def BuilderDeclsWellFormed (builder : Builder) : Prop :=
  ∀ decl ∈ builder.decls, ∀ input ∈ decl.inputs, SigIn builder input.value

/-- The semantic arena invariants and all declaration roots owned by a builder. -/
structure BuilderWellFormed (builder : Builder) : Prop where
  arena : ArenaWellFormed builder.exprs
  decls : BuilderDeclsWellFormed builder

/-- `after` preserves both the old expression arena and old declaration prefix. -/
structure BuilderExtends (before after : Builder) : Prop where
  exprs : Extends before.exprs after.exprs
  decls : ∀ ⦃i : Nat⦄ ⦃decl⦄, before.decls[i]? = some decl →
    after.decls[i]? = some decl

/-- Qualified preservation contract for public builder actions.  No such claim
    is made for arbitrary inhabitants of the public `BuildM` abbreviation. -/
def PreservesBuilderWF {α : Type} (action : BuildM α) : Prop :=
  ∀ builder, BuilderWellFormed builder →
    match action.run builder with
    | .error _ => True
    | .ok (_, after) =>
      BuilderWellFormed after ∧ BuilderExtends builder after

theorem exprIdIn_iff_deref {arena : ExprArena} {id : ExprId} :
    ExprIdIn arena id ↔ ∃ node, arena.deref id = some node := by
  constructor
  · exact deref_of_index_lt
  · rintro ⟨node, h⟩
    exact deref_index_lt h

theorem BuilderExtends.sigIn {before after : Builder}
    (h : BuilderExtends before after) {id : Sig} (hid : SigIn before id) :
    SigIn after id := by
  change ExprIdIn before.exprs id at hid
  change ExprIdIn after.exprs id
  rw [exprIdIn_iff_deref] at hid ⊢
  obtain ⟨node, hnode⟩ := hid
  exact ⟨node, h.exprs hnode⟩

theorem BuilderExtends.refl (builder : Builder) :
    BuilderExtends builder builder := by
  constructor
  · exact Extends.refl builder.exprs
  · intro i decl h
    exact h

theorem BuilderExtends.trans {a b c : Builder}
    (hab : BuilderExtends a b) (hbc : BuilderExtends b c) :
    BuilderExtends a c := by
  constructor
  · exact Extends.trans hab.exprs hbc.exprs
  · intro i decl h
    exact hbc.decls (hab.decls h)

/-- The executable arena check proves precisely the child-descent component. -/
theorem childrenDescend_of_wf {arena : ExprArena} (h : arena.wf = true) :
    ChildrenDescend arena := by
  intro id node hd child hc
  exact ExprArena.forall_children_lt h hd child hc

theorem enodeChildrenIn_iff_childrenInPrefix {arena : ExprArena} {node : ENode} :
    ENodeChildrenIn arena node ↔ ChildrenInPrefix arena node := by
  rfl

-- Program roots --------------------------------------------------------------

/-- Every expression root stored directly in a resolved program is addressable. -/
def ProgramExprRefsIn (arena : ExprArena) (program : Program) : Prop :=
  (∀ input ∈ program.inputs, ∀ id ∈ input.default?, ExprIdIn arena id) ∧
  (∀ decl ∈ program.decls, match decl with
    | .inst _ _ inputs => ∀ input ∈ inputs, ExprIdIn arena input.value
    | _ => True) ∧
  (∀ assign ∈ program.assigns, ExprIdIn arena assign.expr)

/-- Local index references made by one expression node in a resolved program. -/
def ENodeIndicesWellFormed (programs : Array Program) (program : Program) :
    ENode → Prop
  | .inputRef idx => idx.idx < program.inputs.size
  | .paramRef idx => idx.idx < program.params.size
  | .nestedOut instance_ output =>
      ∃ name typeKey inputs target child,
        program.instances[instance_.idx]? = some (.inst name typeKey inputs) ∧
        program.registryGet? typeKey = some target ∧
        programs[target.idx]? = some child ∧
        output.idx < child.outputs.size
  | _ => True

/-- Every expression reachable from a root has valid program-local indices. -/
inductive ExprIndicesWellFormed (arena : ExprArena) (programs : Array Program)
    (program : Program) : ExprId → Prop where
  | intro {id node} :
      arena.deref id = some node →
      ENodeIndicesWellFormed programs program node →
      (∀ child ∈ node.children,
        ExprIndicesWellFormed arena programs program child) →
      ExprIndicesWellFormed arena programs program id

/-- All local expression, instance-input, and output-target indices are valid. -/
def ProgramIndicesWellFormed (arena : ExprArena) (programs : Array Program)
    (program : Program) : Prop :=
  (∀ input ∈ program.inputs, ∀ id ∈ input.default?,
    ExprIndicesWellFormed arena programs program id) ∧
  (∀ decl ∈ program.decls, match decl with
    | .inst _ typeKey inputs =>
        ∃ target child,
          program.registryGet? typeKey = some target ∧
          programs[target.idx]? = some child ∧
          ∀ input ∈ inputs,
            input.port.idx < child.inputs.size ∧
            ExprIndicesWellFormed arena programs program input.value
    | _ => True) ∧
  (∀ assign ∈ program.assigns,
    (match assign.target with
      | .port idx => idx.idx < program.outputs.size
      | .dac => True) ∧
    ExprIndicesWellFormed arena programs program assign.expr)

/-- Registry keys agree with their targets, instances resolve through the
    registry, and all explicit program-pool links descend from `index`. -/
def ProgramRegistryWellFormed (programs : Array Program) (index : Nat)
    (program : Program) : Prop :=
  (∀ child ∈ program.progChildren, child.idx < index) ∧
  (∀ entry ∈ program.registry, ∃ child,
    programs[entry.2.idx]? = some child ∧ entry.1 = child.name) ∧
  (∀ decl ∈ program.decls, match decl with
    | .inst _ typeKey _ => ∃ target child,
        program.registryGet? typeKey = some target ∧
        programs[target.idx]? = some child ∧ typeKey = child.name
    | .prog name target => ∃ child,
        programs[target.idx]? = some child ∧ name = child.name
    | .param .. => True)

/-- Full resolved-program certificate at one program-pool index. -/
structure ProgramWellFormed (arena : ExprArena) (programs : Array Program)
    (index : Nat) : Prop where
  exists_program : ∃ program, programs[index]? = some program
  arenaWellFormed : ArenaWellFormed arena
  programPoolWellFormed : progPoolWf programs = true
  expressionRefs : ∀ ⦃program⦄, programs[index]? = some program →
    ProgramExprRefsIn arena program
  indices : ∀ ⦃program⦄, programs[index]? = some program →
    ProgramIndicesWellFormed arena programs program
  registry : ∀ ⦃program⦄, programs[index]? = some program →
    ProgramRegistryWellFormed programs index program

-- Materialized CoreProgram roots ---------------------------------------------

/-- Every expression root stored directly in a materialized core program. -/
def CoreProgramExprRefsIn (arena : ExprArena) (program : CoreProgram) : Prop :=
  (∀ input ∈ program.inputs, ∀ id ∈ input.default?, ExprIdIn arena id) ∧
  (∀ decl ∈ program.decls, match decl with
    | .inst _ _ inputs => ∀ input ∈ inputs, ExprIdIn arena input.value
    | _ => True) ∧
  (∀ assign ∈ program.assigns, ExprIdIn arena assign.expr)

/-- Local index references made by one core expression node. -/
def CoreENodeIndicesWellFormed (program : CoreProgram) : ENode → Prop
  | .inputRef idx => idx.idx < program.inputs.size
  | .paramRef idx => idx.idx < program.params.size
  | .nestedOut instance_ output =>
      ∃ name typeKey inputs child,
        program.instances[instance_.idx]? = some (.inst name typeKey inputs) ∧
        program.registryGet? typeKey = some child ∧
        output.idx < child.outputs.size
  | _ => True

inductive CoreExprIndicesWellFormed (arena : ExprArena) (program : CoreProgram) :
    ExprId → Prop where
  | intro {id node} :
      arena.deref id = some node →
      CoreENodeIndicesWellFormed program node →
      (∀ child ∈ node.children,
        CoreExprIndicesWellFormed arena program child) →
      CoreExprIndicesWellFormed arena program id

def CoreProgramIndicesWellFormed (arena : ExprArena)
    (program : CoreProgram) : Prop :=
  (∀ input ∈ program.inputs, ∀ id ∈ input.default?,
    CoreExprIndicesWellFormed arena program id) ∧
  (∀ decl ∈ program.decls, match decl with
    | .inst _ typeKey inputs => ∃ child,
        program.registryGet? typeKey = some child ∧
        ∀ input ∈ inputs,
          input.port.idx < child.inputs.size ∧
          CoreExprIndicesWellFormed arena program input.value
    | _ => True) ∧
  (∀ assign ∈ program.assigns,
    (match assign.target with
      | .port idx => idx.idx < program.outputs.size
      | .dac => True) ∧
    CoreExprIndicesWellFormed arena program assign.expr)

def CoreProgramRegistryWellFormed (program : CoreProgram) : Prop :=
  (∀ entry ∈ program.registry, entry.1 = entry.2.name) ∧
  (∀ decl ∈ program.decls, match decl with
    | .inst _ typeKey _ => ∃ child,
        program.registryGet? typeKey = some child ∧ typeKey = child.name
    | _ => True)

/-- Recursive well-formedness for the materialized core-program tree. -/
inductive CoreProgramWellFormed (arena : ExprArena) : CoreProgram → Prop where
  | intro {program} :
      ArenaWellFormed arena →
      CoreProgramExprRefsIn arena program →
      CoreProgramIndicesWellFormed arena program →
      CoreProgramRegistryWellFormed program →
      (∀ entry ∈ program.registry, CoreProgramWellFormed arena entry.2) →
      CoreProgramWellFormed arena program

end Tropical.Semantics
