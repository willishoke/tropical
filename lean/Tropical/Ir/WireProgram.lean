import Tropical.WireExpr
import Tropical.Ir.Nodes

/-!
# Wire-program lift

Lift a wire expression to a raw resolved `Program`: one `InputDecl`
per free instance-output ref (sorted by canonical `instance:port`
key), one output `out`, and inline `ParamDecl`s for `param`/`trigger`
refs. Shape-identical to a user-authored single-assign program; the
lowering accepts it unmodified.

Pure and total (structural over `WireExpr`); error strings keep the TS
messages (they surface as `internal_error` envelopes). State-op and
retired-op refusals no longer appear here — the `WireExpr` decoder is
the refusal site, so this module only refuses the forms that are
storable but not liftable (`clock`, `broadcastTo`, the export-file and
session-slot forms).
-/

namespace Tropical.Ir.WireProgram

open Lean (Json JsonNumber)
open Tropical (WireExpr RefOut)

/-- A free instance-output reference discovered in a wire expression. -/
structure FreeRef where
  instanceName : String
  outputName : String

/-- The raw program synthesized from a wire expression, together with the
    information needed to attach it to the session graph. -/
structure LiftResult where
  program : Program
  freeRefs : Array FreeRef
  exprs : ExprArena

/-- `String(output)` for the freeRefs error message (numeric outputs
    render bare, as JS does). -/
private def refOutStr : RefOut → String
  | .name s => s
  | .index n => n.toString

/-- Port of `freeRefs`: every `ref(instance, output)` in the wire,
    deduplicated by canonical key in first-encounter order. Wire-form
    refs must carry string port names — numeric output indices are
    post-elaboration. -/
def freeRefs (expr : WireExpr) : Except String (Array FreeRef) :=
  walk #[] expr
where
  walk (acc : Array FreeRef) (e : WireExpr) :
      Except String (Array FreeRef) := do
    match e with
    | .ref inst (.name outName) =>
      pure <| if acc.any (fun ref =>
        ref.instanceName == inst && ref.outputName == outName)
        then acc
        else acc.push { instanceName := inst, outputName := outName }
    | .ref inst (.index n) =>
      throw <| s!"freeRefs: ref({inst}, {n.toString}) — output must be a " ++
        "string port name in wire-form, got number"
    | .arr items =>
      items.attach.foldlM (fun a ⟨x, _⟩ => walk a x) acc
    | .binary _ l r => walk (← walk acc l) r
    | .unary _ a | .broadcastTo a _ => walk acc a
    | .clamp a b c | .select a b c | .arraySet a b c =>
      walk (← walk (← walk acc a) b) c
    | .index a b => walk (← walk acc a) b
    | _ => pure acc
  termination_by sizeOf e
  decreasing_by
    all_goals first
      | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
      | (simp; omega)

/-- Port of `inferOutputPortType`: array-shaped wires get an
    array-typed output; `none` means scalar. (The old `delay()`
    auto-wrap unwrapping is gone with `delay` itself — the decoder
    refuses state ops.) -/
def inferOutputPortType (expr : WireExpr) : Option PortType :=
  match expr with
  | .arr items => some (.array .float #[⟨items.size, 0⟩])
  | _ => none

/-- Translation state: param decls (insertion-ordered, position =
    `ParamIdx`) plus the shared expression DAG the translated ids intern into.
    CF-only — the lifted body has no reg decls. -/
private structure Ctx where
  params : Array String := #[]
  exprs : ExprArena := {}

/-- Intern a node into the ctx's DAG, returning its id and the advanced ctx. -/
private def internW (n : ENode) (ctx : Ctx) : ExprId × Ctx :=
  let (id, ex) := (eintern n).run ctx.exprs
  (id, { ctx with exprs := ex })

private def wireKeyOf (inst port : String) : String := s!"{inst}:{port}"

/-- Port of `translateExpr`: wire expression → resolved id, interning into
    the ctx's DAG and threading the param accumulator. -/
private def translate (refToInput : Array (String × Nat))
    (e : WireExpr) (ctx : Ctx) : Except String (ExprId × Ctx) := do
  match e with
  | .num n => pure (internW (.num n) ctx)
  | .bool b => pure (internW (.bool b) ctx)
  | .arr items =>
    let (out, ctx) ← items.attach.foldlM (fun (acc, ctx) ⟨x, _⟩ => do
      let (t, ctx') ← translate refToInput x ctx
      pure (acc.push t, ctx')) ((#[] : Array ExprId), ctx)
    pure (internW (.arr out) ctx)
  | .ref inst output =>
    let key := wireKeyOf inst (refOutStr output)
    match refToInput.find? (·.1 == key) with
    | some (_, i) => pure (internW (.inputRef ⟨i⟩) ctx)
    | none =>
      throw <| s!"liftWireToProgram: ref {key} not in freeRefSet — " ++
        "pass the same set returned by freeRefs(expr)"
  | .param name | .trigger name =>
    let ctx := if ctx.params.contains name then ctx
               else { ctx with params := ctx.params.push name }
    let some pi := ctx.params.idxOf? name
      | throw "liftWireToProgram: param accumulator lost a name (internal)"
    pure (internW (.paramRef ⟨pi⟩) ctx)
  | .sampleRate => pure (internW .sampleRate ctx)
  | .sampleIndex => pure (internW .sampleIndex ctx)
  | .binary tag l r =>
    let (a, ctx) ← translate refToInput l ctx
    let (b, ctx) ← translate refToInput r ctx
    pure (internW (.binary tag a b) ctx)
  | .unary tag x =>
    let (a, ctx) ← translate refToInput x ctx
    pure (internW (.unary tag a) ctx)
  | .clamp x y z =>
    let (a, ctx) ← translate refToInput x ctx
    let (b, ctx) ← translate refToInput y ctx
    let (c, ctx) ← translate refToInput z ctx
    pure (internW (.clamp a b c) ctx)
  | .select x y z =>
    let (a, ctx) ← translate refToInput x ctx
    let (b, ctx) ← translate refToInput y ctx
    let (c, ctx) ← translate refToInput z ctx
    pure (internW (.select a b c) ctx)
  | .arraySet x y z =>
    let (a, ctx) ← translate refToInput x ctx
    let (b, ctx) ← translate refToInput y ctx
    let (c, ctx) ← translate refToInput z ctx
    pure (internW (.arraySet a b c) ctx)
  | .index x y =>
    let (a, ctx) ← translate refToInput x ctx
    let (b, ctx) ← translate refToInput y ctx
    pure (internW (.index a b) ctx)
  | .clock | .broadcastTo .. | .input _ | .nestedOut ..
  | .sessionSlot _ | .sessionArraySlot .. =>
    throw s!"liftWireToProgram: unhandled wire-form op '{e.opName}'"
termination_by sizeOf e
decreasing_by
  all_goals first
    | (have := Array.sizeOf_lt_of_mem ‹_ ∈ items›; simp; omega)
    | (simp; omega)

/-- Port of `liftWireToProgram`: build the raw lifted `Program`.
    Returns the program AND the sorted free refs (the caller wires
    each `instance__port` input back to its source). -/
def lift (expr : WireExpr) (synthName : String) (exprs0 : ExprArena := {}) :
    Except String LiftResult := do
  let refs ← freeRefs expr
  -- Sort by canonical key — deterministic input order across calls.
  let sortedRefs := refs.qsort fun a b =>
    wireKeyOf a.instanceName a.outputName < wireKeyOf b.instanceName b.outputName
  let inputDecls : Array InputDecl := sortedRefs.map fun ref =>
    -- Double-underscore separator avoids collisions with user port
    -- names; dots in instance paths flatten to underscores.
    { name := s!"{ref.instanceName.replace "." "_"}__{ref.outputName}" }
  let refToInput := sortedRefs.mapIdx fun i ref =>
    (wireKeyOf ref.instanceName ref.outputName, i)
  let outputDecl : OutputDecl := { name := "out", type? := inferOutputPortType expr }
  let (translated, ctx) ← translate refToInput expr { exprs := exprs0 }
  let prog : Program := {
    name := synthName
    inputs := inputDecls
    outputs := #[outputDecl]
    decls := ctx.params.map (BodyDecl.param · none)
    assigns := #[{ target := .port ⟨0⟩, expr := translated }]
    registry := #[] }
  return { program := prog, freeRefs := sortedRefs, exprs := ctx.exprs }

end Tropical.Ir.WireProgram
