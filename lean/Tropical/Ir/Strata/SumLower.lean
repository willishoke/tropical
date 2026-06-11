import Tropical.Ir.Nodes
import Tropical.Ir.Strata.Basic

/-!
# sumLower — port of compiler/ir/sum_lower.ts (Phase 5 pass 2)

Decomposes every sum-typed `RegDecl` (one whose `init` is a `tag`)
into N+1 scalar regs — a discriminator slot (int) plus one slot per
(variant, field) pair across all variants — and lowers every
`match`/`tag` to scalar select-chains and variant-index literals.

Three pure phases, mirroring the TS structure: `buildSpecs` (assign
the new RegIdx layout), `buildNewDecls` (construct replacements with
init/update set), assign rewriting. Refs resolve via the spec table
(old reg position → spec → new idx).

TS identity discipline, mirrored exactly where it is observable:
  - variant/field `===` checks → (typeDef pool idx, position) equality
  - `slotKey` map lookups → NAME-keyed (`variant__field`), because TS
    keys those maps by name strings — a cross-type lookup (match arms
    over a different sum type than the scrutinee reg's) matches by
    name in TS, and must here too.

Errors are comparable outputs (byte-exact TS messages).
-/

namespace Tropical.Ir.Strata.SumLower

open Lean (JsonNumber)
open Tropical.Ir

private def nat0 (n : Nat) : Expr := .num ⟨Int.ofNat n, 0⟩

private def zeroE : Expr := nat0 0

-- ─────────────────────────────────────────────────────────────
-- Specs — pure structural data describing each old reg's new shape
-- ─────────────────────────────────────────────────────────────

/-- A single bundle slot replacing one sum-typed `RegDecl`.
    `variant?`/`field?` are `(position, name)` pairs; `none` for the
    tag slot. -/
private structure SlotSpec where
  idx : RegIdx
  variant? : Option (Nat × String) := none
  field? : Option (Nat × String) := none
deriving Inhabited

private structure SumRegInfo where
  def_ : TypeDefIdx
  sumName : String
  variants : Array SumVariant
  /-- Slot order: [tag, ...per-variant per-field-in-payload-order]. -/
  slots : Array SlotSpec
  tagSlot : SlotSpec
deriving Inhabited

/-- TS `payloadByKey.get(slotKey(variant, field))` — name-keyed. -/
private def SumRegInfo.payloadByKey (info : SumRegInfo)
    (variantName fieldName : String) : Option SlotSpec :=
  info.slots.find? fun s =>
    s.variant?.any (·.2 == variantName) && s.field?.any (·.2 == fieldName)

private inductive RegSpec where
  | sum (info : SumRegInfo)
  | nonSum (idx : RegIdx)
deriving Inhabited

/-- Aligned with the OLD program's reg-table positions. -/
private abbrev SpecMap := Array RegSpec

private def mangle (base suffix : String) : String := s!"{base}#{suffix}"

/-- The sum TypeDef behind a reg's `init`, when the init is a `tag`
    whose def resolves to a sum (TS: `init.variant.parent`). -/
private def sumTypeOfRegInit (arena : Arena) (init : Expr) :
    Option (TypeDefIdx × String × Array SumVariant) :=
  match init with
  | .tag d _ _ =>
    match arena.typeDef? d with
    | some (.sum name variants) => some (d, name, variants)
    | _ => none
  | _ => none

-- ─────────────────────────────────────────────────────────────
-- Phase 1 — spec collection
-- ─────────────────────────────────────────────────────────────

private def buildSpecs (arena : Arena) (decls : Array BodyDecl) : SpecMap := Id.run do
  let mut specs : SpecMap := #[]
  let mut nextIdx := 0
  for decl in decls do
    match decl with
    | .reg _ init _ _ _ =>
      match sumTypeOfRegInit arena init with
      | some (d, sumName, variants) =>
        let tagSlot : SlotSpec := { idx := ⟨nextIdx⟩ }
        nextIdx := nextIdx + 1
        let mut slots : Array SlotSpec := #[tagSlot]
        for hv : vi in [0:variants.size] do
          let v := variants[vi]
          for hf : fi in [0:v.payload.size] do
            let f := v.payload[fi]
            slots := slots.push
              { idx := ⟨nextIdx⟩, variant? := some (vi, v.name), field? := some (fi, f.name) }
            nextIdx := nextIdx + 1
        specs := specs.push (.sum { def_ := d, sumName, variants, slots, tagSlot })
      | none =>
        specs := specs.push (.nonSum ⟨nextIdx⟩)
        nextIdx := nextIdx + 1
    | _ => pure ()
  return specs

-- ─────────────────────────────────────────────────────────────
-- Rewriting context
-- ─────────────────────────────────────────────────────────────

private structure Ctx where
  arena : Arena
  /-- Old reg table (regDecls of the input in order); resolves old
      `RegRef.idx`. -/
  oldRegs : Array BodyDecl
  specs : SpecMap
  /-- Active per-binder substitutions from match arms, innermost
      first (lookup is find-first, so inner shadows outer). -/
  bindings : List (BinderIdx × Expr) := []

private def Ctx.spec? (ctx : Ctx) (oldIdx : Nat) : Option RegSpec :=
  ctx.specs[oldIdx]?

private def Ctx.withBindings (ctx : Ctx) (extra : List (BinderIdx × Expr)) : Ctx :=
  if extra.isEmpty then ctx else { ctx with bindings := extra ++ ctx.bindings }

/-- Resolve a match's sum def: variants + name. -/
private def sumDef (ctx : Ctx) (d : TypeDefIdx) :
    Except Error (String × Array SumVariant) :=
  match ctx.arena.typeDef? d with
  | some (.sum name variants) => .ok (name, variants)
  | _ => .error ⟨s!"sumLower: typeDef pool index {d.idx} is not a sum type (internal)"⟩

-- ─────────────────────────────────────────────────────────────
-- Expression rewriting (mutually recursive with match lowering)
-- ─────────────────────────────────────────────────────────────

mutual

private partial def rewriteExpr (ctx : Ctx) (e : Expr) : Except Error Expr := do
  match e with
  | .num _ | .bool _ => return e
  | .arr items => return .arr (← items.mapM (rewriteExpr ctx))
  -- Bindings: substitute when the binder is in the active map.
  | .bindingRef i =>
    match ctx.bindings.find? (·.1 == i) with
    | some (_, sub) => return sub
    | none => return e
  -- RegRef: remap to the new RegIdx via the spec map (tag slot for a
  -- sum-typed source; the fresh decl's idx for non-sum).
  | .regRef i =>
    let some srcDecl := ctx.oldRegs[i.idx]?
      | throw ⟨s!"sumLower: regRef idx={i.idx} has no source in input program"⟩
    let _ := srcDecl
    match ctx.spec? i.idx with
    | none => return e
    | some (.sum info) => return .regRef info.tagSlot.idx
    | some (.nonSum idx) => return .regRef idx
  -- Tag in expression position (no payload) → variant index.
  | .tag d variant payload =>
    let (_, variants) ← sumDef ctx d
    let some v := variants[variant]?
      | throw ⟨s!"sumLower: tag variant index {variant} out of range (internal)"⟩
    if payload.isEmpty then
      return nat0 variant
    throw ⟨s!"sumLower: bare tag with payload (variant '{v.name}') in non-update context"⟩
  -- Match: lower to a scalar select chain.
  | .match_ d scrutinee arms => lowerMatchToSelectChain ctx d scrutinee arms
  -- Pass-through references / leaves.
  | .inputRef _ | .paramRef _ | .typeParamRef _
  | .sampleRate | .sampleIndex | .nestedOut _ _ => return e
  -- Operators.
  | .binary tag lhs rhs => return .binary tag (← rewriteExpr ctx lhs) (← rewriteExpr ctx rhs)
  | .unary tag arg => return .unary tag (← rewriteExpr ctx arg)
  | .clamp a b c => return .clamp (← rewriteExpr ctx a) (← rewriteExpr ctx b) (← rewriteExpr ctx c)
  | .select a b c => return .select (← rewriteExpr ctx a) (← rewriteExpr ctx b) (← rewriteExpr ctx c)
  | .index a b => return .index (← rewriteExpr ctx a) (← rewriteExpr ctx b)
  | .arraySet a b c => return .arraySet (← rewriteExpr ctx a) (← rewriteExpr ctx b) (← rewriteExpr ctx c)
  | .zeros count => return .zeros (← rewriteExpr ctx count)
  -- Combinators (binders pass through; bodies get rewritten).
  | .fold over init acc elem body =>
    return .fold (← rewriteExpr ctx over) (← rewriteExpr ctx init) acc elem
      (← rewriteExpr ctx body)
  | .scan over init acc elem body =>
    return .scan (← rewriteExpr ctx over) (← rewriteExpr ctx init) acc elem
      (← rewriteExpr ctx body)
  | .generate count iter body =>
    return .generate (← rewriteExpr ctx count) iter (← rewriteExpr ctx body)
  | .iterate count init iter body =>
    return .iterate (← rewriteExpr ctx count) (← rewriteExpr ctx init) iter
      (← rewriteExpr ctx body)
  | .chain count init iter body =>
    return .chain (← rewriteExpr ctx count) (← rewriteExpr ctx init) iter
      (← rewriteExpr ctx body)
  | .map2 over elem body =>
    return .map2 (← rewriteExpr ctx over) elem (← rewriteExpr ctx body)
  | .zipWith a b x y body =>
    return .zipWith (← rewriteExpr ctx a) (← rewriteExpr ctx b) x y
      (← rewriteExpr ctx body)
  | .letIn binders body =>
    return .letIn
      (← binders.mapM fun b => do pure (.mk b.binder (← rewriteExpr ctx b.value)))
      (← rewriteExpr ctx body)

/-- Lower a `match` to a select chain over the scrutinee's tag read. -/
private partial def lowerMatchToSelectChain (ctx : Ctx) (d : TypeDefIdx)
    (scrutinee : Expr) (arms : Array MatchArm) : Except Error Expr := do
  let tagRead ← rewriteExpr ctx scrutinee
  let (typeName, variants) ← sumDef ctx d
  let armFor (vi : Nat) (v : SumVariant) : Except Error MatchArm :=
    match arms.find? (·.variant == vi) with
    | some arm => .ok arm
    | none => .error ⟨s!"sumLower: match on '{typeName}' missing arm for '{v.name}'"⟩
  let lowerArmBody (arm : MatchArm) : Except Error Expr := do
    if arm.binders.isEmpty then
      rewriteExpr ctx arm.body
    else
      let subs ← bindingsForArm ctx scrutinee d arm
      rewriteExpr (ctx.withBindings subs) arm.body
  if variants.isEmpty then
    throw ⟨s!"sumLower: match on '{typeName}' has no variants (internal)"⟩
  let lastIdx := variants.size - 1
  let mut chain : Expr ← lowerArmBody (← armFor lastIdx variants[lastIdx]!)
  for k in [1:variants.size] do
    let i := lastIdx - k
    let arm ← armFor i variants[i]!
    chain := .select (.binary .eq tagRead (nat0 i)) (← lowerArmBody arm) chain
  return chain

/-- Per-arm payload bindings: only a `RegRef` scrutinee to a sum-typed
    reg is supported; binders rewrite to slot reads of that reg. -/
private partial def bindingsForArm (ctx : Ctx) (scrutinee : Expr)
    (matchDef : TypeDefIdx) (arm : MatchArm) :
    Except Error (List (BinderIdx × Expr)) := do
  if arm.binders.isEmpty then return []
  let (_, matchVariants) ← sumDef ctx matchDef
  let some armVariant := matchVariants[arm.variant]?
    | throw ⟨s!"sumLower: match arm variant index {arm.variant} out of range (internal)"⟩
  let .regRef rIdx := scrutinee
    | throw ⟨s!"sumLower: match arm '{armVariant.name}' has payload bindings but scrutinee is not a reg_ref"⟩
  let some srcDecl := ctx.oldRegs[rIdx.idx]?
    | throw ⟨s!"sumLower: match arm scrutinee regRef idx={rIdx.idx} has no source decl"⟩
  let some (.sum info) := ctx.spec? rIdx.idx
    | throw ⟨s!"sumLower: match arm '{armVariant.name}' scrutinee references non-sum reg '{srcDecl.name}'"⟩
  unless arm.binders.size == armVariant.payload.size do
    throw ⟨s!"sumLower: match arm '{armVariant.name}': binders/payload arity mismatch"⟩
  let mut subs : List (BinderIdx × Expr) := []
  for i in [0:arm.binders.size] do
    let field := armVariant.payload[i]!
    let some slot := info.payloadByKey armVariant.name field.name
      | throw ⟨s!"sumLower: match arm '{armVariant.name}': missing slot for field '{field.name}'"⟩
    subs := subs ++ [(arm.binders[i]!.idx, Expr.regRef slot.idx)]
  return subs

/-- Extract the scalar update for one slot of a sum-typed reg's update
    expression (tag / match / regRef / select shapes; anything else
    is 0 — caller's malformed update, TS-faithfully). -/
private partial def extractSlotFromSumExpr (ctx : Ctx) (info : SumRegInfo)
    (slot : SlotSpec) (e : Expr) : Except Error Expr := do
  match e with
  | .tag d variant payload =>
    -- TS: info.sumType.variants.indexOf(expr.variant) — same def ⇒
    -- the variant position; different def ⇒ not found.
    if d != info.def_ then
      let (_, variants) ← sumDef ctx d
      let vname := (variants[variant]?).map (·.name) |>.getD s!"#{variant}"
      throw ⟨s!"sumLower: tag variant '{vname}' not in '{info.sumName}'"⟩
    match slot.variant? with
    | none => return nat0 variant
    | some (vi, _) =>
      if vi == variant then
        let some (fi, _) := slot.field? | return zeroE
        match payload.find? (·.field == fi) with
        | some entry => rewriteExpr ctx entry.value
        | none => return zeroE
      else
        return zeroE
  | .match_ d scrutinee arms =>
    let tagRead ← rewriteExpr ctx scrutinee
    let (typeName, variants) ← sumDef ctx d
    let armFor (vi : Nat) (v : SumVariant) : Except Error MatchArm :=
      match arms.find? (·.variant == vi) with
      | some arm => .ok arm
      | none => .error ⟨s!"sumLower: match on '{typeName}' missing arm for '{v.name}'"⟩
    let armSlot (arm : MatchArm) : Except Error Expr := do
      if arm.binders.isEmpty then
        extractSlotFromSumExpr ctx info slot arm.body
      else
        let subs ← bindingsForArm ctx scrutinee d arm
        extractSlotFromSumExpr (ctx.withBindings subs) info slot arm.body
    if variants.isEmpty then
      throw ⟨s!"sumLower: match on '{typeName}' has no variants (internal)"⟩
    let lastIdx := variants.size - 1
    let mut chain : Expr ← armSlot (← armFor lastIdx variants[lastIdx]!)
    for k in [1:variants.size] do
      let i := lastIdx - k
      let arm ← armFor i variants[i]!
      chain := .select (.binary .eq tagRead (nat0 i)) (← armSlot arm) chain
    return chain
  | .regRef rIdx =>
    let some _ := ctx.oldRegs[rIdx.idx]? | return zeroE
    match ctx.spec? rIdx.idx with
    | some (.sum srcInfo) =>
      match slot.variant? with
      | none => return .regRef srcInfo.tagSlot.idx
      | some (_, vname) =>
        let fname := (slot.field?.map (·.2)).getD ""
        match srcInfo.payloadByKey vname fname with
        | some srcSlot => return .regRef srcSlot.idx
        | none => return zeroE
    | _ => return zeroE  -- reading a scalar reg as a sum value is malformed
  | .select cond then_ else_ =>
    return .select (← rewriteExpr ctx cond)
      (← extractSlotFromSumExpr ctx info slot then_)
      (← extractSlotFromSumExpr ctx info slot else_)
  | _ => return zeroE

end

-- ─────────────────────────────────────────────────────────────
-- Phase 2 — decl construction
-- ─────────────────────────────────────────────────────────────

private def buildSumRegSlots (ctx : Ctx) (origName : String) (init : Expr)
    (update? : Option Expr) (info : SumRegInfo) :
    Except Error (Array BodyDecl) := do
  let .tag initDef initVariant initPayload := init
    | throw ⟨s!"sumLower: reg '{origName}': init must be a constant tag expression"⟩
  -- TS: variants.indexOf(initTag.variant) — same def ⇒ position.
  if initDef != info.def_ then
    let (_, variants) ← sumDef ctx initDef
    let vname := (variants[initVariant]?).map (·.name) |>.getD s!"#{initVariant}"
    throw ⟨s!"sumLower: reg '{origName}': init variant '{vname}' not in '{info.sumName}'"⟩
  let mut out : Array BodyDecl := #[]
  for slot in info.slots do
    let slotInit : Expr ←
      match slot.variant? with
      | none => pure (nat0 initVariant)
      | some (vi, _) =>
        if vi == initVariant then
          match slot.field? with
          | some (fi, _) =>
            -- TS keys the init payload by FIELD NAME within the init
            -- variant; positions are bijective with names there.
            match initPayload.find? (·.field == fi) with
            | some entry => rewriteExpr ctx entry.value
            | none => pure zeroE
          | none => pure zeroE
        else
          pure zeroE
    let slotName :=
      match slot.variant?, slot.field? with
      | some (_, vname), some (_, fname) => mangle origName s!"{vname}__{fname}"
      | _, _ => mangle origName "tag"
    let update? ← match update? with
      | some u => some <$> extractSlotFromSumExpr ctx info slot u
      | none => pure none
    out := out.push (.reg slotName slotInit update? none none)
  return out

private def buildNonSumReg (ctx : Ctx) (name : String) (init : Expr)
    (update? : Option Expr) (type? : Option ScalarOrAlias)
    (liftedFrom? : Option String) : Except Error BodyDecl := do
  return .reg name (← rewriteExpr ctx init)
    (← match update? with
       | some u => some <$> rewriteExpr ctx u
       | none => pure none)
    type? liftedFrom?

private def buildNewDecls (ctx : Ctx) (decls : Array BodyDecl) :
    Except Error (Array BodyDecl) := do
  let mut out : Array BodyDecl := #[]
  let mut regPos := 0
  for decl in decls do
    match decl with
    | .reg name init update? type? liftedFrom? =>
      let some spec := ctx.spec? regPos
        | throw ⟨s!"sumLower: no spec for reg '{name}' (internal)"⟩
      regPos := regPos + 1
      match spec with
      | .sum info => out := out ++ (← buildSumRegSlots ctx name init update? info)
      | .nonSum _ => out := out.push (← buildNonSumReg ctx name init update? type? liftedFrom?)
    | .inst name typeKey tArgs inputs =>
      out := out.push (.inst name typeKey tArgs
        (← inputs.mapM fun i => do pure { i with value := ← rewriteExpr ctx i.value }))
    | .param .. | .prog .. =>
      out := out.push decl
  return out

-- ─────────────────────────────────────────────────────────────
-- Fast-path detection: skip if no work to do
-- ─────────────────────────────────────────────────────────────

private partial def exprHasSumExpr : Expr → Bool
  | .tag .. | .match_ .. => true
  | .num _ | .bool _
  | .inputRef _ | .regRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
  | .sampleRate | .sampleIndex | .nestedOut _ _ => false
  | .arr items => items.any exprHasSumExpr
  | .binary _ a b | .index a b => exprHasSumExpr a || exprHasSumExpr b
  | .unary _ a => exprHasSumExpr a
  | .clamp a b c | .select a b c | .arraySet a b c =>
    exprHasSumExpr a || exprHasSumExpr b || exprHasSumExpr c
  | .zeros count => exprHasSumExpr count
  | .fold over init _ _ body | .scan over init _ _ body =>
    exprHasSumExpr over || exprHasSumExpr init || exprHasSumExpr body
  | .iterate count init _ body | .chain count init _ body =>
    exprHasSumExpr count || exprHasSumExpr init || exprHasSumExpr body
  | .generate count _ body => exprHasSumExpr count || exprHasSumExpr body
  | .map2 over _ body => exprHasSumExpr over || exprHasSumExpr body
  | .zipWith a b _ _ body =>
    exprHasSumExpr a || exprHasSumExpr b || exprHasSumExpr body
  | .letIn binders body =>
    binders.any (fun b => exprHasSumExpr b.value) || exprHasSumExpr body

private def declHasSumExpr : BodyDecl → Bool
  | .reg _ init update? _ _ =>
    exprHasSumExpr init || update?.any exprHasSumExpr
  | .inst _ _ _ inputs => inputs.any (fun i => exprHasSumExpr i.value)
  | .param .. | .prog .. => false

private def progHasAnySumWork (arena : Arena) (prog : Program) : Bool :=
  prog.decls.any (fun d => match d with
    | .reg _ init _ _ _ => (sumTypeOfRegInit arena init).isSome
    | _ => false)
  || prog.decls.any declHasSumExpr
  || prog.assigns.any (fun a => exprHasSumExpr a.expr)

-- ─────────────────────────────────────────────────────────────
-- Public entry
-- ─────────────────────────────────────────────────────────────

def run (arena : Arena) (rootIdx : ProgramIdx) :
    Except Error (Arena × ProgramIdx) := do
  let some prog := arena.program? rootIdx
    | throw ⟨s!"sumLower: program pool index {rootIdx.idx} out of range"⟩
  if !progHasAnySumWork arena prog then
    return (arena, rootIdx)
  let specs := buildSpecs arena prog.decls
  let ctx : Ctx := { arena, oldRegs := prog.regs, specs }
  let newDecls ← buildNewDecls ctx prog.decls
  let newAssigns ← prog.assigns.mapM fun a => do
    pure { a with expr := ← rewriteExpr ctx a.expr }
  let fresh : Program := { prog with decls := newDecls, assigns := newAssigns }
  return ({ arena with programs := arena.programs.push fresh },
          ⟨arena.programs.size⟩)

end Tropical.Ir.Strata.SumLower
