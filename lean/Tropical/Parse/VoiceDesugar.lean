import Tropical.Parse.Nodes

/-!
# Voice-application desugar (voice-ports sprint, Phase 1)

Rewrite voice applications `v(clockExpr)` into instances of the bound voice
program clocked at `clockExpr`, plus a `nestedOut` reference to that instance's
output. Pure `ParsedProgram → ParsedProgram`, given the per-voice bindings (the
program wired into a voice port, its clock-input and output port names, and the
non-clock params threaded from the binding).

This is the whole novelty of voice ports at the IR level: `v(clk − δ)` is sugar
for "a fresh instance of the bound voice, `clk: clk − δ`, take its output" — the
FlangeSin shape Phase 0 proved renders. No new IR node downstream; the generated
instances flow through elaborate → strata → emit unchanged.

Scope note (Phase 1): voice applications are expected at expression position in
output assigns / instance inputs, not inside a combinator binder — lifting a
binder-scoped application to a top-level instance would escape its binder. The
MVP effects (flanger) apply voices at the top level; binder-scoped voices are a
later concern.
-/

namespace Tropical.Parse.VoiceDesugar

open Lean (JsonNumber)
open Tropical.Parse

/-- One voice port's binding: the program wired into it, that program's
    clock-input and output port names, and the non-clock params carried from the
    binding site. -/
structure VoiceBinding where
  voiceName : String
  programName : String
  clkInput : String
  output : String
  params : Array (String × ParsedExpr) := #[]
deriving Inhabited

private structure St where
  counter : Nat := 0
  newInsts : Array BodyDecl := #[]

private abbrev M := StateT St (Except String)

private def findVoice (bs : Array VoiceBinding) (name : String) : Option VoiceBinding :=
  bs.find? (·.voiceName == name)

partial def rw (bs : Array VoiceBinding) : ParsedExpr → M ParsedExpr
  | .call (.nameRef fname) args => do
    match findVoice bs fname with
    | some vb =>
      unless args.size == 1 do
        throw s!"voice application '{fname}(…)' takes exactly one clock argument, got {args.size}"
      let clkArg ← rw bs args[0]!
      let st ← get
      let instName := s!"__{fname}_{st.counter}"
      let inputs := #[(vb.clkInput, clkArg)] ++ vb.params
      set { st with
        counter := st.counter + 1
        newInsts := st.newInsts.push (.inst instName vb.programName none (some inputs)) }
      pure (.nestedOut instName vb.output)
    | none =>
      pure (.call (.nameRef fname) (← args.mapM (rw bs)))
  | .call callee args => do pure (.call (← rw bs callee) (← args.mapM (rw bs)))
  | .num n => do pure (.num n)
  | .bool b => do pure (.bool b)
  | .arr items => do pure (.arr (← items.mapM (rw bs)))
  | .binary tag l r => do pure (.binary tag (← rw bs l) (← rw bs r))
  | .unary tag a => do pure (.unary tag (← rw bs a))
  | .nameRef n => do pure (.nameRef n)
  | .binding n => do pure (.binding n)
  | .nestedOut r o => do pure (.nestedOut r o)
  | .index a i => do pure (.index (← rw bs a) (← rw bs i))
  | .letIn binds body => do
    let binds' ← binds.mapM fun (k, e) => do pure (k, ← rw bs e)
    pure (.letIn binds' (← rw bs body))
  | .fold over init av ev body => do
    pure (.fold (← rw bs over) (← rw bs init) av ev (← rw bs body))
  | .scan over init av ev body => do
    pure (.scan (← rw bs over) (← rw bs init) av ev (← rw bs body))
  | .generate count v body => do pure (.generate (← rw bs count) v (← rw bs body))
  | .iterate count v init body => do
    pure (.iterate (← rw bs count) v (← rw bs init) (← rw bs body))
  | .chain count v init body => do
    pure (.chain (← rw bs count) v (← rw bs init) (← rw bs body))
  | .map2 over ev body => do pure (.map2 (← rw bs over) ev (← rw bs body))
  | .zipWith a b xv yv body => do
    pure (.zipWith (← rw bs a) (← rw bs b) xv yv (← rw bs body))
  | .tag variant payload => do
    match payload with
    | none => do pure (.tag variant none)
    | some entries =>
      pure (.tag variant (some (← entries.mapM fun e => do
        pure (TagPayloadEntry.mk e.field (← rw bs e.value)))))
  | .match_ scrutinee arms => do
    let arms' ← arms.mapM fun arm => do
      pure (MatchArm.mk arm.variant arm.binds (← rw bs arm.body))
    pure (.match_ (← rw bs scrutinee) arms')

/-- Desugar every voice application in a program body, appending the generated
    voice instances to the body's decls. -/
def desugarBody (bs : Array VoiceBinding) (body : Block) : Except String Block := do
  let walkAssign : BodyAssign → M BodyAssign := fun
    | .output name e => do pure (.output name (← rw bs e))
  let walkDecl : BodyDecl → M BodyDecl := fun
    | .inst name prog ta (some inputs) => do
      pure (.inst name prog ta (some (← inputs.mapM fun (k, e) => do pure (k, ← rw bs e))))
    | d => pure d
  let action : M Block := do
    let assigns' ← body.assigns.mapM walkAssign
    let decls' ← body.decls.mapM walkDecl
    pure (Block.mk (decls' ++ (← get).newInsts) assigns')
  match action.run {} with
  | .ok (block, _) => pure block
  | .error e => throw e

end Tropical.Parse.VoiceDesugar


