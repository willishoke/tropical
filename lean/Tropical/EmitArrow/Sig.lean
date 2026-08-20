import Tropical.Ir.Nodes

/-!
# EmitArrow.Sig — ID-native authoring foundation

Signals are stable `ExprId`s in the `ExprArena` owned by `Builder`; smart
constructors intern `ENode`s immediately, after their child IDs already exist.

These declarations are the stable `Tropical.EmitArrow` authoring API.
-/

namespace Tropical.EmitArrow

open Lean (JsonNumber)
open Tropical.Ir

/-- An authored signal is a stable handle into the active builder arena. -/
abbrev Sig := ExprId

/-- The Q32.32 clock rail uses the same expression vocabulary and handles. -/
abbrev Clock := Sig

/-- An instance input already carrying an arena-native expression ID. -/
structure AInput where
  port : InputIdx
  value : Sig
deriving Inhabited, Repr

/-- An authoring-side instance declaration.  Declaration order determines
    `InstanceIdx`, just as it does on the recursive path. -/
structure AInst where
  name : String
  programName : String
  inputs : Array AInput := #[]
deriving Inhabited, Repr

/-- An input declaration whose optional default is already an expression ID. -/
structure AInputDecl where
  name : String
  type? : Option PortType := none
  defaultSig : Option Sig := none
deriving Inhabited, Repr

/-- The complete state needed while authoring one program.  Expression IDs are
    scoped by convention to this append-only arena; declarations share that
    same scope. -/
structure Builder where
  exprs : ExprArena := {}
  decls : Array AInst := #[]
deriving Inhabited

/-- Expression allocation and declaration construction are atomic with
    respect to the outer `Except`: a failed build never publishes a `Program`.
-/
abbrev BuildM := StateT Builder (Except String)

/-- The ID-valued portion returned by a program build.  Keeping this inside the
    build action makes input defaults and output assignments use IDs from the
    same active builder arena. -/
structure ProgramBody where
  inputs : Array AInputDecl := #[]
  assigns : Array (OutputTarget × Sig) := #[]
deriving Inhabited, Repr

/-- A program whose output surface is discovered during the arena build.  The
    playground uses this for optional taps: a refused projection must publish
    neither an assignment nor a dangling output declaration. -/
structure CompleteProgramBody where
  inputs : Array AInputDecl := #[]
  outputs : Array OutputDecl := #[]
  assigns : Array (OutputTarget × Sig) := #[]
deriving Inhabited, Repr

/-- The sole expression-arena mutation in the authoring API.  Raw `ENode`
    construction stays local to this module; ordinary callers use the smart
    constructors below. -/
private def internSig (node : ENode) : BuildM Sig := do
  let builder ← get
  let (id, exprs) := (eintern node).run builder.exprs
  set { builder with exprs }
  pure id

-- ─────────────────────────────────────────────────────────────
-- Direct expression smart constructors
-- ─────────────────────────────────────────────────────────────

def num (n : JsonNumber) : BuildM Sig := internSig (.num n)

/-- Decimal literal `mantissa · 10^(-exponent)`. -/
def lit (mantissa : Int) (exponent : Nat := 0) : BuildM Sig :=
  num ⟨mantissa, exponent⟩

def binary (tag : BinaryOpTag) (lhs rhs : Sig) : BuildM Sig :=
  internSig (.binary tag lhs rhs)

def unary (tag : UnaryOpTag) (arg : Sig) : BuildM Sig :=
  internSig (.unary tag arg)

def clamp (value lo hi : Sig) : BuildM Sig :=
  internSig (.clamp value lo hi)

def select (cond then_ else_ : Sig) : BuildM Sig :=
  internSig (.select cond then_ else_)

def inputRef (idx : InputIdx) : BuildM Sig := internSig (.inputRef idx)

def paramRef (idx : ParamIdx) : BuildM Sig := internSig (.paramRef idx)

def nestedOut (instance_ : InstanceIdx) (output : OutputIdx) : BuildM Sig :=
  internSig (.nestedOut instance_ output)

def sampleRate : BuildM Sig := internSig .sampleRate

def sampleIndex : BuildM Sig := internSig .sampleIndex

/-- Dispatch-local `[0,1)` coordinate used only by the absolute-time tile
    terminal.  Exact/JIT evaluation deliberately reads zero. -/
def tilePhase : BuildM Sig := internSig .tilePhase

/-- Absolute clock leaf used only by the tile endpoint materializer. -/
def tileSampleIndex : BuildM Sig := internSig .tileSampleIndex

def arr (items : Array Sig) : BuildM Sig := internSig (.arr items)

/-- Mark an array as an immutable endpoint image owned by the tile-time
    materializer. -/
def tileArray (items : Array Sig) : BuildM Sig := internSig (.tileArray items)

def index (array index : Sig) : BuildM Sig :=
  internSig (.index array index)

def loopIdx (id : Nat) : BuildM Sig := internSig (.loopIdx id)

/-- An indexed reduction.  Table order, optional dynamic count, and binder ID
    are stored verbatim; callers construct the body with the matching
    `loopIdx` ID before interning this parent. -/
def bankSum (count : Nat) (tables : Array Sig) (body : Sig)
    (dynCount? : Option Sig := none) (idxId : Nat := 0) : BuildM Sig :=
  internSig (.bankSum count tables body dynCount? idxId)

/-- Static-capacity pure map followed by an authored-order routed additive
    fold.  Routes, tables, values, and binder ID are stored verbatim. -/
def routedSum (capacity outputCount : Nat) (routes : Array (Option Nat))
    (tables values : Array Sig) (dynCount? : Option Sig := none)
    (idxId : Nat := 0) : BuildM Sig :=
  internSig (.routedSum capacity outputCount routes tables values dynCount? idxId)

-- ────────────────────────────────────────────────────────────────────────
-- Absolute-coordinate substitution
-- ────────────────────────────────────────────────────────────────────────

/-- Rebuild `roots` in the current arena while substituting every
    `sampleIndex` leaf with `tileSampleIndex + frames`.  The distinct base
    leaf prevents the tile dependency slice from capturing shared audio-clock
    instructions.  The walk freezes the
    pre-rewrite arena and proceeds in child-before-parent ID order, so address
    warps and control trajectories are shifted as complete expressions rather
    than approximated by advancing only a visible phasor. -/
def shiftSampleIndex (roots : Array Sig) (frames : Nat) : BuildM (Array Sig) := do
  let snapshot := (← get).exprs.nodes
  let rawTick ← internSig .tileSampleIndex
  let shiftedTick ← if frames == 0 then
      pure rawTick
    else do
      let frameLiteral ← internSig (.num ⟨Int.ofNat frames, 0⟩)
      let frameInt ← internSig (.unary .toInt frameLiteral)
      internSig (.binary .add rawTick frameInt)
  let mut mapped : Array Sig := #[]
  let mapId := fun (mapping : Array Sig) (id : Sig) =>
    mapping[id.idx]?.getD id
  for node in snapshot do
    let id ← match node with
      | .sampleIndex => pure shiftedTick
      | .tileSampleIndex => internSig .tileSampleIndex
      | .num n => internSig (.num n)
      | .bool b => internSig (.bool b)
      | .arr items => internSig (.arr (items.map (mapId mapped)))
      | .tileArray items => internSig (.tileArray (items.map (mapId mapped)))
      | .binary tag a b =>
        internSig (.binary tag (mapId mapped a) (mapId mapped b))
      | .unary tag a => internSig (.unary tag (mapId mapped a))
      | .clamp a b c => internSig (.clamp
          (mapId mapped a) (mapId mapped b) (mapId mapped c))
      | .select a b c => internSig (.select
          (mapId mapped a) (mapId mapped b) (mapId mapped c))
      | .arraySet a b c =>
        internSig (.arraySet
          (mapId mapped a) (mapId mapped b) (mapId mapped c))
      | .index a b =>
        internSig (.index (mapId mapped a) (mapId mapped b))
      | .inputRef i => internSig (.inputRef i)
      | .paramRef i => internSig (.paramRef i)
      | .nestedOut i o => internSig (.nestedOut i o)
      | .sampleRate => internSig .sampleRate
      | .tilePhase => internSig .tilePhase
      | .loopIdx i => internSig (.loopIdx i)
      | .bankSum count tables body dynCount? idxId =>
        internSig (.bankSum count (tables.map (mapId mapped))
          (mapId mapped body) (dynCount?.map (mapId mapped)) idxId)
      | .routedSum capacity outputCount routes tables values dynCount? idxId =>
        internSig (.routedSum capacity outputCount routes
          (tables.map (mapId mapped)) (values.map (mapId mapped))
          (dynCount?.map (mapId mapped)) idxId)
    mapped := mapped.push id
  pure (roots.map (mapId mapped))

-- The production scalar helper vocabulary, preserving its existing names and
-- operand order.  Each helper performs exactly one intern step.

def litI (mantissa : Int) : BuildM Sig := do
  let value ← lit mantissa
  unary .toInt value

def mul (a b : Sig) : BuildM Sig := binary .mul a b
def add (a b : Sig) : BuildM Sig := binary .add a b
def sub (a b : Sig) : BuildM Sig := binary .sub a b
def div (a b : Sig) : BuildM Sig := binary .div a b
def bitAnd (a b : Sig) : BuildM Sig := binary .bitAnd a b
def bitOr (a b : Sig) : BuildM Sig := binary .bitOr a b
def rshift (a b : Sig) : BuildM Sig := binary .rshift a b
def lshift (a b : Sig) : BuildM Sig := binary .lshift a b
def gt (a b : Sig) : BuildM Sig := binary .gt a b
def ldexpE (mantissa exponent : Sig) : BuildM Sig :=
  binary .ldexp mantissa exponent

def toIntE (a : Sig) : BuildM Sig := unary .toInt a
def neg (a : Sig) : BuildM Sig := unary .neg a
def absE (a : Sig) : BuildM Sig := unary .abs a
def floatExponentE (a : Sig) : BuildM Sig := unary .floatExponent a
def roundE (a : Sig) : BuildM Sig := unary .round a
def toFloatE (a : Sig) : BuildM Sig := unary .toFloat a
def clampE (value lo hi : Sig) : BuildM Sig := clamp value lo hi
def selectE (cond then_ else_ : Sig) : BuildM Sig := select cond then_ else_

/-- Authored-order, left-associated addition over an array of already-built
    IDs.  The empty sum constructs the same numeric zero as the recursive
    authoring surface. -/
def sumLeft (items : Array Sig) : BuildM Sig := do
  match items[0]? with
  | none => lit 0
  | some first => (items.extract 1 items.size).foldlM add first

/-- Encode a build-time `Float` as the same decimal literal used by the
    recursive authoring path. -/
def litF (x : Float) : BuildM Sig :=
  let scaled := x * 1000000000000.0
  let mantissa : Int :=
    if scaled ≥ 0.0 then Int.ofNat (scaled + 0.5).toUInt64.toNat
    else -(Int.ofNat (0.5 - scaled).toUInt64.toNat)
  if mantissa == 0 then lit 0 else lit mantissa 12

-- ─────────────────────────────────────────────────────────────
-- Ordered declaration construction and ID-native assembly
-- ─────────────────────────────────────────────────────────────

/-- Append one instance in authored order and return its stable positional
    index.  Inputs already contain IDs from the active builder arena. -/
def declareInst (decl : AInst) : BuildM InstanceIdx := do
  let builder ← get
  let idx : InstanceIdx := ⟨builder.decls.size⟩
  set { builder with decls := builder.decls.push decl }
  pure idx

/-- Convenience form of `declareInst` retaining input-port order verbatim. -/
def inst (name programName : String) (inputs : Array AInput := #[]) :
    BuildM InstanceIdx :=
  declareInst { name, programName, inputs }

/-- Mirror the elaborator's deterministic transitive registry merge. -/
def buildRegistry (arena : Arena) (resolved : Array (String × ProgramIdx))
    (programNames : Array String) : Except String (Array (String × ProgramIdx)) := do
  let mut registry : Array (String × ProgramIdx) := #[]
  for programName in programNames do
    let some index := (resolved.find? (·.1 == programName)).map (·.2)
      | .error s!"EmitArrow: program '{programName}' not found in the elaborated stdlib chain"
    let some program := arena.program? index
      | .error s!"EmitArrow: program '{programName}' index out of range"
    if !registry.any (·.1 == program.name) then
      registry := registry.push (program.name, index)
    for (name, registeredIndex) in program.registry do
      if !registry.any (·.1 == name) then
        registry := registry.push (name, registeredIndex)
  pure registry

/-- Assemble one ID-native program.  The build starts from `arena.exprs`; only
    an `.ok` result publishes the updated expression arena and appends exactly
    one program.  No recursive expression lowering occurs at this seam. -/
def assemble (arena : Arena) (name : String) (outputs : Array OutputDecl)
    (registry : Array (String × ProgramIdx)) (build : BuildM ProgramBody)
    (extraDecls : Array BodyDecl := #[]) :
    Except String (Arena × ProgramIdx) := do
  let initial : Builder := { exprs := arena.exprs }
  let (body, builder) ← build.run initial
  let inputs : Array InputDecl := body.inputs.map fun decl =>
    { name := decl.name, type? := decl.type?, default? := decl.defaultSig }
  let decls : Array BodyDecl := builder.decls.map fun decl =>
    .inst decl.name decl.programName (decl.inputs.map fun input =>
      { port := input.port, value := input.value })
  let assigns : Array OutputAssign := body.assigns.map fun (target, expr) =>
    { target, expr }
  let program : Program :=
    { name, inputs, outputs, decls := decls ++ extraDecls, assigns, registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with
    programs := arena.programs.push program
    exprs := builder.exprs }, idx)

/-- Assemble a program whose output declarations, plus arbitrary pure caller
    metadata, are determined inside the same atomic builder transaction. -/
def assembleCompleteWithResult {α : Type} (arena : Arena) (name : String)
    (registry : Array (String × ProgramIdx))
    (build : BuildM (CompleteProgramBody × α))
    (extraDecls : Array BodyDecl := #[]) :
    Except String (Arena × ProgramIdx × α) := do
  let initial : Builder := { exprs := arena.exprs }
  let ((body, result), builder) ← build.run initial
  let inputs : Array InputDecl := body.inputs.map fun decl =>
    { name := decl.name, type? := decl.type?, default? := decl.defaultSig }
  let decls : Array BodyDecl := builder.decls.map fun decl =>
    .inst decl.name decl.programName (decl.inputs.map fun input =>
      { port := input.port, value := input.value })
  let assigns : Array OutputAssign := body.assigns.map fun (target, expr) =>
    { target, expr }
  let program : Program := {
    name, inputs, outputs := body.outputs
    decls := decls ++ extraDecls, assigns, registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  pure ({ arena with
    programs := arena.programs.push program
    exprs := builder.exprs }, idx, result)

end Tropical.EmitArrow
