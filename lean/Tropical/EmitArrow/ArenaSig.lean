import Tropical.Ir.Nodes

/-!
# EmitArrow.ArenaSig — arena-native authoring foundation

This is the phase-1 authoring substrate for the recursive-`Sig` cutover.
Signals are stable `ExprId`s in the `ExprArena` owned by `Builder`; smart
constructors intern `ENode`s immediately, after their child IDs already exist.

The `ArenaNative` namespace is deliberately temporary while the recursive
`Tropical.EmitArrow.Sig` API is still present.  Later migration phases move
callers here; phase 5 removes the old API and promotes these names.
-/

namespace Tropical.EmitArrow.ArenaNative

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

def arr (items : Array Sig) : BuildM Sig := internSig (.arr items)

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
def roundE (a : Sig) : BuildM Sig := unary .round a
def toFloatE (a : Sig) : BuildM Sig := unary .toFloat a
def clampE (value lo hi : Sig) : BuildM Sig := clamp value lo hi
def selectE (cond then_ else_ : Sig) : BuildM Sig := select cond then_ else_

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

end Tropical.EmitArrow.ArenaNative
