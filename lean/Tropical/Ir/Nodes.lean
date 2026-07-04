import Lean.Data.Json
import Tropical.Parse.Nodes

/-!
# Resolved IR — pool-shaped port of compiler/ir/nodes.ts

The TS resolved IR uses object identity and back-pointers in four
reference families (programs shared across registries, type defs
shared across scope chains, type params shared across shape dims,
`SumVariant.parent` cycles). Lean represents those natively as a
top-level `Arena` of three identity pools — the same three pools the
TS codec (`compiler/ir/resolved_codec.ts`, schema `tropical_resolved_1`)
already serializes — and every cross-pool reference is a typed index
newtype.

Within a program, refs carry the positional de Bruijn levels of the TS
IR: `RegIdx` / `InputIdx` / `OutputIdx` / `ParamIdx` / `InstanceIdx` /
`TypeParamIdx` index the enclosing program's typed decl tables (which
this port *computes* from `decls` — see `Program.regs` etc. — so the
body↔table invariant holds by construction); `BinderIdx` is the
unique-per-program ID minted by the elaborator.

`Tag` / `Match` reference their sum type as (typeDef pool idx, variant
position) and payload fields by position within the variant — exactly
the wire encoding, so the codec is nearly an identity map.

Numbers are `Lean.JsonNumber` (decimal text preserved) so encode output
re-parses to the bit-identical double on the TS side, mirroring
`Tropical/Parse/Nodes.lean`.
-/

namespace Tropical.Ir

open Lean (JsonNumber)

/-- Scalar kinds are shared verbatim with the parse layer. -/
abbrev ScalarKind := Tropical.Parse.ScalarKind

-- ─────────────────────────────────────────────────────────────
-- Typed index newtypes
-- ─────────────────────────────────────────────────────────────

/-- Position in `ports.inputs` (of the enclosing program, or — on
    `InstanceDecl.inputs` — of the *target* program). -/
structure InputIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Position in `ports.outputs` (enclosing program; on `NestedOut`, the
    target program's). -/
structure OutputIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Position in the enclosing program's param table. -/
structure ParamIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Position in the enclosing program's instance table. -/
structure InstanceIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Position in `typeParams` (enclosing program; on
    `InstanceDecl.typeArgs`, the target's). -/
structure TypeParamIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Unique-per-program binder ID (NOT a table position). -/
structure BinderIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

-- Pool indices (arena-level identity).

/-- Index into `Arena.programs`. -/
structure ProgramIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Index into `Arena.typeDefs`. -/
structure TypeDefIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Index into `Arena.typeParams`. -/
structure TypeParamPoolIdx where
  idx : Nat
deriving BEq, Repr, Inhabited

/-- Dense index into a hash-consed expression arena (`CoreArena` /
    `ExprArena`). Defined here (not in `CoreArena`) so `Core`'s
    post-strata leaves can be `ExprId`s without importing the arena
    modules that in turn import `Core` (the circular dependency the
    Phase B DAG-to-emit reshape would otherwise hit). -/
structure ExprId where
  idx : Nat
deriving BEq, Hashable, Repr, Inhabited

-- ─────────────────────────────────────────────────────────────
-- Type defs (pool entries)
-- ─────────────────────────────────────────────────────────────

/-- `ScalarKind | AliasTypeDef` positions in the TS IR. The alias arm
    must reference an `alias` pool entry. -/
inductive ScalarOrAlias where
  | scalar (k : ScalarKind)
  | alias (td : TypeDefIdx)
deriving BEq, Repr, Inhabited

structure StructField where
  name : String
  type : ScalarOrAlias
deriving BEq, Repr, Inhabited

structure SumVariant where
  name : String
  payload : Array StructField
deriving BEq, Repr, Inhabited

/-- A typeDef pool entry. The TS `SumVariant.parent` back-pointer is
    implicit: a variant is identified by (pool idx of its sum, position). -/
inductive TypeDef where
  | alias (name : String) (base : ScalarKind)
  | struct (name : String) (fields : Array StructField)
  | sum (name : String) (variants : Array SumVariant)
deriving BEq, Repr, Inhabited

def TypeDef.name : TypeDef → String
  | .alias n _ => n
  | .struct n _ => n
  | .sum n _ => n

/-- The TS `op` discriminator string, used in error messages
    (`resolveElement`'s "got ${td.op}"). -/
def TypeDef.opName : TypeDef → String
  | .alias _ _ => "aliasTypeDef"
  | .struct _ _ => "structTypeDef"
  | .sum _ _ => "sumTypeDef"

/-- A typeParam pool entry. -/
structure TypeParamDecl where
  name : String
  default? : Option JsonNumber := none
deriving Repr, Inhabited

-- ─────────────────────────────────────────────────────────────
-- Port types
-- ─────────────────────────────────────────────────────────────

inductive ShapeDim where
  | lit (n : JsonNumber)
  | typeParam (tp : TypeParamPoolIdx)
deriving Repr, Inhabited

inductive PortType where
  | scalar (k : ScalarKind)
  | alias (td : TypeDefIdx)
  | array (element : ScalarOrAlias) (shape : Array ShapeDim)
deriving Repr, Inhabited

-- ─────────────────────────────────────────────────────────────
-- Op tags (resolved phase: parse tags plus call-surfaced builtins)
-- ─────────────────────────────────────────────────────────────

inductive BinaryOpTag where
  | add | sub | mul | div | mod
  | lt | lte | gt | gte | eq | neq
  | and | or
  | bitAnd | bitOr | bitXor | lshift | rshift
  | floorDiv | ldexp
deriving BEq, Repr, Inhabited

def BinaryOpTag.wire : BinaryOpTag → String
  | .add => "add" | .sub => "sub" | .mul => "mul" | .div => "div" | .mod => "mod"
  | .lt => "lt" | .lte => "lte" | .gt => "gt" | .gte => "gte"
  | .eq => "eq" | .neq => "neq"
  | .and => "and" | .or => "or"
  | .bitAnd => "bitAnd" | .bitOr => "bitOr" | .bitXor => "bitXor"
  | .lshift => "lshift" | .rshift => "rshift"
  | .floorDiv => "floorDiv" | .ldexp => "ldexp"

def BinaryOpTag.ofWire? : String → Option BinaryOpTag
  | "add" => some .add | "sub" => some .sub | "mul" => some .mul
  | "div" => some .div | "mod" => some .mod
  | "lt" => some .lt | "lte" => some .lte | "gt" => some .gt | "gte" => some .gte
  | "eq" => some .eq | "neq" => some .neq
  | "and" => some .and | "or" => some .or
  | "bitAnd" => some .bitAnd | "bitOr" => some .bitOr | "bitXor" => some .bitXor
  | "lshift" => some .lshift | "rshift" => some .rshift
  | "floorDiv" => some .floorDiv | "ldexp" => some .ldexp
  | _ => none

/-- Lift a parse-phase binary tag (the infix subset) into the resolved tag. -/
def BinaryOpTag.ofParse : Tropical.Parse.BinaryOpTag → BinaryOpTag
  | .add => .add | .sub => .sub | .mul => .mul | .div => .div | .mod => .mod
  | .lt => .lt | .lte => .lte | .gt => .gt | .gte => .gte
  | .eq => .eq | .neq => .neq
  | .and => .and | .or => .or
  | .bitAnd => .bitAnd | .bitOr => .bitOr | .bitXor => .bitXor
  | .lshift => .lshift | .rshift => .rshift

inductive UnaryOpTag where
  | neg | not | bitNot
  | sqrt | abs | floor | ceil | round
  | floatExponent | toInt | toBool | toFloat
deriving BEq, Repr, Inhabited

def UnaryOpTag.wire : UnaryOpTag → String
  | .neg => "neg" | .not => "not" | .bitNot => "bitNot"
  | .sqrt => "sqrt" | .abs => "abs" | .floor => "floor"
  | .ceil => "ceil" | .round => "round"
  | .floatExponent => "floatExponent" | .toInt => "toInt"
  | .toBool => "toBool" | .toFloat => "toFloat"

def UnaryOpTag.ofWire? : String → Option UnaryOpTag
  | "neg" => some .neg | "not" => some .not | "bitNot" => some .bitNot
  | "sqrt" => some .sqrt | "abs" => some .abs | "floor" => some .floor
  | "ceil" => some .ceil | "round" => some .round
  | "floatExponent" => some .floatExponent | "toInt" => some .toInt
  | "toBool" => some .toBool | "toFloat" => some .toFloat
  | _ => none

def UnaryOpTag.ofParse : Tropical.Parse.UnaryOpTag → UnaryOpTag
  | .neg => .neg | .not => .not | .bitNot => .bitNot

-- ─────────────────────────────────────────────────────────────
-- Expressions
-- ─────────────────────────────────────────────────────────────

/-- An anonymous binder decl: identity label + unique-per-program idx. -/
structure Binder where
  name : String
  idx : BinderIdx
deriving Repr, Inhabited, BEq

mutual

/-- Port of `ResolvedExpr` / `ResolvedExprOp`. -/
inductive Expr where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array Expr)
  | binary (tag : BinaryOpTag) (lhs rhs : Expr)
  | unary (tag : UnaryOpTag) (arg : Expr)
  | clamp (value lo hi : Expr)
  | select (cond then_ else_ : Expr)
  | arraySet (arr idx value : Expr)
  | index (arr idx : Expr)
  | zeros (count : Expr)
  | inputRef (idx : InputIdx)
  | paramRef (idx : ParamIdx)
  | typeParamRef (idx : TypeParamIdx)
  | bindingRef (idx : BinderIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
  | fold (over init : Expr) (acc elem : Binder) (body : Expr)
  | scan (over init : Expr) (acc elem : Binder) (body : Expr)
  | generate (count : Expr) (iter : Binder) (body : Expr)
  | iterate (count init : Expr) (iter : Binder) (body : Expr)
  | chain (count init : Expr) (iter : Binder) (body : Expr)
  | map2 (over : Expr) (elem : Binder) (body : Expr)
  | zipWith (a b : Expr) (x y : Binder) (body : Expr)
  | letIn (binders : Array LetBinder) (body : Expr)
  /-- Variant constructor: sum pool idx + variant position; payload
      entries pair a field position (into the variant's payload) with
      the value bound to it. -/
  | tag (def_ : TypeDefIdx) (variant : Nat) (payload : Array TagPayload)
  | match_ (def_ : TypeDefIdx) (scrutinee : Expr) (arms : Array MatchArm)
deriving Repr, Inhabited

inductive LetBinder where
  | mk (binder : Binder) (value : Expr)
deriving Repr, Inhabited

inductive TagPayload where
  | mk (field : Nat) (value : Expr)
deriving Repr, Inhabited

inductive MatchArm where
  | mk (variant : Nat) (binders : Array Binder) (body : Expr)
deriving Repr, Inhabited

end

def LetBinder.binder : LetBinder → Binder
  | .mk b _ => b

def LetBinder.value : LetBinder → Expr
  | .mk _ v => v

def TagPayload.field : TagPayload → Nat
  | .mk f _ => f

def TagPayload.value : TagPayload → Expr
  | .mk _ v => v

def MatchArm.variant : MatchArm → Nat
  | .mk v _ _ => v

def MatchArm.binders : MatchArm → Array Binder
  | .mk _ b _ => b

def MatchArm.body : MatchArm → Expr
  | .mk _ _ b => b

-- ─────────────────────────────────────────────────────────────
-- Decls + assigns + program
-- ─────────────────────────────────────────────────────────────

structure InputDecl where
  name : String
  type? : Option PortType := none
  default? : Option Expr := none
deriving Repr, Inhabited

structure OutputDecl where
  name : String
  type? : Option PortType := none
deriving Repr, Inhabited

structure InstanceTypeArg where
  param : TypeParamIdx
  value : JsonNumber
deriving Repr, Inhabited

structure InstanceInput where
  port : InputIdx
  value : Expr
deriving Repr, Inhabited

inductive BodyDecl where
  | param (name : String) (value? : Option JsonNumber)
  | inst (name : String) (typeKey : String)
      (typeArgs : Array InstanceTypeArg) (inputs : Array InstanceInput)
  | prog (name : String) (program : ProgramIdx)
deriving Repr, Inhabited

def BodyDecl.name : BodyDecl → String
  | .param n _ => n | .inst n .. => n | .prog n _ => n

inductive OutputTarget where
  | port (idx : OutputIdx)
  | dac
deriving BEq, Repr, Inhabited

structure OutputAssign where
  target : OutputTarget
  expr : Expr
deriving Repr, Inhabited

/-- One program pool entry. `registry` is the `programRegistry` as an
    ordered association array — TS `Map` insertion order is observable
    through the codec, so it is load-bearing here. `typeParams` entries
    are pool indices (a nested program's shape dim may reference an
    *enclosing* program's TypeParamDecl — exactly why the pool exists). -/
structure Program where
  name : String
  typeParams : Array TypeParamPoolIdx := #[]
  inputs : Array InputDecl := #[]
  outputs : Array OutputDecl := #[]
  typeDefs : Array TypeDefIdx := #[]
  decls : Array BodyDecl := #[]
  assigns : Array OutputAssign := #[]
  binderCount : Nat := 0
  registry : Array (String × ProgramIdx) := #[]
deriving Repr, Inhabited

/-- Projected param table (positions are `ParamIdx`). -/
def Program.params (p : Program) : Array BodyDecl :=
  p.decls.filter fun d => match d with | .param .. => true | _ => false

/-- Projected instance table (positions are `InstanceIdx`). -/
def Program.instances (p : Program) : Array BodyDecl :=
  p.decls.filter fun d => match d with | .inst .. => true | _ => false

def Program.registryGet? (p : Program) (key : String) : Option ProgramIdx :=
  (p.registry.find? (·.1 == key)).map (·.2)

-- ─────────────────────────────────────────────────────────────
-- Arena — the three identity pools
-- ─────────────────────────────────────────────────────────────

/-- The identity pools of the resolved IR. References into a pool are
    by index; sharing an index is the Lean image of TS pointer sharing.
    Invariant (maintained by both the codec's forward-pass decode and
    the elaborator's children-before-parents construction): a program
    at index `i` references only programs at indices `< i`, so the
    arena is acyclic by construction. -/
structure Arena where
  typeParams : Array TypeParamDecl := #[]
  typeDefs : Array TypeDef := #[]
  programs : Array Program := #[]
deriving Repr, Inhabited

def Arena.program? (a : Arena) (i : ProgramIdx) : Option Program :=
  a.programs[i.idx]?

def Arena.typeDef? (a : Arena) (i : TypeDefIdx) : Option TypeDef :=
  a.typeDefs[i.idx]?

def Arena.typeParam? (a : Arena) (i : TypeParamPoolIdx) : Option TypeParamDecl :=
  a.typeParams[i.idx]?

end Tropical.Ir
