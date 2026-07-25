import Std.Data.HashMap
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

-- The tree `Expr` (and its `LetBinder`/`TagPayload`/`MatchArm` binders) is gone:
-- the resolved expression IS the hash-consed `ENode`/`ExprArena` DAG below, and
-- `Program`'s leaves are `ExprId`s. Authoring frontends (the surface parser's
-- `ParsedExpr`, `EmitArrow`'s combinator tree) build their own shapes and lower
-- into the arena.

-- ─────────────────────────────────────────────────────────────
-- ExprArena — the hash-consed (DAG) form of the resolved expression
--
-- The native-DAG representation for the whole lowering (issue #190). An
-- `ENode` is flat — its children are `ExprId`s — so it is O(1) to hash and
-- compare, and interning at construction makes two equal subtrees one node.
-- This is THE resolved-expression representation: `Program`'s expression leaves
-- are `ExprId`s into an `ExprArena`; there is no tree `Expr` twin.
-- ─────────────────────────────────────────────────────────────

/-- A let binder with an id-valued body. -/
structure ELetBinder where
  binder : Binder
  value : ExprId
deriving BEq, Repr, Inhabited

/-- A tag payload with an id-valued field. -/
structure ETagPayload where
  field : Nat
  value : ExprId
deriving BEq, Repr, Inhabited

/-- A match arm with an id-valued body. -/
structure EMatchArm where
  variant : Nat
  binders : Array Binder
  body : ExprId
deriving BEq, Repr, Inhabited

/-- A resolved expression node with children referenced by `ExprId`; flat (no
    inlined subtrees). The full resolved op set — combinators (`fold`,
    `generate`, …), `letIn`, `tag`/`match` and their binders — so the arena is
    live from the elaborator through the lowering to emit.

    Binders carry their `BinderIdx`, so two otherwise-identical combinators with
    *different* binder indices stay distinct (alpha-correct); within one program,
    identical subtrees carry identical binder indices and so share. -/
inductive ENode where
  | num (n : JsonNumber)
  | bool (b : Bool)
  | arr (items : Array ExprId)
  | binary (tag : BinaryOpTag) (lhs rhs : ExprId)
  | unary (tag : UnaryOpTag) (arg : ExprId)
  | clamp (value lo hi : ExprId)
  | select (cond then_ else_ : ExprId)
  | arraySet (arr idx value : ExprId)
  | index (arr idx : ExprId)
  | zeros (count : ExprId)
  | inputRef (idx : InputIdx)
  | paramRef (idx : ParamIdx)
  | typeParamRef (idx : TypeParamIdx)
  | bindingRef (idx : BinderIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
  | fold (over init : ExprId) (acc elem : Binder) (body : ExprId)
  | scan (over init : ExprId) (acc elem : Binder) (body : ExprId)
  | generate (count : ExprId) (iter : Binder) (body : ExprId)
  | iterate (count init : ExprId) (iter : Binder) (body : ExprId)
  | chain (count init : ExprId) (iter : Binder) (body : ExprId)
  | map2 (over : ExprId) (elem : Binder) (body : ExprId)
  | zipWith (a b : ExprId) (x y : Binder) (body : ExprId)
  | letIn (binders : Array ELetBinder) (body : ExprId)
  | tag (def_ : TypeDefIdx) (variant : Nat) (payload : Array ETagPayload)
  | match_ (def_ : TypeDefIdx) (scrutinee : ExprId) (arms : Array EMatchArm)
  /-- The iteration index of the `bankSum` region whose `idxId` equals `id`
      (the post-strata analogue of `Plan.NOperand.loopIdx`). Unlike the
      combinators above, `bankSum`/`loopIdx` are NOT unrolled by arrayLower —
      they survive to the post-strata IR as an indexed reduction (banks-as-data
      slice 3b). `id` is a UNIQUE BINDER ID, not a de Bruijn index: it is
      stable under nesting (wrapping an inner bank in an outer one changes
      nothing about the inner's spelling — no shifting exists anywhere), and it
      participates in the structural hash so two distinct indices are two
      distinct DAG nodes. Ids need only be unique along a NESTING CHAIN
      (ancestors): resolution is "search the stack of open regions for this
      id", and the emitters fail on an unresolved id or an ancestor collision. -/
  | loopIdx (id : Nat)
  /-- An indexed reduction `Σ_{k<count} body(k)`, i64-modular so the sum is
      associative (reordering modes moves no bit — the bit-exactness argument).
      `tables` are the loop-invariant coefficient columns the body indexes at
      `loopIdx`; carried explicitly so emit materializes them ONCE before the
      region. `body` is the per-iteration contribution (references `loopIdx` and
      `index table loopIdx`); the accumulation is emit's job, not the body's.
      `dynCount?` is the OPTIONAL runtime effective count (trip-count-as-data):
      `count` stays the static CAPACITY (= tables' length, the topology); when
      `dynCount?` is present the emitters clamp it to `[0, count]` at the loop
      head and trip that many iterations — the room-size knob, no recompile.
      `none` is today's static path, byte-identical output.
      `idxId` names the binder the body's `loopIdx id` refers to (nested banks:
      the id must be unique along the region's nesting chain — see `loopIdx`). -/
  | bankSum (count : Nat) (tables : Array ExprId) (body : ExprId)
      (dynCount? : Option ExprId := none) (idxId : Nat := 0)
deriving BEq, Repr, Inhabited

/-- O(1) structural hash — children are ids (no subtree recursion). Op tags and
    binders fold through hashable components (`.wire`, `.idx`, names). -/
def enodeHash : ENode → UInt64
  | .num n          => mixHash 1 (hash n)
  | .bool b         => mixHash 2 (hash b)
  | .arr items      => mixHash 3 (hash (items.map (·.idx)))
  | .binary t a b   => mixHash (mixHash (mixHash 4 (hash t.wire)) (hash a.idx)) (hash b.idx)
  | .unary t a      => mixHash (mixHash 5 (hash t.wire)) (hash a.idx)
  | .clamp a b c    => mixHash (mixHash (mixHash 6 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .select a b c   => mixHash (mixHash (mixHash 7 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .arraySet a b c => mixHash (mixHash (mixHash 8 (hash a.idx)) (hash b.idx)) (hash c.idx)
  | .index a b      => mixHash (mixHash 9 (hash a.idx)) (hash b.idx)
  | .zeros c        => mixHash 10 (hash c.idx)
  | .inputRef i     => mixHash 11 (hash i.idx)
  | .paramRef i     => mixHash 12 (hash i.idx)
  | .typeParamRef i => mixHash 13 (hash i.idx)
  | .bindingRef i   => mixHash 14 (hash i.idx)
  | .nestedOut i o  => mixHash (mixHash 15 (hash i.idx)) (hash o.idx)
  | .sampleRate     => 16
  | .sampleIndex    => 17
  | .fold o i a e b => mixHash (mixHash (mixHash (mixHash (mixHash 18 (hash o.idx)) (hash i.idx)) (hash a.idx.idx)) (hash e.idx.idx)) (hash b.idx)
  | .scan o i a e b => mixHash (mixHash (mixHash (mixHash (mixHash 19 (hash o.idx)) (hash i.idx)) (hash a.idx.idx)) (hash e.idx.idx)) (hash b.idx)
  | .generate c i b => mixHash (mixHash (mixHash 20 (hash c.idx)) (hash i.idx.idx)) (hash b.idx)
  | .iterate c i it b => mixHash (mixHash (mixHash (mixHash 21 (hash c.idx)) (hash i.idx)) (hash it.idx.idx)) (hash b.idx)
  | .chain c i it b => mixHash (mixHash (mixHash (mixHash 22 (hash c.idx)) (hash i.idx)) (hash it.idx.idx)) (hash b.idx)
  | .map2 o e b     => mixHash (mixHash (mixHash 23 (hash o.idx)) (hash e.idx.idx)) (hash b.idx)
  | .zipWith a b x y bd => mixHash (mixHash (mixHash (mixHash (mixHash 24 (hash a.idx)) (hash b.idx)) (hash x.idx.idx)) (hash y.idx.idx)) (hash bd.idx)
  | .letIn bs b     => mixHash (mixHash 25 (hash (bs.map (fun lb => (lb.binder.idx.idx, lb.value.idx))))) (hash b.idx)
  | .tag d v p      => mixHash (mixHash (mixHash 26 (hash d.idx)) (hash v)) (hash (p.map (fun tp => (tp.field, tp.value.idx))))
  | .match_ d s arms => mixHash (mixHash (mixHash 27 (hash d.idx)) (hash s.idx)) (hash (arms.map (fun a => (a.variant, a.body.idx))))
  | .loopIdx id     => mixHash 28 (hash id)
  | .bankSum c ts b dc ii => mixHash (mixHash (mixHash (mixHash (mixHash 29 (hash c)) (hash (ts.map (·.idx)))) (hash b.idx)) (hash (dc.map (·.idx)))) (hash ii)

instance : Hashable ENode := ⟨enodeHash⟩

/-- Interned resolved-expression node store. Append-only; ids are assigned in
    first-seen order; `dedup` collapses equal nodes. -/
structure ExprArena where
  nodes : Array ENode := #[]
  dedup : Std.HashMap ENode ExprId := {}
deriving Inhabited

/-- `Repr` over the node array (the `dedup` map is a derived index). Lets
    containers of `ExprArena` (`Arena`) keep their derived `Repr`. -/
instance : Repr ExprArena where
  reprPrec a _ := repr a.nodes

abbrev EArenaM := StateM ExprArena

/-- Intern a flat node, returning its (possibly shared) id. -/
def eintern (n : ENode) : EArenaM ExprId := do
  let a ← get
  match a.dedup.get? n with
  | some id => pure id
  | none =>
    let id : ExprId := ⟨a.nodes.size⟩
    set { a with nodes := a.nodes.push n, dedup := a.dedup.insert n id }
    pure id

def ExprArena.deref (a : ExprArena) (id : ExprId) : Option ENode :=
  a.nodes[id.idx]?

-- ─────────────────────────────────────────────────────────────
-- Decls + assigns + program
-- ─────────────────────────────────────────────────────────────

structure InputDecl where
  name : String
  type? : Option PortType := none
  default? : Option ExprId := none
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
  value : ExprId
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
  expr : ExprId
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
  /-- The shared hash-consed expression DAG every program's leaf `ExprId`s
      index into. (Populated once `Program` is id-valued; `{}` until then.) -/
  exprs : ExprArena := {}
deriving Repr, Inhabited

def Arena.program? (a : Arena) (i : ProgramIdx) : Option Program :=
  a.programs[i.idx]?

def Arena.typeDef? (a : Arena) (i : TypeDefIdx) : Option TypeDef :=
  a.typeDefs[i.idx]?

def Arena.typeParam? (a : Arena) (i : TypeParamPoolIdx) : Option TypeParamDecl :=
  a.typeParams[i.idx]?

end Tropical.Ir
