import Std.Data.HashMap
import Lean.Data.Json
import Tropical.Parse.Nodes

/-!
# Resolved IR — the trunk program shape

A `Program` is a typed signal-flow graph: ports, params, instances,
output assigns, with every expression leaf an `ExprId` into the shared
hash-consed `ExprArena` DAG. The `Arena` is the identity pool — a
program at index `i` references only programs below it, so the pool is
acyclic by construction.

Within a program, refs are positional: `InputIdx` / `OutputIdx` /
`ParamIdx` / `InstanceIdx` index the enclosing program's typed decl
tables (computed from `decls`, so the body↔table invariant holds by
construction).

`ENode` is the trunk constructor set — the same vocabulary `Sig`
authors and the backends consume. Retired constructs (combinators, sum
types, binders, generics) have no representation here; the JSON front
doors refuse their spellings at ingest.

Numbers are `Lean.JsonNumber` (decimal text preserved) so encode output
re-parses to the bit-identical double, mirroring
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

-- Pool indices (arena-level identity).

/-- Index into `Arena.programs`. -/
structure ProgramIdx where
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
-- Port types
--
-- Trunk-only: user type defs (alias/struct/sum) and type-param shape
-- dims were retired with generics and the sum-type lowering — a port
-- is a builtin scalar kind or an array of one with literal dims.
-- ─────────────────────────────────────────────────────────────

inductive PortType where
  | scalar (k : ScalarKind)
  | array (element : ScalarKind) (shape : Array JsonNumber)
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

theorem BinaryOpTag.ofWire_wire (t : BinaryOpTag) :
    BinaryOpTag.ofWire? t.wire = some t := by
  cases t <;> rfl

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

theorem UnaryOpTag.ofWire_wire (t : UnaryOpTag) :
    UnaryOpTag.ofWire? t.wire = some t := by
  cases t <;> rfl

def UnaryOpTag.ofParse : Tropical.Parse.UnaryOpTag → UnaryOpTag
  | .neg => .neg | .not => .not | .bitNot => .bitNot

-- ─────────────────────────────────────────────────────────────
-- Expressions
-- ─────────────────────────────────────────────────────────────

-- The tree `Expr` (and its binder machinery) is gone: the resolved
-- expression IS the hash-consed `ENode`/`ExprArena` DAG below, and
-- `Program`'s leaves are `ExprId`s. Authoring frontends (the JSON codec's
-- `ParsedExpr`, `EmitArrow`'s combinator tree) build their own shapes and
-- lower into the arena.

-- ─────────────────────────────────────────────────────────────
-- ExprArena — the hash-consed (DAG) form of the resolved expression
--
-- The native-DAG representation for the whole lowering (issue #190). An
-- `ENode` is flat — its children are `ExprId`s — so it is O(1) to hash and
-- compare, and interning at construction makes two equal subtrees one node.
-- This is THE resolved-expression representation: `Program`'s expression leaves
-- are `ExprId`s into an `ExprArena`; there is no tree `Expr` twin.
-- ─────────────────────────────────────────────────────────────

/-- A resolved expression node with children referenced by `ExprId`; flat (no
    inlined subtrees). The trunk constructor set — the same vocabulary `Sig`
    authors and emit consumes. The combinator/sum-type/binder tail (`fold`,
    `scan`, `generate`, `iterate`, `chain`, `map2`, `zipWith`, `let`,
    `tag`/`match`, `bindingRef`, `typeParamRef`, `zeros`) was retired with its
    producers; those ops are refused at the JSON front doors, so they are
    unspellable here by construction. -/
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
  | inputRef (idx : InputIdx)
  | paramRef (idx : ParamIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
  /-- The iteration index of the `bankSum` region whose `idxId` equals `id`
      (the post-strata analogue of `Plan.NOperand.loopIdx`).
      `id` is a UNIQUE BINDER ID, not a de Bruijn index: it is
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
  | .inputRef i     => mixHash 11 (hash i.idx)
  | .paramRef i     => mixHash 12 (hash i.idx)
  | .nestedOut i o  => mixHash (mixHash 15 (hash i.idx)) (hash o.idx)
  | .sampleRate     => 16
  | .sampleIndex    => 17
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

/-- The edge set of the DAG: every child id the node references
    (including `bankSum` tables/body/dynCount). -/
def ENode.children : ENode → Array ExprId
  | .num _ | .bool _ | .inputRef _ | .paramRef _
  | .nestedOut _ _ | .sampleRate | .sampleIndex
  | .loopIdx _ => #[]
  | .arr items => items
  | .binary _ a b => #[a, b]
  | .unary _ a => #[a]
  | .clamp a b c => #[a, b, c]
  | .select a b c => #[a, b, c]
  | .arraySet a b c => #[a, b, c]
  | .index a b => #[a, b]
  | .bankSum _ ts b dc _ => (ts.push b) ++ dc.toArray

/-- Child-descending well-formedness: every edge points to a strictly
    smaller id. `eintern` assigns `⟨nodes.size⟩` to each fresh node, so
    every arena built through it satisfies this by construction; the
    check makes the fact available as data, and the conversion walks
    terminate by descent on `idx` against it. -/
def ExprArena.wf (a : ExprArena) : Bool :=
  (Array.range a.nodes.size).all fun i =>
    match a.nodes[i]? with
    | some n => n.children.all (fun c => c.idx < i)
    | none => true

/-- The elimination form of `wf`: a dereferenced node's children are all
    strictly below it — the decrease fact for `termination_by id.idx`. -/
theorem ExprArena.forall_children_lt {a : ExprArena} (hw : a.wf = true)
    {id : ExprId} {n : ENode} (hd : a.deref id = some n) :
    ∀ c ∈ n.children, c.idx < id.idx := by
  intro c hc
  have hd' : a.nodes[id.idx]? = some n := hd
  obtain ⟨hi, -⟩ := Array.getElem?_eq_some_iff.mp hd'
  have hnode := Array.all_eq_true.mp hw id.idx (by simpa using hi)
  rw [Array.getElem_range, hd'] at hnode
  obtain ⟨i, hilt, rfl⟩ := Array.mem_iff_getElem.mp hc
  exact of_decide_eq_true (Array.all_eq_true.mp hnode i hilt)

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

structure InstanceInput where
  port : InputIdx
  value : ExprId
deriving Repr, Inhabited

inductive BodyDecl where
  | param (name : String) (value? : Option JsonNumber)
  | inst (name : String) (typeKey : String) (inputs : Array InstanceInput)
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
    ordered association array — insertion order is observable through
    the codec, so it is load-bearing here. -/
structure Program where
  name : String
  inputs : Array InputDecl := #[]
  outputs : Array OutputDecl := #[]
  decls : Array BodyDecl := #[]
  assigns : Array OutputAssign := #[]
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

/-- The identity pools of the resolved IR: the program pool plus the
    shared expression DAG. References into the pool are by index;
    sharing an index is pointer sharing. Invariant (maintained by both
    the codec's forward-pass decode and the elaborator's
    children-before-parents construction): a program at index `i`
    references only programs at indices `< i`, so the arena is acyclic
    by construction. -/
structure Arena where
  programs : Array Program := #[]
  /-- The shared hash-consed expression DAG every program's leaf `ExprId`s
      index into. -/
  exprs : ExprArena := {}
deriving Repr, Inhabited

def Arena.program? (a : Arena) (i : ProgramIdx) : Option Program :=
  a.programs[i.idx]?

end Tropical.Ir
