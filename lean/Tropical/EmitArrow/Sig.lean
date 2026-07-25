import Tropical.Ir.Nodes

/-!
# EmitArrow.Sig — the authoring substrate

The build-time expression tree (`Sig`), its lowering into the resolved-IR
arena (`lowerSig` — pointer-memoized), the authoring-side decls, and
`assemble`, THE one boundary through which every EmitArrow-built `Program`
enters the arena. Plus the smart-constructor op set (`lit`/`mul`/…/`ldexpE`):
one algebra of scalar operations, applied to values or to the clock alike.
-/

namespace Tropical.EmitArrow

open Lean (JsonNumber)
open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- Builder substrate: smart constructors over a local authoring tree
-- ─────────────────────────────────────────────────────────────

/-- EmitArrow's authoring-tree expression: the scalar subset the combinators
    build, kept as a small tree (like the parser's `ParsedExpr`) and lowered
    into the resolved-IR arena (`lowerSig`) when a `Program` is assembled. The
    resolved IR itself is id-only (`ExprId`); this is a build-time frontend. -/
inductive Sig where
  | num (n : JsonNumber)
  | binary (tag : BinaryOpTag) (lhs rhs : Sig)
  | unary (tag : UnaryOpTag) (arg : Sig)
  | clamp (value lo hi : Sig)
  | select (cond then_ else_ : Sig)
  | inputRef (idx : InputIdx)
  | paramRef (idx : ParamIdx)
  | nestedOut (instance_ : InstanceIdx) (output : OutputIdx)
  | sampleRate
  | sampleIndex
  -- Banks-as-data: the authoring tree can express an indexed
  -- reduction over coefficient columns. `arr`/`index`/`loopIdx` lower to the
  -- like-named IR nodes; `bankSum` to `ENode.bankSum` (a `ReduceBegin` region).
  -- `loopIdx` carries the UNIQUE BINDER ID of the `bankSum` it reads (nested
  -- banks: unique along a nesting chain; Sig-level authoring uses `bankFold`,
  -- whose banks never nest today — tables materialize before their region —
  -- so `BankCols.idxId` defaults to 0).
  | arr (items : Array Sig)
  | index (arr idx : Sig)
  | loopIdx (id : Nat)
  -- `count` is the static CAPACITY; `dynCount?` (trip-count-as-data) is the
  -- optional RUNTIME effective count, clamped to `[0, count]` at the loop head.
  -- (No `:= none` default here: `Option Sig` is a NESTED occurrence of the
  -- inductive, and the kernel rejects optParam defaults on nested fields.)
  -- `idxId` names the binder the body's `loopIdx id` refers to.
  | bankSum (count : Nat) (tables : Array Sig) (body : Sig)
      (dynCount? : Option Sig) (idxId : Nat)
deriving Repr, Inhabited

/-- A clock-as-value expression (Q32.32 fixed-point sample coordinate). -/
abbrev Clock := Sig

/-- Reference (tree-walk) lowering of an authoring `Sig` into the resolved
    arena, interning bottom-up. Runtime uses the memoized `lowerSigPtr`:
    the combinators build `Sig` values that share subterms by Lean object
    REFERENCE (e.g. residue composition embeds one voice's H(ν) sum into
    every coupling amp), so the structural walk pays for the fully
    expanded tree only for `eintern` to collapse it straight back into a
    shared DAG — super-linear on composed patches. -/
partial def lowerSigTree : Sig → EArenaM ExprId
  | .num n          => eintern (.num n)
  | .binary t a b   => do eintern (.binary t (← lowerSigTree a) (← lowerSigTree b))
  | .unary t a      => do eintern (.unary t (← lowerSigTree a))
  | .clamp a b c    => do eintern (.clamp (← lowerSigTree a) (← lowerSigTree b) (← lowerSigTree c))
  | .select a b c   => do eintern (.select (← lowerSigTree a) (← lowerSigTree b) (← lowerSigTree c))
  | .inputRef i     => eintern (.inputRef i)
  | .paramRef i     => eintern (.paramRef i)
  | .nestedOut i o  => eintern (.nestedOut i o)
  | .sampleRate     => eintern .sampleRate
  | .sampleIndex    => eintern .sampleIndex
  | .arr items      => do eintern (.arr (← items.mapM lowerSigTree))
  | .index a b      => do eintern (.index (← lowerSigTree a) (← lowerSigTree b))
  | .loopIdx id     => eintern (.loopIdx id)
  | .bankSum c ts b dc ii => do
    eintern (.bankSum c (← ts.mapM lowerSigTree) (← lowerSigTree b) (← dc.mapM lowerSigTree) ii)

/-- Pointer-identity-memoized lowering: a hash map from object address to
    interned id, threaded alongside the arena. Sound because `Sig` values
    are immutable — a shared pointer IS the same subterm — and lowering a
    subterm is deterministic (`eintern` returns equal ids for equal nodes),
    so a pointer MISS merely re-lowers structurally to the same id. -/
private unsafe def lowerSigPtrGo (s : Sig) :
    StateM (ExprArena × Std.HashMap USize ExprId) ExprId := do
  let key := ptrAddrUnsafe s
  if let some r := (← get).2.get? key then return r
  let intern1 (n : ENode) : StateM (ExprArena × Std.HashMap USize ExprId) ExprId :=
    fun (a, m) => let (id, a') := (eintern n).run a; (id, (a', m))
  let r ← match s with
    | .num n          => intern1 (.num n)
    | .binary t a b   => do intern1 (.binary t (← lowerSigPtrGo a) (← lowerSigPtrGo b))
    | .unary t a      => do intern1 (.unary t (← lowerSigPtrGo a))
    | .clamp a b c    => do intern1 (.clamp (← lowerSigPtrGo a) (← lowerSigPtrGo b) (← lowerSigPtrGo c))
    | .select a b c   => do intern1 (.select (← lowerSigPtrGo a) (← lowerSigPtrGo b) (← lowerSigPtrGo c))
    | .inputRef i     => intern1 (.inputRef i)
    | .paramRef i     => intern1 (.paramRef i)
    | .nestedOut i o  => intern1 (.nestedOut i o)
    | .sampleRate     => intern1 .sampleRate
    | .sampleIndex    => intern1 .sampleIndex
    | .arr items      => do intern1 (.arr (← items.mapM lowerSigPtrGo))
    | .index a b      => do intern1 (.index (← lowerSigPtrGo a) (← lowerSigPtrGo b))
    | .loopIdx id     => intern1 (.loopIdx id)
    | .bankSum c ts b dc ii => do
      intern1 (.bankSum c (← ts.mapM lowerSigPtrGo) (← lowerSigPtrGo b) (← dc.mapM lowerSigPtrGo) ii)
  modify fun (a, m) => (a, m.insert key r)
  return r

private unsafe def lowerSigPtr (s : Sig) : EArenaM ExprId := fun arena =>
  let (id, (arena', _)) := (lowerSigPtrGo s).run (arena, {})
  (id, arena')

/-- Lower an authoring `Sig` into the resolved arena, interning bottom-up.
    Compiled implementation is the pointer-memoized walk (`lowerSigPtr`);
    `lowerSigTree` is the structural reference it agrees with. -/
@[implemented_by lowerSigPtr]
def lowerSig (s : Sig) : EArenaM ExprId := lowerSigTree s

/-- Lower an optional authoring default. -/
def lowerSigOpt : Option Sig → EArenaM (Option ExprId)
  | none => pure none
  | some s => some <$> lowerSig s

-- ─────────────────────────────────────────────────────────────
-- Authoring-side decls (tree-valued) + the one lowering boundary
-- ─────────────────────────────────────────────────────────────

/-- An instance input at the authoring layer (tree `Sig` value). -/
structure AInput where
  port : InputIdx
  value : Sig
deriving Inhabited

/-- An instance decl at the authoring layer. -/
structure AInst where
  name : String
  programName : String
  inputs : Array AInput := #[]
deriving Inhabited

/-- An input-port decl at the authoring layer (tree `Sig` default). -/
structure AInputDecl where
  name : String
  type? : Option PortType := none
  defaultSig : Option Sig := none
deriving Inhabited

/-- THE lowering boundary: assemble an authoring carrier into a resolved
    `Program`, interning every tree `Sig` into `arena.exprs`, and push it into
    the arena. All of EmitArrow's `Sig`-into-`Program` conversion funnels here. -/
def assemble (arena : Arena) (name : String) (inputs : Array AInputDecl)
    (outputs : Array OutputDecl) (decls : Array AInst)
    (assigns : Array (OutputTarget × Sig))
    (registry : Array (String × ProgramIdx))
    (extraDecls : Array BodyDecl := #[]) :
    Arena × ProgramIdx :=
  let build : EArenaM Program := do
    let inputsR ← inputs.mapM fun d => do
      pure ({ name := d.name, type? := d.type?, default? := ← lowerSigOpt d.defaultSig } : InputDecl)
    let declsR ← decls.mapM fun a => do
      let ins ← a.inputs.mapM fun i => do
        pure ({ port := i.port, value := ← lowerSig i.value } : InstanceInput)
      pure (BodyDecl.inst a.name a.programName ins)
    let assignsR ← assigns.mapM fun (t, s) => do
      pure ({ target := t, expr := ← lowerSig s } : OutputAssign)
    pure { name, inputs := inputsR, outputs, decls := declsR ++ extraDecls,
           assigns := assignsR, registry }
  let (prog, exprs) := build.run arena.exprs
  ({ arena with programs := arena.programs.push prog, exprs }, ⟨arena.programs.size⟩)

/-- Decimal literal `mantissa · 10^(-exponent)`. The exponent defaults to 0
    (an integer). This mirrors the `JsonNumber` the JSON parser produces, so
    the emitted bytes match: `lit 5 1 = 0.5`, `lit 25 2 = 0.25`, `lit 7 4 =
    0.0007`, `lit 4294967296 = 2³²`. -/
def lit (mantissa : Int) (exponent : Nat := 0) : Sig := .num ⟨mantissa, exponent⟩

/-- An INTEGER literal for clock arithmetic. A bare `lit` only narrows to int
    inside an int-EXPECTED context (bitwise ops, `to_int`); in plain `add`/
    `mul`/`sub` against the integer clock it stays float and PROMOTES the whole
    op to float (`inferResultType`/`promoteTypes`) — the clock gets `sitofp`'d
    and loses sub-sample bits once the value passes 2⁵³ (≈ 47 seconds of
    absolute Q32.32 clock at 44.1k). Wrapping the literal in `toInt` pins the
    op to exact i64. Use this for every literal that meets a clock. -/
def litI (mantissa : Int) : Sig := .unary .toInt (.num ⟨mantissa, 0⟩)

/-- `arr`-level pointwise multiply. -/
def mul (a b : Sig) : Sig := .binary .mul a b
/-- `arr`-level pointwise add. -/
def add (a b : Sig) : Sig := .binary .add a b
/-- `arr`-level pointwise subtract. -/
def sub (a b : Sig) : Sig := .binary .sub a b
/-- `arr`-level truncate-to-int (the Q32.32 offset is an integer sample count). -/
def toIntE (a : Sig) : Sig := .unary .toInt a
/-- Negate — the negate EFFECT (`.unary .neg`, emitted `sub i64 0, c`). The SAME
    operation whether applied to a signal value or to the clock; applied to the
    clock it IS `reverse`. There is one algebra of operations on expressions; the
    clock is one of those expressions. -/
def neg (a : Sig) : Sig := .unary .neg a

/-! More of the universal scalar op set — plain `.binary`/`.unary`/`.clamp`
    wrappers, staying scalar (the post-strata IR is scalar by definition). These
    are what a generator/closed-form voice (phasor arithmetic + polynomial)
    needs beyond the flanger's add/sub/mul: division, the bit ops the fixed-point
    phasor speaks, rounding, the int⇆float casts, and the bounded-type clamp the
    elaborator inserts for `unipolar`/`freq` ports. -/

/-- `arr`-level pointwise divide. -/
def div (a b : Sig) : Sig := .binary .div a b
/-- Bitwise AND on the integer (fixed-point) clock/value (`& (2³²−1)` masks). -/
def bitAnd (a b : Sig) : Sig := .binary .bitAnd a b
/-- Logical/arithmetic right shift (`clk >> 32` etc.; the mask makes the choice
    irrelevant where it follows a `& (2³²−1)`). -/
def rshift (a b : Sig) : Sig := .binary .rshift a b
/-- Left shift (`sampleIndex << 32` — the root clock). -/
def lshift (a b : Sig) : Sig := .binary .lshift a b
/-- `gt` comparison (the bounded-default `select(hz > 0, …)`). -/
def gt (a b : Sig) : Sig := .binary .gt a b
/-- Round to nearest integer-as-float (`Sin`'s half-cycle count `n`). -/
def roundE (a : Sig) : Sig := .unary .round a
/-- Reinterpret int → float (`toFloat(acc & mask)` at the phasor boundary). -/
def toFloatE (a : Sig) : Sig := .unary .toFloat a
/-- Clamp into `[lo, hi]` — the elaborator's lowering of a bounded port type
    (`unipolar` ⇒ `clamp _ 0 1`). -/
def clampE (value lo hi : Sig) : Sig := .clamp value lo hi
/-- Select (`cond ? then : else`) — the bounded-default `select(hz > 0, hz, 0)`. -/
def selectE (cond then_ else_ : Sig) : Sig := .select cond then_ else_
/-- `ldexp(m, n) = m·2ⁿ` — the exact float-exponent bump `stdlib/Exp` uses for the
    integer part of range reduction. `n` is an integer-valued float (`round …`). -/
def ldexpE (m n : Sig) : Sig := .binary .ldexp m n

/-- Does this expression read the sample clock (`.sampleIndex`)? That read is the
    authoring layer's source of per-sample (s1) τ-dependence — a value with none is
    a function of settled parameters alone (s0). -/
partial def readsSampleIndex : Sig → Bool
  | .sampleIndex        => true
  | .binary _ a b       => readsSampleIndex a || readsSampleIndex b
  | .unary _ a          => readsSampleIndex a
  | .clamp v lo hi      => readsSampleIndex v || readsSampleIndex lo || readsSampleIndex hi
  | .select c t e       => readsSampleIndex c || readsSampleIndex t || readsSampleIndex e
  | .index a b          => readsSampleIndex a || readsSampleIndex b
  | .arr items          => items.any readsSampleIndex
  | .bankSum _ ts b _ _  => ts.any readsSampleIndex || readsSampleIndex b
  | _                   => false

private def isZeroLit : Sig → Bool | .num n => n.mantissa == 0 | _ => false
private def isOneLit  : Sig → Bool | .num n => n.mantissa == 1 && n.exponent == 0 | _ => false

/-- Collapse every SATURATING glide ramp to its terminal value — a `clampE(r, 0, 1)`
    whose ramp `r` reads the clock (`glideExpr`'s smoothstep parameter, the only
    converging τ-dependent construct authored today) reaches `1` as the clock runs,
    so it settles to its ceiling. Every other subterm settles pointwise. This is the
    mechanical half of `settle`; use `settle` for the totality contract. -/
partial def settleRamps : Sig → Sig
  | .clamp v lo hi =>
    if isZeroLit lo && isOneLit hi && readsSampleIndex v then lit 1
    else .clamp (settleRamps v) (settleRamps lo) (settleRamps hi)
  | .binary t a b       => .binary t (settleRamps a) (settleRamps b)
  | .unary t a          => .unary t (settleRamps a)
  | .select c t e       => .select (settleRamps c) (settleRamps t) (settleRamps e)
  | .index a b          => .index (settleRamps a) (settleRamps b)
  | .arr items          => .arr (items.map settleRamps)
  | .bankSum c ts b dc ii => .bankSum c (ts.map settleRamps) (settleRamps b) dc ii
  | s                   => s

/-- `settle e` — the τ-independent (s0) value `e` converges to as the clock runs,
    when that value exists; `none` otherwise. The one converging τ-dependent
    construct today is `glideExpr`'s saturating ramp, which `settleRamps` collapses
    to its ceiling; a `.sampleIndex` that SURVIVES that collapse is a genuine
    modulation (an LFO on a pole never settles — it has no fixed point), so `settle`
    returns `none` and the caller must DECLINE rather than measure a value the signal
    never sits at. The guarantee that makes this a contract, not a hope: a `some`
    result reads no clock — **s0 by construction**. So a coefficient built through
    `settle` (a self-measuring gauge norm) cannot emit a per-sample op, and the
    f32≠f64 `floatExponent` split across backends is unREACHABLE, not merely
    documented against. -/
def settle (s : Sig) : Option Sig :=
  let s' := settleRamps s
  if readsSampleIndex s' then none else some s'

/-- Mirror the elaborator's `registerInstanceDecl` over a sequence of
    instantiated program names: each adds `(name, idx)` then merges that
    program's own registry (skipping keys already present), in declaration
    order. Generalizes `buildWarpBank`'s single-voice registry merge to the
    several distinct programs a multi-instance body references. -/
def buildRegistry (arena : Arena) (resolved : Array (String × ProgramIdx))
    (programNames : Array String) : Except String (Array (String × ProgramIdx)) := do
  let mut registry : Array (String × ProgramIdx) := #[]
  for pn in programNames do
    let some idx := (resolved.find? (·.1 == pn)).map (·.2)
      | .error s!"EmitArrow: program '{pn}' not found in the elaborated stdlib chain"
    let some prog := arena.program? idx
      | .error s!"EmitArrow: program '{pn}' index out of range"
    if !registry.any (·.1 == prog.name) then registry := registry.push (prog.name, idx)
    for (k, v) in prog.registry do
      if !registry.any (·.1 == k) then registry := registry.push (k, v)
  pure registry

/-- Encode a build-time `Float` as a decimal `lit` (12 places, round-to-nearest) —
    the bridge that lets residue-computed complex coefficients become a bank's
    literal params. Magnitude must fit `·10¹²` in `UInt64` (fine for audio
    σ/f/amp/phase). -/
def litF (x : Float) : Sig :=
  let s := x * 1000000000000.0
  let m : Int := if s ≥ 0.0 then Int.ofNat (s + 0.5).toUInt64.toNat
                 else -(Int.ofNat (0.5 - s).toUInt64.toNat)
  -- a zero-mantissa NumLit with a nonzero exponent serializes to invalid JSON
  -- ("0.e-12"); a plain `0` avoids it.
  if m == 0 then lit 0 else lit m 12

end Tropical.EmitArrow
