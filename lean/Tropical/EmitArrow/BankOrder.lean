import Tropical.EmitArrow.Modal

/-!
# Bank order-preservation as a theorem (slice 3c)

The claim that was prose (`Ir/Emit.lean`'s region docstring, CLAUDE.md's
"order preservation makes every realization bit-identical") becomes a
theorem here — carrier-parametric, exactly as scoped in
`design/bank-order-proof-handoff.local.md`:

* **`instLoop`** is the β-rule of the `bankSum` binder: instantiate
  `loopIdx id` at a concrete index `k`, reducing the column redex
  `index (arr xs) (loopIdx id)` to `xs[k]` in place. Capture-aware: a
  nested `bankSum` that rebinds the same id shields its body.
* **`unrollBanks`** is the REFERENCE REALIZATION of banks-as-data at the
  `Sig` level: homomorphic on every constructor, and a *static* `bankSum`
  becomes the left-nested `add` fold over `0 … count−1` in array order —
  the same tree the unrolled builder writes by hand. A *dynamic* bank
  (`dynCount? = some _`) has no static unrolling; its law is the emit-half
  clamp statement (WS-B), not a `Sig` equation.
* **WS-A** (`unrollBanks_modalBankSigTable`) states the builder's two
  branches agree SYNTACTICALLY: `unrollBanks (modalBankSigTable …) =
  modalBankSig …`. Syntactic `Sig` equality is stronger than any
  denotational statement — every denotation into every carrier maps equal
  trees to equal values, so no float model and no associativity ever enter.
* **`refFold`** and `denote_unrollBanks_bankSum` restate the fold in the
  handoff's carrier-parametric vocabulary: for ANY `d : Sig → α` that is a
  homomorphism on `add` alone, the reference realization of a bank IS
  `refFold op (d (litI 0)) (fun k => d (body@k)) count`. Which operands
  meet in which order — nothing else.

Freshness side conditions (`LoopFree`, `BankFree`) are hypotheses on the
caller-supplied leaves (clock, anchor, mode scalars), not on anything the
builders construct: they say the caller's Sigs do not capture the bank's
binder id and do not smuggle a bank of their own into the leaves. On a
concrete patch both discharge by unfolding (every real leaf is a literal,
a param ref, or a clock expression, all of which satisfy them trivially).
-/

namespace Tropical.EmitArrow

open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- Freshness predicates
-- ─────────────────────────────────────────────────────────────

/-- `s` has no FREE occurrence of `loopIdx id` — binder-aware: a nested
    `bankSum` whose `idxId` rebinds `id` shields its body (its tables and
    dynamic count are outside the binder's scope and stay checked). -/
def LoopFree (id : Nat) : Sig → Prop
  | .num _ => True
  | .binary _ a b => LoopFree id a ∧ LoopFree id b
  | .unary _ a => LoopFree id a
  | .clamp a b c => LoopFree id a ∧ LoopFree id b ∧ LoopFree id c
  | .select a b c => LoopFree id a ∧ LoopFree id b ∧ LoopFree id c
  | .inputRef _ => True
  | .paramRef _ => True
  | .nestedOut _ _ => True
  | .sampleRate => True
  | .sampleIndex => True
  | .arr items => ∀ x ∈ items, LoopFree id x
  | .index a b => LoopFree id a ∧ LoopFree id b
  | .loopIdx j => j ≠ id
  | .bankSum _ ts b dc ii =>
    (∀ x ∈ ts, LoopFree id x) ∧ (ii = id ∨ LoopFree id b) ∧
      (∀ d ∈ dc, LoopFree id d)
  | .routedSum _ _ _ ts vs dc ii =>
    (∀ x ∈ ts, LoopFree id x) ∧
      (ii = id ∨ ∀ x ∈ vs, LoopFree id x) ∧
      (∀ d ∈ dc, LoopFree id d)

/-- `s` contains no `bankSum` node at all (so the reference realization
    `unrollBanks` is the identity on it). -/
def BankFree : Sig → Prop
  | .num _ => True
  | .binary _ a b => BankFree a ∧ BankFree b
  | .unary _ a => BankFree a
  | .clamp a b c => BankFree a ∧ BankFree b ∧ BankFree c
  | .select a b c => BankFree a ∧ BankFree b ∧ BankFree c
  | .inputRef _ => True
  | .paramRef _ => True
  | .nestedOut _ _ => True
  | .sampleRate => True
  | .sampleIndex => True
  | .arr items => ∀ x ∈ items, BankFree x
  | .index a b => BankFree a ∧ BankFree b
  | .loopIdx _ => True
  | .bankSum _ _ _ _ _ => False
  | .routedSum _ _ _ ts vs dc _ =>
    (∀ x ∈ ts, BankFree x) ∧ (∀ x ∈ vs, BankFree x) ∧
      (∀ d ∈ dc, BankFree d)

-- ─────────────────────────────────────────────────────────────
-- instLoop — the binder's β-rule
-- ─────────────────────────────────────────────────────────────

/-- Collapse an array read at a CONCRETE index: a literal array indexed at
    `k` is its `k`-th element. Any other shape (never builder-produced —
    columns are always `Sig.arr`) keeps the explicit indexed form. -/
def readAt (k : Nat) : Sig → Sig
  | .arr xs => if h : k < xs.size then xs[k] else .index (.arr xs) (litI k)
  | a => .index a (litI k)

/-- Instantiate the loop binder `id` at concrete index `k`: `loopIdx id`
    becomes the integer literal `k`, and the column-read redex
    `index (arr xs) (loopIdx id)` reduces to the instantiated array's
    `k`-th element (`readAt` — the element is itself instantiated first;
    columns may read an OUTER binder). A nested `bankSum` that rebinds `id`
    shields its body — capture-avoidance, though the unique-binder-id
    discipline (WS5) means the shield is never exercised on a
    builder-produced tree. -/
def instLoop (id k : Nat) : Sig → Sig
  | .num n => .num n
  | .binary t a b => .binary t (instLoop id k a) (instLoop id k b)
  | .unary t a => .unary t (instLoop id k a)
  | .clamp a b c => .clamp (instLoop id k a) (instLoop id k b) (instLoop id k c)
  | .select a b c => .select (instLoop id k a) (instLoop id k b) (instLoop id k c)
  | .inputRef i => .inputRef i
  | .paramRef i => .paramRef i
  | .nestedOut i o => .nestedOut i o
  | .sampleRate => .sampleRate
  | .sampleIndex => .sampleIndex
  | .arr items => .arr (items.attach.map fun ⟨x, _⟩ => instLoop id k x)
  | .index a b =>
    match b with
    | .loopIdx j =>
      if j = id then readAt k (instLoop id k a)
      else .index (instLoop id k a) (.loopIdx j)
    | b => .index (instLoop id k a) (instLoop id k b)
  | .loopIdx j => if j = id then litI k else .loopIdx j
  | .bankSum c ts b dc ii =>
    .bankSum c (ts.attach.map fun ⟨x, _⟩ => instLoop id k x)
      (if ii = id then b else instLoop id k b)
      (match dc with | none => none | some d => some (instLoop id k d)) ii
  | .routedSum c oc rs ts vs dc ii =>
    .routedSum c oc rs
      (ts.attach.map fun ⟨x, _⟩ => instLoop id k x)
      (if ii = id then vs else
        vs.attach.map fun ⟨x, _⟩ => instLoop id k x)
      (match dc with | none => none | some d => some (instLoop id k d)) ii

-- ─────────────────────────────────────────────────────────────
-- unrollBanks — the reference realization
-- ─────────────────────────────────────────────────────────────

/-- The reference realization of banks-as-data: every STATIC `bankSum`
    becomes the left-nested `add` fold over its body instantiated at
    `0 … count−1`, in array order — `add (… (add (add 0 b₀) b₁) …) b₍ₙ₋₁₎`,
    the exact tree the unrolled builder (`modalBankSig`'s `foldl`) writes.
    Inner banks unroll first (the recursion is on the body), then the outer
    index instantiates through the result — WS-C's commutation lemma shows
    the order doesn't matter. Tables vanish: every read of them was a redex
    `instLoop` reduced away. A DYNAMIC bank (`dynCount? = some _`) is kept
    as a node (its trip count is runtime data; its law is the emit-half
    clamp, WS-B), with its pieces unrolled recursively. -/
def unrollBanks : Sig → Sig
  | .num n => .num n
  | .binary t a b => .binary t (unrollBanks a) (unrollBanks b)
  | .unary t a => .unary t (unrollBanks a)
  | .clamp a b c => .clamp (unrollBanks a) (unrollBanks b) (unrollBanks c)
  | .select a b c => .select (unrollBanks a) (unrollBanks b) (unrollBanks c)
  | .inputRef i => .inputRef i
  | .paramRef i => .paramRef i
  | .nestedOut i o => .nestedOut i o
  | .sampleRate => .sampleRate
  | .sampleIndex => .sampleIndex
  | .arr items => .arr (items.attach.map fun ⟨x, _⟩ => unrollBanks x)
  | .index a b => .index (unrollBanks a) (unrollBanks b)
  | .loopIdx j => .loopIdx j
  | .bankSum c _ts b none ii =>
    (List.range c).foldl
      (fun acc k => .binary .add acc (instLoop ii k (unrollBanks b)))
      (litI 0)
  | .bankSum c ts b (some d) ii =>
    .bankSum c (ts.attach.map fun ⟨x, _⟩ => unrollBanks x)
      (unrollBanks b) (some (unrollBanks d)) ii
  | .routedSum c oc rs ts vs dc ii =>
    .routedSum c oc rs
      (ts.attach.map fun ⟨x, _⟩ => unrollBanks x)
      (vs.attach.map fun ⟨x, _⟩ => unrollBanks x)
      (match dc with | none => none | some d => some (unrollBanks d)) ii

-- ─────────────────────────────────────────────────────────────
-- refFold — the handoff's carrier-parametric vocabulary
-- ─────────────────────────────────────────────────────────────

/-- Reference semantics of an indexed reduction over an ARBITRARY carrier:
    the fold in array order. `refFold op z f (n+1) = op (refFold op z f n)
    (f n)` — element `n` meets the accumulator last. No associativity, no
    commutativity, no float model: the statement is about which operands
    meet in which order, nothing else. -/
def refFold {α : Sort _} (op : α → α → α) (z : α) (f : Nat → α) : Nat → α
  | 0 => z
  | n + 1 => op (refFold op z f n) (f n)

-- ─────────────────────────────────────────────────────────────
-- Equation lemmas for the overlapping `index` match
-- ─────────────────────────────────────────────────────────────

/-- The column-read β-redex: indexing a literal array at the binder being
    instantiated reduces to the (instantiated) element. -/
theorem instLoop_redex {id k : Nat} {xs : Array Sig} (h : k < xs.size) :
    instLoop id k (.index (.arr xs) (.loopIdx id)) = instLoop id k (xs[k]'h) := by
  simp [instLoop, readAt, h]

-- ─────────────────────────────────────────────────────────────
-- Identity lemmas: fresh trees are fixed points
-- ─────────────────────────────────────────────────────────────

private theorem attach_map_eq_of_id {f : Sig → Sig} (xs : Array Sig)
    (hf : ∀ x ∈ xs, f x = x) :
    (xs.attach.map fun ⟨x, _⟩ => f x) = xs := by
  apply Array.ext
  · simp
  · intro i h1 h2
    simp only [Array.getElem_map, Array.getElem_attach]
    exact hf xs[i] (Array.getElem_mem h2)

/-- Instantiation is the identity on a tree with no free occurrence of the
    binder: `instLoop` can only rewrite at `loopIdx id` (directly or through
    the column redex), and `LoopFree` says there is none. -/
theorem instLoop_of_loopFree {id k : Nat} :
    ∀ {s : Sig}, LoopFree id s → instLoop id k s = s := by
  intro s
  induction s using instLoop.induct id with
  | case1 n => intro _; simp [instLoop]
  | case2 t a b iha ihb =>
    intro h; simp only [LoopFree] at h
    simp [instLoop, iha h.1, ihb h.2]
  | case3 t a ih =>
    intro h; simp only [LoopFree] at h
    simp [instLoop, ih h]
  | case4 a b c iha ihb ihc =>
    intro h; simp only [LoopFree] at h
    simp [instLoop, iha h.1, ihb h.2.1, ihc h.2.2]
  | case5 a b c iha ihb ihc =>
    intro h; simp only [LoopFree] at h
    simp [instLoop, iha h.1, ihb h.2.1, ihc h.2.2]
  | case6 i => intro _; simp [instLoop]
  | case7 i => intro _; simp [instLoop]
  | case8 i o => intro _; simp [instLoop]
  | case9 => intro _; simp [instLoop]
  | case10 => intro _; simp [instLoop]
  | case11 items ih =>
    intro h; simp only [LoopFree] at h
    simp only [instLoop]
    rw [attach_map_eq_of_id items (fun x hx => ih x hx (h x hx))]
  | case12 a iha =>
    -- reading at the binder itself: contradicts LoopFree (`loopIdx id`)
    intro hf; simp only [LoopFree] at hf
    exact absurd rfl hf.2
  | case13 a j hne iha =>
    intro h; simp only [LoopFree] at h
    simp [instLoop, hne, iha h.1]
  | case14 a b hg iha ihb =>
    intro h; simp only [LoopFree] at h
    simp only [instLoop]
    rw [iha h.1, ihb h.2]
  | case15 => intro hf; simp only [LoopFree] at hf; exact absurd rfl hf
  | case16 j hne => intro _; simp [instLoop, hne]
  | case17 c ts b dc ii ihts ihb ihdc =>
    intro h; simp only [LoopFree] at h
    have hbody : (if ii = id then b else instLoop id k b) = b := by
      split
      · rfl
      · rename_i hii
        exact ihb (h.2.1.resolve_left hii)
    cases dc with
    | none =>
      simp only [instLoop]
      rw [attach_map_eq_of_id ts (fun x hx => ihts x hx (h.1 x hx)), hbody]
    | some d =>
      simp only [instLoop]
      rw [attach_map_eq_of_id ts (fun x hx => ihts x hx (h.1 x hx)), hbody,
        ihdc (h.2.2 d rfl)]
  | case18 c oc rs ts vs dc ii ihts ihvs ihdc =>
    intro h; simp only [LoopFree] at h
    have hvalues :
        (if ii = id then vs else
          vs.attach.map fun ⟨x, _⟩ => instLoop id k x) = vs := by
      split
      · rfl
      · rename_i hii
        rw [attach_map_eq_of_id vs (fun x hx =>
          ihvs x hx (h.2.1.resolve_left hii x hx))]
    cases dc with
    | none =>
      simp only [instLoop]
      rw [attach_map_eq_of_id ts (fun x hx => ihts x hx (h.1 x hx)), hvalues]
    | some d =>
      simp only [instLoop]
      rw [attach_map_eq_of_id ts (fun x hx => ihts x hx (h.1 x hx)), hvalues,
        ihdc (h.2.2 d rfl)]

/-- The reference realization is the identity on a bank-free tree. -/
theorem unrollBanks_of_bankFree :
    ∀ {s : Sig}, BankFree s → unrollBanks s = s := by
  intro s
  induction s using unrollBanks.induct with
  | case1 n => intro _; simp [unrollBanks]
  | case2 t a b iha ihb =>
    intro h; simp only [BankFree] at h
    simp [unrollBanks, iha h.1, ihb h.2]
  | case3 t a ih =>
    intro h; simp only [BankFree] at h
    simp [unrollBanks, ih h]
  | case4 a b c iha ihb ihc =>
    intro h; simp only [BankFree] at h
    simp [unrollBanks, iha h.1, ihb h.2.1, ihc h.2.2]
  | case5 a b c iha ihb ihc =>
    intro h; simp only [BankFree] at h
    simp [unrollBanks, iha h.1, ihb h.2.1, ihc h.2.2]
  | case6 i => intro _; simp [unrollBanks]
  | case7 i => intro _; simp [unrollBanks]
  | case8 i o => intro _; simp [unrollBanks]
  | case9 => intro _; simp [unrollBanks]
  | case10 => intro _; simp [unrollBanks]
  | case11 items ih =>
    intro h; simp only [BankFree] at h
    simp only [unrollBanks]
    rw [attach_map_eq_of_id items (fun x hx => ih x hx (h x hx))]
  | case12 a b iha ihb =>
    intro h; simp only [BankFree] at h
    simp [unrollBanks, iha h.1, ihb h.2]
  | case13 j => intro _; simp [unrollBanks]
  | case14 c ts b ii ihb =>
    intro hf; simp only [BankFree] at hf
  | case15 c ts b d ii ihts ihb ihd =>
    intro hf; simp only [BankFree] at hf
  | case16 c oc rs ts vs dc ii ihts ihvs ihdc =>
    intro h; simp only [BankFree] at h
    cases dc with
    | none =>
      simp only [unrollBanks]
      rw [attach_map_eq_of_id ts (fun x hx => ihts x hx (h.1 x hx)),
        attach_map_eq_of_id vs (fun x hx => ihvs x hx (h.2.1 x hx))]
    | some d =>
      simp only [unrollBanks]
      rw [attach_map_eq_of_id ts (fun x hx => ihts x hx (h.1 x hx)),
        attach_map_eq_of_id vs (fun x hx => ihvs x hx (h.2.1 x hx)),
        ihdc (h.2.2 d rfl)]
-- ─────────────────────────────────────────────────────────────
-- Freshness composes through folds and the landing-exponent builder
-- ─────────────────────────────────────────────────────────────

private theorem pred_foldl {β} {P : Sig → Prop} (xs : Array β) (g : Sig → β → Sig)
    (z : Sig) (hz : P z) (hstep : ∀ acc x, x ∈ xs → P acc → P (g acc x)) :
    P (xs.foldl g z) :=
  Array.foldl_induction (motive := fun _ acc => P acc) hz
    (fun i b hb => hstep b xs[i] (Array.getElem_mem i.isLt) hb)

/-- The dynamic branch of `bankLandExp` is `⌊log₂⌋` of the per-mode
    weight-bound max fold — the SHAPE fact the freshness lemmas read (the
    static branch is literals and needs nothing beyond its constructor). -/
theorem bankLandExp_dynamic_shape {modes : Array ModalMode} {ks : Sig}
    (h : bankLandExp modes = .dynamic ks) :
    ks = clampE (sub (.unary .floatExponent (modes.foldl (fun acc m =>
        selectE (gt acc (modeWeightBoundSig m)) acc (modeWeightBoundSig m))
      (lit 0))) (lit 4)) (lit 0) (lit 28) := by
  unfold bankLandExp at h
  simp only [Id.run, bind, pure] at h
  split at h
  split at h
  · exact LandExp.noConfusion h
  · cases h; rfl

private theorem loopFree_litF {id : Nat} (x : Float) : LoopFree id (litF x) := by
  simp only [litF, lit]
  split <;> split <;> simp [LoopFree]

private theorem bankFree_litF (x : Float) : BankFree (litF x) := by
  simp only [litF, lit]
  split <;> split <;> simp [BankFree]

private theorem loopFree_powE {id : Nat} {b : Sig} (hb : LoopFree id b) :
    ∀ n, LoopFree id (powE b n)
  | 0 => by simp [powE, lit, LoopFree]
  | 1 => hb
  | n + 2 => by
    simpa [powE, mul, LoopFree] using ⟨loopFree_powE hb (n + 1), hb⟩

private theorem bankFree_powE {b : Sig} (hb : BankFree b) :
    ∀ n, BankFree (powE b n)
  | 0 => by simp [powE, lit, BankFree]
  | 1 => hb
  | n + 2 => by
    simpa [powE, mul, BankFree] using ⟨bankFree_powE hb (n + 1), hb⟩

private theorem loopFree_modeWeightBoundSig {id : Nat} {m : ModalMode}
    (hs : LoopFree id m.sigma) (hcre : LoopFree id m.cre)
    (hcim : LoopFree id m.cim) : LoopFree id (modeWeightBoundSig m) := by
  unfold modeWeightBoundSig
  split
  · simp [add, LoopFree, hcre, hcim]
  · have hbase : LoopFree id (div (litF m.deg.toFloat)
        (mul m.sigma (litF 2.718281828459045))) := by
      simp [div, mul, LoopFree, loopFree_litF, hs]
    have hp := loopFree_powE hbase m.deg
    simp only [mul, add, LoopFree]
    exact ⟨hp, hcre, hcim⟩

private theorem bankFree_modeWeightBoundSig {m : ModalMode}
    (hs : BankFree m.sigma) (hcre : BankFree m.cre)
    (hcim : BankFree m.cim) : BankFree (modeWeightBoundSig m) := by
  unfold modeWeightBoundSig
  split
  · simp [add, BankFree, hcre, hcim]
  · have hbase : BankFree (div (litF m.deg.toFloat)
        (mul m.sigma (litF 2.718281828459045))) := by
      simp [div, mul, BankFree, bankFree_litF, hs]
    have hp := bankFree_powE hbase m.deg
    simp only [mul, add, BankFree]
    exact ⟨hp, hcre, hcim⟩

/-- The landing scale/shift are as fresh as the bank's mode scalars: the
    static branch is literals; the dynamic branch folds
    `modeWeightBoundSig` over the modes and freshness rides the fold. -/
theorem loopFree_landExp {id : Nat} {modes : Array ModalMode}
    (hm : ∀ m ∈ modes, LoopFree id m.sigma ∧ LoopFree id m.cre ∧ LoopFree id m.cim) :
    LoopFree id (bankLandExp modes).scale ∧ LoopFree id (bankLandExp modes).shift := by
  cases hle : bankLandExp modes with
  | static k => exact ⟨by simp [LandExp.scale, lit, LoopFree],
      by simp [LandExp.shift, lit, LoopFree]⟩
  | dynamic ks =>
    have hks := bankLandExp_dynamic_shape hle
    subst hks
    have hfold : LoopFree id (modes.foldl (fun acc m =>
        selectE (gt acc (modeWeightBoundSig m)) acc (modeWeightBoundSig m)) (lit 0)) := by
      refine pred_foldl _ _ _ (by simp [lit, LoopFree]) ?_
      intro acc x hx hacc
      have h := hm x hx
      simp [selectE, gt, LoopFree, hacc,
        loopFree_modeWeightBoundSig h.1 h.2.1 h.2.2]
    simp only [lit] at hfold
    exact ⟨by simp [LandExp.scale, ldexpE, sub, clampE, lit, LoopFree, hfold],
      by simp [LandExp.shift, sub, clampE, lit, LoopFree, hfold]⟩

theorem bankFree_landExp {modes : Array ModalMode}
    (hm : ∀ m ∈ modes, BankFree m.sigma ∧ BankFree m.cre ∧ BankFree m.cim) :
    BankFree (bankLandExp modes).scale ∧ BankFree (bankLandExp modes).shift := by
  cases hle : bankLandExp modes with
  | static k => exact ⟨by simp [LandExp.scale, lit, BankFree],
      by simp [LandExp.shift, lit, BankFree]⟩
  | dynamic ks =>
    have hks := bankLandExp_dynamic_shape hle
    subst hks
    have hfold : BankFree (modes.foldl (fun acc m =>
        selectE (gt acc (modeWeightBoundSig m)) acc (modeWeightBoundSig m)) (lit 0)) := by
      refine pred_foldl _ _ _ (by simp [lit, BankFree]) ?_
      intro acc x hx hacc
      have h := hm x hx
      simp [selectE, gt, BankFree, hacc,
        bankFree_modeWeightBoundSig h.1 h.2.1 h.2.2]
    simp only [lit] at hfold
    exact ⟨by simp [LandExp.scale, ldexpE, sub, clampE, lit, BankFree, hfold],
      by simp [LandExp.shift, sub, clampE, lit, BankFree, hfold]⟩

-- ─────────────────────────────────────────────────────────────
-- The fold bridge: `List.range` index fold ≡ array element fold
-- ─────────────────────────────────────────────────────────────

private theorem range_foldl_eq_list_foldl {β} (l : List β) (f : Nat → Sig)
    (osc : β → Sig) (z : Sig)
    (h : ∀ (k : Nat) (hk : k < l.length), f k = osc l[k]) :
    (List.range l.length).foldl (fun acc k => add acc (f k)) z
      = l.foldl (fun acc m => add acc (osc m)) z := by
  induction l generalizing f z with
  | nil => simp
  | cons m l' ih =>
    have h0 : f 0 = osc m := h 0 (by simp)
    simp only [List.length_cons, List.range_succ_eq_map, List.foldl_cons,
      List.foldl_map, h0]
    exact ih (fun k => f (k + 1)) (add z (osc m))
      (fun k hk => by simpa using h (k + 1) (by simpa))

/-- The reference realization's index fold over `0 … n−1` IS the builder's
    element fold: same operands, same order, same left-nested `add` tree. -/
private theorem range_foldl_eq_array_foldl {β} (xs : Array β) (f : Nat → Sig)
    (osc : β → Sig) (z : Sig)
    (h : ∀ (k : Nat) (hk : k < xs.size), f k = osc xs[k]) :
    (List.range xs.size).foldl (fun acc k => add acc (f k)) z
      = xs.foldl (fun acc m => add acc (osc m)) z := by
  rw [← Array.foldl_toList, ← Array.length_toList]
  exact range_foldl_eq_list_foldl xs.toList f osc z (fun k hk => by
    simpa [Array.getElem_toList] using h k (by simpa using hk))

-- ─────────────────────────────────────────────────────────────
-- Distribution lemmas: instLoop / unrollBanks through the op algebra
-- ─────────────────────────────────────────────────────────────
-- Both walkers are homomorphic on every scalar smart constructor; stating
-- the fact per constructor (rather than unfolding constructors globally)
-- keeps every proof at the combinator level the builders are written in.

section Distribution

variable {id k : Nat} {a b c : Sig}

@[local simp] theorem instLoop_binary {t} :
    instLoop id k (.binary t a b) = .binary t (instLoop id k a) (instLoop id k b) := by
  simp [instLoop]
@[local simp] theorem instLoop_unary {t} :
    instLoop id k (.unary t a) = .unary t (instLoop id k a) := by simp [instLoop]
@[local simp] theorem instLoop_mul : instLoop id k (mul a b) = mul (instLoop id k a) (instLoop id k b) := by simp [mul]
@[local simp] theorem instLoop_add : instLoop id k (add a b) = add (instLoop id k a) (instLoop id k b) := by simp [add]
@[local simp] theorem instLoop_sub : instLoop id k (sub a b) = sub (instLoop id k a) (instLoop id k b) := by simp [sub]
@[local simp] theorem instLoop_div : instLoop id k (div a b) = div (instLoop id k a) (instLoop id k b) := by simp [div]
@[local simp] theorem instLoop_neg : instLoop id k (neg a) = neg (instLoop id k a) := by simp [neg]
@[local simp] theorem instLoop_toIntE : instLoop id k (toIntE a) = toIntE (instLoop id k a) := by simp [toIntE]
@[local simp] theorem instLoop_toFloatE : instLoop id k (toFloatE a) = toFloatE (instLoop id k a) := by simp [toFloatE]
@[local simp] theorem instLoop_rshift : instLoop id k (rshift a b) = rshift (instLoop id k a) (instLoop id k b) := by simp [rshift]
@[local simp] theorem instLoop_lshift : instLoop id k (lshift a b) = lshift (instLoop id k a) (instLoop id k b) := by simp [lshift]
@[local simp] theorem instLoop_bitAnd : instLoop id k (bitAnd a b) = bitAnd (instLoop id k a) (instLoop id k b) := by simp [bitAnd]
@[local simp] theorem instLoop_gt : instLoop id k (gt a b) = gt (instLoop id k a) (instLoop id k b) := by simp [gt]
@[local simp] theorem instLoop_roundE : instLoop id k (roundE a) = roundE (instLoop id k a) := by simp [roundE]
@[local simp] theorem instLoop_ldexpE : instLoop id k (ldexpE a b) = ldexpE (instLoop id k a) (instLoop id k b) := by simp [ldexpE]
@[local simp] theorem instLoop_clampE :
    instLoop id k (clampE a b c) = clampE (instLoop id k a) (instLoop id k b) (instLoop id k c) := by
  simp [clampE, instLoop]
@[local simp] theorem instLoop_selectE :
    instLoop id k (selectE a b c) = selectE (instLoop id k a) (instLoop id k b) (instLoop id k c) := by
  simp [selectE, instLoop]
@[local simp] theorem instLoop_lit {m : Int} {e : Nat} : instLoop id k (lit m e) = lit m e := by
  simp [lit, instLoop]
@[local simp] theorem instLoop_litI {m : Int} : instLoop id k (litI m) = litI m := by
  simp [litI, instLoop]
@[local simp] theorem instLoop_sampleRate : instLoop id k .sampleRate = .sampleRate := by
  simp [instLoop]
@[local simp] theorem instLoop_twoPiE : instLoop id k twoPiE = twoPiE := by
  simp [twoPiE]

@[local simp] theorem unrollBanks_binary {t} :
    unrollBanks (.binary t a b) = .binary t (unrollBanks a) (unrollBanks b) := by
  simp [unrollBanks]
@[local simp] theorem unrollBanks_unary {t} :
    unrollBanks (.unary t a) = .unary t (unrollBanks a) := by simp [unrollBanks]
@[local simp] theorem unrollBanks_mul : unrollBanks (mul a b) = mul (unrollBanks a) (unrollBanks b) := by simp [mul]
@[local simp] theorem unrollBanks_add : unrollBanks (add a b) = add (unrollBanks a) (unrollBanks b) := by simp [add]
@[local simp] theorem unrollBanks_sub : unrollBanks (sub a b) = sub (unrollBanks a) (unrollBanks b) := by simp [sub]
@[local simp] theorem unrollBanks_div : unrollBanks (div a b) = div (unrollBanks a) (unrollBanks b) := by simp [div]
@[local simp] theorem unrollBanks_neg : unrollBanks (neg a) = neg (unrollBanks a) := by simp [neg]
@[local simp] theorem unrollBanks_toIntE : unrollBanks (toIntE a) = toIntE (unrollBanks a) := by simp [toIntE]
@[local simp] theorem unrollBanks_toFloatE : unrollBanks (toFloatE a) = toFloatE (unrollBanks a) := by simp [toFloatE]
@[local simp] theorem unrollBanks_rshift : unrollBanks (rshift a b) = rshift (unrollBanks a) (unrollBanks b) := by simp [rshift]
@[local simp] theorem unrollBanks_lshift : unrollBanks (lshift a b) = lshift (unrollBanks a) (unrollBanks b) := by simp [lshift]
@[local simp] theorem unrollBanks_bitAnd : unrollBanks (bitAnd a b) = bitAnd (unrollBanks a) (unrollBanks b) := by simp [bitAnd]
@[local simp] theorem unrollBanks_gt : unrollBanks (gt a b) = gt (unrollBanks a) (unrollBanks b) := by simp [gt]
@[local simp] theorem unrollBanks_roundE : unrollBanks (roundE a) = roundE (unrollBanks a) := by simp [roundE]
@[local simp] theorem unrollBanks_ldexpE : unrollBanks (ldexpE a b) = ldexpE (unrollBanks a) (unrollBanks b) := by simp [ldexpE]
@[local simp] theorem unrollBanks_clampE :
    unrollBanks (clampE a b c) = clampE (unrollBanks a) (unrollBanks b) (unrollBanks c) := by
  simp [clampE, unrollBanks]
@[local simp] theorem unrollBanks_selectE :
    unrollBanks (selectE a b c) = selectE (unrollBanks a) (unrollBanks b) (unrollBanks c) := by
  simp [selectE, unrollBanks]
@[local simp] theorem unrollBanks_lit {m : Int} {e : Nat} : unrollBanks (lit m e) = lit m e := by
  simp [lit, unrollBanks]
@[local simp] theorem unrollBanks_litI {m : Int} : unrollBanks (litI m) = litI m := by
  simp [litI, unrollBanks]
@[local simp] theorem unrollBanks_sampleRate : unrollBanks .sampleRate = .sampleRate := by
  simp [unrollBanks]
@[local simp] theorem unrollBanks_twoPiE : unrollBanks twoPiE = twoPiE := by
  simp [twoPiE]
@[local simp] theorem unrollBanks_index :
    unrollBanks (.index a b) = .index (unrollBanks a) (unrollBanks b) := by
  simp [unrollBanks]
@[local simp] theorem unrollBanks_loopIdx {j : Nat} :
    unrollBanks (.loopIdx j) = .loopIdx j := by simp [unrollBanks]

/-- The reference realization of a STATIC bank: the left-nested `add` fold
    over the instantiated body, in array order. -/
theorem unrollBanks_bankSum {n : Nat} {ts : Array Sig} {b : Sig} {ii : Nat} :
    unrollBanks (.bankSum n ts b none ii)
      = (List.range n).foldl
          (fun acc j => add acc (instLoop ii j (unrollBanks b))) (litI 0) := by
  simp [unrollBanks, add]

-- Commutation through the numeric kernels: each is a fixed finite tree of
-- smart constructors, so the walkers pass through by the lemmas above.

@[local simp] theorem instLoop_relClockQ {x y : Sig} :
    instLoop id k (relClockQ x y) = relClockQ (instLoop id k x) (instLoop id k y) := by
  simp [relClockQ]
@[local simp] theorem unrollBanks_relClockQ {x y : Sig} :
    unrollBanks (relClockQ x y) = relClockQ (unrollBanks x) (unrollBanks y) := by
  simp [relClockQ]
@[local simp] theorem instLoop_modePhaseQFromIncr {x y : Sig} :
    instLoop id k (modePhaseQFromIncr x y)
      = modePhaseQFromIncr (instLoop id k x) (instLoop id k y) := by
  simp [modePhaseQFromIncr]
@[local simp] theorem unrollBanks_modePhaseQFromIncr {x y : Sig} :
    unrollBanks (modePhaseQFromIncr x y)
      = modePhaseQFromIncr (unrollBanks x) (unrollBanks y) := by
  simp [modePhaseQFromIncr]
@[local simp] theorem instLoop_expSig {x : Sig} :
    instLoop id k (expSig x) = expSig (instLoop id k x) := by
  simp [expSig]
@[local simp] theorem unrollBanks_expSig {x : Sig} :
    unrollBanks (expSig x) = expSig (unrollBanks x) := by
  simp [expSig]
@[local simp] theorem instLoop_fixedSinCycSig {x : Sig} :
    instLoop id k (fixedSinCycSig x) = fixedSinCycSig (instLoop id k x) := by
  simp [fixedSinCycSig]
@[local simp] theorem unrollBanks_fixedSinCycSig {x : Sig} :
    unrollBanks (fixedSinCycSig x) = fixedSinCycSig (unrollBanks x) := by
  simp [fixedSinCycSig]
@[local simp] theorem instLoop_fixedCosCycSig {x : Sig} :
    instLoop id k (fixedCosCycSig x) = fixedCosCycSig (instLoop id k x) := by
  simp [fixedCosCycSig]
@[local simp] theorem unrollBanks_fixedCosCycSig {x : Sig} :
    unrollBanks (fixedCosCycSig x) = fixedCosCycSig (unrollBanks x) := by
  simp [fixedCosCycSig]
@[local simp] theorem instLoop_fixedOutQ {n : Nat} {x : Sig} :
    instLoop id k (fixedOutQ n x) = fixedOutQ n (instLoop id k x) := by
  simp [fixedOutQ]
@[local simp] theorem unrollBanks_fixedOutQ {n : Nat} {x : Sig} :
    unrollBanks (fixedOutQ n x) = fixedOutQ n (unrollBanks x) := by
  simp [fixedOutQ]

/-- Baking the increment into a column and re-`toInt`ing it in the body IS
    the unrolled phase computation — definitionally. -/
theorem modePhaseQ_fromIncr {omega clkRel : Sig} :
    modePhaseQFromIncr
        (toIntE (div (mul (div omega twoPiE) (lit 4294967296)) .sampleRate)) clkRel
      = modePhaseQ omega clkRel := rfl

/-- Instantiating a column read: `index (arr (xs.map f)) (loopIdx id)` at
    concrete `k` is `f xs[k]` — provided the column's entries do not
    themselves read the binder. THE β-step of banks-as-data. -/
theorem instLoop_read_map {id k : Nat} {β} (xs : Array β) (f : β → Sig)
    (hf : ∀ x ∈ xs, LoopFree id (f x)) (hk : k < xs.size) :
    instLoop id k (.index (.arr (xs.map f)) (.loopIdx id)) = f xs[k] := by
  have hcol : ((xs.map f).attach.map fun ⟨x, _⟩ => instLoop id k x) = xs.map f :=
    attach_map_eq_of_id _ (fun x hx => by
      obtain ⟨m, hm', rfl⟩ := Array.mem_map.mp hx
      exact instLoop_of_loopFree (hf m hm'))
  simp [instLoop, readAt, hcol, Array.size_map, hk, Array.getElem_map]

end Distribution

-- ─────────────────────────────────────────────────────────────
-- WS-A: the builder's two branches agree
-- ─────────────────────────────────────────────────────────────

/-- Caller-supplied leaves must be FRESH for the bank's binder: no free
    read of the binder id, and no bank of their own smuggled into a leaf.
    Trivially true of every real call site (literals, param refs, clock
    expressions) — the hypotheses exist because `Sig` cannot say "closed
    term" by type. -/
def Fresh (id : Nat) (s : Sig) : Prop := LoopFree id s ∧ BankFree s

/-- **WS-A.** The banked lowering `modalBankSigTable` and the unrolled
    lowering `modalBankSig` are the SAME tree once the bank region is read
    through its reference realization: `unrollBanks` turns the `bankSum`
    into the left-nested add fold in array order, and the result is
    syntactically `modalBankSig`'s output. Carrier-free: syntactic equality
    is preserved by every denotation, so any interpretation of the ops —
    floats included — renders the two identically. Hypotheses: the bank is
    uniform (`deg = 0`, the table lowering's admission gate) and the
    caller-supplied leaves are fresh. -/
theorem unrollBanks_modalBankSigTable
    (modes : Array ModalMode) (clkInt anchor : Sig)
    (hu : bankIsUniform modes = true)
    (hclk : Fresh 0 clkInt) (hanc : Fresh 0 anchor)
    (hm : ∀ m ∈ modes,
      Fresh 0 m.omega ∧ Fresh 0 m.sigma ∧ Fresh 0 m.cre ∧ Fresh 0 m.cim) :
    unrollBanks (modalBankSigTable modes clkInt anchor none)
      = modalBankSig modes clkInt anchor := by
  -- repackaged freshness
  have hmL : ∀ m ∈ modes, LoopFree 0 m.sigma ∧ LoopFree 0 m.cre ∧ LoopFree 0 m.cim :=
    fun m h => ⟨(hm m h).2.1.1, (hm m h).2.2.1.1, (hm m h).2.2.2.1⟩
  have hmB : ∀ m ∈ modes, BankFree m.sigma ∧ BankFree m.cre ∧ BankFree m.cim :=
    fun m h => ⟨(hm m h).2.1.2, (hm m h).2.2.1.2, (hm m h).2.2.2.2⟩
  have hleL := loopFree_landExp (id := 0) hmL
  have hleB := bankFree_landExp hmB
  have hdeg : ∀ (i : Nat) (hi : i < modes.size), modes[i].deg = 0 := by
    unfold bankIsUniform at hu
    simpa [Array.all_eq_true] using hu
  -- identity facts for the walkers on the opaque leaves
  have huClk : unrollBanks clkInt = clkInt := unrollBanks_of_bankFree hclk.2
  have huAnc : unrollBanks anchor = anchor := unrollBanks_of_bankFree hanc.2
  have huScale : unrollBanks (bankLandExp modes).scale = (bankLandExp modes).scale :=
    unrollBanks_of_bankFree hleB.1
  have huShift : unrollBanks (bankLandExp modes).shift = (bankLandExp modes).shift :=
    unrollBanks_of_bankFree hleB.2
  -- the four coefficient columns are bank-free arrays: unrollBanks fixes them
  have hcolU : ∀ (f : ModalMode → Sig), (∀ m ∈ modes, BankFree (f m)) →
      unrollBanks (Sig.arr (modes.map f)) = Sig.arr (modes.map f) := by
    intro f hf
    refine unrollBanks_of_bankFree ?_
    simp only [BankFree]
    intro x hx
    obtain ⟨m, hm', rfl⟩ := Array.mem_map.mp hx
    exact hf m hm'
  have hcIncr := hcolU (fun m => div (mul (div m.omega twoPiE) (lit 4294967296)) .sampleRate)
    (fun m h => by simp [div, mul, lit, twoPiE, BankFree, (hm m h).1.2])
  have hcSigma := hcolU (fun m => m.sigma) (fun m h => (hm m h).2.1.2)
  have hcCre := hcolU (fun m => m.cre) (fun m h => (hm m h).2.2.1.2)
  have hcCim := hcolU (fun m => m.cim) (fun m h => (hm m h).2.2.2.2)
  -- Phase A: expose both builders, push `unrollBanks` to the leaves, erase it
  simp only [modalBankSigTable, modalBankSig, bankFold, bankCols,
    unrollBanks_selectE, unrollBanks_gt, unrollBanks_fixedOutQ, unrollBanks_lit,
    unrollBanks_relClockQ, unrollBanks_bankSum, huClk, huAnc]
  -- Phase A': the bank body under the fold — push the walker through it
  simp only [unrollBanks_rshift, unrollBanks_sub, unrollBanks_mul,
    unrollBanks_toIntE, unrollBanks_expSig, unrollBanks_neg, unrollBanks_div,
    unrollBanks_toFloatE, unrollBanks_modePhaseQFromIncr,
    unrollBanks_fixedSinCycSig, unrollBanks_fixedCosCycSig, unrollBanks_index,
    unrollBanks_loopIdx, unrollBanks_sampleRate, unrollBanks_lit,
    hcIncr, hcSigma, hcCre, hcCim, huScale, huShift, huClk, huAnc,
    unrollBanks_relClockQ]
  -- the two folds: index fold over `range modes.size` = element fold
  congr 1
  congr 1
  refine range_foldl_eq_array_foldl modes _ _ _ ?_
  intro j hj
  -- Phase B: instantiate the body at index j and compare with mode j's unroll
  have hiClk : instLoop 0 j clkInt = clkInt := instLoop_of_loopFree hclk.1
  have hiAnc : instLoop 0 j anchor = anchor := instLoop_of_loopFree hanc.1
  have hiScale : instLoop 0 j (bankLandExp modes).scale = (bankLandExp modes).scale :=
    instLoop_of_loopFree hleL.1
  have hiShift : instLoop 0 j (bankLandExp modes).shift = (bankLandExp modes).shift :=
    instLoop_of_loopFree hleL.2
  have hrIncr := instLoop_read_map (id := 0) (k := j) modes
    (fun m => div (mul (div m.omega twoPiE) (lit 4294967296)) .sampleRate)
    (fun m h => by simp [div, mul, lit, twoPiE, LoopFree, (hm m h).1.1]) hj
  have hrSigma := instLoop_read_map (id := 0) (k := j) modes (fun m => m.sigma)
    (fun m h => (hm m h).2.1.1) hj
  have hrCre := instLoop_read_map (id := 0) (k := j) modes (fun m => m.cre)
    (fun m h => (hm m h).2.2.1.1) hj
  have hrCim := instLoop_read_map (id := 0) (k := j) modes (fun m => m.cim)
    (fun m h => (hm m h).2.2.2.1) hj
  simp only [instLoop_rshift, instLoop_sub, instLoop_mul, instLoop_toIntE,
    instLoop_expSig, instLoop_neg, instLoop_div, instLoop_toFloatE,
    instLoop_modePhaseQFromIncr, instLoop_fixedSinCycSig,
    instLoop_fixedCosCycSig, instLoop_sampleRate, instLoop_lit,
    instLoop_relClockQ,
    hrIncr, hrSigma, hrCre, hrCim, hiScale, hiShift, hiClk, hiAnc,
    modePhaseQ_fromIncr, hdeg j hj]
  simp

-- ─────────────────────────────────────────────────────────────
-- The carrier-parametric corollaries (the handoff's vocabulary)
-- ─────────────────────────────────────────────────────────────

private theorem denote_range_foldl {α} (d : Sig → α) (op : α → α → α)
    (hadd : ∀ a b, d (add a b) = op (d a) (d b)) (f : Nat → Sig) (z0 : Sig) :
    ∀ n, d ((List.range n).foldl (fun acc j => add acc (f j)) z0)
      = refFold op (d z0) (fun j => d (f j)) n
  | 0 => by simp [refFold]
  | n + 1 => by
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [hadd, denote_range_foldl d op hadd f z0 n]
    rfl

/-- The reference realization of a STATIC bank, read through ANY map
    `d : Sig → α` that is a homomorphism on `add` alone (no other structure,
    no properties of `op` — not associativity, not commutativity, nothing):
    it is `refFold` — the fold over `0 … n−1` in array order. This is the
    handoff's `⟦bank⟧ op = refFold op z body n`, with the carrier fully
    abstract; instantiating `α := Float` with IEEE addition is just one
    reading, and needs no float model because the statement never looks
    inside `op`. -/
theorem denote_unrollBanks_bankSum {α} (d : Sig → α) (op : α → α → α) (z : α)
    (hadd : ∀ a b, d (add a b) = op (d a) (d b)) (hz : d (litI 0) = z)
    {n : Nat} {ts : Array Sig} {b : Sig} {ii : Nat} :
    d (unrollBanks (.bankSum n ts b none ii))
      = refFold op z (fun j => d (instLoop ii j (unrollBanks b))) n := by
  rw [unrollBanks_bankSum, denote_range_foldl d op hadd _ _ n, hz]

/-- **WS-A, denotationally.** Any denotation whatsoever agrees on the two
    builder branches once the banked one is read through its reference
    realization — immediate from the syntactic theorem, and the reason the
    syntactic theorem is the strong form: `d` never has to be specified. -/
theorem denote_modalBankSigTable_eq_modalBankSig {α} (d : Sig → α)
    (modes : Array ModalMode) (clkInt anchor : Sig)
    (hu : bankIsUniform modes = true)
    (hclk : Fresh 0 clkInt) (hanc : Fresh 0 anchor)
    (hm : ∀ m ∈ modes,
      Fresh 0 m.omega ∧ Fresh 0 m.sigma ∧ Fresh 0 m.cre ∧ Fresh 0 m.cim) :
    d (unrollBanks (modalBankSigTable modes clkInt anchor none))
      = d (modalBankSig modes clkInt anchor) :=
  congrArg d (unrollBanks_modalBankSigTable modes clkInt anchor hu hclk hanc hm)

-- ─────────────────────────────────────────────────────────────
-- WS-C: nested banks compose (fold-of-folds, WS5's unique binder ids)
-- ─────────────────────────────────────────────────────────────

/-- Instantiating a binder DISTRIBUTES through an unrolled fold: the fold
    structure (count, order, the left-nested `add` spine, the zero) is
    untouched; only the per-iteration body is instantiated. This is the
    load-bearing half of WS-C: an outer bank's index reaches an inner
    bank's unrolled body without disturbing the inner fold's shape. -/
theorem instLoop_range_foldl {id k : Nat} (f : Nat → Sig) (n : Nat) :
    instLoop id k ((List.range n).foldl (fun acc j => add acc (f j)) (litI 0))
      = (List.range n).foldl (fun acc j => add acc (instLoop id k (f j))) (litI 0) := by
  suffices h : ∀ (z : Sig),
      instLoop id k ((List.range n).foldl (fun acc j => add acc (f j)) z)
        = (List.range n).foldl (fun acc j => add acc (instLoop id k (f j)))
            (instLoop id k z) by
    simpa [instLoop_litI] using h (litI 0)
  induction n with
  | zero => intro z; simp
  | succ n ih =>
    intro z
    rw [List.range_succ, List.foldl_append, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [instLoop_add, ih]

/-- **WS-C.** A nested bank (fold-of-folds, WS5) unrolls to the ROW-MAJOR
    double fold: for each outer index `j` in array order, the ENTIRE inner
    fold in array order — the inner region's fold order is manifestly
    independent of the outer's index, because the outer instantiation
    distributes through the inner fold's spine and touches only the body.

    The unique-binder-id discipline is exactly what makes the statement
    meaningful: the inner bank unrolls FIRST (consuming every free
    `loopIdx i₂` in its own body — including reads baked into its column
    entries by the outer builder), and the outer index `i₁` is instantiated
    through the result. Were `i₁ = i₂`, the inner instantiation would
    capture the outer's reads — the failure WS5 removed by correcting
    de Bruijn away to chain-unique ids. -/
theorem unrollBanks_bankSum_nested {c c' : Nat} {ts ts' : Array Sig}
    {b : Sig} {i₁ i₂ : Nat} :
    unrollBanks (.bankSum c ts (.bankSum c' ts' b none i₂) none i₁)
      = (List.range c).foldl (fun acc j => add acc
          ((List.range c').foldl (fun acc' i => add acc'
            (instLoop i₁ j (instLoop i₂ i (unrollBanks b)))) (litI 0)))
        (litI 0) := by
  rw [unrollBanks_bankSum]
  have hstep : (fun (acc : Sig) (j : Nat) =>
        add acc (instLoop i₁ j (unrollBanks (Sig.bankSum c' ts' b none i₂))))
      = (fun acc j => add acc ((List.range c').foldl (fun acc' i => add acc'
          (instLoop i₁ j (instLoop i₂ i (unrollBanks b)))) (litI 0))) := by
    funext acc j
    rw [unrollBanks_bankSum, instLoop_range_foldl]
  rw [hstep]

/-- **WS-C, denotationally.** Read through any `add`-homomorphism, a nested
    bank IS the double `refFold`, row-major — the composable form: the
    inner fold is a closed sub-computation re-run per outer index, operands
    meeting in the same order at both depths. -/
theorem denote_unrollBanks_bankSum_nested {α} (d : Sig → α) (op : α → α → α)
    (z : α) (hadd : ∀ a b, d (add a b) = op (d a) (d b)) (hz : d (litI 0) = z)
    {c c' : Nat} {ts ts' : Array Sig} {b : Sig} {i₁ i₂ : Nat} :
    d (unrollBanks (.bankSum c ts (.bankSum c' ts' b none i₂) none i₁))
      = refFold op z (fun j => refFold op z
          (fun i => d (instLoop i₁ j (instLoop i₂ i (unrollBanks b)))) c') c := by
  rw [unrollBanks_bankSum_nested,
    denote_range_foldl d op hadd _ _ c, hz]
  congr 1
  funext j
  rw [denote_range_foldl d op hadd _ _ c', hz]

end Tropical.EmitArrow
