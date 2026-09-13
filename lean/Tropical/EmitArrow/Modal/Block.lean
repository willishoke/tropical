import Tropical.EmitArrow.Modal.Kernel

/-!
# EmitArrow.Modal.Block — block partial fractions over the retained kernel

The composable modal carrier is the retained factor tree (`ModalKernelExpr`):
cascade, parallel, scale, blend, proper kernels, all kept as authored. This
module is the terminal that consumes that tree WITHOUT collecting it first.

Mathematics (design/block-carrier-research.local.md §2, slice plan §2). For a
causal spine `H(s) = ∏ᵢ Hᵢ(s)`, `Hᵢ = Σ_ν r_ν/(s − z_ν)`, the pole multiset is
the union over the tree and never moves. Group it into CLUSTERS by the same
pole-distance lens the pairwise router uses (θ_acc, over declared σ
intervals). For a cluster `c` with ordered nodes `z₁..z_k`, the block's
contribution to the impulse response is

    h_c(d) = Σ_{m=1}^{k} n_m · exp[z_m..z_k](d),

the suffix divided differences of `e^{zd}`, with Newton coefficients
`n_m = g_c[z₁..z_m]` — the divided differences at the cluster nodes of
`g_c = H · D_c`, `D_c = ∏_{z∈c}(s − z)`. Those are computed by the Leibniz
rule for divided differences over the factors of the tree, using only

    (s − a)            : [i,i] = zᵢ − a, [i,i+1] = 1, else 0        (division-free)
    1/(s − a), a ∉ c   : [i,j] = (−1)^{j−i} / ∏_{l=i..j}(z_l − a)  (cross-cluster gap only)
    (fg)[i,j]          = Σ_{l=i..j} f[i,l]·g[l,j]

so no `1/(zᵢ − zⱼ)` with both nodes in one cluster is ever formed, exact
coincidence needs no branch, and a singleton cluster reduces to the ordinary
residue `r_z · ∏_{other factors} Hᵢ(z)` (a product of sums rather than the
collected fold's sum of products — the same value, better conditioned).
Validated in f64 at 1e-14 to 1e-16 across every gap including 0
(`demos/block_carrier_leibniz.py`).

Every coefficient is a `CplxE` (`Sig × Sig`): with literal poles it
const-folds; with a live pole it rides the stage-0 coefficient kernel, exactly
like the collected residues today. `DyadicI` is consulted only to DECIDE
(cluster membership, causality), never to compute a coefficient.

This module is Phase 1 of the slice: the algebra and the decision. It emits
`BlockRow`s; their realization (size-1 = today's mode, size-2 = a `PairedMode`
plus a mode, size-3 = the nested divided-difference body) is Phase 2.
-/

namespace Tropical.EmitArrow.Block

open Tropical.Ir
open Tropical.Exact (DyadicI)

/-- Why a spine is not served by the block terminal. Every constructor names
    the numerical or structural fact, so the refusal an agent sees says what
    to change. -/
inductive Refusal where
  /-- a proper kernel whose direction does not fold to exactly `0` — the
      bilateral extension (slice plan Phase 4) is not in this module. -/
  | nonCausal
  /-- a mode of positive degree in the input tree — the algebra here is stated
      for simple-pole factors; higher-degree inputs stay on the exact
      `Oriented` path. -/
  | higherDegree
  /-- a cluster of more than `cap` nodes with at least two DISTINCT (by
      expression identity) members — the realizable body size is capped. A
      cluster whose nodes are all one expression is exempt at any size: it
      realizes as polynomial degree (`exp[z,…,z] = dᵏ⁻¹/(k−1)!·e^{zd}`). -/
  | clusterTooLarge (size cap : Nat)
deriving Repr, BEq, Inhabited

def Refusal.describe : Refusal → String
  | .nonCausal => "block carrier: a room's direction is not exactly forward (bilateral chains are not yet served by the block terminal)"
  | .higherDegree => "block carrier: an input mode has positive degree (only simple-pole factors are served)"
  | .clusterTooLarge size cap =>
      s!"block carrier: a cluster of {size} runtime-near-equal poles (not all one expression) exceeds the served body size {cap}"

/-- The default cap on distinct nodes per cluster — the largest body Phase 2
    realizes (size 1 = a mode, 2 = a paired mode, 3 = the nested body). -/
def defaultClusterCap : Nat := 3

-- ── Divided-difference tables ─────────────────────────────────────────────────

/-- The divided differences `f[zᵢ..zⱼ]`, `i ≤ j`, of one rational factor at an
    ordered node list, as a row-major `size × size` array of `CplxE` (entries
    below the diagonal are unused zeros). -/
structure DDTable where
  size : Nat
  entries : Array CplxE

namespace DDTable

def get (t : DDTable) (i j : Nat) : CplxE :=
  (t.entries[i * t.size + j]?).getD (⟨0⟩, ⟨0⟩)

/-- Build a table from an entry function (only `i ≤ j` is consulted). -/
def build (size : Nat) (entry : Nat → Nat → BuildM CplxE) : BuildM DDTable := do
  let zero ← Oriented.natE 0
  let mut entries : Array CplxE := #[]
  for i in [0:size] do
    for j in [0:size] do
      entries := entries.push (← if i ≤ j then entry i j else pure zero)
  pure { size, entries }

/-- The constant `c`: `c` on the diagonal, zero elsewhere. -/
def const (size : Nat) (c : CplxE) : BuildM DDTable := do
  let zero ← Oriented.natE 0
  build size fun i j => pure (if i == j then c else zero)

def one (size : Nat) : BuildM DDTable := do const size (← Oriented.natE 1)
def zero (size : Nat) : BuildM DDTable := do const size (← Oriented.natE 0)

/-- The linear factor `(s − a)`: `[i,i] = zᵢ − a`, `[i,i+1] = 1`, else `0`.
    Division-free — the factor that cancels a cluster's own pole. -/
def linear (nodes : Array CplxE) (a : CplxE) : BuildM DDTable := do
  let zero ← Oriented.natE 0
  let unit ← Oriented.natE 1
  build nodes.size fun i j =>
    if i == j then csubE nodes[i]! a
    else if j == i + 1 then pure unit
    else pure zero

/-- The factor `1/(s − a)` for a pole `a` OUTSIDE the cluster:
    `[i,j] = (−1)^{j−i} / ∏_{l=i..j}(z_l − a)`. The only division in the
    algebra, and every divisor is a cross-cluster gap. -/
def inverse (nodes : Array CplxE) (a : CplxE) : BuildM DDTable :=
  build nodes.size fun i j => do
    let unit ← Oriented.natE 1
    let mut denominator := unit
    for l in [i:j+1] do
      denominator ← cmulE denominator (← csubE nodes[l]! a)
    let quotient ← cdivE unit denominator
    if (j - i) % 2 == 1 then cnegE quotient else pure quotient

def add (a b : DDTable) : BuildM DDTable :=
  build a.size fun i j => caddE (a.get i j) (b.get i j)

/-- The Leibniz rule: `(fg)[i,j] = Σ_{l=i..j} f[i,l]·g[l,j]`. -/
def mul (a b : DDTable) : BuildM DDTable :=
  build a.size fun i j => do
    let mut total ← Oriented.natE 0
    for l in [i:j+1] do
      total ← caddE total (← cmulE (a.get i l) (b.get l j))
    pure total

/-- Scale every entry by a real `Sig`. -/
def scaleReal (s : Sig) (t : DDTable) : BuildM DDTable :=
  build t.size fun i j => do
    let e := t.get i j
    pure (← EmitArrow.mul s e.1, ← EmitArrow.mul s e.2)

/-- The Newton coefficients `f[z₁..z_m]`, `m = 1..size` — the first row. -/
def newton (t : DDTable) : Array CplxE :=
  (Array.range t.size).map fun j => t.get 0 j

end DDTable

-- ── The pole multiset of a causal tree ────────────────────────────────────────

/-- Is this enclosure certifiably the exact number zero? -/
def isExactZero (d : DyadicI) : Bool :=
  d.isExact && Dyadic.ble d.lo 0 && Dyadic.ble 0 d.lo

/-- The modes of a proper kernel with its direction folded to exactly forward.
    A `causalTail` is causal by construction. -/
private def causalModes? (constants : Array (Option DyadicI)) :
    ModalProperKernel → Except Refusal (Array ModalMode)
  | .causalTail tail => pure #[tail]
  | .oriented modes direction =>
      match sigConstDFrom? constants direction with
      | some d => if isExactZero d then pure modes else throw .nonCausal
      | none => throw .nonCausal

/-- Every simple pole of the tree, in traversal order (the GLOBAL index used
    for cluster membership below), with its degree and causality checked. -/
def poleMultiset (constants : Array (Option DyadicI)) :
    ModalKernelExpr → Except Refusal (Array ModalMode)
  | .identity => pure #[]
  | .proper kernel => do
      let modes ← causalModes? constants kernel
      if modes.any (·.deg != 0) then throw .higherDegree
      pure modes
  | .scale _ kernel => poleMultiset constants kernel
  | .parallel kernels => do
      kernels.attach.foldlM (fun acc kernel => do
        pure (acc ++ (← poleMultiset constants kernel.1))) #[]
  | .cascade kernels => do
      kernels.attach.foldlM (fun acc kernel => do
        pure (acc ++ (← poleMultiset constants kernel.1))) #[]
  | .blend _ dry wet => do
      pure ((← poleMultiset constants dry) ++ (← poleMultiset constants wet))
termination_by kernel => sizeOf kernel
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem kernel.2; simp_all; omega)

-- ── Clustering ────────────────────────────────────────────────────────────────

/-- Union-find over the pole multiset under the shared pole-distance lens:
    two poles join a cluster when their min |Δ| over the declared σ intervals
    is certifiably below θ_acc. An unclassifiable pair (a live σ without a
    declared range, a live ω) never joins — the same convention as the
    pairwise router's `cold`. Each cluster lists its global indices in
    traversal order. -/
def clusterPoles (constants : Array (Option DyadicI)) (modes : Array ModalMode) :
    Array (Array Nat) := Id.run do
  let n := modes.size
  let mut parent : Array Nat := Array.range n
  let find := fun (parent : Array Nat) (i : Nat) => Id.run do
    let mut k := i
    for _ in [0:n] do
      let p := parent[k]!
      if p == k then break
      k := p
    return k
  for i in [0:n] do
    for j in [i+1:n] do
      let hot := match modes[i]?, modes[j]? with
        | some a, some b => poleAccuracyHotFrom? constants a b == some true
        | _, _ => false
      if hot then
        let ri := find parent i
        let rj := find parent j
        if ri != rj then parent := parent.set! ri rj
  let mut groups : Array (Array Nat) := #[]
  let mut root : Array Nat := #[]
  for i in [0:n] do
    let r := find parent i
    match root.findIdx? (· == r) with
    | some g => groups := groups.set! g (groups[g]!.push i)
    | none =>
        root := root.push r
        groups := groups.push #[i]
  return groups

/-- The number of DISTINCT nodes in a cluster by expression identity — nodes
    built from the same hash-consed pole expressions are one node with
    multiplicity, and realize as polynomial degree. -/
def distinctNodeCount (nodes : Array CplxE) : Nat := Id.run do
  let mut seen : Array CplxE := #[]
  for z in nodes do
    if !(seen.any (fun w => w.1 == z.1 && w.2 == z.2)) then seen := seen.push z
  return seen.size

-- ── The Leibniz traversal ─────────────────────────────────────────────────────

/-- The divided-difference table, at the cluster nodes, of `H_sub · D_{c,sub}`
    where `H_sub` is the subtree's transfer function and `D_{c,sub}` the
    product of `(s − z)` over the cluster nodes that live in the subtree.
    Returns the table, the cluster poles found in the subtree (so a parallel
    node can multiply each branch by the OTHER branches' cluster factors), and
    the global-index cursor after the subtree. -/
private def leibniz (nodes : Array CplxE) (member : Nat → Bool)
    (cursor : Nat) : ModalKernelExpr → BuildM (DDTable × Array CplxE × Nat)
  | .identity => do pure (← DDTable.one nodes.size, #[], cursor)
  | .proper kernel => do
      let modes := match kernel with
        | .oriented modes _ => modes
        | .causalTail tail => #[tail]
      let k := nodes.size
      let mut total ← DDTable.zero k
      let mut clusterPolesHere : Array CplxE := #[]
      for (m, i) in modes.zipIdx do
        if member (cursor + i) then clusterPolesHere := clusterPolesHere.push (← m.poleE)
      for (m, i) in modes.zipIdx do
        let pole ← m.poleE
        -- term_ν = r_ν · ∏_{ν'∈c∩stage, ν'≠ν}(s − z_ν') · [1/(s − z_ν) if ν ∉ c]
        let mut term ← DDTable.const k m.ampE
        for (m', j) in modes.zipIdx do
          if j != i && member (cursor + j) then
            term ← term.mul (← DDTable.linear nodes (← m'.poleE))
        if !(member (cursor + i)) then
          term ← term.mul (← DDTable.inverse nodes pole)
        total ← total.add term
      pure (total, clusterPolesHere, cursor + modes.size)
  | .scale value kernel => do
      let (table, poles, cursor) ← leibniz nodes member cursor kernel
      pure (← table.scaleReal value, poles, cursor)
  | .cascade kernels => do
      let one ← DDTable.one nodes.size
      kernels.attach.foldlM (fun (state : DDTable × Array CplxE × Nat) kernel => do
        let (table, poles, cursor) := state
        let (t, p, c) ← leibniz nodes member cursor kernel.1
        pure (← table.mul t, poles ++ p, c)) (one, #[], cursor)
  | .parallel kernels => do
      -- Σ_a T_a · ∏_{b≠a} D_{c,b}: each branch carries the other branches'
      -- cluster factors so every summand is (branch · D_{c,node}).
      let (branches, cursor) ← kernels.attach.foldlM
        (fun (state : Array (DDTable × Array CplxE) × Nat) kernel => do
          let (t, p, c) ← leibniz nodes member state.2 kernel.1
          pure (state.1.push (t, p), c)) (#[], cursor)
      let all := branches.foldl (fun acc (_, p) => acc ++ p) #[]
      let mut total ← DDTable.zero nodes.size
      for (branch, a) in branches.zipIdx do
        let mut term := branch.1
        for (other, b) in branches.zipIdx do
          if a != b then
            for z in other.2 do
              term ← term.mul (← DDTable.linear nodes z)
        total ← total.add term
      pure (total, all, cursor)
  | .blend mix dry wet => do
      let one ← lit 1
      let dryWeight ← sub one mix
      let (dt, dp, cursor) ← leibniz nodes member cursor dry
      let (wt, wp, cursor) ← leibniz nodes member cursor wet
      let mut dryTerm ← dt.scaleReal dryWeight
      for z in wp do dryTerm ← dryTerm.mul (← DDTable.linear nodes z)
      let mut wetTerm ← wt.scaleReal mix
      for z in dp do wetTerm ← wetTerm.mul (← DDTable.linear nodes z)
      pure (← dryTerm.add wetTerm, dp ++ wp, cursor)
termination_by kernel => sizeOf kernel
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem kernel.2; simp_all; omega)

-- ── The decomposition ─────────────────────────────────────────────────────────

/-- One block: ordered cluster nodes `z₁..z_k` (pole form `(−σ, ω)`) and the
    Newton coefficients `n₁..n_k` of `h(d) = Σ_m n_m · exp[z_m..z_k](d)`.
    `indices` are the nodes' global positions in the tree's pole multiset. -/
structure BlockRow where
  nodes : Array CplxE
  coeffs : Array CplxE
  indices : Array Nat

/-- Are all of a row's nodes one expression (so the row is a confluent pole of
    multiplicity `k` and realizes as degree, never as a divided difference)? -/
def BlockRow.confluent (row : BlockRow) : Bool := distinctNodeCount row.nodes ≤ 1

/-- Block partial fractions of a causal retained tree: cluster the pole
    multiset, then run the Leibniz traversal once per cluster. Refuses, with
    the reason, a non-causal or higher-degree input or a cluster whose distinct
    node count exceeds `cap`. -/
def decompose (spine : ModalKernelExpr) (cap : Nat := defaultClusterCap) :
    BuildM (Except Refusal (Array BlockRow)) := do
  let builder ← get
  let constants := sigConstTable builder.exprs
  match poleMultiset constants spine with
  | .error refusal => pure (.error refusal)
  | .ok modes =>
    let clusters := clusterPoles constants modes
    let mut rows : Array BlockRow := #[]
    for indices in clusters do
      let nodes ← (indices.filterMap (modes[·]?)).mapM ModalMode.poleE
      if distinctNodeCount nodes > 1 && nodes.size > cap then
        return .error (.clusterTooLarge nodes.size cap)
      let member := fun (g : Nat) => indices.contains g
      let (table, _, _) ← leibniz nodes member 0 spine
      rows := rows.push { nodes, coeffs := table.newton, indices }
    pure (.ok rows)

end Tropical.EmitArrow.Block
