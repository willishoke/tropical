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

Bilateral (slice Phase 4): a room's past arm enters as the MIRRORED pole
`z' = −ν` with the two-sided transform's sign, so both arms are one rational
function; clusters never cross orientation; a degree-`p` mode is `p+1` copies
of its node. `BlockRow`s are realized by `BlockRealize.lean`.
-/

namespace Tropical.EmitArrow.Block

open Tropical.Ir
open Tropical.Exact (DyadicI)

/-- Why a spine is not served by the block terminal. Every constructor names
    the numerical or structural fact, so the refusal an agent sees says what
    to change. -/
inductive Refusal where
  /-- a cluster of more than `cap` nodes with at least two DISTINCT (by
      expression identity) members — the realizable body size is capped. A
      cluster whose nodes are all one expression is exempt at any size: it
      realizes as polynomial degree (`exp[z,…,z] = dᵏ⁻¹/(k−1)!·e^{zd}`). -/
  | clusterTooLarge (size cap : Nat)
deriving Repr, BEq, Inhabited

def Refusal.describe : Refusal → String
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

-- ── The pole multiset of a bilateral tree ─────────────────────────────────────

/-- Is this enclosure certifiably the exact number zero? -/
def isExactZero (d : DyadicI) : Bool :=
  d.isExact && Dyadic.ble d.lo 0 && Dyadic.ble 0 d.lo

/-- Is this enclosure certifiably the exact number one? -/
def isExactOne (d : DyadicI) : Bool :=
  d.isExact && Dyadic.ble d.lo 1 && Dyadic.ble 1 d.lo

/-- One factor of a proper kernel in the s-plane: a pole of multiplicity
    `mult` (= `deg + 1`) with the numerator coefficient of `coeff/(s−z)^mult`,
    tagged with the arm it came from. A FUTURE mode `A·d^p·e^{zd}` is
    `A·p!/(s−z)^{p+1}`; a PAST mode `B·(−d)^p·e^{ν(−d)}` on `d<0` is
    `(−1)^{p+1}·B·p!/(s−z')^{p+1}` at the MIRRORED pole `z' = −ν` — the
    two-sided Laplace transform's sign, which is what makes one rational
    function of both arms. -/
structure PoleFactor where
  mode : ModalMode
  orientation : Oriented.Orientation
  /-- the s-plane node: `poleE` of the future mode, its negation for the past -/
  node : CplxE
  coeff : CplxE
  mult : Nat

/-- Which arms a room's direction leaves alive: exactly `0` ⇒ future only,
    exactly `1` ⇒ past only, anything else (a live knob, a fraction) ⇒ both,
    scaled `(1−δ)` / `δ` as `Bank.kernel` does. -/
private def armsOf (constants : Array (Option DyadicI)) (direction : Sig) :
    Bool × Bool :=
  match sigConstDFrom? constants direction with
  | some d => if isExactZero d then (true, false)
              else if isExactOne d then (false, true)
              else (true, true)
  | none => (true, true)

private def natCoeff (n : Nat) : BuildM CplxE := Oriented.natE n

/-- The s-plane factors of one proper kernel, in mode order, future arm then
    past arm per mode. -/
private def properFactors (constants : Array (Option DyadicI)) :
    ModalProperKernel → BuildM (Array PoleFactor)
  | .causalTail tail => do
      let node ← tail.poleE
      let coeff ← cmulE tail.ampE (← natCoeff (Oriented.factorial tail.deg))
      pure #[{ mode := tail, orientation := .future, node, coeff, mult := tail.deg + 1 }]
  | .oriented modes direction => do
      let (futureArm, pastArm) := armsOf constants direction
      let scaledArms := !(futureArm && !pastArm) && !(pastArm && !futureArm)
      let one ← lit 1
      let forward ← sub one direction
      let mut out : Array PoleFactor := #[]
      for m in modes do
        let base ← cmulE m.ampE (← natCoeff (Oriented.factorial m.deg))
        if futureArm then
          let coeff ← if scaledArms then do
              pure (← mul forward base.1, ← mul forward base.2)
            else pure base
          out := out.push { mode := m, orientation := .future, node := ← m.poleE,
                            coeff, mult := m.deg + 1 }
        if pastArm then
          let scaled ← if scaledArms then do
              pure (← mul direction base.1, ← mul direction base.2)
            else pure base
          -- `(−1)^{p+1}·B·p!`
          let coeff ← if (m.deg + 1) % 2 == 1 then cnegE scaled else pure scaled
          let mirrored : ModalMode := { m with
            sigma := ← neg m.sigma, omega := ← neg m.omega,
            sigmaRange := m.sigmaRange.map fun (lo, hi) => (-hi, -lo) }
          out := out.push { mode := mirrored, orientation := .past, node := ← mirrored.poleE,
                            coeff, mult := m.deg + 1 }
      pure out

/-- Every s-plane factor of the tree, in traversal order — the GLOBAL index
    used for cluster membership below is the position in the EXPANDED list
    (a factor of multiplicity `m` occupies `m` consecutive positions). -/
def poleFactors (constants : Array (Option DyadicI)) :
    ModalKernelExpr → BuildM (Array PoleFactor)
  | .identity => pure #[]
  | .proper kernel => properFactors constants kernel
  | .scale _ kernel => poleFactors constants kernel
  | .parallel kernels =>
      kernels.attach.foldlM (fun acc kernel => do
        pure (acc ++ (← poleFactors constants kernel.1))) #[]
  | .cascade kernels =>
      kernels.attach.foldlM (fun acc kernel => do
        pure (acc ++ (← poleFactors constants kernel.1))) #[]
  | .blend _ dry wet => do
      pure ((← poleFactors constants dry) ++ (← poleFactors constants wet))
termination_by kernel => sizeOf kernel
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem kernel.2; simp_all; omega)

/-- The expanded node list: each factor repeated `mult` times, with its
    orientation — the multiset the clusters partition. -/
def expandNodes (factors : Array PoleFactor) : Array (CplxE × Oriented.Orientation × ModalMode) :=
  factors.foldl (fun acc f =>
    acc ++ Array.replicate f.mult (f.node, f.orientation, f.mode)) #[]

-- ── Clustering ────────────────────────────────────────────────────────────────

/-- Union-find over the expanded pole multiset under the shared pole-distance
    lens: two poles of the SAME orientation join a cluster when their min |Δ|
    over the declared σ intervals is certifiably below θ_acc, or when they are
    one expression (a factor's own copies, or two spellings that interned to
    one node). Opposite orientations never cluster — a future pole and a past
    pole are separated by at least `σ_f + σ_p` and are different signals. An
    unclassifiable pair (a live σ without a declared range) never joins — the
    pairwise router's `cold` convention. Each cluster lists global indices in
    traversal order. -/
def clusterPoles (constants : Array (Option DyadicI))
    (nodes : Array (CplxE × Oriented.Orientation × ModalMode)) : Array (Array Nat) := Id.run do
  let n := nodes.size
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
      let hot := match nodes[i]?, nodes[j]? with
        | some (zi, oi, mi), some (zj, oj, mj) =>
            oi == oj && ((zi.1 == zj.1 && zi.2 == zj.2)
              || poleAccuracyHotFrom? constants mi mj == some true)
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
    product of `(s − z)` over the cluster nodes (with multiplicity) that live
    in the subtree. Returns the table, those cluster nodes (so a parallel node
    can multiply each branch by the OTHER branches' cluster factors), and the
    global-index cursor after the subtree. `factors` is the tree's factor list
    in traversal order; the cursor walks it one factor at a time, its global
    index advancing by the factor's multiplicity. -/
private def leibniz (constants : Array (Option DyadicI)) (nodes : Array CplxE)
    (member : Nat → Bool) (cursor : Nat) :
    ModalKernelExpr → BuildM (DDTable × Array CplxE × Nat)
  | .identity => do pure (← DDTable.one nodes.size, #[], cursor)
  | .proper kernel => do
      let factors ← properFactors constants kernel
      let k := nodes.size
      -- global index of each factor's first copy
      let mut starts : Array Nat := #[]
      let mut g := cursor
      for f in factors do
        starts := starts.push g
        g := g + f.mult
      let inCluster := fun (i : Nat) => member starts[i]!
      let mut clusterPolesHere : Array CplxE := #[]
      for (f, i) in factors.zipIdx do
        if inCluster i then
          clusterPolesHere := clusterPolesHere ++ Array.replicate f.mult f.node
      let mut total ← DDTable.zero k
      for (f, i) in factors.zipIdx do
        -- term_ν = coeff_ν · ∏_{ν'∈c∩stage, ν'≠ν}(s − z_ν')^{mult} · [(s − z_ν)^{−mult} if ν ∉ c]
        let mut term ← DDTable.const k f.coeff
        for (f', j) in factors.zipIdx do
          if j != i && inCluster j then
            for _ in [0:f'.mult] do
              term ← term.mul (← DDTable.linear nodes f'.node)
        if !(inCluster i) then
          let inverse ← DDTable.inverse nodes f.node
          for _ in [0:f.mult] do
            term ← term.mul inverse
        total ← total.add term
      pure (total, clusterPolesHere, g)
  | .scale value kernel => do
      let (table, poles, cursor) ← leibniz constants nodes member cursor kernel
      pure (← table.scaleReal value, poles, cursor)
  | .cascade kernels => do
      let one ← DDTable.one nodes.size
      kernels.attach.foldlM (fun (state : DDTable × Array CplxE × Nat) kernel => do
        let (table, poles, cursor) := state
        let (t, p, c) ← leibniz constants nodes member cursor kernel.1
        pure (← table.mul t, poles ++ p, c)) (one, #[], cursor)
  | .parallel kernels => do
      -- Σ_a T_a · ∏_{b≠a} D_{c,b}: each branch carries the other branches'
      -- cluster factors so every summand is (branch · D_{c,node}).
      let (branches, cursor) ← kernels.attach.foldlM
        (fun (state : Array (DDTable × Array CplxE) × Nat) kernel => do
          let (t, p, c) ← leibniz constants nodes member state.2 kernel.1
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
      let (dt, dp, cursor) ← leibniz constants nodes member cursor dry
      let (wt, wp, cursor) ← leibniz constants nodes member cursor wet
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

/-- One block: ordered cluster nodes `z₁..z_k` in the s-plane and the Newton
    coefficients `n₁..n_k` of `Σ_m n_m · exp[z_m..z_k](d)`. A FUTURE row is the
    causal signal on `d > 0`. A PAST row's nodes are mirrored poles `z' = −ν`
    and its signal is the anti-causal `−Σ_m n_m·exp[z'_m..z'_k](d)` on `d < 0`,
    which on the mirrored clock `d' = −d` is `Σ_m (−1)^{k−m+1} n_m ·
    exp[ν_m..ν_k](d')` at the physical poles — the realizer applies that
    reflection. `indices` are the nodes' global positions in the expanded
    multiset. -/
structure BlockRow where
  nodes : Array CplxE
  coeffs : Array CplxE
  indices : Array Nat
  orientation : Oriented.Orientation

/-- Are all of a row's nodes one expression (so the row is a confluent pole of
    multiplicity `k` and realizes as degree, never as a divided difference)? -/
def BlockRow.confluent (row : BlockRow) : Bool := distinctNodeCount row.nodes ≤ 1

/-- Block partial fractions of a bilateral retained tree: expand the pole
    multiset (arms by direction, copies by degree), cluster it, then run the
    Leibniz traversal once per cluster. Refuses, with the reason, a cluster
    whose distinct node count exceeds `cap`. -/
def decompose (spine : ModalKernelExpr) (cap : Nat := defaultClusterCap) :
    BuildM (Except Refusal (Array BlockRow)) := do
  let factors ← poleFactors (sigConstTable (← get).exprs) spine
  -- the constant table is taken AFTER expansion: a past arm's mirrored pole is
  -- a fresh `neg` node, and the lens must be able to fold it
  let constants := sigConstTable (← get).exprs
  let expanded := expandNodes factors
  let clusters := clusterPoles constants expanded
  let mut rows : Array BlockRow := #[]
  for indices in clusters do
    let entries := indices.filterMap (expanded[·]?)
    let nodes := entries.map (·.1)
    let some (_, orientation, _) := entries[0]? | continue
    if distinctNodeCount nodes > 1 && nodes.size > cap then
      return .error (.clusterTooLarge nodes.size cap)
    let member := fun (g : Nat) => indices.contains g
    let (table, _, _) ← leibniz constants nodes member 0 spine
    rows := rows.push { nodes, coeffs := table.newton, indices, orientation }
  pure (.ok rows)

end Tropical.EmitArrow.Block
