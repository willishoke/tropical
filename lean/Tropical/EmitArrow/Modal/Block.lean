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

/-- The largest divided-difference body realized (size 1 = a mode, 2 = a
    paired mode, 3 = the nested body). A cluster with more value classes than
    this splits into its classes at the collected floor. -/
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

/-- `c/(s − a)` for a pole `a` OUTSIDE the cluster: `[i,j] = (−1)^{j−i}·c /
    ∏_{l=i..j}(z_l − a)`, by running products along each row and ONE complex
    division per entry (the numerator folded into the quotient, as the
    collected fold's `amp/Δ` is): O(k²) complex ops, k² divisions. -/
def scaledInverse (nodes : Array CplxE) (a c : CplxE) : BuildM DDTable := do
  let k := nodes.size
  let zero ← Oriented.natE 0
  let gaps ← nodes.mapM fun z => csubE z a
  let mut entries : Array CplxE := Array.replicate (k * k) zero
  for i in [0:k] do
    let mut denominator : Option CplxE := none
    for j in [i:k] do
      denominator := some (← match denominator with
        | none => pure gaps[j]!
        | some d => cmulE d gaps[j]!)
      let quotient ← cdivE c denominator.get!
      let entry ← if (j - i) % 2 == 1 then cnegE quotient else pure quotient
      entries := entries.set! (i * k + j) entry
  pure { size := k, entries }


/-- Binder id of the coefficient-side Leibniz loops. The hand-maintained binder
    space: 0 the terminal bank, 1 `cauchyFold`, 2–16 the modal families, 23 and
    4300/4301 elsewhere — grep `loopIdx [0-9]` before choosing a new one. -/
def leibnizBinder : Nat := 17

/-- The BANKED stage table: `Σ_ν coeff_ν/(s − z_ν)` over a stage's poles as ONE
    reduction per table entry instead of one `scaledInverse` table per pole.
    The (pole, coeff) pairs ride four coefficient columns (`arr` of node re/im,
    coeff re/im); each of the k(k+1)/2 complex entries is two `bankSum`
    reductions (real, imaginary) whose body is `scaledInverse`'s entry formula
    for the INDEXED pole — the `cauchyFold` discipline (`OrientedRealize`).

    `member? = some (m, z_m)` is the stage owning ONE cluster pole: every
    non-member ν contributes `c_ν + c_ν·(z_ν − z_m)/(s − z_ν)` (the constant plus
    one scaled inverse, as the unrolled path's third branch) and the member
    itself contributes the constant `c_m`; so the body computes the shifted
    coefficient `c·(z_ν − z_m)`, masks the member's lane (`loopIdx = m`) to +0
    and adds `c` on the diagonal. The dead lane's divisor is swapped to 1
    BEFORE the division (the fixed-lane families' convention, never a
    post-division select): a 0/0 in a masked lane is a NaN the JIT would
    tolerate but the GPU need not.

    Bit-identity with the unrolled chain. Unrolled, the stage total is
    `((0 + T₁) + T₂) + …` entrywise in factor order, each `T_ν` computed by the
    same per-pole expression this body computes at index ν; `bankSum` starts
    its accumulator at 0 and adds each item's contribution in index order —
    the same f64 operations in the same order, and a masked lane adds exactly
    +0. The one edge is `−0` (a −0 partial sum plus +0 is +0 either way; never
    observed — the `generic-banking` note), which the `block-banked` gate
    would surface as a 1-sample bit difference. -/
def scaledInverseBanked (nodes : Array CplxE) (poles coeffs : Array CplxE)
    (member? : Option (Nat × CplxE) := none) : BuildM DDTable := do
  let k := nodes.size
  let zero ← Oriented.natE 0
  let one ← Oriented.natE 1
  let nRe ← arr (poles.map (·.1))
  let nIm ← arr (poles.map (·.2))
  let cRe ← arr (coeffs.map (·.1))
  let cIm ← arr (coeffs.map (·.2))
  let tables := #[nRe, nIm, cRe, cIm]
  let idx ← loopIdx leibnizBinder
  let a : CplxE := (← index nRe idx, ← index nIm idx)
  let c : CplxE := (← index cRe idx, ← index cIm idx)
  let (scaled, isMember?) ← match member? with
    | none => pure (c, none)
    | some (m, zm) => do
      let isM ← binary .eq idx (← litI m)
      pure (← cmulE c (← csubE a zm), some isM)
  let mask := fun (v : CplxE) => do
    match isMember? with
    | none => pure v
    | some isM => pure (← selectE isM zero.1 v.1, ← selectE isM zero.2 v.2)
  let swap := fun (d : CplxE) => do
    match isMember? with
    | none => pure d
    | some isM => pure (← selectE isM one.1 d.1, ← selectE isM one.2 d.2)
  let gaps ← nodes.mapM fun z => csubE z a
  let mut entries : Array CplxE := Array.replicate (k * k) zero
  for i in [0:k] do
    let mut denominator : Option CplxE := none
    for j in [i:k] do
      denominator := some (← match denominator with
        | none => pure gaps[j]!
        | some d => cmulE d gaps[j]!)
      let quotient ← cdivE scaled (← swap denominator.get!)
      let signed ← if (j - i) % 2 == 1 then cnegE quotient else pure quotient
      let masked ← mask signed
      let entry ← match isMember? with
        | none => pure masked
        | some _ => caddE (if i == j then c else zero) masked
      let real ← bankSum poles.size tables entry.1 none leibnizBinder
      let imag ← bankSum poles.size tables entry.2 none leibnizBinder
      entries := entries.set! (i * k + j) (real, imag)
  pure { size := k, entries }


/-- The factor `1/(s − a)` for a pole `a` OUTSIDE the cluster:
    `[i,j] = (−1)^{j−i} / ∏_{l=i..j}(z_l − a)`. The only division in the
    algebra, and every divisor is a cross-cluster gap. -/
def inverse (nodes : Array CplxE) (a : CplxE) : BuildM DDTable := do
  scaledInverse nodes a (← Oriented.natE 1)

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

/-- Can two enclosures NOT be separated? A decimal literal enters as a tight
    enclosure, not a point, so two spellings of one number (`9`, `90·10⁻¹`)
    overlap rather than coincide; the three-answer discipline reads an overlap
    as "cannot be told apart" — the conservative side here, since the
    alternative is dividing by a gap that may be exactly zero. -/
private def inseparable (a b : DyadicI) : Bool :=
  a.ok && b.ok && DyadicI.cmp a b == .overlap

/-- Are two s-plane nodes ONE node — the same hash-consed expressions, or two
    constants whose enclosures cannot be separated? Such nodes carry
    multiplicity (polynomial degree), never a divided difference between them:
    no gap that may be exactly zero is ever divided by. Two constants a
    representable distance apart (two doubles 1e-9 rad/s apart enter exactly)
    remain two nodes. -/
def sameNodeValue (constants : Array (Option DyadicI)) (a b : CplxE) : Bool :=
  (a.1 == b.1 && a.2 == b.2) ||
  match sigConstDFrom? constants a.1, sigConstDFrom? constants b.1,
        sigConstDFrom? constants a.2, sigConstDFrom? constants b.2 with
  | some ar, some br, some ai, some bi => inseparable ar br && inseparable ai bi
  | _, _, _, _ => false

/-- Partition a cluster's positions into value classes (traversal order kept). -/
def valueClasses (constants : Array (Option DyadicI)) (nodes : Array CplxE) :
    Array (Array Nat) := Id.run do
  let mut classes : Array (Array Nat) := #[]
  for (z, i) in nodes.zipIdx do
    match classes.findIdx? (fun c => match c[0]? with
        | some j => sameNodeValue constants z nodes[j]!
        | none => false) with
    | some k => classes := classes.set! k (classes[k]!.push i)
    | none => classes := classes.push #[i]
  return classes

/-- Union-find over the expanded pole multiset under the shared pole-distance
    lens: two poles of the SAME orientation join a cluster when their min |Δ|
    over the declared σ intervals is certifiably below θ_acc, or when they are
    one node value (a factor's own copies, or two spellings of one constant). Opposite orientations never cluster — a future pole and a past
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
            oi == oj && (sameNodeValue constants zi zj
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
    (member : Nat → Bool) (banked : Bool) (cursor : Nat) :
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
      let anyMember := (Array.range factors.size).any inCluster
      let members := (Array.range factors.size).filter inCluster
      let memberCount := members.foldl (fun acc i => acc + ((factors[i]?).map (·.mult)).getD 0) 0
      let memberNode : Option CplxE := members[0]?.bind fun i => (factors[i]?).map (·.node)
      -- BANKED (`banked`, the `Ir.banksEnabled` realization knob): a stage of
      -- ≥ 2 simple poles that owns no cluster pole (`bankedA`) or exactly one
      -- (`bankedB`, k > 1) is ONE `scaledInverseBanked` table — the per-pole
      -- loop below is its unrolled twin and the gate oracle (the `modalBankSig`
      -- meta-fold precedent). Multiplicity > 1 and a stage owning ≥ 2 poles of
      -- one cluster (deliberate unisons, over-cap value classes) stay on the
      -- unrolled general branch: rare and small.
      let simple := factors.size ≥ 2 && factors.all (·.mult == 1)
      let bankedA := banked && simple && !anyMember
      let bankedB := banked && simple && anyMember && memberCount == 1 && k > 1
      let mut total ← DDTable.zero k
      if bankedA then
        total ← DDTable.scaledInverseBanked nodes (factors.map (·.node)) (factors.map (·.coeff))
      else if bankedB then
        if let some m := members[0]? then
          if let some zm := memberNode then
            total ← DDTable.scaledInverseBanked nodes (factors.map (·.node))
              (factors.map (·.coeff)) (some (m, zm))
      for (f, i) in factors.zipIdx do
        -- term_ν = coeff_ν · ∏_{ν'∈c∩stage, ν'≠ν}(s − z_ν')^{mult} · [(s − z_ν)^{−mult} if ν ∉ c]
        if bankedA || bankedB then
          pure ()
        else if anyMember && k == 1 && !(inCluster i) then
          -- a singleton cluster {z_m}: every non-member term of z_m's own
          -- stage carries the factor (s − z_m), whose one-node table is
          -- (z_m − z_m) = 0 EXACTLY — nothing to compute
          pure ()
        else if !anyMember && f.mult == 1 then
          -- the common case (a stage none of whose poles is in this cluster):
          -- `coeff_ν/(s − z_ν)` as one scaled inverse table, no table products
          total ← total.add (← DDTable.scaledInverse nodes f.node f.coeff)
        else if f.mult == 1 && !(inCluster i) && memberCount == 1 then
          -- a stage owning ONE cluster pole z_m, a non-member ν of multiplicity 1:
          -- `c·(s − z_m)/(s − z_ν) = c + c·(z_ν − z_m)/(s − z_ν)` — a constant
          -- plus one scaled inverse, instead of two table products
          let some m := memberNode | pure ()
          let shifted ← cmulE f.coeff (← csubE f.node m)
          let term ← (← DDTable.const k f.coeff).add (← DDTable.scaledInverse nodes f.node shifted)
          total ← total.add term
        else
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
      let (table, poles, cursor) ← leibniz constants nodes member banked cursor kernel
      pure (← table.scaleReal value, poles, cursor)
  | .cascade kernels => do
      let one ← DDTable.one nodes.size
      kernels.attach.foldlM (fun (state : DDTable × Array CplxE × Nat) kernel => do
        let (table, poles, cursor) := state
        let (t, p, c) ← leibniz constants nodes member banked cursor kernel.1
        pure (← table.mul t, poles ++ p, c)) (one, #[], cursor)
  | .parallel kernels => do
      -- Σ_a T_a · ∏_{b≠a} D_{c,b}: each branch carries the other branches'
      -- cluster factors so every summand is (branch · D_{c,node}).
      let (branches, cursor) ← kernels.attach.foldlM
        (fun (state : Array (DDTable × Array CplxE) × Nat) kernel => do
          let (t, p, c) ← leibniz constants nodes member banked state.2 kernel.1
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
      let (dt, dp, cursor) ← leibniz constants nodes member banked cursor dry
      let (wt, wp, cursor) ← leibniz constants nodes member banked cursor wet
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
    reflection. `confluent` marks a row whose nodes are ONE value (it realizes
    as polynomial degree). `indices` are the nodes' global positions in the
    expanded multiset. -/
structure BlockRow where
  nodes : Array CplxE
  coeffs : Array CplxE
  indices : Array Nat
  orientation : Oriented.Orientation
  confluent : Bool

/-- Block partial fractions of a bilateral retained tree — TOTAL. Expand the
    pole multiset (arms by direction, copies by degree), cluster it, then run
    the Leibniz traversal once per row. A cluster with at most `cap` value
    classes is one row; a cluster with more splits into one row per value
    class — its classes then divide by their mutual gaps, the collected fold's
    floor, which is exactly what every such spine rendered before (and a gap
    between distinct certified values is never exactly zero). `banked` selects
    the stage-table realization (`scaledInverseBanked` vs the unrolled per-pole
    chain); it defaults to the process-wide `Ir.banksEnabled` and is explicit
    only so the `block-banked` gate can build the unrolled oracle in-process. -/
def decompose (spine : ModalKernelExpr) (cap : Nat := defaultClusterCap)
    (banked : Bool := Tropical.Ir.banksEnabled) : BuildM (Array BlockRow) := do
  let factors ← poleFactors (sigConstTable (← get).exprs) spine
  -- the constant table is taken AFTER expansion: a past arm's mirrored pole is
  -- a fresh `neg` node, and the lens must be able to fold it
  let constants := sigConstTable (← get).exprs
  let expanded := expandNodes factors
  let clusters := clusterPoles constants expanded
  let mut rows : Array BlockRow := #[]
  for cluster in clusters do
    let entries := cluster.filterMap (expanded[·]?)
    let nodes := entries.map (·.1)
    let some (_, orientation, _) := entries[0]? | continue
    let classes := valueClasses constants nodes
    let groups : Array (Array Nat) :=
      if classes.size ≤ cap then #[cluster]
      else classes.map fun cls => cls.filterMap (cluster[·]?)
    for indices in groups do
      let nodes := (indices.filterMap (expanded[·]?)).map (·.1)
      let member := fun (g : Nat) => indices.contains g
      let (table, _, _) ← leibniz constants nodes member banked 0 spine
      let confluent := (valueClasses constants nodes).size ≤ 1
      rows := rows.push { nodes, coeffs := table.newton, indices, orientation, confluent }
  pure rows

end Tropical.EmitArrow.Block
