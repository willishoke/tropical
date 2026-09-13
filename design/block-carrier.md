# The block carrier — repeated rooms without collecting

Status: Phases 1–5 landed 2026-09-13 (`feat/block-carrier`). Every plain
modal spine lowers through the block terminal, any direction, except the three
exact cost schedules that keep their topologies. Cockpit
scripts: `demos/block_carrier_leibniz.py` (the coefficient algebra in f64 vs a
60-digit reference), `demos/block_carrier_body.py` (the size-3 body),
`demos/block_carrier_exp.py` / `dd_union_law.py` (the research experiments).

## The problem it removes

The modal island's composable carrier was the collected partial-fraction bank:
after every room, the residues at the pole union were formed, each carrying a
`1/(λ−ν)` from the fold. That is the parallel realization of a rational
transfer function, the most coefficient-sensitive one for clustered poles. The
divided-difference paired atom (`PairedMode`) repaired the near-coincident case
at the LAST room only, so `room ⋙ room ⋙ room` was refused topologically
(`plainStageSpineAdmitted`), and the generic two-room fallback still folded its
first room through the syntactic classifier — protected by expression identity,
not by value.

## The carrier

The composable carrier is the retained factor tree the island already kept:
`ModalKernelExpr` (identity / proper / scale / parallel / cascade / blend). The
block terminal (`Modal/Block.lean`) consumes it once, at the end of a spine:

1. **Pole multiset.** The union over the tree in the s-plane; composition
   never moves a pole. A room's past arm enters as the MIRRORED pole `z' = −ν`
   with the two-sided Laplace transform's sign (`−δ·r/(s−z')`, and
   `(−1)^{p+1}·B·p!` for degree `p`), so both arms are one rational function;
   a degree-`p` mode is `p+1` copies of its node with coefficient `A·p!`. An
   exactly-forward or exactly-reversed room contributes one arm; a live or
   fractional direction contributes both, scaled `(1−δ)` / `δ`.
2. **Clusters.** Union-find under the one pole-distance lens the pairwise router
   already uses (`poleAccuracyHotFrom?`: min |Δ| over declared σ intervals,
   ω exact, certified below `θ_acc = 0.4642` rad/s on the `DyadicI` carrier),
   or expression identity. Never across orientation: a future and a past pole
   are separated by at least `σ_f + σ_p` and are different signals. Amps play
   no part: membership is a pole question.
3. **Coefficients.** For a cluster with ordered nodes `z₁..z_k`, the Newton
   coefficients `n_m = g_c[z₁..z_m]` of `g_c = H·D_c` are computed by the
   Leibniz rule for divided differences over the tree's factors — linear
   factors `(s − a)` for the cluster's own poles (division-free), inverse
   factors `1/(s − a)` for every other pole (division by cross-cluster gaps
   only), products by `(fg)[i,j] = Σ f[i,l]·g[l,j]`. No `1/(zᵢ − zⱼ)` with both
   nodes in one cluster is ever formed; exact coincidence needs no branch; a
   singleton reduces to the ordinary residue as a product of sums.
4. **Rows.** `h_c(d) = Σ_m n_m · exp[z_m..z_k](d)` — the suffix divided
   differences of `e^{zd}`. A past row is the anti-causal
   `−Σ_m n_m·exp[z'_m..z'_k](d)` on `d < 0`, which on the mirrored clock
   `d' = −d` is `Σ_m (−1)^{k−m+1}·n_m·exp[ν_m..ν_k](d')` at the physical poles —
   the same families, reflected. The strike sample carries `0` for a
   single-sided spine (the causal convention) and the continuous `h(0) =
   Σ_{future rows} n_k` once a past arm exists.

Every coefficient is a `CplxE` (`Sig × Sig`): literal poles const-fold on the
exact carrier and round once; live poles ride the stage-0 kernel, as the
collected residues did. `DyadicI` decides, it never computes a coefficient.

## Realization (`Modal/BlockRealize.lean`)

| row | body |
|---|---|
| size 1 | a plain `ModalMode` — the fixed Q datapath, unchanged |
| size 2 | one `PairedMode` (`n₁·e^{z₂d}·d·cexpm1((z₁−z₂)d)`, the float paired lane) + a plain mode `n₂` at `z₂` |
| size 3 | one `TripleMode` (`n₁·e^{z₃d}·d²·Φ(u,v)`, `u = (z₁−z₃)d`, `v = (z₂−z₃)d`) + a paired row + a plain mode |
| all nodes one expression, any size | degree modes: `exp[z,…,z] = d^{k−m}/(k−m)!·e^{zd}` |

`Φ` has a series lane (`Σ h_n(u,v)/(n+2)!` to degree 10, exact at the triple
collision) below `|·|² < 0.01` on every pairwise gap, and otherwise one level of
the divided-difference recurrence over `cexpm1` factors, choosing the symmetric
form whose divisor is largest so node order can never put a near-zero gap in a
live divisor. Both lanes are evaluated every sample; dead divisors are swapped
to 1 (the `cexpm1` discipline). The `z₃` carrier is the float terminal carrier
(`expSig` × the Q0.32 rotator); `u`, `v` use raw differences bounded by the
cluster tolerance.

## Admission and refusal

`resolvePlainStages` keeps three exact cost schedules for their topologies —
the time-staged phaser (`stagedPhaserTerminal?`), the fused two-room product
(`factoredTwoRoomTerminal?`) and the fused two-room-with-phaser product — and
sends every other plain spine to `resolveBlockSpine`: rooms (any direction,
live or reversed; sway) and linear kernels are retained factors, decomposed
once at the spine's end. A gauge is nonlinear in the whole bank, so it splits
the spine into segments: the block terminal before it is materialized to a
collected bank (`BlockTerminal.toBank`, the one structure-dropping step, at
the collected fold's `1/Δ` floor), gauged, and re-enters as the next
segment's input factor. The carrier is TOTAL: a cluster with at most three
value classes is one row; a larger one splits into its classes, which then
divide by their mutual gaps — the collected fold's floor, exactly what every
such spine rendered before. Two constant nodes are one value class when their
enclosures cannot be separated (a decimal literal is a tight enclosure, so two
spellings of one number overlap rather than coincide; the three-answer
discipline reads that as "cannot be told apart", the conservative side, since
the alternative is dividing by a gap that may be exactly zero); two doubles a
nanoradian apart enter exactly and stay two nodes. The syntactic nonterminal
fold, the two-room generic fallback, and the topological spine rule are gone.

## Witnesses

- `block-algebra` (`Tropicaltest/Block.lean`): singleton rows equal the
  collected residues up to reassociation; an identity-coincident pair's Newton
  coefficients equal the `Oriented` coincident algebra's degree-1/degree-0
  amplitudes; a reversed room is admitted as past rows; an over-cap comb splits to the
  collected floor; three spellings of one value are one confluent row.
- `block-realize` (`Tropicaltest/BlockRealize.lean`), at the observable:
  singleton render bit-identical to the plain path; a four-fold confluent pole
  vs the closed form; a gap-1e-3 triple plus a separated pole vs the exact
  partial fractions on the 128-bit carrier (enclosure width reported as the
  oracle's certificate); the triple collision vs its closed form; hand-built
  triple rows at gaps 3 and 5 rad/s so the lane seam is crossed in-window.
  Measured 2026-09-13: singleton 0, confluent 1.7e-6, triple 1.0e-5 (the plain
  family's Q4.28 landing LSB is the binding floor), collision (poles a
  nanoradian apart) 9.7e-10, lanes 4.9e-7; fail lines at 1e-7 / 1e-4.
- `blockCompose` seam atom (`Tropicaltest/SeamSweep.lean`): `voice ⋙ room`
  through the block terminal over the residue atom's own interiors and
  boundary probes against the quadrature oracle, total admission, snr 2e-4.
- `ecdd-gauge`: the gauge's scale law now holds at the sub-grid detune too —
  the segment after a gauge re-enters the block terminal, so the tuned pair
  renders on the divided-difference lane; the landing poison that gate once
  recorded no longer reaches the render.
- `arena-native-phase3` re-frozen: its room → phaser → room spine (rooms of
  different frequency topology, never the fused schedule) now lowers through
  the block terminal — (2234, 2232, 0, 799131) nodes/reachable/routed/bytes.
- `modal-oriented-patch`: three separately authored equal rooms render through
  the production lowering against `1/((s+2)(s+5)³)`; a past·future·past chain
  against the bilateral partial fractions; room-room-gauge renders through a
  materialized segment.
- `block-realize` bilateral probe: past·future·past rooms with a hot PAST pair
  against the exact two-sided partial fractions on the 128-bit carrier, both
  arms observed around a mid-window anchor (9.4e-6).

## Not yet

- A fixed-lane landing for block rows via the Hermite–Genocchi sup bound
  `|n₁|·((k−1)/(σ_min·e))^{k−1}/(k−1)!`; Metal. Phase 6.
- Live poles without a declared σ range are unclassifiable and stay singletons
  (the pairwise router's `cold` convention); two such poles becoming
  runtime-equal divide by their gap exactly as the collected fold did.
