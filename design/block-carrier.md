# The block carrier — repeated rooms without collecting

Status: Phases 1–6 landed 2026-09-13 (`feat/block-carrier`). Every plain
modal spine lowers through the block terminal, any direction, except the three
exact cost schedules that keep their topologies; every family renders on the
fixed i64 lane. Cockpit
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
| size 2 | one `PairedMode` (`n₁·e^{z₂d}·d·cexpm1((z₁−z₂)d)`, `modalBankSigTableDD`) + a plain mode `n₂` at `z₂` |
| size 3 | one `TripleMode` (`n₁·e^{z₃d}·d²·Φ(u,v)`, `u = (z₁−z₃)d`, `v = (z₂−z₃)d`, `tripleSig`) + a paired row + a plain mode |
| all nodes one value, any size | degree modes: `exp[z,…,z] = d^{k−m}/(k−m)!·e^{zd}` |

Every family lands on the fixed i64 datapath (Phase 6). The plain family uses
`bankLandExp`; the paired and triple families land at their own per-bank
option-E exponent from the Hermite–Genocchi sup bounds `|c|/(e·σ_min)` and
`2|c|/(e·σ_min)²` (`pairedLandExp`, `tripleLandExp` — static on the exact
carrier when the row folds, a dynamic s0 expression otherwise), so there is no
admission cap: the exponent absorbs the range. `modalBankSigTableDD` took an
optional landing for this; its default is the verbatim q28 literals.

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
  the block terminal — (1495, 1493, 0, 505498) nodes/reachable/routed/bytes after the Phase 6 cost reductions.
- Phase 6 floors (fixed lanes), 2026-09-13: triple 1.55e-5, collision 6.7e-6,
  lanes 4.9e-7, bilateral 1.07e-5, paired headroom (sup ≈ 2649, `k = 7`, 80×
  over the bare rail) 1.7e-9.

## Cost (measured through `diffcli render-graph --stage-census`)

The coefficient algebra costs one complex division per (cluster, outside pole,
table entry): `k²` per pole for a size-`k` cluster, `1` for a singleton. Two
identities keep the singleton cost equal to the collected fold's: a non-member
term of a singleton's own stage carries `(s − z_m)`, whose one-node table is
exactly zero (skipped), and the numerator folds into the reciprocal. A stage
owning one pole of a larger cluster uses `c·(s−z_m)/(s−z_ν) = c +
c·(z_ν−z_m)/(s−z_ν)`, a constant plus one scaled inverse.

| playground graph | base `main@943bf32` | this branch |
|---|---|---|
| resonator ⋙ reverb (6 + 14 modes, live direction ⇒ both arms) | 10825 audio IR lines, 407 stage-0 | 9652 audio instructions, 2106 stage-0 |
| resonator ⋙ reverb ⋙ reverb ⋙ reverb (rt60 0.1 % apart ⇒ 28 triple clusters) | refused | ~580k audio instructions |

The second row is the finding: on this base the playground's room
coefficients are STAGE-1 — the controls are frozen at the terminal coordinate,
so every coefficient tree runs per sample, at base too (the base's one-room
kernel is 10 k lines for 34 modes). The block terminal's trees are correct and
s0 in isolation (`block-algebra`'s stage probe) but larger than the collected
fold's on wide clustered spines, so until the coefficient plane settles to
stage 0 (the in-flight `modal-s0-compose` work on the main checkout, which
brings the one-room kernel to ~860 lines) a wide repeated-room chain is
compile-heavy. On Metal the two-room paired kernel compiles and renders; the
three-room triple kernel exceeds the shader compiler
(`XPC_ERROR_CONNECTION_INTERRUPTED`), and its bun case in
`tests/web/metal_vs_jit.test.ts` is skipped with that reason. `diffcli
render-graph` gained `--dump-plan=<path>` and `--stage-census` for this.
- `modal-oriented-patch`: three separately authored equal rooms render through
  the production lowering against `1/((s+2)(s+5)³)`; a past·future·past chain
  against the bilateral partial fractions; room-room-gauge renders through a
  materialized segment.
- `block-realize` bilateral probe: past·future·past rooms with a hot PAST pair
  against the exact two-sided partial fractions on the 128-bit carrier, both
  arms observed around a mid-window anchor (9.4e-6).

## Not yet

- Re-enable the Metal three-room case and re-measure the cost table once the
  coefficient plane settles to stage 0.
- Live poles without a declared σ range are unclassifiable and stay singletons
  (the pairwise router's `cold` convention); two such poles becoming
  runtime-equal divide by their gap exactly as the collected fold did.
