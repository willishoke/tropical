# The seam-atom contract — the island's one promise

The modal island grew a composition algebra (residue calculus, divided
differences, the Γ-bridge across a pitch bloom) before it grew a statement of
what any of those crossings *promises*. Each atom shipped with the same
artisanal witness chain — derive in a numpy cockpit, eyeball a convergence
boundary, freeze a handful of points into a gate — and so each atom's scope
read as ad hoc: admission regions lived in comments, `∀`-laws were checked at
points, and "is the class closed" lived in conversation, not the repo. This
doc is the contract those atoms were always instances of. One page, because
the promise is one sentence.

## The promise

> **A seam atom's realized signal equals the realized oracle — pointwise,
> within its stated error model, over its stated admission region.**

Everything load-bearing is in that sentence, so read it slowly.

- **Realized signal** — the atom's *representation is its own business*
  (`Array ModalMode`, `Array PairedMode`, `Array BloomPair`, whatever a future
  atom needs). The contract only ever looks at the atom *rendered to a `Sig`
  and evaluated* — the samples that leave the datapath. This is the hourglass
  move the trunk already made once: state the law at the waist, not at any one
  representation above it. Because the law lives at the observable, an atom may
  return any representation with any realizer, and the **Sig-codomain fork**
  (is `Sig` the mainland, or should composition eliminate to it?) is deferred
  *without prejudice* — the math transfers either way; only seam placement is
  at stake, and nothing here presumes the answer.

- **The oracle** — one reference for every seam crossing: direct
  high-resolution quadrature of the composition integral

  ```
  y(d) = Σ_{voice μ, room ν}  a_μ·r_ν · ∫₀^d e^{ν(d−s)}·e^{μ·φ(s)} ds
  ```

  summed over voice-pole × room-pole pairs, `Re` taken, the carrier sink gain
  applied, and ω quantized to the datapath's rotator grid so the reference
  refines toward the frequencies the kernel actually carries. It is generic
  over the warp **φ**, and that genericity is the unification the island never
  wrote down: **every atom is this one oracle at a different φ.** `residueEC`
  and `residueDD` are `φ = id`; the Γ-bridge is `φ(s) = s + B(1−e^{−gs})`. The
  oracle is total, mechanical, slow, and free of cancellation pathology — the
  machine's slow path that every fast atom is gated against. It is NOT the
  pretty pole-ladder oracle (expand the bloom as `Σ(−κ)ⁿ/n!·e^{−ngd}` and
  compose term-by-term): that is the already-refuted Poisson lattice, float64
  dead above the fundamental (coefficients reach `~e^{|κ|}`, `|κ|` up to ~178).
  The ladder is the *story* of why closure holds; quadrature is the
  *computation* that witnesses it.

- **Error model** — an SNR threshold, not float eps. The datapath has an
  honest floor (float-envelope-vs-fixed ≈ 1.4e-5 at 0.74 s, drift-type, scales
  with τ); a threshold sits a comfortable multiple above that floor, never at
  machine precision. This generalizes the two-realization thesis: *n*
  realizations, one oracle, SNR-gated. A crossing that lands below its floor
  passes; one that drifts above it fails.

- **Admission region** — an *executable predicate* over pole configurations
  and warp parameters, not a comment. `|a| < ½` excluded, depth > 300 excluded,
  near-coincident poles routed elsewhere: these are outputs of a predicate the
  harness can probe, not inputs from a human squint. An atom and the harness
  consult the *same* predicate, so "where does this atom promise anything" has
  one answer, in code.

## The four fields

A `SeamAtom` is exactly these, and nothing else:

| field | what it is |
|---|---|
| **algorithm** | `bank × bank × warp-params → representation` — the build-time pole move (`residueComposeEC`, `residueComposeDD`, `bloomCompose`) |
| **realizer** | `representation → Sig` — how the atom's own form becomes samples (`modalBankSigTable`, `modalBankSigTableDD`, `bloomComposedSig`) |
| **admits** | `config → Bool` — the admission region as a predicate |
| **error model** | the SNR threshold vs the oracle, above the datapath floor |

The **law** is not a fifth field; it is the standing obligation stated at the
observable, and the sweep harness is its executable form: for configs the
predicate admits, the realized algorithm agrees with `oracleSeam φ` within the
error model. A frozen golden pins *history* (regression); the sweep checks the
*law* (the region). Different instruments — keep both.

## The role-split rule

The last sprint's sin was letting the derivation lab's one-time output
masquerade as a standing witness. The rule that prevents recurrence:

> **Cockpit = derivation lab. Repo = witness. Nothing the cockpit finds counts
> as verified until it is an instance in the harness.**

The numpy/mpmath cockpits (`demos/*.py`) stay runnable and stay the wide
exploration surface — full-range comparators, boundary maps, the place a
candidate is born. But a candidate is *verified* only when it is a registered
`SeamAtom` whose law the sweep re-checks on every `make validate`. The
apparatus is kept, not frozen.

## The three instances (v1)

| atom | φ | representation · realizer | admission |
|---|---|---|---|
| `residueComposeEC` | `id` | collected `ModalMode` bank · `modalBankSigTable` | separated poles (`|Δ|` above the coincidence floor — near-coincidence is `DD`'s region) |
| `residueComposeDD` | `id` | `PairedMode` bank · `modalBankSigTableDD` | total (the `cexpm1` limit is the τ·e resonance, no branch) |
| `bloomCompose` | `s + B(1−e^{−gs})` | `BloomPair` bank · `bloomComposedSig` | baked poles; `|a| ≥ ½`; envelope depth ≤ 300 |

`bloomCompose` at `B→0` collapses to `residueComposeDD` (the Γ-bridge is the
κ-extension of the divided difference), so the three are one family with a
boundary the sweep can walk.

## What this repairs (the five failures)

1. **∀-claims, point-witnesses** → the sweep re-checks the law over the region
   every run (Halton-sampled, boundary-probed).
2. **Admission regions in comments** → `admits` predicates in code, probed at
   their edges.
3. **The law unstated** → this doc; the atoms are its instances.
4. **Cockpit frozen instead of kept** → the role-split rule; cockpits stay
   runnable, the harness stands witness.
5. **Codomain a default** → the law at the observable defers the fork
   explicitly, rather than presuming `Sig` is the mainland.

## Deferred, on purpose (recorded so they stop being defaults)

- **Sig codomain / Sig-elimination** — arbitrated after atom four's cost datum,
  not here.
- **Seam policy** (typed modal inlet vs convolution descent) — the eventual
  total repair is a costed lattice, not a typed refusal; the realized-observable
  law is deliberately compatible with it.
- **Live-pole blooms** — `bloomCompose` is baked-pole v1; the sweep probes the
  baked region only, and the predicate says so.
- **Proofs** — once the contract is a structure with a law obligation, the Lean
  theorem register is an open door: statement costs nothing, proving instances
  is incremental. Goldens-as-floor stands; we do not pay for proofs this sprint.
