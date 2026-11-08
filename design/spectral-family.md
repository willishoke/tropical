# Spectral family and complex envelope (S2, S3)

Date: 2026-11-07. Status: landed on `feat/spectral-family`.

This note covers the first two slices of the spectral plan: S2, the
partial family term, and S3, the complex spectral envelope. It records
what landed, the owner's decisions behind it, the limits, the measured
cost, and what S4 to S6 need. The plan itself is the spectral handoff
(untracked, 2026-11-04).

## 1. What landed

**The family is a term the algebra can see.** `ArrowTerm.family`
(`lean/Tropical/EmitArrow/Term.lean`) carries a `PartialFamily`
(`lean/Tropical/EmitArrow/Spectral.lean`):

- partials `k = 1..K` at `k·f0`, `f0` uniform;
- one phasor on the exact integer phase rail, with partial `k`'s phase
  `k·φ` taken from it;
- weights `a_k` as a coefficient table (one `Sig` per partial, constant
  or live), or as a closed form in `k`;
- an optional phase offset (partial `k` is offset by `k·phase`, so one
  anchor keeps every partial locked);
- an optional live partial count, clamped to `K`;
- the real projection it emits: `sine` (`Im`) or `cosine` (`Re`,
  zero-phase cosines);
- soft edges at DC and Nyquist.

It lowers to one `bankSum` region. The phase is exact: with the clock's
fraction bits kept, `P ≡ (inc·⌊τ⌋ + off)·2^F + inc·frac(τ) (mod 2^(32+F))`,
and partial `k`'s Q32 phase is `⌊k·P / 2^F⌋ mod 2^32`. That is
`k·inc·τ + k·off`, floored once, at any τ, fractional τ included. The
sine and cosine come from the same fixed-point polynomial every exact
oscillator uses (`fixedSinCycSig`).

**The envelope is a term too.** `ArrowTerm.envelope law controls t`
multiplies each partial of `t` by a complex gain `G(f_k, τ)` at its own
frequency: amplitude × `|G|`, phase + `arg G`. The real projection is
taken after the gain, so the phase term is part of the sound (the
listening set measured it roughly doubling the sideband spread at fast
modulation). An `EnvelopeLaw` is a closed form. `prepare` turns the
control signals and `f0` into per-sample scalars, computed once before
the partial loop. `gain` returns `(Re G, Im G)` per partial.

The controls are signal terms evaluated in the envelope's clock context,
as `genC`'s are. Cutoff and Q can therefore be any signal: an LFO at
audio rate, another kernel's output, a knob.

`svfEnvelope` gives the named shapes, the SVF responses with
`x = f/fc` and `D = 1 − x² + j·x/Q`:

| shape | `G` |
|---|---|
| lowpass | `1/D` |
| bandpass | `(j·x/Q)/D` (unity gain at `fc`) |
| highpass | `−x²/D` |

Per partial a shape costs one multiply for `x = k·(f0/fc)` and one
division (`1/|D|²`).

**The algebra.** `normalize` distributes an envelope over `sum` (`G` is
linear) and commutes it with `scale` (`G` multiplies amplitudes, as
`scale` does). It stops above a family, which is the normal form. Two
envelopes on one family multiply. Emission refuses an envelope over
anything else: an opaque generator, a pointwise map, a product, or a
warp. The refusal names the term.

**Soft edges.** A partial fades with a smoothstep edge `t²(3 − 2t)` as
its frequency nears Nyquist (by default over the top 10% below it) or
0 Hz (by default 20 Hz). The edge is exactly 1 in band, so an in-band
partial's value is untouched bit for bit. Partials at or above Nyquist
are not evaluated: the region trips `min(liveCount, ⌊Nyquist/|f0|⌋)`
partials, and a skipped partial's edge weight is 0 anyway.

**Fixtures and gates.**

- `lean/Tropical/Testing/SpectralFixtures.lean` holds `SpectralSaw`,
  `SpectralLowpassSaw` and `ExpCutoffLfo`. They are registered under
  `--fixtures`, so patches can instantiate them.
- `lean/Tropical/Tropicaltest/Spectral.lean` holds the five tropicaltest
  gates; `tropicaltest --spectral-only` runs them alone.
- `tests/web/wasm_vs_jit.test.ts` holds two WASM≡JIT cases.
- `benchmarks/spectral/bench.ts` takes the cost measurements.

## 2. The owner's decisions (2026-11-07)

1. The family is a new `ArrowTerm` constructor lowering to `bankSum`,
   not an opaque leaf with metadata. The lesson of `90242aa` is that the
   algebra must see the family.
2. Modal keeps its place: `ModalKernelExpr` stays the LTI sub-algebra a
   bundle enters at events. `ModalMode` is not absorbed in this slice.
3. The envelope is complex, not real. Lowpass, bandpass and highpass are
   named shapes. Cutoff and Q are animated signals; the animated case is
   the main one.
4. A partial near Nyquist or 0 Hz fades with a soft edge, not a hard cut.
5. Aliasing from a sharp gain at high Q under fast modulation is
   accepted as character. There is no oversampling (see §5).

## 3. Limits

- **Uniform fundamental.** `f0` must be constant within a kernel call,
  so a partial's frequency `k·f0` is a constant the envelope can be
  evaluated at. The builder refuses an `f0` whose stage signature reads
  τ or an instance output. Program inputs are trusted: the rate-inlet
  rule (a freq inlet accepts only uniforms) belongs to the wiring layer
  (#248). Phase-modulated partials are S5.
- **No envelope under a warp.** A warp between an envelope and its
  family is refused, because the warped partial's frequency needs θ′
  (S5). A delay would be exact, but a delay and a rate-changing warp
  cannot be told apart syntactically, so both are refused. A family
  without an envelope warps freely: it is a function of its clock.
- **Envelopes reach families only.** This is the point of decision 1,
  not an accident.
- **DC from sidebands is not removed.** The soft edges act on partial
  carrier frequencies, which the algebra knows. The sidebands an
  animated envelope creates (`k·f0 ± n·fm`) are not partials the algebra
  can see, so a sideband landing on 0 Hz still carries DC. This is the
  case the listening set measured at −11 dB in `harmonic_q8`. Removing
  it exactly needs the modulation as a family, so the sideband
  frequencies are explicit; that is S4. No stateful DC blocker was
  added.
- **The uniform envelope is not folded into stage 0.** When cutoff and
  Q are uniforms, the per-sample scalars (`f0/fc`, `1/Q`) hoist to the
  coefficient kernel, but the per-partial gain still runs per sample.
  Folding it into stage-0 weights needs per-partial arrays crossing from
  the coefficient kernel, and arrays are per-sample in v1 (`Stage0`).
  Decision 3 made this optional; it is a follow-up.
- **Capacity.** `K ≤ 2^(31−F) − 1 = 32767` partials, because `k·P` must
  stay below `2^63`.
- **Guards.** Cutoff and Q are floored at 1e-9, so a zero control cannot
  divide by zero. The bandpass shape is normalized to unity gain at
  `fc`.
- **Quasi-static.** `G` is evaluated at each partial's constant
  frequency and the current τ. That is exact by definition, since the
  envelope is the primitive and there is no filter for it to fall short
  of. It is not an emulation of SVF dynamics: there is no resonant
  ringing and no smear. The listening set's spectral-vs-SVF tables show
  where the two part.

## 4. Gates and numbers

All gates run in CI, and none depends on an untracked file.

| gate | result |
|---|---|
| `family-plan-size` | 102 instructions at K = 24, 181, 1000 and 8000; operands grow only by the K-entry weight column |
| `family-old-saw` | `90242aa`'s four 24-partial saws, re-expressed as four family terms: **bit-exact** against SHA-256s of that commit's own render (`diffcli render-graph`, channels saw1–saw4, τ₀ = 0 and 1000003, 16384 samples each), and bit-exact against today's compile of the same graph (`tests/fixtures/spectral/synced-scope-90242aa.json`) |
| `envelope-oracle` | `sweep_q0.7/q4/q12` at 9 windows from fm 0.5 Hz to the last window before 40 s (fm ≈ 440 Hz). Operator NMSE −177.8 to −180.3 dB. End to end −152.3 dB worst (Q 12) |
| `envelope-residue` | LP envelope at the modal Filter's (fc, Q) ≡ `residueComposeEC` forced-mode amplitudes `a·H(iω_k)`, worst relative error 1.6e-15 over 4 settings × 12 partials |
| `envelope-algebra` | `G·(w·a + b)` normalizes to `w·(G·a) + G·b` and emits identical IDs; the generator, warp and moving-`f0` cases are refused |
| WASM ≡ JIT | `SpectralSaw` (200 partials) and `SpectralLowpassSaw` with its cutoff wired from a 220 Hz `ExpCutoffLfo`, 512 samples each, within 1e-12 |

**Why the old saw is bit-exact.** The order is preserved.
`90242aa`'s partial `n` had increment `n·inc` and offset `n·K`, written
by the host. Its sine was the same `fixedSinCycSig`, its weight a `1/n`
knob, and its mix a left fold starting from partial 1. The family
computes the same integer phase (§1), multiplies by the same doubles
(`litExact(1/k)`, an exact decimal), and folds in the same order (the
`bankSum` order theorem). The edge weight is exactly 1 in band.

**The oracle.** The reference is a Lean port of the listening set's
`sweepCutoff` and `renderSpectralDirect` (`scripts/dsp.ts`): libm
`cos`/`atan2`/`exp`/`pow`, with the partial phase in the direct form.
There are two comparisons:

- *Operator.* The reference is fed the kernel's own cutoff samples, so
  only the family and envelope arithmetic differ.
- *End to end.* The reference evaluates the cutoff law with libm. The
  kernel's cutoff is good to 4.7e-9 relative: the fixed-point sine's
  Q30 step. That needed a double-precision `exp` in the fixture
  (`expPreciseSig`). The core `expSig` is a ~1e-9 minimax fit. At 40 s
  the chirp's phase is about 1.5e4 rad, so that error alone limited end
  to end to −77 dB at Q 12.

**The phase convention.** The script's partial phase is
`2π·k·f·n/rate`. Tropical's is the integer rail,
`2π·((k·inc·n) mod 2³²)/2³²` with `inc = ⌊f·2³²/rate⌋`. The rail runs
the 110 Hz fundamental 3.6 µHz slow at 48 kHz, so partial 181 drifts
0.16 rad by 40 s. The reference uses the rail. The rail-vs-exact
difference is printed: −40 to −68 dB NMSE, growing with τ. The envelope
evaluates `G` at the authored `f0`, as the script does.

## 5. Measured cost

`bun benchmarks/spectral/bench.ts`, run from the repo root after
`make lean`. It compiles with `diffcli compile-wasm` /
`compile-kernel --fixtures`, drives the kernel through `web/runtime`
(WasmKernel, 128-frame calls) on one thread, warms up for 1 s, and takes
the best of three 10 s renders. The machine is an Apple-silicon Mac.
Bun runs JavaScriptCore; Node 26 runs V8 and loads the same runtime,
bundled.

**The kill criterion:**

- the patch is 4 voices at 55, 68.75, 82.5 and 96.25 Hz;
- each voice is its own 200-partial saw family (all 200 below the
  Nyquist edge) under a complex SVF lowpass at Q 8;
- the cutoff is 800·2^(2·sin(2π·220·τ)), an audio-rate LFO;
- the rate is 48 kHz.

| case | wasm | JSC (bun) | V8 (node) |
|---|---|---|---|
| kill: 4 × 200, animated complex lowpass | 20.5 KB | **15.3–16.4%** of real time | **17.9%** |
| 4 × 200 saw families, no envelope | 18.1 KB | 9.5–10.5% | 9.1% |

This is the direct path. Each partial evaluates the fixed-point sine and
cosine of `k·φ` (about 30 i64 operations each), the complex gain (one
division) and the edges. It uses no recurrence over `k`. The rate is
38.4 M partial evaluations per second at about 4 ns each. The cost
divides roughly as follows:

- base family: ~9% of real time;
- complex envelope: +~6%;
- soft edges: ~1.3% of the total (measured with the edges switched off,
  not shipped).

That is under the 25% kill line, so the bounded fold (Clenshaw, or
rotation across `k`) was not added.

**Against `90242aa`** (44.1 kHz, same method):

| case | wasm | JSC | V8 |
|---|---|---|---|
| `90242aa` synced-scope graph, as measured at that commit | 111.1 KB | 7.47% | — |
| its four 24-partial saws alone, `90242aa` build | 44.0 KB | 1.33% | — |
| the same saws, today's compile of that graph | 40.5 KB | 1.2–1.4% | 1.20% |
| the same saws as four family terms | **18.1 KB** | 1.7–1.9% | 1.74% |

The family is far below 111 KB and 7.5%: the old figure includes four
flangers that read the saws at three clocks each, plus 14 taps. Against
the saws alone, the family is 2.2× smaller, but at 24 partials it costs
about 30–45% more per partial than 96 unrolled sine nodes. The
per-partial arithmetic is the same. The difference is the loop: an
in-order accumulator, a bounds-checked table read, and the edges, where
the unrolled form has 24 independent straight-line chains. The family
wins on plan and wasm size, which are O(1) in K, and on authoring: one
term, no host-written partial frequencies.

**Accepted aliasing.** The renders reproduce the oracle to −152 dB, so
they alias exactly as the listening set measured. In-band aliasing is
negligible for Q ≤ 4 and below fm ≈ 100 Hz. At Q 12 it reaches −46 dB
at fm 200 and −26 dB at fm 400. It comes from the low partials sweeping
through a sharp, fast-moving resonance: a near-impulsive gain spike per
crossing. Decision 5 accepts this as character, so there is no
oversampling. If it is ever wanted, the closed-form fixes are to
evaluate at 4× and decimate, or to band-limit each partial's gain
modulation.

## 6. Findings outside the slice

**Registration expands the DAG.** Registering a program whose
expression nests transcendentals over an input took 40–70 s under
`--fixtures`. The case was `expSig` of `logSig(fb/fa)`, the first
draft of the chirp fixture. The registration path renders a catalog
entry (`Entries.concreteEntry` → `Codec.encodeResolved`) that writes the
resolved expression out as a tree. For a three-level nest the entry was
21 MB of JSON, and `adoptResolved` re-parses it. The fixture now folds
`ln(fb/fa)` at assemble time. The encoding is worth a look: shared
subterms should stay shared.

**An honest playground kind is easy.** A `harmonic` node (a family with
a uniform `f0` knob) and an envelope node whose input must lower to
families would need no decoder tricks: a non-family input is refused at
emission. This landed later; see §8.

## 7. What S4–S6 need

- **S4, products of families.** A product term over two families whose
  partials are pairs `(j, k)`, with phases added on the integer rail
  (Q32 increments add exactly). That is a nested `bankSum` (inner
  binder id ≠ 1) or a flattened pair table. Nyquist and DC soft edges
  then apply per product partial. The sideband-DC gap of §3 closes when
  the envelope's modulator is itself a family. Gate: the product-to-sum
  identity, pointwise.
- **S5, warps inside partial phases.** Lift the warp refusal: `normalize`
  slides a warp into each partial's θ, and the envelope and edges read
  θ′. `f0` can then move (a phase-modulated family). This needs θ′ for
  arbitrary arrow terms: autodiff, or a finite difference of the closed
  form at τ ± h. Random access makes the second legitimate.
- **S6, WebGPU.** The family's phase is already `u32`-shaped: `P` is the
  fundamental's phase with the clock's fraction bits kept, and partial
  `k` is `(k·P >> F) mod 2³²`. A WGSL evaluator needs the host's
  block-start `P` and `k·P` in wrapping `u32`/`u64` emulation.
- **Cost.** If more partials or voices are wanted, the options are:
  - a bounded fold over `k` (a Clenshaw or rotation recurrence, which
    is an IR node and a sibling of `bankSum`);
  - wasm SIMD across partials (the accumulation order is the obstacle;
    a pairwise-tree order would have to be the authored order);
  - skipping the edges for partials a stage-0 bound proves in band;
  - folding a uniform envelope into stage-0 weights.

  None of these was needed to meet the 25% line.

## 8. Playground

The family and the envelope are served as two playground node kinds,
`harmonic` and `envelope`. Each maps one to one onto the term algebra:
a `harmonic` node lowers to one family term, an `envelope` node wraps
the term arriving on its `in` wire. The playground lowering carries the
upstream term across a wire (`lowerNode` in `EmitArrow/Patch.lean`), so
the envelope sees the family as the algebra sees it. It never asks what
kind of node is upstream and never reads another node's params.

### `harmonic`

Partials `k = 1..K` at `k·f0`, with weights
`w_k = base_k · k^(−tilt) · (k even ? even : 1)`.

| port | discipline | notes |
|---|---|---|
| `freq` | uniform inlet, anchored knob (110 Hz) | `f0`. A wire must be constant within a kernel call (a knob, arithmetic on knobs). |
| `tilt` | any-wire inlet, glided knob (1) | The spectral slope. With no table, tilt 1 is a saw, 2 a triangle-like slope, 0 flat. |
| `even` | any-wire inlet, glided knob (1) | The even partials' gain relative to the odd ones. 0 with tilt 1 is a square. |
| `partials` | uniform inlet, raw knob (64) | The live partial count, clamped to the capacity. |

Structural fields in `params` (not slots; a change recompiles and
hot-swaps):

- `partials_max`: the capacity `K` (default 64, at most 32767).
- `wave`: `sine` (`Σ w_k sin kφ`, the default) or `cosine` (zero-phase
  cosines).
- `table`: an optional literal base table `base_k`, one number per
  partial. Without it `base_k = 1`. The table is the node's own
  structural data, not a signal: making it live would need table inputs
  in the kernel port, which is out of scope.

The output is an ordinary signal. It can be heard, scoped, mixed, warped
and modulated like any other.

### `envelope`

| port | discipline | notes |
|---|---|---|
| `in` | signal inlet (fan-in sums) | Must be a family, a mix of families, or a constant scaling of one. |
| `cutoff` | any-wire inlet, glided knob (800 Hz) | An LFO at any rate, slow to audio, may drive it: filter FM. |
| `q` | any-wire inlet, glided knob (0.707) | Any wire, as `cutoff`. |

Structural field: `shape`, one of `lowpass` (default), `bandpass`,
`highpass`. An unknown shape is an error, not a fallback.

### What composes

- `harmonic → envelope`.
- `mix(harmonic, harmonic) → envelope`: the envelope distributes over the
  sum, one gain per family.
- `ring(harmonic, knob) → envelope`, a VCA by a knob: a product whose
  other factor is stage 0 (knobs and arithmetic on knobs) is a constant
  scaling, which the envelope commutes with.
- `harmonic → envelope → anything`: the envelope's output is a signal. A
  delay, flanger or FM after the envelope reads the whole enveloped
  signal at its warped clock.
- Envelopes in series on one family multiply.
- A delay or warp of a bare `harmonic` is fine. The family is a function
  of its clock.

### What is refused

The compile fails with a message that names the envelope node, its `in`
wire, and what was found:

- a plain oscillator (`sine`, `saw`, `source`, …): its partials are not a
  family the algebra can see;
- a `ring` of two signals that vary over time (two harmonics, a harmonic
  and an LFO): a product of families, S4;
- a VCA by a time-varying signal: the same product, S4;
- a delay, flange, comb, reverse or FM between the envelope and the
  family: a warp inside the partial phases, S5;
- a constant, a realized modal node, or any other pointwise map.

### How the envelope sees the family

The playground lowering is one recursion over the wires: `lowerNode`
(`lean/Tropical/EmitArrow/Patch.lean`, the `effectC` case at line 848)
lowers each input wire to the upstream node's *term* and hands the terms
to the node's body. The `envelope` body
(`lean/Tropical/Playground/Decode.lean:347`) is
`ArrowTerm.envelope law [cutoff, q] inTerm`, and the `harmonic` body
(`Decode.lean:331`) is `ArrowTerm.familyC`. `normalize`
(`EmitArrow/Term.lean:170`, `envelopeInto` at line 142) distributes the
envelope over sums and scalings, and emission (`emitTermE`, line 243)
either reaches a family (`familyC` at line 318) or refuses. Nothing reads
an upstream node's kind or params.

Two pieces of core were added for this:

- `ArrowTerm.familyC`, a family whose fundamental, live count and weight
  inputs are signal terms, emitted in the family's clock context like
  `genC`'s controls. Wired freq and partials are checked uniform; tilt
  and even take any wire.
- Under an envelope, a product with one stage-0 factor is a scaling
  (`emitTermE`'s `prod` case, line 275). The factor is found structurally
  (`ArrowTerm.isScalar`: constants and sums, products and maps of them)
  and then confirmed stage 0 on its emitted value. That is the knob VCA.
  Any other product is refused.

The weight law is `PartialWeights.closedForm`: K-entry structural
columns (`1/k` and `ln k`, the latter in exact arithmetic) and per-sample
inputs (tilt, even) that ride the region's loop-invariant columns. Without
a table `k^(−tilt)` is computed as `(1/k)·exp((1 − tilt)·ln k)`, so tilt 1
is the saw's `1/k` bit for bit. It costs one `expSig` per partial per
sample. The playable patch (128 partials, the animated lowpass) runs at
4.8% of real time in wasm under bun at 44.1 kHz.

### Gates

- `playground-envelope-equiv` (tropicaltest): the static graph is
  bit-identical to `SpectralLowpassSaw`'s body at τ = 0 and ten hours in.
  With the cutoff wired from an LFO patched out of playground nodes, it
  matches the oracle closed form at −179 dB NMSE (τ = 0 and one hour).
- `playground-envelope-algebra` (tropicaltest): the four refusals, each
  naming the node and wire, and the three acceptances. The mix is bit for
  bit the mix of two enveloped harmonics, and the knob VCA is bit for bit
  0.5 × the enveloped harmonic.
- `graph_wasm_vs_jit.test.ts`: WASM ≡ JIT on a graph using every port of
  both kinds, at τ = 0 and ten hours in. The typed split equals the
  unsplit kernel bitwise. Harmonic → envelope through the graph front
  door equals the registered `SpectralLowpassSaw` through the program
  front door, bitwise.
- `graph_freq_inlets.test.ts`: `harmonic.freq` and `harmonic.partials`
  refuse time-varying wires. The pitch, measured by zero crossings,
  is 220 Hz at frame 0 and ten hours in, and 330 Hz after a knob write.
- `runtime_params.test.ts`: the player's anchored re-pitch of a harmonic
  is continuous ten hours in.

### Findings

- **The typed stage-0 split hoisted loop-index arithmetic.** The
  intern-time stage treats `loopIdx` as stage-neutral, so `toFloat(k + 1)`
  is `fold` by value. The split's per-instruction walk duplicated it into
  the coefficient stream, where its loop-reading operand has no def and
  reads as zero. Every family built at the root rendered silence through
  `render-graph` and `compile-wasm`. The fixtures never hit it, because
  their arithmetic sits inside an instance. The fix
  (`lean/Tropical/Ir/Stage0.lean`, `loopTainted`) pins any instruction
  whose reaching defs read `loopIdx`, transitively. Whole-region moves
  are unchanged. No golden moved.
- **A glide's discipline is not on any kernel port.** `Kernel.ofPlan`
  attaches a discipline to the port named by its base. A glided knob has
  no base port, only `#v0/#v1/#t0` companions, so a host that reads only
  the kernel type cannot see the glide. The `typed-kernel` gate asserts
  that every discipline names a port, so a corpus patch with a glided
  knob fails it. The web player therefore exposes `knob` nodes (raw or
  anchor), and the shipped patch wires every inlet it plays from one. Attaching the glide to a companion, or
  giving it a port, is a kernel-manifest change, out of scope here.
- **Live tables.** A `table` is structural. Making it live needs table
  inputs in the kernel port (`role: table`), which is out of scope.

### Accepted artifacts

- **Aliasing at high Q under fast modulation.** A sharp resonance swept
  quickly past the low partials puts a near-impulsive gain spike on each
  crossing. Its sidebands can fold. This is decision 5 (§2, §5); there is
  no oversampling.
- **DC from sidebands.** An animated envelope makes sidebands at
  `k·f0 ± n·fm` that the algebra cannot see. One that lands on 0 Hz
  carries DC. The soft edges act on carrier frequencies only. This is
  not removed until S4 (§3).
- **Quasi-static gain.** The envelope is a filter shape evaluated per
  partial at the current τ, not filter dynamics: no ringing, no smear
  (§3).
