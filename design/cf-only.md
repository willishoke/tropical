# Closed-form-only: the commitment

This document is the contract for the CF-only rearchitecture. It states
what the language becomes, what guarantees that buys, and the few
disciplines that make the guarantees checkable. Every phase of the
migration (the sprint plan) cites it; where this document and the code
disagree, this document is the intent and the code is the bug.

## The thesis

Every audio kernel becomes a pure function of a time coordinate and a
parameter point:

> **`f(τ, params)`** — no per-sample state. τ is navigable; the
> parameters are held constant while you navigate.

That single shape is the whole commitment, and three properties the
roadmap has wanted separately turn out to be one property read three
ways:

- the scope (any consumer) can look at **any** signal,
- a consumer at a **different rate** (video, control) is correct for any
  patch,
- **random access in time** is guaranteed for every patch.

They are the same fact: *every signal is a closed-form function of the
time coordinate.* Make that a guarantee rather than a detected property
and every downstream consumer becomes uniform — no "is this signal
closed-form?" branch, no fallback limb. Addressability stops being
conditional and becomes the identity of the instrument.

## Not a tape machine — a parallel-universe explorer

Reversibility here is defined **given baked (constant) parameters**: the
fiber at a fixed parameter point is invertible in τ. You do not unwind
the knobs. Scrubbing τ backward holds the parameters at their *current*
values, so what you hear is not a recording of the past — it is the past
**recomputed under the decisions you are making now**. A tape can only
replay what was recorded; this holds the whole bundle of possible
histories and evaluates the one your current settings select. That is
the product, and it is the thing a stream provably cannot do: the past
is a closed form evaluated at an earlier τ, not a buffer.

Two time axes coexist and must not be conflated:

- **τ** — synthesis time, the coordinate the kernel is total and
  invertible over. You navigate it freely.
- **performance time** — the order you touched the controls. Parameters
  live here; they are frozen-constant during any navigation of τ.

There is no stored performance history, no automation sidecar, no
anchor table. The total state of the instrument is the current
parameter point (a handful of numbers) plus τ. Live parameter
automation, if wanted at all, is itself a closed-form function of τ
(authored, and it reverses) — never a recorded gesture replayed.

## What "no per-sample state" means precisely

⚠ **The landmine.** The codegen overloads the word *register*. There are
two unrelated things:

- **SSA temps** (`rgF`/`rgI`/`rgB` in `EmitLlvm`) — computational
  intermediates within a single sample. These are not state. **They
  stay.**
- **`state_reg`** — the per-sample latch with read-old/write-new
  writeback across samples (surface `reg`/`delay`). This is the state.
  **This is what CF-only forbids in the audio path.**

CF-only deletes the second and never the first. Any pass, test, or
deletion that touches "registers" must target `state_reg` only. The
op-coverage test (a temp-only FlatPlan) is the guard: it must keep
compiling in `cfOnly` mode.

The navigable coordinate is the integer sample index `n`, already the
kernel ABI input (`start_sample_index`), already random-access (no
monotonic constraint). τ is a closed-form function of `n`. The host
owns the clock: integrating a live velocity knob into a render position
is the transport's job, outside the pure kernel — the instrument is a
function from a number to a sound; the tape is something the host waves
in front of it.

## Reversible vs total — name the fork, cede it on purpose

Closed-form gives **totality**: the kernel is defined at every time
coordinate, terminating, random-access. **Reversibility is strictly
stronger** — totality plus an involution on the coordinate (τ ↦ −τ).
A damped sinusoid `e^(−ατ)·sin(ωτ)` is total but not reversible: read
backward it grows without bound. Categorically, the full closed-form
ring is a *monoid* action of time (it plays, it's random-access); the
conservative, undamped subring is a *group* action (it scrubs bit-exact
both ways). The reverse-reverb moat lives in the group.

The discipline is therefore: **damping is spatial, not temporal.** Decay
across modes or taps — geometric weights `gᵏ` over taps at `τ+kD` — is a
constant per tap, unchanged under τ ↦ −τ, so it reverses. Decay in the
coordinate, `e^(−ατ)`, does not. You may have resonance, decay, the
whole filter cabinet, as long as the decay rides the mode/tap index and
not τ. This is invisible at the value level (`gᵏ` and `e^(−ατ)` are both
"a decaying coefficient"), so it is the one structural discipline the
reversibility test leans on — checked, not typed (next section).

## Reversibility is tested, not typed — the hard-won negative result

The first instinct is to make reversibility a *type* — forbid the
irreversible by construction, the way statelessness is forbidden by
deleting `reg`. Two adversarial design passes killed both routes, and the
negative result is recorded here so it is not re-attempted:

- **The growth-grade** (tag every value Reversible/Total by its τ-growth)
  is an inlining-defeatable lint. Irreversibility is unbounded growth,
  which is just `×` over an unbounded coordinate — constructible by hand
  from arithmetic, so the grade only catches the *named* `Exp` program,
  not the structure. It also rejected the actual flagship.
- **The compact coordinate** (expose only the wrapped phase ℤ/2³² so
  nothing can grow) removes the precondition instead of tracking it — but
  it cedes aperiodicity and one-shot events, turns the moat's *acausal*
  "future" into a periodic "next lap," bakes the sample rate into the
  lattice (killing the rate-neutral v2 coordinate), and still leaks through
  division/log poles. And decisively: the exactness it advertised never
  came from compactness — `ReversibleProbe` already palindromes bit-exactly
  over **raw, unbounded `sampleIndex`, no wrap, no register**. The grid, on
  the *line*, was always what delivered exact scrub.

So reversibility is a **tested property**: the palindrome test
(`out[half+k] == out[half−k]` over a symmetric τ, bit-exact) is the
guarantee — decidable, undefeatable by inlining, certifying the numerical
property directly — paired with the one checkable structural discipline
both passes agreed on, *damping is spatial, not temporal*. Everything else
is the test's job. The priority that falls out of this is simply
**stateless kernels**; reversibility rides along for free on any pure
function of the coordinate.

Two corrections this forces:

- **Boundedness comes from `clamp`, not from forbidding division.** The
  domain is audio; everything saturates. `clamp` bounds the range
  regardless of div/log/poles, and because it is a pure function of its
  input it preserves equal-coordinate ⟹ equal-output — the palindrome
  still holds. Unboundedness does not persist; the pole objection is moot.
- **Noise is stateless, not a reason to keep `reg`.** Today's `WhiteNoise`
  happens to use a state register, but the sequence is not sacred:
  `hash(sampleIndex)` is a stateless, random-access, reversible noise, and
  the better form is a few **short-period LFSRs memoized as constant
  tables**, indexed by `sampleIndex mod Pᵢ` and XOR'd at coprime periods —
  combined period `∏Pᵢ`, a handful of table reads, pure in the coordinate,
  and it reverses. The register was an implementation detail, not the
  aperiodicity.

## The capability class: the future is computable

The killer capabilities are one fact wearing several hats — *the source
has a computable future*, because it is a closed form, not a recording:

- **reverse reverb** — geometric taps at `τ+kD` (the future), so the
  reverb swells *into* a hit from before it;
- **zero-latency lookahead compression** — evaluate the envelope at
  `τ+Δ` with no delay, because the future samples are computable;
- **zero-phase / acausal filtering, live** — symmetric offset reads
  `τ±Δ` give a linear-phase response with no block latency and no
  pre-ringing tradeoff.

All of these are **offset reads `x(τ±Δ)` of a closed-form source**. None
require a new IR primitive. A stream cannot do any of them without buying
latency, because its future does not exist yet; ours does.

## No new primitives

The exp-poly ring's generators already exist. `sin`, `cos`, `exp`,
`pow`, `log`, `tanh` are **not** codegen primitives — they are stdlib
programs (polynomial approximation with range reduction over the ~27
arithmetic ops). A damped sinusoid is `Exp(...)·Sin(...)`. An IIR pole on
an in-ring signal is the geometric tap series `Σ gᵏ·x(τ−kD)` — exactly
the reverse reverb, already running. Summation-closure of the ring holds
for everything synthesized within it; it fails only for recursion on
signals from *outside* the ring (see the ceded island). So: **add no
primitives until a profiler on real polyphony demands one.** Effects are
stdlib built from offset reads, not new ops.

## The three checkable disciplines

These keep the reversibility test honest — each is enforced by the test
or a lightweight check, not a type (previous section), and not a
convention maintained by eye.

1. **Centered operators.** Reversible ⟺ commutes with the involution
   τ ↦ −τ. Any difference-based operator (derivative, ADAA, one-pole)
   must use the **centered** form (`x(τ+h/2)`, `x(τ−h/2)`), not the
   backward form, or the palindrome breaks at the one-sample level.
   Note the centered form reads the future — free here, impossible in a
   stream: the correct operator for us is the one nobody downstream of a
   buffer can write.
2. **Bounded-degree polynomial shapers.** A degree-`d` polynomial of a
   bandwidth-`B` signal has bandwidth exactly `d·B`. If `d·B < Nyquist`,
   aliasing is exactly zero — a closed-form inequality, not a hope. ADAA
   is the fallback when `d·B` exceeds Nyquist.
3. **Fixed-point coordinate (the grid, on the line).** Exactness lives in
   the integer *grid*, not in wrapping: `ReversibleProbe` palindromes
   bit-exactly over raw `sampleIndex`. Float is in fact already bit-exact
   to ~2⁵² samples (~3000 years at 48k), so this is **low-urgency** — its
   real payoff is *cross-backend determinism* (integer arithmetic is
   bit-identical across JIT and wasm — the reason `FixedPhasor` is integer;
   float can diverge under fp-contract/reordering). Keep τ **unbounded**
   (the line) so the past, the before-zero, and the acausal future taps
   stay real and the rate-neutral v2 coordinate survives; reserve the
   **wrap** (`FixedPhasor`'s ℤ/2³²) for genuinely *periodic generators* —
   oscillators, wavetables, additive — where "long gesture = low-frequency
   phase" is correct.

## The ceded island

The ring is closed for everything synthesized within it and breaks the
instant you recursively filter something from *outside* it: IIR on live
input, IIR on true broadband noise (`Σ a^(−k)·noise[k]` has no closed
form). That is genuine, irreducible state. It is **ceded on purpose** to
a future stateful sister runtime ("supertropical"), where state and live
I/O are first-class. The boundary between total and partial is a design
commitment, not a footnote: tropical is the total language; the partial
half lives elsewhere. The narrow corner of dense *metallic/inharmonic*
clangor (many non-integer partials) is expensive everywhere and is
ceded with it — mild inharmonicity (a few partials, `ModalVoice`) stays
cheap; harmonic richness comes free from waveshaping.

## What this deletes (and why the scope gets simpler)

CF-only is a net deletion. Gone: the per-wire unit-delay wrapping that
breaks session cycles, the delay-slot registry and extraction, the
save/restore round-trip for delays, state-transfer-by-name on hot-swap,
and the MCP `feedback` tool. The session-acyclicity check stays, but
hardens into a plain "no cycles at all" rule.

One payoff closes the loop with the scope work. The scope's `WARMUP`
lead exists only because the MCP wraps every wire in a unit delay to
break session cycles — so "stateless" patches are secretly stateful, the
render window starts cold, and it needs priming. CF-only pulls the wire
delays at the root: `render_window` becomes cold-start-exact, no warmup,
no priming, and the scope sees everything because there is nothing left
it cannot. The strange-scope smell and the cold-start garbage are the
same residual state, removed at the source instead of patched at two
leaves.

## One-line summary

Make every kernel a pure `f(τ, params)` — *that* is the guarantee, by
deleting `reg`; stateless kernels are the priority and reversibility rides
along for free. Keep damping spatial so the conservative scrub is a group
action; build the future-reading effects from offset reads, not new
primitives; keep noise stateless (hash / memoized-LFSR-XOR) and bound
everything with `clamp`; cede the recursive/live-input island to the
stateful sister runtime. Reversibility is the **palindrome test plus the
spatial-damping discipline** — a tested property, not a type (both type
routes were explored and rejected). Fixed-point is the right coordinate
representation for cross-backend determinism but low-urgency; the wrap is
a generator primitive, not the universe.
