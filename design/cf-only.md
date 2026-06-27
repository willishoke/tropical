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
"a decaying coefficient"), so it is the first thing to make a type.

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

These are the laws that keep the guarantees true. Each is a one-line
property that wants to be a type or a test, not a convention maintained
by eye.

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
3. **Fixed-point τ.** Bit-exact reversibility is a claim on a grid; float
   is not symmetric under negation out where the exponent is wide. τ is
   derived from the integer sample index via fixed-point arithmetic (the
   `FixedPhasor` pattern), so the involution is exact on the grid for
   arbitrarily long renders — the type's theorem and the ear agree.

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

Make every kernel a pure `f(τ, params)`; guarantee it rather than detect
it; keep damping spatial so reversibility is a group action; build the
future-reading effects from offset reads, not new primitives; cede the
recursive/stochastic island to the stateful sister runtime; and let the
three disciplines (centered operators, bounded-degree shapers,
fixed-point τ) make the guarantees checkable rather than hoped for.
