# The Tropical User's Manual — project handoff

Status: **pivot decided 2026-07-10.** The `design/writeups/` explainers stop
being a set of standalone articles and become chapters of one book: **The
Tropical User's Manual.** This document is the source of truth for that
book — the vision, the decisions from the conversation that produced it, the
per-chapter briefs, the editorial + figure conventions, and a set of
parallelizable work packages. Read the top three sections before touching
any chapter; then take a work package.

---

## 1. What this is, and what it is not

It is a **manual**, not a manifesto. The distinction is load-bearing and it
sets the voice of every page:

- A *manifesto* argues a thesis; the reader nods and moves on.
- A *manual* rewires how the reader works. It is patient, it builds, it
  assumes little, and it *earns* its punchlines over pages instead of
  asserting them.

The manifesto energy (dense, declarative, convicted) is the through-line,
but the genre is the manual. When in doubt, teach — don't argue.

The one lesson the whole book teaches is to **put on functional glasses**:

> Every signal is a closed-form function of a time coordinate,
> `f(τ, params)`. What you were trained to call *state* is not fundamental —
> it is a *reading* of that function (the function and its derivatives
> sampled at an instant). Once you see a signal this way, scrubbing,
> reversal, rate-independence, and reading the future stop being features
> you build and become things that **fall out** of the representation.

"Scrubbing falls out" is the kind of line you may only write once you have
*earned* it — built the reader up so it lands as inevitable, not asserted.
That is the whole reason this is a book and not an essay: earning a line
over a hundred pages is what chapters are for.

## 2. The honest boundary (plant it in Chapter 1, restate it per chapter)

The glasses work for systems that **have** a closed form. That is a huge,
rich class — all of linear synthesis, and more than you'd guess (see the
pendulum). They do **not** work where no closed form exists: chaotic
dynamics, and the DSP-specific case, **recursive feedback on live input** —
where the input is a future you do not have, so no `f(τ)` can exist. That is
the *ceded island*, deferred to a future stateful sister runtime.

Stating the limit up front is what makes the book serious instead of
overclaiming. Every chapter names its own edge in a sentence or two. Do not
sell past the boundary; the first reader who finds an unstated limit stops
trusting the rest.

## 3. Structure — three parts (do not fake a linear spine)

The two ends are a real dependency spine; the middle is a **gallery** of
siblings, not a staircase. Flanging, reverse reverb, the scope, and
rate-independence are the *same move* applied to four phenomena — order-
flexible, not sequential. Name this honestly or the book bloats to justify
its numbering.

**Part I — Put on the glasses** (the real load-bearing sequence; each needs
the last)
1. Oscillation without accumulation — *the pendulum* (NEW)
2. The sawtooth is a line (existing draft → chapter reframe)
3. An echo from the future (NEW; extracted precursor)

**Part II — What falls off the shelf** (a room of worked examples you can
walk in any order; each is *light* because Part I did the foundations once)
- Through-zero flanging (existing → rework)
- Reverse reverb (existing → rework)
- The oscilloscope / phase alignment by anchoring (existing → light rework)
- There is no control rate (existing → light rework)
- Convolution as a sum of affine warps (candidate, new)

**Part III — The far side** (the harder mathematics, reachable only from
here)
- Modal recombination, residue algebras (outline only; sourced from the
  modal-island / two-realization-reverb work — see the project memory)

Everything already written is Part II *in draft*, and none of it is wasted.
It gets *lighter*: once Part I establishes "a signal is `f(τ)`" once, each
application chapter stops re-deriving the foundations and spends its whole
length on its own phenomenon.

---

## 4. Chapter briefs

### Ch 1 — Oscillation without accumulation (the pendulum) · NEW

The gentlest possible on-ramp to functional glasses, on the most familiar
object in physics.

- **The stateful instinct.** A pendulum is "an object with state" —
  position and velocity `(θ, θ̇)`. You simulate it by *Euler-integrating*
  `θ'' = −ω²θ`: hold `(θ, θ̇)` and step forward. Everything bad follows from
  the stepping — you can reverse it (flip `dt`) but it is *sequential*; you
  cannot land on `t = 10⁶` without a million steps; and each step accrues
  numerical error, so even going forward it slowly drifts off the true
  motion (Euler famously *adds energy* to an oscillator — it spirals out).
- **The functional inversion.** `θ(t) = A·cos(ωt + φ)`. Build the chapter
  toward this sentence: *the state you thought was fundamental was never
  fundamental.* Position and velocity at time `t` are just `θ(t)` and
  `θ'(t)` — the function and its derivative sampled at an instant. State was
  a *reading* of the function, not the thing. Now jump-anywhere, reversal,
  and no-drift are not features; they are what is left when you stop
  simulating and start evaluating. **Scrubbing falls out.**
- **The boundary, drawn in full here** (the pendulum is the only object that
  shows all of it in one breath): small-angle → a clean cosine, functional;
  large-angle → *still* closed-form (Jacobi `sn`, an elliptic period) — the
  glasses survive nonlinearity, a surprising and reassuring thing to show
  early; drive it and damp it → chaos → no `f(t)`, only integration, the
  very thing you were escaping. That last regime is the ceded island; its
  DSP form is recursive feedback on live input.
- **Figure ideas** (worker's discretion, within the house aesthetic):
  (a) phase space `θ` vs `θ̇` — the true motion is a closed orbit, the Euler
  simulation spirals outward (energy drift), and "a state" is just a *point*
  on the orbit, reachable directly by evaluating at `t`; (b) the true cosine
  vs the drifting Euler staircase over time (can't jump, accumulates error);
  (c) the three-regime boundary as three phase portraits: closed ellipse →
  closed distorted orbit → chaotic tangle.

### Ch 2 — The sawtooth is a line (existing → chapter reframe)

Now that it *follows* the pendulum, it can assume "signals are functions"
and focus on its real subject: the **phase**. The phase is the sawtooth;
the classic waveforms (triangle, square, sine) are memoryless shapings of
that one ramp; the accumulator is the state you no longer need. Existing
figures stay (`accumulator`, `line-mod-one`, `shaping-family`). Lighten the
opening — it no longer has to argue closed-form synthesis from scratch; the
pendulum did that. Trim the standalone throat-clearing and the closing
series-glue.

### Ch 3 — An echo from the future (NEW; extracted precursor)

This is the temporal twin of Ch 2's atom, and it was *inside* the flanger
article wearing a costume. Extract it. It owns one principle:

> A delay reads the *past*: `x(τ − d)`, `d > 0`. A causal stream can read
> only the past — it has never held a future sample. A closed-form signal
> has no such restriction: `x(τ − d)` is defined for negative `d` too, and a
> negative delay reads the *future*. To fake a forward read, a stream must
> delay everything by the lookahead — latency equal to how far ahead it
> reaches. A closed-form kernel pays nothing.

An "echo from the future" — a sound whose echo arrives *before* it — is the
concrete image. Both Part II effects instantiate this: through-zero flanging
needs the copy to *lead* (a negative relative delay), and reverse reverb is
the whole tail doing it at once.

- **The figure** (this fixes the degeneracy that started the whole
  restructure): a **single** signal and its pre-echo — a concrete,
  *directional* waveform (use a lowpass-filtered ramp so time has a visible
  direction, per the aesthetic note below) with a copy of it landing
  *before* it, in the streaming-vs-closed-form two-panel grammar. **Not**
  the old symmetric two-tap. Why the old one was wrong: `d_A = D + m`,
  `d_B = D − m` are exact mirror images, which read as *a thing and its
  negation* — the signature of an inverter, not of two independent delay
  lines — so the eye is pulled toward cancellation semantics that have
  nothing to do with the point. The principle needs no symmetry: one signal,
  one forward tap. This figure absorbs and replaces the old `causality-wall`.

### Through-zero flanging (existing → rework)

- **Keep** the rewritten opening section ("Two copies, slightly apart") — it
  is the *model* for how to open any chapter (see §5). It leads with the
  physical picture (crest-on-trough cancels, crest-on-crest reinforces) and
  drops the transfer function entirely; the math moved to where it does real
  work. Keep the `comb` and `notch-fan` figures.
- **Remove** the causality-wall figure and its section — that is now Ch 3.
  Replace with a one-paragraph cite: "the copy has to *lead* the original,
  which means reading it from the future — see Ch 3." The flanger chapter
  then spends its length on the flange, not on causality.
- **Add** a diagram in the "Nothing clicks" spot: the streaming failure it
  describes, shown next to ours. A modulated *delay line* zippers — updated
  at block boundaries and/or without interpolation, the read pointer jumps,
  and the output has audible steps/discontinuities. Show that stair-stepped
  output beside the closed-form output, which is smooth because the delay is
  a continuous argument and the phase is computed from `τ`, not latched.
- **Cut** the "moat" section entirely. No grand "this is an instance of the
  general addressability moat" editorializing. The diagrams speak; end on a
  quick word or two of conclusion.

### Reverse reverb (existing → rework)

- Cite Ch 3 for the forward-read principle instead of re-deriving it. Keep
  `pre-swell` and `impulse-mirror`. The `latency-wall` figure becomes the
  reverb-specific point — *the lookahead here is the entire tail, seconds*,
  the dramatic specialization — either a trimmed keep or folded into the Ch 3
  cite. Own content: the IR mirror, the anticipatory pre-swell, whole-tail
  latency, workspace-≠-state.
- Cut the moat editorializing → quick conclusion.

### The oscilloscope / phase alignment by anchoring (existing → light rework)

- Chapter framing: the scope is a *consumer* that reads `f(τ)` at chosen
  coordinates; anchoring = *solve* for the phase, don't *search* for it.
  Grounded in real engine primitives (`render_window`, the `FixedPhasor`
  exact-integer Q0.32 phase, `relClockQ = clk − anchor·2³²`). Keep `trigger`,
  `anchor`, `lock-timebase`. Trim the closing moat paragraph → quick
  conclusion.

### There is no control rate (existing → light rework)

- Chapter: rate is a property of the *reading*, not the signal (the Serge
  no-rate-hierarchy refusal — this is a firm project stance). Keep
  `rate-hierarchy` and `many-consumers`. Trim any moat glue → quick
  conclusion.

---

## 5. Editorial conventions (every chapter)

- **Open on the concrete.** Lead with the physical / sensory / familiar, not
  the formula. The model is the flanger's rewritten §1: it explains the comb
  through waves cancelling (crest-on-trough), and the transfer function that
  used to open it is simply gone — moved to the one section where it does
  load-bearing work. Introduce a symbol lightly, in a phrase, only when a
  later section will lean on it.
- **Earn the punchlines.** Build to "scrubbing falls out" / "state was a
  reading of the function" — don't assert them cold.
- **No moat sermons.** Cut the grandiose "one fact read three ways / the
  general addressability moat / same fact that makes τ-scrub free"
  editorializing. It is pretentious and beside the point. The diagrams speak
  for themselves; a quick word or two of conclusion is enough.
- **Name the boundary** (the ceded island) in each chapter, briefly.
- **No emojis.** Unicode math (`τ ≡ → ∘ λ`) is welcome; pictographs never.
  GitHub-flavored LaTeX (`$…$`, `$$…$$`) for load-bearing equations only.
- **No rate hierarchy** as a design primitive (Serge).
- Cross-reference by chapter, not by re-derivation. Foundations are
  established once (Part I) and *used* thereafter.

## 6. Figure production (every figure)

- **Aesthetic:** narrow grayscale lines, Computer-Modern serif, high
  data-ink / Tufte. Vector SVG, glyph outlines embedded as paths (portable),
  deterministic output.
- **Tooling:** matplotlib + numpy in a venv (the repo ships no python):
  `python -m venv venv && venv/bin/pip install matplotlib numpy`. Shared
  style module `design/writeups/figures/_style.py` (`apply_style`, `finish`,
  `tidy`, `boxed_legend`, palette `K/G/L/FILL`; `svg.hashsalt` pinned +
  `metadata={"Date":None}` for byte-reproducibility). One
  `make_<topic>_figures.py` per chapter, imports `_style`.
- **cmr10 gotchas** (learned the hard way): the font has **no em-dash**
  (U+2014) — never put `—` in figure text (titles/labels render in cmr10,
  not mathtext); mathtext has **no `\bmod`** (use `\ \mathrm{mod}\ `) and
  **no `\tfrac`** (use `\frac`). Break vertical discontinuities (phase wraps,
  square-wave edges) by inserting `NaN` so no seam draws.
- **QA loop:** render each figure to PNG in the scratchpad and *look at it*
  (label collisions, overlaps, legibility) before writing the committed SVG.

## 7. Repo state

- Worktree: `/Users/willishoke/tropical-writeups`, branch `docs/writeups`
  (off `origin/main`). Parallel workers operate here.
- Committed: five draft chapters + figures + tooling (`7f3d0df … f17470e`):
  `sawtooth-is-a-line`, `through-zero-flanging`, `reverse-reverb`,
  `phase-anchoring`, `no-control-rate`.
- **Uncommitted working change:** the through-zero-flanging §1 rewrite
  ("Two copies, slightly apart") — keep it.
- **Open decision — directory layout.** The book wants an obvious order.
  Options: a `manual/` tree with `NN-title.md` chapters (numeric prefix =
  reading order), or keep files flat with a `SUMMARY.md`/`README.md` that
  fixes the order. Recommend the numbered `manual/` tree; decide before the
  rework WPs move files. Figures can stay under `figures/` shared.

## 8. Work packages (parallelizable)

Each is a self-contained handoff. WP1, WP2, WP5, WP7 are independent and can
start immediately. WP3/WP4 reference Ch 3, so either sequence them after WP2
or draft against a stub cite and reconcile.

- **WP1 — Ch 1, the pendulum (NEW).** Write the chapter per the brief in §4;
  build its figures (phase-space + drift + boundary ideas). The flagship new
  chapter; give it the most care.
- **WP2 — Ch 3, an echo from the future (NEW).** Write the chapter; build the
  single-pre-echo two-panel figure (absorbs the old `causality-wall`).
- **WP3 — rework the flanger chapter.** Remove causality-wall (→ WP2), add
  the zipper diagram, cut the moat, keep the §1 rewrite and the comb /
  notch-fan figures.
- **WP4 — rework reverse reverb.** Cite Ch 3, resolve the latency-wall
  (trim/keep/fold), cut the moat.
- **WP5 — Ch 2, the sawtooth.** Light chapter reframe (assume the pendulum;
  lighten the opening; drop series-glue).
- **WP6 — the scope + no-control-rate.** Light chapter reframes; trim moat
  paragraphs → quick conclusions.
- **WP7 — front matter.** The book's opening (what a manual is; the
  functional-glasses promise), the table of contents, the three-part
  framing, and the directory layout decision (§7).
- **WP8 — Part III outline (later).** Sketch the residue-algebra / modal-
  recombination arc from the modal-island work; not yet drafted.

Consistency guarantee for parallel work: §5 (editorial) and §6 (figures) are
the shared contract. A worker who honors both produces a chapter that reads
as part of one book.
