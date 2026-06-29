# Voice-source absorption & shared-source semantics — design fork (for architect review)

**Status:** REFRAMED + down to one spike (2026-06-28 discussion). The design
*direction* is settled — **sharing by identity, not exclusive consumption.** The
source consumed as a voice stays a normal instance; the diagonal and the
single-tap absorption both fall out of **hash-cons on `(closed-form, clock)`**
("equal things are equal"), with no special-case deletion. Exclusive consumption
(Option 1) was an implementation reflex, not the right default (a voice has
multiplicity ω — effects tap it many times).

This leaves exactly **one empirical, non-design question** that decides whether
the clean path is implementable now and whether it byte-matches `FlangeSin`:
**does the emit layer dedup an identical closed form at the same clock ACROSS
instance boundaries?** (A standalone osc at `clk` vs. a flanger's dry tap at
`clk` → one eval or two?) If yes → the source stays, hash-cons gives byte-
identity, nothing is special-cased. If no → fall back to explicit exclusion-from-
lowering (Option 1) for the MVP, with cross-instance hash-cons as the proper fix.
**A spike is running to settle this** (read the CSE/emit granularity + probe a
shared-form session). The "architect design session" this doc once implied is no
longer needed — it is a code question, not a semantics one.

---

_Original fork (for provenance):_

## The problem (verified)

When `osc → flanger.v` is wired, the osc is **consumed as a voice**: the
desugared Flanger builds its *own* internal `__v_*` instances of the osc's
program at the warped clocks (`v(clk)`, `v(clk±δ)`) and sums them. For
byte-identity with `FlangeSin` — which has exactly three internal oscillators
and **no standalone osc** — the wired source osc must **not** also be emitted
standalone.

But "absorb the source" is not what the obvious mechanism does. **Absorption via
`graphOutputs` filtering only suppresses *sinks*, not *computation*:**

- `emitSinks` (`Compile.lean:529-537`) creates a `SinkSpec` only for instances
  in `graphOutputs`. An osc wired into `flanger.v` (not to the dac) already gets
  no sink.
- **But every instance still becomes an `InstanceFunction`** (`Compile.lean:436,
  604`) and is **computed every sample**, sink or no sink.

So leaving the standalone osc in the session instance list both (a) wastes a
kernel and (b) **breaks byte-identity** with `FlangeSin` (which has no such
instance). Real absorption requires **excluding the consumed source from session
lowering entirely** (`sessionToParsed`'s instance list, `Lowering.lean:362-392`),
not merely sink-filtering. There is **no existing "consumed/absorbed" flag** on
an instance (`Plan.lean` `InstanceFunction` has no such field).

## The fork

The hard part is not the single-consumer case — it is: **what if the osc also
feeds another consumer** (the dac directly, or another instance's input)?

### Option 1 — exclusive consumption + error on shared (the clean MVP)

A source consumed as a voice is **removed from lowering entirely**; if it has
**any other consumer**, the wire is an **error** ("a voice source is consumed
exclusively"). Simplest graph operation; guarantees byte-identity; matches the
handoff's own "absorbed, not emitted standalone."

**Con:** it **forbids the diagonal** — and the diagonal is exactly what Phase 2
wants to celebrate ("two flangers on a shared sine, different δ ≡ two
independent"). Under Option 1, a shared sine is illegal, so Phase 2's headline
golden can't even be expressed. It is a temporary simplification that the very
next phase has to walk back.

### Option 2 — remove-if-unshared, else duplicate

Exclude from lowering when the source has no other consumer; otherwise keep it
standalone **and** inline it as a voice, relying on emit-layer hash-cons to
dedupe the overlap. Avoids the error, but introduces a representation split
(sometimes absorbed, sometimes not) and leans on a dedup mechanism that may not
exist yet (see below).

### Option 3 — never remove; lean on hash-cons (the vision-aligned one)

Always keep the source as a normal instance; rely entirely on hash-cons over
`(closedform, clock)` to dedupe it against the desugared warped taps. This is
the option the **plan's stated cost model already describes**:

> "you pay one evaluation per distinct `(cone, clock)`; … hash-cons dedups
> coincidences." (`voice-ports-plan.md:38-42`)

Walk it through. The desugared 3-tap flanger evaluates the osc's closed form at
`clk`, `clk−δ`, `clk+δ`. A standalone osc consumed at the **ambient** clock is
evaluated at `clk` — which **coincides with the dry tap** `v(clk)`, so hash-cons
folds them into one eval. The `±δ` taps are distinct clocks → distinct evals.
**This is the correct sharing, derived for free** — and it generalizes to the
diagonal (two flangers / shared sine / different δ: the shared dry tap dedups,
the warped taps don't). Option 3 is the only one where the diagonal law falls
out of the cost model rather than being special-cased.

**The catch:** does hash-cons on `(closedform, clock)` **actually exist** in the
emit layer today, working **across** the standalone-instance / desugared-tap
boundary? The recon did **not** confirm it. If it's aspirational, Option 3 is
not implementable now and may not byte-match `FlangeSin` (the standalone osc's
`InstanceFunction` would still emit unless CSE removes it). **This is the
load-bearing unknown for this fork** and should be resolved first.

## The deeper tension (the real reason this needs you)

The MVP wants **exclusivity** (Option 1: simplest, byte-exact). The **vision
wants sharing** (Option 3: the cartesian diagonal, the cost model, Phase 2's
golden). These pull opposite directions:

- Pick Option 1 and the MVP ships fast, but Phase 2 has to **reverse** the
  "voice sources are exclusive" rule to express the diagonal — a churn cost and
  a conceptual walk-back.
- Pick Option 3 and the semantics are right from the start, but it **depends on
  a dedup mechanism whose existence is unconfirmed**, and the byte-identity gate
  with `FlangeSin` becomes contingent on CSE behavior rather than on graph shape.

The question is whether to **buy byte-identity now with a rule you'll delete**
(1), or **invest in the sharing substrate now** (3) so the diagonal is free and
the MVP gate is just its degenerate single-consumer case.

## Recommendation (for discussion)

- **First, settle the load-bearing unknown:** does `(closedform, clock)`
  hash-cons exist and work across the instance/tap boundary? (One focused
  spike: wire a sine to a flanger *and* to the dac; inspect the emitted plan for
  one vs. two evaluations at the ambient clock.)
- **If hash-cons exists:** prefer **Option 3**. The MVP gate is then the
  single-consumer special case, no rule gets walked back, and Phase 2's diagonal
  is free. This is the categorically honest choice.
- **If it doesn't:** ship **Option 1** for the MVP gate, but **scope it
  explicitly as temporary** — write the Phase-2 diagonal requirement into the
  exclusivity error message so the walk-back is visible, not silent.

## Interaction with the unbound-voice decision (Q4)

Willis's stance on unbound voices: **graceful exclusion, and unwired islands are
NOT warnings** — legal-but-incomplete session states compile to silence, never
to a warning that could confuse an MCP agent mid-sequence. That principle bears
on this fork: under Option 1, wiring a *second* consumer onto a voice source
must surface as a **hard error** (it is genuinely illegal), but adding the osc
*before* wiring it into the flanger (a transient island) must be **silent**. The
two must not be conflated — "not yet consumed" is legal and quiet; "consumed
twice" is an error. Option 3 dissolves this entirely (no consumer count is ever
illegal).

## Questions for the architect

1. Does `(closedform, clock)` hash-cons exist in emit today, across the
   standalone-instance / desugared-tap boundary? (Resolve before choosing.)
2. Is buying byte-identity now with an exclusivity rule you'll delete in Phase 2
   acceptable, or should the sharing substrate be built first?
3. Should a voice source ever be **legally shareable** (Option 3 / the
   diagonal), or is "the voice IS the source, consumed" an exclusive ownership
   that the language should enforce (Option 1)?

## Anchors

- `Compile.lean:529-537` emitSinks (graphOutputs filter); `:436,604` every instance → InstanceFunction
- `Lowering.lean:362-392` sessionToParsed instance list (where exclusion must happen)
- `Plan.lean` InstanceFunction (no consumed/absorbed field)
- `voice-ports-plan.md:38-42` the cost model (hash-cons on (closedform, clock)); `:90-94` Phase 2 diagonal
- `design/voice-admission-handoff.md` — the companion fork (how the program is admitted at all)
