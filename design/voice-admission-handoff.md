# Voice-port admission — design fork (for architect review)

**Status:** DECIDED 2026-06-28 in discussion (the fork below is recorded for
provenance, but the question is settled — no architect session needed).

**Decision: voice-bearing programs are TEMPLATES (Option B, reframed as
principled).** A program with a `voice` port is parsed and stored at
`define_program` but **not core-elaborated** there; it is elaborated *per
binding* at `syncCompile`, after the wire-derived `VoiceDesugar` produces a
concrete (uniquely-named) form. The justification is not "dodge the checker"
but: **a voice port is a binding-site, dual to a type parameter.** Generics
already defer this way (`Delay<N>` is specialized before it is elaborated for
codegen); a voice is the same shape — bind first, elaborate second — so the core
elaborator never needs to understand `voice` at all. The `voice(out: float)`
**bound survives as interface metadata** for cable-UX validation (the principal
type the row-inference would derive — keeps the parametricity argument), but the
core IR never carries a `voice` type. A1 (voice in the IR) and A2 (recognized-then-
erased PortType) are therefore both unnecessary; C (generics) was rejected
earlier. Residual narrow, non-blocking question: should the bound be a *checked*
type at the wire boundary (validate the wired source's output signature against
it) or an informal annotation? Defer; it is a small decision, not a fork.

---

_Original fork (for provenance):_

## The problem (verified, not hypothetical)

The landed voice-ports work (PR #194) proves the **file pipeline**:
`parse-md flanger.md | voice-desugar | emit-file` is **byte-identical** to the
hand-wired `FlangeSin`. That path works because `VoiceDesugar.desugarProgram`
**strips the `voice` input port before anything elaborates** — the elaborator
never sees the `voice` type.

The **session/MCP path elaborates eagerly and at the wrong time**, so the same
program cannot even be *registered*:

- `define_program → registerOne` (`Engine.lean:557`) calls
  `Tropical.Ir.elaborateInto` on **every** program at registration. Failure →
  `internalError`. (Generics defer only *strata*, not elaboration —
  `Engine.lean:561-564`.)
- `resolvePortType` (`Elaborator.lean:232-242`) on a scalar port type name that
  is neither a builtin nor an in-scope alias throws
  `throwElab "unknown port type 'voice'"`.
- `builtinTypeToScalar?` (`Elaborator.lean:158-166`) has **no `voice` case**
  (verified: float/int/bool/signal/freq/unipolar/bipolar/phase/clock/time only).

So `define_program Flanger(v: voice, …)` dies at registration, **before any
instance exists and long before the `osc → flanger.v` wire can supply a
binding.** The handoff's deferral story is pitched at *instance resolution*
(`resolveInstanceMeta`, ~600-680); the real first wall is one level earlier, at
*registration*. Any wire-binding implementation must clear this wall first.

## The fork

How does a voice-bearing program become a legal session citizen?

### Option A — first-class `voice` PortType (the rigorous one)

Add a `voice` variant to the `PortType` inductive (`Wiring.lean:56-81`,
`Ir/Nodes.lean PortType`), carrying the output signature/bound
(`v: voice(out: float)`). Teach `resolvePortType`, `parsePortType?`, and
`checkArrayConnection` to recognize it. The voice becomes a **real type in the
category** — a port whose inhabitant is a `Clock → Signal` morphism — and the
program elaborates *structurally*, marked unresolvable until a wire binds it.

There is a sub-fork inside A, and it matters more than the headline choice:

- **A1 — voice survives into the IR.** `voice` is a genuine port type that
  flows through elaboration into `ResolvedProgram` as an abstract/unresolved
  port, resolved per-binding. **Heavy:** strata + emit must now reason about a
  port that isn't a scalar/array. Touches the whole post-strata surface. This is
  the version that conflicts hardest with the representation lock and the
  "smallest sub-IR sufficient for any evaluator" framing.
- **A2 — voice is recognized but resolved-away (likely sweet spot).** `voice`
  is a real `PortType` at the **parse/elaborate boundary** — validated, readable
  by the cable UX, the principal type a row-inference engine would derive — but
  it is **desugared/stripped before strata**, exactly as the file pipeline does
  today. The elaborator *recognizes* it (doesn't throw) instead of the desugar
  having to *delete* it pre-elaboration. Strata/emit never see it. Rigorous as a
  type; invisible to the runtime.

**Categorical reading.** A makes the type system express "this port expects a
`Clock → Signal`." That is the principal type the eventual row-inference would
derive, and it is the honest statement of what an effect *is*: a morphism with a
voice domain. It is the option most aligned with the backbone (programs are
typed signal-flow graphs; a voice port is a typed hole). A2 gets that honesty
without paying for it at every post-strata call site.

**Cost.** A1: large, cross-cutting (elaborator + strata + emit + wiring). A2:
moderate (elaborator + wiring recognize a new `PortType`; the desugar already
exists). Both touch `checkArrayConnection` so a `float`-source wire into a
`voice` port type-checks (see the absorption handoff and the wire-interception
note below).

**Tension with the representation lock (commit `a65b078`).** The lock says
"voice ports are sugar over instances; **no new IR node**." A `PortType` variant
is a new *type*, not a new *node* — and under A2 it is resolved away before the
IR proper, so arguably the lock survives. The architect should decide whether a
type that exists only to be erased before strata is principled (a real
interface, like a phantom type) or a half-measure that wants to be either A1
(commit to it in the IR) or B (keep it out of the core entirely).

### Option B — session-layer parsed retention (the pragmatic MVP)

Detect the `voice` port at `define_program`, **skip elaboration entirely**,
retain the `ParsedProgram` in a **new `SessionSt` field** (none exists today —
`Session.lean:102-142` holds `templateByName` arena indices for generics, i.e.
*elaborated* forms, not parsed ones), and synthesize a `ProgMeta` from the
parsed ports so `add_instance`/wire-checking can find `v`. Desugar + elaborate +
strata happen **only at bind time**, once the voice wire lands, producing a
concrete (named-uniquely) program — exactly the proven file-pipeline ordering,
lifted into the session.

**Pro:** elaborator, IR, strata, and the acyclic invariants are **untouched**.
All voice logic is isolated to the session layer. Maximally consistent with the
representation lock (voice never becomes a type anywhere in the core). Smallest
blast radius; fastest to a green wire-binding gate.

**Con:** the session layer grows a second program-representation (parsed,
unelaborated) alongside the existing elaborated catalog — a genuine new state
to maintain. The `voice` "type" lives only as an informal parsed annotation, so
the cable UX reads an un-validated string, not a checked type. It is the *less*
categorically honest option: voice-ness is a session-layer special case, not a
property the type system knows.

### Option C — generics over explicit `<V>` (rejected)

Author effects as generics parametric over a voice type param, reusing
`templateByName`/specialization. **Off the table** unless the pointfree decision
is reversed: the handoff records that Willis deliberately chose pointfree
wire-binding over the explicit-`<V>` surface (`voice-ports-handoff.md:48-50`),
knowing the wire-binding is more work.

## Cross-cutting consequence (true under A or B)

Whatever admits the program, **two downstream walls remain** and the architect
should know they exist when weighing cost:

1. **`handleWire` will reject the voice wire.** The normal set-op path runs
   `adaptInputExpr`/`checkArrayConnection` (`Engine.lean:938`), type-checking
   `float` (`osc.sine`) against the voice port. The voice wire must be
   **intercepted before** type-checking. Under A this can be a `checkArrayConnection`
   rule (`float` is a legal inhabitant of `voice(out: float)`); under B it is a
   session-layer special case keyed off the parsed annotation.
2. **Name collision across bindings.** The desugared program keeps the name
   `Flanger`; `syncCompile`'s resolver table and `registryGet` are **first-wins
   by program name** (`Engine.lean:343, 378`). Two flangers over different
   voices would link to the same resolved type. Voice-bound forms need **distinct
   synthetic names** — the generic-specialization precedent (`Delay<N=8>` display
   keys). This is independent of A vs B.

## Recommendation (for discussion, not a decision)

- For a **fast, low-risk MVP**: **B**. It ships the wire-binding gate without
  touching the IR, and it is the most faithful to the representation lock.
- For the **language Willis actually wants**: **A2**. A `voice` is a real type;
  making the elaborator *recognize* it (rather than the desugar *delete* it)
  states that in the type system, at moderate cost, without polluting strata.
- **A1** only if the architect wants voices to be first-class *at runtime* (e.g.
  a future where an effect is shipped without its voice bound and the binding
  happens in the engine, not the compiler) — that is a bigger bet than piece #2.

## Questions for the architect

1. Is a port type that is **erased before strata** (A2) principled, or does
   rigor demand A1 (voice survives into the IR)?
2. Does the representation lock ("voice ports are sugar, no new IR node") forbid
   a `PortType` *variant*, or only an IR *node*? Is A2 inside or outside the lock?
3. Is the cable UX reading a **checked type** (A) worth a cross-cutting
   elaborator/wiring change, vs. an informal parsed annotation (B) for now?
4. Should the binding mechanism be **compiler-time** (desugar→elaborate at bind,
   B and A2) or leave room for **engine-time** binding (A1) where a voice could
   be (un)bound without recompiling the effect's structure?

## Anchors

- `Engine.lean:557` registerOne eager elaboration; `:343,378` first-wins resolver
- `Elaborator.lean:158-166` builtinTypeToScalar? (no voice); `:232-242` resolvePortType throw
- `Wiring.lean:56-81` PortType inductive + parsePortType?; `:123-161` checkArrayConnection
- `Session.lean:102-142` SessionSt (no parsed-program field); `:84-89` InstanceInfo.resolvedIdx
- `Parse/VoiceDesugar.lean:32-38` VoiceBinding; `:126-134` desugarProgram (strips voice port)
- `design/voice-ports-plan.md`, `design/voice-ports-handoff.md` — the locked representation + sprint
