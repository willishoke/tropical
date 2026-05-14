# Operadic IR for Tropical

## How this document came to exist

This document captures architectural commitments that emerged from an
extended design conversation. The conversation began with a specific
technical problem — cross-kernel input wiring under M11's fractal
architecture — and traced backward to find the implicit categorical
foundations that had never been written down. The result was a
recasting of tropical's IR as a typed operad. The existing CLAUDE.md
tagline ("cartesian category of typed signal-flow graphs with a
guarded trace") gestures at this framing but never makes it explicit;
this document is that explicitness.

The journey through the conversation, in compressed form: a specific
cross-kernel wiring problem → recognition of the slotified-memory
unification → arrival at operadic encapsulation as the structural
principle → realization that the language *is* the operad → exploration
of cycle-handling options as different operadic realizations →
identification of Wave Digital Filters as the no-compromise structural
option for analog circuit modeling → understanding that multiple
realizations can coexist behind operadic interfaces → glimpse of
tropical as a polymorphic operadic DSL spanning multiple computational
paradigms.

What follows is the architectural picture in declarative form: what
tropical IS, categorically, and what design choices follow from that
commitment.

## The shape of the domain

Tropical models signal flow through bounded, port-typed devices
arranged hierarchically. Eurorack modules patch into other modules
through audio and CV jacks. Synthesizers expose MIDI input and audio
output but encapsulate their internals (LFOs, oscillators, filters)
behind a single interface. Microcontrollers expose their GPIO pins
but hide their internal computation. At every level, devices interact
through explicit port boundaries and never through internal state.

This is the structural fact about the domain — not a design choice we
make about the IR, but a property of what we're modeling. The IR's job
is to faithfully represent this domain's compositional structure. The
categorical name for this kind of compositional structure is a **typed
(colored) operad** — specifically, a colored PROP for handling
multi-input/multi-output operations and parallel composition.

## Tropical is a typed colored PROP

A typed (colored) operad consists of:

- A set of **colors** (port types: scalars, arrays, sums)
- A set of **operations**, each with a typed input signature and a
  typed output signature
- A **composition** rule that lets you plug operations into other
  operations' compatible ports
- A **tensor product** (in the PROP variant) that lets operations run
  in parallel

In tropical:

- *Colors* = post-strata port types (`float`, `int`, `bool`, fixed-shape
  arrays of these)
- *Operations* = primitive operations (`add`, `mul`, `sin`, `delay`, ...)
  plus user-defined programs
- *Composition* = wiring (output of one operation feeds input of another)
- *Tensor* = parallel placement (operations running side-by-side)

Each `ResolvedProgram` in tropical's IR is, viewed correctly, an
operation in this operad. Its `body` is a finite tree representation
of that operation expressed as a composition of sub-operations. The
tree's leaves are primitive operations (or sub-program references);
the tree's internal nodes are operadic plug points.

This recasting is not a reimplementation — it's an articulation of
what the IR has always been. The mathematical structure was always
there; we just hadn't named it.

## Encapsulation is structural, not a discipline

Operadic composition has one rule about what's visible across the
boundary of an operation: **only the port signature**. An operation's
internal definition is categorically opaque from outside. When you
compose two operations, you don't peer into either of them; you only
see how they fit together at their ports.

This isn't a discipline we impose to keep the IR clean. It's a
structural property of operadic composition. Any IR that violates it —
for example, by letting a sub-operation's body reference the
*enclosing* operation's input ports directly — has stepped outside the
operadic category. That's exactly what we saw with the
`cloneWithInputSubst` scope-leak during the M11 fractal-activation
attempt: an attempt to substitute the parent's wired expression into
the child's body produced references that crossed the operadic
boundary. The error wasn't a bug in the substitution; it was an
attempt to do something categorically invalid.

The correct architecture: child operations only see their own input
ports. Cross-program communication happens at the *parent's* level —
the parent's body computes the wired expression in the parent's scope
and writes the result to the child's input port slot. The child reads
its input port slot in its body. Each operation works inside its own
categorically-closed world.

## Pre- and post-trace operads

Tropical's IR has two distinct operadic shapes connected by a known
transformation:

**Pre-trace operad**: a traced colored PROP. Operations can be composed
cyclically — feedback paths in source code translate directly to cyclic
plug patterns in this operad. The trace operator is implicit in cyclic
structure: a cycle through operations means "trace this composition."
This is the user-facing operad.

**Post-trace operad**: a plain colored PROP (no trace operator) with
`delay` promoted to a first-class primitive in the operadic signature.
Every composition is acyclic. Cycles from the pre-trace operad have
become compositions through `delay`. This is the runtime-facing operad.

The transformation `T: pre-trace → post-trace` is a **functor in the
category of operads (Op)**. It is the entire categorical content of
compilation. Within tropical's strata pipeline:

- `specialize` — substitution in a polymorphic operad
- `sumLower` — coproduct elimination
- `traceCycles` — **the functor T**, mapping cyclic operadic
  compositions to delay-broken acyclic ones
- `arrayLower` — combinator unrolling
- `identityElim` — categorical identity law

`traceCycles` is the load-bearing pass. The others are normalization.
This is the categorical reading of compilation: each strata pass is a
named operad morphism; the whole pipeline is a composition of operad
morphisms.

## The trace functor is not unique

Here is where a key design decision lives: **the functor T is not
uniquely determined by the categorical structure.** Given a cyclic
composition, there are infinitely many delay-broken realizations of
it, differing in where the unit delays sit. Each realization is a
valid trace functor; each produces a slightly different concrete
behavior under discrete-time evaluation.

The categorically precise statement: there is an *equivalence class*
of post-trace realizations that all denote the same morphism in the
continuous-time TSMC, but which are not equal as concrete morphisms
in the discrete-time DAG operad. `traceCycles` picks a representative
of that equivalence class. The choice is a language design decision,
not a categorical theorem.

The cycle-handling options:

### A. Strict acyclic (no implicit cycles)

The operad has no trace operator. Users must use explicit `delay`
primitives to express feedback. The compiler never inserts delays.
The user always knows where the unit-delay sample shift sits.

Verbose source; predictable semantics. This is what Haskell-pure-FP
does: separate effects (state) from pure computation.

### B. Universal unit delay between operations

Every wire between operations carries an implicit unit delay. The
post-trace operad is one whose composition operator implicitly
inserts a delay. Trivial implementation; no cycle-breaking algorithm
needed.

Cost: latency accumulates through pipeline depth (a 4-stage filter
chain has 4 samples of delay). This is what VCV Rack and Pure Data
approximate at coarse-grained module boundaries.

### C. Explicit feedback operator (Faustian)

Users mark feedback points syntactically (`feedback (fb => ...)` or
`~`). The pre-trace operad has the trace operator as a first-class
syntactic construct; the compiler mechanically rewrites each
`feedback` to a `delay` insertion. Implicit cycles outside `feedback`
are a type error.

Categorically clean: the user explicitly specifies the trace, the
compiler trivially realizes it.

### D. Implicit cycle-breaking with documented convention

The compiler auto-breaks cycles using a documented rule (e.g.,
Tarjan's natural back-edge selection). Users can override via explicit
`delay` annotations. Convenient; relies on users reading documentation
to predict behavior.

Current tropical sits here, but undocumented. This is the position
that motivated the architectural conversation.

### E. Iterative fixed-point per sample

The trace operator is realized as iterative evaluation: within each
sample, the cyclic composition is evaluated N times, with each
iteration using the previous iteration's slot values as the next
iteration's inputs. As N grows, this converges to the actual
categorical fixed point.

Crucially: this realization is more categorically faithful than
delay-substitution, which is just N=1 iteration. The IR keeps cycles;
no `traceCycles` pass is required. Cost: N× compute per sample in
cyclic regions.

### F. Wave Digital Filters / structural mapping

Operations are wave-domain adaptors; reactive elements (capacitors,
inductors) provide intrinsic delays. The analog-to-digital mapping is
one-to-one structurally; cycles in the source circuit become acyclic
dataflow in the wave domain without any compiler-side cycle breaking.
Stability is guaranteed by construction.

Constraint: requires modeling in circuit-topology terms (not block-
diagram or transfer-function terms).

### G. Physics-aware per-cycle integration

The compiler analyzes each cycle's transfer characteristics and
chooses a discretization scheme (forward Euler, backward Euler,
trapezoidal/bilinear) per cycle. This is what serious analog modeling
does (Zavalishin's ZDF, others).

---

Each of these is a valid choice of trace functor. The categorical
structure doesn't pick one for us; the language designer does. The
categorically cleanest options for general-purpose use are A, C, and E.
F is the no-compromise option for analog modeling specifically.

## Wave Digital Filters in depth

WDF deserves special attention because it's the only option in the
cycle-handling spectrum that's structurally lossless — the only one
where the analog-to-digital mapping is a true functor between operads.

### The wave variable transformation

Instead of tracking node voltages `v` and currents `i`, track:

```
a = v + R·i   (incident wave)
b = v - R·i   (reflected wave)
```

The wave variables are a change of basis — `v` and `i` can be
recovered from `a` and `b` and vice versa. The free parameter `R`
(port resistance) is chosen per port.

### Circuit elements in the wave basis

Under this transformation, circuit elements become simple wave-domain
operations:

- **Resistor with port-matched** R: `b = 0` (perfect absorption)
- **Capacitor** (bilinear discretized): `b[n] = a[n-1]` (unit delay)
- **Inductor**: `b[n] = -a[n-1]` (unit delay with sign flip)
- **Voltage source**: `b = 2V - a`
- **Current source**: `b = a + 2R·I`

Series, parallel, and three-port junctions become *scattering matrices*
— small fixed matrices computed at compile time from the port
resistances.

### Why cycle-breaking dissolves

Every closed analog loop must pass through at least one energy-storing
element (capacitor or inductor) — otherwise it would be a short
circuit. In the wave basis, those elements provide intrinsic unit
delays. So analog cycles become acyclic wave-domain compositions
naturally — the delays are placed exactly where the circuit puts the
energy-storing elements, never at arbitrary back-edges.

### The functor

WDF is what you get when you constrain the analog-to-digital map to be
a true functor of operads. There is a unique (up to port-resistance
choice) such functor for tree-structured passive circuits. The math
forces the answer; no language design choice is needed.

### The stability guarantee

WDF preserves "pseudo-power" (the wave-domain analog of `v · i`) under
composition. If the original analog circuit is passive (energy-
dissipating), the WDF model is provably stable. This is unique to WDF
— no other discretization scheme has this constructive stability
guarantee.

### The limitations

The limitations are real:

- **Tree-structured topologies only** in pure WDF; non-tree topologies
  require R-type adaptors (manageable but adds machinery).
- **Nonlinear elements** (diodes, transistors, soft clippers) require
  local iteration at the nonlinear element. Handled, but not free.
- **The user must think in circuit-topology terms**, not block-diagram
  terms. This is a serious modeling constraint.

For tropical's design: WDF is one realization of certain operations
(filters, distortion stages, analog emulations). It's not a candidate
for the default cycle-handling strategy because it requires too-specific
modeling. It IS a candidate for a per-operation realization choice —
operations defined as WDF blocks live in tropical's operad with the
same port semantics as standard operations, but with WDF realization
internally.

## Multi-realization architecture

The deepest implication of operadic encapsulation: **the realization
of an operation can vary while preserving its operadic identity**. Two
operations with the same port signature and the same I/O behavior are
*the same operation* in the operad's view, regardless of how they're
implemented internally.

This is the Yoneda-ish principle: an operation is fully characterized
by its observable port behavior. An operation realized as a WDF
subnetwork, an operation realized as a standard tropical kernel, and
an operation realized via FFI to a hand-written C++ function are the
same operation if they have the same signature and behavior. They can
be substituted for each other in any operadic composition.

This makes tropical's identity the operadic interface, not the
implementation. The implementation is a *realization* — a specific way
to evaluate the operadic operation in a runtime. Multiple realizations
are first-class. The compiler routes each operation to its appropriate
realization compiler.

Concrete realization candidates:

- **Standard tropical** — the current compilation through strata.
  Generic signal-flow.
- **Iterative fixed-point** — for cycles where no-extra-latency
  matters; runtime iterates N times per sample.
- **Wave Digital Filters** — for analog circuit modeling; exact
  topology mapping, stability guarantees.
- **FFI / native** — escape hatch for performance-critical hand-
  written kernels.
- **Neural network inference** — for learned approximations of analog
  circuits.
- **State machine / event-driven** — for sequencers, MIDI parsers,
  anything that isn't time-step driven.

Each realization has:

- A signature interpretation (what kind of values flow through its
  ports)
- Adapters at the boundary if the internal representation differs from
  the operadic port type
- A compiler that maps its realization-specific IR to the runtime's
  slot model
- Possibly its own intermediate IR

The runtime is uniform: it executes kernels that read input slots, do
their thing, write output slots. The kernels' internal mechanics are
realization-specific; the slot model glues them together.

## The continuous-discrete semantic hierarchy

Tropical's compilation pipeline implicitly crosses three semantic
levels:

1. **Continuous-time semantics** — what the user "means" when they
   write a feedback loop or a filter. The mathematical referent is a
   continuous-time signal flow model: time is real-valued, signals are
   continuous functions.

2. **Discrete-time semantics with chosen delay placement** — what
   compiles. Time is sample-indexed; signals are sample sequences;
   feedback is broken by unit-delay primitives (or solved iteratively,
   or handled wave-domain — depending on the realization chosen). Each
   cycle-handling option from earlier represents a different choice of
   discrete-time semantics.

3. **Finite-precision arithmetic** — what runs. Floating-point
   representation; rounding; accumulator behavior. Lossy approximation
   of the discrete-time semantics.

Each level introduces choices and lossy approximations. Each level can
be made categorically rigorous. The functors between levels are the
discretization and quantization steps.

Most DSL implementations collapse all three levels and call it a day.
Being explicit about which level is the source of truth, and which
discretization choices the compiler makes, is part of language design
clarity. Tropical's current compilation pipeline collapses level 1 to
level 2 silently (via implicit `traceCycles`). Making this collapse
visible — either by requiring explicit feedback markers, or by exposing
the realization choice per operation — is part of the operadic vision.

## Memory model: slotified, port-bordered, hierarchically named

Under the operadic framing, memory has one role: holding the
persistent state that supports the trace operator. Different memory
namespaces (state registers, module slots, array storage, temps) were
historical implementation artifacts, not categorical distinctions.

The architectural commitment: **one slot namespace, port-typed and
hierarchically named**. Every persistent state — register, delay,
input port, output port, inter-program communication channel — lives
in one global slot array. Ownership is per-operation:

- **Internal slots** (regs, delays, intermediate state) — owned by an
  operation, visible only within its body.
- **Input port slots** — owned by an operation, *written* by its
  immediate parent, *read* by its own body.
- **Output port slots** — owned by an operation, *written* by its own
  body, *read* by its parent or siblings.

Each operation's slots occupy a contiguous range in the global
namespace. Names follow the operadic plug path: `voice1.env.cutoff` is
"the `cutoff` state of the `env` operation plugged into `voice1`'s
`env` slot." Hot-swap matches by full path; replacing an operation's
internals preserves the slot range; the surrounding operations see no
change.

Per-sample scratch state (intermediate temps within a kernel's body)
is JIT-internal — never named in the IR. LLVM's `mem2reg` + GVN
promote slot operations that have register-like access patterns to SSA
values automatically.

Cross-operation communication is via slot writes from parent to child
input port slots and slot reads from sibling output port slots. The
encapsulation is structural: an operation's body has no language-level
mechanism to reference slots outside its own range (plus its immediate
sub-operations' port slots). Slot allocation enforces this; emit-level
operand resolution honors it.

## The category Op and what compilation is

`Op` — the category of typed operads with operadic morphisms (functors
that preserve colors, operations, and composition) — is the
metacategory in which tropical's compilation lives. The pipeline is a
sequence of morphisms in `Op`:

```
Surface Operad (typed, traced, primitives = base set)
  │  specialize   — substitute type params
  ▼
Monomorphic Surface Operad
  │  sumLower     — replace coproduct operations
  ▼
Sum-Free Operad
  │  traceCycles  — choose a trace functor; insert delay primitive
  ▼
Choice-of-Trace Operad
  │  arrayLower   — unroll combinators
  ▼
Scalar DAG Operad
  │  identityElim — categorical identity law
  ▼
Normal-Form Operad
  │  operationalize — assign slots; convert operadic plugs
  ▼              to slot writes
Slot-Operational Operad (runtime ready)
```

Each pass is a named morphism. Each preserves the categorical content
modulo the specified rewrites. The total compilation is the composition
of these morphisms. Equivalence tests (`jit_vs_interp`, `wasm_vs_jit`,
etc.) are claims that two paths through this composition produce the
same final operad — that two compositions of morphisms agree.

This framing makes the compilation pipeline self-documenting at the
type level. Each pass has a categorical signature; the IR shapes
between passes are precisely the intermediate operads.

## Design implications

What this framing tells us about tropical's design choices:

**1. The IR types should make operadic structure explicit.** A
`ResolvedProgram` is not "a program with a body of declarations"; it's
an operation in the operad with a body expressing its definition as a
composition. The naming, the type structure, and the pass signatures
should reflect this.

**2. Encapsulation is structural, not a discipline.** The IR should
make boundary-violating operations literally inexpressible. The current
`cloneWithInputSubst`-style substitution that produces scope leaks is
categorically invalid and should be impossible to write under the
right IR types.

**3. The cycle-handling convention is a language design choice that
should be visible.** Whichever option (A through G) tropical adopts,
the choice should be explicit — in source-language syntax, in
documentation, and in the IR types. Users should never have to guess
where the compiler placed a delay.

**4. Multi-realization is the long-term vision.** The IR carries a
realization tag per operation. v1 has one realization (standard
tropical compilation through strata). Future versions add realizations
(iterative, WDF, FFI, etc.) without changing the language's identity.
The operadic interface is the stable contract.

**5. Memory unification (slotified) is the foundational refactor.** It
is the precondition for the rest of the vision. The four-namespace
memory model is an obstacle; one slot namespace with hierarchical
naming and structural enforcement of ownership is the destination.

**6. The compilation pipeline is a composition of operad morphisms.**
Strata passes can be reorganized, replaced, or added by composing
additional morphisms in `Op`. The framework is generative — new
compilation strategies are new morphism compositions.

## What remains free for the language designer

The categorical structure doesn't pick everything. Several decisions
are genuinely free:

- **The choice of cycle-handling realization** (the trace functor's
  instantiation). Multiple valid options; tropical picks one as default
  per language version.
- **The set of primitive operations** (the operadic generators). Add
  or remove primitives; the operadic framework adapts.
- **The level of language exposure for realizations**. Does the user
  select per-operation realization, or does the compiler default
  everything? Both are valid.
- **The discretization scheme for `delay`**. Forward Euler is the
  standard but trapezoidal/bilinear or specialized schemes are equally
  valid alternatives.
- **The numerical precision and accumulator behavior**. Floats vs
  fixed-point; saturation vs wrap. Implementation choices that don't
  affect operadic structure.

These are language design choices. Tropical commits to specific
positions; future tropical versions can re-decide.

## Closing note

Operads are not a clever lens through which to view tropical. They are
the structural fact about what tropical *is*. The domain — signal flow
through bounded port-typed devices — is operadic; the IR is an
operadic presentation; the compilation pipeline is a composition of
operad morphisms.

Making this explicit — in IR types, in language design, in
documentation — is the architectural commitment. Everything else
(cycle handling, slotified memory, multi-realization, specific
implementations) is the working out of details under this commitment.
