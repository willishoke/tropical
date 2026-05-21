# tropical

A realtime audio synthesis language whose compiler treats a patch
as one program: every oscillator, filter, envelope, and wire gets
fused into a single per-sample kernel that the LLVM ORC JIT emits
fresh on every topology change. The browser backend does the same
thing in WebAssembly, off the same intermediate representation, and
sample-for-sample equivalence between the two is a CI gate.

This README is the high-level pitch. The full architecture lives
in [`design/architecture.md`](design/architecture.md); the IR walks
top-to-bottom from `.trop` source through six structure-dropping
passes to the slot-typed instruction stream the runtime executes.

## What tropical actually is

A genuinely cross-domain problem, attacked with one consistent
discipline.

Three domains, one project:

- **Compilers.** A real strata-style IR pipeline with name
  resolution, monomorphization, sum-type lowering, instance
  inlining, combinator unrolling, and a final categorical
  identity-law rewrite. Each pass produces an IR strictly poorer
  than its input — the dropped structure is something the next pass
  doesn't have to reason about. By the end, you have a flat,
  scalar-only, monomorphic, acyclic, non-nested, combinator-free
  graph that's the smallest sub-IR any per-sample evaluator needs.
- **Realtime audio.** Single-kernel fusion, double-buffered
  hot-swap with state transfer by name (so delay lines and
  oscillator phase survive a recompile without clicking), no libm
  in the kernel (transcendentals are stdlib `.trop` files that
  inline at strata time, deterministic across platforms), lock-free
  atomic control parameters, sub-millisecond JIT compile budgets,
  RtAudio with device hot-swap and disconnect recovery.
- **Agentic interfaces.** Patches are graphs that an agent edits
  incrementally over MCP. Cycles in MCP-built graphs are broken at
  the wire layer (every wire is wrapped in a unit delay; the
  compile-side extractor hoists those delays into the scheduler's
  `state_evolution` phase) so the agent can describe feedback loops
  without thinking about topological order. This is the shape that
  let "build me a four-pole resonant filter with self-oscillation"
  go from a sentence to audible signal without any structural
  intermediate step.

The interesting bit is that these three are not independent
subsystems with adapters between them; they share one IR. The
compiler's correctness criterion is "the JIT and the pure-TS
interpreter agree on every sample for every input"; the realtime
constraint is "every edit must produce a fresh kernel inside the
audio thread's latency budget"; the agentic constraint is "every
sentence-scale edit must produce a syntactically well-typed graph
or a structured error the model can recover from." All three
criteria are pinned simultaneously by the same set of tests.

The structural fact about the domain — signal flow through bounded,
port-typed devices that interact only through their interfaces — is
operadic. The IR makes this explicit and the compiler is, in the
strict categorical sense, a functor from the source operad to the
slot-operational operad consumed by the runtime. You don't have to
think in those terms to read the code, but the framing is what
makes the layout of `compiler/ir/` predictable and makes "the JIT
and the interpreter must agree" the right correctness criterion
rather than an ambitious one. The longer design conversation that
arrived at this lives in
[`design/archive/operadic_ir.md`](design/archive/operadic_ir.md).

## How it gets built

Tropical is built almost entirely through Claude Code with Opus 4.7
(1M context). The model writes the code; I drive the architecture,
review the changes, and own the design decisions. I'm not going to
overstate this — I haven't built out a fleet of specialized
subagents or a custom multi-agent review loop. What I have is a
tight feedback cycle between a person who knows what the system
should look like and a frontier model that can hold an unusual
amount of the codebase in context at once. That's the leverage.

The discipline that makes this work is end-to-end validation. The
codebase is built around the assumption that a model is going to
make subtle structural mistakes and the test suite has to catch
them before they reach production:

- **Sample-for-sample equivalence gates** between three backends —
  the LLVM ORC JIT, a pure-TS reference interpreter, and the
  WebAssembly emitter — running across the full stdlib corpus
  (`tests/equiv/`). Every IR-touching change has to keep these
  three in agreement. A regression in any pass surfaces here, not
  in audible artifacts a week later.
- **Round-trip and acyclicity invariants** at every IR boundary —
  the elaborator throws on cyclic source code, the strata pipeline
  asserts acyclic input, the session compiler runs a defensive
  cycle check after delay extraction. Mistakes that would once have
  surfaced as silent JIT-vs-interpreter divergence now throw at the
  boundary with a port-detailed error message.
- **A stdlib audit** that parses, elaborates, and lowers every
  `stdlib/*.trop` file on every change, asserting every post-strata
  invariant.
- **CI** runs all of it (TypeScript typecheck, C++ tests, TS tests,
  YAML lint) on every PR against GitHub Actions, with LLVM 20
  pinned. Local development runs the same gates via `make validate`.

The result is that the model can refactor aggressively — the
recent removal of triggers as a first-class primitive, the removal
of alive-gating, the switch from `tropical_plan_4` to a per-instance
`tropical_plan_5` with a scheduler — without me having to read
every diff carefully. The tests do the reading.

That's the actual workflow: a frontier model with deep context,
plus a test surface dense enough that the model's mistakes show up
as failed CI runs rather than as bugs.

## Why you might care

If you're a hiring manager who got here from a résumé: I build
ambitious systems in domains that don't usually overlap, using the
current generation of frontier models as a force multiplier, and I
have a real working theory of how to organize a codebase so that
this scales rather than producing slop. The shape of this repo is
the demonstration. Read
[`design/architecture.md`](design/architecture.md), look at
[`tests/equiv/`](tests/equiv/), or grep for "Phase" comments in
`compiler/ir/` to see how the IR has migrated over time without
breaking the equivalence gates.

If you want to run it: see [`INSTALL.md`](INSTALL.md). If you want
to point an agent at it: the MCP server is documented in
[`mcp/CLAUDE.md`](mcp/CLAUDE.md). If you want to read the code: the
top-level [`CLAUDE.md`](CLAUDE.md) is the contributor map.

## License

MIT.
