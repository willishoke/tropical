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
  compile-side extractor hoists those delays into per-wire registers)
  so the agent can describe feedback loops
  without thinking about topological order. This is the shape that
  let "build me a four-pole resonant filter with self-oscillation"
  go from a sentence to audible signal without any structural
  intermediate step.

The interesting bit is that these three are not independent
subsystems with adapters between them; they share one IR. The
compiler's correctness criterion is "the JIT and the WebAssembly
backend agree on every sample for every input"; the realtime
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
and WebAssembly must agree" the right correctness criterion
rather than an ambitious one. The longer design conversation that
arrived at this lives in
[`design/archive/operadic_ir.md`](design/archive/operadic_ir.md).

## How it gets built

Tropical is built through Claude Code with Opus 4.7 (1M context).
The workflow is task-level parallelism: a fleet of agents working
on independent pieces of the codebase in parallel, managed by one
person who holds the architectural picture. Opus 4.7's context
window is the leverage — it can hold a large fraction of this
codebase at once and reason about cross-cutting changes that would
otherwise require careful staging.

Architectural decisions go through rigorous written specs: phased
implementation plans, clear correctness criteria, agreement on
what counts as "done" before code gets written. The specs are
cross-disciplinary by default. Each decision gets viewed through
whichever practitioner traditions are relevant to the problem
(category theory, type theory, systems programming, DSP,
real-time audio), and the tensions between those views get worked
out in the spec rather than in the code. The structure of the IR,
the runtime, and the stdlib all carry marks of that process.

What makes this tractable is end-to-end validation. The codebase
is built around the assumption that a model is going to make
subtle structural mistakes and the test suite has to catch them
before they reach production:

- **Sample-for-sample equivalence gates** between the two backends —
  the LLVM ORC JIT and the WebAssembly emitter, both consuming the
  same `tropical_plan_5` — plus realization-variant differentials
  inside the JIT (fused vs. per-instance microkernel; flat vs.
  nested) and byte-for-byte audio goldens (`tests/equiv/`). A
  regression in any pass surfaces here, not in audible artifacts a
  week later.
- **Round-trip and acyclicity invariants** at every IR boundary —
  the elaborator throws on cyclic source code, the strata pipeline
  asserts acyclic input, the session compiler runs a defensive
  cycle check after delay extraction. Mistakes that would once have
  surfaced as silent cross-backend divergence now throw at the
  boundary with a port-detailed error message.
- **A stdlib audit** that parses, elaborates, and lowers every
  `stdlib/*.trop` file on every change, asserting every post-strata
  invariant.
- **CI** runs all of it (TypeScript typecheck, C++ tests, TS tests,
  YAML lint) on every PR against GitHub Actions, with LLVM 20
  pinned. Local development runs the same gates via `make validate`.

With this surface in place, refactors that would otherwise need
careful manual review go through cleanly. The recent removal of
triggers as a first-class primitive, the removal of alive-gating,
the switch from `tropical_plan_4` to a per-instance
`tropical_plan_5` with a scheduler — all landed as multi-commit
branches without per-diff human inspection. The tests do the
reading.

## Where to read next

- [`design/architecture.md`](design/architecture.md) — the full IR
  walkthrough, top to bottom.
- [`tests/equiv/`](tests/equiv/) — the sample-for-sample equivalence
  suites that pin the JIT and WebAssembly backends to each other.
- [`CLAUDE.md`](CLAUDE.md) — contributor map.
- [`INSTALL.md`](INSTALL.md) — build prerequisites.
- [`mcp/CLAUDE.md`](mcp/CLAUDE.md) — MCP server and tool reference.

## License

MIT.
