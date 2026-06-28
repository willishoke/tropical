# tropical

Tropical is a substrate for stateless computation. Every program —
every oscillator, filter, envelope, and wire — compiles to a single
pure function `f(τ, params)` of a time coordinate, with no per-sample
state anywhere in the kernel. A whole patch is one such function: the
compiler fuses every device into one per-sample kernel that the LLVM
ORC JIT emits fresh on each edit.

Statelessness is the point, not an implementation detail. Because a
kernel computes sample *n* directly from `τ = n` instead of stepping a
recurrence, the language gets what a stream-of-state engine cannot:

- **Random access in time.** Any sample is computable in isolation —
  scrub, jump, or render a window with zero warm-up.
- **Reversibility.** Negate the time coordinate and the signal runs
  exactly backwards, sample-for-sample; there is no accumulator to unwind.
- **Rate- and backend-agnosticism.** Nothing latches a sample rate or a
  block size, so the same IR lowers to the LLVM JIT *and* to WebAssembly,
  and sample-for-sample agreement between the two is a CI gate.
- **No clicks, structurally.** Every edit hot-swaps a fresh kernel;
  matching params transfer by name. Phase is recomputed from `τ`, never
  latched, so there is no discontinuity to smooth — nothing clicks because
  there is no state to carry.

Audio synthesis is the first instantiation of the substrate, not its
definition: `τ` is a time coordinate, the values are samples, the runtime
is an audio device. None of that is load-bearing in the IR.

This README is the high-level pitch. The full architecture lives
in [`design/architecture.md`](design/architecture.md); the IR walks
top-to-bottom from literate `.md` source through five structure-dropping
passes to the slot-typed instruction stream the runtime executes.

## What tropical actually is

One IR, three domains it has to satisfy at once:

- **Compilers.** A real strata-style IR pipeline with name
  resolution, monomorphization, sum-type lowering, instance
  inlining, combinator unrolling, and a final categorical
  identity-law rewrite. Each pass produces an IR strictly poorer
  than its input — the dropped structure is something the next pass
  doesn't have to reason about. By the end, you have a flat,
  scalar-only, monomorphic, acyclic, non-nested, combinator-free
  graph that's the smallest sub-IR any per-sample evaluator needs.
- **Realtime audio.** Single-kernel fusion, double-buffered
  hot-swap with params transferred by name (there is no per-sample
  state to carry across a recompile — phase is recomputed from `τ`, so
  edits never click), no libm in the kernel (transcendentals are stdlib
  programs that inline at strata time, deterministic across platforms),
  lock-free atomic control parameters, sub-millisecond JIT compile
  budgets, RtAudio with device hot-swap and disconnect recovery.
- **Agentic interfaces.** Patches are graphs an agent edits
  incrementally over MCP. The IR is acyclic by construction: there is
  no state primitive to break a cycle through, so feedback — in source
  or in an MCP mutation — is a compile error with a port-detailed
  message, not a silent one-sample loop. That constraint is what lets
  "give me a through-zero flanger" or "an additive voice that runs
  backwards" go from a sentence to audible signal without any structural
  intermediate step. The MCP front door is itself a native Lean 4
  server built on [Turnstile](https://github.com/willishoke/turnstile)
  ([`lean/Main.lean`](lean/Main.lean)): every tool's arguments are a
  typed Lean structure, the wire schema is derived from that type,
  and each call is decoded and validated at the boundary before it
  reaches the engine.

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
makes the layout of `lean/Tropical/Ir/` predictable and makes "the JIT
and WebAssembly must agree" the right correctness criterion
rather than an ambitious one. The longer design conversation that
arrived at this lives in
[`design/archive/operadic_ir.md`](design/archive/operadic_ir.md).

## Everything is legible

The same discipline that makes the IR legible to the compiler makes
the source legible to everyone else. There is no `.trop` file format:
**a tropical program is a markdown document.** Every program in
[`stdlib/`](stdlib/) is a literate document — prose explaining the
DSP, a mermaid diagram of the signal flow, and exactly one
```` ```tropical ```` code block holding the source. The compiler reads
the code fence; humans and agents read the whole document; GitHub
renders the diagrams inline.

There is deliberately no parallel metadata layer: the signature in the
code block *is* the interface, and the prose *is* the documentation —
nothing is written twice, so nothing can fall out of sync. What can be
enforced structurally, is: the literate gate in the Lean
golden runner (the built `tropicaltest` binary, off
`lean/Tropical/Parse/Surface/Markdown.lean`) fails the build on a
mismatched title, a missing diagram, or anything other than exactly
one code block. An agent that wants to know what `SVF` exposes reads
the same document a human does. See
[`stdlib/OnePole.md`](stdlib/OnePole.md) for the canonical example.

## Why you can trust the output

There is no second implementation to diff against and no interpreter
fallback on the audio path, so correctness is pinned by gates, not by
review. Every change runs the full surface:

- **Sample-for-sample equivalence gates** between the two backends —
  the LLVM ORC JIT and the WebAssembly emitter, both consuming the
  same `tropical_plan_5` (`tests/web/`) — plus realization-variant
  differentials inside the JIT (fused vs. per-instance microkernel;
  flat vs. nested) and byte-for-byte audio goldens, both run by the
  built `tropicaltest` binary. A regression in any pass surfaces here,
  not in audible artifacts a week later. Correctness is anchored by the
  frozen goldens; the cross-backend agreement proves the two emitters
  *agree*, the goldens prove they're *right*.
- **Acyclicity by construction** at every IR boundary — the elaborator
  throws on cyclic source code, the strata pipeline asserts acyclic
  input, and the session compiler runs the same cycle check as a
  compile-time invariant. With no state primitive to break a cycle
  through, feedback is rejected with a port-detailed error rather than
  silently turned into a one-sample loop.
- **A stdlib audit** that parses, elaborates, and lowers every
  `stdlib/*.md` file on every change, asserting every post-strata
  invariant — plus the literate gate, which fails the build if any
  program document loses its title, diagram, or single-code-block
  shape.
- **CI** runs all of it (the Lean build, the `tropicaltest` golden
  runner, the C++ tests, and the behavioral bun suites against the live
  Lean engine) on every PR, with LLVM pinned. Local development runs the
  same gates via `make validate`.

With this surface in place, load-bearing refactors land as multi-commit
branches without per-diff human inspection — most recently the move to a
closed-form-only IR (cutting per-sample state entirely) and the
fixed-point clock substrate underneath reversible, random-access voices.
The tests do the reading.

## Where to read next

- [`design/architecture.md`](design/architecture.md) — the full IR
  walkthrough, top to bottom.
- [`tests/web/`](tests/web/) — the sample-for-sample equivalence
  suites that pin the JIT and WebAssembly backends to each other
  (run against the live Lean engine).
- [`CLAUDE.md`](CLAUDE.md) — contributor map.
- [`INSTALL.md`](INSTALL.md) — build prerequisites.
- [`mcp/CLAUDE.md`](mcp/CLAUDE.md) — MCP server and tool reference.
- [`lean/Main.lean`](lean/Main.lean) — the Lean 4 MCP front door,
  built on [Turnstile](https://github.com/willishoke/turnstile);
  the whole tool surface as typed, schema-derived Lean structures.

## License

MIT.
