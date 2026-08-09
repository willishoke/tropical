# mcp/

The MCP stack. The production server is the **Lean `frontend` binary**
(`lean/.lake/build/bin/frontend`) — it is the whole stack: session,
registration, compiler (patch-bay ingest → direct root construction →
lower → emit → partition; there is no raise-to-AST and no elaborator),
runtime FFI, save/export/load/merge, resources/prompts.
There is no compiler-service subprocess and no TypeScript engine; the
former `engine.ts` / `ir_service.ts` / `resources.ts` / `envelope.ts`
are deleted, and Lean implements them natively
(`Tropical/Engine.lean`, `Tropical/Resources.lean`,
`Tropical/Errors.lean`, `Tropical/Tools.lean`, `Tropical/Rpc.lean`,
`Tropical/Frontend.lean`).

What lives in this directory now is the **behavioral protocol suite** —
two bun tests that run against the live Lean engine over its
JSON-RPC surface — plus this doc and `ERRORS.md`:

```
errors.test.ts    Error-envelope protocol: codes, suggestions, retryability.
wire_dac.test.ts  Wiring + dac-output protocol behavior.
CLAUDE.md         This file.
ERRORS.md         The error-code catalog (mirrors Tropical/Errors.lean).
```

Both suites are engine-agnostic: they speak the protocol via
`TROPICAL_ENGINE_CMD`, which points at the Lean engine in `--rpc`
mode. `make test-lean-engine` runs them against
`lean/.lake/build/bin/frontend --rpc`.

## Running

```bash
make mcp-lean   # build C++ core + Lean front door, then launch the MCP server
make lean       # build just the lean/ front door (compiler + MCP server, one binary)
```

The server is also configured in `.mcp.json` for Claude Code — it
builds the front door (`make -s lean`) and execs
`lean/.lake/build/bin/frontend`. Two modes:

- default — the MCP server over stdio (`make mcp-lean`).
- `--rpc` — the newline JSON-RPC surface the bun protocol suites and
  the differential harness drive.

## Compile pipeline behind every mutation

Every tool that changes the signal graph ultimately recompiles the
whole session to a single kernel and hot-swaps it. The shape (all
inside the Lean engine):

```
SessionState
  → syncCompile
       → liftIfNeeded          (anonymous instances for array-literal wires)
       → buildSessionInputVia
            → assertSessionAcyclic
            → sessionToResolvedRoot
       → compileSessionStaged  (partition + typed stage-0 split)
       → emit LLVM/MSL artifacts + tropical_plan_6 metadata
  → loadKernel                (C API → OrcJitEngine/MetalKernel → FlatRuntime publication)
```

The direct `Ir.Strata` lowering (`assertAcyclic → inlineInstances →
identityElim → toResolved`) runs when a program type is registered. The live
synthetic session root instead links those already-resolved snapshots,
checks the session graph, and copies the reachable resolved root before
partitioning. (The old five-pass strata drop sequence was retired
2026-07-25, and the elaborator followed it 2026-07-26 — program bodies
no longer cross the wire at all: a loaded file carrying a `programDecl`
is refused at ingest with the retirement message, and the session wire
grammar cannot spell `fold`/`tag`/….) tropical is
closed-form-only: every kernel is
a pure `f(τ, params)` with no per-sample state, so there is nothing to
break a cycle around. `assertSessionAcyclic` is therefore a plain "no
cycles at all" rule — inter-instance cycles are rejected outright, at
every boundary that constructs a graph (`Ir/Cycles.lean`). (A
recursive filter on live or broadband input has no closed form; that
island is ceded on purpose to a future stateful sister runtime,
"supertropical" — see `design/cf-only.md`.) The WASM backend consumes
the same `tropical_plan_6` and is held to sample-for-sample equivalence
with the JIT (`tests/web/wasm_vs_jit`).

A compile error doesn't kill the session; it returns a structured
error envelope (see below) and the previous kernel keeps playing.

## SessionState

The engine owns one `SessionState`. The fields tools read and mutate:

- `programs` / `templateByName` — registered concrete program metadata and
  resolved roots. Populated by `Tropical.Stdlib` boot and
  `export_program`; loaded patch files cannot carry definitions.
- `instances` — ordered live instances with resolved type snapshots.
- `wires` — typed `WireExpr` connections.
- `graphOutputs` — what wires to dac.
- `params` — current control values by name (the session compiler
  materializes `param:<name>` module slots, which `set_param` writes without
  relowering).
- `paramDisciplines` — plan-carried `raw`, `glide`, `anchor`, or `velocity`
  dispatch metadata for the loaded kernel.

The engine environment also owns the native runtime handle and a DAC created
lazily on first `start_audio`.

The instance name `dac` is reserved — it's the audio-output boundary,
not a real instance.

## Tools

### Program management

New DSP types are **not** defined over the wire. They are authored as
`Tropical.Stdlib` arrow-combinator builders in Lean and booted directly
by the engine (15 at present) — no parse bridge, no literate `.md`
surface language. The former `define_program` tool and its entire
generics apparatus (`type_params` / `type_args`, monomorphization, the
specialization cache) are gone; every registered program is concrete.

- `add_instance` — instantiate a registered type by name. Validates
  uniqueness. Since no program declares `type_params`, passing
  `type_args` is always rejected with `invalid_type_args`.
- `remove_instance` — delete an instance, cascade-clean wiring that
  references it.
- `replicate` — create N instances in one call (does not trigger
  recompile by itself; pair with `wire`).
- `list_programs` — the registered concrete types with their ports and
  input defaults.
- `list_instances` — live instances with their base type and ports.
- `get_info` — detailed port / wiring info for one instance.

### Wiring

All of these update the same ordered, typed `SessionSt.wires` collection and
recompile; they are shape conveniences for common graph patterns.

- `wire` — set and/or remove input wires in a single recompile. The
  audio-output bus is `instance: "dac", input: "out"`; multiple wires
  to it sum into the mono mix.
- `wire_chain` — N instances in series, optional initial expression
  into the first input.
- `wire_zip` — pairwise sources → targets.
- `fan_out` — one source (literal, param, or ref) to N targets.
- `fan_in` — N sources, optional per-source gain, summed to one target.
- `list_wiring` — show current input expressions, optional instance filter.

(There is no `feedback` tool. Feedback loops require per-sample state,
which the closed-form-only language does not have — cycles are rejected
outright. Recursive filtering of live input belongs to the future
stateful sister runtime, "supertropical"; see `design/cf-only.md`.)

### Program I/O

- `export_program` — crystallize selected session instances into a
  reusable `ProgramType`. Current wiring becomes input defaults.
  Optionally removes the exported instances. It builds the resolved IR
  **directly** off the session mirror — there is no JSON round-trip.
  This is the only route that registers a new program type at runtime.
- `load` — `tropical_program_2` JSON (path or inline). Stops audio,
  recreates the session from the file's instances + wiring over
  already-registered programs. It is a PATCH BAY: instances, wiring,
  and params of types that are **already registered**. A file carrying
  an inline program definition (`programDecl`) is refused at ingest —
  program definitions over the wire are retired. To get a new type,
  either author it in Lean as an arrow builder (`Tropical.Stdlib` /
  `EmitArrow`) or build it from instances and crystallize it with
  `export_program`.
- `save` — session → `tropical_program_2` JSON.
- `merge` — additive: instances + wiring of already-registered programs
  without clearing the session.

### Control parameters

- `set_param` — apply the loaded plan's declared `raw`, `glide`, `anchor`, or
  `velocity` host discipline, then update the corresponding runtime slots
  without relowering. Glide ramps and anchor/velocity rebasing are explicit
  closed-form companion-slot writes, not hidden per-sample state. This is the
  only public parameter-write method; discipline-specific method aliases are
  retired.
- `list_params` — registered params and their current values.

### Audio control

- `start_audio` — open output device (optional name substring).
  `sample_rate` / `channels` apply only to the first DAC creation in
  the session.
- `stop_audio` — stop playback.
- `audio_status` — running flag, device info, callback stats
  (callback count, avg/max ms, underruns, overruns).

## Error envelope

The engine returns structured errors so agents can recover
programmatically (the Lean source of truth is `Tropical/Errors.lean`;
the catalog is in `ERRORS.md`).

```typescript
type ErrorCode =
  | 'unknown_program' | 'unknown_instance' | 'unknown_input' | 'unknown_output'
  | 'unknown_param'   | 'unknown_device'
  | 'instance_exists' | 'invalid_type_args'
  | 'type_mismatch'   | 'shape_mismatch' | 'length_mismatch' | 'arity_error'
  | 'missing_argument' | 'invalid_value' | 'invalid_state'
  | 'compile_failed'   | 'audio_error'   | 'internal_error'

type ErrorEnvelope = {
  code:        ErrorCode
  message:     string
  retryable:   boolean
  param?:      string                  // which input parameter triggered
  value?:      unknown                 // what the user passed
  valid?:                              // validity descriptor
    | { kind: 'enum';      options: string[] }
    | { kind: 'record';    fields: Record<string, FieldSpec> }
    | { kind: 'predicate'; predicate: string; expected: unknown; got: unknown }
  suggestion?: unknown                 // nearest-match correction (Levenshtein)
}
```

The four fail-shapes:

- bare — plain error (`code`, `message`, optional `param` / `value`).
- enum — invalid enum-valued argument; `suggestion` is the nearest
  valid option by Levenshtein distance (≤ max(2, ⌊len/3⌋)).
- record — invalid object argument; `valid.fields` describes expected
  types/required-ness/bounds.
- predicate — domain check failed (e.g. range, ordering).

`compile_failed` carries the lowering or emit error verbatim in
`message` and is `retryable: true` — the previous kernel keeps
playing while the agent edits and retries.

`errors.test.ts` and `wire_dac.test.ts` pin this behavior against the
live Lean engine.
