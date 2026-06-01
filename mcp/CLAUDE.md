# mcp/

The IR engine and its JSON-RPC service. The MCP *server* itself now lives in
`lean/` — a native Lean front door built on Turnstile — and this directory is
what it drives. (The old `mcp/server.ts`, an `@modelcontextprotocol/sdk` server,
has been retired; the Lean front door replaced it, validating each tool call
against a typed schema before relaying.)

- `engine.ts` owns a long-lived `SessionState` (`compiler/session.ts`) and the
  23 tool handlers (`handleTool`), plus the MCP resources (program catalog,
  program-format doc) and the build-patch prompt. Transport-agnostic.
- `ir_service.ts` exposes that engine over plain newline JSON-RPC on stdio
  (method = tool name; plus `resources/list`·`read` and `prompts/list`·`get`).
  The Lean front door spawns it and relays validated calls.

## Running

```bash
make mcp-lean   # build C++ core + Lean front door, then launch the MCP server
```

`make lean` builds just the `lean/` front door. Also configured in `.mcp.json`
for Claude Code (it builds the front door and execs it).

## Layout

```
engine.ts                  Transport-agnostic IR engine: SessionState + handleTool
                           + resources/prompts. No transport.
ir_service.ts              Exposes the engine over newline JSON-RPC/stdio (the
                           Lean front door spawns this).
program_format_example.ts  The canonical tropical_program_2 example the
                           program-format resource renders from.
test_patch.ts              CLI smoke-tester: bun run mcp/test_patch.ts <patch.json> [n_frames]
*.test.ts                  errors / wire_dac / program_format_example — drive ir_service
```

## Compile pipeline behind every mutation

Every tool that changes the signal graph ultimately calls `wire()`,
which runs `applyFlatPlan(session, runtime)`:

```
SessionState
  → compileSession (compiler/ir/compile_session.ts)
       → liftWiresToInstances  (anonymous-instance lift for array-literal wires)
       → extractSessionDelays  (hoist auto-wrap delays into session.delaySlotRegistry)
       → assertSessionAcyclic  (defensive invariant)
       → compileSessionSlotted (default: materialize one root ResolvedProgram
                                → partitionKernel → instance_functions[]+scheduler)
       → tropical_plan_5 JSON
  → JSON.stringify
  → runtime.loadPlan  (C++: NumericProgramParser → OrcJitEngine → FlatRuntime hot-swap)
```

The strata pipeline (now `assertAcyclic → specialize → sumLower →
inlineInstances → arrayLower → identityElim`) runs per-instance
inside `compileResolved`. Session-level cycle handling lives in the
wire layer: `setWireExpr` wraps every wire in a unit delay,
`extractSessionDelays` hoists it into `session.delaySlotRegistry`. The
default root-program lowering realizes each entry as a root `RegDecl`
writeback (the scheduler's `state_evolution` phase is then empty — the
legacy per-instance path uses it for one `WriteSlot` per entry). The
WASM backend consumes the same `tropical_plan_5` and is held to
sample-for-sample equivalence with the JIT (`tests/equiv/wasm_vs_jit`).

A compile error doesn't kill the session; it returns a structured
error envelope (see below) and the previous kernel keeps playing.

## SessionState

`engine.ts` owns one `SessionState`. The fields tools read and mutate:

- `typeRegistry: Map<string, ProgramType>` — registered concrete types
  (`define_program`, stdlib loading); `ProgramType` is the
  metadata-wrapper form
- `programs: Map<string, ResolvedProgram>` — unified registry of
  every registered program, both concrete (post-strata, `typeParams=[]`)
  and generic templates (raw, `typeParams.length > 0`). Pre-Phase-5
  this was split into `resolvedRegistry` + `genericTemplatesResolved`.
- `specializationCache: Map<string, ProgramType>` — keyed by
  `Type<N=8>`-style cache keys
- `instanceRegistry: Map<string, ProgramInstance>` — live instances
- `inputExprNodes: Map<"inst:input", ExprNode>` — wiring
- `graphOutputs: Array<{instance, output}>` — what wires to dac
- `paramRegistry` — control parameters by name
  (the session compiler turns names into FFI handles at compile time)
- `runtime: Runtime` — native `tropical_runtime_t`
- `dac: DAC | null` — created lazily on first `start_audio`

The instance name `dac` is reserved — it's the audio-output boundary,
not a real instance.

## Tools

### Program management

- `define_program` — register a reusable type from a `tropical_program_2`
  object. Generic programs (declaring `type_params`) become templates
  that monomorphize at instance time.
- `add_instance` — instantiate a registered type by name. `type_args`
  for generics (e.g. `{N: 8}`). Validates uniqueness.
- `remove_instance` — delete an instance, cascade-clean wiring that
  references it.
- `replicate` — create N instances in one call (does not trigger
  recompile by itself; pair with `wire`).
- `list_programs` — concrete types + generic templates with ports,
  defaults, and `type_params`.
- `list_instances` — live instances with their `type_args`.
- `get_info` — detailed port / wiring / register info for one instance.

### Wiring

All five of these compile down to the same `inputExprNodes` mutation +
`wire()` recompile; they're shape-conveniences for the most common
graph patterns.

- `wire` — set and/or remove input wires in a single recompile. The
  audio-output bus is `instance: "dac", input: "out"`; multiple wires
  to it sum into the mono mix.
- `wire_chain` — N instances in series, optional initial expression
  into the first input.
- `wire_zip` — pairwise sources → targets.
- `fan_out` — one source (literal, param, or ref) to N targets.
- `fan_in` — N sources, optional per-source gain, summed to one target.
- `feedback` — one-sample delay loop with a stable `delay_id` so
  state survives hot-swap.
- `list_wiring` — show current input expressions, optional instance filter.

### Program I/O

- `export_program` — crystallize selected session instances into a
  reusable `ProgramType`. Current wiring becomes input defaults.
  Optionally removes the exported instances.
- `load` — `tropical_program_2` JSON (path or inline). Stops audio,
  recreates the session.
- `save` — session → `tropical_program_2` JSON.
- `merge` — additive: instances + wiring without clearing the session.

### Control parameters

- `set_param` — update a smoothed `Param`. Thread-safe (atomic
  store on the C++ side); the smoothing time-constant is set at
  param creation. (Wire format still accepts `{op:'trigger', name}`
  for backcompat — aliased to `{op:'param', name}` at materialization.)
- `list_params` — registered params and their current values.

### Audio control

- `start_audio` — open output device (optional name substring).
  `sample_rate` / `channels` apply only to the first DAC creation in
  the session.
- `stop_audio` — stop playback.
- `audio_status` — running flag, device info, callback stats
  (callback count, avg/max ms, underruns, overruns).

## Error envelope

`engine.ts` returns structured errors so agents can recover programmatically.

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

Helpers in `engine.ts`:

- `failBare({ code, message, retryable?, param?, value? })` — plain
  error.
- `failEnum({ code, param, value, options })` — invalid enum-valued
  argument; `suggestion` is the nearest valid option by Levenshtein
  distance (≤ max(2, ⌊len/3⌋)).
- `failRecord({ code, param, value, fields })` — invalid object
  argument; `valid.fields` describes expected types/required-ness/bounds.
- `failPredicate({ code, param, value, predicate, expected, got })` —
  domain check failed (e.g. range, ordering).

`compile_failed` carries the strata or emit error verbatim in
`message` and is `retryable: true` — the previous kernel keeps
playing while the agent edits and retries.

## Smoke test

```bash
bun run mcp/test_patch.ts <patch.json> [n_frames]
```

Loads the patch, runs `runtime.process()` `n_frames` times, reports
peak output, exits non-zero on silence or NaN. No audio device
required. Useful for proving the full TS → JIT → kernel pipeline
without hooking up RtAudio.
