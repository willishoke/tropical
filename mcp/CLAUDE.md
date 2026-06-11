# mcp/

The TypeScript side of the MCP stack. As of Phase 1 of the Lean port
(`design/lean_port.md`), the *engine* — session state and the 23 tool
handlers — lives in Lean (`lean/Tropical/Engine.lean`), running
in-process behind the Turnstile front door. What remains here is the
**compiler service** the Lean engine drives, plus the retired-but-kept
TS engine that serves as the differential oracle.

- `compiler_service.ts` — what the Lean front door spawns. As of
  Phase 4 the service performs **no raise and no elaboration** on any
  production path; Lean owns the compiler front (raise + elaborate,
  `lean/Tropical/Parse/*` + `lean/Tropical/Ir/*`) and ships resolved
  IR over the `tropical_resolved_1` codec
  (`compiler/ir/resolved_codec.ts`). What's left here:
  `register_program` (typeDef registries + `decodeResolved` + strata),
  `compile` (rebuilds the session mirror from Lean's snapshot, decodes
  `resolved_root`, runs `partitionKernel`, returns the plan JSON
  string), wire lifting (needs strata), and save/export/load/merge
  (they need the v2 ingest + compiler until their phases). The service
  boots empty — the engine registers the stdlib from the pre-parsed
  bridge (`stdlib/parsed/`). The Lean engine owns the native runtime,
  the DAC, and params over its own FFI (`lean/Tropical/Ffi.lean`);
  plans returned by `compile`/`load`/`merge` hot-swap into the
  Lean-owned runtime. Newline JSON-RPC; tool-level failures return
  `{result: {error: <ErrorEnvelope>}}`.
- `engine.ts` + `ir_service.ts` — the full TS engine and its protocol
  surface. No longer on the production path; kept as the differential
  oracle for `make diff-engine` and the in-process unit tests until
  the TS pipeline retires (Phase 6/8).

## Running

```bash
make mcp-lean   # build C++ core + Lean front door, then launch the MCP server
```

`make lean` builds just the `lean/` front door. Also configured in `.mcp.json`
for Claude Code (it builds the front door and execs it).

## Layout

```
compiler_service.ts        The TS compiler behind the Lean engine (spawned by
                           the front door). Shrinks phase by phase until Phase 6.
envelope.ts                ErrorEnvelope types + fail helpers (shared source for
                           the service; Lean mirrors it in Tropical/Errors.lean).
resources.ts               MCP resources/prompts surface (program catalog,
                           program-format doc, build-patch prompt).
engine.ts                  The retired TS engine: SessionState + handleTool.
                           Differential oracle only.
ir_service.ts              engine.ts over newline JSON-RPC/stdio. Oracle side A
                           of make diff-engine; the Lean engine speaks the same
                           protocol via `frontend --rpc`.
program_format_example.ts  The canonical tropical_program_2 example the
                           program-format resource renders from.
test_patch.ts              CLI smoke-tester: bun run mcp/test_patch.ts <patch.json> [n_frames]
*.test.ts                  errors / wire_dac (protocol suites — run against any
                           engine via TROPICAL_ENGINE_CMD; `make test-lean-engine`
                           gates the Lean engine) / program_format_example /
                           remove_feedback (in-process TS-engine tests)
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
       → compileSessionSlotted (buildSessionRoot → elaborate one root
                                ResolvedProgram → partitionKernel →
                                instance_functions[] (root, nested) + sinks[])
       → tropical_plan_5 JSON
  → JSON.stringify
  → runtime.loadPlan  (C++: NumericProgramParser → OrcJitEngine → FlatRuntime hot-swap)
```

The strata pipeline (now `assertAcyclic → specialize → sumLower →
inlineInstances → arrayLower → identityElim`) runs per-instance
inside `compileResolved`. Session-level cycle handling lives in the
wire layer: `setWireExpr` wraps every wire in a unit delay,
`extractSessionDelays` hoists it into `session.delaySlotRegistry`; the
root-program lowering serializes each entry to a root `RegDecl`
read-old/write-new writeback (no scheduler tier — outputs are `sinks`).
The WASM backend consumes the same `tropical_plan_5` and is held to
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
