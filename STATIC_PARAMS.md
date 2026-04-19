# Generic `Delay<N>` via Compile-Time Type Parameters

## Context

The stdlib currently ships six structurally identical delay types — `Delay1`, `Delay8`, `Delay16`, `Delay512`, `Delay4410`, `Delay44100` — differing only in buffer size. Each is a separate JSON file with the buffer length hardcoded in both the `regs.buf.zeros` count and the `mod` operator inside the process body. This is ugly, not extensible, and defeats the point of the ProgramJSON abstraction.

Introduce **compile-time type parameters** so a single `stdlib/Delay.json` parameterized by integer `N` replaces the whole family. The mechanism is general: `type_params` on a program declaration, `type_args` on an instance, and a new `{op:"type_param",name}` ExprNode. Monomorphization happens at type-resolution time by cloning the ProgramJSON template, substituting `type_param` refs, and running through the existing `loadProgramDef` pipeline. Downstream (flatten, emit, JIT) is untouched.

This branch is named `feat/static_params`. Intended outcome: the six Delay files are gone, one generic `Delay` with `type_params: {N: {type:"int", default: 44100}}` remains, and the mechanism is available for future generic types.

## Naming

- `type_params` — on program declarations (ProgramJSON)
- `type_args` — on instance entries
- `{op: "type_param", name: "N"}` — ExprNode op substituted at specialization time

Chose over `static` because `static` is overloaded (C, C++, Java, Rust all mean different things); `type_params`/`type_args` cleanly mirrors the C++ template analogy.

## Phase 1 — Schema + specialization module (pure, unused)

Add the engine and types. Nothing is wired into runtime yet, so nothing else breaks.

**New file: `compiler/specialize.ts`**
- `specializeProgramJSON(prog: ProgramJSON, args: Record<string, number>): ProgramJSON` — deep-clones via `structuredClone`, walks ExprNodes (`process.outputs`, `process.next_regs`, `delays[*].update`, `input_defaults`, `instances[*].inputs`, `instances[*].type_args`) substituting `{op:"type_param",name}` with `{op:"float",value:N}` (or just a numeric literal — ExprNode supports bare numbers). Also substitutes in `regs[*]` when the value is `{zeros: {type_param: "N"}}`. Recurses into inline `programs[*]` with the outer args.
- `specializationCacheKey(typeName, args)` — sorts keys, `JSON.stringify`s.
- `resolveTypeArgsInContext(rawArgs, outerArgs): Record<string, number>` — evaluates each arg value; a numeric literal passes through, a `{op:"type_param",name}` looks up `outerArgs[name]`. Throws on unresolved names. Used for nested forwarding.
- Validates: every declared `type_param` has a resolved value (from args or declared default); rejects unknown arg keys.

**Type edits:**
- `compiler/program.ts:23` ProgramJSON — add `type_params?: Record<string, { type: 'int'; default?: number }>`; extend `instances[*]` with `type_args?: Record<string, number | ExprNode>`.
- `compiler/program_types.ts:78` ProgramInstance — add readonly `typeArgs?: Record<string, number>`; thread through constructor and `instantiateAs`.
- `compiler/session.ts:61` SessionState — add `specializationCache: Map<string, ProgramType>`; init in `makeSession`.

**Schema edits (`compiler/schema.ts`):**
- `RegValueSchema` (~38) — add `z.object({ zeros: z.union([z.number().int().positive(), z.object({ type_param: z.string() })]) })`. This also closes a pre-existing gap: the current schema rejects plain `{zeros: 44100}`, which stdlib relies on but MCP `load` would refuse.
- `ProgramInstanceSchema` (~116) — add `type_args: z.record(z.string(), z.union([z.number().int(), ExprNodeSchema])).optional()`.
- `ProgramJSONSchema` (~121) — add `type_params: z.record(z.string(), z.object({ type: z.literal('int'), default: z.number().int().optional() })).optional()`.
- `ExprOpNode` is already a passthrough, so `{op:"type_param",name}` parses.

**New test: `compiler/specialize.test.ts`**
- substitutes in `process.outputs`, `process.next_regs`, `regs.*.zeros`, nested `programs[*]` bodies
- applies declared defaults when arg absent; `type_args:{}` and absent both mean defaults
- errors on: unknown arg key, missing required param (no default), non-integer value
- deep-clone: mutating output leaves input untouched
- nested forwarding: outer `{N: 44100}` threads into an inner instance's `type_args: {N: {op:"type_param",name:"N"}}`

**Verify:** `bun test compiler/specialize.test.ts`. No other tests should change.

**Commit:** `feat(compiler): add type_params/type_args schema + specialize module`

## Phase 2 — Wire specialization through instance resolution

All six instance-resolution sites funnel through a new helper.

**New helper** (in `compiler/specialize.ts` or `compiler/session.ts`):
`resolveProgramType(session, baseName, rawTypeArgs?, outerArgs?): ProgramType` — if the base type has no `type_params`, enforce that `rawTypeArgs` is absent/empty and return the base. Otherwise: resolve each rawArg against `outerArgs`, fill defaults, build cache key, return cached or clone-substitute-loadProgramDef-insert. **Never cache partially-resolved args** (all cache keys are over fully-resolved integers).

**Update sites** (call the helper, then stash resolved `typeArgs` onto the `ProgramInstance`):
- `compiler/program.ts:122` `loadProgramAsSession`
- `compiler/program.ts:232` `mergeProgramIntoSession`
- `compiler/session.ts:340,348` `loadProgramDef` nested instances — pass the parent program's resolved args as `outerArgs` to support forwarding
- `mcp/server.ts:491` `handleAddInstance` — plumb `typeArgs`
- `mcp/server.ts:502` `handleReplicate` — single `typeArgs` applied to all copies
- `compiler/bench_compile.ts:29,41`, `compiler/transcendentals.test.ts:49` — audit; likely unchanged (they instantiate non-generic types)

`ProgramInstance` emits its resolved `typeArgs` at construction. Since the specialized ProgramDef carries the already-substituted expressions, flatten/emit/JIT see a concrete type and don't need to know static params exist.

**New test: `compiler/specialize_integration.test.ts`**
- define a tiny generic in-memory, instantiate twice with different N → distinct ProgramTypes, both cached; repeat with same N → same object identity
- composite program with a generic child and forwarded `N` flattens correctly
- regression: instantiating `OnePole` (no `type_params`) still works; passing `type_args` to a non-generic program errors cleanly

**Verify:** `bun test` (everything green); hand-build a `Delay` in-memory and confirm a plan round-trips.

**Commit:** `feat(compiler): monomorphize generic programs at instance resolution`

## Phase 3 — MCP surface + serialization round-trip

**`mcp/server.ts`:**
- `add_instance` input schema (~141): add `type_args: { type: 'object', description: '...' }`. Thread through to `handleAddInstance`.
- `replicate` (~162): add `type_args`. Thread through to `handleReplicate`.
- `list_programs` (~778): include each program's `type_params` in the response so agents discover how to instantiate generics.
- `get_info` (~808): include each instance's `typeArgs`.
- `define_program`, `load`, `merge`, `export_program`, `save`: audit pass-through. `parseProgram` now accepts the new schema fields automatically.

**Serialization (`compiler/program.ts`):**
- `saveProgramFromSession` (line 348): write `{ program: inst.typeName, type_args: inst.typeArgs }` when `inst.typeArgs` is set.
- `exportSessionAsProgram` (line 520): same pattern in the exported instance emission.

**New tests:**
- Round-trip in `compiler/program.test.ts`: define a generic, `add_instance` with `type_args`, `saveProgramFromSession`, clear, `loadProgramAsSession` → cache rebuilds, instance has same `typeArgs`.
- `mcp/test_patch.ts` or a new test: invoke `add_instance Delay {N: 100}` then `get_info` and verify `type_args` surfaces.

**Verify:** `make mcp-ts`, then via MCP: `define_program` generic Delay, `add_instance` with `type_args`, `save`, inspect JSON.

**Commit:** `feat(mcp): expose type_params/type_args through tools and serialization`

## Phase 4 — Stdlib migration

**New `stdlib/Delay.json`:**
```json
{
  "schema": "tropical_program_1",
  "name": "Delay",
  "type_params": { "N": { "type": "int", "default": 44100 } },
  "inputs": ["x"],
  "outputs": ["y"],
  "regs": { "buf": { "zeros": { "type_param": "N" } } },
  "input_defaults": { "x": 0 },
  "breaks_cycles": true,
  "process": {
    "outputs": {
      "y": { "op": "index", "args": [
        { "op": "reg", "name": "buf" },
        { "op": "mod", "args": [{ "op": "sample_index" }, { "op": "type_param", "name": "N" }] }
      ]}
    },
    "next_regs": {
      "buf": { "op": "array_set", "args": [
        { "op": "reg", "name": "buf" },
        { "op": "mod", "args": [{ "op": "sample_index" }, { "op": "type_param", "name": "N" }] },
        { "op": "input", "name": "x" }
      ]}
    }
  }
}
```

**Migrate patches:**
- `patches/cross_fm_4.json` lines 134,144,154,164 (4x Delay8) → `{program:"Delay", type_args:{N:8}}`
- `patches/cross_fm_evolved.json` lines 100–103 (4x Delay8) → same
- `patches/melancholy_house.json` line 368 (Delay44100) → `{program:"Delay", type_args:{N:44100}}`

**Delete:** `stdlib/Delay{1,8,16,512,4410,44100}.json`

Note: `Delay1` currently uses a scalar `prev` register, not a 1-element circular buffer. Under the generic `Delay<N=1>`, it becomes a 1-element buffer. Behaviorally identical (reads then writes slot 0); different register layout. This is acceptable — nothing in the codebase currently instantiates `Delay1` (verified in the audit: no patch, no test, only stdlib self and docs).

**New tests:**
- `compiler/apply_plan.test.ts` (or new `compiler/delay_generic.test.ts`): load `melancholy_house.json`, render a short buffer, assert finite output and expected RMS ballpark.
- Flatten test: `Delay` with N=8 produces the same FlatProgram shape as the old `Delay8` did.

**Verify:** `make build && bun test`, then `bun run mcp/test_patch.ts patches/melancholy_house.json 4096` — audio runs without errors.

**Commit:** `refactor(stdlib): replace Delay<N> family with generic Delay`

## Phase 5 — Documentation

- `README.md:62` — collapse delay family to `Delay` in the stdlib table
- `design/architecture.md:130` — same
- `compiler/CLAUDE.md:65` — stdlib listing
- `design/TESTING.md:105` — remove the "untested: Delay8/16/…" note
- `CLAUDE.md` stdlib count (currently "24 types")
- `mcp/CLAUDE.md` — brief note that `add_instance` / `replicate` accept `type_args`, and `list_programs` surfaces `type_params`

**Commit:** `docs: type parameters and generic Delay`

## Critical files

- New: `compiler/specialize.ts`, `compiler/specialize.test.ts`, `compiler/specialize_integration.test.ts`, `stdlib/Delay.json`
- Modified: `compiler/program.ts`, `compiler/program_types.ts`, `compiler/session.ts`, `compiler/schema.ts`, `mcp/server.ts`
- Migrated: `patches/{cross_fm_4,cross_fm_evolved,melancholy_house}.json`
- Deleted: `stdlib/Delay{1,8,16,512,4410,44100}.json`

## Design risks (closed)

1. **Partial resolution.** `resolveProgramType` refuses to cache unless every arg is a concrete integer. Nested forwarding is resolved against the outer frame before caching.
2. **Specialization before validation.** `specializeProgramJSON` runs before `loadProgramDef`, so `loadProgramDef` never sees `type_param` ops or `{zeros:{type_param:...}}`.
3. **`list_programs` leakage.** Specializations live in `session.specializationCache`, never in `typeRegistry`. `list_programs` reads only `typeRegistry`.
4. **Round-trip.** `ProgramInstance.typeArgs` survives save/load via `saveProgramFromSession` and `loadProgramAsSession`.
5. **Replicate semantics.** `replicate` takes one `type_args` that applies to every copy. Agents wanting different Ns call `add_instance` per copy.
6. **Default handling.** Absent `type_args` and `type_args: {}` both mean "use defaults". Explicit unknown key errors.
7. **Delay1 layout change.** Noted above; no callers affected.

## Out of scope (future)

- Promoting a concrete value to a generic parameter during `export_program` (requires user to specify which constants become `type_params`).
- Non-integer type parameters (floats, strings).
- Type-parameterized *output shape* (e.g. `Delay<N>` producing array output of size N).
