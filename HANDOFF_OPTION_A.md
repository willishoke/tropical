# Handoff — Option A, Phases A + B (session → root ResolvedProgram)

**Date:** 2026-05-31
**Branch:** `feat/single-mcp-server` (working tree; nothing committed for this work)
**Plan:** `/Users/rhizome/.claude/plans/great-let-s-build-a-fluttering-lake.md` (full phased plan, approved)
**Background memory:** `project_session_program_divergence.md` (architecture rationale).

## Status: Phases A + B COMPLETE and green.

Option A makes a **session** materialize into a single root `ResolvedProgram`
(instances → `InstanceDecl`s, wires → `InstanceDecl.inputs`, per-wire unit
delays → root `RegDecl`s) and lowers it through the **existing**
`partitionKernel` path, behind a flag (default OFF). The C++ engine is
semantically unchanged; the scheduler becomes a DAC-stitch postamble only.

### What's done

**Phase A — materializer + unit test (green).**
- `compiler/ir/materialize_session.ts` — `materializeSessionToResolvedIR(session)`.
  - Builds each instance's `InstanceDecl` from **`inst.compiled.prog`** (the exact
    fully-strata-lowered program the flat path compiles, with its own
    `programRegistry` so `partitionKernel`'s recursive `getInstanceType` resolves
    the whole subtree). *Do not* re-fetch/re-specialize from `session.programs` —
    that risked a less-lowered child (surviving `let`/`fold`) that `compileResolved`
    rejects. This was a real bug found and fixed during Phase B.
  - Scalar session delays → reserve-then-fill synthetic `RegDecl{init,update}`
    (two-pass `slotToReg` so nested `delay(delay(x))` resolves). Array delays throw.
  - No strata call, no DAC branch (DAC stays on `emitDacStitch`/`graphOutputs`).
- `compiler/ir/materialize_session.test.ts` — 5 cases, all pass
  (`bun test compiler/ir/materialize_session.test.ts`). Uses `resolveProgramType` +
  `instantiate` to build instances; dac goes on `session.graphOutputs` (NOT
  `setWireExpr`, which would auto-delay it). Cycle test keys `inputExprNodes`
  via `wireKey(...)`.

**extractSessionDelays concern (from prior handoff): VERIFIED CLEAN — non-issue.**
A real osc→amp→dac session yields exactly one `delaySlotRegistry` entry per
non-dac wire, distinct `sourceExpr`s, no phantom/aliased entries. The prior
worry was a test artifact (dac routed through `setWireExpr`). No blocker.

**Phase B — root lowering behind a flag (green).**
- Flag `rootProgram?: boolean` on `CompileSessionOptions` +
  `CompileSessionSlottedOptions`; env `TROPICAL_ROOT_PROGRAM=1` forces it on.
- `compileSessionSlottedRoot(session, mode)` in `compile_session_slotted.ts`:
  materialize → `makeCompiled(root, {displayName:'__session__'})` →
  `partitionKernel(ROOT_INSTANCE_PATH, root, ...)` → `instance_functions=[rootFn]`;
  `emitDacStitch` postamble; `state_evolution=[]`.
- **Naming transparency** (`partition_recursive.ts`): `ROOT_INSTANCE_PATH`
  (`'__root__'`) is a sentinel `instancePath` that `joinInstancePath` treats as
  no-prefix, so the root's children and own registers carry BARE names
  (`amp.out`, `osc.phase`) — `emitDacStitch`'s `graphOutputs` lookup and
  register names land exactly where the flat path puts them.
- **Array fallback:** `sessionHasArrayWiring(session)` (array-typed instance
  ports OR array session delays) routes back to `compileSessionSlottedPerInstance`.
  Phase B is scalar-only; this keeps `TROPICAL_ROOT_PROGRAM=1` safe to force on
  across the whole corpus (internal array state like `Delay`'s ring buffer does
  NOT count — it never crosses a session wire).
- **Gate:** `tests/equiv/root_vs_flat.test.ts` — flag-off vs flag-on via
  `renderFramesJit`, 1e-12, over the stdlib corpus + a multi-instance
  auto-delayed osc→amp session + 8× polyphony. **23/23 pass.**
- **Migration goldens forced-on:** `TROPICAL_ROOT_PROGRAM=1 bun test
  tests/equiv/migration_audio.test.ts` → **11/11 pass** (array patches fall back).

### Two engine bugs found + fixed (cache-key completeness, `OrcJitEngine.cpp`)

The JIT cache key serialization (BOTH the fused `kernel_cache_` key ~L1100 and
the microkernel/deep `microkernel_cache_` key ~L1437) hashed only each top-level
function's own `instructions` + `instance_name` + `register_count` — it never
recursed into `children` / `preamble_instructions` / `pre_input_instructions` /
`writebacks`. The flat/nested paths dodged this because their top-level body
always differs per type. The **root path exposes it**: every root plan's single
top-level function is empty-bodied and identically named (`instance___root__`),
so all root plans collided and served each other's cached kernels (Cos got Exp's
kernel, etc.). Fixed by making both keys recurse via a `serialize_fn` lambda.
This is a pure cache-key fix — no emitted-code or golden change. Rebuild required
(`cmake --build build`); build-id auto-invalidates the disk cache.

## Verification (all green unless noted)

- `npx tsc --noEmit` — clean.
- `bun test` (flag OFF, default) — **1035 pass / 0 fail.**
- `bun test tests/equiv/` flag OFF — 107 pass.
- `TROPICAL_ROOT_PROGRAM=1 bun test tests/equiv/root_vs_flat.test.ts` — 23 pass.
- `TROPICAL_ROOT_PROGRAM=1 bun test tests/equiv/migration_audio.test.ts` — 11 pass.
- `ctest --test-dir build` — pass.

## Known limitations / next steps

1. **WASM backend does not support root plans yet.** With the flag forced ON
   globally, `tests/equiv/wasm_vs_jit.test.ts` has 4 failures (SinOsc×2,
   SinOsc→OnePole, the array-zipWith case). JIT-root is correct (proven by
   `root_vs_flat`); the diverging side is `emit_wasm`, which doesn't handle the
   root-program plan shape (deeper nesting + root `RegDecl` writebacks +
   empty top-level body). Phase B targets the **JIT** path; WASM root support is
   Phase C/E. The flag defaults OFF, so normal WASM operation is unaffected. If
   you force the flag on, scope it to JIT suites.
2. **Unwired typed-bound defaults differ (documented, benign).** An UNWIRED
   input with a typed-bound default (`freq: freq = 440`) resolves to 0 on the
   flat top-level path (its plain-number-only `defaults` builder skips the
   `ExprNode` default) but to the declared 440 on the root child path
   (`rawInputDefaults`). Every realistic session and every equiv test wires its
   inputs, so this never bites in practice; the `root_vs_flat` osc→amp case wires
   `freq` explicitly. NOT fixed (one-axis-at-a-time; arguably root is *more*
   correct). If a future patch relies on unwired typed defaults, revisit.
3. **Phase D1 — array-typed PORT wiring: DONE.** Array-typed instance ports now
   lower on the root path. Two materializer fixes: (a) emit the root's child
   instances in `computeInstanceTopoOrder` (producer before consumer) — array
   wires are same-sample (not auto-delayed), so the producer must run first;
   (b) keep array ports on `InstanceDecl.inputs` so `compileResolved`'s
   `arrayInfo` branch copies the array `nestedOut` into the child's session-array
   slot. The dispatch fallback narrowed from `sessionHasArrayWiring` to
   `sessionHasArrayDelay`. Gate: `root_vs_flat` Sequencer array-literal case;
   migration goldens forced-on stay 11/11 (now via the root path).
4. **Phase D2 — array session DELAYS: deferred (fallback is correct).** A
   `delay()` over an array-shaped source (hoisted to an `ioArraySlot`, `isArray`
   registry entry) still routes to the per-instance path via
   `sessionHasArrayDelay`. There is **no consumer in the current corpus** (no
   fixture/patch produces one; they're hard to even construct — the source must
   be a `ref` to an array output at extract time), and the fallback yields
   byte-identical output. Per "don't build ahead of need," left as the narrow
   fallback until a real array-delay session appears; then either add array
   `RegDecl` support to the materializer or keep array delays on
   `state_evolution` in the root path.
5. **Phase E — cutover** (after the WASM-root gap closes): flip the default and
   delete `compileSessionSlottedPerInstance` + the `state_evolution` emission.
   Keep `delaySlotRegistry`, `emitDacStitch`, `graphOutputs` as the bridge/DAC tap.

## Key invariants to preserve

- One-axis-at-a-time: no golden re-baselining. The cache-key fix doesn't move
  goldens (it can only AVOID wrong-kernel reuse).
- DAC stays on `emitDacStitch`/`graphOutputs`; root has no output ports.
- Delay ordering (R1): root `RegDecl` writebacks are a trailing read-old/write-new
  batch; the engine runs children(pre_input+body) → root body → root writebacks,
  so one-sample latency is preserved by construction. `root_vs_flat` proves it.
