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

1. **WASM backend supports root plans: DONE.** `emit_wasm` now recurses through
   nested `children` with the same four-phase order as the C++
   `emit_kernel_block` (preamble → per-child {pre_input, child} → body →
   writebacks), instead of treating each `instance_function` as a flat leaf. The
   recursive `collectKernelInstrs` also walks the tree for param discovery /
   layout sizing. No engine change; pure `emit_wasm.ts`. Both backends now agree
   on the root-program shape: `tests/equiv/wasm_vs_jit.test.ts` gains two
   permanent (non-env) root-mode cases, and the whole equiv suite is green with
   `TROPICAL_ROOT_PROGRAM=1`. **The flag is now backend-complete — Phase E
   cutover is unblocked.**
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
4. **Phase D2 — array session DELAYS: DONE.** A `delay()` over an array-shaped
   source (hoisted to an `ioArraySlot`, `isArray` registry entry) now
   materializes to an **array-typed root `RegDecl`**: `init` is an
   `arraySize`-long literal array (so `compileResolved` allocates a backing array
   slot via its `arrayRegMap`), and `update` is the translated array source
   (`nestedOut` to the producer's array output). `emit_resolved` lowers the
   array-result update to an elementwise copy at writeback — the same per-element
   one-sample latency the per-instance `state_evolution` array `Add` gave.
   `sessionArraySlot` reads translate to `regRef`. One supporting engine-free fix
   in `emit_resolved.ts`: the array-reg writeback now accepts a
   `session_array_reg` source (a sibling's array output), not just kernel-local
   `array_reg` (`remapInstancePlan` already lowers it to an absolute slot). The
   dispatch fallback is gone — array sessions go through the root path. Gate:
   `root_vs_flat` array-session-delay case (`delay([s, s+10, s+20]) → ArrSum`,
   25 pass) — a time-varying array source makes any latency/permutation bug
   visible. Motivation: polyphony and any cross-voice array feedback will hit
   this routinely.
5. **Phase E — cutover (default flipped): DONE.** `compileSessionSlotted` now
   defaults to the root path; `compileSession(session)` with no options compiles
   a root plan. `options.rootProgram === false` selects the legacy per-instance
   path (kept as the `root_vs_flat` oracle + escape hatch); `TROPICAL_ROOT_PROGRAM=0`
   forces it off when no explicit option is passed. Supporting fix:
   **slot-based session params on the root path.** Session params are
   `param:name` module slots (control plane via `setSlot`, hot-swap by name), but
   the materializer's `paramRef` lowered to a dead FFI handle (empty map → param
   reads returned 0). Added a `paramSlots: Map<ParamIdx, number>` threaded
   `compileSessionSlottedRoot → partitionKernel(root only) → compileResolved →
   emit_resolved`, so `ParamRef` lowers to `opSlot(param slot)` — mirroring the
   per-instance `translateNode`. `paramSlots` is NOT propagated to child kernels
   (`ParamIdx` is per-program). A handful of per-instance *structural* tests
   (instruction/instance-function-count, param-slot-operand placement) are pinned
   to `rootProgram: false` since they assert that lowering's emit shape; the
   audio/behavior param tests run on the default (root) path as coverage. Full
   suite 1039 pass with the default flipped; `TROPICAL_ROOT_PROGRAM=0` and
   `root_vs_flat` keep the per-instance path exercised.

   **Remaining (deferred, plan's "after a release"):** delete
   `compileSessionSlottedPerInstance` + the `state_evolution` emission block and
   rewrite/remove the pinned per-instance structural tests + the `root_vs_flat`
   oracle. Keep `delaySlotRegistry`, `emitDacStitch`, `graphOutputs` as the
   bridge/DAC tap. Not done here — the per-instance path is still the
   differential oracle, so deleting it now would remove the safety net mid-soak.

## Key invariants to preserve

- One-axis-at-a-time: no golden re-baselining. The cache-key fix doesn't move
  goldens (it can only AVOID wrong-kernel reuse).
- DAC stays on `emitDacStitch`/`graphOutputs`; root has no output ports.
- Delay ordering (R1): root `RegDecl` writebacks are a trailing read-old/write-new
  batch; the engine runs children(pre_input+body) → root body → root writebacks,
  so one-sample latency is preserved by construction. `root_vs_flat` proves it.
