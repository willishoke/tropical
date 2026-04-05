# Phase 2 Fusion Attempt — Post-Mortem

This document covers everything tried after commit `b6c7a18` (which is now the tip of PR #57). The goal was to enable primitive body fusion for patches with array inputs (e.g. Clock's `ratios_in: [1.0]`).

---

## What was working at b6c7a18

- Audio played correctly via per-module processing
- The fused input kernel ran when `fusion_enabled_ = true`
- Primitive body fusion was blocked with: `"primitive body fusion requires scalar top-level input bindings"`
- Phase 2 (bypass Value intermediary for fused inputs) had been designed but not yet applied

---

## Phase 2 commit — `b0ad7ea` (reverted)

**What it did:** In `run_fused_input_kernel`, replaced the fallback `module->inputs[id] = float_value(...)` / `array_value(...)` writes with direct writes to `numeric_input_scalar_override_` and `numeric_array_storage_`, setting `numeric_input_override_active_ = true`. This eliminates the Value tagged-union round-trip in the fused input → per-module body path.

**Status:** Committed, then reverted because it exposed the fusion-blocking bug described below.

---

## Root cause of fusion block

`build_fused_graph_state` calls `module.build_numeric_program(compiled_program, state, module.inputs, ...)` to determine input types. If `module.inputs[i]` is `float_value(0.0)` (scalar), the input is compiled as scalar. If it is `array_value([1.0])`, it is compiled as array and gets an array slot in the fused body layout.

Modules are constructed with `inputs(input_count, expr::float_value(0.0))`. `build_fused_graph_state` runs at startup — before any `eval_input_program` has executed — so all inputs appear scalar. Clock's `ratios_in` (which should be `[1.0]`) is seen as scalar `0.0`. This causes `primitive_body_inputs_ok = false` and blocks fusion.

The previous fix (`adf2747` — stop zeroing inputs) was necessary but not sufficient: it preserved array types *between* samples, but didn't help at first build because no samples had run yet.

---

## Attempted fix: seed inputs before build

Added a loop in `build_runtime_locked()` just before `rebuild_fused_graph_state`:

```cpp
for (auto & slot : runtime.modules)
{
  if (slot.module)
    eval_input_program(runtime, slot.input_program, slot.input_registers, slot.module->inputs);
}
```

This evaluates each module's input program once so `module.inputs` has the correct types (including `array_value([1.0])` for Clock's `ratios_in`) before the fused body compiler reads them.

**First attempt** also called `ensure_numeric_jit_current()` after `eval_input_program`. Result: audio corruption (massive pop + high-frequency noise). Diagnosis: `ensure_numeric_jit_current()` recompiles the per-module JIT kernel and reinitializes `numeric_array_storage_`, `numeric_registers_`, etc. — state that `build_fused_graph_state` is about to read for the fused body layout.

**Second attempt** removed `ensure_numeric_jit_current()`, keeping only `eval_input_program`. Result: **no audio**. Diagnosis below.

---

## Silence bug: fused body fails, fallback path broken

With seeding (no `ensure_numeric_jit_current()`), `primitive_body_available = true` and `fused->kernel` is compiled. But `run_fused_primitive_body_kernel` immediately failed:

```
"primitive body runtime module array output slot missing"
```

**Cause:** Clock has an array output (`ratios_out`). The extraction loop after the kernel call tries to copy the output to `module.numeric_array_storage_[local_slot]`, but that vector is empty — `ensure_numeric_jit_current()` was never called to size it.

When the fused body returns `false`:
- `used_fused_body = false`
- The fallback re-runs `run_fused_input_kernel(runtime, false)`, but `fusion_enabled_ = false` by default, so it returns `false` immediately
- `execute_parallel_module_work` is called with `use_fused_inputs = true`
- Modules in `primitive_body_module_mask` are **skipped** even though the body didn't run
- All 40 modules produce no output → silence

**Fix attempt:** auto-allocate `module.numeric_array_storage_` on first use in the array output extraction block, instead of failing.

---

## Stutter bug: array register sync also fails

With the array-output fix, audio returned but as "stutter/impulse every few seconds". Diagnosis: a second check in the same function:

```cpp
// array register sync (delay state, e.g. ADEnvelope's delay(gate, 0.0))
if (local_array_slot >= module.numeric_array_storage_.size())
  return fail("primitive body runtime local array register sync mismatch");
```

ADEnvelope's `prevGate = delay(gate, 0.0)` uses a 1-element array register (delay state). The extraction loop hit this check before syncing the delay state back, failed mid-loop, and left register state inconsistent. The kernel had already run and written to `fused->temps`, so some modules had partial output updates while others didn't.

**Fix attempt:** same auto-allocation pattern applied to the register sync block.

---

## Outcome

Both allocation fixes built and passed tests. The user reported audio was still broken ("completely broken"). At that point the attempted fixes were reverted and the branch was reset to `b6c7a18`.

---

## What is still needed

The seeding approach is fundamentally sound: `eval_input_program` correctly populates array input types before the fused body compiler runs, without the side effects of `ensure_numeric_jit_current()`. The failure cascade (array output → fail → bad fallback → silence; then array register → fail → stutter) was plausibly fixable with the auto-allocation approach, but audio quality was never verified clean.

Open questions before re-attempting:

1. **Are there more hidden `numeric_array_storage_` size checks** in `run_fused_primitive_body_kernel`? A full audit of every `return fail(...)` call in that function is needed.
2. **Does the fallback path need fixing regardless?** When `fusion_enabled_ = false` and the fused body fails, `run_fused_input_kernel(runtime, false)` returns `false`, leaving primitive-body-masked modules without inputs before `execute_parallel_module_work` skips them. This silent data loss should be fixed independently of the fusion work.
3. **Should the fix happen at build time or runtime?** An alternative to seeding is to have `build_fused_graph_state` evaluate input types from the compiled input program directly (dry-run with temp registers), avoiding any mutation of `slot.input_registers`.
