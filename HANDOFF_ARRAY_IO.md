# Handoff — Array-I/O first-class refactor (WIP)

Branch: `feat/array-io-first-class` off `origin/main` at `be6d107`.
Plan: `/Users/willishoke/.claude/plans/okay-let-s-make-a-goofy-mango.md`.

## Status

**~30% of Phase 1 complete.** Data-layer changes are in; consumer updates
(the actual hard work) are pending. Tests are green (1036/1045 pass,
9 skip including one new deferred test, 0 fail) but only because the
data-layer changes are currently *inert*: array-I/O ports now allocate
array slots, but nothing downstream reads them yet. End-to-end array I/O
(Sequencer, Clock, BubbleCloud) still doesn't work — just doesn't error
loudly.

## What's done

### Slot allocator (`compiler/session.ts`)

- `WirePortMeta` extended with optional `arraySlot`, `arraySize`,
  `arrayElemType` fields. For scalar/alias ports these are undefined; for
  array ports they're set and `scalarSlotNames`/`scalarTypes` are empty.
- `SessionState` extended with `ioArraySlotCount`, `ioArraySlotSizes`,
  `ioArraySlotNames` — session-level array-slot registry for array-typed
  I/O ports. Initialized empty in `makeSession`.
- `expandPortToSlots` for array kind: returns `{ names: [], types: [],
  arraySize, arrayElemType }`. Scalar/alias unchanged.
- `allocateOutputSlots` / `allocateInputSlots`: for array ports,
  allocate a session-level array slot (pushed onto `ioArraySlot*`
  registries) and record the slot index + size on the port's
  `WirePortMeta`. For scalar/alias ports, unchanged behavior.

### Emitter schema (`compiler/ir/emit_resolved.ts`)

- `EmitSlots` extended with three new optional maps:
  - `inputArraySlots: Map<InputIdx, { slot, size }>` — for THIS kernel's
    array-typed input ports
  - `nestedInputArraySlots: Map<InstanceIdx, Map<InputIdx, { slot, size }>>`
    — for parent → child array-input wiring
  - `nestedOutputArraySlots: Map<InstanceIdx, Map<OutputIdx, { slot, size }>>`
    — for parent reading child array outputs
- `tryTerminal` `inputRef` case: returns `null` (defers to
  `compileNodeUncached`) when the input idx is in `inputArraySlots`.
- `tryTerminal` `nestedOut` case: returns `null` when the (instance,
  output) pair is in `nestedOutputArraySlots`.
- `compileNodeUncached` gains `inputRef` and `nestedOut` cases that
  return `array_reg` operands (`opArray`) for array-typed ports/outputs.

### Test updates

- `compiler/session_slots.test.ts` — `expandPortToSlots` array-port
  tests rewritten to assert the new shape (empty scalar names, populated
  array fields).
- `compiler/ir/n_write_slot_expansion.test.ts` — the "array_out emits
  N WriteSlots" test marked `.skip` with a comment noting the shape is
  changing.

## What's NOT done — concrete remaining steps

### 1. `compiler/ir/partition_recursive.ts`

Populate the new EmitSlots fields:

- Inside `partitionKernel`, after `allocateOutputSlots`/
  `allocateInputSlots`, build `childOutputArrayMap` and
  `childInputArrayMap` from `session.outputPortMeta`/`inputPortMeta`
  for array-typed ports (read `meta.arraySlot`, `meta.arraySize`).
- Pass them to `compileResolved` via `nestedOutputArraySlots` /
  `nestedInputArraySlots`.
- When recursing into a child, the child's own `inputArraySlots`
  should come from the child's own input ports (lookup via
  `session.inputPortMeta` keyed by `childPath`). This needs threading
  similar to how `inputSlotOverride` is threaded.

`lookupOutputSlot` and `lookupInputSlot` (lines ~91, ~104) currently
return undefined for array ports (because `scalarSlotNames` is empty).
Either:
- (a) Add `lookupOutputArraySlot` / `lookupInputArraySlot` siblings
  that return the array-slot info; callers branch on whether the port
  is array-typed.
- (b) Extend the existing functions to return a discriminated union.

Option (a) is more explicit; option (b) is shorter. Either works.

### 2. `compiler/ir/compile_session_slotted.ts`

Prepend session's I/O array slots to the per-instance `acc` so they
appear in the final FlatPlan:

```ts
// At the top of compileSessionSlottedPerInstance, before iterating
// instances:
for (let i = 0; i < session.ioArraySlotCount; i++) {
  acc.arraySlotNames.push(session.ioArraySlotNames[i])
  acc.arraySlotSizes.push(session.ioArraySlotSizes[i])
}
acc.nextArrayRaw = session.ioArraySlotCount  // starts state-array
                                              // allocation after I/O
```

(Verify exact field names on `acc` — see `makeAccumulators` in
`partition_recursive.ts`.)

### 3. `compiler/ir/compile_session_slotted_helpers.ts`

Two changes:

**(a) `translateNode` `ref` case (line ~120):** for an array-typed
source (`session.outputPortMeta.get(key).arraySlot !== undefined`),
return an `opArray` operand with the recorded array slot, not a
scalar slot read. The CURRENT lookup `session.outputSlotRegistry.get(key)`
returns undefined for array sources (no scalar slot) — the existing
"SlotShapeUnsupportedError" throw will fire. Catch this case explicitly
and return the array operand.

**(b) Output WriteSlot emission (lines ~422-439):** the existing loop
iterates `meta.scalarSlotNames` to emit one WriteSlot per scalar
element. For array outputs, `scalarSlotNames` is now empty so the loop
does nothing — but the array output needs SOME emission. Two options:

- Direct array-slot write: the kernel's body Packs into the output array
  slot directly. The `output_targets` machinery would need to track
  array slots distinctly from temp slots.
- Sequence of `arraySet` ops: emit `arraySet(output_array_slot, k,
  element_k)` for k in [0, N). Reuses existing arraySet infra; simpler
  but per-sample overhead is O(N) writes.

The second is simpler for the initial implementation. The first is
more efficient if performance matters at scale (see Phase 3 stress
test in the plan).

### 4. Parent→child `pre_input` wiring (in compileResolved's nested
emission)

Currently `pre_input_instructions[k]` for child k contains N WriteSlots
(one per scalar input slot). For array-typed inputs, replace those N
WriteSlots with N `arraySet` ops against the child's input array slot.

This happens in `emit_resolved.ts`'s nested-instance emission — search
for `per_child_pre_input` / `pre_input_instructions`.

### 5. `compiler/ir/materialize_session.ts`

Session-level array I/O: when a top-level instance has an array input
wired (e.g., `seq.values = [60, 64, 67, 72]` lifted to `__wire_N`),
the materializer's synthetic top-level program needs to set up the
correct wiring. The lifted `__wire_N` program has an array output —
its array slot. The consumer's array input slot is a different slot.
Either:
- Wire them as one slot (lifted writes directly to consumer's input
  slot), OR
- Wire them as two slots with a copy in between.

The first is more efficient but requires the lift to know its
consumer's slot.

### 6. `compiler/ir/compile_resolved.ts`

Thread the new `EmitSlots` fields (`inputArraySlots`,
`nestedInputArraySlots`, `nestedOutputArraySlots`) through the function
signature and into the constructor that builds `EmitSlots`. Mechanical.

### 7. `compiler/emit_wasm.ts`

Parallel changes:
- `InputRef` on array port produces array operand
- `NestedOut` on array output produces array operand
- The existing `emitIndex` / `emitSetElement` already work on `array_reg`
  operands via linear-memory load/store; no changes there.
- Parallel handling for parent→child wiring and session-level I/O.

### 8. `compiler/interpret_resolved.ts`

Parallel changes; mostly trivial since the interpreter already handles
`number | boolean | number[]` values. The `inputRef` case needs to
look up array values from the environment when the port is array-typed;
the env needs an `inputArrays` map populated by the session-level
wiring layer.

## Tests to unblock once Phase 1 lands

These are currently `.skip`ped due to the array-I/O limitation:

- `compiler/sequencer.test.ts:27` — `Sequencer<4> compiles end-to-end`
  (the test body is currently empty; needs to be written to actually
  exercise the path)
- `compiler/apply_plan.test.ts:304` — `Clock module through flat
  runtime produces output`
- `compiler/bubble_cloud.test.ts:53` — `BubbleCloud JIT matches
  interpreter bit-exact`
- `tests/equiv/migration_audio.test.ts` `KNOWN_UNSUPPORTED` —
  `patch_int_seq_test.json`, `patch_sequencer_demo.json`,
  `stdlib_sequencer.json`
- `compiler/ir/n_write_slot_expansion.test.ts` — newly `.skip`ped;
  needs rewriting for the new shape

## Phases 2-3 (post Phase 1)

Per the approved plan:
- Phase 2: unblock the skipped tests + add `tests/equiv/array_io_vs_oracle.test.ts`
- Phase 3: Sequencer<1024> stress test + bench

## How to verify progress

After each consumer is updated, run:

```bash
cd ../tropical-arrays
bun test compiler/session_slots.test.ts  # data layer
bun test compiler/sequencer.test.ts      # smoke: just the type tests pass
make build && bun test                    # full suite, requires native lib
```

Once Sequencer<4> can compile + run, the work is meaningfully complete
and the equivalence tests should engage.
