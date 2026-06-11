# Bugs surfaced by the Lean port

Production defects found while porting the engine to Lean
(`design/lean_port.md`). None were found by the existing test suite;
all were found by the port's differential gates — running two
independent implementations of the same contract against each other
and treating any divergence as a defect in one of them. This file is
append-only: new findings go at the bottom with the same structure.

A pattern worth naming up front: every entry below involves **state
that existed in two representations** — pre- vs post-extraction wires,
Param handles vs `param:` slots, registries accumulated across
recompiles vs rebuilt fresh. The TS engine compiled by mutating the
session in place, so the second representation leaked into the first
across recompiles. The port's canonical-form discipline (the Lean
session stores wires in authored pre-extraction form, and every
compile lowers from scratch) removed the class, and the differential
gates caught each instance where the two designs disagreed.

---

## 1. Re-wiring any input failed after its first compile

**Symptom.** Setting a wire on a port that had already been compiled
failed with `internal_error: duplicate reg/delay '__autodelay:<key>'`.
Since every MCP mutation compiles, this meant *any second `wire` call
to the same input* errored. The previous kernel kept playing, so the
failure presented as edits silently not taking effect.

**Mechanism.** `extractSessionDelays` is a compile-time pass that
mutates the session: it hoists each `delay()`-wrapped wire into
`session.delaySlotRegistry` and rewrites the wire to a `sessionSlot`
read. The registry entry persisted across compiles. Re-setting the
wire stored a fresh `delay()` with the same auto-generated id; the
next compile re-extracted it, appending a second registry entry with
the same `slotName`; the root lowering then declared two regs with one
name and threw.

**Why tests missed it.** No test re-wired a port after a compile had
run. The MCP test suites built graphs monotonically.

**Found by.** `make diff-engine` step parity: the Lean engine
(rebuilding slot state per compile by construction) succeeded where
the TS engine errored, on a recorded script that re-wired `osc1.freq`.

**Fix.** `wire()` canonicalizes wires (`reconstructWireDelays`) and
rebuilds the slot registries before every compile — the same reset the
compiler service performs per snapshot. Commit `999ef14`.

---

## 2. `combine` double-delayed the existing source

**Symptom.** `wire` with `combine` (fan-in onto an occupied input)
gave the previously-wired source two samples of latency instead of
one — but only if a compile had run between the two wire calls, i.e.
always, in practice.

**Mechanism.** The combine path strips the stored wire's auto-delay
wrapper (`unwrapDelay`) before composing, so the merged expression
carries a single outer delay. But after a compile, the stored wire was
no longer a `delay()` node — extraction had rewritten it to a bare
`sessionSlot` read, which `unwrapDelay` passes through. Wrapping that
read in a fresh delay stacked one sample on top of the slot's one
sample. The code's intent (a comment explicitly explaining why
`unwrapDelay` exists) was defeated by the post-extraction
representation it never anticipated. The same representation leak made
`get_info` echo `{op:'sessionSlot',index}` internals in its `expr`
field instead of the authored form `save` emits.

**Found by.** Reading `engine.ts` against the canonical-wire design
during the Phase 1 port — the divergence was identified before the
gate could run, then pinned by it.

**Fix.** Resolve the stored wire back through `reconstructWireDelays`
before unwrapping; canonicalize the `get_info` echo the same way.
Commit `9f3376a`.

---

## 3. `set_param` was audibly inert

**Symptom.** `set_param` returned ok, `list_params` showed the new
value, and the sound did not change. The value took effect only at the
next recompile (via `slot_defaults`), and even then only for slots the
hot-swap didn't preserve by name.

**Mechanism.** On the session path, a param compiles to a
`param:<name>` module slot; the kernel reads the slot every sample.
The control plane for slots is `tropical_runtime_set_slot`. But
`set_param` wrote only the detached C `Param` object
(`tropical_param_set`) — a smoothing handle that nothing on the
session path ever reads. The handle's value reached audio only as a
`slot_defaults` seed at compile time. Verified empirically: output
unchanged after `set_param`, changed immediately after `set_slot` on
the same session.

**Why tests missed it.** Tests asserted the response envelope and the
registry value — both of which were correct. Nothing measured audio
output across a `set_param`.

**Found by.** Phase 2 FFI design work: deciding what the Lean engine's
`set_param` must call forced the question "how does the value reach
the kernel?", and the answer was that it didn't.

**Fix.** `set_param` drives the live slot
(`setSlot(slotIndex("param:<name>"), value)`) in both engines, in
addition to the handle write. Commit `adb4349`.

---

## 4. Any rebuild dropped the `param:` slots

**Symptom.** After any wire mutation following a `load`, the plan lost
its `param:<name>` slots entirely: `slot_names` no longer carried
them, `ParamRef`s lowered against an empty slot registry, and the
fix from finding 3 silently no-opped (no slot to write).

**Mechanism.** `allocateParamSlot` had exactly one caller —
`applyParamSpecs` on the load path. The compile-time rebuild paths
(the TS engine's `wire()` after finding 1's fix; the service's
snapshot rebuild) cleared `paramSlotRegistry` and reallocated outputs
and delays but never params. Params got slots at load and lost them at
the first rebuild.

**Why tests missed it.** Same blind spot as finding 3 plus
sequencing: a test would have needed `load` (params), then a wire
mutation, then an audio-effective `set_param` — three steps across two
subsystems with an audio assertion at the end.

**Found by.** Reading the allocation call graph while designing the
Phase 3 canonical allocation order (the Lean side had to know where
param slots come from, and the answer was "nowhere, after a rebuild").

**Fix.** Canonical allocation order in every rebuild path — params
first (matching `loadJSON`), then instance outputs, then extraction
delay slots — in the TS engine, the service, and the Lean lowering.
Commit `3f89907`.

---

## 5. Recompiles renamed delay registers; feedback graphs fell silent

**Symptom.** After adding a feedback edge, the TS engine rendered
exactly zero — 2048/2048 samples — while the Lean engine rendered the
expected signal. After the *next* recompile both agreed again, masking
the failure as a transient glitch.

**Mechanism.** Three pieces compose:

1. `reconstructWireDelays` (used by the canonicalize-then-rebuild
   compile from finding 1) rebuilt `delay()` nodes *without their
   `id`*. Re-extraction then minted fresh registry-length-dependent
   names (`__autodelay:vca1:audio#0` vs the original
   `__autodelay:vca1:audio`).
2. Register state transfers across hot-swap *by name*. The renamed
   register matched nothing, so it restarted from `state_init` —
   losing the live value the same wire's register held one kernel ago.
3. The patch under test computed `out = audio · cv` with `cv` fed back
   from `out`. With `audio`'s register reset to 0, the first sample's
   `out` was 0, which became the next `cv` — an **absorbing state**:
   silence regenerating silence, forever, in a kernel whose plan was
   structurally identical (modulo one register name) to the working
   one and which rendered perfectly when loaded into a fresh runtime.

**Why tests missed it.** Plan-level comparison can't see it (the plans
are equivalent up to a name); fresh-runtime rendering can't see it
(both plans render identically from `state_init`). Only a *live
hot-swap under feedback* exhibits it — a sequence no test performed.

**Found by.** The `debug_render` audio probes added to the recorded
tool scripts in Phase 3: the engine differ compares rendered samples
after every mutation, and the post-feedback probe diverged
(all-zeros vs signal).

**Fix.** Reconstruction is id-preserving: the rebuilt `delay()`
carries its slot name as `id`, so register names are stable across
recompiles and hot-swap state transfer finds them. This also makes
`save`/reload state-stable. Commit `3f89907`.

---

## 6. Wire creation order depended on client JSON key order

**Symptom.** Two semantically identical `load` payloads whose
`instanceDecl.inputs` objects listed keys in different orders produced
sessions whose wire maps iterated differently — observable in
`list_wiring` order and, via slot-allocation order, in plan layout.

**Mechanism.** `loadProgramAsSession`/`mergeProgramIntoSession` wired
instance inputs via `Object.entries(inst.inputs)` — insertion order of
the parsed JSON. JSON object key order is not a contract: clients can
emit any order, and the Lean front door's RBMap-backed `Json` sorts
keys on re-serialization, so the same patch arrived at the service in
a different iteration order than the oracle saw.

**Found by.** `make diff-engine` on the load happy-path script:
`list_wiring` entries swapped between engines.

**Fix.** Wires are created in the program's declared input-port order
— the canonical order — regardless of JSON key order. Commit
`adb4349`.

---

## 7. A second `load` could ship a plan the engine parser rejects

**Symptom.** Loading a patch after a session that had allocated input
slots (any fractal compile — i.e., every session compile since the
root lowering) or session array slots (array ports, lifted array
wires) made the new session's plan unloadable:
`runtime loadPlan failed: [json.exception.type_error.302] type must be
string, but is null`. The previous kernel kept playing; every
subsequent load of any patch failed the same way.

**Mechanism.** `loadProgramAsSession` reset `slotCount` and cleared the
output/param slot registries, but not `inputSlotRegistry`,
`inputPortMeta`, the `ioArraySlot*` triple, or `delaySlotRegistry`.
`buildSlotMetadata` then wrote stale input-slot names at indices past
the fresh session's `slot_count`, leaving array holes that
`JSON.stringify` serializes as `null` — which the C++ plan parser
correctly rejects. Stale `ioArraySlot*` entries additionally inflated
`array_slot_count` with phantom slots carrying the dead session's
names.

**Found by.** `make diff-engine` on the new `ingest.json` script
(stage 6d corpus-discrimination pass): the Lean engine — which
recomputes slot allocation from scratch on every compile and has no
persistent slot registries to leak — loaded the second patch cleanly
while the oracle returned `internal_error`.

**Fix.** `loadProgramAsSession` clears the input-slot registries, the
session array-slot space, and the delay registry alongside the rest of
the slot state.

---

## Appendix: not a code defect

- **`export_program` rejects MCP-wired subgraphs (latent, behavior
  preserved)** — exported instance inputs and exposed-port defaults are
  baked from the session's *post-extraction* wires, so any MCP-set wire
  (always delay-wrapped, hence extraction-rewritten) lands in the
  exported node as `{op:'sessionSlot'}` — which `raise` rejects
  (`raise: unknown expression op 'sessionSlot'`). Export therefore
  works for load-built sessions (raw wires) and fails for MCP-wired
  ones. Both engines reproduce this identically (gated by
  `ingest.json`'s `WiredVoice` probe at stage 6e). A real fix needs
  delay-expression support in raise/elaborate (or reconstruction before
  baking) — deferred as feature work, not silently changed during the
  port.

- **Stale golden hashes on `origin/main`** — the three committed
  `tests/golden/*.hash` values disagree with the current pipeline's
  output, identically across both engine implementations and with the
  JIT cache cleared (verified on a clean origin/main worktree). The
  goldens predate this branch; likely not regenerated after the
  DAC-as-sink/sources work. Pending a decision to re-baseline
  (`make validate-write`).
