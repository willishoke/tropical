Status: Current

# Compatibility matrix

This matrix separates Tropical's closed-form production compiler from
compatibility and dead state-shaped surfaces below its waist. It is based on
source reachability at commit `9c492dd` and should be re-audited when a schema
or public load API changes.

## Reading the reachability columns

- **Y**: the path produces or intentionally consumes the row.
- **N**: the row is not reachable from that path.
- **Empty**: the path carries the current empty/default form but cannot make
  the legacy feature effective.
- **C API**: reachable only when an external native caller deliberately
  supplies it.

“MCP/session” includes `load`, `merge`, ordinary MCP graph mutations, and
`export_program`. “Playground” is `load_patch_graph`. `diffcli` means its
compile/render/emit verbs. “Web build” means `web/build_patches.ts` plus the
browser runtime.

## Reachability and classification

| Surface | Classification | Lean builder | Playground | MCP/session | `diffcli` | Web build | Direct C API | Owner / review |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `tropical_program_2` patch document | production-required | N | N | Y | Y | Y | N | Lean engine / continuous |
| `programDecl` inside `program_2` | historical/dead | N | N | refused | refused | refused | N | Lean engine; removal means deleting refusal fixture only / 2027-01-01 |
| `Tropical.WireExpr` state spellings (`reg`, `delay`, `delayRef`, `delayValue`) | historical/dead as source features | N | N | refused by decoder | refused | refused | N | Lean engine / continuous refusal gate |
| `tropical_plan_5` typed/wire plan | production-required | Y | Y | Y | Y | Y | Y | Compiler/runtime / continuous |
| `tropical_plan_4` manifest dispatch in `FlatRuntime::load_ir_staged` | compatibility-supported | N | N | N | N | N | C API | Runtime maintainers; external-caller inventory required / 2026-10-01 |
| Plan-4 single `instructions[]` lift to one `InstanceProgram` | test substrate | N | N | N | N | N | C API | Runtime maintainers / 2026-10-01 |
| Plan-4 `config.sampleRate`, `register_count`, array/slot metadata | compatibility-supported | N | N | N | N | N | C API | Runtime maintainers / 2026-10-01 |
| Plan-4 `output_targets` + `outputs` temp-mix metadata | compatibility-supported | N | N | N | N | N | C API | Runtime maintainers / 2026-10-01 |
| Legacy operand kinds `rate` / `tick` | compatibility-supported | N | N | N | N | N | C API | Runtime maintainers / 2026-10-01 |
| Missing instruction `dst_kind` inference | compatibility-supported | N | N | N | N | N | C API | Runtime maintainers / 2026-10-01 |
| Current scalar, slot, array, and reduce instructions | production-required | Y | Y | Y | Y | Y | metadata only | Compiler/backends / continuous |
| `SmoothParam` parser tag | historical/dead | N | N | N | N | N | parsed metadata only | Runtime maintainers; remove with dead plan-instruction parser / 2026-10-01 |
| `OperandKind::StateReg`, `Operand::make_state` | historical/dead | N | N | N | N | N | N | Runtime maintainers; safe-removal candidate / 2026-10-01 |
| `InstanceProgram::writebacks` | historical/dead | N | N | N | N | N | N | Runtime maintainers; safe-removal candidate / 2026-10-01 |
| `FlatProgram::register_targets` / `register_types` | historical/dead | N | N | N | N | N | N | Runtime maintainers; safe-removal candidate / 2026-10-01 |
| Plan keys `state_init`, `register_names`, `register_types`, `register_targets`, `state_reg_offset` on native load | historical/dead | N | N | N | N | N | ignored | Runtime maintainers; reject-or-document decision / 2026-10-01 |
| Native `KernelState::registers` ABI buffer | test substrate | Empty | Empty | Empty | Empty | Empty | caller IR may address ABI pointer, but runtime allocates zero cells | Runtime/ABI owner / 2027-01-01 |
| Native per-sample state allocation | historical/dead | N | N | N | N | N | N | Runtime maintainers |
| Hot-swap transfer of registers, arrays, or slots | historical/dead | N | N | N | N | N | N | Runtime maintainers |
| Hot-swap transfer of `sample_index` | production-required | Y | Y | Y | Y | N (player has no hot-swap) | Y | Runtime maintainers / continuous |
| Reapplication of current parameter values through plan defaults/control host | production-required | Y | Y | Y | Y | built defaults | caller-owned | Frontend/host owners / continuous |
| Web `KernelManifest.registerTypes` / `stateInit` fields | compatibility-supported | Empty | Empty | Empty | Empty | Empty | N | Web runtime; extraction/API review / 2026-10-01 |
| Web register-memory initialization loop | sister-runtime candidate | N | N | N | N | empty input | N | Web runtime; do not treat as Tropical semantics / 2026-10-01 |
| `tests/fixtures/flat_plan/stdlib_sin.json.expected_plan` plan-4 snapshot | historical/dead | N | N | N | N | N | N | Test owners; migration record / 2026-10-01 |
| CTest `current_module_process` | production-required | N | N | N | N | N | exercises current C API/IR ABI | Runtime test owners / continuous |
| CTest `compat_legacy_plan4_manifest` | test substrate | N | N | N | N | N | exercises bounded plan-4 acceptance | Runtime test owners / while plan-4 is retained |
| Docs claiming source registers, delay declarations, or state transfer | historical/dead | N | N | N | N | N | N | Documentation owner; stale-token audit |

## What plan 4 actually accepts

The dispatch is in
[`FlatRuntime::load_ir_staged`](../engine/runtime/FlatRuntime.cpp). All three
public loads reach it:

- `tropical_runtime_load_ir`;
- `tropical_runtime_load_ir_msl`;
- `tropical_runtime_load_ir_staged`.

For `schema: "tropical_plan_4"`,
[`NumericProgramParser::parse_plan4`](../engine/runtime/NumericProgramParser.hpp)
reads sample rate, compilation mode, scratch count, array sizes/names,
legacy output mixing, slots/defaults, and an instruction list. It upgrades
legacy `rate`/`tick` operands and infers a missing `dst_kind`.

The caller, however, also supplies the LLVM IR that defines behavior.
`OrcJitEngine::compile_ir_text` compiles that IR; the parsed instruction list
does not generate code. State initialization, state-register metadata, and
register writebacks are not read. `FlatRuntime::build_kernel_state` clears the
register buffer, and `publish_state` carries only `sample_index`.

Consequences:

1. A plan-4 document cannot activate state by adding `state_init` or
   `register_targets`.
2. A direct native caller can supply arbitrary valid LLVM using the public
   kernel ABI, but that is caller-authored code, not a Tropical source or plan
   capability.
3. Retaining the plan-4 branch currently preserves metadata compatibility, not
   the historical state machine.

## Production non-emission

The `production-non-emission` tropicaltest gate inspects in-memory
`FlatPlan`/`PerInstancePlan` values from representative builder, playground,
session/patch, and export-derived paths. It rejects legacy state/update tags
before serialization. It then uses the wire form only to assert
`schema == "tropical_plan_5"`.

The existing `patch-bay-refusal` and `cycle-refusal` gates cover the other two
parts of the boundary: a patch cannot define a program body, and source/session
back-edges do not reach any backend.

The dedicated CTest `compat_legacy_plan4_manifest` proves the deliberately
retained opposite direction: a direct C caller can pair a plan-4 metadata
manifest with a closed-form LLVM kernel and load it successfully. Its name and
test output keep that fact quarantined from production-language tests.

## Removal-cost analysis

The retained native plan-4 branch is small and off the audio callback. It adds:

- one schema dispatch arm;
- one plan-4 metadata parser/lift;
- legacy operand and destination inference;
- a CTest;
- dead instruction/state-shaped C++ data types shared with the metadata parser.

It does not add per-sample execution work because emitted LLVM owns execution.
Deleting only the state-shaped fields would reduce review and trust surface but
would not measurably improve runtime or compile latency. Deleting the plan-4
branch without an external-caller inventory risks silently breaking a direct C
consumer.

The browser manifest's `stateInit`/`registerTypes` carrier is likewise not fed
by current builds, but it is part of the extractable runtime package's public
shape. Its cost is a small allocation/initialization branch at instantiation,
not audio-loop work.

## Recommendation

Retain the bounded plan-4 metadata branch until the 2026-10-01 review. Before
that date:

1. inventory direct C API consumers;
2. decide whether ignored state keys should be rejected instead of silently
   ignored;
3. remove the dead `StateReg`, writeback, and `FlatProgram` state fields in a
   separate behavior-preserving change;
4. decide whether the WebAssembly manifest's state initializer belongs in an
   extracted stateful sister runtime or should leave Tropical's public runtime
   shape;
5. if no plan-4 owner remains, announce deprecation and remove the branch in a
   separately reviewed release.

The reusable ABI/data structures are sister-runtime candidates, not permission
to reintroduce registers, delays, or feedback into Tropical's source language.
