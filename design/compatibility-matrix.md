Status: Current

# Serialized-plan and retired-surface matrix

Staff decision S-09 replaced the temporary Plan-4 quarantine with immediate
removal on 2026-07-28. There is no deprecation window and no runtime
compatibility promise for Plan 4.

Audit base: sprint integration line, 2026-07-28.

## Current boundary

| Surface | Status | Lean builder | Playground | MCP/session | `diffcli` | Web build | Direct C API |
|---|---|---:|---:|---:|---:|---:|---:|
| `tropical_program_2` patch document | current input | N | N | Y | Y | Y | N |
| `programDecl` inside `program_2` | retired/refused | N | N | refused | refused | refused | N |
| retired Program-2 root carriers | retired/refused | N | N | refused | refused | refused | N |
| retired wire aliases/state spellings | retired/refused | N | N | refused | refused | refused | N |
| `tropical_plan_5` typed/wire plan | current plan | Y | Y | Y | Y | Y | required |
| any other serialized plan schema | unsupported/refused | N | N | N | refused | N | refused |
| retired top-level state/output carriers in Plan 5 | unsupported/refused | N | N | N | refused | N | refused |
| public parameter method `set_param` | current | N | Y | Y | N | future host | socket/RPC |
| specialized parameter-method aliases | removed | N | N | refused | N | N | refused |

`tropical_program_2` remains the current patch/session input schema. This
retirement concerns older *plan* schemas and does not remove or rename the
patch-bay front door.

## Removed compatibility surface

| Removed surface | Previous role | Current evidence |
|---|---|---|
| Plan-4 branch in `FlatRuntime::load_ir_staged` | direct-C metadata compatibility | `current_module_process` rejects Plan 4 |
| `NumericProgramParser::parse_plan4` and `tropical_plan4` namespace alias | Plan-4 lift | deleted |
| legacy `rate`/`tick` operand translation | Plan-4 shorthand | parser rejects unknown operand kinds |
| missing `dst_kind` inference | Plan-4 shorthand | `dst_kind` is required |
| `output_targets`/`outputs` top-level mix lift | Plan-4 output metadata | retired carriers rejected |
| ignored state/register top-level fields | inert compatibility metadata | retired carriers rejected |
| `SmoothParam`, `StateReg`, legacy operand factories, writebacks, register targets/types | dead C++ carrier types | deleted |
| Web `stateInit`/`registerTypes` and initializer loop | always-empty sister carrier | deleted from manifest contract |
| `set_param_glide`, `set_param_freq`, `set_param_velocity` RPC methods | migration aliases | unknown-tool errors; disciplines remain internal to `set_param` |
| Program-2 root `params`, `audio_outputs`, `breaks_cycles` | parameter/output/cycle compatibility metadata | rejected; use body `paramDecl` and `outputAssign` |
| `paramExpr`, `triggerParamExpr`, `trigger` | parameter-read aliases | rejected; use `param` |
| object `array`/`arrayLiteral` | array-literal aliases | rejected; use a bare JSON array |
| `sampleClock`/`sample_clock` | time-coordinate aliases | rejected; use `clock` |
| `array_set` | array-update alias | rejected; use `arraySet` |
| `sessionSlot`/`sessionArraySlot` | serialized session-state carriers | constructors deleted; rejected as unknown ops |
| Plan-4 compatibility CTest and fixture payload | acceptance evidence | deleted; useful program input remains |

## Required rejection behavior

Both `Tropical.Plan.FlatPlan.ofWire` and the native C load tail require an
explicit `schema: "tropical_plan_5"`. Missing, unknown, and older schemas fail
with an error naming the unsupported boundary. Plan 5 rejects these retired
top-level fields rather than ignoring them:

`state_init`, `register_names`, `register_types`, `register_targets`,
`state_reg_offset`, `output_targets`, `outputs`, top-level `instructions`, and
`scheduler_function`.

Canonical Plan-5 omissions are not legacy compatibility. The current encoder
may omit fused `compilation_mode`, the default tick/rate `sources` pair, empty
child/instruction arrays, and zero loop ids. Decoders retain those current
defaults.

Program 2 likewise has one source spelling per current feature. Root
`params`, `audio_outputs`, and `breaks_cycles` fail normalization. The wire
decoder accepts `param`, bare arrays, `clock`, and `arraySet`; the retired
aliases listed above fail as unknown operations.

## Evidence

- `current_module_process`: native schema/carrier rejection plus current ABI
  rendering.
- `plan6-asset-abi`: Plan-5 wire stability; Plan-6 schema coherence and
  immutable-asset decode; readonly JIT slots; packed Metal buffers 3/4.
- `production-non-emission`: representative current front doors emit typed,
  state-free Plan 5 when no immutable asset is present.
- `mcp/errors.test.ts`: removed parameter methods, Program-2 root carriers,
  and wire aliases are refused at their public boundaries.
- `param_dispatch_conformance.test.ts`: socket and Lean control-plane
  implementations still agree through the one public `set_param`.

Historical planning material remains searchable under `design/sprint-*` and
`design/archive`, but is labeled as historical or superseded where it could
otherwise imply current runtime support.
