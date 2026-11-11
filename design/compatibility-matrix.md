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
| `tropical_kernel_1` kernel manifest (the plan nested) | current artifact manifest | Y | Y | Y | Y | Y | required |
| `tropical_plan_7` typed/wire plan | current plan | Y | Y | Y | Y | Y | nested in the manifest |
| any other serialized plan schema | unsupported/refused | N | N | N | refused | N | refused |
| `tropical_plan_6` (integer time source) | unsupported/refused | N | N | N | refused | N | refused |
| `tropical_plan_5` and older plans | unsupported/refused | N | N | N | refused | N | refused |
| retired top-level state/output carriers in Plan 7 | unsupported/refused | N | N | N | refused | N | refused |
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
| Plan 6's integer `tick` time source | source 0 was the sample index | refused with a recompile message: Plan 7's source 0 is the fixed-point τ (Q48.16, `fixed-carrier.md`), so a Plan 6 document cannot be reinterpreted |
| `tau:i64q32` kernel manifests | the Q32.32 clock rail (2026-10-28 to 2026-11-01, unreleased) | refused by the native and web manifest readers: τ's element type names the rail's format, and the rail is now Q48.16 (`i64q16`) |
| the `MulLo32` scalar op | a 32-bit multiply on the i64 rail (2026-11-09 to 2026-11-10, unreleased) | refused as an unknown instruction tag by the Lean decoder (`PlanDecode`, via `FlatPlan.kindError?`, which admits only `PlanOp` tags and the structural instructions), by both Lean emitters, and by the native manifest reader's tag table; the `int32` kind's `Mul` is the 32-bit multiply |

## Additive changes within Plan 7

A change stays inside `tropical_plan_7` when every existing document still
decodes to the same function and a document using the new vocabulary is
refused, not reinterpreted, by a reader that predates it. A change that
gives an existing spelling a new meaning bumps the schema (Plan 6 → 7
redefined source 0, so a Plan 6 document is refused).

- `int32` scalar type and the `ToInt32` op (2026-11-10). A new
  `scalar_type`/`result_type` value and a new tag; no existing spelling
  changes meaning, and an older reader refuses both as unknown. Retiring
  `MulLo32` at the same time is a refusal, not a reinterpretation.
- Total integer semantics (2026-11-10). Integer results the Plan left
  undefined are now defined: a zero divisor gives 0, `MIN / -1` gives `MIN`
  and `MIN % -1` gives 0, a shift amount is taken modulo the width, and a
  float → integer conversion saturates (NaN → 0). This is a refinement: every
  result that was defined is unchanged, and no shipped program depended on
  an undefined one (the audio goldens are bit-identical).
- Kind checking at decode (2026-11-10). `PlanDecode` refuses a document
  that fails `FlatPlan.kindError?`: an unknown instruction tag, a `reg`
  read at a sort its producer did not write, an op whose result sort is not
  the one its operands give or that mixes `int32` with `int`/`float`, a
  store or array element reached by anything but widening, an `int32` read
  out of an array, and an integer constant outside its kind. Every plan the
  compiler emits passes; the check only refuses hand-edited crossings the
  compiler would refuse.

## Required rejection behavior

`Tropical.Plan.FlatPlan.ofWire` requires an explicit
`schema: "tropical_plan_7"`. The native C load tail (`tropical_runtime_load`)
requires a `schema: "tropical_kernel_1"` manifest (`design/typed-kernel.md`)
whose first input is `tau:i64q16[frames]` with `frames` equal to the
runtime's buffer length, whose one output is `f64[frames, channels]`, and
whose nested `plan` carries `schema: "tropical_plan_7"` and the `fused`
realization. Missing, unknown, and older schemas fail with an error naming
the unsupported boundary; Plan 6 additionally names its integer time source
and asks for a recompile. Plan 7 rejects these retired
top-level fields rather than ignoring them:

`state_init`, `register_names`, `register_types`, `register_targets`,
`state_reg_offset`, `output_targets`, `outputs`, top-level `instructions`, and
`scheduler_function`.

Canonical Plan-7 omissions are not legacy compatibility. The current encoder
may omit fused `compilation_mode`, the default tau/rate `sources` pair, empty
child/instruction arrays, and zero loop ids. Decoders retain those current
defaults.

Program 2 likewise has one source spelling per current feature. Root
`params`, `audio_outputs`, and `breaks_cycles` fail normalization. The wire
decoder accepts `param`, bare arrays, `clock`, and `arraySet`; the retired
aliases listed above fail as unknown operations.

## Evidence

- `current_module_process`: native schema/carrier rejection plus current ABI
  rendering.
- `plan-schema-rejection`: Lean decoder rejection of Plans 4, 5 and 6,
  unknown/missing schemas, and retired carriers.
- `production-non-emission`: representative current front doors emit typed,
  state-free Plan 7 only.
- `mcp/errors.test.ts`: removed parameter methods, Program-2 root carriers,
  and wire aliases are refused at their public boundaries.
- `param_dispatch_conformance.test.ts`: socket and Lean control-plane
  implementations still agree through the one public `set_param`.

Historical planning material remains searchable under `design/sprint-*` and
`design/archive`, but is labeled as historical or superseded where it could
otherwise imply current runtime support.
