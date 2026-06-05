# patches/

Example patches in `tropical_program_2` JSON format. Load via
`make mcp-ts` → MCP `load` tool, or precompile a curated subset for
the browser demo via `bun web/build_patches.ts`.

## Schema: `tropical_program_2`

```json
{
  "schema": "tropical_program_2",
  "name": "MyPatch",
  "body": {
    "op": "block",
    "decls": [
      { "op": "instance_decl", "name": "Osc1", "program": "SinOsc", "inputs": { "freq": 440 } }
    ],
    "assigns": [
      { "op": "output_assign", "name": "dac.out",
        "expr": { "op": "ref", "instance": "Osc1", "output": "sine" } }
    ]
  }
}
```

### Fields

- **body.decls** — ordered list of `reg_decl`, `delay_decl`,
  `instance_decl`, `program_decl`. `instance_decl.program` must match
  a registered type (PascalCase: `Sin`, `Clock`, `LadderFilter`,
  `VCA`, …). For generic types pass `type_args: { N: 8 }` etc.
- **body.assigns** — `output_assign` (a wire to a named output port)
  and `next_update` (a register/delay update). Empty at the top level
  of an audio-only patch.
- **audio_outputs** — list of `{ instance, output }` mixed into the
  mono audio bus. **Legacy.** The modern way is to wire into the
  reserved `dac` instance via `output_assign{name: "dac.out"}` in
  `body.assigns` (or via the MCP `wire` tool with
  `instance: "dac", input: "out"`). Migration is gated on the
  snake_case → camelCase ingest normalization (Phase D5); the schema
  audit grandfathers existing patches that still use `audio_outputs`.
  See `compiler/schema_audit.test.ts` — adding a new patch with
  `audio_outputs` will fail that audit until you append it to the
  grandfathered list.
- **params** — *(optional, deprecated for new patches)* named control
  parameters with initial values and smoothing time constants. New
  patches register params via the MCP `set_param` tool instead.
- **ports** — *(optional)* port declarations for reusable composite
  programs. Top-level patches don't need them.

### Expression format

Input expressions are MCP wire-format `ExprNode`s — the same shape
the materializer reads when translating session wiring into resolved
IR. The closed op set lives in `compiler/expr.ts:WireFormatOp`.

- **Literal number / boolean** — `440`, `0.5`, `true`
- **Inline array** — `[110, 220, 330, 440]`
- **Instance output reference** — `{"op": "ref", "instance": "Osc1", "output": "out"}`
- **Binary operation** — `{"op": "mul", "args": [<expr>, <expr>]}`
- **Unary operation** — `{"op": "neg", "args": [<expr>]}`
- **Ternary** — `{"op": "select", "args": [<cond>, <then>, <else>]}` /
  `{"op": "clamp", "args": [<v>, <lo>, <hi>]}`
- **Sentinels** — `{"op": "sample_rate"}`, `{"op": "sample_index"}`
- **Param** — `{"op": "param", "name": "cutoff"}`. The materializer
  resolves the name to an FFI handle at compile time. Legacy
  `{"op": "trigger", "name": "kick"}` refs are still accepted on the
  wire and aliased to `{op:'param'}` at materialization.

Available scalar ops: `add`, `sub`, `mul`, `div`, `mod`, `floor_div`,
`neg`, `abs`, `sqrt`, `ldexp`, `float_exponent`, `lt`, `lte`, `gt`,
`gte`, `eq`, `neq`, `and`, `or`, `bit_and`, `bit_or`, `bit_xor`,
`lshift`, `rshift`, `bit_not`, `not`, `clamp`, `select`, `index`,
`array_set`, `to_int`, `to_bool`, `to_float`, `round`, `floor`,
`ceil`. Transcendentals (`sin`, `cos`, `tanh`, `exp`, `log`, `pow`)
are stdlib programs (`stdlib/*.md`); instantiate one and reference
its output via `ref`.

### Common program types and their I/O

See `stdlib/README.md` for the full catalogue. The most-used handful:

| Type           | Inputs                                        | Outputs                  |
|----------------|-----------------------------------------------|--------------------------|
| `Sin / Cos / Tanh / Exp / Log` | `x`                                | `out`                    |
| `Pow`          | `x`, `y`                                      | `out`                    |
| `SinOsc`       | `freq`                                        | `sine`                   |
| `Clock`        | `freq`, `ratios_in: float[N]`                 | `output`, `ratios_out`   |
| `VCA`          | `audio`, `cv`                                 | `out`                    |
| `OnePole`      | `input`, `g`                                  | `out`                    |
| `LadderFilter` | `input`, `cutoff`, `resonance`, `drive`       | `lp`, `bp`, `hp`, `notch`|
| `SoftClip`     | `input`, `drive`                              | `out`                    |
| `Delay<N>`     | `x`                                           | `y`                      |
| `Sequencer<N>` | `clock`, `values: float[N]`                   | `value`                  |
