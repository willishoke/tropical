# patches/

Example patches in `tropical_program_2` JSON format. Load via `make mcp-ts` → `load` tool, or from the TUI.

## Schema: `tropical_program_2`

```json
{
  "schema": "tropical_program_2",
  "name": "MyPatch",
  "body": {
    "op": "block",
    "decls": [
      { "op": "instance_decl", "name": "Osc1", "program": "Sin", "inputs": { "x": 440 } }
    ],
    "assigns": []
  },
  "audio_outputs": [
    { "instance": "Osc1", "output": "out" }
  ]
}
```

### Fields

- **body.decls** — ordered list of `reg_decl`, `delay_decl`, `instance_decl`, `program_decl`. Instance programs must match a registered type (PascalCase: `Sin`, `Clock`, `LadderFilter`, `VCA`, etc.).
- **body.assigns** — `output_assign` and `next_update` entries. Empty at the top level of a patch.
- **audio_outputs** — list of `{ instance, output }` specifying which outputs are mixed to the audio output buffer.
- **params** — (optional) named control parameters with initial values and smoothing time constants.
- **ports** — (optional) port declarations for reusable composite programs.

### Expression format

Input expressions can be:
- **Literal number** — `440`, `0.5`
- **Instance output reference** — `{"op": "ref", "instance": "Osc1", "output": "out"}`
- **Binary operation** — `{"op": "mul", "args": [<expr>, <expr>]}`
- **Unary operation** — `{"op": "neg", "args": [<expr>]}`

Available ops: `add`, `sub`, `mul`, `div`, `mod`, `neg`, `abs`, `sqrt`, `ldexp`, `float_exponent`, `lt`, `lte`, `gt`, `gte`, `clamp`, `select`, `array_pack`, `matmul`. Transcendentals (`sin`, `cos`, `tanh`, `exp`, `log`, `pow`) are stdlib programs — instantiate them and read via `nested_out`.

### Common program types and their I/O

| Type | Inputs | Outputs |
|------|--------|---------|
| Sin / Cos / Tanh | x | out |
| Exp / Log | x | out |
| Pow | x, y | out |
| Clock | freq | output |
| VCA | audio, cv | out |
| LadderFilter | input, cutoff, resonance, drive | lp |
| OnePole | input, cutoff | out |
| SoftClip | input, drive | out |
