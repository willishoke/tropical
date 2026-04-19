# patches/

Example patches in `tropical_program_1` JSON format. Load via `make mcp-ts` → `load` tool, or from the TUI.

## Schema: `tropical_program_1`

```json
{
  "schema": "tropical_program_1",
  "name": "MyPatch",
  "instances": {
    "Osc1": { "program": "Sin", "inputs": { "x": 440 } }
  },
  "audio_outputs": [
    { "instance": "Osc1", "output": "out" }
  ]
}
```

### Fields

- **instances** — map of `name → { program, inputs }`. Program must match a registered type (PascalCase: `Sin`, `Clock`, `LadderFilter`, `VCA`, etc.).
- **audio_outputs** — list of `{ instance, output }` specifying which outputs are mixed to the audio output buffer.
- **programs** — (optional) inline subprogram definitions, used before they appear in `instances`.
- **params** — (optional) named control parameters with initial values and smoothing time constants.

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
