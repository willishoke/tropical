# patches/

Example patches in `tropical_program_1` JSON format. Load via `make mcp-ts` → `load` tool, or from the TUI.

## Schema: `tropical_program_1`

```json
{
  "schema": "tropical_program_1",
  "name": "MyPatch",
  "instances": {
    "VCO1": { "program": "VCO", "inputs": { "freq": 440 } }
  },
  "audio_outputs": [
    { "instance": "VCO1", "output": "sin" }
  ]
}
```

### Fields

- **instances** — map of `name → { program, inputs }`. Program must match a registered type (PascalCase: `VCO`, `Clock`, `ADEnvelope`, `VCA`, etc.).
- **audio_outputs** — list of `{ instance, output }` specifying which outputs are mixed to the audio output buffer.
- **programs** — (optional) inline subprogram definitions, used before they appear in `instances`.
- **params** — (optional) named control parameters with initial values and smoothing time constants.

### Expression format

Input expressions can be:
- **Literal number** — `440`, `0.5`
- **Instance output reference** — `{"op": "ref", "instance": "VCO1", "output": "sin"}`
- **Binary operation** — `{"op": "mul", "args": [<expr>, <expr>]}`
- **Unary operation** — `{"op": "neg", "args": [<expr>]}`

Available ops: `add`, `sub`, `mul`, `div`, `mod`, `pow`, `neg`, `abs`, `sin`, `log`, `lt`, `lte`, `gt`, `gte`, `clamp`, `array_pack`, `matmul`.

### Common program types and their I/O

| Type | Inputs | Outputs |
|------|--------|---------|
| VCO | freq | sin, saw, tri, square |
| Clock | freq | output |
| ADEnvelope | gate, attack, decay | env |
| VCA | audio, cv | out |
