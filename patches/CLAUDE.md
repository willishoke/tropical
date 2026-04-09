# patches/

Example patches in JSON format. Load via `make mcp-ts` → `load` tool (also accepts the deprecated `load_patch`), or from the TUI.

Both `tropical_program_1` and `tropical_patch_1` formats are accepted. Existing patches use the legacy `tropical_patch_1` schema.

## Schema: `tropical_patch_1`

```json
{
  "schema": "tropical_patch_1",
  "modules": [
    {"type": "VCO", "name": "VCO1"}
  ],
  "outputs": [
    {"module": "VCO1", "output": "sin"}
  ],
  "input_exprs": [
    {"module": "VCO1", "input": "freq", "expr": 440}
  ]
}
```

### Fields

- **modules** — list of `{type, name}`. Type must match a registered module type (PascalCase: `VCO`, `Clock`, `ADEnvelope`, `VCA`, etc.). Name is a unique instance identifier.
- **outputs** — list of `{module, output}` specifying which module outputs are mixed to the audio output buffer.
- **input_exprs** — list of `{module, input, expr}` setting module input expressions.

### Expression format

Expressions can be:
- **Literal number** — `440`, `0.5`
- **Module output reference** — `{"op": "ref", "module": "VCO1", "output": "sin"}`
- **Binary operation** — `{"op": "mul", "args": [<expr>, <expr>]}`
- **Unary operation** — `{"op": "neg", "args": [<expr>]}`

Available ops: `add`, `sub`, `mul`, `div`, `mod`, `pow`, `neg`, `abs`, `sin`, `log`, `lt`, `lte`, `gt`, `gte`, `clamp`, `array_pack`, `matmul`.

### Common module types and their I/O

| Type | Inputs | Outputs |
|------|--------|---------|
| VCO | freq | sin, saw, tri, square |
| Clock | freq | output |
| ADEnvelope | gate, attack, decay | env |
| VCA | audio, cv | out |
