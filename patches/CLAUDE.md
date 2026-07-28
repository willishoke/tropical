# patches/

Example patches in `tropical_program_2` JSON format. Loaded and
compiled by the Lean engine: launch the MCP server (`make mcp-lean`)
and use the `load` tool, or compile one directly with
`diffcli compile <patch.json>`. The curated browser-demo subset lives
under `web/patches/` and is precompiled via `bun web/build_patches.ts`.

## Schema: `tropical_program_2`

```json
{
  "schema": "tropical_program_2",
  "name": "MyPatch",
  "body": {
    "op": "block",
    "decls": [
      { "op": "instanceDecl", "name": "Osc1", "program": "SinOsc", "inputs": { "freq": 440 } }
    ],
    "assigns": [
      { "op": "outputAssign", "name": "dac.out",
        "expr": { "op": "ref", "instance": "Osc1", "output": "sine" } }
    ]
  }
}
```

### Fields

- **body.decls** — ordered `instanceDecl` and parameter metadata over
  already-registered program types. `instanceDecl.program` must match a
  registered type (PascalCase: `Sin`, `ModalVoice`, `FixedSinOsc`, `VCA`, …).
  Generics are retired — there are no `type_args`. A `programDecl` is rejected
  at ingest; register and delay declarations are not part of the current
  patch-bay language.
- **body.assigns** — `outputAssign` wires expressions to named output
  boundaries, including `dac.out`. There is no next/update assignment.
- **body.decls parameter declarations** — `paramDecl` entries declare
  named control parameters. The MCP `set_param` tool changes values after
  load.
- **ports** — *(optional)* port metadata retained for serialization.
  Top-level patches don't need it; a loaded patch cannot use this field to
  define a new program type.

The retired root fields `audio_outputs`, `params`, and `breaks_cycles` are
rejected. Express audio routing with a body `outputAssign` to `dac.out`,
parameters with body `paramDecl` entries, and graphs as closed-form acyclic
wiring.

### Expression format

Input expressions are MCP wire-format `ExprNode`s — the same shape the
session compile reads when translating wiring into resolved IR. The
grammar is a closed inductive, `Tropical.WireExpr`
(`lean/Tropical/WireExpr.lean`), and its JSON decoder is the single
refusal site: what decodes is exactly what compiles. Anything else —
state ops (`delay`, `reg`), retired combinators (`fold`, `scan`), the
old array/functional ops (`map`, `matmul`, `reduce`, `zeros`, …) — is
rejected at ingest.

The serialized plan boundary is independently strict. See
[`design/compatibility-matrix.md`](../design/compatibility-matrix.md).

- **Literal number / boolean** — `440`, `0.5`, `true`
- **Inline array** — `[110, 220, 330, 440]`
- **Instance output reference** — `{"op": "ref", "instance": "Osc1", "output": "out"}`.
  `output` is a port name or a positional index.
- **Binary operation** — `{"op": "mul", "args": [<expr>, <expr>]}`
- **Unary operation** — `{"op": "neg", "args": [<expr>]}`
- **Ternary** — `{"op": "select", "args": [<cond>, <then>, <else>]}` /
  `{"op": "clamp", "args": [<v>, <lo>, <hi>]}`
- **Sentinels** — `{"op": "sampleRate"}`, `{"op": "sampleIndex"}`,
  `{"op": "clock"}`
- **Param** — `{"op": "param", "name": "cutoff"}`. The session compile
  resolves the name to a `param:<name>` module slot.

Op names are **camelCase** on the wire, not snake_case. Available
scalar ops: `add`, `sub`, `mul`, `div`, `mod`, `floorDiv`, `neg`,
`abs`, `sqrt`, `ldexp`, `floatExponent`, `lt`, `lte`, `gt`, `gte`,
`eq`, `neq`, `and`, `or`, `bitAnd`, `bitOr`, `bitXor`, `lshift`,
`rshift`, `bitNot`, `not`, `clamp`, `select`, `index`, `arraySet`,
`toInt`, `toBool`, `toFloat`, `round`, `floor`, `ceil`.
Alternate spellings—including `paramExpr`, `triggerParamExpr`, `trigger`,
`array`/`arrayLiteral` object forms, `sampleClock`/`sample_clock`,
`array_set`, and `sessionSlot`/`sessionArraySlot`—are retired and rejected.
Transcendentals (`sin`, `cos`, `tanh`, `exp`, `log`, `pow`) are
`Tropical.Stdlib` builder programs
(`lean/Tropical/Stdlib.lean`, booted directly by the engine);
instantiate one and reference its output via `ref`.

Arity is exact: binary ops require exactly 2 `args`, unary exactly 1,
ternary exactly 3. A wrong count is refused at the tool boundary with
`invalid_value`, not at the next compile.

### Common program types and their I/O

See `Tropical.Stdlib` (`lean/Tropical/Stdlib.lean`) for the full
catalogue of builder programs. The most-used handful:

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
