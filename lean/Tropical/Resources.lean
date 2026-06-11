import Lean.Data.Json
import Tropical.Session

/-!
# MCP resources + prompts (Phase 6 stage 6f)

Port of `mcp/resources.ts`: the static resource/prompt texts (embedded
verbatim — the format doc includes the rendered
`PROGRAM_FORMAT_EXAMPLE` JSON) and the program-catalog renderer over
the engine's own catalog mirror.
-/

namespace Tropical.Resources

open Lean (Json toJson)

def resourcesList : Json := Json.arr #[
  Json.mkObj [
    ("uri", Json.str "tropical://programs"),
    ("name", Json.str "Program catalog"),
    ("description", Json.str "Markdown catalog of all registered program types with inputs, outputs, and default values."),
    ("mimeType", Json.str "text/markdown")],
  Json.mkObj [
    ("uri", Json.str "tropical://program-format"),
    ("name", Json.str "Program format"),
    ("description", Json.str "Reference doc for the tropical_program_2 schema."),
    ("mimeType", Json.str "text/markdown")]]

def promptsList : Json := Json.arr #[
  Json.mkObj [
    ("name", Json.str "build-patch"),
    ("description", Json.str "Three-tiered workflow guidance for building and editing tropical patches efficiently.")]]

def programFormatDoc : String := r#"# tropical program format

Programs are the unified representation for all DSP in tropical. A program is an
expression with declared ports and a body that declares regs/delays/instances
and assigns outputs and next-tick register updates.

## tropical_program_2

The body is a `block` of `decls` (reg_decl, delay_decl, instance_decl,
program_decl) and `assigns` (output_assign, next_update). Ports, type_params,
and breaks_cycles sit alongside the body. **Audio output is an `output_assign`
in the body with name `"dac.out"`** — wire the signal you want heard to it.
Session metadata — `params`, `config` — is top-level.

{
  "schema": "tropical_program_2",
  "name": "MyPatch",
  "body": {
    "op": "block",
    "decls": [
      {
        "op": "programDecl",
        "name": "Saw",
        "program": {
          "op": "program",
          "name": "Saw",
          "ports": {
            "inputs": [
              "freq"
            ],
            "outputs": [
              "out"
            ]
          },
          "body": {
            "op": "block",
            "decls": [
              {
                "op": "regDecl",
                "name": "phase",
                "init": 0
              }
            ],
            "assigns": [
              {
                "op": "outputAssign",
                "name": "out",
                "expr": {
                  "op": "sub",
                  "args": [
                    {
                      "op": "mul",
                      "args": [
                        2,
                        {
                          "op": "reg",
                          "name": "phase"
                        }
                      ]
                    },
                    1
                  ]
                }
              },
              {
                "op": "nextUpdate",
                "target": {
                  "kind": "reg",
                  "name": "phase"
                },
                "expr": {
                  "op": "mod",
                  "args": [
                    {
                      "op": "add",
                      "args": [
                        {
                          "op": "reg",
                          "name": "phase"
                        },
                        {
                          "op": "div",
                          "args": [
                            {
                              "op": "input",
                              "name": "freq"
                            },
                            {
                              "op": "sampleRate"
                            }
                          ]
                        }
                      ]
                    },
                    1
                  ]
                }
              }
            ],
            "value": null
          }
        }
      },
      {
        "op": "instanceDecl",
        "name": "osc",
        "program": "Saw",
        "inputs": {
          "freq": 440
        }
      },
      {
        "op": "instanceDecl",
        "name": "filt",
        "program": "LadderFilter",
        "inputs": {
          "input": {
            "op": "ref",
            "instance": "osc",
            "output": "out"
          },
          "cutoff": 2000
        }
      }
    ],
    "assigns": [
      {
        "op": "outputAssign",
        "name": "dac.out",
        "expr": {
          "op": "ref",
          "instance": "filt",
          "output": "lp"
        }
      }
    ],
    "value": null
  }
}

Key fields: schema, name, ports (inputs/outputs/type_defs), body (block),
type_params, sample_rate, breaks_cycles, and top-level session metadata
(params, config). Send a signal to the speakers with a body `output_assign`
named `"dac.out"`. (File-root `audio_outputs` is deprecated — don't use it.)

Decl node shapes:
- reg_decl:      { op, name, init, type? }
- delay_decl:    { op, name, update, init? }
- instance_decl: { op, name, program, inputs?, type_args? }
- program_decl:  { op, name, program: <program node> }

Assign node shapes:
- output_assign: { op, name, expr }
- next_update:   { op, target: { kind: "reg"|"delay", name }, expr }

## Expression node format (ExprNode)

Used in instance input wiring and inline program process definitions.

- **Literal**: `3.14` or `true`
- **Reference**: `{ "op": "ref", "instance": "osc", "output": "out" }` — routes another instance's output to this input
- **Input port**: `{ "op": "input", "name": "freq" }` — (inside program definitions only)
- **Param**: `{ "op": "param", "name": "cutoff" }`
- **Binary**: `{ "op": "mul", "args": [<expr>, <expr>] }` — add, sub, mul, div, floor_div, mod, pow; lt, lte, gt, gte, eq, neq; bit_and, bit_or, bit_xor, lshift, rshift
- **Unary**: `{ "op": "neg", "args": [<expr>] }` — neg, abs, not, bit_not
- **Clamp / Select / Array / Index / Delay / Builtins**: see the program catalog and stdlib for worked examples.
"#

def buildPatchPrompt : String := r#"# build-patch workflow

Before writing any program, always fetch both resources:
- `tropical://programs` — full catalog of available program types with inputs, defaults, and outputs
- `tropical://program-format` — tropical_program_2 schema reference with a worked example

## Choose the right tool for the job

### New program (starting from scratch)
Use `load` with a **complete** tropical_program_2 JSON object in a single call.
Do not call `add_instance` and `wire` one-by-one — that requires many round
trips and recompiles the JIT kernel on every change.

### Extending an existing program (adding new instances)
Use `merge` with a partial program containing only the new instances and wiring.
Then use `wire` to connect the new instances to existing ones.

### Targeted edits (changing a value, tweaking a param)
Use `wire` with a single set entry, or `set_param` for control parameters.

## Writing programs

- Use named ports (e.g. `"output": "out"`) rather than integer indices.
- Wire everything in one program object where possible; minimize round trips.
- Audio output goes through a body `output_assign` named `"dac.out"` (or the
  `wire` tool with `instance: "dac", input: "out"`).
"#

/-- Port of `renderProgramCatalog` over the engine catalog (concrete
    entries only — the TS source iterated `session.typeRegistry`). -/
def renderProgramCatalog (st : Tropical.SessionSt) : String := Id.run do
  let mut lines : Array String := #["# tropical program catalog\n"]
  for name in st.catalogOrder do
    match st.programs.get? name with
    | some pm =>
      if pm.generic then
        continue
      let inputParts := pm.inputs.map fun p =>
        match p.default with
        | some v => s!"{p.name}={v.compress}"
        | none => p.name
      let inputsJoined := String.intercalate ", " inputParts.toList
      let outputsJoined := String.intercalate ", " pm.outputNames.toList
      lines := lines.push s!"## {name}"
      lines := lines.push s!"Inputs:  {inputsJoined}"
      lines := lines.push s!"Outputs: {outputsJoined}"
      lines := lines.push ""
    | none => continue
  return String.intercalate "\n" lines.toList

/-- `readResourceText` by uri. -/
def readResourceText (st : Tropical.SessionSt) (uri : String) : Except String String :=
  if uri == "tropical://programs" || uri == "tropical://modules" then
    .ok (renderProgramCatalog st)
  else if uri == "tropical://program-format" || uri == "tropical://patch-format" then
    .ok programFormatDoc
  else
    .error s!"Unknown resource: {uri}"

/-- `getPromptMessages` by name. -/
def getPromptMessages (name : String) : Except String Json :=
  if name == "build-patch" then
    .ok <| Json.mkObj [("messages", Json.arr #[
      Json.mkObj [("role", Json.str "user"),
        ("content", Json.mkObj [("type", Json.str "text"),
          ("text", Json.str buildPatchPrompt)])]])]
  else
    .error s!"Unknown prompt: {name}"

end Tropical.Resources
