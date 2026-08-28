import Lean.Data.Json
import Tropical.Session

/-!
# MCP resources + prompts

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

Programs are the unified representation for all DSP in tropical. tropical is
closed-form-only: a program is a pure function of a time coordinate, with
declared ports and a body that declares params/instances and assigns outputs.
There is no per-sample state — no `reg`, no `delay`, no feedback.

## tropical_program_2

The wire format is a PATCH BAY: instances of registered program types, wiring,
and params. The body is a `block` of `decls` (`paramDecl`, `instanceDecl`) and
`assigns` (`outputAssign`). **Program definitions over the wire are retired** —
programs are authored in Lean (arrow builders) and registered at boot (or
crystallized from a session with `export_program`); a file carrying a
`programDecl` is refused at ingest.
**Audio output is an `outputAssign` in the body with name `"dac.out"`** — wire
the signal you want heard to it. Add `"channel": 1` (or another non-negative
index) for an independent output channel; an absent channel is the legacy mono
channel 0. Multiple assignments to one channel are mixed in authored order.
Root `params`, `audio_outputs`, and
`breaks_cycles` carriers are retired and refused.

{
  "schema": "tropical_program_2",
  "name": "MyPatch",
  "body": {
    "op": "block",
    "decls": [
      {
        "op": "instanceDecl",
        "name": "osc",
        "program": "FixedSinOsc",
        "inputs": {
          "freq": 440
        }
      },
      {
        "op": "instanceDecl",
        "name": "shaper",
        "program": "SoftClip",
        "inputs": {
          "input": {
            "op": "ref",
            "instance": "osc",
            "output": "sine"
          },
          "drive": 4
        }
      }
    ],
    "assigns": [
      {
        "op": "outputAssign",
        "name": "dac.out",
        "expr": {
          "op": "ref",
          "instance": "shaper",
          "output": "out"
        }
      }
    ],
    "value": null
  }
}

Key fields: schema, name, ports (inputs/outputs), body (block), and sample_rate.
Declare parameters with body `paramDecl` nodes. Send a signal to the speakers
with a body `outputAssign` named `"dac.out"`.

Decl node shapes:
- paramDecl:    { op, name, value? }
- instanceDecl: { op, name, program, inputs? }

Assign node shapes:
- outputAssign: { op, name, expr }

## Expression node format (ExprNode)

Used in instance input wiring.

- **Literal**: `3.14` or `true`
- **Reference**: `{ "op": "ref", "instance": "osc", "output": "out" }` — routes another instance's output to this input
- **Param**: `{ "op": "param", "name": "cutoff" }`
- **Binary**: `{ "op": "mul", "args": [<expr>, <expr>] }` — add, sub, mul, div, floorDiv, mod, ldexp; lt, lte, gt, gte, eq, neq; and, or; bitAnd, bitOr, bitXor, lshift, rshift
- **Unary**: `{ "op": "neg", "args": [<expr>] }` — neg, abs, not, bitNot, sqrt, floor, ceil, round, floatExponent, toInt, toBool, toFloat
- **Clamp / Select / bare arrays / Index / Builtins**: see the program catalog and stdlib for worked examples.
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
- Audio output goes through a body `outputAssign` named `"dac.out"` (or the
  `wire` tool with `instance: "dac", input: "out"`). Add `channel` for
  independent stereo/multichannel routing; omission keeps mono channel 0.
"#

/-- Port of `renderProgramCatalog` over the engine catalog (concrete
    entries only — the TS source iterated `session.typeRegistry`). -/
def renderProgramCatalog (st : Tropical.SessionSt) : String := Id.run do
  let mut lines : Array String := #["# tropical program catalog\n"]
  for name in st.catalogOrder do
    match st.programs.get? name with
    | some pm =>
      let inputParts := pm.inputs.map fun p =>
        match p.default with
        | some v => s!"{p.name}={v.toJson.compress}"
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
