/**
 * resources.ts — the MCP non-tool surface (resources + prompts), shared
 * between engine.ts and compiler_service.ts.
 */

import type { SessionState } from '../compiler/session.js'
import { inputNames, outputNames, rawInputDefaults } from '../compiler/session.js'
import { PROGRAM_FORMAT_EXAMPLE } from './program_format_example.js'

export const RESOURCES = [
  {
    uri:         'tropical://programs',
    name:        'Program catalog',
    description: 'Markdown catalog of all registered program types with inputs, outputs, and default values.',
    mimeType:    'text/markdown',
  },
  {
    uri:         'tropical://program-format',
    name:        'Program format',
    description: 'Reference doc for the tropical_program_2 schema.',
    mimeType:    'text/markdown',
  },
]

export function renderProgramCatalog(session: SessionState): string {
  const lines: string[] = ['# tropical program catalog\n']
  for (const [typeName, type] of session.typeRegistry) {
    lines.push(`## ${typeName}`)
    const defaultsMap = rawInputDefaults(type) as Record<string, unknown>
    const inputParts = inputNames(type).map(n => {
      const val = defaultsMap[n]
      return val !== undefined ? `${n}=${JSON.stringify(val)}` : n
    })
    lines.push(`Inputs:  ${inputParts.join(', ')}`)
    lines.push(`Outputs: ${outputNames(type).join(', ')}`)
    lines.push('')
  }
  return lines.join('\n')
}

export const PROGRAM_FORMAT_DOC = `# tropical program format

Programs are the unified representation for all DSP in tropical. A program is an
expression with declared ports and a body that declares regs/delays/instances
and assigns outputs and next-tick register updates.

## tropical_program_2

The body is a \`block\` of \`decls\` (reg_decl, delay_decl, instance_decl,
program_decl) and \`assigns\` (output_assign, next_update). Ports, type_params,
and breaks_cycles sit alongside the body. **Audio output is an \`output_assign\`
in the body with name \`"dac.out"\`** — wire the signal you want heard to it.
Session metadata — \`params\`, \`config\` — is top-level.

${JSON.stringify(PROGRAM_FORMAT_EXAMPLE, null, 2)}

Key fields: schema, name, ports (inputs/outputs/type_defs), body (block),
type_params, sample_rate, breaks_cycles, and top-level session metadata
(params, config). Send a signal to the speakers with a body \`output_assign\`
named \`"dac.out"\`. (File-root \`audio_outputs\` is deprecated — don't use it.)

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

- **Literal**: \`3.14\` or \`true\`
- **Reference**: \`{ "op": "ref", "instance": "osc", "output": "out" }\` — routes another instance's output to this input
- **Input port**: \`{ "op": "input", "name": "freq" }\` — (inside program definitions only)
- **Param**: \`{ "op": "param", "name": "cutoff" }\`
- **Binary**: \`{ "op": "mul", "args": [<expr>, <expr>] }\` — add, sub, mul, div, floor_div, mod, pow; lt, lte, gt, gte, eq, neq; bit_and, bit_or, bit_xor, lshift, rshift
- **Unary**: \`{ "op": "neg", "args": [<expr>] }\` — neg, abs, not, bit_not
- **Clamp / Select / Array / Index / Delay / Builtins**: see the program catalog and stdlib for worked examples.
`

export const PROMPTS = [
  {
    name:        'build-patch',
    description: 'Three-tiered workflow guidance for building and editing tropical patches efficiently.',
  },
]

export const BUILD_PATCH_PROMPT = `# build-patch workflow

Before writing any program, always fetch both resources:
- \`tropical://programs\` — full catalog of available program types with inputs, defaults, and outputs
- \`tropical://program-format\` — tropical_program_2 schema reference with a worked example

## Choose the right tool for the job

### New program (starting from scratch)
Use \`load\` with a **complete** tropical_program_2 JSON object in a single call.
Do not call \`add_instance\` and \`wire\` one-by-one — that requires many round
trips and recompiles the JIT kernel on every change.

### Extending an existing program (adding new instances)
Use \`merge\` with a partial program containing only the new instances and wiring.
Then use \`wire\` to connect the new instances to existing ones.

### Targeted edits (changing a value, tweaking a param)
Use \`wire\` with a single set entry, or \`set_param\` for control parameters.

## Writing programs

- Use named ports (e.g. \`"output": "out"\`) rather than integer indices.
- Wire everything in one program object where possible; minimize round trips.
- Audio output goes through a body \`output_assign\` named \`"dac.out"\` (or the
  \`wire\` tool with \`instance: "dac", input: "out"\`).
`

/** Read a resource's text content by uri. */
export function readResourceText(uri: string, session: SessionState): string {
  if (uri === 'tropical://programs' || uri === 'tropical://modules') return renderProgramCatalog(session)
  if (uri === 'tropical://program-format' || uri === 'tropical://patch-format') return PROGRAM_FORMAT_DOC
  throw new Error(`Unknown resource: ${uri}`)
}

/** Get a prompt's messages by name. */
export function getPromptMessages(name: string): { messages: unknown[] } {
  if (name === 'build-patch') {
    return { messages: [{ role: 'user', content: { type: 'text', text: BUILD_PATCH_PROMPT } }] }
  }
  throw new Error(`Unknown prompt: ${name}`)
}
