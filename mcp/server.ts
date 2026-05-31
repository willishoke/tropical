/**
 * TypeScript MCP server — replaces tropical/mcp_server.py.
 *
 * Run with:   bun run src/server.ts
 * Spawned by: tui/src/mcp.ts via StdioClientTransport
 */

import { Server }              from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport }from '@modelcontextprotocol/sdk/server/stdio.js'
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js'
import { inputNames, outputNames, rawInputDefaults } from '../compiler/session.js'
import { session, handleTool } from './engine.js'
import { PROGRAM_FORMAT_EXAMPLE } from './program_format_example.js'
// ─── Tool definitions ─────────────────────────────────────────────────────────

const TOOLS = [
  // ── Unified program tools ──────────────────────────────────────────────────

  {
    name: 'define_program',
    description: 'Define a reusable DSP program type and register it. Accepts a tropical_program_2 object. Returns the type name and port names.',
    inputSchema: {
      type: 'object',
      properties: {
        def: { type: 'object', description: 'tropical_program_2 object defining the program' },
      },
      required: ['def'],
    },
  },
  {
    name: 'add_instance',
    description: 'Create a named instance of a registered program type. For generic programs (those declaring type_params — see list_programs), supply type_args with concrete integer values (e.g. {"N": 44100}).',
    inputSchema: {
      type: 'object',
      properties: {
        program:       { type: 'string', description: 'Registered program/type name (builtin or user-defined)' },
        instance_name: { type: 'string', description: 'Unique name for this instance' },
        type_args:     { type: 'object', description: 'Compile-time type args for generic programs, e.g. {"N": 44100}. Omit for non-generic programs.' },
      },
      required: ['program', 'instance_name'],
    },
  },
  {
    name: 'remove_instance',
    description: 'Remove a program instance from the session.',
    inputSchema: {
      type: 'object',
      properties: { instance_name: { type: 'string' } },
      required: ['instance_name'],
    },
  },
  {
    name: 'replicate',
    description: 'Create N instances of a program type in one call. Returns the list of created instance names and their ports. Does NOT trigger recompilation — follow up with wire (use `instance: "dac", input: "out"` to wire to audio output).',
    inputSchema: {
      type: 'object',
      properties: {
        program:     { type: 'string', description: 'Registered program type name' },
        count:       { type: 'number', description: 'Number of instances to create' },
        name_prefix: { type: 'string', description: 'Name prefix for instances (default: lowercase program name). Instances are named prefix1, prefix2, …' },
        type_args:   { type: 'object', description: 'Compile-time type args applied to every created instance, e.g. {"N": 8}. Omit for non-generic programs.' },
      },
      required: ['program', 'count'],
    },
  },
  {
    name: 'wire_chain',
    description: 'Wire N instances in series: instances[i].output → instances[i+1].input. Optionally set the first instance\'s input from an expression. One recompile.',
    inputSchema: {
      type: 'object',
      properties: {
        instances:    { type: 'array', items: { type: 'string' }, description: 'Ordered list of instance names to chain' },
        output:       { description: 'Output port name or index to read from each instance' },
        input:        { description: 'Input port name or index to feed on each instance' },
        initial_expr: { description: 'Optional ExprNode to wire into the first instance\'s input' },
      },
      required: ['instances', 'output', 'input'],
    },
  },
  {
    name: 'wire_zip',
    description: 'Wire two equal-length lists of ports pairwise: sources[i].output → targets[i].input. One recompile.',
    inputSchema: {
      type: 'object',
      properties: {
        sources: {
          type: 'array',
          items: {
            type: 'object',
            properties: { instance: { type: 'string' }, output: { description: 'Output port name or index' } },
            required: ['instance', 'output'],
          },
        },
        targets: {
          type: 'array',
          items: {
            type: 'object',
            properties: { instance: { type: 'string' }, input: { description: 'Input port name or index' } },
            required: ['instance', 'input'],
          },
        },
      },
      required: ['sources', 'targets'],
    },
  },
  {
    name: 'fan_out',
    description: 'Wire one source to many target inputs. Source can be an instance output ({instance, output}) or any ExprNode (literal, param, arithmetic expression, etc.). One recompile.',
    inputSchema: {
      type: 'object',
      properties: {
        source: {
          description: 'Either {instance, output} for an instance output, or any ExprNode (number, {op:"param",...}, {op:"mul",...}, etc.)',
        },
        targets: {
          type: 'array',
          items: {
            type: 'object',
            properties: { instance: { type: 'string' }, input: { description: 'Input port name or index' } },
            required: ['instance', 'input'],
          },
        },
      },
      required: ['source', 'targets'],
    },
  },
  {
    name: 'fan_in',
    description: 'Sum N instance outputs and wire the result to one input. Optional per-source gain. One recompile.',
    inputSchema: {
      type: 'object',
      properties: {
        sources: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              instance: { type: 'string' },
              output:   { description: 'Output port name or index' },
              gain:     { type: 'number', description: 'Optional scale factor for this source' },
            },
            required: ['instance', 'output'],
          },
        },
        target: {
          type: 'object',
          properties: {
            instance: { type: 'string' },
            input:    { description: 'Input port name or index' },
          },
          required: ['instance', 'input'],
        },
      },
      required: ['sources', 'target'],
    },
  },
  {
    name: 'feedback',
    description: 'Wire an instance output back to an input through a 1-sample delay, creating a feedback loop. The delay is inserted inline — no extra instance is created. One recompile.',
    inputSchema: {
      type: 'object',
      properties: {
        from: {
          type: 'object',
          properties: { instance: { type: 'string' }, output: { description: 'Output port name or index' } },
          required: ['instance', 'output'],
        },
        to: {
          type: 'object',
          properties: { instance: { type: 'string' }, input: { description: 'Input port name or index' } },
          required: ['instance', 'input'],
        },
        init:     { type: 'number', description: 'Initial delay register value (default 0)' },
        delay_id: { type: 'string', description: 'Stable name for this delay register — preserves its state across hot-swaps' },
      },
      required: ['from', 'to'],
    },
  },
  {
    name: 'export_program',
    description: 'Crystallize the current session (or a subset) into a reusable program type. ' +
      'Specify which instance ports become the new program\'s inputs and outputs. ' +
      'Current wiring becomes input_defaults. The new type is registered and can be instantiated immediately. ' +
      'Optionally remove the exported instances from the session afterward.',
    inputSchema: {
      type: 'object',
      properties: {
        name:   { type: 'string', description: 'Name for the new program type (PascalCase recommended)' },
        inputs: {
          type: 'object',
          description: 'Map from new input name → "instance:port". The current wiring for each port becomes the default.',
          additionalProperties: { type: 'string' },
        },
        outputs: {
          type: 'object',
          description: 'Map from new output name → {instance, output}.',
          additionalProperties: {
            type: 'object',
            properties: {
              instance: { type: 'string' },
              output:   { type: 'string' },
            },
            required: ['instance', 'output'],
          },
        },
        remove_exported: {
          type: 'boolean',
          description: 'If true, remove the exported instances from the session after registration (default false).',
        },
      },
      required: ['name', 'inputs', 'outputs'],
    },
  },
  {
    name: 'list_programs',
    description: 'List all registered program types (builtins + user-defined) with their input/output port names and input defaults. Call this before building a program to discover what is available.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'list_instances',
    description: 'List all live program instances in the current session with their port names.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'get_info',
    description: 'Return detailed info about a program instance including ports, wiring, and registers.',
    inputSchema: {
      type: 'object',
      properties: { instance_name: { type: 'string' } },
      required: ['instance_name'],
    },
  },
  {
    name: 'wire',
    description: 'Set and/or remove input wiring in a single recompile. Use `set` to wire inputs (each is {instance, input, expr, [combine]}), `remove` to disconnect (each is {instance, input}). Audio output: use `instance: "dac", input: "out"` to wire to the DAC boundary leaf — `expr` must be a ref-shaped node (e.g. {op:"ref",instance,output}). Multiple wires to dac.out sum into the mono output bus. Removing a dac wire (`remove: [{instance:"dac",input:"out"}]`) clears all dac wires at once. The optional per-set `combine` field declares how a second wire to the same input combines with the first (slot model, M3+): named built-in like "add", "or", "max", or a program name. Currently stored but not yet enforced — wire(set:[...]) still replaces a prior expression.',
    inputSchema: {
      type: 'object',
      properties: {
        set: {
          type: 'array', description: 'Inputs to set',
          items: {
            type: 'object',
            properties: {
              instance: { type: 'string' },
              input:    { description: 'Input port name or index' },
              expr:     { description: 'ExprNode: number, bool, array, or {op, ...} object' },
              combine:  { type: 'string', description: 'Optional combine op for fan-in (e.g. "add", "or", "max", or a program name). Required when wiring a second source to the same input. Currently stored but not enforced; M4 wires it into the slot-mode compile path.' },
            },
            required: ['instance', 'input', 'expr'],
          },
        },
        remove: {
          type: 'array', description: 'Inputs to disconnect',
          items: {
            type: 'object',
            properties: {
              instance: { type: 'string' },
              input:    { description: 'Input port name or index' },
            },
            required: ['instance', 'input'],
          },
        },
      },
    },
  },
  {
    name: 'list_wiring',
    description: 'List all wired inputs. Shows the expression assigned to each input.',
    inputSchema: {
      type: 'object',
      properties: {
        instance: { type: 'string', description: 'If provided, filter to inputs of this instance.' },
      },
    },
  },
  {
    name: 'load',
    description: 'Load a tropical_program_2 program. Stops audio, recreates the session, and rebuilds state. Provide either path (preferred) or inline JSON.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Path to a .json file on disk.' },
        program: { type: 'object', description: 'Inline tropical_program_2 object.' },
      },
    },
  },
  {
    name: 'save',
    description: 'Serialize the current session to a tropical_program_2 JSON object.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'merge',
    description: 'Merge a program or patch into the current session without clearing it. Fails fast on name collisions.',
    inputSchema: {
      type: 'object',
      properties: {
        program: { type: 'object', description: 'tropical_program_2 object to merge.' },
      },
      required: ['program'],
    },
  },

  // ── Control parameters ─────────────────────────────────────────────────────

  {
    name: 'set_param',
    description: 'Set the value of a named Param (thread-safe, smoothed).',
    inputSchema: {
      type: 'object',
      properties: {
        name:  { type: 'string' },
        value: { type: 'number' },
      },
      required: ['name', 'value'],
    },
  },
  {
    name: 'list_params',
    description: 'List all registered Params with their current values.',
    inputSchema: { type: 'object', properties: {} },
  },

  // ── Audio control ──────────────────────────────────────────────────────────

  {
    name: 'start_audio',
    description: 'Start audio output. Optional: device by name substring; sample_rate (default 44100); channels (default 2). Sample rate and channel count apply only the first time the DAC is created and are fixed thereafter; subsequent calls ignore those overrides.',
    inputSchema: {
      type: 'object',
      properties: {
        device_name: { type: 'string', description: 'Optional partial device name match' },
        sample_rate: { type: 'integer', minimum: 1, description: 'DAC sample rate in Hz (default 44100). Used only on first DAC creation.' },
        channels:    { type: 'integer', minimum: 1, description: 'DAC output channel count (default 2). Used only on first DAC creation.' },
      },
    },
  },
  {
    name: 'stop_audio',
    description: 'Stop audio output.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'audio_status',
    description: 'Return current audio status including callback statistics.',
    inputSchema: { type: 'object', properties: {} },
  },
]
// ─── Resources ────────────────────────────────────────────────────────────────

const RESOURCES = [
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

function renderProgramCatalog(): string {
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

const PROGRAM_FORMAT_DOC = `# tropical program format

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

## Combinators (compile-time expansion)

- generate: { "op": "generate", "count": 4, "var": "i", "body": <expr> }
- iterate: { "op": "iterate", "count": 4, "init": <expr>, "var": "x", "body": <expr> }
- fold: { "op": "fold", "over": <array>, "init": <expr>, "acc_var": "acc", "elem_var": "x", "body": <expr> }
- scan: like fold but keeps intermediates as array
- map2: { "op": "map2", "over": <array>, "elem_var": "x", "body": <expr> }
- zip_with: { "op": "zipWith", "a": <arr>, "b": <arr>, "x_var": "x", "y_var": "y", "body": <expr> }
- chain: { "op": "chain", "count": 3, "init": <expr>, "var": "x", "body": <expr> }
- let: { "op": "let", "bind": { "x": <expr> }, "in": <expr> }
- binding: { "op": "binding", "name": "x" } — variable reference in combinator bodies

## Expression node format (ExprNode)

Used in instance input wiring and inline program process definitions.

- **Literal**: \`3.14\` or \`true\`
- **Reference**: \`{ "op": "ref", "instance": "osc", "output": "sin" }\` — routes another instance's output to this input
- **Input port**: \`{ "op": "input", "name": "freq" }\` — (inside program definitions only)
- **Param**: \`{ "op": "param", "name": "cutoff" }\`
- **Binary**: \`{ "op": "mul", "args": [<expr>, <expr>] }\`
  - Arithmetic: \`add\`, \`sub\`, \`mul\`, \`div\`, \`floor_div\`, \`mod\`, \`pow\`, \`matmul\`
  - Compare: \`lt\`, \`lte\`, \`gt\`, \`gte\`, \`eq\`, \`neq\`
  - Bitwise: \`bit_and\`, \`bit_or\`, \`bit_xor\`, \`lshift\`, \`rshift\`
- **Unary**: \`{ "op": "neg", "args": [<expr>] }\`
  - ops: \`neg\`, \`abs\`, \`sin\`, \`log\`, \`not\`, \`bit_not\`
- **Clamp**: \`{ "op": "clamp", "args": [<value>, <min>, <max>] }\`
- **Select**: \`{ "op": "select", "args": [<cond>, <if_true>, <if_false>] }\`
- **Array**: \`{ "op": "array", "items": [<expr>, ...] }\`
- **Index**: \`{ "op": "index", "args": [<array_expr>, <index_expr>] }\`
- **Delay**: \`{ "op": "delay", "args": [<expr>], "init": 0.0, "id": "optional_name" }\`
- **Builtins**: \`{ "op": "sampleRate" }\`, \`{ "op": "sampleIndex" }\`
`

// ─── Prompts ──────────────────────────────────────────────────────────────────

const PROMPTS = [
  {
    name:        'build-patch',
    description: 'Three-tiered workflow guidance for building and editing tropical patches efficiently.',
  },
]

const BUILD_PATCH_PROMPT = `# build-patch workflow

Before writing any program, always fetch both resources:
- \`tropical://programs\` — full catalog of available program types with inputs, defaults, and outputs
- \`tropical://program-format\` — tropical_program_2 schema reference with worked examples

## Choose the right tool for the job

### New program (starting from scratch)
Use \`load\` with a **complete** tropical_program_2 JSON object in a single call.
Do not call \`add_instance\` and \`wire\` one-by-one —
that requires many round trips and recompiles the JIT kernel on every change.

### Extending an existing program (adding new instances)
Use \`merge\` with a partial program containing only the new instances and wiring.
Then use \`wire\` to connect the new instances to existing ones.
Do not use \`load\` — it tears down the entire session and loses the existing state.

### Targeted edits (changing a value, tweaking a param)
Use \`wire\` with a single set entry, or \`set_param\` for control parameters.
Do not reload or rebuild the program for a single value change.

## Writing programs

- Use inline \`programs\` for simple arithmetic types (gains, mixers, etc.)
  rather than calling \`define_program\` separately. This keeps the program self-contained.
- Use named ports (e.g. \`"output": "sin"\`) rather than integer indices.
- Wire everything in one program object where possible; minimize round trips.
- Use combinators (generate, fold, chain, map2, etc.) instead of manually unrolling
  repetitive structures. See the program-format resource for the full list.
`
// ─── Server wiring ────────────────────────────────────────────────────────────

const server = new Server(
  { name: 'tropical', version: '0.3.0' },
  { capabilities: { tools: {}, resources: {}, prompts: {} } },
)

server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: RESOURCES }))

server.setRequestHandler(ReadResourceRequestSchema, async (req) => {
  const { uri } = req.params
  if (uri === 'tropical://programs' || uri === 'tropical://modules') {
    return { contents: [{ uri, mimeType: 'text/markdown', text: renderProgramCatalog() }] }
  }
  if (uri === 'tropical://program-format' || uri === 'tropical://patch-format') {
    return { contents: [{ uri, mimeType: 'text/markdown', text: PROGRAM_FORMAT_DOC }] }
  }
  throw new Error(`Unknown resource: ${uri}`)
})

server.setRequestHandler(ListPromptsRequestSchema, async () => ({ prompts: PROMPTS }))

server.setRequestHandler(GetPromptRequestSchema, async (req) => {
  if (req.params.name === 'build-patch') {
    return { messages: [{ role: 'user', content: { type: 'text', text: BUILD_PATCH_PROMPT } }] }
  }
  throw new Error(`Unknown prompt: ${req.params.name}`)
})

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }))

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params
  return handleTool(name, args as Record<string, unknown>)
})

const transport = new StdioServerTransport()
await server.connect(transport)
