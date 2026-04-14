/**
 * TypeScript MCP server — replaces tropical/mcp_server.py.
 *
 * Run with:   bun run src/server.ts
 * Spawned by: tui/src/mcp.ts via StdioClientTransport
 */

import { readFileSync }        from 'node:fs'
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

import {
  makeSession, nextName, loadJSON,
  prettyExpr, SessionState, ExprNode,
} from '../compiler/session.js'
import { parseProgram } from '../compiler/schema.js'
import {
  saveProgramFromSession, loadProgramAsType, mergeProgramIntoSession,
  loadStdlib as loadBuiltins, type ProgramJSON,
} from '../compiler/program.js'
import { DAC }                 from '../compiler/runtime/audio.js'
import { Param, Trigger }      from '../compiler/runtime/param.js'
import { applyFlatPlan }  from '../compiler/apply_plan.js'
import { checkArrayConnection } from '../compiler/array_wiring.js'
import { validateExpr }         from '../compiler/expr.js'
import { exprDependencies }     from '../compiler/compiler.js'

// ─── Session ──────────────────────────────────────────────────────────────────

const session: SessionState = makeSession()
loadBuiltins(session.typeRegistry)

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Type-check an input expression against a destination port type, inserting
 * a broadcast_to wrapper when the shapes are compatible but differ (e.g. scalar→array).
 * Throws if the connection is incompatible (e.g. array→scalar, shape mismatch).
 * Returns the (possibly wrapped) expression and the resultShape if broadcasting occurred.
 */
function adaptInputExpr(
  node: ExprNode,
  dstTypeStr: string | undefined,
  instanceName: string,
  inputName: string,
): { expr: ExprNode; resultShape?: number[] } {
  let srcTypeStr: string | undefined

  if (typeof node === 'number') {
    srcTypeStr = 'float'
  } else if (typeof node === 'boolean') {
    srcTypeStr = 'bool'
  } else if (Array.isArray(node)) {
    srcTypeStr = `float[${(node as unknown[]).length}]`
  } else if (typeof node === 'object' && node !== null) {
    const obj = node as Record<string, unknown>
    if (obj.op === 'ref') {
      const srcInst = session.instanceRegistry.get(obj.instance as string)
      if (srcInst) {
        const outName = obj.output as string | number
        const outIdx = typeof outName === 'number' ? outName : srcInst.outputNames.indexOf(String(outName))
        if (outIdx !== -1) srcTypeStr = srcInst.outputPortType(outIdx)
      }
    }
  }

  if (srcTypeStr === undefined) return { expr: node }

  const check = checkArrayConnection(srcTypeStr, dstTypeStr, node)
  if (!check.compatible) {
    throw new Error(
      `Type mismatch on '${instanceName}'.${inputName}: ${check.error}`
    )
  }
  return { expr: check.broadcastExpr ?? node, resultShape: check.resultShape }
}

const ok  = (data: unknown) =>
  ({ content: [{ type: 'text' as const, text: JSON.stringify({ ok: true,  data  }) }] })

const fail = (e: unknown) =>
  ({ content: [{ type: 'text' as const, text: JSON.stringify({ ok: false, error: String(e) }) }], isError: true as const })

function wrap(fn: () => unknown) {
  try   { return ok(fn())         }
  catch (e) { return fail(e) }
}

function resolveOutputIdx(inst: { outputNames: string[] }, nameOrIdx: string | number): number {
  if (typeof nameOrIdx === 'number') return nameOrIdx
  const idx = inst.outputNames.indexOf(nameOrIdx)
  if (idx === -1) throw new Error(`Unknown output '${nameOrIdx}'. Available: ${inst.outputNames.join(', ')}`)
  return idx
}

function resolveInputIdx(inst: { inputNames: string[] }, nameOrIdx: string | number): number {
  if (typeof nameOrIdx === 'number') return nameOrIdx
  const idx = inst.inputNames.indexOf(nameOrIdx)
  if (idx === -1) throw new Error(`Unknown input '${nameOrIdx}'. Available: ${inst.inputNames.join(', ')}`)
  return idx
}

function instanceSummary(name: string) {
  const inst = session.instanceRegistry.get(name)!
  return { name, type_name: inst.typeName, inputs: inst.inputNames, outputs: inst.outputNames }
}

/** Apply wiring via FlatRuntime. */
function wire(): {} {
  applyFlatPlan(session, session.runtime)
  return {}
}

// ─── Tool definitions ─────────────────────────────────────────────────────────

const TOOLS = [
  // ── Unified program tools ──────────────────────────────────────────────────

  {
    name: 'define_program',
    description: 'Define a reusable DSP program type and register it. Accepts a ProgramJSON (tropical_program_1). Returns the type name and port names.',
    inputSchema: {
      type: 'object',
      properties: {
        def: { type: 'object', description: 'ProgramJSON (tropical_program_1) object defining the program' },
      },
      required: ['def'],
    },
  },
  {
    name: 'add_instance',
    description: 'Create a named instance of a registered program type.',
    inputSchema: {
      type: 'object',
      properties: {
        program:       { type: 'string', description: 'Registered program/type name (builtin or user-defined)' },
        instance_name: { type: 'string', description: 'Unique name for this instance' },
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
    description: 'Create N instances of a program type in one call. Returns the list of created instance names and their ports. Does NOT trigger recompilation — follow up with wire and/or set_output.',
    inputSchema: {
      type: 'object',
      properties: {
        program:     { type: 'string', description: 'Registered program type name' },
        count:       { type: 'number', description: 'Number of instances to create' },
        name_prefix: { type: 'string', description: 'Name prefix for instances (default: lowercase program name). Instances are named prefix1, prefix2, …' },
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
    description: 'Set and/or remove input wiring in a single recompile. Use `set` to wire inputs (each is {instance, input, expr}), `remove` to disconnect (each is {instance, input}).',
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
    name: 'set_output',
    description: 'Set the audio output mix. Provide a complete list of outputs — replaces all current outputs.',
    inputSchema: {
      type: 'object',
      properties: {
        outputs: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              instance:    { type: 'string' },
              output: { description: 'Output port name or index' },
            },
            required: ['instance', 'output'],
          },
        },
      },
      required: ['outputs'],
    },
  },
  {
    name: 'load',
    description: 'Load a program. Accepts tropical_program_1. Stops audio, recreates the session, and rebuilds state. Provide either path (preferred) or inline JSON.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Path to a .json file on disk.' },
        program: { type: 'object', description: 'Inline ProgramJSON (tropical_program_1) object.' },
      },
    },
  },
  {
    name: 'save',
    description: 'Serialize the current session to a tropical_program_1 JSON object.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'merge',
    description: 'Merge a program or patch into the current session without clearing it. Fails fast on name collisions.',
    inputSchema: {
      type: 'object',
      properties: {
        program: { type: 'object', description: 'ProgramJSON (tropical_program_1) object to merge.' },
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
    description: 'List all registered Params and Triggers with their current values.',
    inputSchema: { type: 'object', properties: {} },
  },

  // ── Audio control ──────────────────────────────────────────────────────────

  {
    name: 'start_audio',
    description: 'Start audio output. Optionally specify a device by name substring.',
    inputSchema: {
      type: 'object',
      properties: {
        device_name: { type: 'string', description: 'Optional partial device name match' },
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

// ─── Tool handlers ────────────────────────────────────────────────────────────

// ─── Shared handler logic ─────────────────────────────────────────────────────

function handleDefineProgram(args: Record<string, unknown>) {
  return wrap(() => {
    const prog = parseProgram(args.def) as ProgramJSON
    const type = loadProgramAsType(prog, session)
    session.typeRegistry.set(type.name, type)
    return { program_name: type.name, inputs: type._def.inputNames, outputs: type._def.outputNames }
  })
}

function handleAddInstance(programName: string, instanceName: string) {
  return wrap(() => {
    if (session.instanceRegistry.has(instanceName))
      throw new Error(`Instance '${instanceName}' already exists.`)
    const type = session.typeRegistry.get(programName)
    if (!type)
      throw new Error(`Unknown program '${programName}'. Known: ${[...session.typeRegistry.keys()].join(', ')}`)
    const inst = type.instantiateAs(instanceName)
    session.instanceRegistry.set(instanceName, inst)
    return instanceSummary(instanceName)
  })
}

function handleReplicate(programName: string, count: number, namePrefix?: string) {
  return wrap(() => {
    const type = session.typeRegistry.get(programName)
    if (!type)
      throw new Error(`Unknown program '${programName}'. Known: ${[...session.typeRegistry.keys()].join(', ')}`)
    if (!Number.isInteger(count) || count < 1)
      throw new Error(`count must be a positive integer, got ${count}`)

    const prefix = namePrefix ?? programName.toLowerCase()
    const created = []
    for (let i = 0; i < count; i++) {
      const name = nextName(session, prefix)
      if (session.instanceRegistry.has(name))
        throw new Error(`Instance '${name}' already exists — pick a different name_prefix`)
      const inst = type.instantiateAs(name)
      session.instanceRegistry.set(name, inst)
      created.push(instanceSummary(name))
    }
    return { created }
  })
}

/** Resolve an output name/index on an instance and return the canonical name string. */
function resolveOutputName(inst: { outputNames: string[] }, nameOrIdx: string | number): string {
  const idx = resolveOutputIdx(inst, nameOrIdx)
  return inst.outputNames[idx]
}

/** Resolve an input name/index on an instance and return the canonical name string. */
function resolveInputName(inst: { inputNames: string[] }, nameOrIdx: string | number): string {
  const idx = resolveInputIdx(inst, nameOrIdx)
  return inst.inputNames[idx]
}

function handleWireChain(args: Record<string, unknown>) {
  return wrap(() => {
    const instanceNames = args.instances as string[]
    const outputPort    = args.output as string | number
    const inputPort     = args.input  as string | number
    const initialExpr   = args.initial_expr as import('../compiler/expr.js').ExprNode | undefined

    if (instanceNames.length < 2 && initialExpr === undefined)
      throw new Error('wire_chain needs at least 2 instances, or 1 instance with initial_expr')

    // Validate all instances upfront
    const insts = instanceNames.map(n => {
      const inst = session.instanceRegistry.get(n)
      if (!inst) throw new Error(`No instance named '${n}'`)
      return inst
    })

    // Optionally set first instance's input
    if (initialExpr !== undefined) {
      const firstName  = instanceNames[0]
      const firstInst  = insts[0]
      const inputName  = resolveInputName(firstInst, inputPort)
      const { expr }   = adaptInputExpr(initialExpr, firstInst.inputPortType(firstInst.inputNames.indexOf(inputName)), firstName, inputName)
      session.inputExprNodes.set(`${firstName}:${inputName}`, expr)
    }

    // Wire instances[i].output → instances[i+1].input
    const linked = []
    for (let i = 0; i < instanceNames.length - 1; i++) {
      const srcInst  = insts[i]
      const dstInst  = insts[i + 1]
      const srcName  = instanceNames[i]
      const dstName  = instanceNames[i + 1]
      const outName  = resolveOutputName(srcInst, outputPort)
      const inName   = resolveInputName(dstInst, inputPort)
      const refExpr  = { op: 'ref' as const, instance: srcName, output: outName }
      const { expr } = adaptInputExpr(refExpr, dstInst.inputPortType(dstInst.inputNames.indexOf(inName)), dstName, inName)
      session.inputExprNodes.set(`${dstName}:${inName}`, expr)
      linked.push(`${srcName}.${outName} → ${dstName}.${inName}`)
    }

    return { linked, ...wire() }
  })
}

function handleWireZip(args: Record<string, unknown>) {
  return wrap(() => {
    const sources = args.sources as Array<{ instance: string; output: string | number }>
    const targets = args.targets as Array<{ instance: string; input:  string | number }>

    if (sources.length !== targets.length)
      throw new Error(`sources and targets must be the same length (got ${sources.length} vs ${targets.length})`)

    const linked = []
    for (let i = 0; i < sources.length; i++) {
      const src     = sources[i]
      const dst     = targets[i]
      const srcInst = session.instanceRegistry.get(src.instance)
      if (!srcInst) throw new Error(`No instance named '${src.instance}'`)
      const dstInst = session.instanceRegistry.get(dst.instance)
      if (!dstInst) throw new Error(`No instance named '${dst.instance}'`)

      const outName  = resolveOutputName(srcInst, src.output)
      const inName   = resolveInputName(dstInst, dst.input)
      const refExpr  = { op: 'ref' as const, instance: src.instance, output: outName }
      const { expr } = adaptInputExpr(refExpr, dstInst.inputPortType(dstInst.inputNames.indexOf(inName)), dst.instance, inName)
      session.inputExprNodes.set(`${dst.instance}:${inName}`, expr)
      linked.push(`${src.instance}.${outName} → ${dst.instance}.${inName}`)
    }

    return { linked, ...wire() }
  })
}

function handleFanOut(args: Record<string, unknown>) {
  return wrap(() => {
    const rawSource = args.source as { instance?: string; output?: string | number } | ExprNode
    const targets   = args.targets as Array<{ instance: string; input: string | number }>

    // Resolve source to a canonical ExprNode plus a display label
    let sourceExpr: ExprNode
    let sourceLabel: string
    if (
      rawSource !== null &&
      typeof rawSource === 'object' &&
      !Array.isArray(rawSource) &&
      typeof (rawSource as Record<string, unknown>).instance === 'string' &&
      (rawSource as Record<string, unknown>).output !== undefined
    ) {
      const s       = rawSource as { instance: string; output: string | number }
      const srcInst = session.instanceRegistry.get(s.instance)
      if (!srcInst) throw new Error(`No instance named '${s.instance}'`)
      const outName = resolveOutputName(srcInst, s.output)
      sourceExpr  = { op: 'ref' as const, instance: s.instance, output: outName }
      sourceLabel = `${s.instance}.${outName}`
    } else {
      sourceExpr  = rawSource as ExprNode
      sourceLabel = JSON.stringify(sourceExpr)
    }

    const linked = []
    for (const dst of targets) {
      const dstInst = session.instanceRegistry.get(dst.instance)
      if (!dstInst) throw new Error(`No instance named '${dst.instance}'`)
      const inName   = resolveInputName(dstInst, dst.input)
      const { expr } = adaptInputExpr(sourceExpr, dstInst.inputPortType(dstInst.inputNames.indexOf(inName)), dst.instance, inName)
      session.inputExprNodes.set(`${dst.instance}:${inName}`, expr)
      linked.push(`${sourceLabel} → ${dst.instance}.${inName}`)
    }

    return { linked, ...wire() }
  })
}

function handleFanIn(args: Record<string, unknown>) {
  return wrap(() => {
    const sources = args.sources as Array<{ instance: string; output: string | number; gain?: number }>
    const target  = args.target  as { instance: string; input: string | number }

    if (sources.length === 0) throw new Error('sources must be non-empty')

    const dstInst = session.instanceRegistry.get(target.instance)
    if (!dstInst) throw new Error(`No instance named '${target.instance}'`)

    // Build one term per source: ref, optionally scaled
    const terms: ExprNode[] = sources.map(src => {
      const srcInst = session.instanceRegistry.get(src.instance)
      if (!srcInst) throw new Error(`No instance named '${src.instance}'`)
      const outName = resolveOutputName(srcInst, src.output)
      const ref: ExprNode = { op: 'ref' as const, instance: src.instance, output: outName }
      return src.gain !== undefined
        ? { op: 'mul' as const, args: [ref, src.gain] }
        : ref
    })

    // Left-fold into a binary add tree
    const sumExpr = terms.slice(1).reduce<ExprNode>(
      (acc, t) => ({ op: 'add' as const, args: [acc, t] }),
      terms[0],
    )

    const inName   = resolveInputName(dstInst, target.input)
    const { expr } = adaptInputExpr(sumExpr, dstInst.inputPortType(dstInst.inputNames.indexOf(inName)), target.instance, inName)
    session.inputExprNodes.set(`${target.instance}:${inName}`, expr)

    return { mixed: sources.length, target: `${target.instance}.${inName}`, ...wire() }
  })
}

function handleFeedback(args: Record<string, unknown>) {
  return wrap(() => {
    const from    = args.from     as { instance: string; output: string | number }
    const to      = args.to       as { instance: string; input:  string | number }
    const init    = (args.init    as number | undefined) ?? 0
    const delayId = args.delay_id as string | undefined

    const srcInst = session.instanceRegistry.get(from.instance)
    if (!srcInst) throw new Error(`No instance named '${from.instance}'`)
    const dstInst = session.instanceRegistry.get(to.instance)
    if (!dstInst) throw new Error(`No instance named '${to.instance}'`)

    const outName = resolveOutputName(srcInst, from.output)
    const inName  = resolveInputName(dstInst, to.input)

    const refExpr: ExprNode = { op: 'ref' as const, instance: from.instance, output: outName }
    const delayExpr: ExprNode = delayId !== undefined
      ? { op: 'delay' as const, args: [refExpr], init, id: delayId }
      : { op: 'delay' as const, args: [refExpr], init }

    validateExpr(delayExpr, `${to.instance}.${inName}`)
    const { expr } = adaptInputExpr(delayExpr, dstInst.inputPortType(dstInst.inputNames.indexOf(inName)), to.instance, inName)
    session.inputExprNodes.set(`${to.instance}:${inName}`, expr)

    return {
      feedback: `${from.instance}.${outName} →[delay init=${init}]→ ${to.instance}.${inName}`,
      ...wire(),
    }
  })
}

function handleRemoveInstance(instanceName: string) {
  return wrap(() => {
    if (!session.instanceRegistry.has(instanceName))
      throw new Error(`No instance named '${instanceName}'.`)
    session.instanceRegistry.delete(instanceName)
    for (const key of [...session.inputExprNodes.keys()]) {
      if (key.startsWith(`${instanceName}:`)) session.inputExprNodes.delete(key)
    }
    for (const [key, expr] of [...session.inputExprNodes.entries()]) {
      if (exprDependencies(expr).has(instanceName)) session.inputExprNodes.delete(key)
    }
    session.graphOutputs = session.graphOutputs.filter(o => o.instance !== instanceName)
    return { removed: instanceName, ...wire() }
  })
}

function handleListPrograms() {
  return wrap(() =>
    [...session.typeRegistry.entries()].map(([typeName, type]) => {
      const d = type._def
      const defaultsMap = (d.rawInputDefaults ?? {}) as Record<string, unknown>
      return {
        program_name: typeName,
        inputs:    d.inputNames.map((n, i) => ({
          name: n,
          type: d.inputPortTypes[i] ?? null,
          bounds: d.inputBounds[i] ?? null,
          default: defaultsMap[n] ?? null,
        })),
        outputs: d.outputNames.map((n, i) => ({
          name: n,
          type: d.outputPortTypes[i] ?? null,
          bounds: d.outputBounds[i] ?? null,
        })),
        registers: d.registerNames.map((n, i) => ({ name: n, type: d.registerPortTypes[i] ?? null })),
      }
    }),
  )
}

function handleListInstances() {
  return wrap(() =>
    [...session.instanceRegistry.keys()].map(instanceSummary),
  )
}

function handleGetInfo(instanceName: string) {
  return wrap(() => {
    const inst = session.instanceRegistry.get(instanceName)
    if (!inst) throw new Error(`No instance named '${instanceName}'.`)
    return {
      name: instanceName,
      program: inst.typeName,
      inputs:  inst.inputNames.map((n, i) => ({
        name: n, index: i,
        type: inst._def.inputPortTypes[i] ?? null,
        bounds: inst._def.inputBounds[i] ?? null,
        expr: session.inputExprNodes.get(`${instanceName}:${n}`) ?? null,
        pretty: session.inputExprNodes.has(`${instanceName}:${n}`)
          ? prettyExpr(session.inputExprNodes.get(`${instanceName}:${n}`)!, session.instanceRegistry)
          : null,
      })),
      outputs: inst.outputNames.map((n, i) => ({
        name: n, index: i,
        type: inst._def.outputPortTypes[i] ?? null,
        bounds: inst._def.outputBounds[i] ?? null,
      })),
      registers: inst.registerNames.map((n, i) => ({
        name: n, index: i, type: inst.registerPortType(i) ?? null,
      })),
    }
  })
}

function handleWire(args: Record<string, unknown>) {
  return wrap(() => {
    const setOps = (args.set ?? []) as Array<{ instance: string; input: string | number; expr: ExprNode }>
    const removeOps = (args.remove ?? []) as Array<{ instance: string; input: string | number }>

    // Process removes first
    for (const r of removeOps) {
      const inst = session.instanceRegistry.get(r.instance)
      if (!inst) throw new Error(`No instance named '${r.instance}'.`)
      const inputId = typeof r.input === 'number' ? r.input
        : (String(r.input).match(/^\d+$/) ? parseInt(String(r.input), 10) : inst.inputIndex(String(r.input)))
      const resolvedName = inst.inputNames[inputId] ?? String(inputId)
      session.inputExprNodes.delete(`${r.instance}:${resolvedName}`)
    }

    // Process sets
    const results = []
    for (const s of setOps) {
      const inst = session.instanceRegistry.get(s.instance)
      if (!inst) throw new Error(`No instance named '${s.instance}'.`)
      const raw = s.input
      const inputId = typeof raw === 'number' ? raw
        : (String(raw).match(/^\d+$/) ? parseInt(String(raw), 10) : inst.inputIndex(String(raw)))
      const resolvedName = inst.inputNames[inputId] ?? String(inputId)
      validateExpr(s.expr, `${s.instance}.${resolvedName}`)
      const { expr } = adaptInputExpr(s.expr, inst.inputPortType(inputId), s.instance, resolvedName)
      session.inputExprNodes.set(`${s.instance}:${resolvedName}`, expr)
      results.push({ instance: s.instance, input: resolvedName, expr })
    }

    return { set: results, removed: removeOps.length, ...wire() }
  })
}

function handleListWiring(filterInstance?: string) {
  return wrap(() => {
    const results: Array<{ instance: string; input: string; expr: string }> = []
    for (const [key, node] of session.inputExprNodes) {
      const colonIdx = key.indexOf(':')
      const inst  = key.slice(0, colonIdx)
      const input = key.slice(colonIdx + 1)
      if (filterInstance && inst !== filterInstance) continue
      results.push({ instance: inst, input, expr: prettyExpr(node, session.instanceRegistry) })
    }
    return results
  })
}

function handleSetOutput(args: Record<string, unknown>) {
  return wrap(() => {
    const outputs = args.outputs as Array<{ instance: string; output: string | number }>
    session.graphOutputs.length = 0
    for (const o of outputs) {
      const inst = session.instanceRegistry.get(o.instance)
      if (!inst) throw new Error(`No instance named '${o.instance}'.`)
      const rawOut = o.output
      const outId = typeof rawOut === 'number' ? rawOut
        : (String(rawOut).match(/^\d+$/) ? parseInt(String(rawOut), 10) : inst.outputIndex(String(rawOut)))
      session.graphOutputs.push({ instance: o.instance, output: inst.outputNames[outId] })
    }
    return { outputs: session.graphOutputs, ...wire() }
  })
}

function handleLoad(args: Record<string, unknown>) {
  return wrap(() => {
    let raw: unknown
    if (args.path) {
      raw = JSON.parse(readFileSync(args.path as string, 'utf-8'))
    } else if (args.program) {
      raw = args.program
    } else {
      throw new Error('Provide either path (file) or program (inline JSON).')
    }

    if (session.dac?.isRunning) session.dac.stop()
    const t0 = performance.now()
    loadJSON(raw as { schema: string }, session)
    const wall_ms = performance.now() - t0
    return {
      instances:   [...session.instanceRegistry.keys()],
      wiring:      session.inputExprNodes.size,
      outputs:     session.graphOutputs.length,
      params:      [...session.paramRegistry.keys()],
      triggers:    [...session.triggerRegistry.keys()],
      timing: { wall_ms },
    }
  })
}

function handleSave() {
  return wrap(() => ({ program: saveProgramFromSession(session) }))
}

function handleMerge(args: Record<string, unknown>) {
  return wrap(() => {
    const raw = args.program ?? args.patch
    if (!raw) throw new Error('Provide a program or patch object.')
    const prog = parseProgram(raw) as ProgramJSON
    mergeProgramIntoSession(prog, session)
    return {
      instances:   [...session.instanceRegistry.keys()],
      wiring:      session.inputExprNodes.size,
      outputs:     session.graphOutputs.length,
      params:      [...session.paramRegistry.keys()],
      triggers:    [...session.triggerRegistry.keys()],
    }
  })
}

// ─── Tool dispatcher ─────────────────────────────────────────────────────────

function handleTool(name: string, args: Record<string, unknown>) {
  switch (name) {

    // ── New unified tools ─────────────────────────────────────────────────

    case 'define_program':
      return handleDefineProgram(args)

    case 'add_instance':
      return handleAddInstance(args.program as string, args.instance_name as string)

    case 'remove_instance':
      return handleRemoveInstance(args.instance_name as string)

    case 'replicate':
      return handleReplicate(
        args.program as string,
        args.count as number,
        args.name_prefix as string | undefined,
      )

    case 'wire_chain':
      return handleWireChain(args)

    case 'wire_zip':
      return handleWireZip(args)

    case 'fan_out':
      return handleFanOut(args)

    case 'fan_in':
      return handleFanIn(args)

    case 'feedback':
      return handleFeedback(args)

    case 'list_programs':
      return handleListPrograms()

    case 'list_instances':
      return handleListInstances()

    case 'get_info':
      return handleGetInfo(args.instance_name as string)

    case 'wire':
      return handleWire(args)

    case 'list_wiring':
      return handleListWiring(args.instance as string | undefined)

    case 'set_output':
      return handleSetOutput(args)

    case 'load':
      return handleLoad(args)

    case 'save':
      return handleSave()

    case 'merge':
      return handleMerge(args)

    // ── Audio / params (unchanged) ────────────────────────────────────────

    case 'start_audio': return wrap(() => {
      if (!session.dac) session.dac = DAC.fromRuntime(session.runtime._h)

      const deviceName = args.device_name as string | undefined
      if (deviceName) {
        const devices = DAC.listDevices()
        const match = devices.find(d => d.name.toLowerCase().includes(deviceName.toLowerCase()))
        if (!match)
          throw new Error(`No device matching '${deviceName}'. Available: ${devices.map(d => d.name).join(', ')}`)
        if (session.dac.isRunning) {
          session.dac.switchDevice(match.id)
        } else {
          session.dac.start()
          session.dac.switchDevice(match.id)
        }
        return { is_running: session.dac.isRunning, device: match.name }
      }

      if (!session.dac.isRunning) session.dac.start()
      return { is_running: session.dac.isRunning }
    })

    case 'stop_audio': return wrap(() => {
      if (!session.dac) throw new Error('DAC has not been created yet.')
      session.dac.stop()
      return { is_running: session.dac.isRunning }
    })

    case 'audio_status': return wrap(() => {
      if (!session.dac) return { is_running: false }
      return {
        is_running:      session.dac.isRunning,
        is_reconnecting: session.dac.isReconnecting,
        stats:           session.dac.callbackStats(),
      }
    })

    case 'set_param': return wrap(() => {
      const paramName = args.name as string
      const value     = args.value as number
      const p = session.paramRegistry.get(paramName)
      if (!p) {
        const known = [...session.paramRegistry.keys()].join(', ')
        throw new Error(`No param named '${paramName}'. Known: ${known || '(none)'}`)
      }
      p.value = value
      return { name: paramName, value: p.value }
    })

    case 'list_params': return wrap(() => {
      const params = [...session.paramRegistry.entries()].map(([name, p]) => ({
        name, type: 'param', value: p.value,
      }))
      const triggers = [...session.triggerRegistry.keys()].map(name => ({
        name, type: 'trigger',
      }))
      return [...params, ...triggers]
    })

    default:
      return fail(`Unknown tool: '${name}'`)
  }
}

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
    description: 'Reference doc for the tropical_program_1 schema with worked examples.',
    mimeType:    'text/markdown',
  },
]

function renderProgramCatalog(): string {
  const lines: string[] = ['# tropical program catalog\n']
  for (const [typeName, type] of session.typeRegistry) {
    lines.push(`## ${typeName}`)
    const d = type._def
    const defaultsMap = d.rawInputDefaults as Record<string, unknown>
    const inputParts = d.inputNames.map(n => {
      const val = defaultsMap[n]
      return val !== undefined ? `${n}=${JSON.stringify(val)}` : n
    })
    lines.push(`Inputs:  ${inputParts.join(', ')}`)
    lines.push(`Outputs: ${d.outputNames.join(', ')}`)
    lines.push('')
  }
  return lines.join('\n')
}

const PROGRAM_FORMAT_DOC = `# tropical program format

Programs are the unified representation for all DSP in tropical. A program can be:
- **A leaf program** (has process, computes directly)
- **A graph program** (has instances and audio_outputs)
- **A composite** (has instances, inputs, and outputs — a reusable graph)

## tropical_program_1

A program with instances wired together:

{ "schema": "tropical_program_1", "name": "MyPatch",
  "programs": { "Gain": { "schema": "tropical_program_1", "name": "Gain",
    "inputs": ["audio", "cv"], "outputs": ["out"],
    "process": { "outputs": { "out": { "op": "mul", "args": [{ "op": "input", "name": "audio" }, { "op": "input", "name": "cv" }] } } }
  }},
  "instances": {
    "osc": { "program": "VCO", "inputs": { "freq": 440 } },
    "amp": { "program": "Gain", "inputs": { "audio": { "op": "ref", "instance": "osc", "output": "sin" }, "cv": 0.5 } }
  },
  "audio_outputs": [{ "instance": "amp", "output": "out" }]
}

Key fields: schema, name, inputs/outputs (leaf/composite), process (leaf body),
programs (inline subprogram defs), instances (instantiated subprograms with wiring),
audio_outputs (graph output routing), params, regs, delays.

## Combinators (compile-time expansion)

- generate: { "op": "generate", "count": 4, "var": "i", "body": <expr> }
- iterate: { "op": "iterate", "count": 4, "init": <expr>, "var": "x", "body": <expr> }
- fold: { "op": "fold", "over": <array>, "init": <expr>, "acc_var": "acc", "elem_var": "x", "body": <expr> }
- scan: like fold but keeps intermediates as array
- map2: { "op": "map2", "over": <array>, "elem_var": "x", "body": <expr> }
- zip_with: { "op": "zip_with", "a": <arr>, "b": <arr>, "x_var": "x", "y_var": "y", "body": <expr> }
- chain: { "op": "chain", "count": 3, "init": <expr>, "var": "x", "body": <expr> }
- let: { "op": "let", "bind": { "x": <expr> }, "in": <expr> }
- binding: { "op": "binding", "name": "x" } — variable reference in combinator bodies

## Expression node format (ExprNode)

Used in instance input wiring and inline program process definitions.

- **Literal**: \`3.14\` or \`true\`
- **Reference**: \`{ "op": "ref", "instance": "osc", "output": "sin" }\` — routes another instance's output to this input
- **Input port**: \`{ "op": "input", "name": "freq" }\` — (inside program definitions only)
- **Param / Trigger**: \`{ "op": "param", "name": "cutoff" }\` / \`{ "op": "trigger", "name": "gate" }\`
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
- **Builtins**: \`{ "op": "sample_rate" }\`, \`{ "op": "sample_index" }\`
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
- \`tropical://program-format\` — tropical_program_1 schema reference with worked examples

## Choose the right tool for the job

### New program (starting from scratch)
Use \`load\` with a **complete** tropical_program_1 JSON object in a single call.
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
