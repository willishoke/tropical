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
  makeSession, nextName, loadModuleFromJSON, loadPatchFromJSON, loadJSON, mergePatchFromJSON, savePatchToJSON,
  prettyExpr, SessionState, ModuleDefJSON, ExprNode, PatchJSON,
} from '../compiler/patch.js'
import { parseModuleDef, parsePatch, parseProgram } from '../compiler/schema.js'
import { saveProgramFromSession, type ProgramJSON } from '../compiler/program.js'
import { loadBuiltins }        from '../compiler/module_library.js'
import { DAC }                 from '../compiler/runtime/audio.js'
import { Param, Trigger }      from '../compiler/runtime/param.js'
import { applyFlatPlan }  from '../compiler/apply_plan.js'
import { checkArrayConnection } from '../compiler/array_wiring.js'
import { exprDependencies }     from '../compiler/compiler.js'

// ─── Session ──────────────────────────────────────────────────────────────────

const session: SessionState = makeSession()
loadBuiltins(session.typeRegistry)

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Normalize incoming expressions: convert {op, a, b} shorthand to {op, args: [a, b]}.
 * LLMs sometimes produce the shorthand form instead of the documented args-array format.
 */
function normalizeExprNode(node: ExprNode): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(normalizeExprNode)
  if (typeof node !== 'object' || node === null) return node
  const obj = node as Record<string, unknown>
  // Convert {op, a, b} → {op, args: [a, b]} (binary shorthand)
  if (obj.op && obj.a !== undefined && !obj.args) {
    const args = obj.b !== undefined
      ? [normalizeExprNode(obj.a as ExprNode), normalizeExprNode(obj.b as ExprNode)]
      : [normalizeExprNode(obj.a as ExprNode)]
    const { a: _a, b: _b, ...rest } = obj
    return { ...rest, args } as ExprNode
  }
  // Recursively normalize args and other nested expressions
  if (obj.args && Array.isArray(obj.args)) {
    return { ...obj, args: (obj.args as ExprNode[]).map(normalizeExprNode) } as ExprNode
  }
  return node
}

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
      const srcInst = session.instanceRegistry.get(obj.module as string)
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
    description: 'Define a reusable DSP program type and register it. Accepts a ProgramJSON (tropical_program_1) or ModuleDefJSON. Returns the type name and port names.',
    inputSchema: {
      type: 'object',
      properties: {
        def: { type: 'object', description: 'ProgramJSON or ModuleDefJSON object defining the program' },
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
    description: 'Set and/or remove input wiring in a single recompile. Use `set` to wire inputs (each is {instance, input, expr}), `remove` to disconnect (each is {instance, input}). This replaces connect_modules, disconnect_modules, set_module_input, and set_inputs_batch.',
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
    description: 'Load a program or patch. Accepts tropical_program_1 or tropical_patch_1. Stops audio, recreates the session, and rebuilds state. Provide either path (preferred) or inline JSON.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Path to a .json file on disk.' },
        program: { type: 'object', description: 'Inline ProgramJSON or PatchJSON object.' },
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
        program: { type: 'object', description: 'ProgramJSON or PatchJSON object to merge.' },
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

  // ── Deprecated aliases (backward compatibility) ────────────────────────────

  {
    name: 'define_module',
    description: '[deprecated: use define_program] Define a reusable DSP module type from a JSON definition.',
    inputSchema: {
      type: 'object',
      properties: { def: { type: 'object' } },
      required: ['def'],
    },
  },
  {
    name: 'instantiate_module',
    description: '[deprecated: use add_instance] Instantiate a registered module type.',
    inputSchema: {
      type: 'object',
      properties: {
        type_name:     { type: 'string' },
        instance_name: { type: 'string' },
      },
      required: ['type_name', 'instance_name'],
    },
  },
  {
    name: 'remove_module',
    description: '[deprecated: use remove_instance] Remove a module instance.',
    inputSchema: {
      type: 'object',
      properties: { instance_name: { type: 'string' } },
      required: ['instance_name'],
    },
  },
  {
    name: 'list_module_types',
    description: '[deprecated: use list_programs] List all registered module types.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'list_modules',
    description: '[deprecated: use list_instances] List all live module instances.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'get_module_info',
    description: '[deprecated: use get_info] Return info about a module instance.',
    inputSchema: {
      type: 'object',
      properties: { instance_name: { type: 'string' } },
      required: ['instance_name'],
    },
  },
  {
    name: 'connect_modules',
    description: '[deprecated: use wire] Connect module output to input.',
    inputSchema: {
      type: 'object',
      properties: {
        src_module: { type: 'string' }, src_output: {},
        dst_module: { type: 'string' }, dst_input: {},
      },
      required: ['src_module', 'src_output', 'dst_module', 'dst_input'],
    },
  },
  {
    name: 'disconnect_modules',
    description: '[deprecated: use wire with remove] Disconnect module ports.',
    inputSchema: {
      type: 'object',
      properties: {
        src_module: { type: 'string' }, src_output: {},
        dst_module: { type: 'string' }, dst_input: {},
      },
      required: ['src_module', 'src_output', 'dst_module', 'dst_input'],
    },
  },
  {
    name: 'set_module_input',
    description: '[deprecated: use wire] Set a module input to a value or expression.',
    inputSchema: {
      type: 'object',
      properties: {
        instance_name: { type: 'string' },
        input_name:    {},
        expr:          {},
      },
      required: ['instance_name', 'input_name', 'expr'],
    },
  },
  {
    name: 'set_inputs_batch',
    description: '[deprecated: use wire] Set multiple module inputs in one recompile.',
    inputSchema: {
      type: 'object',
      properties: {
        updates: { type: 'array', items: { type: 'object' } },
      },
      required: ['updates'],
    },
  },
  {
    name: 'list_inputs',
    description: '[deprecated: use list_wiring] List all wired inputs.',
    inputSchema: {
      type: 'object',
      properties: { module: { type: 'string' } },
    },
  },
  {
    name: 'add_graph_output',
    description: '[deprecated: use set_output] Add a module output to the audio mix.',
    inputSchema: {
      type: 'object',
      properties: { module_name: { type: 'string' }, output_name: {} },
      required: ['module_name', 'output_name'],
    },
  },
  {
    name: 'remove_graph_output',
    description: '[deprecated: use set_output] Remove an output from the audio mix.',
    inputSchema: {
      type: 'object',
      properties: { module_name: { type: 'string' }, output_name: {} },
      required: ['module_name', 'output_name'],
    },
  },
  {
    name: 'load_patch',
    description: '[deprecated: use load] Load a tropical_patch_1 patch.',
    inputSchema: {
      type: 'object',
      properties: {
        patch_path: { type: 'string' },
        patch: { type: 'object' },
      },
    },
  },
  {
    name: 'merge_patch',
    description: '[deprecated: use merge] Merge a patch into the session.',
    inputSchema: {
      type: 'object',
      properties: { patch: { type: 'object' } },
      required: ['patch'],
    },
  },
  {
    name: 'save_patch',
    description: '[deprecated: use save] Serialize session to tropical_patch_1.',
    inputSchema: { type: 'object', properties: {} },
  },
]

// ─── Tool handlers ────────────────────────────────────────────────────────────

// ─── Shared handler logic ─────────────────────────────────────────────────────

function handleDefineProgram(args: Record<string, unknown>) {
  return wrap(() => {
    const def = parseModuleDef(args.def) as ModuleDefJSON
    const type = loadModuleFromJSON(def, session)
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
    session.graphOutputs = session.graphOutputs.filter(o => o.module !== instanceName)
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
        inputs:    d.inputNames.map(n => ({ name: n, default: defaultsMap[n] ?? null })),
        outputs: d.outputNames,
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
        expr: session.inputExprNodes.get(`${instanceName}:${n}`) ?? null,
        pretty: session.inputExprNodes.has(`${instanceName}:${n}`)
          ? prettyExpr(session.inputExprNodes.get(`${instanceName}:${n}`)!, session.instanceRegistry)
          : null,
      })),
      outputs: inst.outputNames.map((n, i) => ({ name: n, index: i })),
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
      const { expr } = adaptInputExpr(normalizeExprNode(s.expr), inst.inputPortType(inputId), s.instance, resolvedName)
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
      session.graphOutputs.push({ module: o.instance, output: inst.outputNames[outId] })
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
    } else if (args.patch_path) {
      // backward compat for load_patch alias
      raw = JSON.parse(readFileSync(args.patch_path as string, 'utf-8'))
    } else if (args.patch) {
      raw = args.patch
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
  return wrap(() => ({ program: saveProgramFromSession(session, savePatchToJSON) }))
}

function handleMerge(args: Record<string, unknown>) {
  return wrap(() => {
    const raw = args.program ?? args.patch
    if (!raw) throw new Error('Provide a program or patch object.')
    const parsed = (raw as { schema?: string })
    if (parsed.schema === 'tropical_program_1' || !parsed.schema) {
      // If it looks like a program, convert to patch first
      // For now, merge only supports patch format natively
      const patch = parsePatch(raw) as PatchJSON
      mergePatchFromJSON(patch, session)
    } else {
      const patch = parsePatch(raw) as PatchJSON
      mergePatchFromJSON(patch, session)
    }
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

    // ── Deprecated aliases ────────────────────────────────────────────────

    case 'define_module':
      return handleDefineProgram(args)

    case 'instantiate_module':
      return handleAddInstance(args.type_name as string, args.instance_name as string)

    case 'remove_module':
      return handleRemoveInstance(args.instance_name as string)

    case 'list_module_types':
      return handleListPrograms()

    case 'list_modules':
      return handleListInstances()

    case 'get_module_info':
      return handleGetInfo(args.instance_name as string)

    case 'connect_modules': return wrap(() => {
      const srcInst = session.instanceRegistry.get(args.src_module as string)
      if (!srcInst) throw new Error(`No instance named '${args.src_module}'.`)
      const dstInst = session.instanceRegistry.get(args.dst_module as string)
      if (!dstInst) throw new Error(`No instance named '${args.dst_module}'.`)

      const srcId = resolveOutputIdx(srcInst, args.src_output as string | number)
      const dstId = resolveInputIdx(dstInst,  args.dst_input  as string | number)

      const srcOut = srcInst.outputNames[srcId]
      const dstIn  = dstInst.inputNames[dstId]
      const refExpr: ExprNode = { op: 'ref', module: args.src_module as string, output: srcOut }

      const srcType = srcInst.outputPortType(srcId)
      const dstType = dstInst.inputPortType(dstId)
      const check = checkArrayConnection(srcType, dstType, refExpr)
      if (!check.compatible) {
        throw new Error(
          `Connection '${args.src_module as string}'.${srcOut} → '${args.dst_module as string}'.${dstIn}: ${check.error}`
        )
      }

      const wiringExpr = check.broadcastExpr ?? refExpr
      session.inputExprNodes.set(`${args.dst_module as string}:${dstIn}`, wiringExpr)

      return { src: args.src_module, src_output: srcOut, dst: args.dst_module, dst_input: dstIn,
               ...(check.resultShape ? { broadcast_shape: check.resultShape } : {}),
               ...wire() }
    })

    case 'disconnect_modules': return wrap(() => {
      const dstInst = session.instanceRegistry.get(args.dst_module as string)
      if (!dstInst) throw new Error(`No instance named '${args.dst_module}'.`)
      const dstId = resolveInputIdx(dstInst, args.dst_input as string | number)
      const dstIn = dstInst.inputNames[dstId]
      session.inputExprNodes.delete(`${args.dst_module as string}:${dstIn}`)
      return { dst: args.dst_module, dst_input: dstIn, ...wire() }
    })

    case 'set_module_input': return wrap(() => {
      const instanceName = args.instance_name as string
      const inst = session.instanceRegistry.get(instanceName)
      if (!inst) throw new Error(`No instance named '${instanceName}'.`)
      const rawInput = args.input_name as string | number
      const inputId = typeof rawInput === 'number' ? rawInput
        : (rawInput.match(/^\d+$/) ? parseInt(rawInput, 10) : inst.inputIndex(rawInput))
      const resolvedName = inst.inputNames[inputId] ?? String(inputId)
      const { expr: node, resultShape } = adaptInputExpr(normalizeExprNode(args.expr as ExprNode), inst.inputPortType(inputId), instanceName, resolvedName)
      session.inputExprNodes.set(`${instanceName}:${resolvedName}`, node)
      return { module: instanceName, input: resolvedName, expr: node,
               ...(resultShape ? { broadcast_shape: resultShape } : {}),
               ...wire() }
    })

    case 'set_inputs_batch': return wrap(() => {
      const updates = args.updates as Array<{
        instance_name: string; input_name: string | number; expr: ExprNode
      }>
      const results = []
      for (const u of updates) {
        const inst = session.instanceRegistry.get(u.instance_name)
        if (!inst) throw new Error(`No instance named '${u.instance_name}'.`)
        const raw = u.input_name
        const inputId = typeof raw === 'number' ? raw
          : (String(raw).match(/^\d+$/) ? parseInt(String(raw), 10) : inst.inputIndex(String(raw)))
        const resolvedName = inst.inputNames[inputId] ?? String(inputId)
        const { expr } = adaptInputExpr(normalizeExprNode(u.expr), inst.inputPortType(inputId), u.instance_name, resolvedName)
        session.inputExprNodes.set(`${u.instance_name}:${resolvedName}`, expr)
        results.push({ module: u.instance_name, input: resolvedName, expr })
      }
      return { updates: results, ...wire() }
    })

    case 'list_inputs':
      return handleListWiring(args.module as string | undefined)

    case 'add_graph_output': return wrap(() => {
      const moduleName = args.module_name as string
      const inst = session.instanceRegistry.get(moduleName)
      if (!inst) throw new Error(`No instance named '${moduleName}'.`)
      const rawOut = args.output_name as string | number
      const outId = typeof rawOut === 'number' ? rawOut
        : (String(rawOut).match(/^\d+$/) ? parseInt(String(rawOut), 10) : inst.outputIndex(rawOut))
      const resolvedName = inst.outputNames[outId]
      const entry = { module: moduleName, output: resolvedName }
      if (!session.graphOutputs.some(o => o.module === entry.module && o.output === entry.output))
        session.graphOutputs.push(entry)
      return { ...entry, ...wire() }
    })

    case 'remove_graph_output': return wrap(() => {
      const moduleName = args.module_name as string
      const inst = session.instanceRegistry.get(moduleName)
      if (!inst) throw new Error(`No instance named '${moduleName}'.`)
      const rawOut = args.output_name as string | number
      const outId = typeof rawOut === 'number' ? rawOut
        : (String(rawOut).match(/^\d+$/) ? parseInt(String(rawOut), 10) : inst.outputIndex(rawOut))
      const resolvedName = inst.outputNames[outId]
      const before = session.graphOutputs.length
      session.graphOutputs = session.graphOutputs.filter(
        o => !(o.module === moduleName && o.output === resolvedName),
      )
      return { removed: before - session.graphOutputs.length, ...wire() }
    })

    case 'load_patch':
      return handleLoad(args)

    case 'merge_patch':
      return handleMerge({ program: args.patch })

    case 'save_patch':
      return wrap(() => ({ patch: savePatchToJSON(session) }))

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
    description: 'Reference doc for the tropical_program_1 and tropical_patch_1 schemas with worked examples.',
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
- **A leaf program** (has process, computes directly — like a module definition)
- **A graph program** (has instances and audio_outputs — like a patch)
- **A composite** (has instances, inputs, and outputs — a reusable graph)

## tropical_program_1 (preferred)

A program with instances wired together:

{ "schema": "tropical_program_1", "name": "MyPatch",
  "programs": { "Gain": { "schema": "tropical_program_1", "name": "Gain",
    "inputs": ["audio", "cv"], "outputs": ["out"],
    "process": { "outputs": { "out": { "op": "mul", "args": [{ "op": "input", "name": "audio" }, { "op": "input", "name": "cv" }] } } }
  }},
  "instances": {
    "osc": { "program": "VCO", "inputs": { "freq": 440 } },
    "amp": { "program": "Gain", "inputs": { "audio": { "op": "ref", "module": "osc", "output": "sin" }, "cv": 0.5 } }
  },
  "audio_outputs": [{ "instance": "amp", "output": "out" }]
}

Key fields: schema, name, inputs/outputs (leaf/composite), process (leaf body),
programs (inline subprogram defs), instances (instantiated subprograms with wiring),
audio_outputs (graph output routing), params, regs, delays.

## tropical_patch_1 (legacy, still accepted by load tool)

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

# tropical_patch_1 patch format

Patches are JSON objects with \`"schema": "tropical_patch_1"\`.

## Top-level fields

- \`schema\` (required): \`"tropical_patch_1"\`
- \`types\` (optional): inline module type definitions (avoids a separate \`define_module\` call)
- \`modules\`: list of \`{ type, name }\` objects
- \`input_exprs\`: list of \`{ module, input, expr }\` objects — **canonical wiring mechanism**
- \`connections\` (legacy): list of \`{ src, src_output, dst, dst_input }\` objects; equivalent to \`input_exprs\` entries with \`{ "op": "ref" }\` expressions. Accepted on load for backward compatibility, not emitted on save.
- \`outputs\`: list of \`{ module, output }\` objects to route to the audio mix
- \`params\`: map of \`param_name → value\`

## Inline type definitions

Embed simple arithmetic types directly in \`types\` instead of calling \`define_module\` separately:

\`\`\`json
"types": [
  {
    "name": "VCA",
    "inputs": ["audio", "cv"],
    "outputs": ["out"],
    "regs": {},
    "process": {
      "outputs": { "out": { "op": "mul", "lhs": { "op": "input", "name": "audio" }, "rhs": { "op": "input", "name": "cv" } } },
      "next_regs": {}
    }
  }
]
\`\`\`

## Complete example: VCO → VCA → output

\`\`\`json
{
  "schema": "tropical_patch_1",
  "types": [
    {
      "name": "VCA",
      "inputs": ["audio", "cv"],
      "outputs": ["out"],
      "regs": {},
      "process": {
        "outputs": { "out": { "op": "mul", "lhs": { "op": "input", "name": "audio" }, "rhs": { "op": "input", "name": "cv" } } },
        "next_regs": {}
      }
    }
  ],
  "modules": [
    { "type": "VCO", "name": "osc" },
    { "type": "VCA", "name": "vca" }
  ],
  "input_exprs": [
    { "module": "vca", "input": "audio", "expr": { "op": "ref", "module": "osc", "output": "sin" } }
  ],
  "outputs": [
    { "module": "vca", "output": "out" }
  ],
  "params": {}
}
\`\`\`

## Expression node format (ExprNode)

Used in \`input_exprs\`, \`set_module_input\`, \`set_inputs_batch\`, and inline type process definitions.

- **Literal**: \`3.14\` or \`true\`
- **Reference**: \`{ "op": "ref", "module": "osc", "output": "sin" }\` — routes another module's output to this input
- **Input port**: \`{ "op": "input", "name": "freq" }\` — (inside type definitions only)
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
