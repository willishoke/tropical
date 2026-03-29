/**
 * TypeScript MCP server — replaces egress/mcp_server.py.
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
  makeSession, loadModuleFromJSON, loadPatchFromJSON, mergePatchFromJSON, savePatchToJSON,
  buildExpr, SessionState, ModuleDefJSON, ExprNode, PatchJSON,
} from './patch.js'
import { loadBuiltins }        from './module_library.js'
import { DAC }                 from './audio.js'
import { Param, Trigger }      from './param.js'
import { coerce }              from './expr.js'

// ─── Session ──────────────────────────────────────────────────────────────────

const session: SessionState = makeSession()
loadBuiltins(session.typeRegistry)

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

// ─── Tool definitions ─────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: 'define_module',
    description: 'Define a reusable DSP module type from a JSON definition and register it. Returns the type name and port names.',
    inputSchema: {
      type: 'object',
      properties: {
        def: { type: 'object', description: 'ModuleDefJSON object (name, inputs, outputs, regs, delays, nested, process, sample_rate, input_defaults)' },
      },
      required: ['def'],
    },
  },
  {
    name: 'instantiate_module',
    description: 'Instantiate a registered module type with a specific instance name.',
    inputSchema: {
      type: 'object',
      properties: {
        type_name:     { type: 'string', description: 'Registered type name (builtin or user-defined)' },
        instance_name: { type: 'string', description: 'Unique name for this instance' },
      },
      required: ['type_name', 'instance_name'],
    },
  },
  {
    name: 'remove_module',
    description: 'Remove a module instance from the graph.',
    inputSchema: {
      type: 'object',
      properties: { instance_name: { type: 'string' } },
      required: ['instance_name'],
    },
  },
  {
    name: 'list_module_types',
    description: 'List all registered module types (builtins + user-defined) with their input/output port names and input defaults. Call this before building a patch to discover what is available — do NOT use list_modules for this; list_modules shows live instances only.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'list_modules',
    description: 'List all live module instances (already instantiated in the current session) with their port names. This shows running instances, not available types — use list_module_types to discover what can be instantiated.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'get_module_info',
    description: 'Return detailed info about a module instance including port indices and connections.',
    inputSchema: {
      type: 'object',
      properties: { instance_name: { type: 'string' } },
      required: ['instance_name'],
    },
  },
  {
    name: 'connect_modules',
    description: 'Connect an output port of one module to an input port of another. Ports may be names or integer indices.',
    inputSchema: {
      type: 'object',
      properties: {
        src_module:  { type: 'string' },
        src_output:  { description: 'Output port name or index' },
        dst_module:  { type: 'string' },
        dst_input:   { description: 'Input port name or index' },
      },
      required: ['src_module', 'src_output', 'dst_module', 'dst_input'],
    },
  },
  {
    name: 'disconnect_modules',
    description: 'Disconnect two module ports.',
    inputSchema: {
      type: 'object',
      properties: {
        src_module:  { type: 'string' },
        src_output:  { description: 'Output port name or index' },
        dst_module:  { type: 'string' },
        dst_input:   { description: 'Input port name or index' },
      },
      required: ['src_module', 'src_output', 'dst_module', 'dst_input'],
    },
  },
  {
    name: 'list_connections',
    description: 'List all tracked connections in the current patch.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'set_module_input',
    description: 'Set a module input to a constant value or an expression. Pass a number for simple values, or an ExprNode object for signal-rate expressions (e.g. {"op":"ref","module":"VCO1","output":"saw"}).',
    inputSchema: {
      type: 'object',
      properties: {
        instance_name: { type: 'string' },
        input_name:    { description: 'Input port name or index' },
        expr:          { description: 'ExprNode: number, bool, array, or {op, ...} object' },
      },
      required: ['instance_name', 'input_name', 'expr'],
    },
  },
  {
    name: 'set_inputs_batch',
    description: 'Set multiple module inputs in a single JIT compilation pass. Use this instead of repeated set_module_input calls when updating many inputs at once (e.g. tuning all oscillators). Triggers one recompile per module rather than one per call.',
    inputSchema: {
      type: 'object',
      properties: {
        updates: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              instance_name: { type: 'string' },
              input_name:    { description: 'Input port name or index' },
              expr:          { description: 'ExprNode: number, bool, array, or {op, ...} object' },
            },
            required: ['instance_name', 'input_name', 'expr'],
          },
        },
      },
      required: ['updates'],
    },
  },
  {
    name: 'add_graph_output',
    description: 'Add a module output to the graph mix output.',
    inputSchema: {
      type: 'object',
      properties: {
        module_name:  { type: 'string' },
        output_name:  { description: 'Output port name or index' },
      },
      required: ['module_name', 'output_name'],
    },
  },
  {
    name: 'remove_graph_output',
    description: 'Remove a graph output from session tracking (graph has no remove_output API; output persists until graph rebuild).',
    inputSchema: {
      type: 'object',
      properties: {
        module_name: { type: 'string' },
        output_name: { description: 'Output port name or index' },
      },
      required: ['module_name', 'output_name'],
    },
  },
  {
    name: 'merge_patch',
    description: 'Add modules, connections, outputs, and params from a patch into the existing graph without clearing it. Fails fast with an error on name collisions, leaving the graph intact. Use this for subpatching: load patch A, load patch B with merge_patch, then wire them together with connect_modules.',
    inputSchema: {
      type: 'object',
      properties: {
        patch: { type: 'object', description: 'PatchJSON object with schema: "egress_patch_1"' },
      },
      required: ['patch'],
    },
  },
  {
    name: 'load_patch',
    description: 'Load a patch. Stops audio, recreates the graph, and rebuilds session state. Provide either patch_path (preferred — path to a .json file on disk) or patch (inline PatchJSON object).',
    inputSchema: {
      type: 'object',
      properties: {
        patch_path: { type: 'string', description: 'Absolute or relative path to an egress_patch_1 JSON file on disk. Preferred over inline patch.' },
        patch: { type: 'object', description: 'Inline PatchJSON object with schema: "egress_patch_1". Use patch_path instead when the patch exists as a file.' },
      },
    },
  },
  {
    name: 'save_patch',
    description: 'Serialize the current session to an egress_patch_1 JSON patch object.',
    inputSchema: { type: 'object', properties: {} },
  },
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
  {
    name: 'get_profile_stats',
    description: 'Return graph profiling statistics: per-callback timing, JIT fusion usage, and per-module call timing. Module-level timing and fused-sync detail are only populated when built with -DEGRESS_PROFILE=ON (make profile); JIT fusion flags are always available.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'reset_profile_stats',
    description: 'Reset all accumulated graph profile statistics to zero.',
    inputSchema: { type: 'object', properties: {} },
  },
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
]

// ─── Tool handlers ────────────────────────────────────────────────────────────

function handleTool(name: string, args: Record<string, unknown>) {
  switch (name) {

    // ── define_module ──────────────────────────────────────────────────────
    case 'define_module': return wrap(() => {
      const def = args.def as ModuleDefJSON
      if (!def?.name) throw new Error('def.name is required.')
      const type = loadModuleFromJSON(def, session)
      session.typeRegistry.set(type.name, type)
      return { type_name: type.name, inputs: type._def.inputNames, outputs: type._def.outputNames }
    })

    // ── instantiate_module ─────────────────────────────────────────────────
    case 'instantiate_module': return wrap(() => {
      const typeName     = args.type_name as string
      const instanceName = args.instance_name as string
      if (session.instanceRegistry.has(instanceName))
        throw new Error(`Instance '${instanceName}' already exists.`)
      const type = session.typeRegistry.get(typeName)
      if (!type)
        throw new Error(`Unknown type '${typeName}'. Known: ${[...session.typeRegistry.keys()].join(', ')}`)
      const inst = type.instantiateAs(session.graph, instanceName)
      session.instanceRegistry.set(instanceName, inst)
      return instanceSummary(instanceName)
    })

    // ── remove_module ──────────────────────────────────────────────────────
    case 'remove_module': return wrap(() => {
      const instanceName = args.instance_name as string
      if (!session.instanceRegistry.has(instanceName))
        throw new Error(`No instance named '${instanceName}'.`)
      session.graph.removeModule(instanceName)
      session.instanceRegistry.delete(instanceName)
      session.connections = session.connections.filter(
        c => c.src !== instanceName && c.dst !== instanceName,
      )
      session.graphOutputs = session.graphOutputs.filter(o => o.module !== instanceName)
      return { removed: instanceName }
    })

    // ── list_module_types ──────────────────────────────────────────────────
    case 'list_module_types': return wrap(() =>
      [...session.typeRegistry.entries()].map(([typeName, type]) => {
        const d = type._def
        const defaultsMap = (d.rawInputDefaults ?? {}) as Record<string, unknown>
        return {
          type_name: typeName,
          inputs:    d.inputNames.map(n => ({
            name: n,
            default: defaultsMap[n] ?? null,
          })),
          outputs: d.outputNames,
        }
      }),
    )

    // ── list_modules ───────────────────────────────────────────────────────
    case 'list_modules': return wrap(() =>
      [...session.instanceRegistry.keys()].map(instanceSummary),
    )

    // ── get_module_info ────────────────────────────────────────────────────
    case 'get_module_info': return wrap(() => {
      const instanceName = args.instance_name as string
      const inst = session.instanceRegistry.get(instanceName)
      if (!inst) throw new Error(`No instance named '${instanceName}'.`)
      const conns = session.connections.filter(
        c => c.src === instanceName || c.dst === instanceName,
      )
      return {
        name: instanceName,
        type_name: inst.typeName,
        inputs:  inst.inputNames.map((n, i) => ({ name: n, index: i })),
        outputs: inst.outputNames.map((n, i) => ({ name: n, index: i })),
        connections: conns,
      }
    })

    // ── connect_modules ────────────────────────────────────────────────────
    case 'connect_modules': return wrap(() => {
      const srcInst = session.instanceRegistry.get(args.src_module as string)
      if (!srcInst) throw new Error(`No instance named '${args.src_module}'.`)
      const dstInst = session.instanceRegistry.get(args.dst_module as string)
      if (!dstInst) throw new Error(`No instance named '${args.dst_module}'.`)

      const srcId = resolveOutputIdx(srcInst, args.src_output as string | number)
      const dstId = resolveInputIdx(dstInst,  args.dst_input  as string | number)

      const ok = session.graph.connect(args.src_module as string, srcId, args.dst_module as string, dstId)
      if (!ok) throw new Error(`graph.connect returned false.`)

      const conn = {
        src: args.src_module as string, srcOutput: srcInst.outputNames[srcId],
        dst: args.dst_module as string, dstInput:  dstInst.inputNames[dstId],
      }
      session.connections.push(conn)
      return { src: conn.src, src_output: conn.srcOutput, dst: conn.dst, dst_input: conn.dstInput }
    })

    // ── disconnect_modules ─────────────────────────────────────────────────
    case 'disconnect_modules': return wrap(() => {
      const srcInst = session.instanceRegistry.get(args.src_module as string)
      if (!srcInst) throw new Error(`No instance named '${args.src_module}'.`)
      const dstInst = session.instanceRegistry.get(args.dst_module as string)
      if (!dstInst) throw new Error(`No instance named '${args.dst_module}'.`)

      const srcId = resolveOutputIdx(srcInst, args.src_output as string | number)
      const dstId = resolveInputIdx(dstInst,  args.dst_input  as string | number)

      const ok = session.graph.disconnect(args.src_module as string, srcId, args.dst_module as string, dstId)
      if (!ok) throw new Error(`graph.disconnect returned false.`)

      const srcOut = srcInst.outputNames[srcId]
      const dstIn  = dstInst.inputNames[dstId]
      session.connections = session.connections.filter(
        c => !(c.src === args.src_module && c.srcOutput === srcOut &&
               c.dst === args.dst_module && c.dstInput  === dstIn),
      )
      return { src: args.src_module, src_output: srcOut, dst: args.dst_module, dst_input: dstIn }
    })

    // ── list_connections ───────────────────────────────────────────────────
    case 'list_connections': return wrap(() =>
      session.connections.map(c => ({
        src: c.src, src_output: c.srcOutput,
        dst: c.dst, dst_input:  c.dstInput,
      })),
    )

    // ── set_module_input ───────────────────────────────────────────────────
    case 'set_module_input': return wrap(() => {
      const instanceName = args.instance_name as string
      const inst = session.instanceRegistry.get(instanceName)
      if (!inst) throw new Error(`No instance named '${instanceName}'.`)

      const rawInput = args.input_name as string | number
      const inputId = typeof rawInput === 'number'
        ? rawInput
        : (rawInput.match(/^\d+$/) ? parseInt(rawInput, 10) : inst.inputIndex(rawInput))

      const node = args.expr as ExprNode
      const ctx = {
        inputNames: [], regNames: [],
        paramRegistry: session.paramRegistry, triggerRegistry: session.triggerRegistry,
        instanceRegistry: session.instanceRegistry, typeRegistry: session.typeRegistry,
        delayRefs: new Map(), nestedAliases: new Map(),
      }
      const expr = buildExpr(node, ctx)
      session.graph.setInputExpr(instanceName, inputId, expr)
      session.inputExprNodes.set(`${instanceName}:${inputId}`, node)

      const resolvedName = inst.inputNames[inputId] ?? String(inputId)
      return { module: instanceName, input: resolvedName, expr: node }
    })

    // ── set_inputs_batch ───────────────────────────────────────────────────
    case 'set_inputs_batch': return wrap(() => {
      const updates = args.updates as Array<{
        instance_name: string; input_name: string | number; expr: ExprNode
      }>
      const ctx = {
        inputNames: [], regNames: [],
        paramRegistry: session.paramRegistry, triggerRegistry: session.triggerRegistry,
        instanceRegistry: session.instanceRegistry, typeRegistry: session.typeRegistry,
        delayRefs: new Map(), nestedAliases: new Map(),
      }
      session.graph.beginUpdate()
      try {
        const results = []
        for (const u of updates) {
          const inst = session.instanceRegistry.get(u.instance_name)
          if (!inst) throw new Error(`No instance named '${u.instance_name}'.`)
          const raw = u.input_name
          const inputId = typeof raw === 'number'
            ? raw
            : (String(raw).match(/^\d+$/) ? parseInt(String(raw), 10) : inst.inputIndex(String(raw)))
          const expr = buildExpr(u.expr, ctx)
          session.graph.setInputExpr(u.instance_name, inputId, expr)
          session.inputExprNodes.set(`${u.instance_name}:${inputId}`, u.expr)
          results.push({ module: u.instance_name, input: inst.inputNames[inputId] ?? String(inputId), expr: u.expr })
        }
        return results
      } finally {
        session.graph.endUpdate()
      }
    })

    // ── add_graph_output ───────────────────────────────────────────────────
    case 'add_graph_output': return wrap(() => {
      const moduleName = args.module_name as string
      const inst = session.instanceRegistry.get(moduleName)
      if (!inst) throw new Error(`No instance named '${moduleName}'.`)

      const rawOut = args.output_name as string | number
      const outId = typeof rawOut === 'number'
        ? rawOut
        : (String(rawOut).match(/^\d+$/) ? parseInt(String(rawOut), 10) : inst.outputIndex(rawOut))

      const added = session.graph.addOutput(moduleName, outId)
      if (!added) throw new Error(`graph.addOutput returned false.`)

      const resolvedName = inst.outputNames[outId]
      const entry = { module: moduleName, output: resolvedName }
      if (!session.graphOutputs.some(o => o.module === entry.module && o.output === entry.output))
        session.graphOutputs.push(entry)
      return entry
    })

    // ── remove_graph_output ────────────────────────────────────────────────
    case 'remove_graph_output': return wrap(() => {
      const moduleName = args.module_name as string
      const inst = session.instanceRegistry.get(moduleName)
      if (!inst) throw new Error(`No instance named '${moduleName}'.`)

      const rawOut = args.output_name as string | number
      const outId = typeof rawOut === 'number'
        ? rawOut
        : (String(rawOut).match(/^\d+$/) ? parseInt(String(rawOut), 10) : inst.outputIndex(rawOut))
      const resolvedName = inst.outputNames[outId]

      const before = session.graphOutputs.length
      session.graphOutputs = session.graphOutputs.filter(
        o => !(o.module === moduleName && o.output === resolvedName),
      )
      return {
        removed_from_tracking: before - session.graphOutputs.length,
        note: 'graph has no remove_output API; output persists until graph rebuild.',
      }
    })

    // ── merge_patch ────────────────────────────────────────────────────────
    case 'merge_patch': return wrap(() => {
      const patch = args.patch as PatchJSON
      if (patch?.schema !== 'egress_patch_1')
        throw new Error(`Unsupported schema: ${(patch as any)?.schema}. Expected 'egress_patch_1'.`)
      mergePatchFromJSON(patch, session)
      return {
        modules:     [...session.instanceRegistry.keys()],
        connections: session.connections.length,
        outputs:     session.graphOutputs.length,
        params:      [...session.paramRegistry.keys()],
        triggers:    [...session.triggerRegistry.keys()],
      }
    })

    // ── load_patch ─────────────────────────────────────────────────────────
    case 'load_patch': return wrap(() => {
      let patch: PatchJSON
      if (args.patch_path) {
        const raw = readFileSync(args.patch_path as string, 'utf-8')
        patch = JSON.parse(raw) as PatchJSON
      } else {
        patch = args.patch as PatchJSON
      }
      if (patch?.schema !== 'egress_patch_1')
        throw new Error(`Unsupported schema: ${(patch as any)?.schema}. Expected 'egress_patch_1'.`)
      if (session.dac?.isRunning) session.dac.stop()
      loadPatchFromJSON(patch, session)
      return {
        modules:     [...session.instanceRegistry.keys()],
        connections: session.connections.length,
        outputs:     session.graphOutputs.length,
        params:      [...session.paramRegistry.keys()],
        triggers:    [...session.triggerRegistry.keys()],
      }
    })

    // ── save_patch ─────────────────────────────────────────────────────────
    case 'save_patch': return wrap(() => ({ patch: savePatchToJSON(session) }))

    // ── start_audio ────────────────────────────────────────────────────────
    case 'start_audio': return wrap(() => {
      if (!session.dac) session.dac = new DAC(session.graph._h)

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

    // ── stop_audio ─────────────────────────────────────────────────────────
    case 'stop_audio': return wrap(() => {
      if (!session.dac) throw new Error('DAC has not been created yet.')
      session.dac.stop()
      return { is_running: session.dac.isRunning }
    })

    // ── audio_status ───────────────────────────────────────────────────────
    case 'audio_status': return wrap(() => {
      if (!session.dac) return { is_running: false }
      return {
        is_running:      session.dac.isRunning,
        is_reconnecting: session.dac.isReconnecting,
        stats:           session.dac.callbackStats(),
      }
    })

    // ── get_profile_stats ──────────────────────────────────────────────────
    case 'get_profile_stats': return wrap(() => session.graph.profileStats())

    // ── reset_profile_stats ────────────────────────────────────────────────
    case 'reset_profile_stats': return wrap(() => {
      session.graph.resetProfileStats()
      return { reset: true }
    })

    // ── set_param ──────────────────────────────────────────────────────────
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

    // ── list_params ────────────────────────────────────────────────────────
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
    uri:         'egress://modules',
    name:        'Module catalog',
    description: 'Markdown catalog of all registered module types with inputs, outputs, and default values.',
    mimeType:    'text/markdown',
  },
  {
    uri:         'egress://patch-format',
    name:        'Patch format',
    description: 'Reference doc for the egress_patch_1 patch schema with a complete worked example.',
    mimeType:    'text/markdown',
  },
]

function renderModuleCatalog(): string {
  const lines: string[] = ['# egress module catalog\n']
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

const PATCH_FORMAT_DOC = `# egress_patch_1 patch format

Patches are JSON objects with \`"schema": "egress_patch_1"\`.

## Top-level fields

- \`schema\` (required): \`"egress_patch_1"\`
- \`types\` (optional): inline module type definitions (avoids a separate \`define_module\` call)
- \`modules\`: list of \`{ type_name, instance_name }\` objects
- \`connections\`: list of \`{ src, src_output, dst, dst_input }\` objects
- \`outputs\`: list of \`{ module, output }\` objects to route to the graph mix
- \`params\`: map of \`param_name → value\`
- \`graph\`: optional graph settings (\`worker_count\`, \`fusion_enabled\`, \`buffer_length\`)

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
  "schema": "egress_patch_1",
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
    { "type_name": "VCO", "instance_name": "osc" },
    { "type_name": "VCA", "instance_name": "vca" }
  ],
  "connections": [
    { "src": "osc", "src_output": "sin", "dst": "vca", "dst_input": "audio" }
  ],
  "outputs": [
    { "module": "vca", "output": "out" }
  ],
  "params": {},
  "graph": { "worker_count": 1, "fusion_enabled": false }
}
\`\`\`

## Expression node format (ExprNode)

Used in \`set_module_input\` and inline type process definitions:

- Literal: \`3.14\` or \`true\`
- Reference: \`{ "op": "ref", "module": "osc", "output": "sin" }\`
- Input: \`{ "op": "input", "name": "freq" }\`
- Binary: \`{ "op": "mul", "lhs": <expr>, "rhs": <expr> }\`
  - ops: \`add\`, \`sub\`, \`mul\`, \`div\`, \`mod\`, \`pow\`
- Unary: \`{ "op": "neg", "operand": <expr> }\`
  - ops: \`neg\`, \`abs\`, \`sin\`, \`log\`
- Compare: \`{ "op": "lt", "lhs": <expr>, "rhs": <expr> }\`
  - ops: \`lt\`, \`lte\`, \`gt\`, \`gte\`
- Clamp: \`{ "op": "clamp", "value": <expr>, "min": <expr>, "max": <expr> }\`
- Builtins: \`{ "op": "sample_rate" }\`, \`{ "op": "sample_index" }\`
`

// ─── Prompts ──────────────────────────────────────────────────────────────────

const PROMPTS = [
  {
    name:        'build-patch',
    description: 'Three-tiered workflow guidance for building and editing egress patches efficiently.',
  },
]

const BUILD_PATCH_PROMPT = `# build-patch workflow

Before writing any patch YAML, always fetch both resources:
- \`egress://modules\` — full catalog of available module types with inputs, defaults, and outputs
- \`egress://patch-format\` — egress_patch_1 schema reference with a worked example

## Choose the right tool for the job

### New patch (starting from scratch)
Use \`load_patch\` with a **complete** egress_patch_1 JSON object in a single call.
Do not call \`instantiate_module\`, \`connect_modules\`, or \`set_module_input\` one-by-one —
that requires 40+ round trips and recompiles the JIT graph on every input change.

### Extending an existing patch (adding new modules)
Use \`merge_patch\` with a partial patch containing only the new modules, connections,
outputs, and params. Then use \`connect_modules\` to wire the new modules to existing ones.
Do not use \`load_patch\` — it tears down the entire graph and loses the existing session.

### Targeted edits (changing a value, tweaking a param)
Use \`set_module_input\` or \`set_param\` directly on the specific input or param.
Do not reload or rebuild the patch for a single value change.

## Writing patch YAML

- Embed simple arithmetic types (VCA, mixer, attenuator, etc.) inline in the \`types\` block
  rather than calling \`define_module\` separately. This keeps the patch self-contained.
- Use named ports (e.g. \`"src_output": "sin"\`) rather than integer indices — more readable
  and robust to future port reordering.
- Wire everything in one patch object where possible; minimize round trips.
`

// ─── Server wiring ────────────────────────────────────────────────────────────

const server = new Server(
  { name: 'egress', version: '0.2.0' },
  { capabilities: { tools: {}, resources: {}, prompts: {} } },
)

server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: RESOURCES }))

server.setRequestHandler(ReadResourceRequestSchema, async (req) => {
  const { uri } = req.params
  if (uri === 'egress://modules') {
    return { contents: [{ uri, mimeType: 'text/markdown', text: renderModuleCatalog() }] }
  }
  if (uri === 'egress://patch-format') {
    return { contents: [{ uri, mimeType: 'text/markdown', text: PATCH_FORMAT_DOC }] }
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
