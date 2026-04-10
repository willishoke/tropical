#!/usr/bin/env bun
/**
 * gen_stdlib.ts — Generate stdlib JSON files from TypeScript module definitions.
 *
 * Extracts the ExprNode trees from each built-in module's ModuleDef and
 * converts slot-based references (input id, reg id, etc.) back to named
 * references suitable for ProgramJSON format.
 *
 * Usage: bun scripts/gen_stdlib.ts
 */

import { writeFileSync } from 'fs'
import { join } from 'path'
import type { ModuleDef, NestedCallDef } from '../compiler/module.js'
import type { ExprNode } from '../compiler/expr.js'
import {
  vco, phaser, phaser16, clock, reverb,
  adEnvelope, adsrEnvelope, compressor, bassDrum,
  topoWaveguide, vca, bitCrusher, ladderFilter, noiseLFSR,
  delayLine,
} from '../compiler/module_library.js'

const STDLIB_DIR = join(import.meta.dir, '../stdlib')

interface ProgramJSON {
  schema: 'tropical_program_1'
  name: string
  inputs?: (string | { name: string; type: string })[]
  outputs?: (string | { name: string; type: string })[]
  regs?: Record<string, unknown>
  delays?: Record<string, { update: unknown; init: number }>
  input_defaults?: Record<string, unknown>
  programs?: Record<string, ProgramJSON>
  instances?: Record<string, { program: string; inputs: Record<string, unknown> }>
  process?: {
    outputs: Record<string, unknown>
    next_regs?: Record<string, unknown>
  }
}

/**
 * Convert a ModuleDef's internal ExprNode tree to ProgramJSON-compatible JSON,
 * replacing slot-based references with name-based ones.
 */
function convertNode(
  node: ExprNode,
  def: ModuleDef,
  nestedAliases: Map<number, string>,  // nodeId → alias name
  delayNames: string[],                 // nodeId → delay name
): unknown {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => convertNode(n, def, nestedAliases, delayNames))

  const obj = node as Record<string, unknown>
  const op = obj.op as string

  // Input reference: id → name
  if (op === 'input') {
    const id = obj.id as number
    const name = def.inputNames[id]
    return { op: 'input', name }
  }

  // Register reference: id → name
  if (op === 'reg') {
    const id = obj.id as number
    const name = def.registerNames[id]
    return { op: 'reg', name }
  }

  // Delay value reference: node_id → named delay_ref
  if (op === 'delay_value') {
    const nodeId = obj.node_id as number
    const name = delayNames[nodeId]
    return { op: 'delay_ref', id: name }
  }

  // Nested output: node_id → alias ref
  if (op === 'nested_output') {
    const nodeId = obj.node_id as number
    const outputId = obj.output_id as number
    const alias = nestedAliases.get(nodeId)
    if (!alias) throw new Error(`No alias for nested call node ${nodeId}`)
    return { op: 'nested_out', ref: alias, output: outputId }
  }

  // Generic op with args
  if ('args' in obj) {
    const args = (obj.args as ExprNode[]).map(a => convertNode(a, def, nestedAliases, delayNames))
    const result: Record<string, unknown> = { op }
    // Preserve any extra fields (like 'name' for named ops)
    for (const [k, v] of Object.entries(obj)) {
      if (k === 'op' || k === 'args') continue
      result[k] = v
    }
    result.args = args
    return result
  }

  // Ops without args (sample_rate, sample_index, etc.)
  return { ...obj }
}

/**
 * Convert a full ModuleDef to ProgramJSON.
 * Handles nested calls recursively, building programs/instances.
 */
function defToProgram(def: ModuleDef, visitedDefs = new Set<ModuleDef>()): ProgramJSON {
  // Prevent infinite recursion
  if (visitedDefs.has(def)) throw new Error(`Circular reference: ${def.typeName}`)
  visitedDefs.add(def)

  // Build nested call aliases and track nested programs.
  // Use object identity (same ModuleDef reference) to deduplicate programs,
  // since PureFunction instances all share the typeName '_pure'.
  const nestedAliases = new Map<number, string>()
  const programs: Record<string, ProgramJSON> = {}
  const instances: Record<string, { program: string; inputs: Record<string, unknown> }> = {}

  // Map from ModuleDef reference → unique program name
  const defToName = new Map<ModuleDef, string>()
  let anonCounter = 0

  // First pass: assign unique program names by reference
  for (const nc of def.nestedCalls) {
    if (!defToName.has(nc.moduleDef)) {
      let name = nc.moduleDef.typeName
      // If this name is already taken by a DIFFERENT def, make it unique
      if (name === '_pure' || [...defToName.values()].includes(name)) {
        name = `_sub_${anonCounter++}`
      }
      defToName.set(nc.moduleDef, name)
    }
  }

  // Second pass: count instances per program for alias naming
  const programInstanceCounts = new Map<string, number>()
  for (const nc of def.nestedCalls) {
    const progName = defToName.get(nc.moduleDef)!
    programInstanceCounts.set(progName, (programInstanceCounts.get(progName) ?? 0) + 1)
  }

  // Third pass: assign aliases
  const instanceCounters = new Map<string, number>()
  for (let i = 0; i < def.nestedCalls.length; i++) {
    const nc = def.nestedCalls[i]
    const progName = defToName.get(nc.moduleDef)!
    const totalInstances = programInstanceCounts.get(progName) ?? 1
    const current = instanceCounters.get(progName) ?? 0
    instanceCounters.set(progName, current + 1)

    const alias = totalInstances > 1 ? `${progName}_${current}` : progName

    nestedAliases.set(i, alias)
  }

  // Build delay names
  const delayNames: string[] = []
  for (let i = 0; i < def.delayInitValues.length; i++) {
    delayNames.push(`delay_${i}`)
  }

  // Now convert nested call programs and build instances
  const addedPrograms = new Set<ModuleDef>()
  for (let i = 0; i < def.nestedCalls.length; i++) {
    const nc = def.nestedCalls[i]
    const nestedDef = nc.moduleDef
    const progName = defToName.get(nestedDef)!
    const alias = nestedAliases.get(i)!

    // Only add the program definition once per unique ModuleDef
    if (!addedPrograms.has(nestedDef)) {
      addedPrograms.add(nestedDef)
      const subProg = defToProgram(nestedDef, new Set(visitedDefs))
      subProg.name = progName  // Override the name to match our unique name
      programs[progName] = subProg
    }

    // Build instance inputs
    const inputs: Record<string, unknown> = {}
    for (let j = 0; j < nc.callArgNodes.length; j++) {
      const inputName = nestedDef.inputNames[j]
      inputs[inputName] = convertNode(nc.callArgNodes[j], def, nestedAliases, delayNames)
    }

    instances[alias] = { program: progName, inputs }
  }

  // Build inputs spec (with type annotations if present)
  const inputs: (string | { name: string; type: string })[] = def.inputNames.map((name, i) => {
    const type = def.inputPortTypes[i]
    return type ? { name, type } : name
  })

  // Build outputs spec
  const outputs: (string | { name: string; type: string })[] = def.outputNames.map((name, i) => {
    const type = def.outputPortTypes[i]
    return type ? { name, type } : name
  })

  // Build regs
  const regs: Record<string, unknown> = {}
  for (let i = 0; i < def.registerNames.length; i++) {
    const name = def.registerNames[i]
    const init = def.registerInitValues[i]
    const type = def.registerPortTypes[i]
    if (type) {
      regs[name] = { init, type }
    } else {
      regs[name] = init
    }
  }

  // Build delays
  const delays: Record<string, { update: unknown; init: number }> | undefined =
    def.delayInitValues.length > 0
      ? Object.fromEntries(delayNames.map((name, i) => [
          name,
          {
            update: convertNode(def.delayUpdateNodes[i], def, nestedAliases, delayNames),
            init: def.delayInitValues[i],
          },
        ]))
      : undefined

  // Build input_defaults
  const input_defaults: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(def.rawInputDefaults)) {
    input_defaults[k] = v
  }

  // Build process
  const processOutputs: Record<string, unknown> = {}
  for (let i = 0; i < def.outputNames.length; i++) {
    processOutputs[def.outputNames[i]] = convertNode(
      def.outputExprNodes[i], def, nestedAliases, delayNames,
    )
  }

  const nextRegs: Record<string, unknown> = {}
  for (let i = 0; i < def.registerNames.length; i++) {
    const expr = def.registerExprNodes[i]
    if (expr !== null) {
      nextRegs[def.registerNames[i]] = convertNode(expr, def, nestedAliases, delayNames)
    }
  }

  const prog: ProgramJSON = {
    schema: 'tropical_program_1',
    name: def.typeName,
  }

  if (inputs.length > 0) prog.inputs = inputs
  if (outputs.length > 0) prog.outputs = outputs
  if (Object.keys(regs).length > 0) prog.regs = regs
  if (delays) prog.delays = delays
  if (Object.keys(input_defaults).length > 0) prog.input_defaults = input_defaults
  if (Object.keys(programs).length > 0) prog.programs = programs
  if (Object.keys(instances).length > 0) prog.instances = instances

  prog.process = {
    outputs: processOutputs,
    ...(Object.keys(nextRegs).length > 0 ? { next_regs: nextRegs } : {}),
  }

  return prog
}

// ── Generate all stdlib modules ──

const modules: [string, () => { _def: ModuleDef }][] = [
  ['VCA', () => vca()],
  ['Clock', () => clock()],
  ['BitCrusher', () => bitCrusher()],
  ['NoiseLFSR', () => noiseLFSR()],
  ['Delay8', () => delayLine(8, 'Delay8')],
  ['VCO', () => vco()],
  ['Compressor', () => compressor()],
  ['BassDrum', () => bassDrum()],
  ['ADEnvelope', () => adEnvelope()],
  ['ADSREnvelope', () => adsrEnvelope()],
  ['LadderFilter', () => ladderFilter()],
  ['Phaser', () => phaser()],
  ['Phaser16', () => phaser16()],
  ['Reverb', () => reverb()],
  ['Delay16', () => delayLine(16, 'Delay16')],
  ['Delay512', () => delayLine(512, 'Delay512')],
  ['Delay4410', () => delayLine(4410, 'Delay4410')],
  ['Delay44100', () => delayLine(44100, 'Delay44100')],
  ['TopoWaveguide', () => topoWaveguide()],
]

let generated = 0
for (const [name, factory] of modules) {
  const type = factory()
  const prog = defToProgram(type._def)
  const outPath = join(STDLIB_DIR, `${name}.json`)
  writeFileSync(outPath, JSON.stringify(prog, null, 2) + '\n')
  console.log(`  ${name} → ${outPath}`)
  generated++
}

console.log(`\nGenerated ${generated} stdlib JSON files.`)
