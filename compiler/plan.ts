/**
 * plan.ts — Execution plan generation.
 *
 * Flattens a CompiledPatch into a linear execution schedule (ExecutionPlan)
 * that can be serialized to JSON and loaded by the C++ runtime.
 *
 * The execution plan is the universal frontend interface — the "LLVM IR"
 * of tropical. Any frontend (TS DSL, visual patcher, LLM agent) can emit
 * plans; the C++ runtime loads and JIT-compiles them.
 *
 * Type information is erased at this stage — only layouts, kernel IDs,
 * and wiring expressions survive.
 */

import type { ExprNode } from './patch'
import type { CompiledPatch, ModuleInfo } from './compiler'

// ─────────────────────────────────────────────────────────────
// Plan schema
// ─────────────────────────────────────────────────────────────

export interface ExecutionPlan {
  schema: 'tropical_plan_1'
  config: PlanConfig
  /** Kernels in topological execution order. */
  kernels: KernelSpec[]
  /** How each kernel input is computed. */
  wiring: WiringEntry[]
  /** Which kernel outputs are mixed to the audio output buffer. */
  outputs: OutputEntry[]
}

export interface PlanConfig {
  sample_rate: number
  buffer_length: number
}

export interface KernelSpec {
  /** Unique kernel ID (sequential, stable within a plan). */
  id: number
  /** Instance name (e.g., "VCO1"). */
  name: string
  /** Module type name (e.g., "VCO") — used to look up the JIT kernel. */
  module_type: string
  /** Parallel execution group (topological level). Kernels in the same group can run concurrently. */
  group: number
  /** Port names for inputs, outputs, registers. */
  inputs: string[]
  outputs: string[]
  registers: string[]
  /** Initial values for each register (state slot). */
  state_init: StateValue[]
}

/** A state initial value: scalar, boolean, flat array, or matrix. */
export type StateValue = number | boolean | number[] | number[][]

export interface WiringEntry {
  /** Target kernel ID. */
  kernel: number
  /** Target input index. */
  input: number
  /** Input name (for readability). */
  input_name: string
  /** Expression computing this input value. Uses 'ref' ops with module names. */
  expr: ExprNode
}

export interface OutputEntry {
  /** Source kernel ID. */
  kernel: number
  /** Source output index. */
  output: number
  /** Output name (for readability). */
  output_name: string
}

// ─────────────────────────────────────────────────────────────
// Plan generation
// ─────────────────────────────────────────────────────────────

const DEFAULT_CONFIG: PlanConfig = {
  sample_rate: 44100,
  buffer_length: 512,
}

/**
 * Generate an execution plan from a compiled patch.
 *
 * The plan is a flat, JSON-serializable schedule:
 * - Kernels in topological order, grouped by execution level
 * - Wiring entries mapping each kernel input to its source expression
 * - Output entries specifying which kernel outputs are mixed to audio
 */
export function generatePlan(
  patch: CompiledPatch,
  config?: Partial<PlanConfig>,
): ExecutionPlan {
  const cfg: PlanConfig = { ...DEFAULT_CONFIG, ...config }

  // Assign kernel IDs in topological order
  const kernelIds = new Map<string, number>()
  let nextId = 0
  for (const level of patch.levels) {
    for (const name of level) {
      kernelIds.set(name, nextId++)
    }
  }

  // Build kernel specs
  const kernels: KernelSpec[] = []
  for (let group = 0; group < patch.levels.length; group++) {
    for (const name of patch.levels[group]) {
      const info = patch.modules.get(name)!
      kernels.push({
        id: kernelIds.get(name)!,
        name: info.name,
        module_type: info.typeName,
        group,
        inputs: [...info.inputNames],
        outputs: [...info.outputNames],
        registers: [...info.registerNames],
        state_init: info.registerNames.map(() => 0),
      })
    }
  }

  // Build wiring entries
  const wiring: WiringEntry[] = []
  for (const kernel of kernels) {
    for (let i = 0; i < kernel.inputs.length; i++) {
      const key = `${kernel.name}:${kernel.inputs[i]}`
      const expr = patch.inputExprNodes.get(key)
      if (expr !== undefined) {
        wiring.push({
          kernel: kernel.id,
          input: i,
          input_name: kernel.inputs[i],
          expr,
        })
      }
    }
  }

  // Build output entries
  const outputs: OutputEntry[] = []
  for (const { module, output } of patch.graphOutputs) {
    const kernelId = kernelIds.get(module)
    if (kernelId === undefined) continue
    const info = patch.modules.get(module)!
    const outputIdx = info.outputNames.indexOf(output)
    if (outputIdx === -1) continue
    outputs.push({
      kernel: kernelId,
      output: outputIdx,
      output_name: output,
    })
  }

  return {
    schema: 'tropical_plan_1',
    config: cfg,
    kernels,
    wiring,
    outputs,
  }
}

// ─────────────────────────────────────────────────────────────
// Plan serialization
// ─────────────────────────────────────────────────────────────

/** Serialize an execution plan to JSON. */
export function planToJSON(plan: ExecutionPlan): string {
  return JSON.stringify(plan, null, 2)
}

// ─────────────────────────────────────────────────────────────
// Plan validation
// ─────────────────────────────────────────────────────────────

export class PlanValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'PlanValidationError'
  }
}

/**
 * Validate structural integrity of a plan.
 * Checks: kernel ID uniqueness, wiring targets exist, outputs exist,
 * group ordering is monotonic.
 */
export function validatePlan(plan: ExecutionPlan): void {
  const kernelMap = new Map<number, KernelSpec>()
  const nameMap = new Map<string, number>()

  // Check kernel IDs are unique and sequential
  for (let i = 0; i < plan.kernels.length; i++) {
    const k = plan.kernels[i]
    if (k.id !== i) {
      throw new PlanValidationError(
        `Kernel IDs must be sequential: expected ${i}, got ${k.id}`
      )
    }
    if (nameMap.has(k.name)) {
      throw new PlanValidationError(`Duplicate kernel name: '${k.name}'`)
    }
    kernelMap.set(k.id, k)
    nameMap.set(k.name, k.id)
  }

  // Check group ordering is monotonically non-decreasing
  let lastGroup = -1
  for (const k of plan.kernels) {
    if (k.group < lastGroup) {
      throw new PlanValidationError(
        `Kernel '${k.name}' has group ${k.group} but follows group ${lastGroup}`
      )
    }
    lastGroup = k.group
  }

  // Check wiring targets exist
  for (const w of plan.wiring) {
    const k = kernelMap.get(w.kernel)
    if (!k) {
      throw new PlanValidationError(
        `Wiring targets unknown kernel ID ${w.kernel}`
      )
    }
    if (w.input < 0 || w.input >= k.inputs.length) {
      throw new PlanValidationError(
        `Wiring targets invalid input ${w.input} on kernel '${k.name}' (has ${k.inputs.length} inputs)`
      )
    }
  }

  // Check wiring ref expressions point to known kernels
  for (const w of plan.wiring) {
    const refs = collectRefs(w.expr)
    for (const ref of refs) {
      if (!nameMap.has(ref.module)) {
        throw new PlanValidationError(
          `Wiring expr for '${kernelMap.get(w.kernel)!.name}.${w.input_name}' ` +
          `references unknown module '${ref.module}'`
        )
      }
      // Check that the referenced module is in a prior or same group (no forward refs)
      const refKernelId = nameMap.get(ref.module)!
      const refKernel = kernelMap.get(refKernelId)!
      const targetKernel = kernelMap.get(w.kernel)!
      if (refKernel.group >= targetKernel.group && refKernel.id !== targetKernel.id) {
        // Same-group or later-group ref is a potential ordering issue
        // (but same-group is OK if it's a different module reading prev-sample outputs)
        // For now, only flag strictly forward refs
        if (refKernel.group > targetKernel.group) {
          throw new PlanValidationError(
            `Wiring for '${targetKernel.name}.${w.input_name}' references ` +
            `'${ref.module}' which executes in a later group (${refKernel.group} > ${targetKernel.group})`
          )
        }
      }
    }
  }

  // Check outputs reference valid kernels
  for (const o of plan.outputs) {
    const k = kernelMap.get(o.kernel)
    if (!k) {
      throw new PlanValidationError(
        `Output references unknown kernel ID ${o.kernel}`
      )
    }
    if (o.output < 0 || o.output >= k.outputs.length) {
      throw new PlanValidationError(
        `Output references invalid output ${o.output} on kernel '${k.name}' (has ${k.outputs.length} outputs)`
      )
    }
  }

  // Check state_init length matches registers
  for (const k of plan.kernels) {
    if (k.state_init.length !== k.registers.length) {
      throw new PlanValidationError(
        `Kernel '${k.name}': state_init length (${k.state_init.length}) ` +
        `doesn't match register count (${k.registers.length})`
      )
    }
  }
}

/** Extract all {module, output} refs from an ExprNode. */
function collectRefs(node: ExprNode): Array<{ module: string; output: string }> {
  const refs: Array<{ module: string; output: string }> = []
  walkRefs(node, refs)
  return refs
}

function walkRefs(node: ExprNode, refs: Array<{ module: string; output: string }>): void {
  if (typeof node === 'number' || typeof node === 'boolean') return
  if (Array.isArray(node)) {
    for (const item of node) walkRefs(item, refs)
    return
  }
  const n = node as { op: string; [k: string]: unknown }
  if (n.op === 'ref') {
    refs.push({ module: n.module as string, output: n.output as string })
    return
  }
  for (const arg of ((n.args as ExprNode[]) ?? [])) walkRefs(arg, refs)
  if (n.op === 'construct') {
    const fields = n.fields as Record<string, ExprNode> | undefined
    if (fields) for (const f of Object.values(fields)) walkRefs(f, refs)
  }
  if (n.op === 'project') walkRefs(n.expr as ExprNode, refs)
  if (n.op === 'inject') {
    const payload = n.payload as Record<string, ExprNode> | undefined
    if (payload) for (const p of Object.values(payload)) walkRefs(p, refs)
  }
  if (n.op === 'match') {
    walkRefs(n.scrutinee as ExprNode, refs)
    const branches = n.branches as Record<string, { body: ExprNode }> | undefined
    if (branches) for (const b of Object.values(branches)) walkRefs(b.body, refs)
  }
}

// ─────────────────────────────────────────────────────────────
// Plan statistics
// ─────────────────────────────────────────────────────────────

export interface PlanStats {
  kernel_count: number
  group_count: number
  wiring_count: number
  output_count: number
  total_inputs: number
  total_outputs: number
  total_registers: number
  /** Max parallelism (largest group size). */
  max_parallelism: number
}

export function planStats(plan: ExecutionPlan): PlanStats {
  const groupSizes = new Map<number, number>()
  let totalInputs = 0
  let totalOutputs = 0
  let totalRegisters = 0

  for (const k of plan.kernels) {
    groupSizes.set(k.group, (groupSizes.get(k.group) ?? 0) + 1)
    totalInputs += k.inputs.length
    totalOutputs += k.outputs.length
    totalRegisters += k.registers.length
  }

  return {
    kernel_count: plan.kernels.length,
    group_count: groupSizes.size,
    wiring_count: plan.wiring.length,
    output_count: plan.outputs.length,
    total_inputs: totalInputs,
    total_outputs: totalOutputs,
    total_registers: totalRegisters,
    max_parallelism: Math.max(0, ...groupSizes.values()),
  }
}
