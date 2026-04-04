/**
 * Graph — wraps egress_graph_t. Port of egress/graph.py.
 */

import * as b from './bindings.js'
import { SignalExpr } from './expr.js'

const _registry = new FinalizationRegistry((handle: unknown) => {
  b.egress_graph_free(handle)
})

export class Graph {
  _h: unknown
  private _nameCounters: Map<string, number> = new Map()

  constructor(bufferLength = 512) {
    this._h = b.check(b.egress_graph_new(bufferLength), 'graph_new')
    _registry.register(this, this._h, this)
  }

  dispose(): void {
    _registry.unregister(this)
    b.egress_graph_free(this._h)
    this._h = null
  }

  // ---- Type definitions ----

  /** Register a struct type in the graph's type registry. fields is an array of {name, scalar_type} where scalar_type: 0=float, 1=int, 2=bool. */
  defineStruct(name: string, fields: Array<{ name: string; scalar_type: number }>): boolean {
    const fieldNames = fields.map(f => f.name)
    const fieldTypes = fields.map(f => f.scalar_type)
    const ok = b.egress_typedef_struct(this._h, name, fieldNames, fieldTypes, fields.length) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      if (cErr) throw new Error(`Failed to define struct type '${name}': ${cErr}`)
    }
    return ok
  }

  /** Register a sum type in the graph's type registry. */
  defineSumType(name: string, variants: Array<{ name: string; payload: Array<{ name: string; scalar_type: number }> }>): boolean {
    const variantNames = variants.map(v => v.name)
    const flatFieldNames: string[] = []
    const flatFieldTypes: number[] = []
    const fieldCounts: number[] = []
    for (const v of variants) {
      for (const f of v.payload) {
        flatFieldNames.push(f.name)
        flatFieldTypes.push(f.scalar_type)
      }
      fieldCounts.push(v.payload.length)
    }
    const ok = b.egress_typedef_sum(this._h, name, variantNames, flatFieldNames, flatFieldTypes, fieldCounts, variants.length) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      if (cErr) throw new Error(`Failed to define sum type '${name}': ${cErr}`)
    }
    return ok
  }

  // ---- Module management ----

  addModule(name: string, specHandle: unknown): boolean {
    const ok = b.egress_graph_add_module(this._h, name, specHandle) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      if (cErr) throw new Error(`Failed to add module '${name}' to graph: ${cErr}`)
    }
    return ok
  }

  removeModule(name: string): boolean {
    return b.egress_graph_remove_module(this._h, name) as boolean
  }

  declareInputType(moduleName: string, idx: number, typeName: string): void {
    const ok = b.egress_module_declare_input_type(this._h, moduleName, idx, typeName) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      throw new Error(`Failed to declare input type on '${moduleName}' port ${idx}: ${cErr}`)
    }
  }

  declareOutputType(moduleName: string, idx: number, typeName: string): void {
    const ok = b.egress_module_declare_output_type(this._h, moduleName, idx, typeName) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      throw new Error(`Failed to declare output type on '${moduleName}' port ${idx}: ${cErr}`)
    }
  }

  declareRegisterType(moduleName: string, idx: number, typeName: string): void {
    const ok = b.egress_module_declare_register_type(this._h, moduleName, idx, typeName) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      throw new Error(`Failed to declare register type on '${moduleName}' register ${idx}: ${cErr}`)
    }
  }

  // ---- Input expressions ----

  setInputExpr(moduleName: string, inputId: number, expr: SignalExpr | null): boolean {
    const handle = expr !== null ? expr._h : null
    return b.egress_graph_set_input_expr(this._h, moduleName, inputId, handle) as boolean
  }

  // ---- Outputs ----

  addOutputTap(moduleName: string, outputId: number): number {
    return b.egress_graph_add_output_tap(this._h, moduleName, outputId) as number
  }

  removeOutputTap(tapId: number): boolean {
    return b.egress_graph_remove_output_tap(this._h, tapId) as boolean
  }

  // ---- Processing ----

  process(): void {
    const prevErr = b.egress_last_error() as string | null
    b.egress_graph_process(this._h)
    const err = b.egress_last_error() as string | null
    if (err && err !== prevErr) throw new Error(`graph process failed: ${err}`)
  }

  primeJit(): void {
    b.egress_graph_prime_jit(this._h)
  }

  /** Clear all input expressions and output mix. */
  clearWiring(): void {
    b.egress_graph_clear_wiring(this._h)
  }

  /** Load wiring and outputs from a plan JSON string. Modules must already exist. */
  loadPlan(planJson: string): boolean {
    const ok = b.egress_graph_load_plan(this._h, planJson, planJson.length) as boolean
    if (!ok) {
      const cErr = b.egress_last_error()
      throw new Error(`loadPlan failed: ${cErr}`)
    }
    return ok
  }

  // ---- Buffers ----

  /** Copy the output buffer into a Float64Array. Valid until next process(). */
  get outputBuffer(): Float64Array {
    const ptr = b.egress_graph_output_buffer(this._h)
    return b.decodeDoubleBuffer(ptr, this.bufferLength)
  }

  outputTapBuffer(tapId: number): Float64Array {
    const outLen = [0]
    const ptr = b.egress_graph_tap_buffer(this._h, tapId, outLen)
    return b.decodeDoubleBuffer(ptr, outLen[0] as number)
  }

  // ---- Configuration ----

  get bufferLength(): number {
    return b.egress_graph_get_buffer_length(this._h) as number
  }

  get workerCount(): number {
    return b.egress_graph_get_worker_count(this._h) as number
  }

  set workerCount(n: number) {
    b.egress_graph_set_worker_count(this._h, n)
  }

  get fusionEnabled(): boolean {
    return b.egress_graph_get_fusion_enabled(this._h) as boolean
  }

  set fusionEnabled(v: boolean) {
    b.egress_graph_set_fusion_enabled(this._h, v)
  }

  // ---- Profiling ----

  profileStats(): unknown {
    const json = b.egress_graph_get_profile_stats_json(this._h) as string
    return JSON.parse(json)
  }

  resetProfileStats(): void {
    b.egress_graph_reset_profile_stats(this._h)
  }

  /** Timing entries accumulated since the last begin_update / loadPlan call.
   *  Each entry corresponds to one rebuild_and_publish_runtime_locked() call. */
  buildTimingEntries(): Array<{
    module_count: number
    input_programs_ms: number
    fused_jit_ms: number
    total_ms: number
  }> {
    const json = b.egress_graph_get_build_timing_json(this._h) as string
    return JSON.parse(json)
  }

  // ---- Name generation (used by module builder) ----

  nextName(prefix: string): string {
    const count = (this._nameCounters.get(prefix) ?? 0) + 1
    this._nameCounters.set(prefix, count)
    return `${prefix}${count}`
  }
}
