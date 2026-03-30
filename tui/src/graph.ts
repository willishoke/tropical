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

  // ---- Connections ----

  connect(srcModule: string, srcOutputId: number, dstModule: string, dstInputId: number): boolean {
    return b.egress_graph_connect(this._h, srcModule, srcOutputId, dstModule, dstInputId) as boolean
  }

  disconnect(srcModule: string, srcOutputId: number, dstModule: string, dstInputId: number): boolean {
    return b.egress_graph_disconnect(this._h, srcModule, srcOutputId, dstModule, dstInputId) as boolean
  }

  // ---- Input expressions ----

  setInputExpr(moduleName: string, inputId: number, expr: SignalExpr | null): boolean {
    const handle = expr !== null ? expr._h : null
    return b.egress_graph_set_input_expr(this._h, moduleName, inputId, handle) as boolean
  }

  beginUpdate(): void {
    b.egress_graph_begin_update(this._h)
  }

  endUpdate(): boolean {
    return b.egress_graph_end_update(this._h) as boolean
  }

  getInputExpr(moduleName: string, inputId: number): SignalExpr | null {
    const h = b.egress_graph_get_input_expr(this._h, moduleName, inputId)
    return h ? SignalExpr.fromHandle(h) : null
  }

  // ---- Outputs ----

  addOutput(moduleName: string, outputId: number): boolean {
    return b.egress_graph_add_output(this._h, moduleName, outputId) as boolean
  }

  addOutputExpr(expr: SignalExpr): boolean {
    return b.egress_graph_add_output_expr(this._h, expr._h) as boolean
  }

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

  // ---- Name generation (used by module builder) ----

  nextName(prefix: string): string {
    const count = (this._nameCounters.get(prefix) ?? 0) + 1
    this._nameCounters.set(prefix, count)
    return `${prefix}${count}`
  }
}
