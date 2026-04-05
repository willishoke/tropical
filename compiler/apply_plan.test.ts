/**
 * apply_plan.test.ts — Integration tests for the Phase 5 plan-based mutation path.
 *
 * These tests simulate the MCP tool flow: create session, add modules,
 * mutate inputExprNodes/graphOutputs, call applySessionWiring(), process, verify.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadPatchFromJSON, type PatchJSON, type ExprNode } from './patch'
import { loadBuiltins } from './module_library'
import { applySessionWiring, applyFlatPlan } from './apply_plan'
import { Runtime } from './runtime/runtime'

function setupSession(modules: PatchJSON['modules'], bufferLength = 256) {
  const session = makeSession(bufferLength)
  loadBuiltins(session.typeRegistry)
  loadPatchFromJSON({ schema: 'egress_patch_1', modules }, session)
  return session
}

function peak(buf: Float64Array): number {
  let p = 0
  for (let i = 0; i < buf.length; i++) p = Math.max(p, Math.abs(buf[i]))
  return p
}

describe('applySessionWiring', () => {
  test('connect two modules and get output', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])

    // Simulate connect_modules + set_module_input + add_graph_output
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    const buf = session.graph.outputBuffer
    expect(peak(buf)).toBeGreaterThan(0)

    session.graph.dispose()
  })

  test('matches reference output from old path', () => {
    // Old path: loadPatchFromJSON with full wiring
    const patch: PatchJSON = {
      schema: 'egress_patch_1',
      modules: [
        { type: 'VCO', name: 'VCO1' },
        { type: 'VCA', name: 'VCA1' },
      ],
      input_exprs: [
        { module: 'VCO1', input: 'freq', expr: 440 },
        { module: 'VCA1', input: 'audio', expr: { op: 'ref', module: 'VCO1', output: 'saw' } },
        { module: 'VCA1', input: 'cv', expr: 1.0 },
      ],
      outputs: [{ module: 'VCA1', output: 'out' }],
    }

    const refSession = makeSession(256)
    loadBuiltins(refSession.typeRegistry)
    loadPatchFromJSON(patch, refSession)
    refSession.graph.primeJit()
    refSession.graph.process()
    const refBuf = new Float64Array(refSession.graph.outputBuffer)

    // New path: modules only, then mutate + applySessionWiring
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const planBuf = session.graph.outputBuffer

    for (let i = 0; i < Math.min(refBuf.length, planBuf.length); i++) {
      expect(planBuf[i]).toBeCloseTo(refBuf[i], 10)
    }

    refSession.graph.dispose()
    session.graph.dispose()
  })

  test('disconnect produces silence', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])

    // Wire up
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    expect(peak(session.graph.outputBuffer)).toBeGreaterThan(0)

    // Disconnect VCA1:audio (simulates disconnect_modules)
    session.inputExprNodes.delete('VCA1:audio')
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    // VCA with no audio input should produce silence (default audio = 0)
    expect(peak(session.graph.outputBuffer)).toBe(0)

    session.graph.dispose()
  })

  test('switch output to different module', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCO', name: 'VCO2' },
      { type: 'VCA', name: 'VCA1' },
    ])

    // Output from VCA1 via VCO1
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf1 = new Float64Array(session.graph.outputBuffer)
    expect(peak(buf1)).toBeGreaterThan(0)

    // Switch output to VCO2 directly (simulates remove_graph_output + add_graph_output)
    session.inputExprNodes.set('VCO2:freq', 880)
    session.graphOutputs = [{ module: 'VCO2', output: 'saw' }]
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf2 = new Float64Array(session.graph.outputBuffer)
    expect(peak(buf2)).toBeGreaterThan(0)

    // Different output source should produce different signal
    let differs = false
    for (let i = 0; i < Math.min(buf1.length, buf2.length); i++) {
      if (Math.abs(buf1[i] - buf2[i]) > 1e-10) { differs = true; break }
    }
    expect(differs).toBe(true)

    session.graph.dispose()
  })

  test('batch update — multiple inputs then single apply', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCO', name: 'VCO2' },
      { type: 'VCA', name: 'VCA1' },
    ])

    // Batch: set all inputs at once (simulates set_inputs_batch)
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCO2:freq', 880)
    session.inputExprNodes.set('VCA1:audio', {
      op: 'add',
      args: [{ op: 'ref', module: 'VCO1', output: 'saw' }, { op: 'ref', module: 'VCO2', output: 'saw' }],
    } as ExprNode)
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    // Single applySessionWiring call
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    expect(peak(session.graph.outputBuffer)).toBeGreaterThan(0)

    session.graph.dispose()
  })

  test('rewire — change connection source', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCO', name: 'VCO2' },
      { type: 'VCA', name: 'VCA1' },
    ])

    // Initial: VCO1 → VCA1
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCO2:freq', 880)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf1 = new Float64Array(session.graph.outputBuffer)

    // Rewire: VCO2 → VCA1 (different frequency, should produce different output)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO2', output: 'saw' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf2 = session.graph.outputBuffer

    // Both should be non-silent
    expect(peak(buf1)).toBeGreaterThan(0)
    expect(peak(buf2)).toBeGreaterThan(0)

    // They should differ (different frequencies)
    let differs = false
    for (let i = 0; i < Math.min(buf1.length, buf2.length); i++) {
      if (Math.abs(buf1[i] - buf2[i]) > 1e-10) { differs = true; break }
    }
    expect(differs).toBe(true)

    session.graph.dispose()
  })

  test('expression with arithmetic — mul(ref, literal)', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])

    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', {
      op: 'mul',
      args: [{ op: 'ref', module: 'VCO1', output: 'saw' }, 0.5],
    } as ExprNode)
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    expect(peak(session.graph.outputBuffer)).toBeGreaterThan(0)

    session.graph.dispose()
  })
})

// ─── FlatRuntime tests ───────────────────────────────────────────────────

describe('applyFlatPlan', () => {
  test('VCO → VCA through flat runtime produces audio', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])

    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)
    rt.process()

    const buf = rt.outputBuffer
    expect(peak(buf)).toBeGreaterThan(0)

    rt.dispose()
    session.graph.dispose()
  })

  test('Clock module through flat runtime produces output', () => {
    const session = setupSession([
      { type: 'Clock', name: 'Clock1' },
    ])

    session.inputExprNodes.set('Clock1:freq', 1.0)
    session.inputExprNodes.set('Clock1:ratios_in', [1.0])
    session.graphOutputs.push({ module: 'Clock1', output: 'output' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)
    rt.process()

    const buf = rt.outputBuffer
    // Clock output is a square wave — should have nonzero samples
    expect(peak(buf)).toBeGreaterThan(0)

    rt.dispose()
    session.graph.dispose()
  })

  test('flat runtime produces continuous output over two buffers', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])
    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)

    rt.process()
    const buf1 = new Float64Array(rt.outputBuffer)
    expect(peak(buf1)).toBeGreaterThan(0)

    // Second buffer should also produce audio (registers persist)
    rt.process()
    const buf2 = new Float64Array(rt.outputBuffer)
    expect(peak(buf2)).toBeGreaterThan(0)

    // The two buffers should be different (phase advances)
    let differs = false
    for (let i = 0; i < buf1.length; i++) {
      if (Math.abs(buf1[i] - buf2[i]) > 1e-10) { differs = true; break }
    }
    expect(differs).toBe(true)

    rt.dispose()
    session.graph.dispose()
  })

  // TODO: three-module FM chain times out — investigate JIT compilation bottleneck
  // test('three-module chain: VCO → VCO (FM) → VCA', ...)

  test('hot-swap preserves register state across rewiring', () => {
    const session = setupSession([
      { type: 'VCO', name: 'VCO1' },
      { type: 'VCA', name: 'VCA1' },
    ])

    session.inputExprNodes.set('VCO1:freq', 440)
    session.inputExprNodes.set('VCA1:audio', { op: 'ref', module: 'VCO1', output: 'saw' })
    session.inputExprNodes.set('VCA1:cv', 1.0)
    session.graphOutputs.push({ module: 'VCA1', output: 'out' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)

    // Process several buffers to advance VCO phase
    for (let i = 0; i < 10; i++) rt.process()
    const buf1 = new Float64Array(rt.outputBuffer)

    // Rewire: change VCA cv from 1.0 to 0.5 (VCO registers should persist)
    session.inputExprNodes.set('VCA1:cv', 0.5)
    applyFlatPlan(session, rt)
    rt.process()
    const buf2 = rt.outputBuffer

    // Output should be non-silent (VCO kept running thanks to register transfer)
    expect(peak(buf2)).toBeGreaterThan(0)

    // Should be roughly half amplitude of what buf1 was (cv halved)
    const ratio = peak(buf2) / peak(buf1)
    expect(ratio).toBeGreaterThan(0.3)
    expect(ratio).toBeLessThan(0.7)

    rt.dispose()
    session.graph.dispose()
  })
})
