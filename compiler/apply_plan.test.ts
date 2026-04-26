/**
 * apply_plan.test.ts — Integration tests for the Phase 5 plan-based mutation path.
 *
 * These tests simulate the MCP tool flow: create session, add modules,
 * mutate inputExprNodes/graphOutputs, call applySessionWiring(), process, verify.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ExprNode } from './session'
import { loadStdlib as loadBuiltins, loadProgramAsType } from './program'
import type { ProgramNode, ProgramFile } from './program'
import { applySessionWiring, applyFlatPlan } from './apply_plan'
import { Runtime } from './runtime/runtime'

/** Minimal test oscillator — naive saw + sin, phase accumulator. */
const TEST_OSC: ProgramNode = {
  op: 'program',
  name: 'TestOsc',
  ports: {
    inputs: [{ name: 'freq', type: 'freq', default: 440 }],
    outputs: [
      { name: 'saw', type: 'signal' },
      { name: 'sin', type: 'signal' },
    ],
  },
  body: { op: 'block',
    decls: [
      { op: 'regDecl', name: 'phase', init: 0 },
      { op: 'instanceDecl', name: 'sin1', program: 'Sin', inputs: {
        x: { op: 'mul', args: [6.283185307179586, { op: 'reg', name: 'phase' }] },
      }},
    ],
    assigns: [
      { op: 'outputAssign', name: 'saw', expr: { op: 'sub', args: [{ op: 'mul', args: [2, { op: 'reg', name: 'phase' }] }, 1] } },
      { op: 'outputAssign', name: 'sin', expr: { op: 'nestedOut', ref: 'sin1', output: 'out' } },
      { op: 'nextUpdate', target: { kind: 'reg', name: 'phase' }, expr: { op: 'mod', args: [
        { op: 'add', args: [
          { op: 'reg', name: 'phase' },
          { op: 'div', args: [{ op: 'input', name: 'freq' }, { op: 'sampleRate' }] },
        ]},
        1,
      ]}},
    ],
  },
}

function setupSession(instances: Record<string, { program: string }>, bufferLength = 256) {
  const session = makeSession(bufferLength)
  loadBuiltins(session.typeRegistry)
  session.typeRegistry.set('TestOsc', loadProgramAsType(TEST_OSC, session))
  const decls = Object.entries(instances).map(([name, { program }]) => ({
    op: 'instanceDecl' as const, name, program,
  }))
  loadJSON({
    schema: 'tropical_program_2',
    name: 'test',
    body: { op: 'block', decls },
  } as ProgramFile, session)
  return session
}

function peak(buf: Float64Array): number {
  let p = 0
  for (let i = 0; i < buf.length; i++) p = Math.max(p, Math.abs(buf[i]))
  return p
}

describe('applySessionWiring', () => {
  test('connect two modules and get output', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    const buf = session.graph.outputBuffer
    expect(peak(buf)).toBeGreaterThan(0)

    session.graph.dispose()
  })

  test('matches reference output from loadJSON path', () => {
    const refSession = makeSession(256)
    loadBuiltins(refSession.typeRegistry)
    refSession.typeRegistry.set('TestOsc', loadProgramAsType(TEST_OSC, refSession))
    const prog: ProgramFile = {
      schema: 'tropical_program_2',
      name: 'ref',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'osc1', program: 'TestOsc', inputs: { freq: 440 } },
        { op: 'instanceDecl', name: 'amp1', program: 'VCA', inputs: {
          audio: { op: 'ref', instance: 'osc1', output: 'saw' } as ExprNode,
          cv: 1.0,
        }},
      ]},
      audio_outputs: [{ instance: 'amp1', output: 'out' }],
    }
    loadJSON(prog, refSession)
    refSession.graph.primeJit()
    refSession.graph.process()
    const refBuf = new Float64Array(refSession.graph.outputBuffer)

    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })
    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

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
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    expect(peak(session.graph.outputBuffer)).toBeGreaterThan(0)

    session.inputExprNodes.delete('amp1:audio')
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    expect(peak(session.graph.outputBuffer)).toBe(0)

    session.graph.dispose()
  })

  test('switch output to different module', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      osc2: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf1 = new Float64Array(session.graph.outputBuffer)
    expect(peak(buf1)).toBeGreaterThan(0)

    session.inputExprNodes.set('osc2:freq', 880)
    session.graphOutputs = [{ instance: 'osc2', output: 'saw' }]
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf2 = new Float64Array(session.graph.outputBuffer)
    expect(peak(buf2)).toBeGreaterThan(0)

    let differs = false
    for (let i = 0; i < Math.min(buf1.length, buf2.length); i++) {
      if (Math.abs(buf1[i] - buf2[i]) > 1e-10) { differs = true; break }
    }
    expect(differs).toBe(true)

    session.graph.dispose()
  })

  test('batch update — multiple inputs then single apply', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      osc2: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('osc2:freq', 880)
    session.inputExprNodes.set('amp1:audio', {
      op: 'add',
      args: [{ op: 'ref', instance: 'osc1', output: 'saw' }, { op: 'ref', instance: 'osc2', output: 'saw' }],
    } as ExprNode)
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    expect(peak(session.graph.outputBuffer)).toBeGreaterThan(0)

    session.graph.dispose()
  })

  test('rewire — change connection source', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      osc2: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('osc2:freq', 880)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf1 = new Float64Array(session.graph.outputBuffer)

    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc2', output: 'saw' })
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const buf2 = session.graph.outputBuffer

    expect(peak(buf1)).toBeGreaterThan(0)
    expect(peak(buf2)).toBeGreaterThan(0)

    let differs = false
    for (let i = 0; i < Math.min(buf1.length, buf2.length); i++) {
      if (Math.abs(buf1[i] - buf2[i]) > 1e-10) { differs = true; break }
    }
    expect(differs).toBe(true)

    session.graph.dispose()
  })

  test('expression with arithmetic — mul(ref, literal)', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', {
      op: 'mul',
      args: [{ op: 'ref', instance: 'osc1', output: 'saw' }, 0.5],
    } as ExprNode)
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()

    expect(peak(session.graph.outputBuffer)).toBeGreaterThan(0)

    session.graph.dispose()
  })
})

// ─── FlatRuntime tests ───────────────────────────────────────────────────

describe('applyFlatPlan', () => {
  test('TestOsc → VCA through flat runtime produces audio', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)
    rt.process()

    const buf = rt.outputBuffer
    expect(peak(buf)).toBeGreaterThan(0)

    rt.dispose()
    session.graph.dispose()
  })

  test('Clock module through flat runtime produces output', () => {
    const session = setupSession({
      Clock1: { program: 'Clock' },
    })

    session.inputExprNodes.set('Clock1:freq', 1.0)
    session.inputExprNodes.set('Clock1:ratios_in', [1.0])
    session.graphOutputs.push({ instance: 'Clock1', output: 'output' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)
    rt.process()

    const buf = rt.outputBuffer
    expect(peak(buf)).toBeGreaterThan(0)

    rt.dispose()
    session.graph.dispose()
  })

  test('flat runtime produces continuous output over two buffers', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })
    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)

    rt.process()
    const buf1 = new Float64Array(rt.outputBuffer)
    expect(peak(buf1)).toBeGreaterThan(0)

    rt.process()
    const buf2 = new Float64Array(rt.outputBuffer)
    expect(peak(buf2)).toBeGreaterThan(0)

    let differs = false
    for (let i = 0; i < buf1.length; i++) {
      if (Math.abs(buf1[i] - buf2[i]) > 1e-10) { differs = true; break }
    }
    expect(differs).toBe(true)

    rt.dispose()
    session.graph.dispose()
  })

  test('hot-swap preserves register state across rewiring', () => {
    const session = setupSession({
      osc1: { program: 'TestOsc' },
      amp1: { program: 'VCA' },
    })

    session.inputExprNodes.set('osc1:freq', 440)
    session.inputExprNodes.set('amp1:audio', { op: 'ref', instance: 'osc1', output: 'saw' })
    session.inputExprNodes.set('amp1:cv', 1.0)
    session.graphOutputs.push({ instance: 'amp1', output: 'out' })

    const rt = new Runtime(256)
    applyFlatPlan(session, rt)

    for (let i = 0; i < 10; i++) rt.process()
    const buf1 = new Float64Array(rt.outputBuffer)

    session.inputExprNodes.set('amp1:cv', 0.5)
    applyFlatPlan(session, rt)
    rt.process()
    const buf2 = rt.outputBuffer

    expect(peak(buf2)).toBeGreaterThan(0)

    const ratio = peak(buf2) / peak(buf1)
    expect(ratio).toBeGreaterThan(0.3)
    expect(ratio).toBeLessThan(0.7)

    rt.dispose()
    session.graph.dispose()
  })
})
