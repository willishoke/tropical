/**
 * jit_vs_interp.test.ts — Differential test: TS interpreter vs. LLVM JIT.
 *
 * Runs alive-conditional patches through both the pure-TS interpreter
 * (interpret_resolved.ts / interpretSession) and the native LLVM JIT
 * (apply_plan → Runtime.process), then asserts the output buffers
 * agree.
 *
 * The two backends realize alive semantics differently — the JIT
 * skips the instance's kernel when alive evaluates false (slot
 * retains its last write), the interpreter wraps outputs in
 * `select(alive, raw, 0)` and reg/delay updates in `select(alive,
 * next, current)` (legacy-gateable shape). For literal-true alive,
 * both reduce to the un-wrapped form and outputs match byte-for-byte.
 * For literal-false alive starting at the default slot value of 0,
 * both also produce 0. Transitional alive (alive flips during the
 * buffer) DIVERGES because the interpreter publishes 0 on asleep
 * samples and the JIT publishes the last live value. Tests that need
 * transitional behavior wrap their output explicitly:
 *
 *     audio_output = select(alive, computed, 0)
 *
 * which makes the JIT's "retain last write" semantic produce
 * select(alive, computed, slot[0]_default=0). With explicit user-side
 * wrapping, both backends agree.
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ExprNode } from '../../compiler/session'
import { loadStdlib as loadBuiltins, loadProgramAsType, type ProgramNode } from '../../compiler/program'
import { applySessionWiring } from '../../compiler/apply_plan'
import { interpretSession } from '../../compiler/interpret_resolved'

const ACCUM: ProgramNode = {
  op: 'program',
  name: 'Accum',
  ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['out'] },
  body: { op: 'block',
    decls: [{ op: 'regDecl', name: 'acc', init: 0 }],
    assigns: [
      { op: 'outputAssign', name: 'out', expr: { op: 'reg', name: 'acc' } },
      { op: 'nextUpdate', target: { kind: 'reg', name: 'acc' },
        expr: { op: 'add', args: [{ op: 'reg', name: 'acc' }, { op: 'input', name: 'x' }] } },
    ],
  },
}

function setupAlive(aliveInput: ExprNode, bufferLength = 32) {
  const session = makeSession(bufferLength)
  loadBuiltins(session)
  loadProgramAsType(ACCUM, session)
  loadJSON({
    schema: 'tropical_program_2',
    name: 'patch',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'a1', program: 'Accum',
        inputs: { x: 1.0 }, alive_input: aliveInput },
    ]},
    audio_outputs: [{ instance: 'a1', output: 'out' }],
  }, session)
  return session
}

function runJit(session: ReturnType<typeof setupAlive>, nFrames = 1): Float64Array {
  applySessionWiring(session)
  session.graph.primeJit()
  const acc: number[] = []
  for (let f = 0; f < nFrames; f++) {
    session.graph.process()
    for (const v of session.graph.outputBuffer) acc.push(v)
  }
  return Float64Array.from(acc)
}

function runInterp(session: ReturnType<typeof setupAlive>, nSamples: number): Float64Array {
  return interpretSession(session, nSamples)
}

describe('JIT ↔ interpreter equivalence for alive-conditional instances', () => {
  test('baseline: instance without alive_input matches interpreter', () => {
    const session = makeSession(16)
    loadBuiltins(session)
    loadProgramAsType(ACCUM, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'Accum', inputs: { x: 1.0 } },
      ]},
      audio_outputs: [{ instance: 'a1', output: 'out' }],
    }, session)
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const interp = interpretSession(session, jit.length)
    for (let i = 0; i < jit.length; i++) expect(jit[i]).toBeCloseTo(interp[i], 10)
    session.graph.dispose()
  })

  test('alive=true literal: JIT matches interpreter', () => {
    const session = setupAlive(true, 16)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)
    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) expect(jit[i]).toBeCloseTo(interp[i], 10)
    session.graph.dispose()
  })

  test('alive=false literal: both backends produce 0 (slot default)', () => {
    const session = setupAlive(false, 16)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)
    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBe(0)
      expect(interp[i]).toBe(0)
    }
    session.graph.dispose()
  })

  test('alive driven by sampleIndex: both agree when output explicitly zeroed on asleep', () => {
    // Explicit user-side wrap: the audio output is `select(alive,
    // raw_out, 0)`. With the JIT's retain-semantics, slot[0] holds the
    // computed select() value, which is 0 on asleep samples — matching
    // the interpreter's `select(alive, raw, 0)` output wrap.
    //
    // We model this by adding a Passthrough instance whose `x` is wired
    // to `select(alive, accum.out, 0)` (alive driven by sample index).
    const PT: ProgramNode = {
      op: 'program', name: 'Passthrough',
      ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
      body: { op: 'block',
        assigns: [{ op: 'outputAssign', name: 'y', expr: { op: 'input', name: 'x' } }],
      },
    }
    const session = makeSession(32)
    loadBuiltins(session)
    loadProgramAsType(ACCUM, session)
    loadProgramAsType(PT, session)

    const alive: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 4] }, 2],
    }
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'Accum',
          inputs: { x: 1.0 }, alive_input: alive },
        { op: 'instanceDecl', name: 'gate_out', program: 'Passthrough',
          inputs: { x: { op: 'select', args: [
            alive,
            { op: 'ref', instance: 'a1', output: 'out' },
            0,
          ]}},
        },
      ]},
      audio_outputs: [{ instance: 'gate_out', output: 'y' }],
    }, session)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)
    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) expect(jit[i]).toBeCloseTo(interp[i], 10)
    session.graph.dispose()
  })
})
