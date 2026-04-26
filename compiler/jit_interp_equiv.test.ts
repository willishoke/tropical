/**
 * jit_interp_equiv.test.ts — Differential test: TS interpreter vs. LLVM JIT.
 *
 * Runs a small gateable-subgraph patch through both the pure-TS interpreter
 * (interpret.ts / interpretSamples) and the native LLVM JIT (apply_plan →
 * Runtime.process), then asserts the output buffers match sample-for-sample.
 *
 * This is the merge gate for the gateable-subgraph feature: if the JIT's
 * conditional-block optimization diverges from the interpreter's reference
 * source_tag semantics, this test will flag it.
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ExprNode } from './session'
import { loadStdlib as loadBuiltins, loadProgramAsType, type ProgramNode } from './program'
import { applySessionWiring } from './apply_plan'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'

// A trivial leaf program: one input, one state register, output = reg; reg updates
// to (reg + input). Exercises both output and register-update wrapping paths.
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

function setupGated(gateInput: ExprNode, bufferLength = 32) {
  const session = makeSession(bufferLength)
  loadBuiltins(session)
  loadProgramAsType(ACCUM, session)

  // Patch: one gated Accum. x drives the register; output is the register value,
  // wrapped in source_tag (zero on skip) / state update wrapped with on_skip=reg.
  loadJSON({
    schema: 'tropical_program_2',
    name: 'patch',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'a1', program: 'Accum',
        inputs: { x: 1.0 }, gateable: true, gate_input: gateInput },
    ]},
    audio_outputs: [{ instance: 'a1', output: 'out' }],
  }, session)

  return session
}

function runJit(session: ReturnType<typeof setupGated>, nFrames = 1): Float64Array {
  applySessionWiring(session)
  session.graph.primeJit()
  const acc: number[] = []
  for (let f = 0; f < nFrames; f++) {
    session.graph.process()
    for (const v of session.graph.outputBuffer) acc.push(v)
  }
  return Float64Array.from(acc)
}

function runInterp(session: ReturnType<typeof setupGated>, nSamples: number): Float64Array {
  // applySessionWiring already flattened + loaded the plan; we re-run
  // flattenExpressions here to get the pre-emission ExprNode trees that
  // the interpreter consumes.
  const flat = flattenExpressions(session)
  return interpretSamples(flat, nSamples)
}

describe('JIT ↔ interpreter equivalence for gateable subgraphs', () => {
  test('baseline: ungated Accum accumulates correctly in JIT', () => {
    // No source_tag at all — direct Accum instance. Sanity check the test
    // harness itself before asserting anything about gateable semantics.
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
    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }
    session.graph.dispose()
  })

  test('gate always true: JIT output matches interpreter (output + reg update)', () => {
    const session = setupGated(true, 16)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)

    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }

    session.graph.dispose()
  })

  test('gate always false: JIT output is exactly 0 and matches interpreter', () => {
    const session = setupGated(false, 16)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)

    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      // On skip, output should be zero AND the accumulator should hold (stay 0).
      expect(jit[i]).toBe(0)
      expect(interp[i]).toBe(0)
    }

    session.graph.dispose()
  })

  test('gate driven by param flips: JIT matches interpreter sample-for-sample', () => {
    // Use a sample-index-derived gate: (sample_index % 4) < 2. This exercises
    // a gate that changes every two samples, so half the time the group is
    // skipped and the accumulator must hold on those samples.
    const gate: ExprNode = {
      op: 'lt',
      args: [
        { op: 'mod', args: [{ op: 'sampleIndex' }, 4] },
        2,
      ],
    }
    const session = setupGated(gate, 32)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)

    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }

    session.graph.dispose()
  })

  // ─── Bob-recommended coverage: delay register, nested calls, gate dependencies ───

  // Like ACCUM but with a delay register alongside the named reg, so both
  // register-update paths (reg_decl and delay_decl) get wrapped with on_skip.
  const ACCUM_WITH_DELAY: ProgramNode = {
    op: 'program',
    name: 'AccumDelay',
    ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'acc', init: 0 },
        { op: 'delayDecl', name: 'prev_x', update: { op: 'input', name: 'x' }, init: 0 },
      ],
      assigns: [
        // Output is current register value.
        { op: 'outputAssign', name: 'out', expr: { op: 'reg', name: 'acc' } },
        // Accumulator adds the PREVIOUS sample's x (via the delay reg).
        { op: 'nextUpdate', target: { kind: 'reg', name: 'acc' },
          expr: { op: 'add', args: [{ op: 'reg', name: 'acc' }, { op: 'delayRef', id: 'prev_x' }] } },
      ],
    },
  }

  test('gateable instance with both reg and delay registers: JIT matches interpreter', () => {
    const session = makeSession(32)
    loadBuiltins(session)
    loadProgramAsType(ACCUM_WITH_DELAY, session)

    // Dynamic gate: (sample_index % 4) < 2 — alternates live / skip.
    const gate: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 4] }, 2],
    }
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'AccumDelay',
          inputs: { x: 1.0 }, gateable: true, gate_input: gate },
      ]},
      audio_outputs: [{ instance: 'a1', output: 'out' }],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }
    session.graph.dispose()
  })

  test('gateable LadderFilter (4 nested OnePoles): JIT matches interpreter', () => {
    // LadderFilter from stdlib has 4 internal OnePole instances — this
    // exercises the nested-call register wrapping path that simple Accum
    // doesn't hit.
    const session = makeSession(64)
    loadBuiltins(session)

    // Gate alternates every 8 samples so the filter's state is meaningfully
    // held through skip windows and resumes cleanly on live windows.
    const gate: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 16] }, 8],
    }
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'ladder', program: 'LadderFilter',
          inputs: { input: 0.5, freq: 1000, res: 0.5 },
          gateable: true, gate_input: gate },
      ]},
      audio_outputs: [{ instance: 'ladder', output: 'out' }],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 8)
    }
    session.graph.dispose()
  })

  test('gate chain (a2 gated on a1 output): JIT matches interpreter', () => {
    // Two gateable instances where the second's gate is derived from the
    // first's output. Verifies gate-dependency tracking in the topo graph
    // and correct hot-swap behaviour when one gate toggles the other.
    const session = makeSession(32)
    loadBuiltins(session)
    loadProgramAsType(ACCUM, session)

    // a1 gates on (sample_index % 6) < 3 — alternates live/skip in 3-sample runs.
    const a1Gate: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 6] }, 3],
    }
    // a2's gate depends on a1's output: live only when a1 has accumulated
    // past some threshold.
    const a2Gate: ExprNode = {
      op: 'gt',
      args: [{ op: 'ref', instance: 'a1', output: 'out' }, 1.5],
    }

    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'Accum',
          inputs: { x: 1.0 }, gateable: true, gate_input: a1Gate },
        { op: 'instanceDecl', name: 'a2', program: 'Accum',
          inputs: { x: 1.0 }, gateable: true, gate_input: a2Gate },
      ]},
      audio_outputs: [{ instance: 'a2', output: 'out' }],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }
    session.graph.dispose()
  })
})
