/**
 * microkernel_deep.test.ts — three-way equivalence: fused JIT,
 * shallow microkernel JIT, deep microkernel JIT.
 *
 * For each stdlib program in the equivalence corpus, compile and run
 * the same nested-IR session three times — once with each
 * `compilation_mode` — and assert sample-for-sample agreement across
 * all three runs.
 *
 * All three sessions are constructed with `inlineNested: false` so the
 * IR carries non-empty `children` arrays at every nesting level. This
 * gives deep mode something to dispatch over (otherwise it
 * degenerates to shallow microkernel mode for single-instance plans),
 * and keeps fused / shallow comparisons honest — they consume the
 * same nested plan as deep mode, just lower it to a single LLVM
 * function (fused) or one-function-per-top-level-instance with
 * inlined children (shallow).
 *
 * Depth-2 stdlib programs (Pow, OnePole, LadderFilter, Phaser,
 * Phaser16, AllpassDelay, CombDelay, Delay, SVF) are where the
 * deep-mode dispatch surface actually engages — they have nested
 * InstanceDecls that survive as kernel boundaries under
 * `inlineNested: false`.
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, inputNames, outputNames } from '../../compiler/session.js'
import type { ExprNode } from '../../compiler/expr.js'
import { loadStdlib } from '../../compiler/program.js'
import { applyFlatPlan } from '../../compiler/apply_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'
import type { CompilationMode } from '../../compiler/flat_plan.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

const BUFFER_LENGTH  = 256
const N_BUFFERS      = 4
const TOTAL_SAMPLES  = BUFFER_LENGTH * N_BUFFERS
const TOLERANCE      = 1e-12   // same-codegen-path tolerance

function pulseEvery(n: number): ExprNode {
  return { op: 'lt', args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, n] }, 1] }
}

const DEFAULT_INPUTS: Record<string, ExprNode> = {
  freq: 220, x: 0.5, y: 0.5, audio: 0.5, input: 0.5, cv: 0.5,
  cutoff: 1000, q: 0.5, drive: 1.0, mix: 0.5, a: 0.3, b: 0.7,
  coeff: 0.4, feedback: 0.4, lfo_speed: 0.2, decay: 0.99,
  rate: 5, g: 0.1, resonance: 0.5,
  trigger: pulseEvery(64), clock: pulseEvery(32),
}

// Corpus mirrors microkernel_vs_fused but emphasizes depth-2 programs
// where the deep-mode dispatch surface actually engages. Bubble /
// BubbleCloud were the Phase-4 nested-mode gap closed by PR #158;
// included here as regression coverage.
const STDLIB_TARGETS: Array<[string, Record<string, number>?]> = [
  ['SinOsc'], ['Sin'], ['Cos'], ['Tanh'], ['Exp'], ['Log'], ['Pow'],
  ['OnePole'], ['BlepSaw'], ['SoftClip'], ['VCA'], ['CrossFade'],
  ['SVF'], ['LadderFilter'], ['Phaser'], ['Phaser16'],
  ['Bubble'], ['BubbleCloud'],
  ['AllpassDelay'], ['CombDelay'],
  ['Delay', { N: 1024 }],
]

function setupInstance(
  typeName: string,
  typeArgs?: Record<string, number>,
) {
  // inlineNested:false so the post-strata IR carries the children we
  // want deep mode to dispatch over. Fused and shallow microkernel
  // both happily consume the same shape — they just emit different
  // LLVM code.
  const session = makeSession(BUFFER_LENGTH, { inlineNested: false })
  loadStdlib(session)
  const { type, typeArgs: resolved } = resolveProgramType(session, typeName, typeArgs, undefined)
  const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: resolved })
  session.instanceRegistry.set('inst', inst)
  for (const portName of inputNames(inst)) {
    if (portName in DEFAULT_INPUTS) {
      session.inputExprNodes.set(wk('inst', portName), DEFAULT_INPUTS[portName])
    }
  }
  session.graphOutputs.push({ instance: 'inst', output: outputNames(inst)[0] })
  return session
}

function captureOutput(
  session: ReturnType<typeof setupInstance>,
  mode: CompilationMode,
): Float64Array {
  applyFlatPlan(session, session.runtime, { compilation_mode: mode })
  session.graph.primeJit()
  const out = new Float64Array(TOTAL_SAMPLES)
  for (let f = 0; f < N_BUFFERS; f++) {
    session.runtime.process()
    const buf = session.runtime.outputBuffer
    out.set(buf.subarray(0, BUFFER_LENGTH), f * BUFFER_LENGTH)
  }
  return out
}

function maxAbsDiff(a: Float64Array, b: Float64Array): { maxAbsDiff: number, firstDiffIdx: number } {
  let maxAbsDiff = 0
  let firstDiffIdx = -1
  for (let i = 0; i < TOTAL_SAMPLES; i++) {
    const x = a[i], y = b[i]
    if (Number.isNaN(x) && Number.isNaN(y)) continue
    const d = Math.abs(x - y)
    if (d > maxAbsDiff) maxAbsDiff = d
    if (d > TOLERANCE && firstDiffIdx < 0) firstDiffIdx = i
  }
  return { maxAbsDiff, firstDiffIdx }
}

describe('microkernel-deep stdlib equivalence (three-way: fused / shallow / deep)', () => {
  for (const [typeName, typeArgs] of STDLIB_TARGETS) {
    test(`${typeName}${typeArgs ? `<${JSON.stringify(typeArgs)}>` : ''}`, () => {
      // Three independent sessions so each mode starts from identical
      // register/slot inits. Sharing a session across modes would
      // invoke hot-swap state transfer, which is a separate axis.
      const fusedOut    = captureOutput(setupInstance(typeName, typeArgs), 'fused')
      const shallowOut  = captureOutput(setupInstance(typeName, typeArgs), 'microkernel')
      const deepOut     = captureOutput(setupInstance(typeName, typeArgs), 'microkernel-deep')

      // Deep vs fused
      const dfDiff = maxAbsDiff(deepOut, fusedOut)
      if (dfDiff.maxAbsDiff > TOLERANCE) {
        const i = dfDiff.firstDiffIdx
        throw new Error(
          `${typeName}: deep/fused diverged at sample ${i} ` +
          `(deep=${deepOut[i]}, fused=${fusedOut[i]}, maxAbsDiff=${dfDiff.maxAbsDiff})`,
        )
      }
      expect(dfDiff.maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)

      // Deep vs shallow microkernel
      const dsDiff = maxAbsDiff(deepOut, shallowOut)
      if (dsDiff.maxAbsDiff > TOLERANCE) {
        const i = dsDiff.firstDiffIdx
        throw new Error(
          `${typeName}: deep/shallow-microkernel diverged at sample ${i} ` +
          `(deep=${deepOut[i]}, shallow=${shallowOut[i]}, maxAbsDiff=${dsDiff.maxAbsDiff})`,
        )
      }
      expect(dsDiff.maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
    })
  }
})

// Polyphony: 8 SinOsc voices in one session. Each voice is a top-level
// instance with a nested Sin child — deep mode dispatches 16 LLVM
// functions (8 SinOsc + 8 Sin) where shallow dispatches 8 and fused
// emits 1.
describe('microkernel-deep polyphony (8x SinOsc voices)', () => {
  test('all three modes agree', () => {
    function setupPolyphony() {
      const session = makeSession(BUFFER_LENGTH, { inlineNested: false })
      loadStdlib(session)
      const { type, typeArgs } = resolveProgramType(session, 'SinOsc', undefined, undefined)
      for (let i = 0; i < 8; i++) {
        const name = `osc${i}`
        const inst = instantiate(type, name, { baseTypeName: 'SinOsc', typeArgs })
        session.instanceRegistry.set(name, inst)
        session.inputExprNodes.set(wk(name, 'freq'), 110 + 22 * i)
        session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
      }
      return session
    }
    const fusedOut   = captureOutput(setupPolyphony(), 'fused')
    const shallowOut = captureOutput(setupPolyphony(), 'microkernel')
    const deepOut    = captureOutput(setupPolyphony(), 'microkernel-deep')

    expect(maxAbsDiff(deepOut, fusedOut).maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
    expect(maxAbsDiff(deepOut, shallowOut).maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
  })
})
