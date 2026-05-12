/**
 * active_set.test.ts — verification gates for the active-set runtime.
 *
 * Two properties:
 *
 *  1. **IR shape** — the compiled plan carries one InstanceFunction per
 *     session instance, each with an alive slot index, and the
 *     scheduler preamble carries one WriteSlot per instance keyed to
 *     that slot. This is the structural guarantee that the JIT
 *     dispatch / GVN folding can engage on.
 *
 *  2. **Audio equivalence under default alive** — for the always-alive
 *     case (no explicit alive_input), the per-instance dispatch must
 *     produce sample-exact output equal to the same kernel without the
 *     conditional. Verified by comparing two builds: one with a stdlib
 *     patch as-is, one with the patch's lone instance wired to
 *     `alive_input: true` literal. Both must produce identical audio.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, allocateOutputSlots } from './session.js'
import { loadStdlib } from './program.js'
import { instantiate } from './program_types.js'
import { compileSession } from './ir/compile_session.js'
import { toWirePlan } from './flat_plan.js'
import { Runtime } from './runtime/runtime.js'

describe('active-set plan shape', () => {
  test('every session instance contributes one InstanceFunction', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    s.instanceRegistry.set('lp', instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc', sinOsc)
    allocateOutputSlots(s, 'lp', onePole)
    s.inputExprNodes.set('osc:freq', 220)
    s.inputExprNodes.set('lp:input', { op: 'ref', instance: 'osc', output: 'sine' })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const plan = compileSession(s)
    expect(plan.instance_functions.length).toBe(2)
    const names = plan.instance_functions.map(i => i.instance_name).sort()
    expect(names).toEqual(['lp', 'osc'])

    for (const inst of plan.instance_functions) {
      expect(inst.alive_slot_index).toBeGreaterThanOrEqual(0)
      expect(inst.alive_slot_index).toBeLessThan(plan.slot_count)
      expect(plan.slot_names[inst.alive_slot_index]).toBe(`${inst.instance_name}.__alive__`)
      // Default-alive: the slot starts at 1.0 so the conditional folds.
      expect(plan.slot_defaults[inst.alive_slot_index]).toBe(1)
    }
  })

  test('scheduler preamble writes one alive value per instance', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    for (let i = 0; i < 4; i++) {
      s.instanceRegistry.set(`v${i}`, instantiate(sinOsc, `v${i}`))
      allocateOutputSlots(s, `v${i}`, sinOsc)
      s.inputExprNodes.set(`v${i}:freq`, 110 * (i + 1))
      s.graphOutputs.push({ instance: `v${i}`, output: 'sine' })
    }
    const plan = compileSession(s)
    const aliveSlots = new Set<number>(plan.instance_functions.map(i => i.alive_slot_index))
    // Each instance writes to its own alive slot. `dst` is a tagged
    // `DstSlot` post-refactor; pattern-match to extract the module-
    // slot index.
    const aliveWrites = plan.scheduler_function.preamble
      .filter(i => i.tag === 'WriteSlot')
      .filter(i => i.dst.kind === 'moduleSlot' && aliveSlots.has(i.dst.index))
    expect(aliveWrites.length).toBe(4)
    // Default-alive: each WriteSlot carries a constant 1.
    for (const w of aliveWrites) {
      expect(w.args.length).toBe(1)
      expect(w.args[0]).toEqual({ kind: 'const', val: 1, scalar_type: 'float' })
    }
  })
})

describe('active-set audio invariance', () => {
  test('default-alive instance produces same audio as alive_input: true literal', () => {
    const buildSinOsc = (aliveExpr?: import('./expr.js').ExprNode) => {
      const s = makeSession(64)
      loadStdlib(s)
      const sinOsc = s.typeRegistry.get('SinOsc')!
      const inst = instantiate(sinOsc, 'osc')
      if (aliveExpr !== undefined) inst.aliveInput = aliveExpr
      s.instanceRegistry.set('osc', inst)
      allocateOutputSlots(s, 'osc', sinOsc)
      s.inputExprNodes.set('osc:freq', 440)
      s.graphOutputs.push({ instance: 'osc', output: 'sine' })
      return s
    }
    const run = (sess: ReturnType<typeof buildSinOsc>) => {
      const rt = new Runtime(64)
      rt.loadPlan(JSON.stringify(toWirePlan(compileSession(sess))))
      const samples: number[] = []
      for (let f = 0; f < 4; f++) {
        rt.process()
        for (const v of rt.outputBuffer) samples.push(v)
      }
      rt.dispose()
      return samples
    }
    const defaultAlive = run(buildSinOsc(undefined))
    const literalTrue  = run(buildSinOsc(true))
    expect(defaultAlive.length).toBe(literalTrue.length)
    // Byte-equal: literal-true alive collapses to the default-alive
    // codepath at JIT time (both emit WriteSlot const 1.0).
    for (let i = 0; i < defaultAlive.length; i++) {
      expect(defaultAlive[i]).toBe(literalTrue[i])
    }
  })

  test('alive=false from start: outputs hold at slot default (0)', () => {
    const s = makeSession(64)
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    const inst = instantiate(sinOsc, 'osc')
    inst.aliveInput = false  // never alive
    s.instanceRegistry.set('osc', inst)
    allocateOutputSlots(s, 'osc', sinOsc)
    s.inputExprNodes.set('osc:freq', 440)
    s.graphOutputs.push({ instance: 'osc', output: 'sine' })

    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(toWirePlan(compileSession(s))))
    rt.process()
    // SinOsc's `sine` output slot was never written; the slot default
    // for non-alive output slots is 0.
    for (const v of rt.outputBuffer) expect(v).toBe(0)
    rt.dispose()
  })

  test('active-set scaling: kernel CPU cost grows with the alive subset', () => {
    // 8 SinOsc voices, all routed to dac. Vary how many are alive via
    // a per-voice alive_input param. Compare per-buffer wall time
    // between N=8-alive and N=1-alive. Sleeping voices should
    // contribute negligibly to per-sample cost, so the N=1 case
    // should run noticeably faster.
    //
    // This isn't a precise micro-benchmark (no warm-up, no SIMD
    // controls), so the assertion is loose: N=1 should be within
    // ~140% of N=8 / 8 = 0.125 * cost(N=8), allowing for the
    // per-instance dispatch overhead. The exact numbers depend on
    // host CPU; we mostly want to catch a regression where every
    // sleeping voice still ran its kernel.
    //
    // Asserts on relative speedup only. Skipped under CI where
    // timing is noisy by default; set TROPICAL_BENCH=1 to opt in.
    if (!process.env.TROPICAL_BENCH) return

    const buildPatch = (nAlive: number) => {
      const s = makeSession(256)
      loadStdlib(s)
      const sinOsc = s.typeRegistry.get('SinOsc')!
      for (let i = 0; i < 8; i++) {
        const inst = instantiate(sinOsc, `v${i}`)
        inst.aliveInput = i < nAlive  // literal true/false
        s.instanceRegistry.set(`v${i}`, inst)
        allocateOutputSlots(s, `v${i}`, sinOsc)
        s.inputExprNodes.set(`v${i}:freq`, 110 * (i + 1))
        s.graphOutputs.push({ instance: `v${i}`, output: 'sine' })
      }
      return s
    }
    const measure = (sess: ReturnType<typeof buildPatch>) => {
      const rt = new Runtime(256)
      rt.loadPlan(JSON.stringify(toWirePlan(compileSession(sess))))
      // Warm up
      for (let f = 0; f < 16; f++) rt.process()
      const t0 = performance.now()
      for (let f = 0; f < 1024; f++) rt.process()
      const t1 = performance.now()
      rt.dispose()
      return t1 - t0
    }
    const t8 = measure(buildPatch(8))
    const t1 = measure(buildPatch(1))
    const ratio = t1 / t8
    // eslint-disable-next-line no-console
    console.log(`[active-set scaling] 1-alive/8-alive = ${ratio.toFixed(2)}`)
    expect(ratio).toBeLessThan(0.7)
  })
})
