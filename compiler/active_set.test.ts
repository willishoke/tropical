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

  test('active-set scaling: kernel CPU cost is monotonic and roughly linear in alive count', () => {
    // 8 SinOsc voices, all routed to dac. Vary how many are alive
    // (N ∈ {0, 1, 2, 4, 8}) via per-voice `aliveInput`. Measure
    // per-buffer wall time at each point. Verify three properties
    // simultaneously:
    //
    //   1. **Monotonicity** — cost is non-decreasing in N. If alive
    //      false is somehow MORE expensive than alive true (a
    //      regression where the conditional adds overhead instead
    //      of skipping), this fails.
    //   2. **Sleep skip works** — cost(0) is ≤ ~30% of cost(8). The
    //      8-asleep case should run essentially no DSP, just the
    //      slot reads and DAC stitch.
    //   3. **Roughly linear scaling** — for each non-zero N, the
    //      measured cost is within a 2× tolerance of the linear
    //      projection `(cost(8) - cost(0)) * N / 8 + cost(0)`. The
    //      tolerance is loose because the host clock is noisy and
    //      LLVM may apply different optimization to different N
    //      (e.g. unswitching at N=8 but not smaller), but a
    //      regression that runs every kernel regardless of alive
    //      would blow this out completely (cost(N) ≈ cost(8) for
    //      all N, which would fail #1 between N=1 and N=0 if
    //      anything, but more importantly would fail this check).
    //
    // Multiple measurement runs per N to denoise. Skipped under CI
    // where timing is noisy by default; set TROPICAL_BENCH=1 to
    // opt in.
    if (!process.env.TROPICAL_BENCH) return

    const TOTAL_VOICES = 8
    const N_VALUES     = [0, 1, 2, 4, 8] as const
    const RUNS_PER_N   = 5
    const FRAMES       = 1024
    const WARMUP       = 16

    const buildPatch = (nAlive: number) => {
      const s = makeSession(256)
      loadStdlib(s)
      const sinOsc = s.typeRegistry.get('SinOsc')!
      for (let i = 0; i < TOTAL_VOICES; i++) {
        const inst = instantiate(sinOsc, `v${i}`)
        inst.aliveInput = i < nAlive
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
      for (let f = 0; f < WARMUP; f++) rt.process()
      // Take the minimum of multiple runs — that's the cleanest
      // signal when the host clock is contended.
      let best = Infinity
      for (let r = 0; r < RUNS_PER_N; r++) {
        const t0 = performance.now()
        for (let f = 0; f < FRAMES; f++) rt.process()
        const dt = performance.now() - t0
        if (dt < best) best = dt
      }
      rt.dispose()
      return best
    }

    const costs: Record<number, number> = {}
    for (const n of N_VALUES) costs[n] = measure(buildPatch(n))

    // eslint-disable-next-line no-console
    console.log(
      `[active-set scaling] cost(ms): ` +
      N_VALUES.map(n => `N=${n}→${costs[n]!.toFixed(2)}`).join('  '),
    )

    // 1. Monotonic non-decreasing in N. Allow a small downward
    //    slack for host-clock jitter — strict `>=` is fragile when
    //    a higher-N measurement happens to land slightly faster
    //    than its lower-N neighbor due to scheduling noise.
    const NOISE_SLACK = 0.85
    for (let i = 1; i < N_VALUES.length; i++) {
      const prev = costs[N_VALUES[i - 1]!]!
      const curr = costs[N_VALUES[i]!]!
      expect(curr).toBeGreaterThanOrEqual(prev * NOISE_SLACK)
    }

    // 2. Sleep skip works. N=0 should be much cheaper than N=8 —
    //    the entire DSP body is conditionally skipped.
    expect(costs[0]! / costs[8]!).toBeLessThan(0.35)

    // 3. Roughly linear scaling. For each measured N, compare to
    //    the linear projection and assert within ~2× tolerance.
    const c0 = costs[0]!, c8 = costs[8]!
    const slope = (c8 - c0) / 8
    for (const n of N_VALUES) {
      if (n === 0 || n === 8) continue
      const projection = c0 + slope * n
      const actual     = costs[n]!
      const ratio      = actual / projection
      // 0.5× lower bound catches regressions where small-N is much
      // SLOWER than projected (e.g. if dispatch overhead dominated).
      // 2.0× upper bound catches regressions where small-N is
      // suspiciously close to large-N (e.g. all kernels running
      // regardless of alive).
      expect(ratio).toBeGreaterThan(0.5)
      expect(ratio).toBeLessThan(2.0)
    }
  })
})
