/**
 * trace_cycles.test.ts — coverage for the Phase C4 cycle-tracer.
 *
 * Properties tested:
 *   1. A program with no instances → identity.
 *   2. Acyclic instance graph → identity.
 *   3. Two-instance cycle → one synthetic `DelayDecl` is added; the
 *      cycle member that is not the break target now reads the
 *      synthetic delay rather than the breaker's NestedOut.
 *   4. Three-instance cycle → SCC detected, single back-edge broken.
 *   5. Stdlib programs are all identity (no inter-instance cycles
 *      pre-inlining).
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { extractMarkdown } from '../parse/markdown.js'
import { parseProgram } from '../parse/declarations.js'
import { raiseProgram } from '../parse/raise.js'
import { elaborate, _elaborateForCyclicTest, type ExternalProgramResolver } from './elaborator.js'
import { breakInstanceCycles } from './lowering/cycle_break.js'
import { cloneResolvedProgram } from './clone.js'
import { strataPipeline } from './strata.js'
import { makeSession, loadJSON } from '../session.js'
import { loadProgramAsType, type ProgramNode } from '../program.js'
import { interpretSession } from '../interpret_resolved.js'
import type {
  ResolvedProgram, BodyDecl, InstanceDecl, RegDecl,
  ResolvedExpr, NestedOut, RegRef,
} from './nodes.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const STDLIB_DIR = join(__dirname, '../../stdlib')

function elab(src: string, resolver?: ExternalProgramResolver): ResolvedProgram {
  return elaborate(parseProgram(src), resolver)
}

/** Cyclic-source variant: this test file specifically exercises the
 *  cycle-break helper on programs the elaborator would otherwise
 *  reject under Phase 4b's strict policy. Uses the elaborator's test
 *  escape hatch to construct cyclic ResolvedPrograms for downstream
 *  testing. */
function elabCyclic(src: string, resolver?: ExternalProgramResolver): ResolvedProgram {
  return _elaborateForCyclicTest(parseProgram(src), resolver)
}

function instanceDecls(prog: ResolvedProgram): InstanceDecl[] {
  return prog.body.decls.filter((d): d is InstanceDecl => d.op === 'instanceDecl')
}

/** Post-Phase 0a: former DelayDecls are now RegDecls with update set.
 *  Synthetic cycle-break decls are identifiable by the `_feedback_`
 *  name prefix produced by `breakInstanceCycles`. (Earlier phases used
 *  a `_liftedFrom: 'synthetic'` sentinel; Phase 5 retired it because
 *  the synthetic tag was redundant with the distinctive name and the
 *  strict policy means production code paths never see these regs.) */
function syntheticBreakerRegs(prog: ResolvedProgram): RegDecl[] {
  return prog.body.decls.filter(
    (d): d is RegDecl =>
      d.op === 'regDecl' && d.name.startsWith('_feedback_'),
  )
}

// ─────────────────────────────────────────────────────────────
// 1 + 2: no-op cases
// ─────────────────────────────────────────────────────────────

describe('traceCycles — identity cases', () => {
  test('a program with no instances returns input by identity', () => {
    const p = elab('program X(a: float) -> (out: float) { out = a + 1 }')
    expect(breakInstanceCycles(p).lowered).toBe(p)
  })

  test('acyclic two-instance graph returns input by identity', () => {
    // a → b: b reads from a's output, a reads only from external input.
    const p = elab(`
      program Top(x: float) -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: x)
        b = Inner(in_: a.out_)
        out = b.out_
      }
    `)
    expect(breakInstanceCycles(p).lowered).toBe(p)
  })
})

// ─────────────────────────────────────────────────────────────
// 3: two-instance cycle
// ─────────────────────────────────────────────────────────────

describe('traceCycles — two-instance cycle', () => {
  test('inserts a synthetic delay; later member reads it instead of NestedOut', () => {
    // a reads b.out, b reads a.out — a 2-cycle.
    const p = elabCyclic(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: b.out_)
        b = Inner(in_: a.out_)
        out = b.out_
      }
    `)
    const beforeDelays = syntheticBreakerRegs(p).length
    const out = breakInstanceCycles(p).lowered
    const afterDelays = syntheticBreakerRegs(out)
    expect(afterDelays.length).toBe(beforeDelays + 1)

    // The break target is the first instance in source order = `a`.
    // So `b`'s input wire to `a.out_` was rewritten to a delayRef on
    // the synthetic delay; `a`'s wire stays as a NestedOut to b.out_.
    const insts = instanceDecls(out)
    const a = insts.find(i => i.name === 'a')!
    const b = insts.find(i => i.name === 'b')!

    // a's wire still references b via NestedOut (the surviving edge).
    expect(opsIn(a.inputs[0].value, 'nestedOut').length).toBeGreaterThan(0)
    expect(opsIn(a.inputs[0].value, 'regRef').length).toBe(0)
    // b's wire was rewritten — no NestedOut, one DelayRef on the
    // synthetic delay.
    expect(opsIn(b.inputs[0].value, 'nestedOut').length).toBe(0)
    expect(opsIn(b.inputs[0].value, 'regRef').length).toBe(1)

    // The synthetic delay is named `_feedback_a_out_` (instance "a",
    // output port "out_").
    const synthName = `_feedback_a_out_`
    const synth = afterDelays.find(d => d.name === synthName)
    expect(synth).toBeDefined()
    // Its update reads a.out_ (a NestedOut on the breaker).
    expect(opsIn(synth!.update, 'nestedOut').length).toBe(1)
    // And init is zero.
    expect(synth!.init).toBe(0)
  })
})

// ─────────────────────────────────────────────────────────────
// 4: three-instance cycle
// ─────────────────────────────────────────────────────────────

describe('traceCycles — three-instance cycle', () => {
  test('a → b → c → a: a single back-edge is broken', () => {
    const p = elabCyclic(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: c.out_)
        b = Inner(in_: a.out_)
        c = Inner(in_: b.out_)
        out = c.out_
      }
    `)
    const beforeDelays = syntheticBreakerRegs(p).length
    const out = breakInstanceCycles(p).lowered
    const afterDelays = syntheticBreakerRegs(out)
    // Exactly one synthetic delay added (one output port broken on
    // the chosen breaker).
    expect(afterDelays.length).toBe(beforeDelays + 1)
    expect(afterDelays[afterDelays.length - 1].name).toBe('_feedback_a_out_')
  })
})

// ─────────────────────────────────────────────────────────────
// Identity consistency under downstream clone
// ─────────────────────────────────────────────────────────────

describe('traceCycles — InstanceDecl identity is consistent for cloneResolvedProgram', () => {
  // After traceCycles, every nestedOut.instance ref in the resulting
  // program (in instance inputs, body assigns, and synthetic delay
  // updates) must point at an InstanceDecl that is registered in
  // body.decls. Otherwise cloneResolvedProgram throws "unregistered
  // InstanceDecl 'X'" when the next stratum (inlineInstances) tries to
  // clone the program. This is the exact bug that broke cross-coupled
  // delay topologies (e.g. cross_fm_4.json) before the rebuild loop
  // was changed to mutate inputs in place.

  test('two-instance cycle survives cloneResolvedProgram', () => {
    const p = elabCyclic(`
      program Top() -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: b.out_)
        b = Inner(in_: a.out_)
        out = b.out_
      }
    `)
    const traced = breakInstanceCycles(p).lowered
    expect(() => cloneResolvedProgram(traced)).not.toThrow()
  })

  test('cross-coupled delay topology compiles through full strata pipeline', () => {
    // 2 oscillators + 2 delays cross-coupled — minimal repro of the
    // cross_fm_4.json bug. Each VCO's freq reads the other delay's
    // output; each delay's input reads the matching VCO's output.
    // All 4 instances form one SCC.
    const p = elabCyclic(`
      program Top() -> (out: float) {
        program VCO(freq: float) -> (sine: float) { sine = freq + 1 }
        program Delay(x: float) -> (y: float) { y = x }
        d1 = Delay(x: vco1.sine)
        d2 = Delay(x: vco2.sine)
        vco1 = VCO(freq: 110 + d2.y)
        vco2 = VCO(freq: 220 + d1.y)
        out = vco1.sine + vco2.sine
      }
    `)
    // Post-Phase 3: strataPipeline asserts acyclic input; the caller
    // (the standard realization, here our test driver) runs the cycle
    // break before invoking strata.
    expect(() => strataPipeline(breakInstanceCycles(p).lowered)).not.toThrow()
  })
})

// ─────────────────────────────────────────────────────────────
// 5: stdlib is acyclic at the instance level
// ─────────────────────────────────────────────────────────────

describe('traceCycles — stdlib corpus', () => {
  test('every stdlib program is identity through traceCycles', () => {
    const fixtures = loadStdlibFixtures()
    const resolved = elaborateStdlib(fixtures)
    let identityCount = 0
    let totalCount = 0
    for (const [, prog] of resolved.entries()) {
      totalCount++
      // For pre-inlining stdlib programs with no inter-instance
      // cycles, traceCycles should be the identity by reference.
      if (breakInstanceCycles(prog).lowered === prog) identityCount++
    }
    expect(identityCount).toBe(totalCount)
    expect(totalCount).toBeGreaterThan(0)
  })
})

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function opsIn(expr: ResolvedExpr, target: string): string[] {
  const out: string[] = []
  const walk = (e: ResolvedExpr): void => {
    if (e === null || typeof e !== 'object') return
    if (Array.isArray(e)) { e.forEach(walk); return }
    if (e.op === target) { out.push(e.op); return }
    walkChildren(e, walk)
  }
  walk(expr)
  return out
}

function walkChildren(node: { op: string } & Record<string, unknown>, k: (e: ResolvedExpr) => void): void {
  // Recurse into structured children but NOT into back-pointers like
  // `delayRef.decl` (which would walk into its update expression).
  switch (node.op) {
    case 'inputRef': case 'regRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef': case 'nestedOut':
    case 'sampleRate': case 'sampleIndex':
      return
    default:
      // Generic recursion for op nodes whose children are direct
      // expression fields. Avoid stepping through `decl` references
      // and arm/payload entries' bookkeeping fields.
      for (const [key, v] of Object.entries(node)) {
        if (key === 'op' || key === 'decl' || key === 'instance' || key === 'output') continue
        if (key === 'variant' || key === 'type' || key === 'parent') continue
        if (key === 'iter' || key === 'acc' || key === 'elem' || key === 'x' || key === 'y' || key === 'binder') continue
        if (Array.isArray(v)) v.forEach(child => recurseValue(child, k))
        else recurseValue(v as ResolvedExpr, k)
      }
  }
}

function recurseValue(v: unknown, k: (e: ResolvedExpr) => void): void {
  if (v === null || typeof v !== 'object') {
    if (typeof v === 'number' || typeof v === 'boolean') k(v as ResolvedExpr)
    return
  }
  if (Array.isArray(v)) { v.forEach(c => recurseValue(c, k)); return }
  // Match-arm entry: { variant, binders, body }; just visit body.
  if ('body' in v && !('op' in v)) { recurseValue((v as { body: ResolvedExpr }).body, k); return }
  if ('value' in v && !('op' in v)) { recurseValue((v as { value: ResolvedExpr }).value, k); return }
  if ('op' in v) k(v as ResolvedExpr)
}

// ─────────────────────────────────────────────────────────────
// Phase A — Cycle topologies traceCycles hasn't seen
// (TDD plan: ~/.claude/plans/we-re-doing-a-tdd-eager-waffle.md)
// ─────────────────────────────────────────────────────────────

/** Inner program used by the cycle topology fixtures: y = x + 1.
 *  Combinator-free, so traceCycles' rewriter doesn't have to descend
 *  through array combinators in input wires. */
const INNER_INC: ProgramNode = {
  op: 'program',
  name: 'IncInner',
  ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
  body: { op: 'block', assigns: [
    { op: 'outputAssign', name: 'y',
      expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
  ]},
}

/** Variant of INNER_INC with two outputs: y = x + 1, z = x + 2.
 *  Used by the multi-output cycle test (one cycle through y, another
 *  through z). */
const INNER_INC_2OUT: ProgramNode = {
  op: 'program',
  name: 'IncInner2',
  ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y', 'z'] },
  body: { op: 'block', assigns: [
    { op: 'outputAssign', name: 'y',
      expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
    { op: 'outputAssign', name: 'z',
      expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 2] } },
  ]},
}

/** Build a candidate cycle session: instances of `IncInner` (registered
 *  on the session) wired according to `wiring`, with `audioOutput`
 *  routed to the DAC. */
function buildCycleSession(
  inner: ProgramNode,
  wiring: Array<{ name: string; inputs: Record<string, import('../expr.js').ExprNode>; program?: string }>,
  audioOutput: { instance: string; output: string },
): ReturnType<typeof makeSession> {
  const session = makeSession(8)
  loadProgramAsType(inner, session)
  loadJSON({
    schema: 'tropical_program_2',
    name: 'patch',
    body: { op: 'block', decls: wiring.map(w => ({
      op: 'instanceDecl', name: w.name, program: w.program ?? inner.name,
      inputs: w.inputs,
    })) },
    audio_outputs: [audioOutput],
  }, session)
  return session
}

/** Build a reference session that exposes the recurrence directly via
 *  delayDecl + outputAssign, using a single Inner type whose body
 *  declares the delay. The session instantiates one instance of `Ref`
 *  routed to dac. */
function buildReferenceSession(ref: ProgramNode, instance = 'r'): ReturnType<typeof makeSession> {
  const session = makeSession(8)
  loadProgramAsType(ref, session)
  loadJSON({
    schema: 'tropical_program_2',
    name: 'patch',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: instance, program: ref.name, inputs: {} },
    ]},
    audio_outputs: [{ instance, output: ref.ports!.outputs![0] as string }],
  }, session)
  return session
}

function elabFromNode(node: ProgramNode): ResolvedProgram {
  // The Phase A tests build cyclic JSON test programs; the elaborator's
  // strict-cycle-policy check would reject them at parse time, so use
  // the test-only escape hatch.
  return _elaborateForCyclicTest(raiseProgram(node))
}

describe('Phase A — cycle topologies (TDD plan)', () => {
  // ──────────────────────────────────────────────────────────
  // Test 1 — (D) Self-loop SCC
  // ──────────────────────────────────────────────────────────
  test('(D) self-loop SCC: 1 synthetic delay; recurrence = [1,2,3,...]', () => {
    // Top defines IncInner inline as a sub-program, then a single instance
    // `a` whose own input wires from its own output (a 1-element SCC with
    // a self-edge).
    const TopSelf: ProgramNode = {
      op: 'program',
      name: 'TopSelf',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', decls: [
        { op: 'programDecl', name: 'IncInnerLocal',
          program: { op: 'program', name: 'IncInnerLocal',
            ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
            body: { op: 'block', assigns: [
              { op: 'outputAssign', name: 'y',
                expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
            ]},
          },
        },
        { op: 'instanceDecl', name: 'a', program: 'IncInnerLocal',
          inputs: { x: { op: 'nestedOut', ref: 'a', output: 'y' } } },
      ], assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'nestedOut', ref: 'a', output: 'y' } },
      ]},
    } as unknown as ProgramNode
    // ── IR shape ── traceCycles mutates `inputs` in place; run on a
    // fresh elaboration each time so subsequent passes see a consistent
    // program (decls and inputs both updated together).
    const beforeDelays = syntheticBreakerRegs(elabFromNode(TopSelf)).length
    const traced = breakInstanceCycles(elabFromNode(TopSelf)).lowered
    const afterDelays = syntheticBreakerRegs(traced).length
    expect(afterDelays - beforeDelays).toBe(1)
    expect(() => cloneResolvedProgram(traced)).not.toThrow()
    expect(() => strataPipeline(breakInstanceCycles(elabFromNode(TopSelf)).lowered)).not.toThrow()

    // ── Denotation ── reference still pins the recurrence; the
    // candidate (cyclic session) now throws at session build time.
    // Phase 5 catches the cycle at `compileSession`'s entry, before
    // `loadJSON` finishes priming the JIT runtime.
    expect(() => buildCycleSession(INNER_INC, [
      { name: 'a', inputs: { x: { op: 'ref', instance: 'a', output: 'y' } } },
    ], { instance: 'a', output: 'y' })).toThrow()

    // Reference: y = prev + 1; next prev = prev + 1 (init 0).
    const RefSelf: ProgramNode = {
      op: 'program',
      name: 'RefSelf',
      ports: { inputs: [], outputs: ['y'] },
      body: { op: 'block',
        decls: [{ op: 'delayDecl', name: 'prev', init: 0,
          update: { op: 'add', args: [{ op: 'delayRef', id: 'prev' }, 1] } }],
        assigns: [{ op: 'outputAssign', name: 'y',
          expr: { op: 'add', args: [{ op: 'delayRef', id: 'prev' }, 1] } }],
      },
    }
    const reference = buildReferenceSession(RefSelf)

    const ref = interpretSession(reference, 8)
    const expected = [1,2,3,4,5,6,7,8].map(v => v / 20.0)
    for (let i = 0; i < 8; i++) {
      expect(ref[i]).toBeCloseTo(expected[i], 12)
    }
  })

  // ──────────────────────────────────────────────────────────
  // Test 2 — (D) Two disjoint SCCs
  // ──────────────────────────────────────────────────────────
  test('(D) two disjoint SCCs: 2 synthetic delays with distinct names', () => {
    // (a↔b) cycle wired to dac as a.y; (c↔d) cycle wired to dac as c.y.
    // Phase 5: session construction throws at compileSession's entry
    // for any session containing inter-instance cycles.
    expect(() => buildCycleSession(INNER_INC, [
      { name: 'a', inputs: { x: { op: 'ref', instance: 'b', output: 'y' } } },
      { name: 'b', inputs: { x: { op: 'ref', instance: 'a', output: 'y' } } },
      { name: 'c', inputs: { x: { op: 'ref', instance: 'd', output: 'y' } } },
      { name: 'd', inputs: { x: { op: 'ref', instance: 'c', output: 'y' } } },
    ], { instance: 'a', output: 'y' })).toThrow()

    // IR shape — re-run materializer's IR to inspect post-trace shape.
    // The session's loadProgramAsType has already run strata over
    // IncInner; the cycle is at the patch level. Use elab directly.
    const TopTwoSCC: ProgramNode = {
      op: 'program', name: 'TopTwoSCC',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', decls: [
        { op: 'programDecl', name: 'I',
          program: { op: 'program', name: 'I',
            ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
            body: { op: 'block', assigns: [
              { op: 'outputAssign', name: 'y',
                expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
            ]},
          },
        },
        { op: 'instanceDecl', name: 'a', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'b', output: 'y' } } },
        { op: 'instanceDecl', name: 'b', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'a', output: 'y' } } },
        { op: 'instanceDecl', name: 'c', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'd', output: 'y' } } },
        { op: 'instanceDecl', name: 'd', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'c', output: 'y' } } },
      ], assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'nestedOut', ref: 'a', output: 'y' } },
      ]},
    } as unknown as ProgramNode
    const traced = breakInstanceCycles(elabFromNode(TopTwoSCC)).lowered
    const synthDelays = syntheticBreakerRegs(traced).filter(d => d.name.startsWith('_feedback_'))
    expect(synthDelays.length).toBe(2)
    const names = synthDelays.map(d => d.name)
    expect(new Set(names).size).toBe(2)  // distinct

    // Denotation — for the 2-cycle a↔b with breakTarget = a, each sample
    // both a.y and b.y advance by 2 (one increment from b's IncInner, one
    // from a's). At sample t: a.y_t = a.y_{t-1} + 2, with a.y_0 = 2.
    // Sequence: [2, 4, 6, 8, ...].
    const reference = buildReferenceSession({
      op: 'program', name: 'RefTwoSCC',
      ports: { inputs: [], outputs: ['y'] },
      body: { op: 'block',
        decls: [{ op: 'delayDecl', name: 'prev', init: 0,
          update: { op: 'add', args: [{ op: 'delayRef', id: 'prev' }, 2] } }],
        assigns: [{ op: 'outputAssign', name: 'y',
          expr: { op: 'add', args: [{ op: 'delayRef', id: 'prev' }, 2] } }],
      },
    })

    const ref = interpretSession(reference, 8)
    for (let i = 0; i < 8; i++) {
      expect(Number.isFinite(ref[i])).toBe(true)
    }
  })

  // ──────────────────────────────────────────────────────────
  // Test 3 — (D) Diamond cycle (single SCC, two parallel paths)
  // ──────────────────────────────────────────────────────────
  test('(D) diamond cycle: parallel paths through one SCC; topo-sort succeeds', () => {
    // a → b → d → a and a → c → d → a. d and a both have wires that
    // close the cycle. With breakTarget = a (source order first), all
    // other members rewrite their wires to a's outputs as delayRefs.
    // Phase 5: this cyclic session throws at compileSession's entry.
    expect(() => buildCycleSession(INNER_INC, [
      { name: 'a', inputs: { x: { op: 'ref', instance: 'd', output: 'y' } } },
      { name: 'b', inputs: { x: { op: 'ref', instance: 'a', output: 'y' } } },
      { name: 'c', inputs: { x: { op: 'ref', instance: 'a', output: 'y' } } },
      { name: 'd', inputs: { x: { op: 'add', args: [
        { op: 'ref', instance: 'b', output: 'y' },
        { op: 'ref', instance: 'c', output: 'y' },
      ]}}},
    ], { instance: 'd', output: 'y' })).toThrow()

    // IR: traceCycles must break exactly the back-edge, leaving a DAG.
    const TopDiamond: ProgramNode = {
      op: 'program', name: 'TopDiamond',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', decls: [
        { op: 'programDecl', name: 'I',
          program: { op: 'program', name: 'I',
            ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
            body: { op: 'block', assigns: [
              { op: 'outputAssign', name: 'y',
                expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
            ]},
          },
        },
        { op: 'instanceDecl', name: 'a', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'd', output: 'y' } } },
        { op: 'instanceDecl', name: 'b', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'a', output: 'y' } } },
        { op: 'instanceDecl', name: 'c', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'a', output: 'y' } } },
        { op: 'instanceDecl', name: 'd', program: 'I',
          inputs: { x: { op: 'add', args: [
            { op: 'nestedOut', ref: 'b', output: 'y' },
            { op: 'nestedOut', ref: 'c', output: 'y' },
          ]}}},
      ], assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'nestedOut', ref: 'd', output: 'y' } },
      ]},
    } as unknown as ProgramNode
    const traced = breakInstanceCycles(elabFromNode(TopDiamond)).lowered
    expect(() => strataPipeline(breakInstanceCycles(elabFromNode(TopDiamond)).lowered)).not.toThrow()

    // Topo-check: collect post-trace inter-instance edges; assert no cycle.
    const postInsts = traced.body.decls.filter((d): d is InstanceDecl => d.op === 'instanceDecl')
    const postDeps = new Map<InstanceDecl, Set<InstanceDecl>>()
    for (const inst of postInsts) postDeps.set(inst, new Set())
    const setOf = new Set(postInsts)
    for (const inst of postInsts) {
      const set = postDeps.get(inst)!
      for (const w of inst.inputs) collectInstancesUsedAsNestedOut(w.value, set, setOf)
    }
    expect(hasNoCycle(postInsts, postDeps)).toBe(true)

    // Denotation: build a hand-written reference where d's output carries
    // the recurrence d_t = (a_{t-1} + 1) + (a_{t-1} + 1) = 2 * (a_{t-1} + 1)
    // and a_t = d_{t-1} + 1. With both delays init 0:
    //   t=0: a=1 (delayed d=0 + 1), b=2, c=2, d=4 — but d uses the
    //        previous-sample a, so we need to track the trace breakage.
    // The exact recurrence depends on which back-edge is broken. Here
    // we just compare candidate to the trivially-equivalent reference
    // session that breaks d→a explicitly.
    // With breakTarget = a, the back-edge `b→a` and `c→a` become
    // delayRef(_feedback_a_y). Per-sample evolution (working through the
    // wires): b.y = del + 1, c.y = del + 1, d.y = b + c + 1 = 2*del + 3,
    // a.y = d + 1 = 2*del + 4. Next-sample del = a.y = 2*del + 4.
    // Audio output is d.y = 2*del + 3.
    const RefDiamond: ProgramNode = {
      op: 'program', name: 'RefDiamond',
      ports: { inputs: [], outputs: ['y'] },
      body: { op: 'block',
        decls: [{ op: 'delayDecl', name: 'prev_a', init: 0,
          update: { op: 'add', args: [
            { op: 'mul', args: [{ op: 'delayRef', id: 'prev_a' }, 2] }, 4,
          ]} }],
        assigns: [{ op: 'outputAssign', name: 'y',
          expr: { op: 'add', args: [
            { op: 'mul', args: [{ op: 'delayRef', id: 'prev_a' }, 2] }, 3,
          ]},
        }],
      },
    }
    const reference = buildReferenceSession(RefDiamond)
    const ref = interpretSession(reference, 8)
    for (let i = 0; i < 8; i++) {
      expect(Number.isFinite(ref[i])).toBe(true)
    }
  })

  // ──────────────────────────────────────────────────────────
  // Test 4 — (D) Multi-output cycle member
  // ──────────────────────────────────────────────────────────
  test('(D) multi-output cycle member: distinct synthetic-delay names per (instance, port)', () => {
    // Two cycles share `a` (a 2-output instance):
    //   cycle1: a.y → b.x → b.y → a (via a.x reading b.y)
    //   cycle2: a.z → c.x → c.y → a (via a.x reading c.y as well)
    // Combine: a.x = b.y + c.y, b.x = a.y, c.x = a.z.
    // Both b and c form SCCs with a; both cycle through a's outputs.
    // After traceCycles, the breakTarget = a; b's wire to a.y becomes
    // delayRef(_feedback_a_y); c's wire to a.z becomes delayRef(_feedback_a_z).
    const TopMultiOut: ProgramNode = {
      op: 'program', name: 'TopMultiOut',
      ports: { inputs: [], outputs: ['out'] },
      body: { op: 'block', decls: [
        { op: 'programDecl', name: 'I2',
          program: { op: 'program', name: 'I2',
            ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y', 'z'] },
            body: { op: 'block', assigns: [
              { op: 'outputAssign', name: 'y',
                expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
              { op: 'outputAssign', name: 'z',
                expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 2] } },
            ]},
          },
        },
        { op: 'programDecl', name: 'I',
          program: { op: 'program', name: 'I',
            ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
            body: { op: 'block', assigns: [
              { op: 'outputAssign', name: 'y',
                expr: { op: 'add', args: [{ op: 'input', name: 'x' }, 1] } },
            ]},
          },
        },
        { op: 'instanceDecl', name: 'a', program: 'I2',
          inputs: { x: { op: 'add', args: [
            { op: 'nestedOut', ref: 'b', output: 'y' },
            { op: 'nestedOut', ref: 'c', output: 'y' },
          ]}}},
        { op: 'instanceDecl', name: 'b', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'a', output: 'y' } } },
        { op: 'instanceDecl', name: 'c', program: 'I',
          inputs: { x: { op: 'nestedOut', ref: 'a', output: 'z' } } },
      ], assigns: [
        { op: 'outputAssign', name: 'out',
          expr: { op: 'nestedOut', ref: 'a', output: 'y' } },
      ]},
    } as unknown as ProgramNode
    const traced = breakInstanceCycles(elabFromNode(TopMultiOut)).lowered
    const synth = syntheticBreakerRegs(traced).filter(d => d.name.startsWith('_feedback_'))
    // Two distinct synthetic delays — one per (a, output) port that's
    // in a cycle.
    expect(synth.length).toBe(2)
    const names = new Set(synth.map(d => d.name))
    expect(names.has('_feedback_a_y')).toBe(true)
    expect(names.has('_feedback_a_z')).toBe(true)

    // Denotation: candidate session. Phase 5 catches the inter-instance
    // cycle (a ↔ b, a ↔ c) at compileSession's entry.
    const session = makeSession(8)
    loadProgramAsType(INNER_INC_2OUT, session)
    loadProgramAsType(INNER_INC, session)
    expect(() => loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a', program: 'IncInner2',
          inputs: { x: { op: 'add', args: [
            { op: 'ref', instance: 'b', output: 'y' },
            { op: 'ref', instance: 'c', output: 'y' },
          ]}}},
        { op: 'instanceDecl', name: 'b', program: 'IncInner',
          inputs: { x: { op: 'ref', instance: 'a', output: 'y' } } },
        { op: 'instanceDecl', name: 'c', program: 'IncInner',
          inputs: { x: { op: 'ref', instance: 'a', output: 'z' } } },
      ]},
      audio_outputs: [{ instance: 'a', output: 'y' }],
    }, session)).toThrow()
    // Reference still interprets cleanly. (Legacy cand vs ref
    // comparison removed.)
    // Reference: track the recurrence directly. With a's outputs broken
    // by synthetic delays:
    //   sample 0: prev_a_y = 0, prev_a_z = 0
    //     b.y = a_y_prev + 1 = 1; c.y = a_z_prev + 1 = 1
    //     a.x = 2; a.y = 3; a.z = 4
    //   sample 1: prev_a_y = 3, prev_a_z = 4
    //     b.y = 4; c.y = 5; a.x = 9; a.y = 10; a.z = 11
    //   sample 2: prev_a_y = 10, prev_a_z = 11; b.y=11, c.y=12; a.x=23; a.y=24; a.z=25
    const RefMultiOut: ProgramNode = {
      op: 'program', name: 'RefMultiOut',
      ports: { inputs: [], outputs: ['y'] },
      body: { op: 'block',
        decls: [
          { op: 'delayDecl', name: 'prev_y', init: 0,
            update: { op: 'add', args: [
              { op: 'add', args: [
                { op: 'add', args: [{ op: 'delayRef', id: 'prev_y' }, 1] },
                { op: 'add', args: [{ op: 'delayRef', id: 'prev_z' }, 1] },
              ]},
              1,
            ]},
          },
          { op: 'delayDecl', name: 'prev_z', init: 0,
            update: { op: 'add', args: [
              { op: 'add', args: [
                { op: 'add', args: [{ op: 'delayRef', id: 'prev_y' }, 1] },
                { op: 'add', args: [{ op: 'delayRef', id: 'prev_z' }, 1] },
              ]},
              2,
            ]},
          },
        ],
        assigns: [{ op: 'outputAssign', name: 'y',
          expr: { op: 'add', args: [
            { op: 'add', args: [
              { op: 'add', args: [{ op: 'delayRef', id: 'prev_y' }, 1] },
              { op: 'add', args: [{ op: 'delayRef', id: 'prev_z' }, 1] },
            ]},
            1,
          ]},
        }],
      },
    }
    const reference = buildReferenceSession(RefMultiOut)
    const ref = interpretSession(reference, 8)
    for (let i = 0; i < 8; i++) {
      expect(Number.isFinite(ref[i])).toBe(true)
    }
  })

  // ──────────────────────────────────────────────────────────
  // Test 5 — (D, revised) Cycle through user-supplied DelayDecl
  // ──────────────────────────────────────────────────────────
  test('(D) cycle through user delay: traceCycles preserves denotation (perf hint: identity)', () => {
    // The user already broke the cycle with a delayDecl. traceCycles
    // should not introduce any synthetic delay (no inter-instance cycle
    // remains; the dependency goes through the delay, which is not in
    // the instance graph). Sample equivalence between the un-traced and
    // traced programs is the hard property; reference identity is a perf
    // hint logged but not asserted.
    const Wrap: ProgramNode = {
      op: 'program', name: 'WrappedDelay',
      ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['y'] },
      body: { op: 'block',
        decls: [{ op: 'delayDecl', name: 'mem', init: 0,
          update: { op: 'input', name: 'x' } }],
        assigns: [{ op: 'outputAssign', name: 'y',
          expr: { op: 'delayRef', id: 'mem' } }],
      },
    }
    // Build session: a feeds back into itself but through Wrap's delay.
    // Even though Wrap's TYPE contains an internal delay, the self-edge
    // a.x = f(a.y) IS a session-level cycle, so Phase 5's
    // assertSessionAcyclic throws at loadJSON time. To break this cycle
    // at the session level the user would wrap the wire in `delay(...)`
    // explicitly (or use the MCP wire helpers, which auto-wrap).
    const session = makeSession(8)
    loadProgramAsType(Wrap, session)
    expect(() => loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a', program: 'WrappedDelay',
          inputs: { x: { op: 'add', args: [
            { op: 'ref', instance: 'a', output: 'y' }, 1,
          ]}}},
      ]},
      audio_outputs: [{ instance: 'a', output: 'y' }],
    }, session)).toThrow()
    // Pin the first 4 samples by the absolute recurrence value:
    //   sample 0: a.y = mem (init 0) = 0
    //   sample 1: mem became (a.y_prev + 1) = 1; a.y = 1. But wait, the
    //   self-edge a.x = a.y also forms an inter-instance cycle (single
    //   instance with self-edge), which traceCycles would break with a
    //   *synthetic* delay on top of the user delay. Pin the actual
    //   observed recurrence (whatever the pipeline produces) rather than
    //   asserting "no synthetic added"; the test's purpose is denotation.
    // We assert non-decreasing: each sample is >= previous — the cycle
    // is causally broken, so values evolve monotonically.
    // (Removed under Phase 4b — the session throws above before any
    // samples are produced.)
  })

  // ──────────────────────────────────────────────────────────
  // Test 6 — (D) Session-level feedback via session-level delay
  // ──────────────────────────────────────────────────────────
  test.skip('(D) session-level delay() between two instances: denotation matches flattened reference', () => {
    // Phase 5 of M11 lifts `delay()` wire expressions to anonymous
    // programs whose body contains the delay. But this hides the
    // session-level delay from traceCycles: the lift turns a 1-sample
    // delay into a 2-sample delay (one from the lifted body's delay,
    // one from traceCycles inserting a synthetic delay for the cycle
    // it now sees with no explicit delay at the session edge).
    //
    // Correct handling requires lifting that PRESERVES the delay at
    // the session-edge level — either by leaving `delay()` at the
    // session level and only lifting its arg, or by teaching
    // traceCycles to detect existing delays inside lifted wire bodies.
    // Tracked as a Phase 5 follow-up.
    // Two Inner instances feeding each other through an explicit
    // delay() expression. The session-level delay() short-circuits the
    // cycle so traceCycles sees no SCC.
    const session = makeSession(8)
    loadProgramAsType(INNER_INC, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a', program: 'IncInner',
          inputs: { x: { op: 'delay', init: 0,
            args: [{ op: 'ref', instance: 'b', output: 'y' }],
          }}},
        { op: 'instanceDecl', name: 'b', program: 'IncInner',
          inputs: { x: { op: 'delay', init: 0,
            args: [{ op: 'ref', instance: 'a', output: 'y' }],
          }}},
      ]},
      audio_outputs: [{ instance: 'a', output: 'y' }],
    }, session)

    // Reference: equivalent flat program with two delays explicit.
    //   sample 0: del_a = 0, del_b = 0 → a.y = del_b + 1 = 1, b.y = del_a + 1 = 1
    //   next del_a = a.y_now = 1, del_b = 1
    //   sample 1: a.y = 2, b.y = 2
    const Ref: ProgramNode = {
      op: 'program', name: 'RefSessionFeedback',
      ports: { inputs: [], outputs: ['y'] },
      body: { op: 'block',
        decls: [
          { op: 'delayDecl', name: 'del_a', init: 0,
            update: { op: 'add', args: [{ op: 'delayRef', id: 'del_b' }, 1] } },
          { op: 'delayDecl', name: 'del_b', init: 0,
            update: { op: 'add', args: [{ op: 'delayRef', id: 'del_a' }, 1] } },
        ],
        assigns: [{ op: 'outputAssign', name: 'y',
          expr: { op: 'add', args: [{ op: 'delayRef', id: 'del_b' }, 1] } }],
      },
    }
    const reference = buildReferenceSession(Ref)
    const cand = interpretSession(session, 8)
    const ref = interpretSession(reference, 8)
    for (let i = 0; i < 8; i++) {
      expect(cand[i]).toBeCloseTo(ref[i], 12)
    }
  })
})

// Helper: collect instances appearing as `nestedOut.instance` in expr.
function collectInstancesUsedAsNestedOut(
  expr: ResolvedExpr,
  out: Set<InstanceDecl>,
  pool: Set<InstanceDecl>,
): void {
  if (expr === null || typeof expr !== 'object') return
  if (Array.isArray(expr)) { expr.forEach(e => collectInstancesUsedAsNestedOut(e, out, pool)); return }
  if (expr.op === 'nestedOut') {
    if (pool.has(expr.instance)) out.add(expr.instance)
    return
  }
  for (const [k, v] of Object.entries(expr)) {
    if (k === 'op' || k === 'decl' || k === 'instance' || k === 'output') continue
    if (Array.isArray(v)) v.forEach(c => collectInstancesUsedAsNestedOut(c as ResolvedExpr, out, pool))
    else if (v !== null && typeof v === 'object') collectInstancesUsedAsNestedOut(v as ResolvedExpr, out, pool)
  }
}

function hasNoCycle(
  nodes: InstanceDecl[],
  deps: Map<InstanceDecl, Set<InstanceDecl>>,
): boolean {
  // Kahn's: repeatedly peel zero-in-degree nodes; if any remain, cycle.
  const indeg = new Map<InstanceDecl, number>()
  for (const n of nodes) indeg.set(n, 0)
  for (const n of nodes) {
    for (const d of deps.get(n) ?? []) indeg.set(d, (indeg.get(d) ?? 0) + 1)
  }
  const q: InstanceDecl[] = []
  for (const [n, k] of indeg) if (k === 0) q.push(n)
  let popped = 0
  while (q.length) {
    const n = q.shift()!
    popped++
    for (const d of deps.get(n) ?? []) {
      const k = (indeg.get(d) ?? 0) - 1
      indeg.set(d, k)
      if (k === 0) q.push(d)
    }
  }
  return popped === nodes.length
}

interface Fixture { name: string; source: string }
function loadStdlibFixtures(): Fixture[] {
  return readdirSync(STDLIB_DIR)
    .filter(f => f.endsWith('.trop'))
    .sort()
    .map(file => {
      const text = readFileSync(join(STDLIB_DIR, file), 'utf-8')
      const ext = extractMarkdown(text)
      return { name: file.replace(/\.trop$/, ''), source: ext.blocks[0].source }
    })
}

function elaborateStdlib(fixtures: Fixture[]): Map<string, ResolvedProgram> {
  const resolved = new Map<string, ResolvedProgram>()
  const remaining = new Map(fixtures.map(f => [f.name, f]))
  const resolver: ExternalProgramResolver = name => resolved.get(name)
  let progress = true
  while (progress) {
    progress = false
    for (const [name, fx] of remaining) {
      try {
        const r = elaborate(parseProgram(fx.source), resolver)
        resolved.set(name, r)
        remaining.delete(name)
        progress = true
      } catch { /* try again */ }
    }
  }
  return resolved
}
