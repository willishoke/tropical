/**
 * Phase 3 — sum-typed delay decomposition end-to-end.
 *
 * Hand-authored tropical_program_2 fixtures with sum-typed delay_decls,
 * tag/match expressions, and exercises the loader → flatten → emit pipeline.
 */

import { describe, expect, test } from 'bun:test'
import { makeSession, loadJSON } from './session.js'
import { loadProgramAsType, loadStdlib } from './program.js'
import type { ProgramNode } from './program.js'
import { applySessionWiring } from './apply_plan.js'
import { flattenExpressions } from './flatten.js'
import { interpretSamples } from './interpret.js'

// ─────────────────────────────────────────────────────────────
// Fixture: Toggle — Sum{Off, On}, alternates each sample.
// Simplest possible sum-typed program: two nullary variants, no payload.
// Just exercises the discriminator slot.
// ─────────────────────────────────────────────────────────────

function toggleProgram(): ProgramNode {
  return {
    op: 'program',
    name: 'Toggle',
    ports: {
      outputs: [{ name: 'value', type: 'float' }],
      type_defs: [{
        kind: 'sum',
        name: 'TogState',
        variants: [
          { name: 'Off', payload: [] },
          { name: 'On',  payload: [] },
        ],
      }],
    },
    body: {
      op: 'block',
      decls: [
        {
          op: 'delayDecl',
          name: 'state',
          type: 'TogState',
          init: { op: 'tag', type: 'TogState', variant: 'Off' },
          update: {
            op: 'match',
            type: 'TogState',
            scrutinee: { op: 'delayRef', id: 'state' },
            arms: {
              Off: { body: { op: 'tag', type: 'TogState', variant: 'On' } },
              On:  { body: { op: 'tag', type: 'TogState', variant: 'Off' } },
            },
          },
        },
      ],
      assigns: [
        {
          op: 'outputAssign',
          name: 'value',
          expr: {
            op: 'match',
            type: 'TogState',
            scrutinee: { op: 'delayRef', id: 'state' },
            arms: {
              Off: { body: 0 },
              On:  { body: 1 },
            },
          },
        },
      ],
    },
  }
}

describe('Toggle — Sum{Off, On} loaded into a ProgramDef', () => {
  test('the program loads without error', () => {
    const session = makeSession()
    const prog = toggleProgram()
    expect(() => loadProgramAsType(prog, session)).not.toThrow()
  })

  test('the sum type is registered', () => {
    const session = makeSession()
    loadProgramAsType(toggleProgram(), session)
    expect(session.sumTypeRegistry.has('TogState')).toBe(true)
    const meta = session.sumTypeRegistry.get('TogState')!
    expect(meta.variants).toHaveLength(2)
    expect(meta.variants[0].name).toBe('Off')
    expect(meta.variants[1].name).toBe('On')
  })

  test('the program registers as a type', () => {
    const session = makeSession()
    loadProgramAsType(toggleProgram(), session)
    expect(session.typeRegistry.has('Toggle')).toBe(true)
  })

  test('the resulting ProgramDef has a single-slot delay (just the tag)', () => {
    const session = makeSession()
    const type = loadProgramAsType(toggleProgram(), session)!
    const def = type._def
    // For Sum{Off, On} with no payloads, the bundle has exactly one slot:
    // the discriminator. So delayUpdateNodes should have length 1.
    expect(def.delayUpdateNodes).toHaveLength(1)
    expect(def.delayInitValues).toHaveLength(1)
    // Init: variant index of 'Off' is 0.
    expect(def.delayInitValues[0]).toBe(0)
  })

  test('the delay update lowers to a select chain dispatching on the tag', () => {
    const session = makeSession()
    const type = loadProgramAsType(toggleProgram(), session)!
    const update = type._def.delayUpdateNodes[0]
    // Expected shape (paraphrased after slottification):
    //   select(eq(delay_value(state#tag), 0), 1 /* On */, 0 /* Off */)
    // The two variant arms swap: when state == Off (idx 0), become On (idx 1);
    // otherwise (state == On), become Off (idx 0).
    expect(typeof update).toBe('object')
    const u = update as Record<string, unknown>
    expect(u.op).toBe('select')
    const args = u.args as ExprNode[]
    expect(args).toHaveLength(3)
    // Condition: eq(delay_value(state#tag), 0)
    const cond = args[0] as Record<string, unknown>
    expect(cond.op).toBe('eq')
    // Then: 1 (On's index)
    expect(args[1]).toBe(1)
    // Else: 0 (Off's index)
    expect(args[2]).toBe(0)
  })

  test('the output expression lowers to a select chain', () => {
    const session = makeSession()
    const type = loadProgramAsType(toggleProgram(), session)!
    const out = type._def.outputExprNodes[0]
    expect(typeof out).toBe('object')
    const o = out as Record<string, unknown>
    expect(o.op).toBe('select')
    const args = o.args as ExprNode[]
    expect(args).toHaveLength(3)
    // When tag == Off (0): output 0; else output 1.
    expect(args[1]).toBe(0)
    expect(args[2]).toBe(1)
  })
})

// ─────────────────────────────────────────────────────────────
// Type alias for ExprNode — used in the assertions above.
// ─────────────────────────────────────────────────────────────
type ExprNode = import('./expr.js').ExprNode

// ─────────────────────────────────────────────────────────────
// Fixture: Counter — Sum{Idle, Counting{n: int}}.
//
// On a trigger (here, an internal trigger that fires every sample once
// after entering Counting via the cycle-breaker pattern), advances n.
// We use a simpler shape for the test: starts in Idle, then unconditionally
// transitions to Counting{n: 0} on first sample, then increments n each
// subsequent sample.
//
// This exercises:
//   - delay_decl with a sum type that has a payload field (n: int).
//   - tag with payload (Tag<Counting>{n: 0}, Tag<Counting>{n: n+1}).
//   - match arm with `bind: 'n'` reading the payload.
//   - tag-slot init for non-init variants populating field slots with 0.
// ─────────────────────────────────────────────────────────────

function counterProgram(): ProgramNode {
  return {
    op: 'program',
    name: 'Counter',
    ports: {
      outputs: [{ name: 'count', type: 'float' }],
      type_defs: [{
        kind: 'sum',
        name: 'CounterState',
        variants: [
          { name: 'Idle', payload: [] },
          { name: 'Counting', payload: [{ name: 'n', scalar_type: 'int' }] },
        ],
      }],
    },
    body: {
      op: 'block',
      decls: [
        {
          op: 'delayDecl',
          name: 'state',
          type: 'CounterState',
          init: { op: 'tag', type: 'CounterState', variant: 'Idle' },
          // Update: in Idle, transition to Counting{n: 0}; in Counting{n},
          // become Counting{n+1}.
          update: {
            op: 'match',
            type: 'CounterState',
            scrutinee: { op: 'delayRef', id: 'state' },
            arms: {
              Idle: {
                body: { op: 'tag', type: 'CounterState', variant: 'Counting',
                        payload: { n: 0 } },
              },
              Counting: {
                bind: 'n',
                body: {
                  op: 'tag', type: 'CounterState', variant: 'Counting',
                  payload: {
                    n: { op: 'add', args: [{ op: 'binding', name: 'n' }, 1] },
                  },
                },
              },
            },
          },
        },
      ],
      assigns: [
        {
          op: 'outputAssign',
          name: 'count',
          expr: {
            op: 'match',
            type: 'CounterState',
            scrutinee: { op: 'delayRef', id: 'state' },
            arms: {
              Idle:     { body: 0 },
              Counting: { bind: 'n', body: { op: 'binding', name: 'n' } },
            },
          },
        },
      ],
    },
  }
}

describe('Counter — Sum{Idle, Counting{n: int}} payload support', () => {
  test('the program loads without error', () => {
    const session = makeSession()
    expect(() => loadProgramAsType(counterProgram(), session)).not.toThrow()
  })

  test('the resulting ProgramDef has 2 delay slots (tag + Counting.n)', () => {
    const session = makeSession()
    const type = loadProgramAsType(counterProgram(), session)!
    const def = type._def
    expect(def.delayUpdateNodes).toHaveLength(2)
    expect(def.delayInitValues).toHaveLength(2)
    // Init: Idle has variant index 0; Counting.n slot is 0 (non-init variant).
    expect(def.delayInitValues[0]).toBe(0) // tag = Idle = 0
    expect(def.delayInitValues[1]).toBe(0) // Counting.n = 0 (init variant is not Counting)
  })

  test('end-to-end interpreter run produces correct values before audio clamp kicks in', () => {
    // Output of n exceeds the audio-output [-1, 1] clamp once n > 1, so
    // we only verify the first 3 samples (n stays at 0, 0, 1). The full
    // payload-passing semantics is covered by JIT-vs-interp equivalence
    // below, which doesn't depend on the clamp.
    const session = makeSession(16)
    loadStdlib(session)
    loadProgramAsType(counterProgram(), session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'c1', program: 'Counter' },
      ]},
      audio_outputs: [{ instance: 'c1', output: 'count' }],
    } as never, session)
    applySessionWiring(session)

    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, 3)
    // Sample 0: state=Idle → output 0. next_state=Counting{n:0}
    // Sample 1: state=Counting{0} → output 0/20. next_state=Counting{n:1}
    // Sample 2: state=Counting{1} → output 1/20. next_state=Counting{n:2}
    expect(interp[0]).toBeCloseTo(0, 10)
    expect(interp[1]).toBeCloseTo(0, 10)
    expect(interp[2]).toBeCloseTo(1 / 20, 10)
  })

  test('JIT and interpreter agree sample-for-sample on Counter', () => {
    const session = makeSession(16)
    loadStdlib(session)
    loadProgramAsType(counterProgram(), session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'c1', program: 'Counter' },
      ]},
      audio_outputs: [{ instance: 'c1', output: 'count' }],
    } as never, session)
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)

    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(interp[i]).toBeCloseTo(jit[i], 10)
    }
  })
})

// ─────────────────────────────────────────────────────────────
// End-to-end execution: JIT and interpreter produce the same alternating
// 0,1,0,1,... output stream for the Toggle program.
// ─────────────────────────────────────────────────────────────

describe('Toggle — end-to-end execution', () => {
  test('interpreter produces alternating 0,1,0,1,... values', () => {
    const session = makeSession(16)
    loadStdlib(session)
    loadProgramAsType(toggleProgram(), session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 't1', program: 'Toggle' },
      ]},
      audio_outputs: [{ instance: 't1', output: 'value' }],
    } as never, session)
    applySessionWiring(session)

    const flat = flattenExpressions(session)
    const N = 8
    const interp = interpretSamples(flat, N)
    // Output goes through mixed / 20.0 scaling.
    // Sample 0: state=Off → output 0.   next_state=On
    // Sample 1: state=On  → output 1.   next_state=Off
    // Sample 2: state=Off → output 0.   ...
    // Expected stream after scaling: 0, 0.05, 0, 0.05, 0, 0.05, 0, 0.05
    for (let i = 0; i < N; i++) {
      const expected = (i % 2 === 0) ? 0 : 0.05
      expect(interp[i]).toBeCloseTo(expected, 10)
    }
  })

  test('JIT and interpreter agree sample-for-sample', () => {
    const session = makeSession(16)
    loadStdlib(session)
    loadProgramAsType(toggleProgram(), session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 't1', program: 'Toggle' },
      ]},
      audio_outputs: [{ instance: 't1', output: 'value' }],
    } as never, session)
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)

    const flat = flattenExpressions(session)
    const interp = interpretSamples(flat, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(interp[i]).toBeCloseTo(jit[i], 10)
    }
  })
})
