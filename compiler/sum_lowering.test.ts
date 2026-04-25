/**
 * Phase 3 — sum-typed delay decomposition end-to-end.
 *
 * Hand-authored tropical_program_2 fixtures with sum-typed delay_decls,
 * tag/match expressions, and exercises the loader → flatten → emit pipeline.
 */

import { describe, expect, test } from 'bun:test'
import { makeSession } from './session.js'
import { loadProgramAsType } from './program.js'
import type { ProgramNode } from './program.js'

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
          op: 'delay_decl',
          name: 'state',
          type: 'TogState',
          init: { op: 'tag', type: 'TogState', variant: 'Off' },
          update: {
            op: 'match',
            type: 'TogState',
            scrutinee: { op: 'delay_ref', id: 'state' },
            arms: {
              Off: { body: { op: 'tag', type: 'TogState', variant: 'On' } },
              On:  { body: { op: 'tag', type: 'TogState', variant: 'Off' } },
            },
          },
        },
      ],
      assigns: [
        {
          op: 'output_assign',
          name: 'value',
          expr: {
            op: 'match',
            type: 'TogState',
            scrutinee: { op: 'delay_ref', id: 'state' },
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
