/**
 * Phase 1 — sum-type metadata: helpers and registry behavior.
 */

import { describe, expect, test } from 'bun:test'
import {
  type SumTypeMeta, sumVariantIndex, sumBundleSlots, sumVariantField, mangleSumSlot,
} from './term.js'
import { decodePortTypeDecl, makeSession } from './session.js'
import { loadStdlib } from './program.js'
import { loadProgramAsType } from './program.js'

const Env: SumTypeMeta = {
  name: 'Env',
  variants: [
    { name: 'Idle', payload: [] },
    { name: 'Decaying', payload: [{ name: 'level', scalar: 'float' }] },
  ],
}

const Multi: SumTypeMeta = {
  name: 'Multi',
  variants: [
    { name: 'A', payload: [{ name: 'x', scalar: 'int' }, { name: 'y', scalar: 'float' }] },
    { name: 'B', payload: [{ name: 'z', scalar: 'bool' }] },
    { name: 'C', payload: [] },
  ],
}

describe('sumVariantIndex', () => {
  test('returns 0-based index in declaration order', () => {
    expect(sumVariantIndex(Env, 'Idle')).toBe(0)
    expect(sumVariantIndex(Env, 'Decaying')).toBe(1)
    expect(sumVariantIndex(Multi, 'A')).toBe(0)
    expect(sumVariantIndex(Multi, 'B')).toBe(1)
    expect(sumVariantIndex(Multi, 'C')).toBe(2)
  })

  test('returns -1 for unknown variant', () => {
    expect(sumVariantIndex(Env, 'Sustaining')).toBe(-1)
  })
})

describe('sumBundleSlots', () => {
  test('first slot is always the discriminator (int)', () => {
    expect(sumBundleSlots(Env)[0]).toEqual({ suffix: 'tag', scalar: 'int' })
    expect(sumBundleSlots(Multi)[0]).toEqual({ suffix: 'tag', scalar: 'int' })
  })

  test('payload slots follow tag, in (variant, field) declaration order', () => {
    expect(sumBundleSlots(Env)).toEqual([
      { suffix: 'tag', scalar: 'int' },
      { suffix: 'Decaying__level', scalar: 'float', variant: 'Decaying', field: 'level' },
    ])
  })

  test('handles multiple variants with multiple fields', () => {
    expect(sumBundleSlots(Multi)).toEqual([
      { suffix: 'tag', scalar: 'int' },
      { suffix: 'A__x', scalar: 'int', variant: 'A', field: 'x' },
      { suffix: 'A__y', scalar: 'float', variant: 'A', field: 'y' },
      { suffix: 'B__z', scalar: 'bool', variant: 'B', field: 'z' },
    ])
  })

  test('nullary-only sum has only the tag slot', () => {
    const Mode: SumTypeMeta = {
      name: 'Mode',
      variants: [{ name: 'On', payload: [] }, { name: 'Off', payload: [] }],
    }
    expect(sumBundleSlots(Mode)).toEqual([{ suffix: 'tag', scalar: 'int' }])
  })
})

describe('sumVariantField', () => {
  test('returns scalar and suffix for declared (variant, field) pair', () => {
    expect(sumVariantField(Env, 'Decaying', 'level')).toEqual({
      scalar: 'float', suffix: 'Decaying__level',
    })
    expect(sumVariantField(Multi, 'B', 'z')).toEqual({
      scalar: 'bool', suffix: 'B__z',
    })
  })

  test('returns undefined for unknown variant', () => {
    expect(sumVariantField(Env, 'Sustaining', 'level')).toBeUndefined()
  })

  test('returns undefined for unknown field', () => {
    expect(sumVariantField(Env, 'Decaying', 'velocity')).toBeUndefined()
  })

  test('returns undefined for nullary variant', () => {
    expect(sumVariantField(Env, 'Idle', 'level')).toBeUndefined()
  })
})

describe('mangleSumSlot', () => {
  test('joins name and suffix with #', () => {
    expect(mangleSumSlot('state', 'tag')).toBe('state#tag')
    expect(mangleSumSlot('state', 'Decaying__level')).toBe('state#Decaying__level')
  })
})

describe('decodePortTypeDecl with sum registry', () => {
  test('resolves registered sum name to SumType', () => {
    const sumTypes = new Set(['Env'])
    const t = decodePortTypeDecl('Env', undefined, 'test', sumTypes)
    expect(t).toEqual({ tag: 'sum', name: 'Env' })
  })

  test('falls back to StructType when sum registry is empty', () => {
    const t = decodePortTypeDecl('UnknownType', undefined, 'test')
    expect(t).toEqual({ tag: 'struct', name: 'UnknownType' })
  })

  test('falls back to StructType when name not in sum registry', () => {
    const sumTypes = new Set(['OtherSum'])
    const t = decodePortTypeDecl('SomeStruct', undefined, 'test', sumTypes)
    expect(t).toEqual({ tag: 'struct', name: 'SomeStruct' })
  })

  test('built-in scalar names beat sum registry', () => {
    const sumTypes = new Set(['float'])  // pathological case — should still resolve scalar
    expect(decodePortTypeDecl('float', undefined, 'test', sumTypes)).toEqual({
      tag: 'scalar', scalar: 'float',
    })
  })
})

describe('SumTypeDef registration through program loading', () => {
  test('a program with type_defs registers its sum type', async () => {
    const prog = {
      schema: 'tropical_program_2' as const,
      name: 'WithSum',
      ports: {
        inputs: [],
        outputs: [{ name: 'out', type: 'float' as const }],
        type_defs: [{
          kind: 'sum' as const,
          name: 'Env',
          variants: [
            { name: 'Idle', payload: [] },
            { name: 'Decaying', payload: [{ name: 'level', scalar_type: 'float' as const }] },
          ],
        }],
      },
      body: { op: 'block' as const, decls: [], assigns: [
        { op: 'outputAssign' as const, name: 'out', expr: 0 },
      ] },
    }
    const session = makeSession()
    loadProgramAsType(prog, session)
    expect(session.sumTypeRegistry.has('Env')).toBe(true)
    const meta = session.sumTypeRegistry.get('Env')!
    expect(meta.variants).toHaveLength(2)
    expect(meta.variants[0]).toEqual({ name: 'Idle', payload: [] })
    expect(meta.variants[1]).toEqual({
      name: 'Decaying', payload: [{ name: 'level', scalar: 'float' }],
    })
  })
})
