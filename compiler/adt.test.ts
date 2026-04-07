/**
 * adt.test.ts — Tests for algebraic data types: TypeRegistry, expression lowering,
 * and integration through the flatten pipeline.
 */

import { describe, test, expect } from 'bun:test'
import {
  TypeRegistry,
  parseSourceType,
  sourceTypeEqual,
  sourceTypeToString,
  type SourceType,
  type ProductDef,
  type CoproductDef,
} from './type_registry'
import {
  Float, Int, Bool, Unit, CoproductType,
  product, portTypeEqual, portTypeToString, scalarCount,
} from './term'
import { lowerAdtOps } from './lower_adts'
import type { ExprNode } from './expr'

// ─────────────────────────────────────────────────────────────
// TypeRegistry
// ─────────────────────────────────────────────────────────────

describe('TypeRegistry', () => {
  function makeRegistry(): TypeRegistry {
    const reg = new TypeRegistry()

    // Temp = Hot | Cold
    reg.register({
      kind: 'coproduct',
      name: 'Temp',
      variants: [
        { name: 'Hot', payloadFields: [] },
        { name: 'Cold', payloadFields: [] },
      ],
    })

    // NoteEvent = NoteOn(pitch: float, vel: float) | NoteOff(pitch: float)
    reg.register({
      kind: 'coproduct',
      name: 'NoteEvent',
      variants: [
        { name: 'NoteOn', payloadFields: [
          { name: 'pitch', type: { tag: 'scalar', scalar: 'float' } },
          { name: 'vel', type: { tag: 'scalar', scalar: 'float' } },
        ]},
        { name: 'NoteOff', payloadFields: [
          { name: 'pitch', type: { tag: 'scalar', scalar: 'float' } },
        ]},
      ],
    })

    // Point = { x: float, y: float }
    reg.register({
      kind: 'product',
      name: 'Point',
      fields: [
        { name: 'x', type: { tag: 'scalar', scalar: 'float' } },
        { name: 'y', type: { tag: 'scalar', scalar: 'float' } },
      ],
    })

    // RGB = { r: float, g: float, b: float }
    reg.register({
      kind: 'product',
      name: 'RGB',
      fields: [
        { name: 'r', type: { tag: 'scalar', scalar: 'float' } },
        { name: 'g', type: { tag: 'scalar', scalar: 'float' } },
        { name: 'b', type: { tag: 'scalar', scalar: 'float' } },
      ],
    })

    return reg
  }

  test('register and resolve', () => {
    const reg = makeRegistry()
    expect(reg.has('Temp')).toBe(true)
    expect(reg.has('Point')).toBe(true)
    expect(reg.has('Unknown')).toBe(false)

    const temp = reg.resolve('Temp')
    expect(temp?.kind).toBe('coproduct')
    expect(temp?.name).toBe('Temp')

    const point = reg.resolve('Point')
    expect(point?.kind).toBe('product')
    expect(point?.name).toBe('Point')
  })

  test('variantTag', () => {
    const reg = makeRegistry()
    expect(reg.variantTag('Temp', 'Hot')).toBe(0)
    expect(reg.variantTag('Temp', 'Cold')).toBe(1)
    expect(reg.variantTag('NoteEvent', 'NoteOn')).toBe(0)
    expect(reg.variantTag('NoteEvent', 'NoteOff')).toBe(1)
  })

  test('variantTag throws for unknown', () => {
    const reg = makeRegistry()
    expect(() => reg.variantTag('Temp', 'Warm')).toThrow()
    expect(() => reg.variantTag('Unknown', 'X')).toThrow()
  })

  test('fieldIndex', () => {
    const reg = makeRegistry()
    expect(reg.fieldIndex('Point', 'x')).toBe(0)
    expect(reg.fieldIndex('Point', 'y')).toBe(1)
    expect(reg.fieldIndex('RGB', 'r')).toBe(0)
    expect(reg.fieldIndex('RGB', 'g')).toBe(1)
    expect(reg.fieldIndex('RGB', 'b')).toBe(2)
  })

  test('fieldOffset', () => {
    const reg = makeRegistry()
    expect(reg.fieldOffset('Point', 'x')).toBe(0)
    expect(reg.fieldOffset('Point', 'y')).toBe(1)
    expect(reg.fieldOffset('RGB', 'r')).toBe(0)
    expect(reg.fieldOffset('RGB', 'g')).toBe(1)
    expect(reg.fieldOffset('RGB', 'b')).toBe(2)
  })

  test('coproductSlotCount', () => {
    const reg = makeRegistry()
    // Temp = Hot | Cold → 1 (tag only, no payload)
    expect(reg.coproductSlotCount('Temp')).toBe(1)
    // NoteEvent = NoteOn(float, float) | NoteOff(float) → 1 + max(2, 1) = 3
    expect(reg.coproductSlotCount('NoteEvent')).toBe(3)
  })

  test('productSlotCount', () => {
    const reg = makeRegistry()
    expect(reg.productSlotCount('Point')).toBe(2)
    expect(reg.productSlotCount('RGB')).toBe(3)
  })

  test('variantNames', () => {
    const reg = makeRegistry()
    expect(reg.variantNames('Temp')).toEqual(['Hot', 'Cold'])
    expect(reg.variantNames('NoteEvent')).toEqual(['NoteOn', 'NoteOff'])
  })

  test('fieldNames', () => {
    const reg = makeRegistry()
    expect(reg.fieldNames('Point')).toEqual(['x', 'y'])
    expect(reg.fieldNames('RGB')).toEqual(['r', 'g', 'b'])
  })

  // ── Elaboration ──

  test('elaborate product → structural product', () => {
    const reg = makeRegistry()
    const pt = reg.toPortType('Point')
    expect(portTypeEqual(pt, product([Float, Float]))).toBe(true)
  })

  test('elaborate coproduct (nullary) → structural coproduct', () => {
    const reg = makeRegistry()
    const temp = reg.toPortType('Temp')
    expect(portTypeEqual(temp, CoproductType([Unit, Unit]))).toBe(true)
  })

  test('elaborate coproduct (with payloads) → structural coproduct', () => {
    const reg = makeRegistry()
    const ne = reg.toPortType('NoteEvent')
    // NoteOn → product([Float, Float]), NoteOff → Float
    expect(portTypeEqual(ne, CoproductType([product([Float, Float]), Float]))).toBe(true)
  })

  test('scalarCount matches coproductSlotCount', () => {
    const reg = makeRegistry()
    const tempType = reg.toPortType('Temp')
    expect(scalarCount(tempType)).toBe(1)

    const neType = reg.toPortType('NoteEvent')
    expect(scalarCount(neType)).toBe(3)
  })

  test('scalarCount matches productSlotCount', () => {
    const reg = makeRegistry()
    const ptType = reg.toPortType('Point')
    expect(scalarCount(ptType)).toBe(2)

    const rgbType = reg.toPortType('RGB')
    expect(scalarCount(rgbType)).toBe(3)
  })
})

// ─────────────────────────────────────────────────────────────
// SourceType
// ─────────────────────────────────────────────────────────────

describe('SourceType', () => {
  test('parseSourceType scalars', () => {
    const f = parseSourceType('float')
    expect(f).toEqual({ tag: 'scalar', scalar: 'float' })
    expect(parseSourceType('int')).toEqual({ tag: 'scalar', scalar: 'int' })
    expect(parseSourceType('bool')).toEqual({ tag: 'scalar', scalar: 'bool' })
    expect(parseSourceType('unit')).toEqual({ tag: 'unit' })
  })

  test('parseSourceType named', () => {
    expect(parseSourceType('Point')).toEqual({ tag: 'named', name: 'Point' })
    expect(parseSourceType('NoteEvent')).toEqual({ tag: 'named', name: 'NoteEvent' })
  })

  test('parseSourceType array', () => {
    const arr = parseSourceType('float[4]')
    expect(arr).toEqual({ tag: 'array', element: { tag: 'scalar', scalar: 'float' }, shape: [4] })
  })

  test('sourceTypeEqual nominal', () => {
    const a: SourceType = { tag: 'named', name: 'Temp' }
    const b: SourceType = { tag: 'named', name: 'Temp' }
    const c: SourceType = { tag: 'named', name: 'Bool' }
    expect(sourceTypeEqual(a, b)).toBe(true)
    expect(sourceTypeEqual(a, c)).toBe(false)
  })

  test('sourceTypeToString', () => {
    expect(sourceTypeToString({ tag: 'scalar', scalar: 'float' })).toBe('float')
    expect(sourceTypeToString({ tag: 'named', name: 'Point' })).toBe('Point')
    expect(sourceTypeToString({ tag: 'unit' })).toBe('unit')
  })
})

// ─────────────────────────────────────────────────────────────
// ADT lowering
// ─────────────────────────────────────────────────────────────

describe('lowerAdtOps', () => {
  function makeRegistry(): TypeRegistry {
    const reg = new TypeRegistry()
    reg.register({
      kind: 'coproduct',
      name: 'Temp',
      variants: [
        { name: 'Hot', payloadFields: [] },
        { name: 'Cold', payloadFields: [] },
      ],
    })
    reg.register({
      kind: 'coproduct',
      name: 'NoteEvent',
      variants: [
        { name: 'NoteOn', payloadFields: [
          { name: 'pitch', type: { tag: 'scalar', scalar: 'float' } },
          { name: 'vel', type: { tag: 'scalar', scalar: 'float' } },
        ]},
        { name: 'NoteOff', payloadFields: [
          { name: 'pitch', type: { tag: 'scalar', scalar: 'float' } },
        ]},
      ],
    })
    reg.register({
      kind: 'product',
      name: 'Point',
      fields: [
        { name: 'x', type: { tag: 'scalar', scalar: 'float' } },
        { name: 'y', type: { tag: 'scalar', scalar: 'float' } },
      ],
    })
    return reg
  }

  test('construct product → inline array', () => {
    const reg = makeRegistry()
    const node: ExprNode = { op: 'construct', type_name: 'Point', fields: { x: 1.0, y: 2.0 } }
    const result = lowerAdtOps(node, reg)
    expect(result).toEqual([1.0, 2.0])
  })

  test('project product → index', () => {
    const reg = makeRegistry()
    const pointExpr: ExprNode = { op: 'construct', type_name: 'Point', fields: { x: 1.0, y: 2.0 } }
    const projX: ExprNode = { op: 'project', type_name: 'Point', field: 'x', expr: pointExpr }
    const projY: ExprNode = { op: 'project', type_name: 'Point', field: 'y', expr: pointExpr }

    const resultX = lowerAdtOps(projX, reg)
    expect(resultX).toEqual({ op: 'index', args: [[1.0, 2.0], 0] })

    const resultY = lowerAdtOps(projY, reg)
    expect(resultY).toEqual({ op: 'index', args: [[1.0, 2.0], 1] })
  })

  test('inject nullary coproduct → tag scalar', () => {
    const reg = makeRegistry()
    const hot: ExprNode = { op: 'inject', type_name: 'Temp', variant: 'Hot' }
    const cold: ExprNode = { op: 'inject', type_name: 'Temp', variant: 'Cold' }

    expect(lowerAdtOps(hot, reg)).toBe(0)
    expect(lowerAdtOps(cold, reg)).toBe(1)
  })

  test('inject coproduct with payload → Pack([tag, payload..., padding...])', () => {
    const reg = makeRegistry()
    const noteOn: ExprNode = {
      op: 'inject', type_name: 'NoteEvent', variant: 'NoteOn',
      payload: { pitch: 440.0, vel: 0.8 },
    }
    const result = lowerAdtOps(noteOn, reg)
    // NoteEvent slots = 3 (tag + max(2,1) = 3)
    // NoteOn: [tag=0, pitch=440, vel=0.8]
    expect(result).toEqual([0, 440.0, 0.8])

    const noteOff: ExprNode = {
      op: 'inject', type_name: 'NoteEvent', variant: 'NoteOff',
      payload: { pitch: 261.6 },
    }
    const resultOff = lowerAdtOps(noteOff, reg)
    // NoteOff: [tag=1, pitch=261.6, pad=0]
    expect(resultOff).toEqual([1, 261.6, 0])
  })

  test('match nullary coproduct → nested Select', () => {
    const reg = makeRegistry()
    const scrutinee: ExprNode = { op: 'input', id: 0 }
    const node: ExprNode = {
      op: 'match', type_name: 'Temp',
      scrutinee,
      branches: {
        Hot: { body: 1.0 },
        Cold: { body: 0.0 },
      },
    }
    const result = lowerAdtOps(node, reg)
    // Should be: Select(Equal(input(0), 0), 1.0, Select(Equal(input(0), 1), 0.0, 0))
    expect(result).toEqual({
      op: 'select',
      args: [
        { op: 'eq', args: [{ op: 'input', id: 0 }, 0] },
        1.0,
        {
          op: 'select',
          args: [
            { op: 'eq', args: [{ op: 'input', id: 0 }, 1] },
            0.0,
            0,
          ],
        },
      ],
    })
  })

  test('match with payload + bound substitution', () => {
    const reg = makeRegistry()
    const scrutinee: ExprNode = { op: 'input', id: 0 }
    const node: ExprNode = {
      op: 'match', type_name: 'NoteEvent',
      scrutinee,
      branches: {
        NoteOn: {
          bind: ['pitch', 'vel'],
          body: { op: 'mul', args: [{ op: 'bound', name: 'pitch' }, { op: 'bound', name: 'vel' }] },
        },
        NoteOff: {
          bind: ['pitch'],
          body: { op: 'bound', name: 'pitch' },
        },
      },
    }
    const result = lowerAdtOps(node, reg) as Record<string, unknown>

    // tag = Index(scrutinee, 0)
    // NoteOn body: mul(Index(scrutinee, 1), Index(scrutinee, 2))
    // NoteOff body: Index(scrutinee, 1)
    // Result: Select(eq(tag, 0), NoteOn_body, Select(eq(tag, 1), NoteOff_body, 0))
    expect(result.op).toBe('select')
    const args = result.args as ExprNode[]
    // Condition: eq(Index(scrutinee, 0), 0)
    const cond = args[0] as Record<string, unknown>
    expect(cond.op).toBe('eq')
    const condArgs = cond.args as ExprNode[]
    expect(condArgs[0]).toEqual({ op: 'index', args: [{ op: 'input', id: 0 }, 0] })
    expect(condArgs[1]).toBe(0)

    // NoteOn body: mul(Index(scrutinee, 1), Index(scrutinee, 2))
    const noteOnBody = args[1] as Record<string, unknown>
    expect(noteOnBody.op).toBe('mul')
    const mulArgs = noteOnBody.args as ExprNode[]
    expect(mulArgs[0]).toEqual({ op: 'index', args: [{ op: 'input', id: 0 }, 1] })
    expect(mulArgs[1]).toEqual({ op: 'index', args: [{ op: 'input', id: 0 }, 2] })

    // Else branch: Select(eq(tag, 1), NoteOff_body, 0)
    const elseBranch = args[2] as Record<string, unknown>
    expect(elseBranch.op).toBe('select')
    const elseArgs = elseBranch.args as ExprNode[]
    const noteOffBody = elseArgs[1] as Record<string, unknown>
    expect(noteOffBody).toEqual({ op: 'index', args: [{ op: 'input', id: 0 }, 1] })
  })

  test('passthrough for non-ADT ops', () => {
    const reg = makeRegistry()
    const node: ExprNode = { op: 'add', args: [1.0, 2.0] }
    const result = lowerAdtOps(node, reg)
    expect(result).toBe(node)  // identity — unchanged
  })

  test('recursion into nested expressions', () => {
    const reg = makeRegistry()
    const node: ExprNode = {
      op: 'add',
      args: [
        { op: 'inject', type_name: 'Temp', variant: 'Hot' },
        { op: 'inject', type_name: 'Temp', variant: 'Cold' },
      ],
    }
    const result = lowerAdtOps(node, reg) as Record<string, unknown>
    expect(result.op).toBe('add')
    expect((result.args as ExprNode[])[0]).toBe(0)
    expect((result.args as ExprNode[])[1]).toBe(1)
  })

  test('memoization preserves DAG identity', () => {
    const reg = makeRegistry()
    const shared: ExprNode = { op: 'inject', type_name: 'Temp', variant: 'Hot' }
    const node: ExprNode = { op: 'add', args: [shared, shared] }
    const memo = new WeakMap<object, ExprNode>()
    const result = lowerAdtOps(node, reg, memo)
    // Both args should be the same value
    const args = (result as Record<string, unknown>).args as ExprNode[]
    expect(args[0]).toBe(args[1])
  })
})
