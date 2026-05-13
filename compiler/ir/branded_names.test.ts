import { describe, test, expect } from 'bun:test'
import {
  instanceName, portName, rawName,
  childInstance, instancePathParts, isTopLevel, parentOf, leafOf,
  portRef, portRefEq, portRefKey, parsePortRefKey,
  type InstanceName, type PortName, type PortRef,
} from './branded_names.js'

describe('InstanceName / PortName constructors', () => {
  test('round-trip through rawName', () => {
    const n = instanceName('voice1')
    expect(rawName(n)).toBe('voice1')
  })

  test('reject empty string', () => {
    expect(() => instanceName('')).toThrow(/empty string/)
    expect(() => portName('')).toThrow(/empty string/)
  })

  test('PortName rejects dots', () => {
    expect(() => portName('foo.bar')).toThrow(/dots are not allowed/)
  })

  test('InstanceName allows dots (nested paths)', () => {
    const n = instanceName('voice1.env')
    expect(rawName(n)).toBe('voice1.env')
  })
})

describe('InstanceName path operations', () => {
  test('childInstance composes nested paths', () => {
    const parent = instanceName('voice1')
    const child = childInstance(parent, 'env')
    expect(rawName(child)).toBe('voice1.env')
  })

  test('childInstance composes deeply nested paths', () => {
    const root = instanceName('voice1')
    const mid  = childInstance(root, 'filter')
    const leaf = childInstance(mid, 'pole1')
    expect(rawName(leaf)).toBe('voice1.filter.pole1')
  })

  test('childInstance rejects child names with dots', () => {
    const parent = instanceName('voice1')
    expect(() => childInstance(parent, 'env.sub')).toThrow(/must not contain dots/)
  })

  test('childInstance rejects empty child', () => {
    const parent = instanceName('voice1')
    expect(() => childInstance(parent, '')).toThrow(/cannot be empty/)
  })

  test('instancePathParts splits on dots', () => {
    expect(instancePathParts(instanceName('voice1'))).toEqual(['voice1'])
    expect(instancePathParts(instanceName('voice1.env'))).toEqual(['voice1', 'env'])
    expect(instancePathParts(instanceName('a.b.c'))).toEqual(['a', 'b', 'c'])
  })

  test('isTopLevel is true iff no dots', () => {
    expect(isTopLevel(instanceName('voice1'))).toBe(true)
    expect(isTopLevel(instanceName('voice1.env'))).toBe(false)
    expect(isTopLevel(instanceName('a.b.c'))).toBe(false)
  })

  test('parentOf returns undefined for top-level', () => {
    expect(parentOf(instanceName('voice1'))).toBeUndefined()
  })

  test('parentOf strips the last segment', () => {
    expect(rawName(parentOf(instanceName('voice1.env'))!)).toBe('voice1')
    expect(rawName(parentOf(instanceName('a.b.c'))!)).toBe('a.b')
  })

  test('leafOf returns the last segment', () => {
    expect(leafOf(instanceName('voice1'))).toBe('voice1')
    expect(leafOf(instanceName('voice1.env'))).toBe('env')
    expect(leafOf(instanceName('a.b.c'))).toBe('c')
  })

  test('childInstance + parentOf round-trip', () => {
    const parent = instanceName('voice1')
    const child = childInstance(parent, 'env')
    expect(parentOf(child)).toBe(parent)
  })
})

describe('PortRef', () => {
  test('portRef constructs a typed pair', () => {
    const r: PortRef = portRef(instanceName('voice1'), portName('out'))
    expect(rawName(r.instance)).toBe('voice1')
    expect(rawName(r.port)).toBe('out')
  })

  test('portRefEq compares structurally', () => {
    const a = portRef(instanceName('v'), portName('out'))
    const b = portRef(instanceName('v'), portName('out'))
    const c = portRef(instanceName('v'), portName('in'))
    const d = portRef(instanceName('w'), portName('out'))
    expect(portRefEq(a, b)).toBe(true)
    expect(portRefEq(a, c)).toBe(false)
    expect(portRefEq(a, d)).toBe(false)
  })

  test('portRefKey serializes to canonical form', () => {
    const r = portRef(instanceName('voice1'), portName('freq'))
    expect(portRefKey(r)).toBe('voice1:freq')
  })

  test('portRefKey handles nested instance paths', () => {
    const r = portRef(instanceName('voice1.env'), portName('alive'))
    expect(portRefKey(r)).toBe('voice1.env:alive')
  })

  test('parsePortRefKey is inverse of portRefKey', () => {
    const r = portRef(instanceName('voice1.env'), portName('alive'))
    const k = portRefKey(r)
    const back = parsePortRefKey(k)
    expect(portRefEq(r, back)).toBe(true)
  })

  test('parsePortRefKey throws on missing separator', () => {
    expect(() => parsePortRefKey('voice1')).toThrow(/missing ':'/)
  })

  test('parsePortRefKey splits on first colon only', () => {
    // contrived but: instance names can contain dots, but not colons.
    // ports also cannot contain colons. so the first colon is canonical.
    const back = parsePortRefKey('voice1:in')
    expect(rawName(back.instance)).toBe('voice1')
    expect(rawName(back.port)).toBe('in')
  })
})

describe('type-system discipline (compile-only smoke)', () => {
  // These checks are mostly that the types compile in the expected
  // ways. Any line that compiles is a positive test; any line that
  // would have to be commented out to compile demonstrates the brand.

  test('InstanceName and PortName are not assignable to each other', () => {
    const inst = instanceName('voice1')
    const port = portName('out')
    // @ts-expect-error — PortName is not InstanceName
    const _bad1: InstanceName = port
    // @ts-expect-error — InstanceName is not PortName
    const _bad2: PortName = inst
    void _bad1; void _bad2
  })

  test('raw strings are not assignable to branded names', () => {
    // @ts-expect-error — raw string must be branded first
    const _bad: InstanceName = 'voice1'
    void _bad
  })
})
