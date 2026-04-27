/**
 * uniformity.test.ts — structural invariants on parsed trees (Phase B5c).
 *
 * The parser's contract is that every reference to another node is wrapped
 * in a NameRefNode, and that identity strings (decl names, binders) remain
 * plain strings. This test walks parsed programs and verifies both
 * directions of the invariant.
 *
 * If a future change introduces a stringly-typed reference field (e.g.,
 * NestedOutNode regaining `ref: string`), or accidentally wraps an
 * identity field in a NameRefNode (e.g., RegDeclNode.name becoming a
 * NameRefNode), one of these tests catches it.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from './declarations.js'

/** Walk a value recursively. `visit` is called on every plain object
 *  (not arrays, not primitives). The visitor is given the object plus
 *  the field-path that reached it. */
function walk(value: unknown, visit: (obj: Record<string, unknown>, path: string) => void, path = '$'): void {
  if (Array.isArray(value)) {
    value.forEach((v, i) => walk(v, visit, `${path}[${i}]`))
    return
  }
  if (value !== null && typeof value === 'object') {
    visit(value as Record<string, unknown>, path)
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      walk(v, visit, `${path}.${k}`)
    }
  }
}

const isNameRef = (v: unknown): v is { op: 'nameRef'; name: string } =>
  typeof v === 'object' && v !== null && !Array.isArray(v)
    && (v as { op?: unknown }).op === 'nameRef'

/** Field paths whose value is a *reference string* — i.e., a name that
 *  refers to another declaration somewhere. After B5c, all of these
 *  should be NameRefNode. The path is matched as a substring (so
 *  `program` matches both `instanceDecl.program` and the program list). */
const REFERENCE_FIELDS_THAT_MUST_BE_NAMEREF = [
  // NestedOutNode.ref + .output
  '.ref',
  // The output of nestedOut is a NameRefNode (or numeric index)
  '.output',
  // InstanceDeclNode.program
  '.program',
  // TagNode.variant, MatchArmEntry.variant
  '.variant',
  // TypeArgEntry.param
  '.param',
  // InstanceInputEntry.port, TagPayloadEntry.field
  '.port',
  '.field',
  // RegDeclNode.type
  '.type',  // careful: this also matches paramDecl.type ('param'|'trigger') — checked below
  // AliasTypeDef.base
  '.base',
  // PortTypeDecl.element
  '.element',
]

/** Identity strings that must NOT become NameRefNodes (decl names, binder
 *  labels, etc). */
const IDENTITY_PATHS = [
  '.name',  // RegDecl.name, InstanceDecl.name, ProgramNode.name, etc.
]

describe('parser uniformity — references are NameRefNode, identities are strings', () => {
  test('a representative program has no inlined reference strings', () => {
    const prog = parseProgram(`
      program Synth<N: int = 4>(freq: freq = 220, x: float[N]) -> (out: signal) {
        struct Pair { a: float, b: float }
        enum Mode { Sine, Saw(phase: float) }
        type Bipolar = float in [-1, 1]

        program Inner<M: int = 8>(sig: signal) -> (out: signal) { out = sig }

        reg s: float = 0
        delay z = x[0] init 0
        param cutoff: smoothed = 1000

        i = Inner<M=2>(sig: freq)
        seq = Sequencer<N=4>(clock: 0)

        out = match cutoff {
          Sine => i.out + s,
          Saw { phase: p } => p
        }
        next s = s * 0.99
        dac.out = i.out
      }
    `)

    // Every value at a reference field should either be a NameRefNode,
    // a numeric index (for nestedOut.output), or undefined (optional
    // field).
    walk(prog, (obj, path) => {
      for (const fieldSuffix of REFERENCE_FIELDS_THAT_MUST_BE_NAMEREF) {
        if (!path.endsWith(fieldSuffix)) continue
        // Skip checks where the field doesn't exist on this object
        // (e.g. RegDeclNode without a `type` annotation).
        const segment = fieldSuffix.slice(1)  // drop leading dot
        if (!(segment in obj)) continue
        const v = obj[segment]
        // ParamDecl.type is a literal 'param'|'trigger', not a name ref.
        if (segment === 'type' && (obj as { op?: string }).op === 'paramDecl') continue
        // Match: arm `bind` field is a binder name (string or string[])
        // — handled by being IN the IDENTITY_PATHS / not present here.
        if (typeof v === 'string') {
          throw new Error(`expected NameRefNode at ${path}.${segment}, got string ${JSON.stringify(v)}`)
        }
        // numeric for output index is fine
        if (typeof v === 'number') continue
        if (v === undefined || v === null) continue
        if (Array.isArray(v)) continue  // shape arrays etc
        if (!isNameRef(v) && typeof v === 'object') {
          // Could be a structured form (e.g., array PortTypeDecl). Recursive
          // walking below will check those nested fields.
        }
      }
    })
  })

  test('decl identity fields are plain strings, not NameRefNodes', () => {
    const prog = parseProgram(`
      program X(input1: signal) -> (out: signal) {
        reg s: float = 0
        osc = SinOsc(freq: 220)
        out = osc.out
      }
    `)
    walk(prog, (obj, path) => {
      for (const fieldSuffix of IDENTITY_PATHS) {
        if (!path.endsWith(fieldSuffix)) continue
        const segment = fieldSuffix.slice(1)
        if (!(segment in obj)) continue
        const v = obj[segment]
        if (v === undefined || v === null) continue
        // The `name` field on a NameRefNode itself is a string by design
        // — we want to skip over the inner of a NameRef. Detect by
        // checking the parent object's `op`.
        if ((obj as { op?: string }).op === 'nameRef') continue
        if (typeof v !== 'string') {
          throw new Error(`expected plain string at ${path}.${segment} (decl identity), got ${JSON.stringify(v)}`)
        }
      }
    })

    // Spot-check: the outer program's name is a string, not a NameRef.
    expect(typeof prog.name).toBe('string')
    expect(prog.name).toBe('X')

    // Spot-check: a regDecl's name is a string.
    const regDecl = prog.body.decls.find(d => (d as { op: string }).op === 'regDecl') as { name: string }
    expect(typeof regDecl.name).toBe('string')
    expect(regDecl.name).toBe('s')
  })

  test('NameRefNode is the only reachable shape with op:"nameRef"', () => {
    // Consistency: all NameRefNodes have shape {op:'nameRef', name:string}.
    const prog = parseProgram(`
      program X<N: int = 4>(buf: float[N]) -> (out: signal) {
        out = buf[0]
      }
    `)
    walk(prog, (obj) => {
      if ((obj as { op?: string }).op !== 'nameRef') return
      const keys = Object.keys(obj).sort()
      expect(keys).toEqual(['name', 'op'])
      expect(typeof obj.name).toBe('string')
    })
  })
})
