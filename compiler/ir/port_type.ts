/**
 * port_type.ts — utilities and constants for the post-strata `PortType`
 * defined in `ir/nodes.ts`.
 *
 * Mirrors the source-level helpers in `compiler/term.ts` but operates on
 * the narrower IR form: `{ kind: 'scalar', scalar }`,
 * `{ kind: 'alias', alias }`, `{ kind: 'array', element, shape }`. No
 * sums, structs, products, or units — those source-level cases have all
 * been lowered by the strata pipeline before any value of this type
 * appears in a post-strata API.
 *
 * Two parallel utilities (here vs `term.ts`) are intentional: each
 * operates on the type system appropriate to its phase. Consumers reading
 * port-type metadata off a `Compiled` should import from here; consumers
 * walking user-authored declarations pre-strata should import from
 * `term.ts`.
 */

import type { PortType, ScalarKind, AliasTypeDef } from './nodes.js'

// ---------- Constants ----------

export const Float: PortType = { kind: 'scalar', scalar: 'float' }
export const Int:   PortType = { kind: 'scalar', scalar: 'int' }
export const Bool:  PortType = { kind: 'scalar', scalar: 'bool' }

// ---------- Constructor ----------

/** Construct a post-strata array port type. Element is a scalar kind, a
 *  scalar PortType (Float/Int/Bool — unwrapped to its scalar kind), or an
 *  alias declaration. Nested arrays are represented by widening the shape
 *  tuple, not by recursive element types. */
export function ArrayType(
  element: ScalarKind | AliasTypeDef | PortType,
  shape: number[],
): PortType {
  let elem: ScalarKind | AliasTypeDef
  if (typeof element === 'string') {
    elem = element
  } else if ('kind' in element) {
    if (element.kind === 'scalar') elem = element.scalar
    else if (element.kind === 'alias') elem = element.alias
    else throw new Error('ArrayType: post-strata arrays cannot have array element type')
  } else {
    elem = element  // AliasTypeDef
  }
  return { kind: 'array', element: elem, shape }
}

// ---------- Equality ----------

export function portTypeEqual(a: PortType, b: PortType): boolean {
  if (a.kind !== b.kind) return false
  switch (a.kind) {
    case 'scalar':
      return a.scalar === (b as typeof a).scalar
    case 'alias':
      return a.alias === (b as typeof a).alias
    case 'array': {
      const ba = b as typeof a
      if (a.shape.length !== ba.shape.length) return false
      for (let i = 0; i < a.shape.length; i++) {
        const da = a.shape[i]
        const db = ba.shape[i]
        // Shapes can carry TypeParamDecl in pre-specialize forms; equality
        // is reference identity for those, integer compare for literals.
        if (da !== db) return false
      }
      return scalarOrAliasEqual(a.element, ba.element)
    }
  }
}

function scalarOrAliasEqual(
  a: ScalarKind | AliasTypeDef,
  b: ScalarKind | AliasTypeDef,
): boolean {
  if (typeof a === 'string') return typeof b === 'string' && a === b
  if (typeof b === 'string') return false
  return a === b
}

// ---------- Display ----------

export function portTypeToString(t: PortType): string {
  switch (t.kind) {
    case 'scalar': return t.scalar
    case 'alias':  return t.alias.name
    case 'array': {
      const elem = typeof t.element === 'string' ? t.element : t.element.name
      const shape = t.shape.map(d => typeof d === 'number' ? String(d) : d.name).join(',')
      return `${elem}[${shape}]`
    }
  }
}
