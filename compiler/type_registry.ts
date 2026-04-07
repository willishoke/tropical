/**
 * type_registry.ts — Source-level type definitions and elaboration.
 *
 * Three layers:
 *   SourceType  — nominal types with named fields/variants (pre-elaboration)
 *   TypeDef     — product/coproduct definitions in the registry
 *   PortType    — structural types in the categorical IR (post-elaboration)
 *
 * The TypeRegistry maps type names to definitions. The elaboration functor
 * (elaborate) resolves named SourceTypes to structural PortTypes, erasing
 * all nominal distinctions. This is safe because the type checker has already
 * proven the wiring is correct before elaboration fires.
 */

import {
  type PortType, type ScalarKind,
  Float, Int, Bool, Unit,
  scalarCount as portScalarCount,
} from './term.js'

// ─────────────────────────────────────────────────────────────
// Source types (pre-elaboration, checked by type checker)
// ─────────────────────────────────────────────────────────────

export type SourceType =
  | { tag: 'scalar'; scalar: ScalarKind }
  | { tag: 'array'; element: SourceType; shape: number[] }
  | { tag: 'named'; name: string }
  | { tag: 'unit' }

export function sourceTypeEqual(a: SourceType, b: SourceType): boolean {
  if (a.tag !== b.tag) return false
  switch (a.tag) {
    case 'scalar':
      return a.scalar === (b as typeof a).scalar
    case 'array': {
      const ba = b as typeof a
      if (a.shape.length !== ba.shape.length) return false
      if (!a.shape.every((d, i) => d === ba.shape[i])) return false
      return sourceTypeEqual(a.element, ba.element)
    }
    case 'named':
      return a.name === (b as typeof a).name  // nominal equality
    case 'unit':
      return true
  }
}

export function sourceTypeToString(t: SourceType): string {
  switch (t.tag) {
    case 'scalar': return t.scalar
    case 'array': return `${sourceTypeToString(t.element)}[${t.shape.join(',')}]`
    case 'named': return t.name
    case 'unit': return 'unit'
  }
}

// ─────────────────────────────────────────────────────────────
// Type definitions
// ─────────────────────────────────────────────────────────────

export interface ProductField {
  name: string
  type: SourceType
}

export interface CoproductVariant {
  name: string
  payloadFields: Array<{ name: string; type: SourceType }>  // empty for nullary
}

export type ProductDef = {
  kind: 'product'
  name: string
  fields: ProductField[]
}

export type CoproductDef = {
  kind: 'coproduct'
  name: string
  variants: CoproductVariant[]
}

export type TypeDef = ProductDef | CoproductDef

// ─────────────────────────────────────────────────────────────
// Type registry
// ─────────────────────────────────────────────────────────────

export class TypeRegistry {
  private defs = new Map<string, TypeDef>()

  register(def: TypeDef): void {
    this.defs.set(def.name, def)
  }

  resolve(name: string): TypeDef | undefined {
    return this.defs.get(name)
  }

  has(name: string): boolean {
    return this.defs.has(name)
  }

  // ── Lowering queries ──

  /** Index of a field within a product type (declaration order). */
  fieldIndex(typeName: string, fieldName: string): number {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'product') throw new Error(`'${typeName}' is not a product type`)
    const idx = def.fields.findIndex(f => f.name === fieldName)
    if (idx === -1) throw new Error(`Product '${typeName}' has no field '${fieldName}'`)
    return idx
  }

  /** Tag value (0-based index) for a variant within a coproduct type. */
  variantTag(typeName: string, variantName: string): number {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'coproduct') throw new Error(`'${typeName}' is not a coproduct type`)
    const idx = def.variants.findIndex(v => v.name === variantName)
    if (idx === -1) throw new Error(`Coproduct '${typeName}' has no variant '${variantName}'`)
    return idx
  }

  /** The payload fields for a specific variant. */
  variantPayloadFields(typeName: string, variantName: string): Array<{ name: string; type: SourceType }> {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'coproduct') throw new Error(`'${typeName}' is not a coproduct type`)
    const v = def.variants.find(v => v.name === variantName)
    if (!v) throw new Error(`Coproduct '${typeName}' has no variant '${variantName}'`)
    return v.payloadFields
  }

  /** All variant names for a coproduct type (in declaration order). */
  variantNames(typeName: string): string[] {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'coproduct') throw new Error(`'${typeName}' is not a coproduct type`)
    return def.variants.map(v => v.name)
  }

  /** All field names for a product type (in declaration order). */
  fieldNames(typeName: string): string[] {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'product') throw new Error(`'${typeName}' is not a product type`)
    return def.fields.map(f => f.name)
  }

  /** Scalar slot count for a coproduct: 1 (tag) + max payload scalar count. */
  coproductSlotCount(typeName: string): number {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'coproduct') throw new Error(`'${typeName}' is not a coproduct type`)
    const maxPayload = def.variants.reduce((max, v) => {
      const payloadCount = v.payloadFields.reduce((sum, f) => sum + this.sourceTypeScalarCount(f.type), 0)
      return Math.max(max, payloadCount)
    }, 0)
    return 1 + maxPayload
  }

  /** Scalar slot count for a product: sum of field scalar counts. */
  productSlotCount(typeName: string): number {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'product') throw new Error(`'${typeName}' is not a product type`)
    return def.fields.reduce((sum, f) => sum + this.sourceTypeScalarCount(f.type), 0)
  }

  /** Scalar count for a SourceType (resolves named types via registry). */
  sourceTypeScalarCount(t: SourceType): number {
    switch (t.tag) {
      case 'scalar': return 1
      case 'unit': return 0
      case 'array': {
        const elemCount = this.sourceTypeScalarCount(t.element)
        return t.shape.reduce((a, b) => a * b, 1) * elemCount
      }
      case 'named': {
        const def = this.defs.get(t.name)
        if (!def) throw new Error(`Unknown type '${t.name}'`)
        if (def.kind === 'product') return this.productSlotCount(t.name)
        return this.coproductSlotCount(t.name)
      }
    }
  }

  /**
   * Field offset within a product's scalar layout.
   * E.g. for Point = {x: float, y: float}, fieldOffset("Point", "y") = 1.
   */
  fieldOffset(typeName: string, fieldName: string): number {
    const def = this.defs.get(typeName)
    if (!def || def.kind !== 'product') throw new Error(`'${typeName}' is not a product type`)
    let offset = 0
    for (const f of def.fields) {
      if (f.name === fieldName) return offset
      offset += this.sourceTypeScalarCount(f.type)
    }
    throw new Error(`Product '${typeName}' has no field '${fieldName}'`)
  }

  // ── Elaboration functor: SourceType → PortType ──

  /** Elaborate a SourceType to a structural PortType, erasing all names. */
  elaborate(t: SourceType): PortType {
    switch (t.tag) {
      case 'scalar':
        return { tag: 'scalar', scalar: t.scalar }
      case 'unit':
        return { tag: 'unit' }
      case 'array':
        return { tag: 'array', element: this.elaborate(t.element), shape: [...t.shape] }
      case 'named': {
        const def = this.defs.get(t.name)
        if (!def) throw new Error(`Cannot elaborate unknown type '${t.name}'`)
        return this.elaborateDef(def)
      }
    }
  }

  /** Elaborate a TypeDef to its structural PortType. */
  private elaborateDef(def: TypeDef): PortType {
    if (def.kind === 'product') {
      const factors = def.fields.map(f => this.elaborate(f.type))
      if (factors.length === 0) return { tag: 'unit' }
      if (factors.length === 1) return factors[0]
      return { tag: 'product', factors }
    }
    // coproduct — each variant's payload fields become a product summand
    const summands = def.variants.map(v => this.elaboratePayloadFields(v.payloadFields))
    if (summands.length === 0) return { tag: 'unit' }
    if (summands.length === 1) return summands[0]
    return { tag: 'coproduct', summands }
  }

  /** Elaborate a variant's payload fields to a structural PortType. */
  private elaboratePayloadFields(fields: Array<{ name: string; type: SourceType }>): PortType {
    if (fields.length === 0) return { tag: 'unit' }
    const factors = fields.map(f => this.elaborate(f.type))
    if (factors.length === 1) return factors[0]
    return { tag: 'product', factors }
  }

  /** Convert a type name to its structural PortType (convenience). */
  toPortType(name: string): PortType {
    return this.elaborate({ tag: 'named', name })
  }
}

// ─────────────────────────────────────────────────────────────
// SourceType from string (for port declarations)
// ─────────────────────────────────────────────────────────────

/**
 * Parse a type string into a SourceType.
 * Scalar types (float, int, bool, unit) → scalar/unit.
 * Array syntax (float[4]) → array.
 * Anything else → named (resolved via registry during elaboration).
 */
export function parseSourceType(s: string): SourceType {
  if (s === 'unit') return { tag: 'unit' }

  const arrayMatch = s.match(/^(\w+)\[([^\]]+)\]$/)
  if (arrayMatch) {
    const element = parseSourceType(arrayMatch[1])
    const shape = arrayMatch[2].split(',').map(d => {
      const n = parseInt(d.trim(), 10)
      if (isNaN(n) || n <= 0) throw new Error(`Invalid array dimension '${d.trim()}' in type '${s}'`)
      return n
    })
    return { tag: 'array', element, shape }
  }

  switch (s) {
    case 'float': return { tag: 'scalar', scalar: 'float' }
    case 'int':   return { tag: 'scalar', scalar: 'int' }
    case 'bool':  return { tag: 'scalar', scalar: 'bool' }
    default:      return { tag: 'named', name: s }
  }
}
