/**
 * term.ts — Port types and shape algebra for tropical.
 *
 * PortType describes the type of a signal port (scalar, array, product, etc.).
 * Shape algebra (broadcast, strides, size) follows numpy conventions with
 * static shapes.
 */

// ─────────────────────────────────────────────────────────────
// Port types
// ─────────────────────────────────────────────────────────────

export type ScalarKind = 'float' | 'int' | 'bool'

export type PortType =
  | { tag: 'scalar'; scalar: ScalarKind }
  | { tag: 'array'; element: PortType; shape: number[] }
  | { tag: 'struct'; name: string }
  | { tag: 'sum'; name: string }
  | { tag: 'product'; factors: PortType[] }
  | { tag: 'unit' }

// Constructors
export const ScalarType = (s: ScalarKind): PortType => ({ tag: 'scalar', scalar: s })
export const Float: PortType = ScalarType('float')
export const Int: PortType = ScalarType('int')
export const Bool: PortType = ScalarType('bool')
export const Unit: PortType = { tag: 'unit' }
export const StructType = (name: string): PortType => ({ tag: 'struct', name })
export const SumType = (name: string): PortType => ({ tag: 'sum', name })

/**
 * Construct an array type: element type with a static shape.
 * Shape is a list of dimension sizes, e.g. [4] for a vector, [4,4] for a matrix.
 */
export const ArrayType = (element: PortType, shape: number[]): PortType => {
  if (shape.length === 0) return element // shape [] degenerates to scalar
  return { tag: 'array', element, shape }
}

// ─────────────────────────────────────────────────────────────
// Shape algebra (numpy-style, static shapes)
// ─────────────────────────────────────────────────────────────

/**
 * Broadcast two shapes following numpy rules:
 * 1. Shapes are right-aligned
 * 2. For each dimension pair: equal sizes pass through, size-1 stretches, mismatch is an error
 * Returns the broadcasted shape, or null if incompatible.
 */
export function broadcastShapes(a: number[], b: number[]): number[] | null {
  const rank = Math.max(a.length, b.length)
  const result: number[] = new Array(rank)
  for (let i = 0; i < rank; i++) {
    const da = i < a.length ? a[a.length - 1 - i] : 1
    const db = i < b.length ? b[b.length - 1 - i] : 1
    if (da === db) {
      result[rank - 1 - i] = da
    } else if (da === 1) {
      result[rank - 1 - i] = db
    } else if (db === 1) {
      result[rank - 1 - i] = da
    } else {
      return null // incompatible
    }
  }
  return result
}

/** Compute row-major strides for a shape. */
export function shapeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length)
  let stride = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= shape[i]
  }
  return strides
}

/** Total number of elements in a shape. */
export function shapeSize(shape: number[]): number {
  let size = 1
  for (const d of shape) size *= d
  return size
}

/** Convert multi-dimensional indices to a flat index using row-major strides. */
export function flattenIndex(indices: number[], strides: number[]): number {
  let idx = 0
  for (let i = 0; i < indices.length; i++) idx += indices[i] * strides[i]
  return idx
}

/**
 * Count the total number of scalars in a PortType.
 * Scalars = 1, arrays = product(shape) * scalarCount(element), etc.
 */
export function scalarCount(t: PortType): number {
  switch (t.tag) {
    case 'scalar': return 1
    case 'unit': return 0
    case 'array': return shapeSize(t.shape) * scalarCount(t.element)
    case 'struct': return 1 // opaque — struct scalar count comes from registry
    case 'sum': return 1
    case 'product': return t.factors.reduce((s, f) => s + scalarCount(f), 0)
  }
}

/**
 * Build a product type, flattening nested products and eliminating units.
 * product([A]) = A, product([]) = Unit, product([A, Unit, B]) = product([A, B])
 */
export function product(factors: PortType[]): PortType {
  // Flatten nested products and filter units
  const flat: PortType[] = []
  for (const f of factors) {
    if (f.tag === 'unit') continue
    if (f.tag === 'product') flat.push(...f.factors)
    else flat.push(f)
  }
  if (flat.length === 0) return Unit
  if (flat.length === 1) return flat[0]
  return { tag: 'product', factors: flat }
}

// ─────────────────────────────────────────────────────────────
// Port type equality
// ─────────────────────────────────────────────────────────────

export function portTypeEqual(a: PortType, b: PortType): boolean {
  if (a.tag !== b.tag) return false
  switch (a.tag) {
    case 'scalar':
      return a.scalar === (b as typeof a).scalar
    case 'array': {
      const ba = b as typeof a
      if (a.shape.length !== ba.shape.length) return false
      if (!a.shape.every((d, i) => d === ba.shape[i])) return false
      return portTypeEqual(a.element, ba.element)
    }
    case 'struct':
    case 'sum':
      return a.name === (b as typeof a).name
    case 'product': {
      const bp = b as typeof a
      if (a.factors.length !== bp.factors.length) return false
      return a.factors.every((f, i) => portTypeEqual(f, bp.factors[i]))
    }
    case 'unit':
      return true
  }
}

/**
 * Human-readable string for a port type.
 */
export function portTypeToString(t: PortType): string {
  switch (t.tag) {
    case 'scalar': return t.scalar
    case 'array': return `${portTypeToString(t.element)}[${t.shape.join(',')}]`
    case 'struct': return t.name
    case 'sum': return t.name
    case 'product':
      return t.factors.map(portTypeToString).join(' ⊗ ')
    case 'unit': return 'I'
  }
}

/**
 * Check if two array shapes are broadcast-compatible.
 * Returns the result shape, or null if incompatible.
 */
export function shapesCompatible(a: PortType, b: PortType): number[] | null {
  if (a.tag === 'array' && b.tag === 'array') {
    return broadcastShapes(a.shape, b.shape)
  }
  if (a.tag === 'array' && b.tag === 'scalar') return a.shape  // scalar broadcasts to any array
  if (a.tag === 'scalar' && b.tag === 'array') return b.shape
  if (a.tag === 'scalar' && b.tag === 'scalar') return []      // both scalar
  return null
}

// ─────────────────────────────────────────────────────────────
// Sum type metadata and bundle decomposition
//
// A sum type T = V_1 | V_2{f_a: A, f_b: B} | V_3{...} decomposes at flatten
// time into a bundle of scalar wires:
//   <wire>#tag                  (int) — variant index, 0-based
//   <wire>#V_2__f_a             (A)
//   <wire>#V_2__f_b             (B)
//   <wire>#V_3__...             (...)
// The bundle has one slot per (variant, field) pair plus the discriminator.
// Inactive variants' field slots carry zeros (written by tag/match lowering).
// ─────────────────────────────────────────────────────────────

/** A field in a variant's payload. */
export interface SumVariantField {
  name: string
  scalar: ScalarKind
}

/** A variant of a sum type: a name plus a (possibly empty) payload of named fields. */
export interface SumVariantMeta {
  name: string
  payload: SumVariantField[]
}

/** Structural metadata for a sum type — what term.ts needs to compute bundle layout. */
export interface SumTypeMeta {
  name: string
  variants: SumVariantMeta[]
}

/** A single slot in a sum-typed bundle. */
export interface SumBundleSlot {
  /** Suffix appended to the bundle's logical name to form the slot's mangled name. */
  suffix: string
  /** Scalar type of this slot. The discriminator is `int`; payload slots take their field type. */
  scalar: ScalarKind
  /** For payload slots: the variant they belong to. Undefined for the discriminator. */
  variant?: string
  /** For payload slots: the field name within the variant. Undefined for the discriminator. */
  field?: string
}

/**
 * Look up a variant's index (0-based, in declaration order) within a sum type.
 * Returns -1 if the variant is not declared.
 */
export function sumVariantIndex(meta: SumTypeMeta, variant: string): number {
  return meta.variants.findIndex(v => v.name === variant)
}

/**
 * Compute the full bundle decomposition for a sum-typed wire.
 * Returns an ordered list: [discriminator, then payload slots in (variant, field) order].
 */
export function sumBundleSlots(meta: SumTypeMeta): SumBundleSlot[] {
  const slots: SumBundleSlot[] = [{ suffix: 'tag', scalar: 'int' }]
  for (const variant of meta.variants) {
    for (const field of variant.payload) {
      slots.push({
        suffix: `${variant.name}__${field.name}`,
        scalar: field.scalar,
        variant: variant.name,
        field: field.name,
      })
    }
  }
  return slots
}

/**
 * Look up a specific (variant, field) pair in a sum type, returning its scalar
 * type and the slot suffix used in the bundle. Returns undefined if the variant
 * or field is not declared.
 */
export function sumVariantField(
  meta: SumTypeMeta,
  variant: string,
  field: string,
): { scalar: ScalarKind; suffix: string } | undefined {
  const v = meta.variants.find(x => x.name === variant)
  if (v === undefined) return undefined
  const f = v.payload.find(x => x.name === field)
  if (f === undefined) return undefined
  return { scalar: f.scalar, suffix: `${variant}__${field}` }
}

/**
 * Mangle a logical wire name with a slot suffix using the bundle separator.
 * E.g. mangleSumSlot('state', 'Decaying__level') === 'state#Decaying__level'.
 */
export function mangleSumSlot(name: string, suffix: string): string {
  return `${name}#${suffix}`
}
