/**
 * morphism_registry.ts — Registry for coercion morphisms between types.
 *
 * A morphism is a named, expression-bodied conversion from one type to another.
 * Example: 'et_realize' converts Z_12 → Float via equal-temperament realization.
 *
 * The compiler consults this registry when it encounters a type mismatch at a
 * composition boundary. If a canonical morphism exists for the type pair, it's
 * automatically inserted into the term. Non-canonical morphisms are available
 * for explicit application.
 */

import type { ExprNode } from './patch'
import type { PortType } from './term'
import { portTypeEqual, portTypeToString } from './term'

// ─────────────────────────────────────────────────────────────
// Morphism definitions
// ─────────────────────────────────────────────────────────────

export interface MorphismDef {
  /** Unique name for this morphism (e.g. 'et_realize', 'round_to_int'). */
  name: string
  /** Source type. */
  fromType: PortType
  /** Destination type. */
  toType: PortType
  /**
   * Expression body. The expression receives a single input (index 0) of
   * type `fromType` and must produce a value of type `toType`.
   */
  body: ExprNode
}

// ─────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────

/** Key for type-pair lookups. */
function pairKey(from: PortType, to: PortType): string {
  return `${portTypeToString(from)} → ${portTypeToString(to)}`
}

export class MorphismRegistry {
  /** All registered morphisms, keyed by name. */
  private _byName = new Map<string, MorphismDef>()

  /** Morphisms indexed by type pair for fast lookup. */
  private _byPair = new Map<string, MorphismDef[]>()

  /** Canonical (auto-inserted) morphism per type pair. */
  private _canonical = new Map<string, string>()

  /**
   * Register a named morphism.
   * Throws if a morphism with the same name already exists.
   */
  register(def: MorphismDef): void {
    if (this._byName.has(def.name)) {
      throw new Error(`Morphism '${def.name}' is already registered.`)
    }
    this._byName.set(def.name, def)

    const key = pairKey(def.fromType, def.toType)
    const list = this._byPair.get(key) ?? []
    list.push(def)
    this._byPair.set(key, list)
  }

  /**
   * Designate a morphism as canonical for its type pair.
   * The canonical morphism is auto-inserted by the compiler at type boundaries.
   */
  setCanonical(name: string): void {
    const def = this._byName.get(name)
    if (!def) throw new Error(`Morphism '${name}' not found.`)
    const key = pairKey(def.fromType, def.toType)
    this._canonical.set(key, name)
  }

  /**
   * Find all registered morphisms from one type to another.
   */
  findMorphisms(from: PortType, to: PortType): MorphismDef[] {
    return this._byPair.get(pairKey(from, to)) ?? []
  }

  /**
   * Find the canonical morphism for a type pair, if one is designated.
   */
  findCanonical(from: PortType, to: PortType): MorphismDef | undefined {
    const key = pairKey(from, to)
    const name = this._canonical.get(key)
    if (name === undefined) return undefined
    return this._byName.get(name)
  }

  /**
   * Look up a morphism by name.
   */
  get(name: string): MorphismDef | undefined {
    return this._byName.get(name)
  }

  /**
   * List all registered morphisms.
   */
  all(): MorphismDef[] {
    return [...this._byName.values()]
  }
}
