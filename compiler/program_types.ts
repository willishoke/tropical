/**
 * program_types.ts — Data types for the compiler's internal program representation.
 *
 * ProgramDef is the slot-indexed IR that the flattener consumes.
 * ProgramType and ProgramInstance are thin wrappers used by the type/instance registries.
 */

import type { SignalExpr, ExprCoercible, ExprNode } from './expr.js'
import type { PortType } from './term.js'

// ---------- Value helpers ----------

export type ValueCoercible = boolean | number | number[] | number[][]

/** A register initialiser: either a bare value or { init, type }. */
export type RegInit = ValueCoercible | { init: ValueCoercible; type: string }

// ---------- Bounded types ----------

/** Value bounds for a port: [lo, hi]. null on either side = unbounded. */
export type Bounds = [number | null, number | null]

// ---------- ProgramDef ----------

/**
 * The compiler's internal representation of a program — slot-indexed ExprNode trees
 * ready for the flattener's register allocation. Built from ProgramJSON (name-based)
 * by converting names to integer slot IDs.
 */
export interface ProgramDef {
  typeName: string
  inputNames: string[]
  outputNames: string[]
  inputPortTypes: (PortType | undefined)[]
  outputPortTypes: (PortType | undefined)[]
  registerNames: string[]
  registerPortTypes: (PortType | undefined)[]
  registerInitValues: ValueCoercible[]
  sampleRate: number
  rawInputDefaults: Record<string, ExprNode>
  inputDefaults: (SignalExpr | null)[]
  delayInitValues: number[]
  outputExprNodes: ExprNode[]
  registerExprNodes: (ExprNode | null)[]
  delayUpdateNodes: ExprNode[]
  nestedCalls: NestedCall[]
  breaksCycles: boolean
  inputBounds: (Bounds | null)[]
  outputBounds: (Bounds | null)[]
}

// ---------- NestedCall ----------

/** Captured metadata for a nested program call. */
export interface NestedCall {
  programDef: ProgramDef
  callArgNodes: ExprNode[]
}

// ---------- ProgramType ----------

export class ProgramType {
  readonly _def: ProgramDef

  constructor(def: ProgramDef) {
    this._def = def
  }

  get name(): string { return this._def.typeName }

  /** Instantiate with an explicit instance name. */
  instantiateAs(name: string, opts?: { baseTypeName?: string; typeArgs?: Record<string, number> }): ProgramInstance {
    return new ProgramInstance(this._def, name, opts?.baseTypeName, opts?.typeArgs)
  }
}

// ---------- ProgramInstance ----------

export class ProgramInstance {
  readonly _def: ProgramDef
  readonly name: string
  /** Base (pre-specialization) type name. Equals _def.typeName for non-generic types. */
  readonly baseTypeName: string
  /** Resolved compile-time args if this instance was specialized. */
  readonly typeArgs?: Record<string, number>

  constructor(def: ProgramDef, name: string, baseTypeName?: string, typeArgs?: Record<string, number>) {
    this._def = def
    this.name = name
    this.baseTypeName = baseTypeName ?? def.typeName
    this.typeArgs = typeArgs
  }

  get inputNames(): string[] { return this._def.inputNames }
  get outputNames(): string[] { return this._def.outputNames }
  get registerNames(): string[] { return this._def.registerNames }
  get typeName(): string { return this.baseTypeName }

  inputPortType(idx: number): PortType | undefined { return this._def.inputPortTypes[idx] }
  outputPortType(idx: number): PortType | undefined { return this._def.outputPortTypes[idx] }
  registerPortType(idx: number): PortType | undefined { return this._def.registerPortTypes[idx] }

  inputIndex(name: string): number {
    const idx = this._def.inputNames.indexOf(name)
    if (idx === -1) throw new Error(`Unknown input '${name}' on instance '${this.name}'.`)
    return idx
  }

  outputIndex(name: string): number {
    const idx = this._def.outputNames.indexOf(name)
    if (idx === -1) throw new Error(`Unknown output '${name}' on instance '${this.name}'.`)
    return idx
  }
}
