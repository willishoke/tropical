/**
 * specialize.ts — Monomorphization of type-parameterized ProgramJSON.
 *
 * A program declares `type_params` (e.g. { N: { type: 'int', default: 44100 } }).
 * An instance supplies `type_args` (e.g. { N: 8 } or, in nested contexts,
 * { N: { op: 'type_param', name: 'N' } } to forward from the outer frame).
 *
 * `specializeProgramJSON` clones the template and substitutes every
 * `{ op: 'type_param', name }` ExprNode with the integer literal, and every
 * `{ zeros: { type_param: name } }` reg value with the resolved integer count.
 * The result is a normal ProgramJSON that `loadProgramDef` can consume
 * without any awareness of type parameters.
 */
import type { ExprNode } from './expr.js'
import type { ProgramJSON } from './program.js'

export type TypeArgValue = number | ExprNode

/** Raw args as they appear on an instance entry (may contain type_param refs). */
export type RawTypeArgs = Record<string, TypeArgValue>

/** Fully resolved args — concrete integers only. Cacheable. */
export type ResolvedTypeArgs = Record<string, number>

/**
 * Resolve raw type_args against the surrounding (outer) program's resolved args.
 * Numeric literals pass through. `{ op: 'type_param', name }` looks up in outerArgs.
 * Throws on unresolved refs, non-integer values, or unknown param names.
 */
export function resolveTypeArgs(
  rawArgs: RawTypeArgs | undefined,
  outerArgs: ResolvedTypeArgs | undefined,
  typeParams: Record<string, { type: 'int'; default?: number }> | undefined,
  contextName: string,
): ResolvedTypeArgs {
  const params = typeParams ?? {}
  const raw = rawArgs ?? {}

  for (const key of Object.keys(raw)) {
    if (!(key in params)) {
      throw new Error(`${contextName}: unknown type_arg '${key}'. Declared: ${Object.keys(params).join(', ') || '(none)'}`)
    }
  }

  const resolved: ResolvedTypeArgs = {}
  for (const [name, spec] of Object.entries(params)) {
    if (name in raw) {
      const v = raw[name]
      const n = resolveValue(v, outerArgs, `${contextName}.type_args.${name}`)
      if (!Number.isInteger(n)) {
        throw new Error(`${contextName}: type_arg '${name}' must be an integer, got ${n}`)
      }
      resolved[name] = n
    } else if (spec.default !== undefined) {
      resolved[name] = spec.default
    } else {
      throw new Error(`${contextName}: missing required type_arg '${name}' (no default)`)
    }
  }
  return resolved
}

function resolveValue(
  v: TypeArgValue,
  outerArgs: ResolvedTypeArgs | undefined,
  context: string,
): number {
  if (typeof v === 'number') return v
  if (v && typeof v === 'object' && !Array.isArray(v) && (v as { op?: string }).op === 'type_param') {
    const name = (v as unknown as { name: string }).name
    if (!outerArgs || !(name in outerArgs)) {
      throw new Error(`${context}: unresolved type_param '${name}' (no outer frame provides it)`)
    }
    return outerArgs[name]
  }
  throw new Error(`${context}: type_arg value must be a number or { op: 'type_param', name }, got ${JSON.stringify(v)}`)
}

/** Build a stable cache key for a specialization. */
export function specializationCacheKey(typeName: string, args: ResolvedTypeArgs): string {
  const sorted = Object.keys(args).sort().map(k => `${k}=${args[k]}`).join(',')
  return `${typeName}<${sorted}>`
}

/**
 * Deep-clone a ProgramJSON and substitute all type_param refs with their
 * resolved integer values. The output contains no `type_param` nodes and
 * no `{ zeros: { type_param } }` forms — it is a plain ProgramJSON ready
 * for `loadProgramDef`.
 *
 * The top-level `type_params` declaration is retained on the clone so the
 * cache can inspect it; `loadProgramDef` ignores unknown fields.
 */
export function specializeProgramJSON(
  prog: ProgramJSON,
  args: ResolvedTypeArgs,
): ProgramJSON {
  const clone = structuredClone(prog) as ProgramJSON

  // regs: { foo: { zeros: { type_param: 'N' } } } → { foo: { zeros: 44100 } }
  if (clone.regs) {
    for (const [name, val] of Object.entries(clone.regs)) {
      if (val && typeof val === 'object' && !Array.isArray(val) && 'zeros' in val) {
        const zeros = (val as { zeros: unknown }).zeros
        if (zeros && typeof zeros === 'object' && !Array.isArray(zeros) && 'type_param' in (zeros as Record<string, unknown>)) {
          const paramName = (zeros as { type_param: string }).type_param
          if (!(paramName in args)) {
            throw new Error(`${prog.name}: reg '${name}' references undeclared type_param '${paramName}'`)
          }
          ;(clone.regs as Record<string, unknown>)[name] = { zeros: args[paramName] }
        }
      }
      // Typed reg: { init: ..., type: <structured-array> } — substitute shape.
      if (val && typeof val === 'object' && !Array.isArray(val) && 'type' in val) {
        const typed = val as { init: unknown; type: unknown }
        typed.type = substituteTypeInDecl(typed.type, args, prog.name, `reg '${name}'`)
      }
    }
  }

  // Port type declarations on inputs/outputs
  if (clone.inputs) {
    clone.inputs = clone.inputs.map(i => substituteTypeOnPortSpec(i, args, prog.name, 'input')) as typeof clone.inputs
  }
  if (clone.outputs) {
    clone.outputs = clone.outputs.map(o => substituteTypeOnPortSpec(o, args, prog.name, 'output')) as typeof clone.outputs
  }

  // ExprNode-bearing fields
  const substNode = (n: ExprNode): ExprNode => substituteTypeParams(n, args, prog.name)

  if (clone.process) {
    const out: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(clone.process.outputs)) out[k] = substNode(v)
    clone.process.outputs = out
    if (clone.process.next_regs) {
      const nr: Record<string, ExprNode> = {}
      for (const [k, v] of Object.entries(clone.process.next_regs)) nr[k] = substNode(v)
      clone.process.next_regs = nr
    }
  }

  if (clone.delays) {
    for (const [k, d] of Object.entries(clone.delays)) {
      d.update = substNode(d.update)
    }
  }

  if (clone.input_defaults) {
    const defs: Record<string, ExprNode> = {}
    for (const [k, v] of Object.entries(clone.input_defaults)) defs[k] = substNode(v)
    clone.input_defaults = defs
  }

  // instances: substitute type_args and input expressions in this frame
  if (clone.instances) {
    for (const [alias, spec] of Object.entries(clone.instances)) {
      if (spec.inputs) {
        const newInputs: Record<string, ExprNode> = {}
        for (const [k, v] of Object.entries(spec.inputs)) newInputs[k] = substNode(v)
        spec.inputs = newInputs
      }
      // type_args that reference outer type_params get resolved to integers;
      // numeric args stay numeric. Downstream resolution will call resolveTypeArgs again.
      const ta = (spec as { type_args?: RawTypeArgs }).type_args
      if (ta) {
        const resolved: RawTypeArgs = {}
        for (const [k, v] of Object.entries(ta)) {
          if (typeof v === 'number') {
            resolved[k] = v
          } else if (v && typeof v === 'object' && !Array.isArray(v) && (v as { op?: string }).op === 'type_param') {
            const pn = (v as unknown as { name: string }).name
            if (!(pn in args)) {
              throw new Error(`${prog.name}: instance '${alias}' forwards unknown type_param '${pn}'`)
            }
            resolved[k] = args[pn]
          } else {
            resolved[k] = v
          }
        }
        ;(spec as { type_args?: RawTypeArgs }).type_args = resolved
      }
    }
  }

  // Inline subprograms are left alone — they're independent type definitions.
  // If an outer program references one as an instance, specialization happens
  // when that nested instance is resolved.

  return clone
}

/** Substitute {op:'type_param',name} refs inside a port-type declaration's
 *  shape array with their resolved integer values. Scalars pass through. */
function substituteTypeInDecl(
  t: unknown,
  args: ResolvedTypeArgs,
  progName: string,
  context: string,
): unknown {
  if (t === undefined || typeof t === 'string') return t
  if (t && typeof t === 'object' && !Array.isArray(t)) {
    const o = t as { kind?: string; shape?: unknown[]; element?: string }
    if (o.kind === 'array' && Array.isArray(o.shape)) {
      const newShape = o.shape.map((dim, i) => {
        if (typeof dim === 'number') return dim
        if (dim && typeof dim === 'object' && (dim as { op?: string }).op === 'type_param') {
          const name = (dim as { name: string }).name
          if (!(name in args)) {
            throw new Error(
              `${progName}: ${context} type shape[${i}] references undeclared type_param '${name}'`,
            )
          }
          return args[name]
        }
        return dim
      })
      return { ...o, shape: newShape }
    }
  }
  return t
}

function substituteTypeOnPortSpec(
  spec: unknown,
  args: ResolvedTypeArgs,
  progName: string,
  kind: string,
): unknown {
  if (typeof spec === 'string') return spec
  if (spec && typeof spec === 'object') {
    const o = spec as { name: string; type?: unknown }
    if (o.type !== undefined) {
      return { ...o, type: substituteTypeInDecl(o.type, args, progName, `${kind} '${o.name}'`) }
    }
  }
  return spec
}

function substituteTypeParams(
  node: ExprNode,
  args: ResolvedTypeArgs,
  progName: string,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteTypeParams(n, args, progName))

  const obj = node as Record<string, unknown>
  if (obj.op === 'type_param') {
    const name = obj.name as string
    if (!(name in args)) {
      throw new Error(`${progName}: ExprNode references undeclared type_param '${name}'`)
    }
    return args[name]
  }

  // Walk structural children. ExprNode carries children in several conventions:
  // args (most ops), items (array), bind/in (let), init/body/over/a/b (combinators),
  // fields/payload/branches/scrutinee/struct_expr (ADT ops).
  const out: Record<string, unknown> = { ...obj }
  const recurseNodeField = (key: string) => {
    if (out[key] !== undefined) out[key] = substituteTypeParams(out[key] as ExprNode, args, progName)
  }
  const recurseArrayField = (key: string) => {
    if (Array.isArray(out[key])) {
      out[key] = (out[key] as ExprNode[]).map(n => substituteTypeParams(n, args, progName))
    }
  }
  const recurseRecordField = (key: string) => {
    if (out[key] && typeof out[key] === 'object' && !Array.isArray(out[key])) {
      const rec: Record<string, ExprNode> = {}
      for (const [k, v] of Object.entries(out[key] as Record<string, ExprNode>)) {
        rec[k] = substituteTypeParams(v, args, progName)
      }
      out[key] = rec
    }
  }

  recurseArrayField('args')
  recurseArrayField('items')
  recurseArrayField('fields')
  recurseArrayField('payload')
  recurseArrayField('branches')
  recurseNodeField('init')
  recurseNodeField('body')
  recurseNodeField('over')
  recurseNodeField('in')
  recurseNodeField('a')
  recurseNodeField('b')
  recurseNodeField('n')
  recurseNodeField('scrutinee')
  recurseNodeField('struct_expr')
  recurseRecordField('bind')

  return out as ExprNode
}
