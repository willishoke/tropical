/**
 * specialize.ts — Monomorphization of type-parameterized ProgramNode.
 *
 * A program declares `type_params` (e.g. { N: { type: 'int', default: 44100 } }).
 * An instance supplies `type_args` (e.g. { N: 8 } or, in nested contexts,
 * { N: { op: 'type_param', name: 'N' } } to forward from the outer frame).
 *
 * `specializeProgramNode` clones the template and substitutes every
 * `{ op: 'type_param', name }` ExprNode with the integer literal, and every
 * `{ zeros: { type_param: name } }` reg init with the resolved integer count.
 * The result is a ProgramNode that `loadProgramDef` can consume without any
 * awareness of type parameters.
 */
import type { ExprNode } from './expr.js'
import type { ProgramNode } from './program.js'

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
 * Deep-clone a ProgramNode and substitute all type_param refs with their
 * resolved integer values. The output contains no `type_param` nodes and
 * no `{ zeros: { type_param } }` forms — it is a plain ProgramNode ready
 * for `loadProgramDef`.
 *
 * The top-level `type_params` declaration is retained on the clone so the
 * cache can inspect it; `loadProgramDef` ignores it.
 */
export function specializeProgramNode(
  prog: ProgramNode,
  args: ResolvedTypeArgs,
): ProgramNode {
  const clone = structuredClone(prog) as ProgramNode
  const typeParams = (prog.type_params ?? {}) as Record<string, { type?: string }>
  const substNode = (n: ExprNode): ExprNode => substituteTypeParams(n, args, prog.name, typeParams)

  // Ports
  if (clone.ports?.inputs) {
    clone.ports.inputs = clone.ports.inputs.map(
      i => substituteTypeOnInputPortSpec(i, args, prog.name, substNode),
    ) as typeof clone.ports.inputs
  }
  if (clone.ports?.outputs) {
    clone.ports.outputs = clone.ports.outputs.map(
      o => substituteTypeOnPortSpec(o, args, prog.name, 'output'),
    ) as typeof clone.ports.outputs
  }

  // Body decls
  if (clone.body?.decls) {
    clone.body.decls = clone.body.decls.map(d => specializeDecl(d, args, prog.name, substNode))
  }

  // Body assigns
  if (clone.body?.assigns) {
    clone.body.assigns = clone.body.assigns.map(a => specializeAssign(a, substNode))
  }

  return clone
}

/** Substitute type_params inside a single block decl. Inline subprograms
 *  (program_decl) are left alone — they are independent type definitions. */
function specializeDecl(
  rawDecl: ExprNode,
  args: ResolvedTypeArgs,
  progName: string,
  substNode: (n: ExprNode) => ExprNode,
): ExprNode {
  if (typeof rawDecl !== 'object' || rawDecl === null || Array.isArray(rawDecl)) return rawDecl
  const d = rawDecl as Record<string, unknown>
  const op = d.op as string
  const out: Record<string, unknown> = { ...d }

  if (op === 'reg_decl') {
    const init = d.init
    if (init && typeof init === 'object' && !Array.isArray(init) && 'zeros' in (init as Record<string, unknown>)) {
      const zeros = (init as { zeros: unknown }).zeros
      if (zeros && typeof zeros === 'object' && !Array.isArray(zeros) && 'type_param' in (zeros as Record<string, unknown>)) {
        const paramName = (zeros as { type_param: string }).type_param
        if (!(paramName in args)) {
          throw new Error(`${progName}: reg '${String(d.name)}' references undeclared type_param '${paramName}'`)
        }
        out.init = { zeros: args[paramName] }
      }
    } else if (init !== undefined) {
      // Init may itself be an ExprNode (e.g. a typed literal referencing type_param).
      out.init = substNode(init as ExprNode)
    }
    if (d.type !== undefined) {
      out.type = substituteTypeInDecl(d.type, args, progName, `reg '${String(d.name)}'`)
    }
    return out as ExprNode
  }

  if (op === 'delay_decl') {
    if (d.update !== undefined) out.update = substNode(d.update as ExprNode)
    // init is a numeric literal — nothing to substitute
    return out as ExprNode
  }

  if (op === 'instance_decl') {
    if (d.inputs && typeof d.inputs === 'object' && !Array.isArray(d.inputs)) {
      const newInputs: Record<string, ExprNode> = {}
      for (const [k, v] of Object.entries(d.inputs as Record<string, ExprNode>)) {
        newInputs[k] = substNode(v)
      }
      out.inputs = newInputs
    }
    // type_args referencing outer type_params get resolved to integers;
    // numeric args stay numeric. Downstream resolveTypeArgs re-validates.
    if (d.type_args && typeof d.type_args === 'object' && !Array.isArray(d.type_args)) {
      const resolved: RawTypeArgs = {}
      for (const [k, v] of Object.entries(d.type_args as RawTypeArgs)) {
        if (typeof v === 'number') {
          resolved[k] = v
        } else if (v && typeof v === 'object' && !Array.isArray(v) && (v as { op?: string }).op === 'type_param') {
          const pn = (v as unknown as { name: string }).name
          if (!(pn in args)) {
            throw new Error(`${progName}: instance '${String(d.name)}' forwards unknown type_param '${pn}'`)
          }
          resolved[k] = args[pn]
        } else {
          resolved[k] = v
        }
      }
      out.type_args = resolved
    }
    return out as ExprNode
  }

  // program_decl and any future decl forms pass through
  return rawDecl
}

/** Substitute type_params inside a single block assign. */
function specializeAssign(rawAssign: ExprNode, substNode: (n: ExprNode) => ExprNode): ExprNode {
  if (typeof rawAssign !== 'object' || rawAssign === null || Array.isArray(rawAssign)) return rawAssign
  const a = rawAssign as Record<string, unknown>
  const op = a.op as string
  if (op === 'output_assign' || op === 'next_update') {
    return { ...a, expr: substNode(a.expr as ExprNode) } as unknown as ExprNode
  }
  return rawAssign
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

function substituteTypeOnInputPortSpec(
  spec: unknown,
  args: ResolvedTypeArgs,
  progName: string,
  substNode: (n: ExprNode) => ExprNode,
): unknown {
  if (typeof spec === 'string') return spec
  if (spec && typeof spec === 'object') {
    const o = spec as { name: string; type?: unknown; default?: ExprNode }
    const out: { name: string; type?: unknown; default?: ExprNode } = { ...o }
    if (out.type !== undefined) {
      out.type = substituteTypeInDecl(out.type, args, progName, `input '${out.name}'`)
    }
    if (out.default !== undefined) {
      out.default = substNode(out.default)
    }
    return out
  }
  return spec
}

function substituteTypeParams(
  node: ExprNode,
  args: ResolvedTypeArgs,
  progName: string,
  typeParams: Record<string, { type?: string }>,
): ExprNode {
  if (typeof node === 'number' || typeof node === 'boolean') return node
  if (Array.isArray(node)) return node.map(n => substituteTypeParams(n, args, progName, typeParams))

  const obj = node as Record<string, unknown>
  if (obj.op === 'type_param') {
    const name = obj.name as string
    if (!(name in args)) {
      throw new Error(`${progName}: ExprNode references undeclared type_param '${name}'`)
    }
    // The declared type of the type_param drives the scalar kind of the
    // substituted const. Int/bool emit a typed-const ExprNode so the emitter
    // preserves integral semantics (vs. float-by-default for a bare number).
    const declType = typeParams[name]?.type
    if (declType === 'int' || declType === 'bool') {
      return { op: 'const', val: args[name], type: declType }
    }
    return args[name]
  }

  // Walk structural children. ExprNode carries children in several conventions:
  // args (most ops), items (array), bind/in (let), init/body/over/a/b (combinators),
  // fields/payload/branches/scrutinee/struct_expr (ADT ops).
  const out: Record<string, unknown> = { ...obj }
  const recurseNodeField = (key: string) => {
    if (out[key] !== undefined) out[key] = substituteTypeParams(out[key] as ExprNode, args, progName, typeParams)
  }
  const recurseArrayField = (key: string) => {
    if (Array.isArray(out[key])) {
      out[key] = (out[key] as ExprNode[]).map(n => substituteTypeParams(n, args, progName, typeParams))
    }
  }
  const recurseRecordField = (key: string) => {
    if (out[key] && typeof out[key] === 'object' && !Array.isArray(out[key])) {
      const rec: Record<string, ExprNode> = {}
      for (const [k, v] of Object.entries(out[key] as Record<string, ExprNode>)) {
        rec[k] = substituteTypeParams(v, args, progName, typeParams)
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
