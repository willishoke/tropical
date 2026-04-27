/**
 * lower.ts — bridge from the strict-typed parser AST (`ParsedProgram` in
 * `nodes.ts`) to the legacy ExprNode-shaped `ProgramNode` consumed by the
 * existing JSON loader (`loadProgramAsType` in `compiler/program.ts`).
 *
 * The parser does no scope analysis: every reference to another decl is
 * wrapped in `NameRefNode { op: 'nameRef', name }`. The legacy schema
 * distinguishes references by op tag — `input` / `reg` / `delayRef` /
 * `typeParam` / `param` / `trigger` — driven by which scope the name
 * resolves against. This lowerer performs that scope analysis in a single
 * top-down walk and emits the corresponding legacy ops.
 *
 * Other rewrites:
 *  - Builtin call recognition: `{op:'call', callee: nameRef('clamp'), args}`
 *    becomes `{op:'clamp', args: [...]}` and similar for the other builtin
 *    names. `sampleRate()` and `sampleIndex()` become their nullary leaves.
 *  - InstanceDecl `inputs: InstanceInputEntry[]` and `type_args:
 *    TypeArgEntry[]` collapse into the legacy object form keyed by port /
 *    type-param name.
 *  - `NestedOutNode` field NameRefs unwrap to bare strings.
 *  - `TagNode` and `MatchNode` gain a `type` field naming the owning sum
 *    type, looked up by variant name in scope.
 *  - `LetNode` is straightforward; `let`-bound names enter scope while
 *    lowering the body.
 *  - Combinator nodes (`fold`, `scan`, `generate`, `iterate`, `chain`,
 *    `map2`, `zipWith`) pass through, with their binders pushed onto
 *    scope while lowering body / nested expressions.
 *  - Nested `programDecl`: recurse with a fresh inner scope. Sum types,
 *    structs, and aliases declared in the outer's `ports.type_defs` are
 *    visible inside the inner program (the legacy loader registers them
 *    at the outer level before recursing into nested programs).
 *
 * Pure: no global state, no input mutation. Same input ⇒ same output.
 */

import type {
  ProgramNode as ParsedProgramNode,
  BlockNode as ParsedBlockNode,
  BodyDecl as ParsedBodyDecl,
  BodyAssign as ParsedBodyAssign,
  RegDeclNode as ParsedRegDeclNode,
  DelayDeclNode as ParsedDelayDeclNode,
  ParamDeclNode as ParsedParamDeclNode,
  InstanceDeclNode as ParsedInstanceDeclNode,
  ProgramDeclNode as ParsedProgramDeclNode,
  OutputAssignNode as ParsedOutputAssignNode,
  NextUpdateNode as ParsedNextUpdateNode,
  ParsedExprNode,
  ExprOpNode as ParsedExprOpNode,
  NameRefNode,
  BindingNode as ParsedBindingNode,
  NestedOutNode as ParsedNestedOutNode,
  IndexNode as ParsedIndexNode,
  CallNode as ParsedCallNode,
  LetNode as ParsedLetNode,
  FoldNode as ParsedFoldNode,
  ScanNode as ParsedScanNode,
  GenerateNode as ParsedGenerateNode,
  IterateNode as ParsedIterateNode,
  ChainNode as ParsedChainNode,
  Map2Node as ParsedMap2Node,
  ZipWithNode as ParsedZipWithNode,
  TagNode as ParsedTagNode,
  MatchNode as ParsedMatchNode,
  BinaryOpNode,
  UnaryOpNode,
  ProgramPort as ParsedProgramPort,
  ProgramPortSpec as ParsedProgramPortSpec,
  ProgramPorts as ParsedProgramPorts,
  PortTypeDecl as ParsedPortTypeDecl,
  ShapeDim as ParsedShapeDim,
  TypeDef as ParsedTypeDef,
} from './nodes.js'
import type {
  ProgramNode as LegacyProgramNode,
  ProgramPorts as LegacyProgramPorts,
  ProgramPortSpec as LegacyProgramPortSpec,
  PortTypeDecl as LegacyPortTypeDecl,
  ShapeDim as LegacyShapeDim,
  BlockNode as LegacyBlockNode,
} from '../program.js'
import type { ExprNode as LegacyExprNode } from '../expr.js'
import type { TypeDefJSON as LegacyTypeDefJSON } from '../session.js'

// ─────────────────────────────────────────────────────────────
// Scope
// ─────────────────────────────────────────────────────────────

/** Per-program scope. Type-level info (sumTypes, structTypes, aliases)
 *  inherits from the outer program; value-level info (inputs, regs,
 *  delays, typeParams, params) is fresh per program. Binders are a stack
 *  managed via lexical push/pop while walking. */
interface Scope {
  inputs: Set<string>
  regs: Set<string>
  delays: Set<string>
  typeParams: Set<string>
  params: Set<string>
  triggers: Set<string>
  aliases: Set<string>
  /** variant name → owning sum-type name. */
  sumTypes: Map<string, string>
  structTypes: Set<string>
  /** Set of currently in-scope binder names. Mutable; callers push/pop. */
  binders: Set<string>
}

function emptyScope(): Scope {
  return {
    inputs: new Set(),
    regs: new Set(),
    delays: new Set(),
    typeParams: new Set(),
    params: new Set(),
    triggers: new Set(),
    aliases: new Set(),
    sumTypes: new Map(),
    structTypes: new Set(),
    binders: new Set(),
  }
}

/** Build a fresh scope for a (possibly nested) program. Type-level info is
 *  inherited from `outer` (or empty at the top level); value-level info is
 *  populated from the program's ports + type_params + body decls. */
function buildScope(prog: ParsedProgramNode, outer: Scope | null): Scope {
  const scope = emptyScope()

  if (outer) {
    for (const k of outer.aliases) scope.aliases.add(k)
    for (const [v, t] of outer.sumTypes) scope.sumTypes.set(v, t)
    for (const k of outer.structTypes) scope.structTypes.add(k)
  }

  // Type defs from this program's ports.type_defs
  for (const td of prog.ports?.type_defs ?? []) {
    if (td.kind === 'alias') {
      scope.aliases.add(td.name)
    } else if (td.kind === 'sum') {
      for (const variant of td.variants) {
        scope.sumTypes.set(variant.name, td.name)
      }
    } else if (td.kind === 'struct') {
      scope.structTypes.add(td.name)
    }
  }

  // Type params
  if (prog.type_params) {
    for (const name of Object.keys(prog.type_params)) {
      scope.typeParams.add(name)
    }
  }

  // Inputs
  for (const port of prog.ports?.inputs ?? []) {
    scope.inputs.add(typeof port === 'string' ? port : port.name)
  }

  // Body decls — populate regs / delays / params before any expression walk.
  for (const decl of prog.body.decls) {
    switch (decl.op) {
      case 'regDecl':
        scope.regs.add(decl.name)
        break
      case 'delayDecl':
        scope.delays.add(decl.name)
        break
      case 'paramDecl':
        if (decl.type === 'trigger') scope.triggers.add(decl.name)
        else scope.params.add(decl.name)
        break
      case 'instanceDecl':
      case 'programDecl':
        // Don't introduce expression-position names.
        break
    }
  }

  return scope
}

// ─────────────────────────────────────────────────────────────
// Builtin call recognition
// ─────────────────────────────────────────────────────────────

/** Names that, when used as a call callee, get rewritten to a structured
 *  legacy op carrying the lowered args. `select`, `clamp`, `arraySet`
 *  carry their args; `floatExponent`, `floorDiv`, `ldexp`, `pow`, `sqrt`,
 *  `abs`, `round` likewise. */
const BUILTIN_CALL_OPS: ReadonlySet<string> = new Set([
  'select', 'clamp', 'round', 'ldexp', 'floorDiv', 'pow',
  'sqrt', 'abs', 'floatExponent', 'arraySet',
])

/** Names that are nullary builtins emitting a leaf node. */
const BUILTIN_NULLARY_OPS: ReadonlySet<string> = new Set([
  'sampleRate', 'sampleIndex',
])

// ─────────────────────────────────────────────────────────────
// Lowerer entry point
// ─────────────────────────────────────────────────────────────

/** Lower a parsed `ProgramNode` (every reference is `NameRefNode`) to the
 *  legacy `ProgramNode` shape consumed by `loadProgramAsType`. */
export function lowerProgram(prog: ParsedProgramNode): LegacyProgramNode {
  return lowerProgramWith(prog, null)
}

function lowerProgramWith(
  prog: ParsedProgramNode,
  outerScope: Scope | null,
): LegacyProgramNode {
  const scope = buildScope(prog, outerScope)

  const decls: LegacyExprNode[] = []
  for (const decl of prog.body.decls) {
    decls.push(lowerBodyDecl(decl, scope))
  }

  const assigns: LegacyExprNode[] = []
  for (const assign of prog.body.assigns) {
    assigns.push(lowerBodyAssign(assign, scope))
  }

  const body: LegacyBlockNode = { op: 'block', decls, assigns }

  const out: LegacyProgramNode = {
    op: 'program',
    name: prog.name,
    body,
  }
  if (prog.type_params !== undefined) out.type_params = prog.type_params
  const ports = lowerPorts(prog.ports, scope)
  if (ports !== undefined) out.ports = ports
  if (prog.breaks_cycles === true) out.breaks_cycles = true
  return out
}

// ─────────────────────────────────────────────────────────────
// Ports + type defs
// ─────────────────────────────────────────────────────────────

function lowerPorts(
  ports: ParsedProgramPorts | undefined,
  scope: Scope,
): LegacyProgramPorts | undefined {
  if (ports === undefined) return undefined

  const out: LegacyProgramPorts = {}
  if (ports.inputs !== undefined) {
    out.inputs = ports.inputs.map(p => lowerPort(p, scope))
  }
  if (ports.outputs !== undefined) {
    out.outputs = ports.outputs.map(p => lowerPort(p, scope))
  }
  if (ports.type_defs !== undefined) {
    // Legacy `TypeDefJSON` (compiler/session.ts) keeps alias.base as a
    // bare string; parser's AliasTypeDef carries it as a NameRef. Sum +
    // struct shapes pass through unchanged.
    out.type_defs = ports.type_defs.map(td => lowerTypeDef(td))
  }
  return out
}

function lowerPort(p: ParsedProgramPort, scope: Scope): string | LegacyProgramPortSpec {
  if (typeof p === 'string') return p

  const spec: LegacyProgramPortSpec = { name: p.name }
  if (p.type !== undefined) spec.type = lowerPortType(p.type, scope)
  if (p.default !== undefined) spec.default = lowerExpr(p.default, scope)
  if (p.bounds !== undefined) spec.bounds = p.bounds
  return spec
}

function lowerPortType(pt: ParsedPortTypeDecl, scope: Scope): LegacyPortTypeDecl {
  // Bare scalar / alias / sum / struct name (NameRefNode at the parsed phase).
  if ('op' in pt) return pt.name

  // Array: { kind: 'array', element: NameRef, shape: ShapeDim[] }
  return {
    kind: 'array',
    element: pt.element.name,
    shape: pt.shape.map(d => lowerShapeDim(d, scope)),
  }
}

function lowerShapeDim(d: ParsedShapeDim, scope: Scope): LegacyShapeDim {
  if (typeof d === 'number') return d
  if (!scope.typeParams.has(d.name)) {
    throw new Error(
      `lower: shape-dim '${d.name}' is not a declared type-param of the enclosing program`,
    )
  }
  return { op: 'typeParam', name: d.name }
}

function lowerTypeDef(td: ParsedTypeDef): LegacyTypeDefJSON {
  if (td.kind === 'alias') {
    return {
      kind: 'alias',
      name: td.name,
      base: td.base.name,
      bounds: td.bounds,
    }
  }
  if (td.kind === 'sum') {
    return {
      kind: 'sum',
      name: td.name,
      variants: td.variants.map(v => ({
        name: v.name,
        payload: v.payload.map(f => ({ name: f.name, scalar_type: f.scalar_type })),
      })),
    }
  }
  return {
    kind: 'struct',
    name: td.name,
    fields: td.fields.map(f => ({ name: f.name, scalar_type: f.scalar_type })),
  }
}

// ─────────────────────────────────────────────────────────────
// Body decls
// ─────────────────────────────────────────────────────────────

function lowerBodyDecl(decl: ParsedBodyDecl, scope: Scope): LegacyExprNode {
  switch (decl.op) {
    case 'regDecl':       return lowerRegDecl(decl, scope)
    case 'delayDecl':     return lowerDelayDecl(decl, scope)
    case 'paramDecl':     return lowerParamDecl(decl)
    case 'instanceDecl':  return lowerInstanceDecl(decl, scope)
    case 'programDecl':   return lowerProgramDecl(decl, scope)
  }
}

function lowerRegDecl(d: ParsedRegDeclNode, scope: Scope): LegacyExprNode {
  const out: { op: 'regDecl'; name: string; init: LegacyExprNode; type?: string } = {
    op: 'regDecl',
    name: d.name,
    init: lowerRegInit(d.init, scope),
  }
  if (d.type !== undefined) out.type = d.type.name
  return out as LegacyExprNode
}

/** Reg init has one legacy-only form the parser/printer don't speak as a
 *  general expression: `{zeros: <N>}` (or `{zeros: {typeParam: <name>}}`)
 *  for array-typed registers. The printer raises this to the surface
 *  `zeros(N)`; round-tripping requires the lowerer to detect the special
 *  shape `call(nameRef('zeros'), [...])` in reg-init position and emit the
 *  sugar form, since the legacy session loader expects it as a literal
 *  property bag. Plain scalar inits pass straight through. */
function lowerRegInit(init: ParsedExprNode, scope: Scope): LegacyExprNode {
  if (typeof init === 'object' && init !== null && !Array.isArray(init)
      && init.op === 'call'
      && typeof init.callee === 'object' && init.callee !== null && !Array.isArray(init.callee)
      && init.callee.op === 'nameRef'
      && (init.callee as NameRefNode).name === 'zeros') {
    if (init.args.length !== 1) {
      throw new Error(`lower: zeros(...) reg init must take exactly 1 argument, got ${init.args.length}`)
    }
    const arg = init.args[0]
    // Type-param dimension: emit `{typeParam: <name>}` — the legacy nested sugar.
    if (typeof arg === 'object' && arg !== null && !Array.isArray(arg) && arg.op === 'nameRef') {
      const name = (arg as NameRefNode).name
      if (scope.typeParams.has(name)) {
        return { zeros: { typeParam: name } } as unknown as LegacyExprNode
      }
    }
    // Numeric literal dimension.
    if (typeof arg === 'number') {
      return { zeros: arg } as unknown as LegacyExprNode
    }
    throw new Error(`lower: zeros(...) reg init dimension must be a numeric literal or a type-param identifier`)
  }
  return lowerExpr(init, scope)
}

function lowerDelayDecl(d: ParsedDelayDeclNode, scope: Scope): LegacyExprNode {
  const out: { op: 'delayDecl'; name: string; update: LegacyExprNode; init: LegacyExprNode; type?: string } = {
    op: 'delayDecl',
    name: d.name,
    update: lowerExpr(d.update, scope),
    init: lowerExpr(d.init, scope),
  }
  if (d.type !== undefined) out.type = d.type.name
  return out as LegacyExprNode
}

function lowerParamDecl(d: ParsedParamDeclNode): LegacyExprNode {
  const out: { op: 'paramDecl'; name: string; type: 'param' | 'trigger'; value?: number } = {
    op: 'paramDecl',
    name: d.name,
    type: d.type,
  }
  if (d.value !== undefined) out.value = d.value
  return out as LegacyExprNode
}

function lowerInstanceDecl(d: ParsedInstanceDeclNode, scope: Scope): LegacyExprNode {
  const out: {
    op: 'instanceDecl'
    name: string
    program: string
    type_args?: Record<string, number>
    inputs?: Record<string, LegacyExprNode>
  } = {
    op: 'instanceDecl',
    name: d.name,
    program: d.program.name,
  }
  if (d.type_args !== undefined && d.type_args.length > 0) {
    const ta: Record<string, number> = {}
    for (const entry of d.type_args) ta[entry.param.name] = entry.value
    out.type_args = ta
  }
  if (d.inputs !== undefined && d.inputs.length > 0) {
    const inputs: Record<string, LegacyExprNode> = {}
    for (const entry of d.inputs) {
      inputs[entry.port.name] = lowerExpr(entry.value, scope)
    }
    out.inputs = inputs
  }
  return out as LegacyExprNode
}

function lowerProgramDecl(d: ParsedProgramDeclNode, scope: Scope): LegacyExprNode {
  return {
    op: 'programDecl',
    name: d.name,
    program: lowerProgramWith(d.program, scope),
  } as unknown as LegacyExprNode
}

// ─────────────────────────────────────────────────────────────
// Body assigns
// ─────────────────────────────────────────────────────────────

function lowerBodyAssign(a: ParsedBodyAssign, scope: Scope): LegacyExprNode {
  switch (a.op) {
    case 'outputAssign': return lowerOutputAssign(a, scope)
    case 'nextUpdate':   return lowerNextUpdate(a, scope)
  }
}

function lowerOutputAssign(a: ParsedOutputAssignNode, scope: Scope): LegacyExprNode {
  return {
    op: 'outputAssign',
    name: a.name,
    expr: lowerExpr(a.expr, scope),
  } as LegacyExprNode
}

function lowerNextUpdate(a: ParsedNextUpdateNode, scope: Scope): LegacyExprNode {
  return {
    op: 'nextUpdate',
    target: { kind: a.target.kind, name: a.target.name },
    expr: lowerExpr(a.expr, scope),
  } as LegacyExprNode
}

// ─────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────

function lowerExpr(e: ParsedExprNode, scope: Scope): LegacyExprNode {
  if (typeof e === 'number')  return e
  if (typeof e === 'boolean') return e
  if (Array.isArray(e))       return e.map(x => lowerExpr(x, scope))
  return lowerOpNode(e, scope)
}

function lowerOpNode(node: ParsedExprOpNode, scope: Scope): LegacyExprNode {
  switch (node.op) {
    case 'nameRef':   return resolveNameRef(node, scope)
    case 'binding':   return lowerBinding(node)
    case 'nestedOut': return lowerNestedOut(node)
    case 'index':     return lowerIndex(node, scope)
    case 'call':      return lowerCall(node, scope)
    case 'tag':       return lowerTag(node, scope)
    case 'match':     return lowerMatch(node, scope)
    case 'let':       return lowerLet(node, scope)
    case 'fold':      return lowerFold(node, scope)
    case 'scan':      return lowerScan(node, scope)
    case 'generate':  return lowerGenerate(node, scope)
    case 'iterate':   return lowerIterate(node, scope)
    case 'chain':     return lowerChain(node, scope)
    case 'map2':      return lowerMap2(node, scope)
    case 'zipWith':   return lowerZipWith(node, scope)
    // Binary + unary ops pass through with args lowered.
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt':  case 'lte': case 'gt':  case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
      return lowerBinary(node, scope)
    case 'neg': case 'not': case 'bitNot':
      return lowerUnary(node, scope)
  }
}

function resolveNameRef(node: NameRefNode, scope: Scope): LegacyExprNode {
  const { name } = node

  // Binders win — they're lexically nearest. (The parser already emits
  // BindingNode for in-scope binders; this branch handles the case where
  // a name was emitted as a plain NameRef but a current binder shadows
  // it. In practice the parser doesn't introduce that path, so this is
  // defensive.)
  if (scope.binders.has(name))    return { op: 'binding', name } as LegacyExprNode
  if (scope.inputs.has(name))     return { op: 'input', name } as LegacyExprNode
  if (scope.regs.has(name))       return { op: 'reg', name } as LegacyExprNode
  if (scope.delays.has(name))     return { op: 'delayRef', id: name } as LegacyExprNode
  if (scope.typeParams.has(name)) return { op: 'typeParam', name } as LegacyExprNode
  if (scope.params.has(name))     return { op: 'param', name } as LegacyExprNode
  if (scope.triggers.has(name))   return { op: 'trigger', name } as LegacyExprNode

  throw new Error(
    `lower: unresolved reference '${name}' — not an input, reg, delay, type-param, param, trigger, or binder in scope`,
  )
}

function lowerBinding(node: ParsedBindingNode): LegacyExprNode {
  return { op: 'binding', name: node.name } as LegacyExprNode
}

function lowerNestedOut(node: ParsedNestedOutNode): LegacyExprNode {
  return {
    op: 'nestedOut',
    ref: node.ref.name,
    output: node.output.name,
  } as LegacyExprNode
}

function lowerIndex(node: ParsedIndexNode, scope: Scope): LegacyExprNode {
  return {
    op: 'index',
    args: [lowerExpr(node.args[0], scope), lowerExpr(node.args[1], scope)],
  } as LegacyExprNode
}

function lowerCall(node: ParsedCallNode, scope: Scope): LegacyExprNode {
  // Parser emits all function-call surface as `call(callee, args)`. Only
  // a NameRef callee can be resolved to a builtin; user-typed program
  // names showing up here are an error today (.trop has no free function
  // calls in expression position).
  if (typeof node.callee !== 'object' || node.callee === null || Array.isArray(node.callee)
      || node.callee.op !== 'nameRef') {
    throw new Error(`lower: call with non-NameRef callee is not supported in .trop`)
  }
  const calleeName = node.callee.name

  if (BUILTIN_NULLARY_OPS.has(calleeName)) {
    if (node.args.length !== 0) {
      throw new Error(`lower: builtin '${calleeName}' takes no arguments (got ${node.args.length})`)
    }
    return { op: calleeName } as LegacyExprNode
  }

  if (BUILTIN_CALL_OPS.has(calleeName)) {
    return {
      op: calleeName,
      args: node.args.map(a => lowerExpr(a, scope)),
    } as LegacyExprNode
  }

  throw new Error(
    `lower: call to '${calleeName}' is not a recognized builtin and free function calls are not allowed in .trop expressions`,
  )
}

function lowerTag(node: ParsedTagNode, scope: Scope): LegacyExprNode {
  const variantName = node.variant.name
  const typeName = scope.sumTypes.get(variantName)
  if (typeName === undefined) {
    throw new Error(
      `lower: variant '${variantName}' is not declared in any in-scope sum type`,
    )
  }
  const out: { op: 'tag'; type: string; variant: string; payload?: Record<string, LegacyExprNode> } = {
    op: 'tag',
    type: typeName,
    variant: variantName,
  }
  if (node.payload !== undefined && node.payload.length > 0) {
    const payload: Record<string, LegacyExprNode> = {}
    for (const entry of node.payload) {
      payload[entry.field.name] = lowerExpr(entry.value, scope)
    }
    out.payload = payload
  }
  return out as LegacyExprNode
}

function lowerMatch(node: ParsedMatchNode, scope: Scope): LegacyExprNode {
  if (node.arms.length === 0) {
    throw new Error(`lower: match with no arms`)
  }
  const firstVariant = node.arms[0].variant.name
  const typeName = scope.sumTypes.get(firstVariant)
  if (typeName === undefined) {
    throw new Error(
      `lower: match-arm variant '${firstVariant}' is not declared in any in-scope sum type`,
    )
  }
  // Verify all arms share the owning sum type.
  for (const arm of node.arms) {
    const owner = scope.sumTypes.get(arm.variant.name)
    if (owner !== typeName) {
      throw new Error(
        `lower: match arm '${arm.variant.name}' belongs to '${owner ?? '<unknown>'}' but match was inferred to be of type '${typeName}'`,
      )
    }
  }

  const arms: Record<string, { bind?: string | string[]; body: LegacyExprNode }> = {}
  for (const arm of node.arms) {
    const bindNames: string[] = arm.bind === undefined
      ? []
      : (typeof arm.bind === 'string' ? [arm.bind] : arm.bind)

    const pushed: string[] = []
    for (const bn of bindNames) {
      if (!scope.binders.has(bn)) {
        scope.binders.add(bn)
        pushed.push(bn)
      }
    }
    const body = lowerExpr(arm.body, scope)
    for (const bn of pushed) scope.binders.delete(bn)

    const armOut: { bind?: string | string[]; body: LegacyExprNode } = { body }
    if (arm.bind !== undefined) armOut.bind = arm.bind
    arms[arm.variant.name] = armOut
  }

  return {
    op: 'match',
    type: typeName,
    scrutinee: lowerExpr(node.scrutinee, scope),
    arms,
  } as LegacyExprNode
}

function lowerLet(node: ParsedLetNode, scope: Scope): LegacyExprNode {
  // Sequential let*: each binding's value sees the binders introduced by
  // earlier entries. Push each binder before lowering the next entry's
  // value, then lower the body with all binders in scope. Match the
  // semantics of `lower_arrays.ts:lowerLet`, which the legacy combinator
  // pass eventually applies to this node.
  const bind: Record<string, LegacyExprNode> = {}
  const pushed: string[] = []
  for (const [k, v] of Object.entries(node.bind)) {
    bind[k] = lowerExpr(v, scope)
    if (!scope.binders.has(k)) {
      scope.binders.add(k)
      pushed.push(k)
    }
  }
  const body = lowerExpr(node.in, scope)
  for (const k of pushed) scope.binders.delete(k)

  return { op: 'let', bind, in: body } as LegacyExprNode
}

function lowerFold(node: ParsedFoldNode, scope: Scope): LegacyExprNode {
  const over = lowerExpr(node.over, scope)
  const init = lowerExpr(node.init, scope)
  const body = withBinders(scope, [node.acc_var, node.elem_var], () => lowerExpr(node.body, scope))
  return {
    op: 'fold', over, init,
    acc_var: node.acc_var, elem_var: node.elem_var, body,
  } as LegacyExprNode
}

function lowerScan(node: ParsedScanNode, scope: Scope): LegacyExprNode {
  const over = lowerExpr(node.over, scope)
  const init = lowerExpr(node.init, scope)
  const body = withBinders(scope, [node.acc_var, node.elem_var], () => lowerExpr(node.body, scope))
  return {
    op: 'scan', over, init,
    acc_var: node.acc_var, elem_var: node.elem_var, body,
  } as LegacyExprNode
}

function lowerGenerate(node: ParsedGenerateNode, scope: Scope): LegacyExprNode {
  const count = lowerExpr(node.count, scope)
  const body = withBinders(scope, [node.var], () => lowerExpr(node.body, scope))
  return { op: 'generate', count, var: node.var, body } as LegacyExprNode
}

function lowerIterate(node: ParsedIterateNode, scope: Scope): LegacyExprNode {
  const count = lowerExpr(node.count, scope)
  const init = lowerExpr(node.init, scope)
  const body = withBinders(scope, [node.var], () => lowerExpr(node.body, scope))
  return { op: 'iterate', count, var: node.var, init, body } as LegacyExprNode
}

function lowerChain(node: ParsedChainNode, scope: Scope): LegacyExprNode {
  const count = lowerExpr(node.count, scope)
  const init = lowerExpr(node.init, scope)
  const body = withBinders(scope, [node.var], () => lowerExpr(node.body, scope))
  return { op: 'chain', count, var: node.var, init, body } as LegacyExprNode
}

function lowerMap2(node: ParsedMap2Node, scope: Scope): LegacyExprNode {
  const over = lowerExpr(node.over, scope)
  const body = withBinders(scope, [node.elem_var], () => lowerExpr(node.body, scope))
  return { op: 'map2', over, elem_var: node.elem_var, body } as LegacyExprNode
}

function lowerZipWith(node: ParsedZipWithNode, scope: Scope): LegacyExprNode {
  const a = lowerExpr(node.a, scope)
  const b = lowerExpr(node.b, scope)
  const body = withBinders(scope, [node.x_var, node.y_var], () => lowerExpr(node.body, scope))
  return {
    op: 'zipWith', a, b,
    x_var: node.x_var, y_var: node.y_var, body,
  } as LegacyExprNode
}

function lowerBinary(node: BinaryOpNode, scope: Scope): LegacyExprNode {
  return {
    op: node.op,
    args: [lowerExpr(node.args[0], scope), lowerExpr(node.args[1], scope)],
  } as LegacyExprNode
}

function lowerUnary(node: UnaryOpNode, scope: Scope): LegacyExprNode {
  return {
    op: node.op,
    args: [lowerExpr(node.args[0], scope)],
  } as LegacyExprNode
}

// ─────────────────────────────────────────────────────────────
// Binder stack helper
// ─────────────────────────────────────────────────────────────

/** Push the named binders onto `scope.binders` for the duration of `body`,
 *  then pop them. Names that were already present (collision) are not
 *  re-pushed and not popped — the existing binding remains intact. */
function withBinders<T>(scope: Scope, names: readonly string[], body: () => T): T {
  const pushed: string[] = []
  for (const n of names) {
    if (!scope.binders.has(n)) {
      scope.binders.add(n)
      pushed.push(n)
    }
  }
  try {
    return body()
  } finally {
    for (const n of pushed) scope.binders.delete(n)
  }
}
