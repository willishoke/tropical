/**
 * raise.ts — bridge from the legacy ExprNode-shaped `ProgramNode`
 * (`compiler/program.ts`, schema `tropical_program_2`) to the strict-typed
 * parser AST `ParsedProgram` (`compiler/parse/nodes.ts`).
 *
 * This is the categorical inverse of `lower.ts`: every transformation that
 * lower performs is undone here, and `lower(raise(legacy))` recovers the
 * input. Used by the round-trip golden tests in `raise.test.ts` — never on
 * the production runtime path.
 *
 * Mapping summary (op tag → parser shape):
 *   {op:'input',name}          → nameRef(name)
 *   {op:'reg',name}            → nameRef(name)
 *   {op:'delayRef',id}         → nameRef(id)
 *   {op:'typeParam',name}      → nameRef(name)              [expr position]
 *   {op:'param',name}          → nameRef(name)
 *   {op:'trigger',name}        → nameRef(name)
 *   {op:'paramExpr',name}      → nameRef(name)
 *   {op:'triggerParamExpr',n}  → nameRef(name)
 *   {op:'binding',name}        → BindingNode {op:'binding',name}    [pass-through]
 *   {op:'sampleRate'}          → call(nameRef('sampleRate'), [])
 *   {op:'sampleIndex'}         → call(nameRef('sampleIndex'), [])
 *   {op:'select'|'clamp'|...}  → call(nameRef(opname), [...args raised])
 *   {op:'tag',type,variant,..} → TagNode (drop `type`, lift variant + payload entries)
 *   {op:'match',type,arms,..}  → MatchNode (drop `type`, lift arms array)
 *   {op:'nestedOut',ref,output}→ NestedOutNode {ref:nameRef, output:nameRef}
 *   regDecl.type:string        → regDecl.type:NameRef
 *   regDecl.init:{zeros:<N>}   → call(nameRef('zeros'), [N raised])
 *   instanceDecl.inputs:Record → InstanceInputEntry[] (Object.entries order)
 *   instanceDecl.type_args:Rec → TypeArgEntry[] (Object.entries order)
 *   port type: string          → nameRef(string)
 *   port type: array{el,shape} → array{element:nameRef(el),shape:[...]}
 *   shape entry {typeParam,n}  → nameRef(n)
 *   alias.base:string          → nameRef(string)
 *
 * Pure: no global state, no input mutation. No scope tracking is needed —
 * the legacy form is unambiguous (every reference already carries its op).
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
  TagPayloadEntry,
  MatchArmEntry,
  ProgramPort as ParsedProgramPort,
  ProgramPortSpec as ParsedProgramPortSpec,
  ProgramPorts as ParsedProgramPorts,
  PortTypeDecl as ParsedPortTypeDecl,
  ShapeDim as ParsedShapeDim,
  TypeDef as ParsedTypeDef,
  InstanceInputEntry,
  TypeArgEntry,
} from './nodes.js'
import { nameRef } from './nodes.js'
import type {
  ProgramNode as LegacyProgramNode,
  ProgramPorts as LegacyProgramPorts,
  ProgramPortSpec as LegacyProgramPortSpec,
  PortTypeDecl as LegacyPortTypeDecl,
  ShapeDim as LegacyShapeDim,
} from '../program.js'
import type { ExprNode as LegacyExprNode } from '../expr.js'
import type { TypeDefJSON as LegacyTypeDefJSON } from '../session.js'

// ─────────────────────────────────────────────────────────────
// Op classification
// ─────────────────────────────────────────────────────────────

/** Legacy ops that collapse to `nameRef(<the carried name>)`. The
 *  field carrying the name is uniformly `name` except for `delayRef`,
 *  which uses `id` (a leftover from the slottification path). */
const REF_OPS_NAME: ReadonlySet<string> = new Set([
  'input', 'reg', 'typeParam', 'param', 'trigger',
  'paramExpr', 'triggerParamExpr',
])

/** Legacy nullary builtins that raise to `call(nameRef(<op>), [])`. */
const BUILTIN_NULLARY_OPS: ReadonlySet<string> = new Set([
  'sampleRate', 'sampleIndex',
])

/** Legacy n-ary builtins that raise to `call(nameRef(<op>), [...args])`.
 *  Mirrors `lower.ts`'s BUILTIN_CALL_OPS. */
const BUILTIN_CALL_OPS: ReadonlySet<string> = new Set([
  'select', 'clamp', 'round', 'ldexp', 'floorDiv', 'pow',
  'sqrt', 'abs', 'floatExponent', 'arraySet',
])

/** Legacy binary ops that pass through unchanged (parser-shape identical). */
const BINARY_OPS: ReadonlySet<string> = new Set([
  'add', 'sub', 'mul', 'div', 'mod',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'and', 'or',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
])

/** Legacy unary ops that pass through unchanged. */
const UNARY_OPS: ReadonlySet<string> = new Set([
  'neg', 'not', 'bitNot',
])

// ─────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────

/** Raise a legacy ProgramNode (on-disk `tropical_program_2` shape) to a
 *  strict-typed parser ProgramNode. Pure; throws on shapes the function
 *  doesn't recognize. */
export function raiseProgram(legacy: LegacyProgramNode): ParsedProgramNode {
  const decls: ParsedBodyDecl[] = []
  for (const d of legacy.body?.decls ?? []) {
    decls.push(raiseBodyDecl(d))
  }
  const assigns: ParsedBodyAssign[] = []
  for (const a of legacy.body?.assigns ?? []) {
    assigns.push(raiseBodyAssign(a))
  }
  const body: ParsedBlockNode = { op: 'block', decls, assigns }

  const out: ParsedProgramNode = {
    op: 'program',
    name: legacy.name,
    body,
  }
  if (legacy.type_params !== undefined) out.type_params = legacy.type_params
  if (legacy.ports !== undefined) {
    const ports = raisePorts(legacy.ports)
    if (ports !== undefined) out.ports = ports
  }
  if (legacy.breaks_cycles === true) out.breaks_cycles = true
  return out
}

// ─────────────────────────────────────────────────────────────
// Ports + type defs
// ─────────────────────────────────────────────────────────────

function raisePorts(ports: LegacyProgramPorts): ParsedProgramPorts | undefined {
  const out: ParsedProgramPorts = {}
  if (ports.inputs !== undefined) {
    out.inputs = ports.inputs.map(raisePort)
  }
  if (ports.outputs !== undefined) {
    out.outputs = ports.outputs.map(raisePort)
  }
  if (ports.type_defs !== undefined) {
    out.type_defs = ports.type_defs.map(raiseTypeDef)
  }
  return out
}

function raisePort(p: string | LegacyProgramPortSpec): ParsedProgramPort {
  if (typeof p === 'string') return p
  const spec: ParsedProgramPortSpec = { name: p.name }
  if (p.type !== undefined) spec.type = raisePortType(p.type)
  if (p.default !== undefined) spec.default = raiseExpr(p.default)
  if (p.bounds !== undefined) spec.bounds = p.bounds
  return spec
}

function raisePortType(pt: LegacyPortTypeDecl): ParsedPortTypeDecl {
  if (typeof pt === 'string') return nameRef(pt)
  return {
    kind: 'array',
    element: nameRef(pt.element),
    shape: pt.shape.map(raiseShapeDim),
  }
}

function raiseShapeDim(d: LegacyShapeDim): ParsedShapeDim {
  if (typeof d === 'number') return d
  // d is { op: 'typeParam', name }
  return nameRef(d.name)
}

function raiseTypeDef(td: LegacyTypeDefJSON): ParsedTypeDef {
  if (td.kind === 'alias') {
    return {
      kind: 'alias',
      name: td.name,
      base: nameRef(td.base),
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

function raiseBodyDecl(decl: LegacyExprNode): ParsedBodyDecl {
  const obj = asObj(decl, 'body decl')
  switch (obj.op) {
    case 'regDecl':      return raiseRegDecl(obj)
    case 'delayDecl':    return raiseDelayDecl(obj)
    case 'paramDecl':    return raiseParamDecl(obj)
    case 'instanceDecl': return raiseInstanceDecl(obj)
    case 'programDecl':  return raiseProgramDecl(obj)
    default:
      throw new Error(`raise: unknown body decl op '${String(obj.op)}'`)
  }
}

function raiseRegDecl(d: Record<string, unknown>): ParsedRegDeclNode {
  const init = d.init as LegacyExprNode | undefined
  if (init === undefined) {
    throw new Error(`raise: regDecl '${String(d.name)}' missing init`)
  }
  const out: ParsedRegDeclNode = {
    op: 'regDecl',
    name: d.name as string,
    init: raiseRegInit(init),
  }
  if (typeof d.type === 'string') out.type = nameRef(d.type)
  return out
}

/** Reg init is normally an ExprNode, but `Delay.json` uses the legacy
 *  sugar `{zeros: <N>}` (no `op` field) — sometimes paired with the
 *  inner sugar `{typeParam: <name>}` for the dimension. Recognize both
 *  here and raise to a `zeros(N)` builtin call. */
function raiseRegInit(init: LegacyExprNode): ParsedExprNode {
  if (typeof init === 'object' && init !== null && !Array.isArray(init)) {
    const obj = init as unknown as Record<string, unknown>
    if (!('op' in obj) && 'zeros' in obj) {
      const n = obj.zeros as LegacyExprNode
      return {
        op: 'call',
        callee: nameRef('zeros'),
        args: [raiseZerosArg(n)],
      }
    }
  }
  return raiseExpr(init)
}

/** Raise the dimension argument of the `{zeros: <N>}` sugar. The on-disk
 *  form is either a number literal or `{typeParam: <name>}` (no `op`
 *  field — yet another legacy sugar shape used only here). */
function raiseZerosArg(arg: LegacyExprNode): ParsedExprNode {
  if (typeof arg === 'object' && arg !== null && !Array.isArray(arg)) {
    const obj = arg as unknown as Record<string, unknown>
    if (!('op' in obj) && 'typeParam' in obj && typeof obj.typeParam === 'string') {
      return nameRef(obj.typeParam)
    }
  }
  return raiseExpr(arg)
}

function raiseDelayDecl(d: Record<string, unknown>): ParsedDelayDeclNode {
  const out: ParsedDelayDeclNode = {
    op: 'delayDecl',
    name: d.name as string,
    update: raiseExpr(d.update as LegacyExprNode),
    init: raiseExpr(d.init as LegacyExprNode),
  }
  if (typeof d.type === 'string') out.type = nameRef(d.type)
  return out
}

function raiseParamDecl(d: Record<string, unknown>): ParsedParamDeclNode {
  const out: ParsedParamDeclNode = {
    op: 'paramDecl',
    name: d.name as string,
    type: d.type === 'trigger' ? 'trigger' : 'param',
  }
  if (typeof d.value === 'number') out.value = d.value
  return out
}

function raiseInstanceDecl(d: Record<string, unknown>): ParsedInstanceDeclNode {
  const out: ParsedInstanceDeclNode = {
    op: 'instanceDecl',
    name: d.name as string,
    program: nameRef(d.program as string),
  }
  if (d.type_args !== undefined) {
    const ta = d.type_args as Record<string, number>
    const entries: TypeArgEntry[] = []
    for (const [param, value] of Object.entries(ta)) {
      entries.push({ param: nameRef(param), value })
    }
    if (entries.length > 0) out.type_args = entries
  }
  if (d.inputs !== undefined) {
    const ins = d.inputs as Record<string, LegacyExprNode>
    const entries: InstanceInputEntry[] = []
    for (const [port, value] of Object.entries(ins)) {
      entries.push({ port: nameRef(port), value: raiseExpr(value) })
    }
    if (entries.length > 0) out.inputs = entries
  }
  return out
}

function raiseProgramDecl(d: Record<string, unknown>): ParsedProgramDeclNode {
  return {
    op: 'programDecl',
    name: d.name as string,
    program: raiseProgram(d.program as LegacyProgramNode),
  }
}

// ─────────────────────────────────────────────────────────────
// Body assigns
// ─────────────────────────────────────────────────────────────

function raiseBodyAssign(a: LegacyExprNode): ParsedBodyAssign {
  const obj = asObj(a, 'body assign')
  switch (obj.op) {
    case 'outputAssign': return raiseOutputAssign(obj)
    case 'nextUpdate':   return raiseNextUpdate(obj)
    default:
      throw new Error(`raise: unknown body assign op '${String(obj.op)}'`)
  }
}

function raiseOutputAssign(a: Record<string, unknown>): ParsedOutputAssignNode {
  return {
    op: 'outputAssign',
    name: a.name as string,
    expr: raiseExpr(a.expr as LegacyExprNode),
  }
}

function raiseNextUpdate(a: Record<string, unknown>): ParsedNextUpdateNode {
  const target = a.target as { kind: 'reg' | 'delay'; name: string }
  return {
    op: 'nextUpdate',
    target: { kind: target.kind, name: target.name },
    expr: raiseExpr(a.expr as LegacyExprNode),
  }
}

// ─────────────────────────────────────────────────────────────
// Expressions
// ─────────────────────────────────────────────────────────────

function raiseExpr(e: LegacyExprNode): ParsedExprNode {
  if (typeof e === 'number')  return e
  if (typeof e === 'boolean') return e
  if (Array.isArray(e))       return e.map(raiseExpr)
  if (typeof e !== 'object' || e === null) {
    throw new Error(`raise: invalid expr value: ${JSON.stringify(e)}`)
  }
  return raiseOpNode(e as Record<string, unknown>)
}

function raiseOpNode(node: Record<string, unknown>): ParsedExprNode {
  const op = node.op
  if (typeof op !== 'string') {
    throw new Error(`raise: expression object missing 'op' field: ${JSON.stringify(node)}`)
  }

  // ── Reference collapse ────────────────────────────────────
  if (REF_OPS_NAME.has(op)) {
    return nameRef(node.name as string)
  }
  if (op === 'delayRef') {
    return nameRef(node.id as string)
  }
  if (op === 'binding') {
    return { op: 'binding', name: node.name as string }
  }

  // ── Builtin → call ───────────────────────────────────────
  if (BUILTIN_NULLARY_OPS.has(op)) {
    return { op: 'call', callee: nameRef(op), args: [] }
  }
  if (BUILTIN_CALL_OPS.has(op)) {
    const args = (node.args as LegacyExprNode[]).map(raiseExpr)
    return { op: 'call', callee: nameRef(op), args }
  }

  // ── Pass-through binary / unary ──────────────────────────
  if (BINARY_OPS.has(op)) {
    const args = node.args as [LegacyExprNode, LegacyExprNode]
    return { op: op as never, args: [raiseExpr(args[0]), raiseExpr(args[1])] }
  }
  if (UNARY_OPS.has(op)) {
    const args = node.args as [LegacyExprNode]
    return { op: op as never, args: [raiseExpr(args[0])] }
  }

  // ── Structured / ADT ─────────────────────────────────────
  switch (op) {
    case 'nestedOut': return raiseNestedOut(node)
    case 'index':     return raiseIndex(node)
    case 'tag':       return raiseTag(node)
    case 'match':     return raiseMatch(node)
    case 'let':       return raiseLet(node)
    case 'fold':      return raiseFold(node)
    case 'scan':      return raiseScan(node)
    case 'generate':  return raiseGenerate(node)
    case 'iterate':   return raiseIterate(node)
    case 'chain':     return raiseChain(node)
    case 'map2':      return raiseMap2(node)
    case 'zipWith':   return raiseZipWith(node)
    default:
      throw new Error(`raise: unknown expression op '${op}'`)
  }
}

function raiseNestedOut(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'nestedOut',
    ref: nameRef(node.ref as string),
    output: nameRef(String(node.output)),
  }
}

function raiseIndex(node: Record<string, unknown>): ParsedExprNode {
  const args = node.args as [LegacyExprNode, LegacyExprNode]
  return { op: 'index', args: [raiseExpr(args[0]), raiseExpr(args[1])] }
}

function raiseTag(node: Record<string, unknown>): ParsedExprNode {
  const out: { op: 'tag'; variant: ReturnType<typeof nameRef>; payload?: TagPayloadEntry[] } = {
    op: 'tag',
    variant: nameRef(node.variant as string),
  }
  if (node.payload !== undefined) {
    const payload = node.payload as Record<string, LegacyExprNode>
    const entries: TagPayloadEntry[] = []
    for (const [field, value] of Object.entries(payload)) {
      entries.push({ field: nameRef(field), value: raiseExpr(value) })
    }
    if (entries.length > 0) out.payload = entries
  }
  return out
}

function raiseMatch(node: Record<string, unknown>): ParsedExprNode {
  const arms = node.arms as Record<string, { bind?: string | string[]; body: LegacyExprNode }>
  const armEntries: MatchArmEntry[] = []
  for (const [variant, arm] of Object.entries(arms)) {
    const armOut: MatchArmEntry = {
      variant: nameRef(variant),
      body: raiseExpr(arm.body),
    }
    if (arm.bind !== undefined) armOut.bind = arm.bind
    armEntries.push(armOut)
  }
  return {
    op: 'match',
    scrutinee: raiseExpr(node.scrutinee as LegacyExprNode),
    arms: armEntries,
  }
}

function raiseLet(node: Record<string, unknown>): ParsedExprNode {
  const bind: Record<string, ParsedExprNode> = {}
  for (const [k, v] of Object.entries(node.bind as Record<string, LegacyExprNode>)) {
    bind[k] = raiseExpr(v)
  }
  return {
    op: 'let',
    bind,
    in: raiseExpr(node.in as LegacyExprNode),
  }
}

function raiseFold(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'fold',
    over: raiseExpr(node.over as LegacyExprNode),
    init: raiseExpr(node.init as LegacyExprNode),
    acc_var: node.acc_var as string,
    elem_var: node.elem_var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

function raiseScan(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'scan',
    over: raiseExpr(node.over as LegacyExprNode),
    init: raiseExpr(node.init as LegacyExprNode),
    acc_var: node.acc_var as string,
    elem_var: node.elem_var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

function raiseGenerate(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'generate',
    count: raiseExpr(node.count as LegacyExprNode),
    var: node.var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

function raiseIterate(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'iterate',
    count: raiseExpr(node.count as LegacyExprNode),
    init: raiseExpr(node.init as LegacyExprNode),
    var: node.var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

function raiseChain(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'chain',
    count: raiseExpr(node.count as LegacyExprNode),
    init: raiseExpr(node.init as LegacyExprNode),
    var: node.var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

function raiseMap2(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'map2',
    over: raiseExpr(node.over as LegacyExprNode),
    elem_var: node.elem_var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

function raiseZipWith(node: Record<string, unknown>): ParsedExprNode {
  return {
    op: 'zipWith',
    a: raiseExpr(node.a as LegacyExprNode),
    b: raiseExpr(node.b as LegacyExprNode),
    x_var: node.x_var as string,
    y_var: node.y_var as string,
    body: raiseExpr(node.body as LegacyExprNode),
  }
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function asObj(node: LegacyExprNode, ctx: string): Record<string, unknown> {
  if (typeof node !== 'object' || node === null || Array.isArray(node)) {
    throw new Error(`raise: ${ctx} must be an object, got ${JSON.stringify(node)}`)
  }
  return node as Record<string, unknown>
}
