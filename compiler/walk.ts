/**
 * walk.ts — `mapChildren`: structure-preserving traversal over ExprOpNodeStrict.
 *
 * One shared utility that knows for each op kind which fields are children
 * (positional `args`, named subexpressions, payload values, arm bodies, etc.).
 * All compiler walkers (cloneExpr, inlineCalls, substituteInputs, ...) compose
 * around it instead of reinventing the 25-line generic-iteration pattern.
 *
 * The categorical view: this is the canonical structure-preserving functor
 * over the ExprOpNodeStrict union. Each walker is a natural transformation
 * derived from a per-op intervention plus mapChildren's recursion.
 *
 * Adding a new op variant produces a TypeScript compile error at the
 * `assertNever` default arm of mapChildren — exactly one update site for
 * any op addition.
 */

import type {
  ExprNode,
  ExprOpNodeStrict,
  Op,
  BinaryNode, UnaryNode, TernaryNode,
  ReshapeNode, TransposeNode, SliceNode, ReduceNode,
  BroadcastToNode, IndexNode, MatmulNode, MapNode,
  ZerosNode, OnesNode, FillNode, ArrayLiteralNode, MatrixNode, ArrayNode,
  TagNode, MatchNode, MatchArmStrict, LetNode, FunctionNode, CallNode,
  DelayNode, SourceTagNode,
  GenerateNode, IterateNode, FoldNode, ScanNode,
  Map2Node, ZipWithNode, ChainNode, StrConcatNode, GenerateDeclsNode,
  LeafNode,
  RefNode, InstanceDeclNode, RegDeclNode, DelayDeclNode, ProgramDeclNode,
  OutputAssignNode, NextUpdateNode, ProgramBlockNode, ProgramOpNode,
} from './expr.js'

// ─────────────────────────────────────────────────────────────
// Type guards for the Op<N> family
// ─────────────────────────────────────────────────────────────

const BINARY_TAGS: ReadonlySet<string> = new Set([
  'add', 'sub', 'mul', 'div', 'mod', 'floorDiv', 'ldexp', 'pow',
  'lt', 'lte', 'gt', 'gte', 'eq', 'neq',
  'bitAnd', 'bitOr', 'bitXor', 'lshift', 'rshift',
  'and', 'or',
])

const UNARY_TAGS: ReadonlySet<string> = new Set([
  'neg', 'abs', 'sqrt', 'floor', 'ceil', 'round',
  'floatExponent', 'not', 'bitNot',
  'toInt', 'toBool', 'toFloat',
])

const TERNARY_TAGS: ReadonlySet<string> = new Set(['select', 'clamp', 'arraySet'])

/** True when the op is a member of the Op<N> family — args is a flat
 *  ExprNode tuple and traversal is purely positional. */
function isOpNFamily(node: ExprOpNodeStrict): node is BinaryNode | UnaryNode | TernaryNode {
  return BINARY_TAGS.has(node.op) || UNARY_TAGS.has(node.op) || TERNARY_TAGS.has(node.op)
}

// ─────────────────────────────────────────────────────────────
// Small helpers for Record/Array field traversal
// ─────────────────────────────────────────────────────────────

/** Map each value of a Record<string, V> through f. Returns the original
 *  record if no value changed (preserves reference identity for memoization). */
export function mapValues<V>(rec: Record<string, V>, f: (v: V) => V): Record<string, V> {
  let changed = false
  const out: Record<string, V> = {}
  for (const [k, v] of Object.entries(rec)) {
    const nv = f(v)
    if (nv !== v) changed = true
    out[k] = nv
  }
  return changed ? out : rec
}

/** Map each match arm's body through f. Bind names are leaf strings and
 *  pass through unchanged. */
export function mapArms(
  arms: Record<string, MatchArmStrict>,
  f: (e: ExprNode) => ExprNode,
): Record<string, MatchArmStrict> {
  let changed = false
  const out: Record<string, MatchArmStrict> = {}
  for (const [variant, arm] of Object.entries(arms)) {
    const nb = f(arm.body)
    if (nb !== arm.body) changed = true
    out[variant] = arm.bind === undefined ? { body: nb } : { bind: arm.bind, body: nb }
  }
  return changed ? out : arms
}

// ─────────────────────────────────────────────────────────────
// mapChildren: the canonical structure-preserving traversal
// ─────────────────────────────────────────────────────────────

/**
 * Apply `f` to every direct ExprNode child of `node`, returning a new node
 * with substituted children. Structure-preserving — the result has the same
 * op kind. Reference identity is preserved when no child changes.
 *
 * The internal `assertNever` default arm is the safety net: adding a new op
 * to ExprOpNodeStrict without updating this function produces a TypeScript
 * compile error at the default arm. This is what makes the closed union
 * load-bearing.
 *
 * Ops with non-child fields (shape, axis, reduce_op, etc.) pass those
 * fields through unchanged — they are not children to recurse into.
 */
export function mapChildren<T extends ExprOpNodeStrict>(
  node: T,
  f: (child: ExprNode) => ExprNode,
): T {
  // ── Op<N> family: positional-args traversal handles ~45 ops in one branch.
  if (isOpNFamily(node)) {
    const newArgs = (node.args as ExprNode[]).map(f)
    const same = newArgs.length === node.args.length
                && newArgs.every((n, i) => n === (node.args as ExprNode[])[i])
    if (same) return node
    return { ...node, args: newArgs as unknown as typeof node.args }
  }

  // ── Op<N> + extras and named-children ops (per-op cases).
  switch (node.op) {
    // Op<N> + extras: same args traversal, extras pass through.
    case 'reshape':
    case 'transpose':
    case 'slice':
    case 'reduce':
    case 'broadcastTo':
    case 'index':
    case 'matmul': {
      const n = node as ReshapeNode | TransposeNode | SliceNode | ReduceNode
                       | BroadcastToNode | IndexNode | MatmulNode
      const newArgs = (n.args as ExprNode[]).map(f)
      const same = newArgs.every((c, i) => c === (n.args as ExprNode[])[i])
      if (same) return node
      return { ...node, args: newArgs as unknown as typeof n.args } as T
    }
    case 'map': {
      const n = node as MapNode
      const newCallee = f(n.callee)
      const newArg = f(n.args[0])
      if (newCallee === n.callee && newArg === n.args[0]) return node
      return { ...n, callee: newCallee, args: [newArg] } as unknown as T
    }

    // ── Inline array (variadic, but uses `items` not `args`) ──
    case 'array': {
      const n = node as ArrayNode
      const newItems = n.items.map(f)
      const same = newItems.every((c, i) => c === n.items[i])
      return same ? node : ({ ...n, items: newItems } as unknown as T)
    }

    // ── Construction ops ───────────────────────────────────────
    case 'zeros':
    case 'ones':
    case 'matrix':
      return node  // no ExprNode children
    case 'fill': {
      const n = node as FillNode
      const newValue = f(n.value)
      return newValue === n.value ? node : ({ ...n, value: newValue } as unknown as T)
    }
    case 'arrayLiteral': {
      const n = node as ArrayLiteralNode
      const newValues = n.values.map(f)
      const same = newValues.every((c, i) => c === n.values[i])
      return same ? node : ({ ...n, values: newValues } as unknown as T)
    }

    // ── Named-children ops ─────────────────────────────────────
    case 'tag': {
      const n = node as TagNode
      if (n.payload === undefined) return node
      const newPayload = mapValues(n.payload, f)
      return newPayload === n.payload ? node : ({ ...n, payload: newPayload } as unknown as T)
    }
    case 'match': {
      const n = node as MatchNode
      const newScrutinee = f(n.scrutinee)
      const newArms = mapArms(n.arms, f)
      if (newScrutinee === n.scrutinee && newArms === n.arms) return node
      return { ...n, scrutinee: newScrutinee, arms: newArms } as unknown as T
    }
    case 'let': {
      const n = node as LetNode
      const newBind = mapValues(n.bind, f)
      const newIn = f(n.in)
      if (newBind === n.bind && newIn === n.in) return node
      return { ...n, bind: newBind, in: newIn } as unknown as T
    }
    case 'function': {
      const n = node as FunctionNode
      const newBody = f(n.body)
      return newBody === n.body ? node : ({ ...n, body: newBody } as unknown as T)
    }
    case 'call': {
      const n = node as CallNode
      const newCallee = f(n.callee)
      const newArgs = n.args.map(f)
      const argsSame = newArgs.every((c, i) => c === n.args[i])
      if (newCallee === n.callee && argsSame) return node
      return { ...n, callee: newCallee, args: newArgs } as unknown as T
    }
    case 'delay': {
      const n = node as DelayNode
      const newArg = f(n.args[0])
      return newArg === n.args[0] ? node : ({ ...n, args: [newArg] } as unknown as T)
    }
    case 'sourceTag': {
      const n = node as SourceTagNode
      const newGate = f(n.gate_expr)
      const newExpr = f(n.expr)
      const newOnSkip = n.on_skip === undefined ? undefined : f(n.on_skip)
      if (newGate === n.gate_expr && newExpr === n.expr
          && (n.on_skip === undefined || newOnSkip === n.on_skip)) return node
      return {
        ...n,
        gate_expr: newGate,
        expr: newExpr,
        ...(n.on_skip === undefined ? {} : { on_skip: newOnSkip }),
      } as unknown as T
    }

    // Combinators
    case 'generate': {
      const n = node as GenerateNode
      const newBody = f(n.body)
      return newBody === n.body ? node : ({ ...n, body: newBody } as unknown as T)
    }
    case 'chain': {
      const n = node as ChainNode
      const newInit = f(n.init)
      const newBody = f(n.body)
      if (newInit === n.init && newBody === n.body) return node
      return { ...n, init: newInit, body: newBody } as unknown as T
    }
    case 'iterate': {
      const n = node as IterateNode
      const newInit = f(n.init)
      const newBody = f(n.body)
      if (newInit === n.init && newBody === n.body) return node
      return { ...n, init: newInit, body: newBody } as unknown as T
    }
    case 'fold':
    case 'scan': {
      const n = node as FoldNode | ScanNode
      const newOver = f(n.over)
      const newInit = f(n.init)
      const newBody = f(n.body)
      if (newOver === n.over && newInit === n.init && newBody === n.body) return node
      return { ...n, over: newOver, init: newInit, body: newBody } as unknown as T
    }
    case 'map2': {
      const n = node as Map2Node
      const newOver = f(n.over)
      const newBody = f(n.body)
      if (newOver === n.over && newBody === n.body) return node
      return { ...n, over: newOver, body: newBody } as unknown as T
    }
    case 'zipWith': {
      const n = node as ZipWithNode
      const newA = f(n.a)
      const newB = f(n.b)
      const newBody = f(n.body)
      if (newA === n.a && newB === n.b && newBody === n.body) return node
      return { ...n, a: newA, b: newB, body: newBody } as unknown as T
    }
    case 'strConcat': {
      const n = node as StrConcatNode
      const newParts = n.parts.map(f)
      const same = newParts.every((c, i) => c === n.parts[i])
      return same ? node : ({ ...n, parts: newParts } as unknown as T)
    }
    case 'generateDecls': {
      const n = node as GenerateDeclsNode
      const newDecls = n.decls.map(f)
      const same = newDecls.every((c, i) => c === n.decls[i])
      return same ? node : ({ ...n, decls: newDecls } as unknown as T)
    }

    // ── Decl ops ───────────────────────────────────────────────
    case 'ref':
      return node  // wiring ref carries no ExprNode children
    case 'instanceDecl': {
      const n = node as InstanceDeclNode
      const newInputs = n.inputs === undefined ? undefined : mapValues(n.inputs, f)
      const newGate = n.gate_input === undefined ? undefined : f(n.gate_input)
      if ((n.inputs === undefined || newInputs === n.inputs)
          && (n.gate_input === undefined || newGate === n.gate_input)) return node
      return {
        ...n,
        ...(n.inputs === undefined ? {} : { inputs: newInputs }),
        ...(n.gate_input === undefined ? {} : { gate_input: newGate }),
      } as unknown as T
    }
    case 'regDecl': {
      const n = node as RegDeclNode
      if (n.init === undefined) return node
      const newInit = f(n.init)
      return newInit === n.init ? node : ({ ...n, init: newInit } as unknown as T)
    }
    case 'delayDecl': {
      const n = node as DelayDeclNode
      const initIsExpr = n.init !== undefined && typeof n.init === 'object'
      const newInit = initIsExpr ? f(n.init as ExprNode) : n.init
      const newUpdate = n.update === undefined ? undefined : f(n.update)
      const initSame = !initIsExpr || newInit === n.init
      const updateSame = n.update === undefined || newUpdate === n.update
      if (initSame && updateSame) return node
      return {
        ...n,
        ...(n.init === undefined ? {} : { init: newInit }),
        ...(n.update === undefined ? {} : { update: newUpdate }),
      } as unknown as T
    }
    case 'programDecl': {
      const n = node as ProgramDeclNode
      if (n.program === undefined) return node
      const newProgram = f(n.program)
      return newProgram === n.program ? node : ({ ...n, program: newProgram } as unknown as T)
    }
    case 'outputAssign': {
      const n = node as OutputAssignNode
      if (n.expr === undefined) return node
      const newExpr = f(n.expr)
      return newExpr === n.expr ? node : ({ ...n, expr: newExpr } as unknown as T)
    }
    case 'nextUpdate': {
      const n = node as NextUpdateNode
      if (n.expr === undefined) return node
      const newExpr = f(n.expr)
      return newExpr === n.expr ? node : ({ ...n, expr: newExpr } as unknown as T)
    }
    case 'block': {
      const n = node as ProgramBlockNode
      const newDecls = n.decls === undefined ? undefined : n.decls.map(f)
      const newAssigns = n.assigns === undefined ? undefined : n.assigns.map(f)
      const declsSame = n.decls === undefined
                      || (newDecls!.length === n.decls.length && newDecls!.every((c, i) => c === n.decls![i]))
      const assignsSame = n.assigns === undefined
                       || (newAssigns!.length === n.assigns.length && newAssigns!.every((c, i) => c === n.assigns![i]))
      if (declsSame && assignsSame) return node
      return {
        ...n,
        ...(n.decls === undefined ? {} : { decls: newDecls }),
        ...(n.assigns === undefined ? {} : { assigns: newAssigns }),
      } as unknown as T
    }
    case 'program': {
      const n = node as ProgramOpNode
      // Strict ProgramBlockNode lacks the [k: string]: unknown index signature
      // that the broad ExprNode requires, so cast at the strict↔broad boundary.
      const newBody = f(n.body as unknown as ExprNode) as unknown as ProgramBlockNode
      return newBody === n.body ? node : ({ ...n, body: newBody } as unknown as T)
    }

    // ── Leaves: no children. Catches every LeafNode variant. ──
    case 'input':
    case 'reg':
    case 'delayRef':
    case 'delayValue':
    case 'nestedOut':
    case 'nestedOutput':
    case 'binding':
    case 'typeParam':
    case 'sampleRate':
    case 'sampleIndex':
    case 'param':
    case 'trigger':
    case 'smoothedParam':
    case 'triggerParam':
    case 'const':
      return node

    // ── Exhaustiveness check ─────────────────────────────────
    default: {
      // If a new op variant is added to ExprOpNodeStrict without a case
      // here, TypeScript errors on the next line: the parameter inferred
      // for `_` is no longer `never`.
      const _: never = node
      void _
      throw new Error(`mapChildren: unhandled op '${(node as ExprOpNodeStrict).op}'`)
    }
  }
}
