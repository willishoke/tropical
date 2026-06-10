/**
 * resolved_codec.test.ts — property gates for the Resolved⇄JSON codec
 * (Phase 4, stage 1 of the Lean port: gated in TS before any Lean
 * consumer exists).
 *
 * Gate 1 — round-trip:
 *   decodeResolved(JSON.parse(JSON.stringify(encodeResolved(p)))) is
 *   structurally equal to `p`, AND encode∘decode∘encode = encode (the
 *   second encode is byte-identical JSON to the first — catches
 *   identity-loss asymmetries the structural compare can miss).
 *
 * Gate 2 — recompile-equality:
 *   compiling the decoded program produces byte-identical plan JSON to
 *   compiling the original, at BOTH pipeline points:
 *     - post-elaborate (pre-strata): strata + compileResolved both sides
 *     - post-strata: compileResolved both sides
 *   and for session roots: partitionKernel both sides.
 *
 * Corpus: every stdlib program (elaborated and post-strata), generic
 * programs specialized with explicit type args, sum-typed programs
 * (tag/match), nested programDecls + InstanceDecls, and two session
 * roots built from patches/*.json through the session→parsed→elaborate
 * path. Pure-TS gates: plan JSON comparison only, no rendering — the
 * native runtime handle created by `makeSession` is never asked to
 * load a plan (its `loadPlan` is stubbed to a no-op for the session
 * fixtures, matching the compile-only usage in compile_session.test.ts).
 */

import { describe, test, expect } from 'bun:test'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { parseProgram } from '../parse/declarations.js'
import { extractMarkdown } from '../parse/markdown.js'
import { elaborate, type ExternalProgramResolver } from './elaborator.js'
import { strataPipeline } from './strata.js'
import { compileResolved } from './compile_resolved.js'
import { encodeResolved, decodeResolved } from './resolved_codec.js'
import { mkProgram } from './decl_tables.js'
import {
  programKey,
  paramIdx,
  type ResolvedProgram, type TypeParamDecl, type ParamIdx, type ProgramKey,
} from './nodes.js'

import { makeSession, loadJSON, allocateOutputSlots, allocateInputSlots, type SessionState } from '../session.js'
import { loadStdlib } from '../program.js'
import { makeCompiled, type Compiled } from '../program_types.js'
import { sessionToParsedProgram, sessionTypeResolver } from './session_to_parsed.js'
import { partitionKernel, makeAccumulators, ROOT_INSTANCE_PATH } from './partition_recursive.js'
import { getInstanceType } from './decl_tables.js'
import { childInstance, instanceName as toInstanceName, type InstanceName } from './branded_names.js'
import { emitSinks } from './compile_session_slotted_helpers.js'

const STDLIB_DIR = join(import.meta.dir, '../../stdlib')
const PATCHES_DIR = join(import.meta.dir, '../../patches')

// ─────────────────────────────────────────────────────────────
// Gate helpers
// ─────────────────────────────────────────────────────────────

/** Gate 1: round-trip + encode∘decode∘encode = encode. Returns the
 *  decoded program for the caller's Gate-2 use. */
function roundTrip(p: ResolvedProgram): ResolvedProgram {
  const enc1 = encodeResolved(p)
  const json1 = JSON.stringify(enc1)
  const decoded = decodeResolved(JSON.parse(json1) as unknown)
  // encode∘decode∘encode = encode (byte-identical JSON).
  const json2 = JSON.stringify(encodeResolved(decoded))
  expect(json2).toBe(json1)
  // Structural equality of the decoded value against the original.
  expect(decoded).toEqual(p)
  return decoded
}

/** Explicit type args for a generic program, keyed by the program's
 *  OWN TypeParamDecl objects (specialize requires identity keys).
 *  Small explicit values keep the array unrolls test-sized. */
function explicitArgs(p: ResolvedProgram): Map<TypeParamDecl, number> {
  return new Map(p.typeParams.map(tp => [tp, 8]))
}

/** Compile outcome: the plan JSON, or the exact compile error. A few
 *  stdlib programs (Clock, Sequencer) index array-typed INPUT ports and
 *  cannot compile standalone through `compileResolved` at all — the
 *  per-instance path needs the session's array-slot wiring (the same
 *  reason compile_session.test.ts excludes Clock). For those, gate 2
 *  asserts both sides fail identically here; their real plan-level
 *  equality is covered by the sequencer_demo session root below, which
 *  compiles both through the slot-based session path. */
function compileOutcome(f: () => string): string {
  try {
    return `plan:${f()}`
  } catch (e) {
    return `error:${e instanceof Error ? e.message : String(e)}`
  }
}

/** Per-program plan JSON: strata (with each side's own identity-keyed
 *  type args) then compileResolved. */
function planJsonPreStrata(p: ResolvedProgram): string {
  return compileOutcome(() => {
    const post = strataPipeline(p, explicitArgs(p))
    return JSON.stringify(compileResolved(post))
  })
}

function planJsonPostStrata(p: ResolvedProgram): string {
  return compileOutcome(() => JSON.stringify(compileResolved(p)))
}

/** Gate 2 at the pre-strata point. */
function expectRecompileEqualPreStrata(p: ResolvedProgram, decoded: ResolvedProgram): void {
  expect(planJsonPreStrata(decoded)).toBe(planJsonPreStrata(p))
}

/** Gate 2 at the post-strata point. */
function expectRecompileEqualPostStrata(p: ResolvedProgram, decoded: ResolvedProgram): void {
  expect(planJsonPostStrata(decoded)).toBe(planJsonPostStrata(p))
}

// ─────────────────────────────────────────────────────────────
// Stdlib corpus
// ─────────────────────────────────────────────────────────────

/** Parse + elaborate every stdlib program (fixed-point sibling
 *  resolution, mirroring program.ts:loadStdlibFromResolved) WITHOUT
 *  running strata — these are the post-elaborate (pre-strata) forms. */
function stdlibElaborated(): Map<string, ResolvedProgram> {
  const files = readdirSync(STDLIB_DIR)
    .filter(f => f.endsWith('.md') && f !== 'README.md')
    .sort()
  const parsedByName = new Map<string, ReturnType<typeof parseProgram>>()
  for (const file of files) {
    const text = readFileSync(join(STDLIB_DIR, file), 'utf-8')
    const ext = extractMarkdown(text)
    if (ext.blocks.length !== 1) {
      throw new Error(`${file}: expected exactly 1 tropical code block, got ${ext.blocks.length}`)
    }
    const parsed = parseProgram(ext.blocks[0].source)
    parsedByName.set(parsed.name, parsed)
  }

  const resolved = new Map<string, ResolvedProgram>()
  const resolver: ExternalProgramResolver = name => resolved.get(name)
  const remaining = new Map(parsedByName)
  let progress = true
  while (progress && remaining.size > 0) {
    progress = false
    for (const [name, parsed] of remaining) {
      try {
        resolved.set(name, elaborate(parsed, resolver))
        remaining.delete(name)
        progress = true
      } catch {
        // Sibling not yet elaborated; retry next pass.
      }
    }
  }
  if (remaining.size > 0) {
    throw new Error(`stdlibElaborated: failed to elaborate ${[...remaining.keys()].join(', ')}`)
  }
  return resolved
}

const elaboratedStdlib = stdlibElaborated()

describe('resolved codec — stdlib, post-elaborate (pre-strata)', () => {
  for (const [name, prog] of elaboratedStdlib) {
    test(`round-trip + recompile: ${name}`, () => {
      const decoded = roundTrip(prog)
      expectRecompileEqualPreStrata(prog, decoded)
    })
  }
})

describe('resolved codec — stdlib, post-strata', () => {
  // loadStdlib with a bare Map target runs the full elaborate+strata
  // pipeline with no session and no native runtime.
  const registry = new Map<string, Compiled>()
  loadStdlib(registry)
  for (const [name, compiled] of registry) {
    test(`round-trip + recompile: ${name}`, () => {
      const decoded = roundTrip(compiled.prog)
      expectRecompileEqualPostStrata(compiled.prog, decoded)
    })
  }
})

describe('resolved codec — generic programs (explicit type args)', () => {
  const generics = [...elaboratedStdlib.entries()].filter(([, p]) => p.typeParams.length > 0)
  test('the stdlib corpus actually contains generic templates', () => {
    expect(generics.length).toBeGreaterThan(0)
  })
  for (const [name, prog] of generics) {
    test(`round-trip + specialize<8> + recompile: ${name}`, () => {
      const decoded = roundTrip(prog)
      // Each side specializes with type args keyed by its OWN
      // TypeParamDecl identities — exactly the identity contract
      // specializeProgram enforces.
      expectRecompileEqualPreStrata(prog, decoded)
      // And the specialized (post-strata) images round-trip too.
      const post = strataPipeline(prog, explicitArgs(prog))
      const postDecoded = roundTrip(post)
      expectRecompileEqualPostStrata(post, postDecoded)
    })
  }
})

// ─────────────────────────────────────────────────────────────
// Sum-typed fixtures (tag / match / payload variants)
// ─────────────────────────────────────────────────────────────

const SUM_FIXTURES: Record<string, string> = {
  Toggle: `
    program Toggle() -> (value: float) {
      enum St { Off, On }
      delay state: St = match state { Off => On { }, On => Off { } } init Off { }
      value = match state { Off => 0.0, On => 1.0 }
    }
  `,
  EnvLite: `
    program EnvLite(trigger: float = 0, decay: float = 0.999) -> (env: float) {
      enum Env { Idle, Decaying(level: float) }
      delay state: Env = match state {
        Idle => select(trigger > 0.5, Decaying { level: 1 }, Idle { }),
        Decaying { level: level } => select(trigger > 0.5, Decaying { level: 1 }, Decaying { level: level * decay })
      } init Idle { }
      env = match state { Idle => 0, Decaying { level: level } => level }
    }
  `,
}

describe('resolved codec — sum-typed programs', () => {
  for (const [name, src] of Object.entries(SUM_FIXTURES)) {
    test(`round-trip + recompile: ${name}`, () => {
      const prog = elaborate(parseProgram(src))
      const decoded = roundTrip(prog)
      expectRecompileEqualPreStrata(prog, decoded)
    })
  }
})

// ─────────────────────────────────────────────────────────────
// Nested programDecls + InstanceDecls
// ─────────────────────────────────────────────────────────────

const NESTED_FIXTURES: Record<string, string> = {
  TwoStage: `
    program TwoStage(a: float) -> (out: float) {
      program A(x: float) -> (y: float) {
        reg s: float = 0
        y = s + x
        next s = x * 0.5
      }
      program B(z: float) -> (w: float) { w = z + 10 }
      i1 = A(x: a)
      i2 = B(z: i1.y)
      out = i1.y + i2.w
    }
  `,
  NestedGeneric: `
    program NestedGeneric(x: float) -> (out: float) {
      program Buf<N: int = 4>(v: float) -> (head: float) {
        reg buf = zeros(N)
        head = buf[sampleIndex() % N]
        next buf = arraySet(buf, sampleIndex() % N, v)
      }
      d = Buf<N=4>(v: x)
      out = d.head
    }
  `,
}

describe('resolved codec — nested programDecls + instances', () => {
  for (const [name, src] of Object.entries(NESTED_FIXTURES)) {
    test(`round-trip + recompile: ${name}`, () => {
      const prog = elaborate(parseProgram(src))
      const decoded = roundTrip(prog)
      expectRecompileEqualPreStrata(prog, decoded)
    })
  }
})

// ─────────────────────────────────────────────────────────────
// Session roots — session → parsed → elaborate → partitionKernel
// ─────────────────────────────────────────────────────────────

/** Replicates compile_session_slotted.ts:preallocate*Recursive (not
 *  exported there) — idempotent two-phase slot pre-allocation. */
function preallocOutputs(
  session: SessionState, path: InstanceName, prog: ResolvedProgram, compiled: Compiled,
): void {
  allocateOutputSlots(session, path, compiled)
  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') continue
    const declType = getInstanceType(prog, decl)
    preallocOutputs(session, childInstance(path, decl.name), declType, makeCompiled(declType, { displayName: declType.name }))
  }
}

function preallocInputs(
  session: SessionState, path: InstanceName, prog: ResolvedProgram, compiled: Compiled,
): void {
  allocateInputSlots(session, path, compiled)
  for (const decl of prog.body.decls) {
    if (decl.op !== 'instanceDecl') continue
    const declType = getInstanceType(prog, decl)
    preallocInputs(session, childInstance(path, decl.name), declType, makeCompiled(declType, { displayName: declType.name }))
  }
}

/** Root-lowering plan JSON: the partitionKernel slice of
 *  compile_session_slotted.ts:compileSessionSlottedFromParsed, with the
 *  root `ResolvedProgram` injected so the codec sits in the loop. The
 *  session-only slot metadata is omitted — both sides share one session,
 *  so it carries no codec signal. */
function compileRootPlanJson(session: SessionState, root: ResolvedProgram): string {
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) {
      preallocOutputs(session, toInstanceName(name), inst.compiled.prog, inst.compiled)
    }
  }
  for (const [name, inst] of session.instanceRegistry) {
    if (inst.compiled !== undefined) {
      preallocInputs(session, toInstanceName(name), inst.compiled.prog, inst.compiled)
    }
  }

  const acc = makeAccumulators()
  for (let i = 0; i < session.ioArraySlotCount; i++) {
    acc.arraySlotSizes.push(session.ioArraySlotSizes[i])
    acc.arraySlotNames.push(session.ioArraySlotNames[i])
  }
  acc.nextArrayRaw = session.ioArraySlotCount

  const rootCompiled = makeCompiled(root, { displayName: '__session__' })
  const paramSlots = new Map<ParamIdx, number>()
  for (let i = 0; i < root.params.length; i++) {
    const slot = session.paramSlotRegistry.get(root.params[i].name)
    if (slot !== undefined) paramSlots.set(paramIdx(i), slot)
  }

  const { fn } = partitionKernel(
    ROOT_INSTANCE_PATH,
    root,
    rootCompiled,
    () => undefined,
    {},
    new Map(),
    session,
    acc,
    undefined,
    undefined,
    paramSlots,
  )

  return JSON.stringify({
    fn,
    state_init: acc.stateInit,
    register_names: acc.registerNames,
    register_types: acc.registerTypes,
    register_count: acc.nextRegRaw,
    array_slot_names: acc.arraySlotNames,
    array_slot_sizes: acc.arraySlotSizes,
    array_slot_count: acc.nextArrayRaw,
    sinks: emitSinks(session),
  })
}

/** Build a session from a patch and elaborate its synthetic root.
 *  `loadPlan` is stubbed to keep the gate compile-only (no native JIT);
 *  `loadJSON`'s internal `compileSession` still runs in full, leaving
 *  the session post-extraction — exactly what `sessionToParsedProgram`
 *  consumes. */
function sessionRoot(patchFile: string): { session: SessionState; root: ResolvedProgram } {
  const session = makeSession()
  session.runtime.loadPlan = () => true
  loadStdlib(session)
  const json = JSON.parse(readFileSync(join(PATCHES_DIR, patchFile), 'utf-8')) as
    { schema: string; [k: string]: unknown }
  loadJSON(json, session)
  const parsed = sessionToParsedProgram(session)
  const root = elaborate(parsed, sessionTypeResolver(session))
  return { session, root }
}

describe('resolved codec — session roots', () => {
  for (const patch of ['cross_fm_4.json', 'sequencer_demo.json']) {
    test(`round-trip + partition recompile: ${patch}`, () => {
      const { session, root } = sessionRoot(patch)
      try {
        const decoded = roundTrip(root)
        const planA = compileRootPlanJson(session, root)
        // Idempotence guard: a second compile of the ORIGINAL root must
        // be byte-identical, so any decoded-side diff is the codec's.
        expect(compileRootPlanJson(session, root)).toBe(planA)
        expect(compileRootPlanJson(session, decoded)).toBe(planA)
      } finally {
        session.graph.dispose()
      }
    })
  }
})

// ─────────────────────────────────────────────────────────────
// Defensive: cyclic program references
// ─────────────────────────────────────────────────────────────

describe('resolved codec — cycle defense', () => {
  test('encode throws on a cyclic program registry', () => {
    const registry = new Map<ProgramKey, ResolvedProgram>()
    const prog = mkProgram({
      name: 'Cyclic',
      typeParams: [],
      ports: { inputs: [], outputs: [], typeDefs: [] },
      body: { op: 'block', decls: [], assigns: [] },
      binderCount: 0,
      programRegistry: registry,
    })
    // Close the loop after construction (mkProgram holds the same Map).
    registry.set(programKey('Cyclic'), prog)
    expect(() => encodeResolved(prog)).toThrow(/cyclic program reference/)
  })

  test('encode throws on non-finite literals instead of silently corrupting', () => {
    const prog = elaborate(parseProgram('program X(a: float) -> (out: float) { out = a + 1 }'))
    // Forge a non-finite literal in a fresh copy of the assign expr.
    const broken = mkProgram({
      name: prog.name,
      typeParams: prog.typeParams,
      ports: prog.ports,
      body: {
        op: 'block',
        decls: prog.body.decls,
        assigns: [{ op: 'outputAssign', target: prog.body.assigns[0].target, expr: Infinity }],
      },
      binderCount: prog.binderCount,
      programRegistry: prog.programRegistry,
    })
    expect(() => encodeResolved(broken)).toThrow(/non-finite/)
  })
})
