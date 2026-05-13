/**
 * branded_names.ts — phantom-typed identifiers for the IR vocabulary.
 *
 * The IR has two kinds of string identifiers that get confused if they
 * share a type:
 *
 *   - `InstanceName` — names an instance in the session graph. May be a
 *     dotted path (`voice1.env.lp`) when sub-instances are addressable at
 *     depth. Used for hot-swap state matching, slot naming, and the MCP
 *     surface.
 *   - `PortName` — names an input or output port of an instance.
 *
 * Both are runtime-`string`. The brand is a phantom type the compiler
 * enforces at boundaries. Calling `instanceName(s)` is the single place a
 * raw `string` gets lifted into the typed world; after that, the type
 * system tracks which strings are which.
 *
 * Sibling to `slot_indices.ts`: same parse-don't-validate discipline,
 * different namespace (names are strings, not arithmetic-able).
 *
 * Discipline:
 * - Only this module produces values of the branded types.
 * - At the MCP boundary, raw user input gets branded via
 *   `instanceName` / `portName` before flowing inward.
 * - At the JSON-serialization boundary, `rawName` documents the exit.
 * - Direct casts (`s as InstanceName`) should be rare and reviewed.
 */

declare const __nameBrand: unique symbol

/** Phantom-tag a string. Erased at runtime. */
export type BrandedName<B extends string> = string & { readonly [__nameBrand]: B }

// ─── Name brands ────────────────────────────────────────────────────────────

/** Name of an instance in the session graph. May be a dotted path
 *  (`voice1.env`) for nested instances; the dot separator is reserved
 *  for nesting and must not appear in a single-level instance name. */
export type InstanceName = BrandedName<'InstanceName'>

/** Name of an input or output port on an instance. Single-level —
 *  dots are not allowed (ports are not nested). */
export type PortName = BrandedName<'PortName'>

// ─── Constructors ───────────────────────────────────────────────────────────

const construct = <B extends string>(name: B, s: string, allowDots: boolean): BrandedName<B> => {
  if (typeof s !== 'string') {
    throw new Error(`${name}: expected string, got ${typeof s}`)
  }
  if (s.length === 0) {
    throw new Error(`${name}: empty string is not a valid name`)
  }
  if (!allowDots && s.includes('.')) {
    throw new Error(`${name}: dots are not allowed in '${s}' (dots reserved for nested paths)`)
  }
  return s as BrandedName<B>
}

export const instanceName = (s: string): InstanceName =>
  construct('InstanceName', s, /* allowDots */ true)

export const portName = (s: string): PortName =>
  construct('PortName', s, /* allowDots */ false)

/** Identity at the serialization boundary. Documents "leaving the
 *  typed world; raw string from here." */
export const rawName = <B extends string>(n: BrandedName<B>): string => n

// ─── Path operations on InstanceName ────────────────────────────────────────

/** Compose a child instance under a parent. `child(parent, "env")` →
 *  `"parent.env"`. The parent may itself be a nested path. */
export const childInstance = (parent: InstanceName, child: string): InstanceName => {
  if (child.includes('.')) {
    throw new Error(`childInstance: child name '${child}' must not contain dots`)
  }
  if (child.length === 0) {
    throw new Error(`childInstance: child name cannot be empty`)
  }
  return `${parent}.${child}` as InstanceName
}

/** Split a nested InstanceName into its path components. `["voice1",
 *  "env"]` for `"voice1.env"`; `["voice1"]` for `"voice1"`. */
export const instancePathParts = (name: InstanceName): string[] => name.split('.')

/** True if this is a top-level (non-nested) instance name. */
export const isTopLevel = (name: InstanceName): boolean => !name.includes('.')

/** The parent of a nested name, or undefined if top-level.
 *  `parentOf("voice1.env")` → `"voice1"`; `parentOf("voice1")` → undefined. */
export const parentOf = (name: InstanceName): InstanceName | undefined => {
  const idx = name.lastIndexOf('.')
  if (idx < 0) return undefined
  return name.slice(0, idx) as InstanceName
}

/** The leaf segment of a nested name. `leafOf("voice1.env")` → `"env"`;
 *  `leafOf("voice1")` → `"voice1"`. Returns a plain string, not a
 *  PortName — the leaf of an instance path is itself an instance label,
 *  not a port. */
export const leafOf = (name: InstanceName): string => {
  const idx = name.lastIndexOf('.')
  return idx < 0 ? name : name.slice(idx + 1)
}

// ─── PortRef — instance + port pair ─────────────────────────────────────────

/** A reference to a specific port on a specific instance. Replaces ad-hoc
 *  string keys like `${instance}:${port}`. */
export interface PortRef {
  readonly instance: InstanceName
  readonly port:     PortName
}

export const portRef = (inst: InstanceName, port: PortName): PortRef => ({
  instance: inst,
  port,
})

/** Structural equality on PortRefs. */
export const portRefEq = (a: PortRef, b: PortRef): boolean =>
  a.instance === b.instance && a.port === b.port

/** Canonical string form for use as a Map key or in JSON. Uses `:` as
 *  the instance/port separator (distinct from `.` which separates
 *  nesting levels in the instance path). */
export const portRefKey = (r: PortRef): string => `${r.instance}:${r.port}`

/** Inverse of `portRefKey`. Throws on malformed input. */
export const parsePortRefKey = (key: string): PortRef => {
  const idx = key.indexOf(':')
  if (idx < 0) {
    throw new Error(`parsePortRefKey: malformed key '${key}' — missing ':'`)
  }
  const instPart = key.slice(0, idx)
  const portPart = key.slice(idx + 1)
  return {
    instance: instanceName(instPart),
    port:     portName(portPart),
  }
}
