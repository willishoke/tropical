/**
 * structural.ts — order-insensitive structural JSON comparison.
 *
 * The Lean-port differential harness never compares JSON *text*: two
 * pipelines may serialize the same plan with different key order or
 * float formatting. Values are compared structurally — objects by key
 * set, arrays element-wise, numbers by IEEE-754 identity (`Object.is`,
 * which is bit equality for doubles: it distinguishes +0/−0 and treats
 * the JSON-unrepresentable NaN payloads as equal).
 *
 * Used by diff_plan.ts (plan-vs-plan) and diff_engine.ts
 * (response-vs-response).
 */

export interface DiffEntry {
  /** JSONPath-ish location of the divergence, e.g. `$.instance_functions[0].instructions[3].dst` */
  path: string
  a: unknown
  b: unknown
}

const MAX_DIFFS = 20

/** Collect up to MAX_DIFFS structural divergences between two JSON values. */
export function structuralDiff(a: unknown, b: unknown): DiffEntry[] {
  const out: DiffEntry[] = []
  walk(a, b, '$', out)
  return out
}

function walk(a: unknown, b: unknown, path: string, out: DiffEntry[]): void {
  if (out.length >= MAX_DIFFS) return

  if (typeof a === 'number' && typeof b === 'number') {
    if (!Object.is(a, b)) out.push({ path, a, b })
    return
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      out.push({ path: `${path}.length`, a: a.length, b: b.length })
      return
    }
    for (let i = 0; i < a.length; i++) walk(a[i], b[i], `${path}[${i}]`, out)
    return
  }

  if (isPlainObject(a) && isPlainObject(b)) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)])
    for (const k of [...keys].sort()) {
      if (out.length >= MAX_DIFFS) return
      const inA = k in a
      const inB = k in b
      if (!inA || !inB) {
        out.push({ path: `${path}.${k}`, a: inA ? a[k] : '<absent>', b: inB ? b[k] : '<absent>' })
        continue
      }
      walk(a[k], b[k], `${path}.${k}`, out)
    }
    return
  }

  if (a !== b) out.push({ path, a, b })
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

/** Render a diff list for terminal output. */
export function formatDiffs(diffs: DiffEntry[]): string {
  return diffs
    .map(d => `    ${d.path}\n      a: ${JSON.stringify(d.a)}\n      b: ${JSON.stringify(d.b)}`)
    .join('\n')
}
