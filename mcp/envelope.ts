/**
 * envelope.ts — the structured tool-error envelope (spec: mcp/ERRORS.md).
 *
 * Shared by the IR engine (engine.ts) and the compiler service
 * (compiler_service.ts) so both speak exactly one error dialect. The
 * Lean engine implements this same envelope natively
 * (lean/Tropical/Errors.lean); the differential harness holds the
 * implementations to byte-level agreement on the JSON payloads.
 */

export type ErrorCode =
  | 'unknown_program'
  | 'unknown_instance'
  | 'unknown_input'
  | 'unknown_output'
  | 'unknown_param'
  | 'unknown_device'
  | 'instance_exists'
  | 'invalid_type_args'
  | 'type_mismatch'
  | 'shape_mismatch'
  | 'length_mismatch'
  | 'arity_error'
  | 'missing_argument'
  | 'invalid_value'
  | 'invalid_state'
  | 'compile_failed'
  | 'audio_error'
  | 'internal_error'

export type FieldSpec = {
  type:     'int' | 'float' | 'string' | 'bool'
  required: boolean
  min?:     number
  max?:     number
  options?: string[]
}

export type Valid =
  | { kind: 'enum';      options: string[] }
  | { kind: 'record';    fields: Record<string, FieldSpec> }
  | { kind: 'predicate'; predicate: string; expected: unknown; got: unknown }

export type ErrorEnvelope = {
  code:        ErrorCode
  message:     string
  retryable:   boolean
  param?:      string
  value?:      unknown
  valid?:      Valid
  suggestion?: unknown
}

export class ToolError extends Error {
  envelope: ErrorEnvelope
  constructor(envelope: ErrorEnvelope) {
    super(envelope.message)
    this.envelope = envelope
  }
}

export function nearestMatch(value: string, candidates: string[]): string | undefined {
  if (candidates.length === 0 || typeof value !== 'string') return undefined
  const dist = (a: string, b: string): number => {
    const m = a.length, n = b.length
    const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0))
    for (let i = 0; i <= m; i++) dp[i][0] = i
    for (let j = 0; j <= n; j++) dp[0][j] = j
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        dp[i][j] = a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
  }
  let best = candidates[0], bestD = dist(value, best)
  for (const c of candidates.slice(1)) {
    const d = dist(value, c)
    if (d < bestD) { best = c; bestD = d }
  }
  return bestD <= Math.max(2, Math.floor(value.length / 3)) ? best : undefined
}

export function failEnum(opts: {
  code: ErrorCode; param: string; value: unknown;
  options: string[]; message?: string;
}): never {
  const suggestion = typeof opts.value === 'string'
    ? nearestMatch(opts.value, opts.options)
    : undefined
  const message = opts.message
    ?? `Invalid ${opts.param}: ${JSON.stringify(opts.value)}${suggestion ? `. Did you mean '${suggestion}'?` : ''}`
  throw new ToolError({
    code:      opts.code,
    message,
    retryable: false,
    param:     opts.param,
    value:     opts.value,
    valid:     { kind: 'enum', options: opts.options },
    ...(suggestion !== undefined ? { suggestion } : {}),
  })
}

export function failRecord(opts: {
  code: ErrorCode; param: string; value: unknown;
  fields: Record<string, FieldSpec>; message?: string; suggestion?: unknown;
}): never {
  throw new ToolError({
    code:      opts.code,
    message:   opts.message ?? `Invalid ${opts.param}`,
    retryable: false,
    param:     opts.param,
    value:     opts.value,
    valid:     { kind: 'record', fields: opts.fields },
    ...(opts.suggestion !== undefined ? { suggestion: opts.suggestion } : {}),
  })
}

export function failPredicate(opts: {
  code: ErrorCode; param: string; value: unknown;
  predicate: string; expected: unknown; got: unknown;
  suggestion?: unknown; message?: string;
}): never {
  throw new ToolError({
    code:      opts.code,
    message:   opts.message ?? `${opts.predicate} failed on ${opts.param}`,
    retryable: false,
    param:     opts.param,
    value:     opts.value,
    valid:     { kind: 'predicate', predicate: opts.predicate, expected: opts.expected, got: opts.got },
    ...(opts.suggestion !== undefined ? { suggestion: opts.suggestion } : {}),
  })
}

export function failBare(opts: {
  code: ErrorCode; message: string; retryable?: boolean;
  param?: string; value?: unknown;
}): never {
  throw new ToolError({
    code:      opts.code,
    message:   opts.message,
    retryable: opts.retryable ?? false,
    ...(opts.param !== undefined ? { param: opts.param } : {}),
    ...(opts.value !== undefined ? { value: opts.value } : {}),
  })
}

/** Coerce an arbitrary thrown value to an ErrorEnvelope. */
export function toEnvelope(e: unknown): ErrorEnvelope {
  return e instanceof ToolError
    ? e.envelope
    : { code: 'internal_error', message: e instanceof Error ? e.message : String(e), retryable: false }
}

export const ok = (data: unknown) =>
  ({ content: [{ type: 'text' as const, text: JSON.stringify({ status: 'ok', data }) }] })

export const fail = (e: unknown) => {
  const envelope = toEnvelope(e)
  return {
    content: [{ type: 'text' as const, text: JSON.stringify({ status: 'error', error: envelope }) }],
    isError: true as const,
  }
}

export function wrap(fn: () => unknown) {
  try   { return ok(fn())    }
  catch (e) { return fail(e) }
}
