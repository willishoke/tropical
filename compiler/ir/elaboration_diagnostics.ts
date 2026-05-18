/**
 * compiler/ir/elaboration_diagnostics.ts — diagnostic records and
 * formatters for elaboration-time warnings and errors.
 *
 * Today this carries the cycle-detection diagnostics used by Phase 4a
 * (warn-on-cycle, before flipping to strict-error in Phase 4b). The
 * shape is designed so that:
 *
 *   - Warnings are structured records (not just strings), so future
 *     MCP/agent integrations can route them to dedicated channels.
 *   - Error formatting is centralized: Phase 4b's `CycleViolation`
 *     error uses the same `formatCycleSuggestion` helper, so warning
 *     and error messages stay in lockstep.
 *
 * The Phase 4a default sink is `console.warn`. Future work can replace
 * `emitWarning` with a callback registered via the elaborator's
 * options, but the synchronous warn-now default is fine for the
 * current single-process pipeline.
 */

import type { InstanceDecl, OutputDecl } from './nodes.js'
import type { BrokenCycle } from './lowering/cycle_break.js'

// ─────────────────────────────────────────────────────────────
// Diagnostic record types
// ─────────────────────────────────────────────────────────────

/** A single cycle-detection finding produced by the elaborator. */
export interface CycleDiagnostic {
  readonly kind: 'cycle'
  /** The strongly connected component (cycle members in source order). */
  readonly scc: ReadonlyArray<InstanceDecl>
  /** The program in which the cycle was found (its name, not a ref —
   *  the resolved program object identity is not load-bearing for
   *  diagnostics). */
  readonly programName: string
  /** A suggested-fix snippet: an explicit `delay` (or `reg`) the user
   *  could add to break the cycle. Mirrors the cycle-break helper's
   *  choice of break-target so the suggestion matches what the
   *  auto-fix would do. */
  readonly suggestedFix: string
}

// ─────────────────────────────────────────────────────────────
// Formatting
// ─────────────────────────────────────────────────────────────

/** Pretty-print a CycleDiagnostic as a human-readable warning string.
 *  Same format used by `CycleViolation` errors in Phase 4b so warning
 *  and error messages agree. */
export function formatCycleDiagnostic(d: CycleDiagnostic): string {
  const memberPath = d.scc.map(i => i.name).join(' → ')
  const lines = [
    `tropical: cycle in program '${d.programName}' without a user register`,
    `  Instances in cycle: ${memberPath}`,
    `  ${d.suggestedFix}`,
  ]
  return lines.join('\n')
}

/** Build a Tier-2 "suggested fix" snippet for a broken cycle. Names
 *  the break-target instance, the ports promoted to synthetic regs,
 *  and proposes an explicit `delay` statement the user could add. */
export function buildSuggestedFix(broken: BrokenCycle): string {
  const target = broken.breakTarget.name
  if (broken.breakPorts.length === 0) {
    // No port has been used yet (the cycle was detected but the
    // synthetic-reg allocation didn't run, e.g. detection-only path
    // in Phase 4a). Fall back to a generic suggestion.
    return (
      `Suggested fix: insert a 'delay' statement on one of '${target}'’s ` +
      `output ports to break the cycle explicitly.`
    )
  }
  const portNames: ReadonlyArray<OutputDecl> = broken.breakPorts
  const suggestions = portNames.map(p => {
    const synthName = `${target}_${p.name}_delayed`
    return (
      `  Insert: delay ${synthName} = ${target}.${p.name} init 0\n` +
      `  Then route cycle members from ${synthName} instead of ${target}.${p.name}`
    )
  }).join('\n')
  return `Suggested fix:\n${suggestions}`
}

// ─────────────────────────────────────────────────────────────
// Emission
// ─────────────────────────────────────────────────────────────

/** Default sink: console.warn. Single function so tests can spy on
 *  it and future versions can wire it to MCP error channels. */
export function emitWarning(d: CycleDiagnostic): void {
  console.warn(formatCycleDiagnostic(d))
}

// ─────────────────────────────────────────────────────────────
// Phase 4b — strict error class
// ─────────────────────────────────────────────────────────────

/** Thrown by the elaborator (or any other resolved-IR producer) when
 *  source contains an inter-instance cycle that doesn't pass through
 *  an explicit user register. Carries the SCCs as structured data so
 *  callers can render the violation precisely. */
export class CycleViolation extends Error {
  readonly diagnostics: ReadonlyArray<CycleDiagnostic>
  constructor(diagnostics: ReadonlyArray<CycleDiagnostic>) {
    const body = diagnostics.map(formatCycleDiagnostic).join('\n\n')
    super(`tropical: strict cycle policy violated:\n${body}`)
    this.name = 'CycleViolation'
    this.diagnostics = diagnostics
  }
}
