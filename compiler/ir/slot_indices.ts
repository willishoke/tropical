/**
 * slot_indices.ts — branded namespace types for the flat IR.
 *
 * The plan_5 layout carries five distinct integer namespaces:
 *
 *   - TempIdx       — per-sample scratch (`temps[]` in the kernel)
 *   - StateRegIdx   — per-instance persistent state (`registers[]`)
 *   - ArraySlotIdx  — per-instance array buffers (`arrays[]`)
 *   - ModuleSlotIdx — inter-module slot array (`slots[]`)
 *   - InputPortIdx  — input port position on an instance (compile-time)
 *
 * Pre-refactor they were all `number`. Two production bugs landed
 * from confusing them:
 *
 *   1. A `-1` "no scalar writeback" sentinel in `register_targets`
 *      was being arithmetically shifted by `regOffset`, turning the
 *      sentinel into a valid temp index that the JIT then dutifully
 *      wrote to.
 *   2. `instr.dst` for an elementwise loop (or Pack/SetElement)
 *      is an ArraySlotIdx, not a TempIdx, so shifting it by
 *      `regOffset` produced array-write targets far outside the
 *      allocated array storage. Direct cause of the Phaser segfault.
 *
 * Branding makes both of these compile errors. The runtime
 * representation is still `number` (zero overhead, JSON output
 * unchanged) — the brand is a phantom type the compiler enforces
 * at boundaries (constructors, typed arithmetic helpers).
 *
 * ## Discipline
 *
 * Only this module produces values of the branded types. Code that
 * wants to convert a raw `number` into a branded index calls the
 * constructor (e.g. `tempIdx(n)`); all internal arithmetic goes
 * through the typed shift helpers (`shiftTemp`, etc.). Direct casts
 * (`n as TempIdx`) should be rare and reviewed.
 *
 * Offsets get their own brands too so that `shiftTemp(local,
 * arraySlotOffset)` is rejected by the compiler — exactly the
 * mistake that caused bug 2.
 */

declare const __brand: unique symbol

/** Phantom-tag a numeric type. Erased at runtime. */
export type Branded<B extends string> = number & { readonly [__brand]: B }

// ─── Index brands ───────────────────────────────────────────────────────────
// "Index" = an absolute slot position within a single namespace's
// flat array. Constructed by allocators or by adding a same-namespace
// Offset to another Index.

export type TempIdx       = Branded<'TempIdx'>
export type StateRegIdx   = Branded<'StateRegIdx'>
export type ArraySlotIdx  = Branded<'ArraySlotIdx'>
export type ModuleSlotIdx = Branded<'ModuleSlotIdx'>
export type InputPortIdx  = Branded<'InputPortIdx'>

// ─── Offset brands ──────────────────────────────────────────────────────────
// "Offset" = a per-instance cumulative shift in the same namespace.
// Distinct brand from Index so the compiler refuses to subtract two
// indices and accidentally use the result as an Index. Same numeric
// representation; the brand is a stronger lint.

export type TempOffset       = Branded<'TempOffset'>
export type StateRegOffset   = Branded<'StateRegOffset'>
export type ArraySlotOffset  = Branded<'ArraySlotOffset'>

// ─── Constructors ───────────────────────────────────────────────────────────
// Single-source-of-truth conversion from raw `number` to branded.
// Negative values are rejected — the `-1` sentinel for array-managed
// regs lives in `RegTarget`, not in `TempIdx`.

const construct = <B extends string>(name: B, n: number): Branded<B> => {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`${name}: invalid index ${n} — must be a non-negative integer`)
  }
  return n as Branded<B>
}

export const tempIdx       = (n: number): TempIdx       => construct('TempIdx', n)
export const stateRegIdx   = (n: number): StateRegIdx   => construct('StateRegIdx', n)
export const arraySlotIdx  = (n: number): ArraySlotIdx  => construct('ArraySlotIdx', n)
export const moduleSlotIdx = (n: number): ModuleSlotIdx => construct('ModuleSlotIdx', n)
export const inputPortIdx  = (n: number): InputPortIdx  => construct('InputPortIdx', n)

export const tempOffset       = (n: number): TempOffset       => construct('TempOffset', n)
export const stateRegOffset   = (n: number): StateRegOffset   => construct('StateRegOffset', n)
export const arraySlotOffset  = (n: number): ArraySlotOffset  => construct('ArraySlotOffset', n)

// ─── Typed shifts ───────────────────────────────────────────────────────────
// Adding an Offset to an Index of the SAME namespace produces a new
// Index in that namespace. The signatures are what enforces the
// discipline — `shiftTemp(local, arraySlotOffset)` fails to
// type-check because `ArraySlotOffset` isn't assignable to
// `TempOffset`.

export const shiftTemp = (local: TempIdx, off: TempOffset): TempIdx =>
  (local + off) as TempIdx
export const shiftStateReg = (local: StateRegIdx, off: StateRegOffset): StateRegIdx =>
  (local + off) as StateRegIdx
export const shiftArraySlot = (local: ArraySlotIdx, off: ArraySlotOffset): ArraySlotIdx =>
  (local + off) as ArraySlotIdx

// ─── Index → raw number (JSON-emit boundary) ────────────────────────────────
// Identity functions used at the JSON-serialization boundary to make
// the namespace transition explicit. Calling `rawIdx(x)` documents
// "I'm leaving the typed world; integers go on the wire from here."

export const rawIdx    = (i: number): number => i
export const rawOffset = (o: number): number => o

// ─── Zero-offset constants ──────────────────────────────────────────────────

export const ZERO_TEMP_OFFSET:       TempOffset       = 0 as TempOffset
export const ZERO_STATE_REG_OFFSET:  StateRegOffset   = 0 as StateRegOffset
export const ZERO_ARRAY_SLOT_OFFSET: ArraySlotOffset  = 0 as ArraySlotOffset
