# Tool error envelope

All MCP tool failures return a single uniform JSON shape. The goal: an LLM client can discriminate the error class, identify the offending argument, see what *would* have been accepted, and decide whether to retry — without parsing prose.

## Top-level shape

```ts
type ToolResult =
  | { status: "ok",    data:  unknown }
  | { status: "error", error: ErrorEnvelope }

type ErrorEnvelope = {
  code:        ErrorCode    // machine-readable class
  message:     string       // human-readable, single sentence, ≤120 chars
  retryable:   boolean      // false if retrying with identical args will fail identically
  param?:      string       // name of the offending tool argument
  value?:      unknown      // value the caller supplied for `param`
  valid?:      Valid        // structured description of what `param` would accept
  suggestion?: Suggestion   // best guess at what the caller meant, same type as `param` takes
}
```

`message` is fallback copy for humans. Agents read `code` + `param` + `valid` + `suggestion`. Do not put information in `message` that isn't also in a structured field.

## The `Valid` ADT

```ts
type Valid = EnumValid | RecordValid | PredicateValid
```

Three constructors. The `code` already tells the caller *which axis* of the input is wrong; `valid` just describes the alternatives along that one axis.

### `enum` — flat finite set

Use when the valid set is a finite list of atoms. Covers every lookup error (program name, instance name, port name, param name). For errors about a port on a specific instance, the list is scoped to that instance's ports — not the cross-product of all instances and ports.

```ts
type EnumValid = {
  kind:    "enum"
  options: string[]   // sorted
}
```

```json
{ "kind": "enum", "options": ["input", "cutoff", "resonance"] }
```

### `record` — object with typed fields

Use when the valid value is an object — e.g. `type_args: { N: int }`. Each field has its own spec.

```ts
type RecordValid = {
  kind:   "record"
  fields: Record<string, FieldSpec>
}

type FieldSpec = {
  type:     "int" | "float" | "string" | "bool"
  required: boolean
  min?:     number
  max?:     number
  options?: string[]   // for enumerated string fields
}
```

```json
{
  "kind": "record",
  "fields": {
    "N": { "type": "int", "required": true, "min": 1 }
  }
}
```

### `predicate` — non-enumerable constraint

Use when the valid set is defined by a relation, not enumerable as a finite list. Canonical case: type/shape compatibility between a wiring source and dest. Any matching `PortType` is valid; we can't list them.

```ts
type PredicateValid = {
  kind:      "predicate"
  predicate: string   // short name of the relation, e.g. "type_compatible"
  expected:  unknown  // the constraint side (typically a PortType)
  got:       unknown  // what the caller supplied (typically a PortType)
}
```

```json
{
  "kind": "predicate",
  "predicate": "type_compatible",
  "expected": { "tag": "scalar", "scalar": "float" },
  "got":      { "tag": "array",  "element": { "tag": "scalar", "scalar": "float" }, "shape": [4] }
}
```

## The `suggestion` field

`suggestion` is the agent's recommended next value for `param` — a ready-to-use value of the type `param` takes, not freeform prose.

- For `enum`: `string` — one element of `options`.
- For `record`: `object` — a fully-specified record.
- For `predicate`: usually omit. In some `type_mismatch` cases a canonical fix is knowable (see "Suggestions for type_mismatch" below).

Computed for `enum` by Levenshtein nearest-neighbor against `options`. Threshold: `max(2, ⌊len/3⌋)` edits. If no candidate is within threshold, omit `suggestion` rather than guess.

**Agent contract:** agents that don't parse `valid` should retry with `suggestion` when present, and surface `message` to the user when it isn't. Models that do parse `valid` may use it directly. This makes `suggestion` the single field small models need to read.

## Suggestions for `type_mismatch`

The `predicate` case doesn't trivially yield a suggestion, but for wiring type mismatches the input space is finite and enumerable: ~5 `PortType` tags × a small number of shape combinations = maybe 30-50 structural patterns. Canonical fixes per pattern live in `mcp/type_mismatch_fixes.json` (TBD), indexed by `(expected.tag, got.tag)` plus shape relation. The compiler looks up the pattern at error time and includes a `suggestion` with a structured fix description:

```json
{ "action": "insert_broadcast", "from_shape": [],  "to_shape": [4] }
{ "action": "no_automatic_fix", "reason": "length mismatch: [4] vs [8]" }
```

Until the table exists, `type_mismatch` errors ship without a `suggestion`.

## The `code` taxonomy

Codes are stable identifiers. Add new codes; never repurpose existing ones.

| Code | When | `valid` kind | `retryable` |
|---|---|---|---|
| `unknown_program` | `add_instance` / `replicate` get a name not in the registry | `enum` | false |
| `unknown_instance` | any tool gets an instance name that doesn't exist | `enum` | false |
| `instance_exists` | `add_instance` / `replicate` would shadow a name | — | false |
| `unknown_input` | wiring tool gets a port name not on the named instance | `enum` (scoped to that instance) | false |
| `unknown_output` | wiring tool gets an output name not on the named instance | `enum` (scoped to that instance) | false |
| `unknown_param` | `set_param` gets a name not in the param registry | `enum` | false |
| `invalid_type_args` | generic program instantiated with wrong/missing `type_args` | `record` | false |
| `type_mismatch` | wiring source and dest have incompatible `PortType` | `predicate` | false |
| `shape_mismatch` | array shapes don't broadcast | `predicate` | false |
| `length_mismatch` | `wire_zip` / similar: paired arrays differ in length | `predicate` | false |
| `arity_error` | too few / many args (e.g. `wire_chain` < 2 instances) | `predicate` | false |
| `missing_argument` | required tool arg absent | — | false |
| `invalid_value` | value out of allowed range or wrong primitive type | `record` or `predicate` | false |
| `compile_failed` | `applyFlatPlan` / JIT compilation error | — | false |
| `audio_error` | DAC open / device error | — | varies |
| `internal_error` | uncaught exception not classified above | — | false |

`retryable: true` is reserved for transient failures. Validation errors are never retryable — the args need to change.

## Suggestion computability

| Code | Suggestion computed by |
|---|---|
| `unknown_program`, `unknown_instance`, `unknown_input`, `unknown_output`, `unknown_param` | Levenshtein over `valid.options` |
| `instance_exists` | Append `_2`, `_3`, … until free |
| `invalid_type_args` | Clamp to nearest in-range value, or read default from `FieldSpec` |
| `invalid_value` | Clamp to range |
| `type_mismatch` | Lookup in `type_mismatch_fixes.json` (pending) |
| `shape_mismatch` | Expected shape (compiler knows it — that's why it failed) |
| `length_mismatch`, `arity_error`, `missing_argument`, `compile_failed`, `audio_error`, `internal_error` | No suggestion |

## Conversion rules (mechanical)

For each existing `throw new Error(...)` site:

1. Classify against the `code` table. If it doesn't fit, add a new code to the table *before* converting.
2. Identify `param` and `value` from the throw site's local variables.
3. Build `valid` per the `kind` for that code.
4. Compute `suggestion` per the computability table.
5. Replace `throw new Error(msg)` with `throw new ToolError({ code, message, retryable, param, value, valid, suggestion })`.
6. If the throw originates from a helper that throws plain `Error`, wrap the call site in `try/catch` and re-tag.

## Helper API

```ts
class ToolError extends Error { envelope: ErrorEnvelope }

function failEnum(opts: {
  code: ErrorCode, param: string, value: unknown,
  options: string[], message?: string,
}): never

function failRecord(opts: {
  code: ErrorCode, param: string, value: unknown,
  fields: Record<string, FieldSpec>, message?: string,
}): never

function failPredicate(opts: {
  code: ErrorCode, param: string, value: unknown,
  predicate: string, expected: unknown, got: unknown,
  suggestion?: unknown, message?: string,
}): never

function failBare(opts: {
  code: ErrorCode, message: string, retryable?: boolean,
  param?: string, value?: unknown,
}): never
```

If `message` is omitted the helper synthesizes one from the structured fields. `suggestion` is computed inside `failEnum` automatically.

## What this is NOT

- **Not a JSON Schema for tool inputs.** Tool input shape is declared in each tool's `inputSchema`. `valid` describes what's wrong about a *specific call*.
- **Not a description of the full search space.** `valid` is scoped to the one axis named by `code`. An agent that needs to reconsider multiple axes re-reads the session via `list_instances` / `list_programs` / `list_params`.
- **Not user-facing copy.** `message` is a fallback. The structured fields are the real interface.
