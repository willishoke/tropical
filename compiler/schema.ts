/**
 * Zod schemas for runtime validation of JSON module definitions and patches.
 * Mirrors the TypeScript types in patch.ts but enforces structure at parse time.
 */

import { z } from 'zod'

// ─────────────────────────────────────────────────────────────
// ExprNode — recursive, so we use z.lazy
// ─────────────────────────────────────────────────────────────

const ExprOpNode: z.ZodType<{ op: string; [key: string]: unknown }> = z
  .object({ op: z.string() })
  .passthrough()

export const ExprNodeSchema: z.ZodType = z.lazy(() =>
  z.union([
    z.number(),
    z.boolean(),
    z.array(ExprNodeSchema),
    ExprOpNode,
  ]),
)

// ─────────────────────────────────────────────────────────────
// Bounds + PortType declaration (used by regs and ports)
// ─────────────────────────────────────────────────────────────

const BoundsSchema = z.tuple([z.number().nullable(), z.number().nullable()])

const ShapeDimSchema = z.union([
  z.number().int().nonnegative(),
  z.object({ op: z.literal('type_param'), name: z.string() }),
])

const ArrayTypeDeclSchema = z.object({
  kind: z.literal('array'),
  element: z.string(),
  shape: z.array(ShapeDimSchema).min(1),
})

/** A port/reg type declaration. Scalar/alias names are bare strings; array
 *  types must use the structured form. Old bracketed strings like "float[4]"
 *  are rejected with a clear error. */
const PortTypeDeclSchema = z.union([
  z.string().refine(s => !s.includes('['), {
    message: 'Bracketed array types like "float[4]" are not supported. Use {kind:"array", element:"float", shape:[4]}.',
  }),
  ArrayTypeDeclSchema,
])

// ─────────────────────────────────────────────────────────────
// ArrayStateJSON
// ─────────────────────────────────────────────────────────────

const ArrayStateJSONSchema = z.object({
  array_state: z.string(),
  init: z.number().optional(),
})

// ─────────────────────────────────────────────────────────────
// Register value — scalar, bool, array, nested array, or array_state
// ─────────────────────────────────────────────────────────────

/** Compact form: `{ zeros: N }` or `{ zeros: { type_param: 'N' } }`. */
const ZerosInitSchema = z.object({
  zeros: z.union([
    z.number().int().nonnegative(),
    z.object({ type_param: z.string() }),
  ]),
})

const RegValueSchema = z.union([
  z.number(),
  z.boolean(),
  z.array(z.number()),
  z.array(z.array(z.number())),
  ArrayStateJSONSchema,
  ZerosInitSchema,
])

/** Typed register entry: { init, type } */
const TypedRegValueSchema = z.object({
  init: RegValueSchema,
  type: PortTypeDeclSchema,
})

/** A register entry is either a bare value or a typed { init, type } object. */
const RegEntrySchema = z.union([RegValueSchema, TypedRegValueSchema])


// ─────────────────────────────────────────────────────────────
// ADT type definitions
// ─────────────────────────────────────────────────────────────

const StructFieldSchema = z.object({
  name: z.string(),
  scalar_type: z.number().int(),
})

const StructTypeDefSchema = z.object({
  kind: z.literal('struct'),
  name: z.string(),
  fields: z.array(StructFieldSchema),
})

const VariantPayloadFieldSchema = z.object({
  name: z.string(),
  scalar_type: z.number().int(),
})

const SumVariantSchema = z.object({
  name: z.string(),
  payload: z.array(VariantPayloadFieldSchema),
})

const SumTypeDefSchema = z.object({
  kind: z.literal('sum'),
  name: z.string(),
  variants: z.array(SumVariantSchema),
})

const AliasTypeDefSchema = z.object({
  kind: z.literal('alias'),
  name: z.string(),
  base: z.string(),
  bounds: BoundsSchema,
})

const TypeDefSchema = z.union([StructTypeDefSchema, SumTypeDefSchema, AliasTypeDefSchema])

// ─────────────────────────────────────────────────────────────
// ProgramJSON
// ─────────────────────────────────────────────────────────────

const ProgramInputSchema = z.union([
  z.string(),
  z.object({ name: z.string(), type: PortTypeDeclSchema.optional(), default: ExprNodeSchema.optional(), bounds: BoundsSchema.optional() }),
])

const ProgramOutputSchema = z.union([
  z.string(),
  z.object({ name: z.string(), type: PortTypeDeclSchema.optional(), bounds: BoundsSchema.optional() }),
])

const ProgramInstanceSchema = z.object({
  program: z.string(),
  inputs: z.record(z.string(), ExprNodeSchema).optional(),
  type_args: z.record(z.string(), z.union([z.number().int(), ExprNodeSchema])).optional(),
})

export const ProgramJSONSchema: z.ZodType = z.lazy(() => z.object({
  schema: z.literal('tropical_program_1'),
  name: z.string().min(1),
  inputs: z.array(ProgramInputSchema).optional(),
  outputs: z.array(ProgramOutputSchema).optional(),
  regs: z.record(z.string(), RegEntrySchema).optional(),
  delays: z.record(z.string(), z.object({
    update: ExprNodeSchema,
    init: z.number().optional(),
  })).optional(),
  sample_rate: z.number().positive().optional(),
  input_defaults: z.record(z.string(), ExprNodeSchema).optional(),
  programs: z.record(z.string(), ProgramJSONSchema).optional(),
  instances: z.record(z.string(), ProgramInstanceSchema).optional(),
  process: z.object({
    outputs: z.record(z.string(), ExprNodeSchema),
    next_regs: z.record(z.string(), ExprNodeSchema).optional(),
  }).optional(),
  audio_outputs: z.array(z.union([
    z.object({ instance: z.string(), output: z.union([z.string(), z.number()]) }),
    z.object({ expr: ExprNodeSchema }),
  ])).optional(),
  params: z.array(z.object({
    name: z.string(),
    value: z.number().optional(),
    time_const: z.number().optional(),
    type: z.enum(['param', 'trigger']).optional(),
  })).optional(),
  config: z.object({
    buffer_length: z.number().int().positive().optional(),
    sample_rate: z.number().positive().optional(),
  }).optional(),
  type_defs: z.array(TypeDefSchema).optional(),
  type_params: z.record(z.string(), z.object({
    type: z.literal('int'),
    default: z.number().int().optional(),
  })).optional(),
}))

// ─────────────────────────────────────────────────────────────
// Unified IR — tropical_program_2
// ─────────────────────────────────────────────────────────────
//
// A program is an ExprNode of op `program` whose `body` is a `block` of
// `decls` (reg_decl, delay_decl, instance_decl, program_decl) and `assigns`
// (output_assign, next_update). Session-level metadata — `params`,
// `audio_outputs`, `config` — sits on the file root alongside the program
// fields but is not part of the program definition itself.

const ProgramPortsSchema = z.object({
  inputs: z.array(ProgramInputSchema).optional(),
  outputs: z.array(ProgramOutputSchema).optional(),
  type_defs: z.array(TypeDefSchema).optional(),
})

const TypeParamsSchema = z.record(z.string(), z.object({
  type: z.literal('int'),
  default: z.number().int().optional(),
}))

const BlockNodeSchema = z.object({
  op: z.literal('block'),
  decls: z.array(ExprNodeSchema).optional(),
  assigns: z.array(ExprNodeSchema).optional(),
  value: ExprNodeSchema.nullable().optional(),
})

/** Schema for a `program` ExprNode. Appears at the file root (together with
 *  a `schema` tag) and inside nested `program_decl` nodes. */
export const ProgramNodeSchema: z.ZodType = z.lazy(() => z.object({
  op: z.literal('program'),
  name: z.string().min(1),
  type_params: TypeParamsSchema.optional(),
  sample_rate: z.number().positive().optional(),
  breaks_cycles: z.boolean().optional(),
  ports: ProgramPortsSchema.optional(),
  body: BlockNodeSchema,
}))

/** Schema for an on-disk tropical_program_2 file — program fields plus the
 *  session metadata (params, audio_outputs, config) that only applies at the
 *  top level. Does not carry an `op` field (the schema tag implies `program`). */
export const ProgramFileSchemaV2: z.ZodType = z.lazy(() => z.object({
  schema: z.literal('tropical_program_2'),
  name: z.string().min(1),
  type_params: TypeParamsSchema.optional(),
  sample_rate: z.number().positive().optional(),
  breaks_cycles: z.boolean().optional(),
  ports: ProgramPortsSchema.optional(),
  body: BlockNodeSchema,
  params: z.array(z.object({
    name: z.string(),
    value: z.number().optional(),
    time_const: z.number().optional(),
    type: z.enum(['param', 'trigger']).optional(),
  })).optional(),
  audio_outputs: z.array(z.union([
    z.object({ instance: z.string(), output: z.union([z.string(), z.number()]) }),
    z.object({ expr: ExprNodeSchema }),
  ])).optional(),
  config: z.object({
    buffer_length: z.number().int().positive().optional(),
    sample_rate: z.number().positive().optional(),
  }).optional(),
}))

// ─────────────────────────────────────────────────────────────
// Validation helpers
// ─────────────────────────────────────────────────────────────

/** Format Zod errors into a readable string. */
function formatZodError(error: z.ZodError): string {
  return error.issues.map(issue => {
    const path = issue.path.length > 0 ? `at '${issue.path.join('.')}': ` : ''
    return `${path}${issue.message}`
  }).join('; ')
}

/** Parse and validate a ProgramJSON, throwing a readable error on failure. */
export function parseProgram(raw: unknown): z.infer<typeof ProgramJSONSchema> {
  const result = ProgramJSONSchema.safeParse(raw)
  if (!result.success) throw new Error(`Invalid program: ${formatZodError(result.error)}`)
  return result.data
}

/** Parse and validate a v2 program file (tropical_program_2). */
export function parseProgramV2(raw: unknown): z.infer<typeof ProgramFileSchemaV2> {
  const result = ProgramFileSchemaV2.safeParse(raw)
  if (!result.success) throw new Error(`Invalid program (v2): ${formatZodError(result.error)}`)
  return result.data as z.infer<typeof ProgramFileSchemaV2>
}
