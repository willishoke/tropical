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
// ArrayStateJSON
// ─────────────────────────────────────────────────────────────

const ArrayStateJSONSchema = z.object({
  array_state: z.string(),
  init: z.number().optional(),
})

// ─────────────────────────────────────────────────────────────
// Register value — scalar, bool, array, nested array, or array_state
// ─────────────────────────────────────────────────────────────

const RegValueSchema = z.union([
  z.number(),
  z.boolean(),
  z.array(z.number()),
  z.array(z.array(z.number())),
  ArrayStateJSONSchema,
])

/** Typed register entry: { init, type } */
const TypedRegValueSchema = z.object({
  init: RegValueSchema,
  type: z.string(),
})

/** A register entry is either a bare value or a typed { init, type } object. */
const RegEntrySchema = z.union([RegValueSchema, TypedRegValueSchema])

// ─────────────────────────────────────────────────────────────
// NestedModuleJSON
// ─────────────────────────────────────────────────────────────

const NestedModuleJSONSchema = z.object({
  type: z.string(),
  inputs: z.record(z.string(), ExprNodeSchema),
})

// ─────────────────────────────────────────────────────────────
// ModuleDefJSON
// ─────────────────────────────────────────────────────────────

export const ModuleDefJSONSchema = z.object({
  name: z.string().min(1, 'Module name must be a non-empty string'),
  inputs: z.array(z.string()),
  outputs: z.array(z.string()).min(1, 'Module must have at least one output'),
  regs: z.record(z.string(), RegEntrySchema).optional(),
  delays: z.record(z.string(), z.object({
    update: ExprNodeSchema,
    init: z.number().optional(),
  })).optional(),
  nested: z.record(z.string(), NestedModuleJSONSchema).optional(),
  sample_rate: z.number().positive().optional(),
  input_defaults: z.record(z.string(), ExprNodeSchema).optional(),
  process: z.object({
    outputs: z.record(z.string(), ExprNodeSchema),
    next_regs: z.record(z.string(), ExprNodeSchema).optional(),
  }),
})

// ─────────────────────────────────────────────────────────────
// PatchJSON
// ─────────────────────────────────────────────────────────────

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

const TypeDefSchema = z.union([StructTypeDefSchema, SumTypeDefSchema])

// ─────────────────────────────────────────────────────────────
// PatchJSON
// ─────────────────────────────────────────────────────────────

export const PatchJSONSchema = z.object({
  schema: z.literal('tropical_patch_1'),
  config: z.object({
    buffer_length: z.number().int().positive().optional(),
    worker_count: z.number().int().positive().optional(),
    fusion_enabled: z.boolean().optional(),
  }).optional(),
  type_defs: z.array(TypeDefSchema).optional(),
  module_defs: z.array(ModuleDefJSONSchema).optional(),
  modules: z.array(z.object({
    type: z.string(),
    name: z.string().optional(),
  })),
  connections: z.array(z.object({
    src: z.string(),
    src_output: z.union([z.string(), z.number()]),
    dst: z.string(),
    dst_input: z.union([z.string(), z.number()]),
  })).optional(),
  outputs: z.array(z.union([
    z.object({ module: z.string(), output: z.union([z.string(), z.number()]) }),
    z.object({ expr: ExprNodeSchema }),
  ])).optional(),
  params: z.array(z.object({
    name: z.string(),
    value: z.number().optional(),
    time_const: z.number().optional(),
    type: z.enum(['param', 'trigger']).optional(),
  })).optional(),
  input_exprs: z.array(z.object({
    module: z.string(),
    input: z.union([z.string(), z.number()]),
    expr: ExprNodeSchema,
  })).optional(),
})

// ─────────────────────────────────────────────────────────────
// ProgramJSON
// ─────────────────────────────────────────────────────────────

const ProgramInputSchema = z.union([
  z.string(),
  z.object({ name: z.string(), type: z.string().optional(), default: ExprNodeSchema.optional() }),
])

const ProgramOutputSchema = z.union([
  z.string(),
  z.object({ name: z.string(), type: z.string().optional() }),
])

const ProgramInstanceSchema = z.object({
  program: z.string(),
  inputs: z.record(z.string(), ExprNodeSchema).optional(),
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

/** Parse and validate a ModuleDefJSON, throwing a readable error on failure. */
export function parseModuleDef(raw: unknown): z.infer<typeof ModuleDefJSONSchema> {
  const result = ModuleDefJSONSchema.safeParse(raw)
  if (!result.success) throw new Error(`Invalid module definition: ${formatZodError(result.error)}`)
  return result.data
}

/** Parse and validate a PatchJSON, throwing a readable error on failure. */
export function parsePatch(raw: unknown): z.infer<typeof PatchJSONSchema> {
  const result = PatchJSONSchema.safeParse(raw)
  if (!result.success) throw new Error(`Invalid patch: ${formatZodError(result.error)}`)
  return result.data
}

/** Parse and validate a ProgramJSON, throwing a readable error on failure. */
export function parseProgram(raw: unknown): z.infer<typeof ProgramJSONSchema> {
  const result = ProgramJSONSchema.safeParse(raw)
  if (!result.success) throw new Error(`Invalid program: ${formatZodError(result.error)}`)
  return result.data
}
