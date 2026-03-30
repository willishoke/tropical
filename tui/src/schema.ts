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
  regs: z.record(z.string(), RegValueSchema).optional(),
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

export const PatchJSONSchema = z.object({
  schema: z.literal('egress_patch_1'),
  config: z.object({
    buffer_length: z.number().int().positive().optional(),
    worker_count: z.number().int().positive().optional(),
    fusion_enabled: z.boolean().optional(),
  }).optional(),
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
