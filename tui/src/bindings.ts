/**
 * koffi FFI bindings for libegress — direct port of egress/_bindings.py.
 * All opaque handles are typed as `unknown` in TypeScript (koffi externals).
 */

import koffi from 'koffi'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// ---------- Library loading ----------

function findLib(): string {
  // Mirror Python's _find_lib() search order.
  const here = path.resolve(__dirname, '../../egress')   // tui/src/ → egress/
  const parent = path.resolve(__dirname, '../..')         // tui/src/ → project root

  const allNames = ['libegress.dylib', 'libegress.so', 'egress.dll']
  const unixNames = ['libegress.dylib', 'libegress.so']

  // 1. Next to the egress Python package
  for (const name of allNames) {
    const candidate = path.join(here, name)
    if (fs.existsSync(candidate)) return candidate
  }

  // 2. Parent directory (common cmake build-tree layout)
  for (const name of allNames) {
    const candidate = path.join(parent, name)
    if (fs.existsSync(candidate)) return candidate
  }

  // 3. Common build subdirectories
  const buildDirs = ['build', 'build-jit-profile', 'build-jit-ctypes', 'build-jit', 'build-profile', 'build-ctypes']
  for (const buildDir of buildDirs) {
    for (const name of unixNames) {
      const candidate = path.join(parent, buildDir, name)
      if (fs.existsSync(candidate)) return candidate
    }
  }

  throw new Error(
    'libegress not found. Build with cmake (target egress_core) and ensure ' +
    'libegress.dylib/.so is on the library path or adjacent to the egress package directory.'
  )
}

export const lib = koffi.load(findLib())

// ---------- Struct types ----------

export const EgressDacStats = koffi.struct('EgressDacStats', {
  callback_count:  'uint64',
  avg_callback_ms: 'double',
  max_callback_ms: 'double',
  underrun_count:  'uint64',
  overrun_count:   'uint64',
})

export const EgressDeviceInfo = koffi.struct('EgressDeviceInfo', {
  id:                    'uint32',
  name:                  koffi.array('char', 256),
  output_channels:       'uint32',
  input_channels:        'uint32',
  is_default_output:     'bool',
  preferred_sample_rate: 'uint32',
  sample_rate_count:     'uint32',
  sample_rates:          koffi.array('uint32', 32),
})

// ---------- Error API ----------

export const egress_last_error = lib.func('egress_last_error', 'str', [])

// ---------- Value API ----------

export const egress_value_float    = lib.func('egress_value_float',    'void *', ['double'])
export const egress_value_int      = lib.func('egress_value_int',      'void *', ['int64'])
export const egress_value_bool     = lib.func('egress_value_bool',     'void *', ['bool'])
export const egress_value_array    = lib.func('egress_value_array',    'void *', ['void **', 'size_t'])
export const egress_value_matrix   = lib.func('egress_value_matrix',   'void *', ['void **', 'size_t', 'size_t'])
export const egress_value_to_float = lib.func('egress_value_to_float', 'double', ['void *'])
export const egress_value_to_int   = lib.func('egress_value_to_int',   'int64',  ['void *'])
export const egress_value_free     = lib.func('egress_value_free',     'void',   ['void *'])

// ---------- Expression factory API ----------

export const egress_expr_literal_float  = lib.func('egress_expr_literal_float',  'void *', ['double'])
export const egress_expr_literal_int    = lib.func('egress_expr_literal_int',    'void *', ['int64'])
export const egress_expr_literal_bool   = lib.func('egress_expr_literal_bool',   'void *', ['bool'])
export const egress_expr_literal_value  = lib.func('egress_expr_literal_value',  'void *', ['void *'])
export const egress_expr_input          = lib.func('egress_expr_input',          'void *', ['uint32'])
export const egress_expr_register       = lib.func('egress_expr_register',       'void *', ['uint32'])
export const egress_expr_nested_output  = lib.func('egress_expr_nested_output',  'void *', ['uint32', 'uint32'])
export const egress_expr_delay_value    = lib.func('egress_expr_delay_value',    'void *', ['uint32'])
export const egress_expr_ref            = lib.func('egress_expr_ref',            'void *', ['str', 'uint32'])
export const egress_expr_sample_rate    = lib.func('egress_expr_sample_rate',    'void *', [])
export const egress_expr_sample_index   = lib.func('egress_expr_sample_index',   'void *', [])
export const egress_expr_unary          = lib.func('egress_expr_unary',          'void *', ['int32', 'void *'])
export const egress_expr_binary         = lib.func('egress_expr_binary',         'void *', ['int32', 'void *', 'void *'])
export const egress_expr_clamp          = lib.func('egress_expr_clamp',          'void *', ['void *', 'void *', 'void *'])
export const egress_expr_select         = lib.func('egress_expr_select',         'void *', ['void *', 'void *', 'void *'])
export const egress_expr_array_pack     = lib.func('egress_expr_array_pack',     'void *', ['void **', 'size_t'])
export const egress_expr_index          = lib.func('egress_expr_index',          'void *', ['void *', 'void *'])
export const egress_expr_array_set      = lib.func('egress_expr_array_set',      'void *', ['void *', 'void *', 'void *'])
export const egress_expr_function       = lib.func('egress_expr_function',       'void *', ['uint32', 'void *'])
export const egress_expr_call           = lib.func('egress_expr_call',           'void *', ['void *', 'void **', 'size_t'])
export const egress_expr_free           = lib.func('egress_expr_free',           'void',   ['void *'])

// ---------- ADT expression constructors ----------

export const egress_expr_construct_struct  = lib.func('egress_expr_construct_struct',  'void *', ['str', 'void **', 'size_t'])
export const egress_expr_field_access      = lib.func('egress_expr_field_access',      'void *', ['str', 'void *', 'uint32'])
export const egress_expr_construct_variant = lib.func('egress_expr_construct_variant', 'void *', ['str', 'uint32', 'void **', 'size_t'])
export const egress_expr_match_variant     = lib.func('egress_expr_match_variant',     'void *', ['str', 'void *', 'void **', 'size_t'])

// ---------- Type definition API (graph-scoped) ----------

export const egress_typedef_struct = lib.func('egress_typedef_struct', 'bool', ['void *', 'str', 'void **', 'int *', 'size_t'])
export const egress_typedef_sum    = lib.func('egress_typedef_sum',    'bool', ['void *', 'str', 'void **', 'void **', 'int *', 'size_t *', 'size_t'])

// ---------- Port type annotation API ----------

export const egress_module_declare_input_type    = lib.func('egress_module_declare_input_type',    'bool', ['void *', 'str', 'uint32', 'str'])
export const egress_module_declare_output_type   = lib.func('egress_module_declare_output_type',   'bool', ['void *', 'str', 'uint32', 'str'])
export const egress_module_declare_register_type = lib.func('egress_module_declare_register_type', 'bool', ['void *', 'str', 'uint32', 'str'])

// ---------- ControlParam API ----------

export const egress_param_new          = lib.func('egress_param_new',          'void *', ['double', 'double'])
export const egress_param_free         = lib.func('egress_param_free',         'void',   ['void *'])
export const egress_param_set          = lib.func('egress_param_set',          'void',   ['void *', 'double'])
export const egress_param_get          = lib.func('egress_param_get',          'double', ['void *'])
export const egress_expr_param         = lib.func('egress_expr_param',         'void *', ['void *'])
export const egress_param_new_trigger  = lib.func('egress_param_new_trigger',  'void *', [])
export const egress_expr_trigger_param = lib.func('egress_expr_trigger_param', 'void *', ['void *'])

// ---------- Module spec builder API ----------

export const egress_module_spec_new                    = lib.func('egress_module_spec_new',                    'void *', ['uint32', 'double'])
export const egress_module_spec_add_output             = lib.func('egress_module_spec_add_output',             'void',   ['void *', 'void *'])
export const egress_module_spec_add_register           = lib.func('egress_module_spec_add_register',           'void',   ['void *', 'void *', 'void *'])
export const egress_module_spec_add_delay_state        = lib.func('egress_module_spec_add_delay_state',        'uint32', ['void *', 'void *', 'void *'])
export const egress_module_spec_add_nested             = lib.func('egress_module_spec_add_nested',             'void',   ['void *', 'void *'])
export const egress_module_spec_set_composite_schedule = lib.func('egress_module_spec_set_composite_schedule', 'void',   ['void *', koffi.pointer('uint32'), 'size_t'])
export const egress_module_spec_set_output_boundary    = lib.func('egress_module_spec_set_output_boundary',    'void',   ['void *', 'uint32'])
export const egress_module_spec_free                   = lib.func('egress_module_spec_free',                   'void',   ['void *'])

// ---------- Nested spec builder API ----------

export const egress_nested_spec_new                    = lib.func('egress_nested_spec_new',                    'void *', ['uint32', 'double'])
export const egress_nested_spec_node_id                = lib.func('egress_nested_spec_node_id',                'uint32', ['void *'])
export const egress_nested_spec_add_input_expr         = lib.func('egress_nested_spec_add_input_expr',         'void',   ['void *', 'void *'])
export const egress_nested_spec_add_output             = lib.func('egress_nested_spec_add_output',             'void',   ['void *', 'void *'])
export const egress_nested_spec_add_register           = lib.func('egress_nested_spec_add_register',           'void',   ['void *', 'void *', 'void *'])
export const egress_nested_spec_add_delay_state        = lib.func('egress_nested_spec_add_delay_state',        'uint32', ['void *', 'void *', 'void *'])
export const egress_nested_spec_add_nested             = lib.func('egress_nested_spec_add_nested',             'void',   ['void *', 'void *'])
export const egress_nested_spec_set_composite_schedule = lib.func('egress_nested_spec_set_composite_schedule', 'void',   ['void *', koffi.pointer('uint32'), 'size_t'])
export const egress_nested_spec_set_output_boundary    = lib.func('egress_nested_spec_set_output_boundary',    'void',   ['void *', 'uint32'])
export const egress_nested_spec_free                   = lib.func('egress_nested_spec_free',                   'void',   ['void *'])

// ---------- Graph API ----------

export const egress_graph_new               = lib.func('egress_graph_new',               'void *', ['uint32'])
export const egress_graph_free              = lib.func('egress_graph_free',              'void',   ['void *'])
export const egress_graph_add_module        = lib.func('egress_graph_add_module',        'bool',   ['void *', 'str', 'void *'])
export const egress_graph_remove_module     = lib.func('egress_graph_remove_module',     'bool',   ['void *', 'str'])
export const egress_graph_connect           = lib.func('egress_graph_connect',           'bool',   ['void *', 'str', 'uint32', 'str', 'uint32'])
export const egress_graph_disconnect        = lib.func('egress_graph_disconnect',        'bool',   ['void *', 'str', 'uint32', 'str', 'uint32'])
export const egress_graph_set_input_expr    = lib.func('egress_graph_set_input_expr',    'bool',   ['void *', 'str', 'uint32', 'void *'])
export const egress_graph_begin_update      = lib.func('egress_graph_begin_update',      'void',   ['void *'])
export const egress_graph_end_update        = lib.func('egress_graph_end_update',        'bool',   ['void *'])
export const egress_graph_get_input_expr    = lib.func('egress_graph_get_input_expr',    'void *', ['void *', 'str', 'uint32'])
export const egress_graph_add_output        = lib.func('egress_graph_add_output',        'bool',   ['void *', 'str', 'uint32'])
export const egress_graph_add_output_expr   = lib.func('egress_graph_add_output_expr',   'bool',   ['void *', 'void *'])
export const egress_graph_add_output_tap    = lib.func('egress_graph_add_output_tap',    'size_t', ['void *', 'str', 'uint32'])
export const egress_graph_remove_output_tap = lib.func('egress_graph_remove_output_tap', 'bool',   ['void *', 'size_t'])
export const egress_graph_process           = lib.func('egress_graph_process',           'void',   ['void *'])
export const egress_graph_prime_jit         = lib.func('egress_graph_prime_jit',         'void',   ['void *'])
// These return const double* — use decodeDoubleBuffer() to read samples
export const egress_graph_output_buffer     = lib.func('egress_graph_output_buffer',     'void *', ['void *'])
export const egress_graph_tap_buffer        = lib.func('egress_graph_tap_buffer',        'void *', ['void *', 'size_t', koffi.out(koffi.pointer('size_t'))])
export const egress_graph_set_worker_count  = lib.func('egress_graph_set_worker_count',  'void',   ['void *', 'uint32'])
export const egress_graph_get_worker_count  = lib.func('egress_graph_get_worker_count',  'uint32', ['void *'])
export const egress_graph_set_fusion_enabled = lib.func('egress_graph_set_fusion_enabled', 'void', ['void *', 'bool'])
export const egress_graph_get_fusion_enabled = lib.func('egress_graph_get_fusion_enabled', 'bool', ['void *'])
export const egress_graph_get_buffer_length          = lib.func('egress_graph_get_buffer_length',          'uint32', ['void *'])
export const egress_graph_get_profile_stats_json     = lib.func('egress_graph_get_profile_stats_json',     'string', ['void *'])
export const egress_graph_reset_profile_stats        = lib.func('egress_graph_reset_profile_stats',        'void',   ['void *'])

// ---------- DAC API ----------

export const egress_dac_new              = lib.func('egress_dac_new',              'void *', ['void *', 'uint32', 'uint32'])
export const egress_dac_free             = lib.func('egress_dac_free',             'void',   ['void *'])
export const egress_dac_start            = lib.func('egress_dac_start',            'void',   ['void *'])
export const egress_dac_stop             = lib.func('egress_dac_stop',             'void',   ['void *'])
export const egress_dac_is_running       = lib.func('egress_dac_is_running',       'bool',   ['void *'])
export const egress_dac_get_stats        = lib.func('egress_dac_get_stats',        'void',   ['void *', koffi.out(koffi.pointer(EgressDacStats))])
export const egress_dac_reset_stats      = lib.func('egress_dac_reset_stats',      'void',   ['void *'])
export const egress_dac_is_reconnecting  = lib.func('egress_dac_is_reconnecting',  'bool',   ['void *'])
export const egress_dac_get_active_device = lib.func('egress_dac_get_active_device', 'uint32', ['void *'])
export const egress_dac_switch_device    = lib.func('egress_dac_switch_device',    'bool',   ['void *', 'uint32'])

// ---------- Device enumeration ----------

export const egress_audio_device_count          = lib.func('egress_audio_device_count',          'uint32', [])
// Caller passes a pre-allocated Uint32Array(count); koffi writes device IDs into it.
export const egress_audio_get_device_ids        = lib.func('egress_audio_get_device_ids',        'void',   [koffi.pointer('uint32'), 'uint32'])
export const egress_audio_get_device_info       = lib.func('egress_audio_get_device_info',       'bool',   ['uint32', koffi.out(koffi.pointer(EgressDeviceInfo))])
export const egress_audio_default_output_device = lib.func('egress_audio_default_output_device', 'uint32', [])

// ---------- ExprKind constants ----------

export const EXPR_LITERAL        = 0
export const EXPR_REF            = 1
export const EXPR_INPUT          = 2
export const EXPR_REGISTER       = 3
export const EXPR_NESTED         = 4
export const EXPR_DELAY          = 5
export const EXPR_SAMPLE_RATE    = 6
export const EXPR_SAMPLE_INDEX   = 7
export const EXPR_FUNCTION       = 8
export const EXPR_CALL           = 9
export const EXPR_ARRAY_PACK     = 10
export const EXPR_INDEX          = 11
export const EXPR_ARRAY_SET      = 12
export const EXPR_NOT            = 13
export const EXPR_LESS           = 14
export const EXPR_LESS_EQUAL     = 15
export const EXPR_GREATER        = 16
export const EXPR_GREATER_EQUAL  = 17
export const EXPR_EQUAL          = 18
export const EXPR_NOT_EQUAL      = 19
export const EXPR_ADD            = 20
export const EXPR_SUB            = 21
export const EXPR_MUL            = 22
export const EXPR_DIV            = 23
export const EXPR_MATMUL         = 24
export const EXPR_POW            = 25
export const EXPR_MOD            = 26
export const EXPR_FLOOR_DIV      = 27
export const EXPR_BIT_AND        = 28
export const EXPR_BIT_OR         = 29
export const EXPR_BIT_XOR        = 30
export const EXPR_LSHIFT         = 31
export const EXPR_RSHIFT         = 32
export const EXPR_ABS            = 33
export const EXPR_CLAMP          = 34
export const EXPR_LOG            = 35
export const EXPR_SIN            = 36
export const EXPR_NEG            = 37
export const EXPR_BIT_NOT        = 38
export const EXPR_SMOOTHED_PARAM     = 39
export const EXPR_SELECT             = 40
export const EXPR_TRIGGER_PARAM      = 41
export const EXPR_CONSTRUCT_STRUCT   = 42
export const EXPR_FIELD_ACCESS       = 43
export const EXPR_CONSTRUCT_VARIANT  = 44
export const EXPR_MATCH_VARIANT      = 45

// ---------- Helpers ----------

/** Throw if handle is null, including the last C error message. */
export function check<T>(handle: T, name = 'operation'): NonNullable<T> {
  if (handle === null || handle === undefined) {
    const msg = egress_last_error()
    throw new Error(`${name} failed: ${msg ?? '(no error)'}`)
  }
  return handle as NonNullable<T>
}

/**
 * Decode a `const double*` returned by the graph output/tap buffer functions
 * into a copied Float64Array. The pointer is only valid until the next process() call.
 */
export function decodeDoubleBuffer(ptr: unknown, length: number): Float64Array {
  if (!ptr) return new Float64Array(0)
  const arr: number[] = koffi.decode(ptr, koffi.array('double', length))
  return new Float64Array(arr)
}

/**
 * Decode a fixed-size char[N] field (returned as a number array by koffi)
 * into a JS string, truncated at the first null byte.
 */
export function decodeCharArray(chars: number[]): string {
  const end = chars.indexOf(0)
  return String.fromCharCode(...(end === -1 ? chars : chars.slice(0, end)))
}
