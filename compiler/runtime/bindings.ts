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
  const buildDirs = ['build-profile', 'build', 'build-jit-profile', 'build-jit-ctypes', 'build-jit', 'build-ctypes']
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

// ---------- ControlParam API ----------

export const egress_param_new          = lib.func('egress_param_new',          'void *', ['double', 'double'])
export const egress_param_free         = lib.func('egress_param_free',         'void',   ['void *'])
export const egress_param_set          = lib.func('egress_param_set',          'void',   ['void *', 'double'])
export const egress_param_get          = lib.func('egress_param_get',          'double', ['void *'])
export const egress_param_new_trigger  = lib.func('egress_param_new_trigger',  'void *', [])

// ---------- DAC API ----------

export const egress_dac_new_runtime     = lib.func('egress_dac_new_runtime',     'void *', ['void *', 'uint32', 'uint32'])
export const egress_dac_free             = lib.func('egress_dac_free',             'void',   ['void *'])
export const egress_dac_start            = lib.func('egress_dac_start',            'void',   ['void *'])
export const egress_dac_stop             = lib.func('egress_dac_stop',             'void',   ['void *'])
export const egress_dac_is_running       = lib.func('egress_dac_is_running',       'bool',   ['void *'])
export const egress_dac_get_stats        = lib.func('egress_dac_get_stats',        'void',   ['void *', koffi.out(koffi.pointer(EgressDacStats))])
export const egress_dac_reset_stats      = lib.func('egress_dac_reset_stats',      'void',   ['void *'])
export const egress_dac_is_reconnecting  = lib.func('egress_dac_is_reconnecting',  'bool',   ['void *'])
export const egress_dac_get_active_device = lib.func('egress_dac_get_active_device', 'uint32', ['void *'])
export const egress_dac_switch_device    = lib.func('egress_dac_switch_device',    'bool',   ['void *', 'uint32'])

// ---------- FlatRuntime API ----------

export const egress_runtime_new                    = lib.func('egress_runtime_new',                    'void *', ['uint32'])
export const egress_runtime_free                   = lib.func('egress_runtime_free',                   'void',   ['void *'])
export const egress_runtime_load_plan              = lib.func('egress_runtime_load_plan',              'bool',   ['void *', 'str', 'size_t'])
export const egress_runtime_process                = lib.func('egress_runtime_process',                'void',   ['void *'])
export const egress_runtime_output_buffer          = lib.func('egress_runtime_output_buffer',          'void *', ['void *'])
export const egress_runtime_get_buffer_length      = lib.func('egress_runtime_get_buffer_length',      'uint32', ['void *'])
export const egress_runtime_begin_fade_in          = lib.func('egress_runtime_begin_fade_in',          'void',   ['void *'])
export const egress_runtime_begin_fade_out         = lib.func('egress_runtime_begin_fade_out',         'void',   ['void *'])
export const egress_runtime_is_fade_out_complete   = lib.func('egress_runtime_is_fade_out_complete',   'bool',   ['void *'])

// ---------- Device enumeration ----------

export const egress_audio_device_count          = lib.func('egress_audio_device_count',          'uint32', [])
// Caller passes a pre-allocated Uint32Array(count); koffi writes device IDs into it.
export const egress_audio_get_device_ids        = lib.func('egress_audio_get_device_ids',        'void',   [koffi.pointer('uint32'), 'uint32'])
export const egress_audio_get_device_info       = lib.func('egress_audio_get_device_info',       'bool',   ['uint32', koffi.out(koffi.pointer(EgressDeviceInfo))])
export const egress_audio_default_output_device = lib.func('egress_audio_default_output_device', 'uint32', [])

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
 * Decode a `const double*` returned by the runtime output buffer functions
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
