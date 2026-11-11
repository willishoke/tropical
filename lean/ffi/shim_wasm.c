/*
 * shim_wasm.c — Lean-ABI wrappers over engine/toolchain/tropical_wasm.h.
 *
 * The toolchain shim is a separate object from shim.c on purpose: it is
 * linked only by `diffcli` (the one binary that carries the IR→wasm
 * toolchain), so the runtime binaries never reference libtropical_wasm.
 * Same conventions as shim.c.
 */

#include <lean/lean.h>
#include "tropical_wasm.h"

extern size_t strlen(const char *);
extern void *memcpy(void *, const void *, size_t);

LEAN_EXPORT lean_obj_res shim_wasm_last_error(lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_mk_string(tropical_wasm_last_error()));
}

/* In-process IR→wasm. Empty ByteArray signals failure; Lean reads
   tropical_wasm_last_error. */
LEAN_EXPORT lean_obj_res shim_wasm_compile_ir(b_lean_obj_arg ir, lean_obj_arg world) {
  (void)world;
  const char *i = lean_string_cstr(ir);
  size_t out_len = 0;
  uint8_t *buf = tropical_wasm_compile_ir(i, strlen(i), &out_len);
  lean_obj_res arr;
  if (buf) {
    arr = lean_alloc_sarray(1, out_len, out_len);
    memcpy(lean_sarray_cptr(arr), buf, out_len);
    tropical_wasm_free(buf);
  } else {
    arr = lean_alloc_sarray(1, 0, 0);
  }
  return lean_io_result_mk_ok(arr);
}

/* What a target's FPU realizes, as JSON text. "" signals failure; Lean reads
   tropical_wasm_last_error. */
LEAN_EXPORT lean_obj_res shim_wasm_target_float_info(b_lean_obj_arg triple, b_lean_obj_arg cpu,
                                                     b_lean_obj_arg features,
                                                     lean_obj_arg world) {
  (void)world;
  char *s = tropical_wasm_target_float_info(lean_string_cstr(triple), lean_string_cstr(cpu),
                                            lean_string_cstr(features));
  lean_obj_res out = lean_mk_string(s ? s : "");
  if (s) tropical_wasm_free((uint8_t *)s);
  return lean_io_result_mk_ok(out);
}

/* The undefined symbols of an object, one per line. A NUL byte alone signals
   failure (an empty string is "no symbols"); Lean reads
   tropical_wasm_last_error. */
LEAN_EXPORT lean_obj_res shim_wasm_object_undefined(b_lean_obj_arg obj, lean_obj_arg world) {
  (void)world;
  char *s = tropical_wasm_object_undefined(lean_sarray_cptr(obj), lean_sarray_size(obj));
  lean_obj_res out = s ? lean_mk_string(s) : lean_mk_string("\x01");
  if (s) tropical_wasm_free((uint8_t *)s);
  return lean_io_result_mk_ok(out);
}

/* In-process IR→relocatable object for a triple. Empty ByteArray signals
   failure; Lean reads tropical_wasm_last_error. */
LEAN_EXPORT lean_obj_res shim_wasm_compile_object(b_lean_obj_arg ir, b_lean_obj_arg triple,
                                                  b_lean_obj_arg cpu, b_lean_obj_arg features,
                                                  lean_obj_arg world) {
  (void)world;
  const char *i = lean_string_cstr(ir);
  size_t out_len = 0;
  uint8_t *buf = tropical_wasm_compile_object(i, strlen(i), lean_string_cstr(triple),
                                              lean_string_cstr(cpu),
                                              lean_string_cstr(features), &out_len);
  lean_obj_res arr;
  if (buf) {
    arr = lean_alloc_sarray(1, out_len, out_len);
    memcpy(lean_sarray_cptr(arr), buf, out_len);
    tropical_wasm_free(buf);
  } else {
    arr = lean_alloc_sarray(1, 0, 0);
  }
  return lean_io_result_mk_ok(arr);
}
