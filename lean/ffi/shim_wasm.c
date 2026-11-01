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
