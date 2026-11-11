#ifndef TROPICAL_WASM_H
#define TROPICAL_WASM_H

/* tropical_wasm.h — the IR→wasm toolchain, as a C API.
 *
 * This is libtropical_wasm, a library separate from libtropical (the audio
 * runtime). It lowers Lean's LLVM IR to a complete wasm32 module in-process
 * (wasm32 TargetMachine + lld-as-a-library — no subprocess), which is what
 * makes native≡wasm bit-exactness structural. It is a build-time tool: only
 * `diffcli compile-wasm` (the web build) links it; the shipped runtime and
 * every other binary carry neither LLVM's WebAssembly backend nor lld.
 *
 * Errors are thread-local to this library (tropical_wasm_last_error), not
 * shared with libtropical's tropical_last_error.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Thread-local description of the last failure in this library. */
const char* tropical_wasm_last_error(void);

/* Lower LLVM IR text to a wasm32 module. Returns a malloc'd buffer of length
   *out_len (free with tropical_wasm_free), or NULL on error (see
   tropical_wasm_last_error()). Returns NULL if the library was built without
   TROPICAL_WASM_EMIT. */
uint8_t* tropical_wasm_compile_ir(const char* ir_text, size_t ir_len, size_t* out_len);
/* Lower LLVM IR text to one relocatable object for `triple` ("" = host), with
   optional `cpu` and `features` ("" = generic). No link. Same buffer and error
   conventions as tropical_wasm_compile_ir. */
uint8_t* tropical_wasm_compile_object(const char* ir_text, size_t ir_len,
                                      const char* triple, const char* cpu,
                                      const char* features, size_t* out_len);
void     tropical_wasm_free(uint8_t* buf);

/* What the FPU of `triple` ("" = host) with `cpu`/`features` realizes, as a
   malloc'd NUL-terminated JSON object:
   {"triple":…,"binary32":bool,"binary64":bool,"float_abi":"hard"|"soft"|"n/a"}.
   Free with tropical_wasm_free. NULL on error. */
char*    tropical_wasm_target_float_info(const char* triple, const char* cpu,
                                         const char* features);
/* The undefined symbols of a relocatable object, one per line, as a malloc'd
   NUL-terminated string ("" when there are none). Free with
   tropical_wasm_free. NULL on error. */
char*    tropical_wasm_object_undefined(const uint8_t* obj, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* TROPICAL_WASM_H */
