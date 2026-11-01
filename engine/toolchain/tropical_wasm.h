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
void     tropical_wasm_free(uint8_t* buf);

#ifdef __cplusplus
}
#endif

#endif /* TROPICAL_WASM_H */
