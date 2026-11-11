#pragma once

// The IR→wasm emitter: a toolchain concern, not a runtime one. It lives in
// libtropical_wasm, which only the build-time `diffcli compile-wasm` verb
// links; libtropical (the audio runtime) never sees LLVM's WebAssembly
// backend or lld.

#include <cstdint>
#include <string>
#include <vector>

namespace tropical_toolchain
{
// Lower textual LLVM IR (Lean's EmitLlvm output) to a complete wasm32 module,
// fully in-process: a wasm32 TargetMachine emits an object, lld::wasm::link
// assembles it into a module. No subprocess, no PATH — the same LLVM the JIT
// uses, which makes native≡wasm bit-exactness structural. Returns the .wasm
// bytes; on failure returns empty and sets `err`.
std::vector<uint8_t> compile_ir_to_wasm(const std::string & ir_text, std::string & err);

// Lower the same IR to one relocatable object for `triple` (empty = the
// host), stopping before any link: the artifact a firmware or a host build
// links itself. `cpu`/`features` default to the target's generic CPU. FP
// semantics are the IR's own (no `contract`/`fast` flags are emitted, and
// AllowFPOpFusion stays Standard, so no FMA is formed). Returns the object
// bytes; on failure returns empty and sets `err`.
std::vector<uint8_t> compile_ir_to_object(const std::string & ir_text,
                                          const std::string & triple,
                                          const std::string & cpu,
                                          const std::string & features,
                                          std::string & err);

// What `triple`/`cpu`/`features`'s FPU realizes, from LLVM's subtarget:
// `{"triple":…,"binary32":bool,"binary64":bool,"float_abi":"hard"|"soft"|"n/a"}`.
// On failure returns empty and sets `err`.
std::string target_float_info(const std::string & triple, const std::string & cpu,
                              const std::string & features, std::string & err);

// The undefined symbols of a relocatable object (ELF, Mach-O or wasm): what a
// link must supply. On failure returns empty and sets `err`.
std::vector<std::string> object_undefined_symbols(const uint8_t * data, size_t len,
                                                  std::string & err);
} // namespace tropical_toolchain
