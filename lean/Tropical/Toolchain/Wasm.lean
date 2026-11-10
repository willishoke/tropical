/-!
# The wasm toolchain binding

`@[extern]` bindings over `lean/ffi/shim_wasm.c`, which wraps
`engine/toolchain/tropical_wasm.h` (libtropical_wasm). This is the compiler's
*toolchain* side: it turns emitted LLVM IR into a wasm32 module and knows
nothing about the audio runtime. Only `diffcli` links it; `Tropical.Ffi`
(the runtime binding) does not reach it, so the runtime binaries carry
neither LLVM's WebAssembly backend nor lld.
-/

namespace Tropical.Toolchain.Wasm

@[extern "shim_wasm_last_error"]
private opaque lastError : IO String

/-- Lower textual LLVM IR (Lean's EmitLlvm output) to a complete wasm32 module,
    in-process via the toolchain library's LLVM + lld. Empty result ⇒ failure;
    requires libtropical_wasm built with `TROPICAL_WASM_EMIT`. -/
@[extern "shim_wasm_compile_ir"]
private opaque compileIrRaw (irText : @& String) : IO ByteArray

def compileIr (irText : String) : IO ByteArray := do
  let wasm ← compileIrRaw irText
  if wasm.size == 0 then
    throw <| IO.userError s!"compileIr failed: {← lastError}"
  pure wasm

@[extern "shim_wasm_compile_object"]
private opaque compileObjectRaw (irText triple cpu features : @& String) : IO ByteArray

/-- Lower the same IR to one relocatable object for `triple` (`""` = the
    host), stopping before any link. `cpu`/`features` as LLVM spells them
    (`""` = generic). The bytes are an ELF/Mach-O/wasm object a firmware or a
    host build links itself. -/
def compileObject (irText triple : String) (cpu features : String := "") : IO ByteArray := do
  let obj ← compileObjectRaw irText triple cpu features
  if obj.size == 0 then
    throw <| IO.userError s!"compileObject failed: {← lastError}"
  pure obj

end Tropical.Toolchain.Wasm
