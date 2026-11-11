import Lean.Data.Json

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

@[extern "shim_wasm_target_float_info"]
private opaque targetFloatInfoRaw (triple cpu features : @& String) : IO String

/-- What a target's FPU realizes, from LLVM's own subtarget description. -/
structure FloatHardware where
  /-- The triple LLVM resolved (`""` resolves to the host's). -/
  triple : String
  /-- binary32 arithmetic in hardware. -/
  binary32 : Bool
  /-- binary64 arithmetic in hardware. -/
  binary64 : Bool
  /-- `hard` (VFP registers) or `soft` on ARM; `n/a` elsewhere. -/
  floatAbi : String
deriving Repr

def targetFloatHardware (triple : String) (cpu features : String := "") :
    IO FloatHardware := do
  let text ← targetFloatInfoRaw triple cpu features
  if text.isEmpty then
    throw <| IO.userError s!"targetFloatHardware failed: {← lastError}"
  match Lean.Json.parse text with
  | .error e => throw <| IO.userError s!"targetFloatHardware: {e}"
  | .ok j =>
    let b (k : String) : Bool := match j.getObjValAs? Bool k with | .ok v => v | .error _ => false
    let s (k : String) : String := match j.getObjValAs? String k with | .ok v => v | .error _ => ""
    pure { triple := s "triple", binary32 := b "binary32", binary64 := b "binary64",
           floatAbi := s "float_abi" }

@[extern "shim_wasm_object_undefined"]
private opaque objectUndefinedRaw (obj : @& ByteArray) : IO String

/-- The undefined symbols of a relocatable object: what its link must
    supply. For a port kernel each one is a runtime routine (a libcall). -/
def objectUndefinedSymbols (obj : ByteArray) : IO (Array String) := do
  let text ← objectUndefinedRaw obj
  if text == "\x01" then
    throw <| IO.userError s!"objectUndefinedSymbols failed: {← lastError}"
  pure ((text.splitOn "\n").filter (!·.isEmpty)).toArray

end Tropical.Toolchain.Wasm
