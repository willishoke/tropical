#include "toolchain/tropical_wasm.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#ifdef TROPICAL_WASM_EMIT
#include "toolchain/WasmEmit.hpp"
#endif

static thread_local std::string tls_last_error;

static void set_error(const std::string & msg) { tls_last_error = msg; }

extern "C" const char* tropical_wasm_last_error(void)
{
  return tls_last_error.c_str();
}

extern "C" uint8_t* tropical_wasm_compile_ir(const char* ir_text, size_t ir_len, size_t* out_len)
{
#ifdef TROPICAL_WASM_EMIT
  if (!ir_text || !out_len) { set_error("wasm_compile_ir: null argument"); return nullptr; }
  try
  {
    std::string err;
    std::vector<uint8_t> bytes =
      tropical_toolchain::compile_ir_to_wasm(std::string(ir_text, ir_len), err);
    if (bytes.empty())
    {
      set_error(err.empty() ? "wasm_compile_ir: produced no output" : err);
      return nullptr;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (!buf) { set_error("wasm_compile_ir: out of memory"); return nullptr; }
    std::memcpy(buf, bytes.data(), bytes.size());
    *out_len = bytes.size();
    return buf;
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
#else
  (void)ir_text; (void)ir_len;
  if (out_len) *out_len = 0;
  set_error("libtropical_wasm was built without TROPICAL_WASM_EMIT");
  return nullptr;
#endif
}

extern "C" void tropical_wasm_free(uint8_t* buf) { std::free(buf); }
