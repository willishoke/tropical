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

extern "C" uint8_t* tropical_wasm_compile_object(const char* ir_text, size_t ir_len,
                                                 const char* triple, const char* cpu,
                                                 const char* features, size_t* out_len)
{
#ifdef TROPICAL_WASM_EMIT
  if (!ir_text || !out_len || !triple || !cpu || !features)
  { set_error("compile_object: null argument"); return nullptr; }
  try
  {
    std::string err;
    std::vector<uint8_t> bytes = tropical_toolchain::compile_ir_to_object(
      std::string(ir_text, ir_len), triple, cpu, features, err);
    if (bytes.empty())
    {
      set_error(err.empty() ? "compile_object: produced no output" : err);
      return nullptr;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (!buf) { set_error("compile_object: out of memory"); return nullptr; }
    std::memcpy(buf, bytes.data(), bytes.size());
    *out_len = bytes.size();
    return buf;
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
#else
  (void)ir_text; (void)ir_len; (void)triple; (void)cpu; (void)features;
  if (out_len) *out_len = 0;
  set_error("libtropical_wasm was built without TROPICAL_WASM_EMIT");
  return nullptr;
#endif
}

extern "C" void tropical_wasm_free(uint8_t* buf) { std::free(buf); }

[[maybe_unused]] static char * malloc_cstr(const std::string & s)
{
  char * out = static_cast<char *>(std::malloc(s.size() + 1));
  if (!out) { set_error("out of memory"); return nullptr; }
  std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

extern "C" char* tropical_wasm_target_float_info(const char* triple, const char* cpu,
                                                 const char* features)
{
#ifdef TROPICAL_WASM_EMIT
  if (!triple || !cpu || !features) { set_error("target_float_info: null argument"); return nullptr; }
  try
  {
    std::string err;
    const std::string info = tropical_toolchain::target_float_info(triple, cpu, features, err);
    if (info.empty()) { set_error(err.empty() ? "target_float_info: no answer" : err); return nullptr; }
    return malloc_cstr(info);
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
#else
  (void)triple; (void)cpu; (void)features;
  set_error("libtropical_wasm was built without TROPICAL_WASM_EMIT");
  return nullptr;
#endif
}

extern "C" char* tropical_wasm_object_undefined(const uint8_t* obj, size_t len)
{
#ifdef TROPICAL_WASM_EMIT
  if (!obj) { set_error("object_undefined: null argument"); return nullptr; }
  try
  {
    std::string err;
    const std::vector<std::string> names =
      tropical_toolchain::object_undefined_symbols(obj, len, err);
    if (!err.empty()) { set_error(err); return nullptr; }
    std::string joined;
    for (const auto & n : names) { joined += n; joined += '\n'; }
    return malloc_cstr(joined);
  }
  catch (const std::exception& e) { set_error(e.what()); return nullptr; }
#else
  (void)obj; (void)len;
  set_error("libtropical_wasm was built without TROPICAL_WASM_EMIT");
  return nullptr;
#endif
}
