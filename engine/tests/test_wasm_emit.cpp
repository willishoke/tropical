// test_wasm_emit — the IR→wasm toolchain sizes the kernel's shadow stack.
//
// The port kernel keeps its per-call storage in allocas, which wasm puts on
// the shadow stack. These tests compile hand-written kernels through the
// libtropical_wasm C entry point and read the linked module back: the stack
// must sit first (so an overflow traps instead of running into the data
// segment) and be at least as large as the kernel's storage, and the imported
// memory's declared minimum must cover it.
#include "toolchain/tropical_wasm.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::printf("FAIL\n    assertion failed: %s (line %d)\n",              \
                  #cond, __LINE__);                                           \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

static void run_test(const char * name, const std::function<void()> & fn)
{
  std::printf("  %-56s", name);
  std::fflush(stdout);
  const int failures_before = g_fail;
  fn();
  if (g_fail == failures_before)
  {
    std::printf("PASS\n");
    ++g_pass;
  }
}

static std::optional<std::vector<uint8_t>> compile(const std::string & ir, std::string & err)
{
  size_t len = 0;
  uint8_t * buf = tropical_wasm_compile_ir(ir.data(), ir.size(), &len);
  if (!buf) { err = tropical_wasm_last_error(); return std::nullopt; }
  std::vector<uint8_t> bytes(buf, buf + len);
  tropical_wasm_free(buf);
  return bytes;
}

// ---- a minimal reader for the two facts under test ------------------------

struct WasmFacts
{
  std::optional<uint32_t> memory_min_pages;   // the imported memory's minimum
  std::optional<int64_t>  stack_pointer_init; // global 0 (__stack_pointer)
};

static WasmFacts read_facts(const std::vector<uint8_t> & b)
{
  WasmFacts facts;
  size_t p = 8; // magic + version
  auto u32 = [&]() {
    uint32_t r = 0; int shift = 0; uint8_t byte;
    do { byte = b[p++]; r |= uint32_t(byte & 0x7f) << shift; shift += 7; } while (byte & 0x80);
    return r;
  };
  auto s64 = [&]() {
    int64_t r = 0; int shift = 0; uint8_t byte;
    do { byte = b[p++]; r |= int64_t(byte & 0x7f) << shift; shift += 7; } while (byte & 0x80);
    if (shift < 64 && (byte & 0x40)) r |= -(int64_t(1) << shift);
    return r;
  };
  auto limits_min = [&]() {
    const uint8_t flags = b[p++];
    const uint32_t min = u32();
    if (flags & 1) u32();
    return min;
  };
  while (p < b.size())
  {
    const uint8_t id = b[p++];
    const uint32_t size = u32();
    const size_t end = p + size;
    if (id == 2) // import
    {
      for (uint32_t n = u32(), i = 0; i < n; ++i)
      {
        p += u32(); p += u32(); // module, field
        switch (b[p++])
        {
          case 0: u32(); break;                         // func: type index
          case 1: ++p; limits_min(); break;             // table: reftype + limits
          case 2: facts.memory_min_pages = limits_min(); break;
          case 3: p += 2; break;                        // global: valtype + mut
        }
      }
    }
    else if (id == 6) // global
    {
      if (u32() > 0)
      {
        p += 2;                                         // valtype + mut
        if (b[p] == 0x41) { ++p; facts.stack_pointer_init = s64(); } // i32.const
      }
    }
    p = end;
  }
  return facts;
}

// ---- the kernels ----------------------------------------------------------

// 200,000 i64 cells (1.6 MB) of per-call storage, indexed by τ so codegen
// keeps the alloca: three times lld's default 64 KiB stack would hold.
static const char * kBigFrameIr = R"(
define void @tropical_kernel(ptr %tau, ptr %out) {
entry:
  %big = alloca i64, i64 200000, align 8
  %t = load i64, ptr %tau
  %i = and i64 %t, 131071
  %p = getelementptr inbounds i64, ptr %big, i64 %i
  store i64 7, ptr %p
  %v = load i64, ptr %p
  %f = sitofp i64 %v to double
  store double %f, ptr %out
  ret void
}
)";

static const char * kDynamicAllocaIr = R"(
define void @tropical_kernel(ptr %tau, ptr %out) {
entry:
  %n = load i64, ptr %tau
  %d = alloca i64, i64 %n, align 8
  store i64 0, ptr %d
  store double 0.0, ptr %out
  ret void
}
)";

static void test_stack_is_first_and_covers_the_frame()
{
  std::string err;
  const auto wasm = compile(kBigFrameIr, err);
  if (!wasm) std::printf("\n    %s\n", err.c_str());
  ASSERT(wasm.has_value());
  const WasmFacts facts = read_facts(*wasm);
  ASSERT(facts.stack_pointer_init.has_value());
  ASSERT(facts.memory_min_pages.has_value());
  const int64_t frame = 200000 * 8;
  // --stack-first: the stack is [0, sp) and grows down toward address 0, so
  // its initial pointer is its size, and it must hold the whole frame.
  ASSERT(*facts.stack_pointer_init >= frame);
  ASSERT(*facts.stack_pointer_init % 16 == 0);
  ASSERT(int64_t(*facts.memory_min_pages) * 65536 >= *facts.stack_pointer_init);
}

// 16 cells (128 bytes): the stack is the frame plus 64 KiB of headroom for
// the codegen's own frame, rounded up to 64 KiB.
static const char * kSmallFrameIr = R"(
define void @tropical_kernel(ptr %tau, ptr %out) {
entry:
  %small = alloca i64, i64 16, align 8
  %t = load i64, ptr %tau
  %i = and i64 %t, 15
  %p = getelementptr inbounds i64, ptr %small, i64 %i
  store i64 7, ptr %p
  %v = load i64, ptr %p
  %f = sitofp i64 %v to double
  store double %f, ptr %out
  ret void
}
)";

static void test_small_kernel_keeps_a_small_stack()
{
  std::string err;
  const auto wasm = compile(kSmallFrameIr, err);
  ASSERT(wasm.has_value());
  const WasmFacts facts = read_facts(*wasm);
  ASSERT(facts.stack_pointer_init.has_value());
  ASSERT(*facts.stack_pointer_init == 2 * 65536);
}

static void test_dynamic_alloca_is_refused()
{
  std::string err;
  const auto wasm = compile(kDynamicAllocaIr, err);
  ASSERT(!wasm.has_value());
  ASSERT(err.find("non-constant size") != std::string::npos);
}

int main()
{
  std::printf("test_wasm_emit\n");
  run_test("the stack is linked first and covers the kernel's frame",
           test_stack_is_first_and_covers_the_frame);
  run_test("a small frame gets 64 KiB of headroom, rounded up",
           test_small_kernel_keeps_a_small_stack);
  run_test("an alloca of non-constant size is refused",
           test_dynamic_alloca_is_refused);
  std::printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
