// test_rounding_f32 — the binary32 floor/ceil/round sequences, exhaustively.
//
// A binary32 artifact (`--float f32`) expands floor, ceil and round inline
// (`EmitLlvm.roundingF32`): FPv4-SP, the Cortex-M4F's FPU, has no rounding
// instruction, so the intrinsics would become `floorf`/`ceilf`/`roundf`
// calls. `tests/fixtures/rounding_f32.ll` is the emitter's own text for the
// three sequences (`EmitLlvm.roundingF32Fixture`; tropicaltest's
// `f32-rounding-fixture` gate holds the file to it), wrapped in a loop over a
// buffer. This test compiles it through the production JIT (O2, the host's
// TargetMachine) and runs it on every one of the 2^32 binary32 bit patterns,
// comparing each result with the C library's floorf/ceilf/roundf:
//
//   * every non-NaN result must have the reference's exact bits (signed
//     zeros included: floor(-0) = -0, ceil(-0.5) = -0, round(-0.3) = -0);
//   * every NaN input must give a NaN (the payload is not part of the
//     numeric contract; the sequence returns the input itself).
//
// Halfway cases (x = n + 1/2), infinities, subnormals and the 2^23 boundary
// are all inside the 2^32 inputs; the test counts them so the report says
// they were reached.
#include "jit/OrcJitEngine.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef TROPICAL_SOURCE_DIR
#error "TROPICAL_SOURCE_DIR must name the repository root"
#endif

using RoundingFn = void (*)(const float * in, float * floor_out, float * ceil_out,
                            float * round_out, int64_t n);

namespace
{

uint32_t bits_of(float x)
{
  uint32_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

float float_of(uint32_t b)
{
  float x;
  std::memcpy(&x, &b, sizeof x);
  return x;
}

struct Tally
{
  uint64_t checked = 0;
  uint64_t nans = 0;
  uint64_t infinities = 0;
  uint64_t zeros = 0;
  uint64_t subnormals = 0;
  uint64_t halfway = 0;      // |x| < 2^23 and x - trunc(x) = ±1/2
  uint64_t failures = 0;
  std::string first_failure;
};

// One result against its reference: the same bits, or both NaN.
bool agrees(float got, float want)
{
  if (std::isnan(want)) return std::isnan(got);
  return bits_of(got) == bits_of(want);
}

void check_range(RoundingFn fn, uint64_t begin, uint64_t end, Tally & t)
{
  constexpr uint64_t kChunk = uint64_t{1} << 18;
  std::vector<float> in(kChunk), fl(kChunk), ce(kChunk), ro(kChunk);
  for (uint64_t base = begin; base < end; base += kChunk)
  {
    const uint64_t n = std::min(kChunk, end - base);
    for (uint64_t k = 0; k < n; ++k) in[k] = float_of(static_cast<uint32_t>(base + k));
    fn(in.data(), fl.data(), ce.data(), ro.data(), static_cast<int64_t>(n));
    for (uint64_t k = 0; k < n; ++k)
    {
      const float x = in[k];
      ++t.checked;
      if (std::isnan(x)) ++t.nans;
      else if (std::isinf(x)) ++t.infinities;
      else if (x == 0.0f) ++t.zeros;
      else if (std::fpclassify(x) == FP_SUBNORMAL) ++t.subnormals;
      if (std::fabs(x) < 8388608.0f && std::fabs(x - std::trunc(x)) == 0.5f) ++t.halfway;
      const float want[3] = {std::floor(x), std::ceil(x), std::round(x)};
      const float got[3] = {fl[k], ce[k], ro[k]};
      static const char * const names[3] = {"floor", "ceil", "round"};
      for (int op = 0; op < 3; ++op)
      {
        if (agrees(got[op], want[op])) continue;
        if (t.failures++ == 0)
        {
          char buf[160];
          std::snprintf(buf, sizeof buf, "%s(0x%08x = %.9g) = 0x%08x (%.9g), libm 0x%08x (%.9g)",
                        names[op], bits_of(x), static_cast<double>(x), bits_of(got[op]),
                        static_cast<double>(got[op]), bits_of(want[op]),
                        static_cast<double>(want[op]));
          t.first_failure = buf;
        }
      }
    }
  }
}

} // namespace

int main()
{
  const std::string path = std::string(TROPICAL_SOURCE_DIR) + "/tests/fixtures/rounding_f32.ll";
  std::ifstream file(path);
  if (!file)
  {
    std::printf("FAIL  cannot read %s\n", path.c_str());
    return 1;
  }
  std::stringstream text;
  text << file.rdbuf();

  auto & jit = tropical_jit::OrcJitEngine::instance();
  if (!jit.available())
  {
    std::printf("FAIL  ORC JIT unavailable: %s\n", jit.init_error().c_str());
    return 1;
  }
  auto compiled = jit.compile_ir_text(text.str(), false, "tropical_rounding_f32");
  if (!compiled)
  {
    std::printf("FAIL  compile %s: %s\n", path.c_str(),
                llvm::toString(compiled.takeError()).c_str());
    return 1;
  }
  const auto fn = reinterpret_cast<RoundingFn>(*compiled);

  // Spot checks first, so a broken sequence names a familiar input.
  struct Case { float x, floor, ceil, round; };
  const Case cases[] = {
    {-0.0f, -0.0f, -0.0f, -0.0f}, {0.0f, 0.0f, 0.0f, 0.0f},
    {-0.5f, -1.0f, -0.0f, -1.0f}, {0.5f, 0.0f, 1.0f, 1.0f},
    {-0.3f, -1.0f, -0.0f, -0.0f}, {2.5f, 2.0f, 3.0f, 3.0f},
    {-2.5f, -3.0f, -2.0f, -3.0f}, {8388607.5f, 8388607.0f, 8388608.0f, 8388608.0f},
    {-8388607.5f, -8388608.0f, -8388607.0f, -8388608.0f},
    {8388608.0f, 8388608.0f, 8388608.0f, 8388608.0f},
    {1e-45f, 0.0f, 1.0f, 0.0f}, {-1e-45f, -1.0f, -0.0f, -0.0f},
  };
  int spot_failures = 0;
  for (const Case & c : cases)
  {
    float fl, ce, ro;
    fn(&c.x, &fl, &ce, &ro, 1);
    if (bits_of(fl) != bits_of(c.floor) || bits_of(ce) != bits_of(c.ceil)
        || bits_of(ro) != bits_of(c.round))
    {
      std::printf("FAIL  spot %.9g: floor %.9g ceil %.9g round %.9g\n",
                  static_cast<double>(c.x), static_cast<double>(fl),
                  static_cast<double>(ce), static_cast<double>(ro));
      ++spot_failures;
    }
  }
  if (spot_failures) return 1;

  // All 2^32 inputs, split over at most four threads.
  const unsigned threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
  const uint64_t total = uint64_t{1} << 32;
  std::vector<Tally> tallies(threads);
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < threads; ++i)
    pool.emplace_back([&, i] {
      check_range(fn, total * i / threads, total * (i + 1) / threads, tallies[i]);
    });
  for (auto & th : pool) th.join();

  Tally sum;
  for (const Tally & t : tallies)
  {
    sum.checked += t.checked; sum.nans += t.nans; sum.infinities += t.infinities;
    sum.zeros += t.zeros; sum.subnormals += t.subnormals; sum.halfway += t.halfway;
    if (t.failures && sum.first_failure.empty()) sum.first_failure = t.first_failure;
    sum.failures += t.failures;
  }
  if (sum.checked != total || sum.failures != 0)
  {
    std::printf("FAIL  %llu of %llu inputs checked, %llu mismatches; first: %s\n",
                static_cast<unsigned long long>(sum.checked),
                static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(sum.failures), sum.first_failure.c_str());
    return 1;
  }
  std::printf("PASS  floor/ceil/round at binary32 = libm on all %llu inputs "
              "(%llu NaN, %llu infinite, %llu zero, %llu subnormal, %llu halfway)\n",
              static_cast<unsigned long long>(sum.checked),
              static_cast<unsigned long long>(sum.nans),
              static_cast<unsigned long long>(sum.infinities),
              static_cast<unsigned long long>(sum.zeros),
              static_cast<unsigned long long>(sum.subnormals),
              static_cast<unsigned long long>(sum.halfway));
  return 0;
}
