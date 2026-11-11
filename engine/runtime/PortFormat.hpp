#pragma once

// A port's element type and fixed-point format, as a tropical_kernel_1
// manifest declares them (`lean/Tropical/PortFormat.lean`,
// design/typed-kernel.md "Typed ports"). An integer port's word `w` means
// `w * 2^-frac_bits`. The host's control value is that real number; the
// word it writes is `encode_word(value)`, the kernel's own float -> integer
// conversion (truncate toward zero, saturate, NaN -> 0; `llvm.fptosi.sat`,
// Lean's `Kernel.encodeWord`), so host and compiler agree on every knob.
//
// The runtime keeps every param in an 8-byte slot cell. A typed param's cell
// holds the word's bits (an int32 sign-extended to 64 bits, so a little-endian
// i32 load of the cell reads it), and the kernel is handed a pointer to the
// cell: no conversion runs on the audio path.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tropical_port
{

enum class Elem : uint8_t { F64, I32, I64 };

struct Format
{
  Elem elem = Elem::F64;
  uint32_t frac_bits = 0;

  bool is_word() const { return elem != Elem::F64; }
};

inline Elem elem_of_wire(const std::string & wire)
{
  if (wire == "f64") return Elem::F64;
  if (wire == "i32") return Elem::I32;
  if (wire == "i64") return Elem::I64;
  throw std::runtime_error("PortFormat: unsupported element '" + wire + "'");
}

inline uint32_t word_bits(Elem e)
{
  return e == Elem::I32 ? 32u : (e == Elem::I64 ? 64u : 0u);
}

// The word for value `x`: x * 2^frac, truncated toward zero and saturated to
// the word, NaN -> 0. The scaling by a power of two is exact in binary64.
inline int64_t encode_word(const Format & f, double x)
{
  if (std::isnan(x)) return 0;
  const double t = std::trunc(std::ldexp(x, static_cast<int>(f.frac_bits)));
  if (f.elem == Elem::I32)
  {
    if (t >= 2147483647.0) return INT32_MAX;
    if (t <= -2147483648.0) return INT32_MIN;
    return static_cast<int64_t>(t);
  }
  if (t >= 9223372036854775808.0) return INT64_MAX;
  if (t <= -9223372036854775808.0) return INT64_MIN;
  return static_cast<int64_t>(t);
}

// The real value word `w` means.
inline double word_value(const Format & f, int64_t w)
{
  return std::ldexp(static_cast<double>(w), -static_cast<int>(f.frac_bits));
}

// An 8-byte slot cell holding word `w`'s bits, and back.
inline double cell_of_word(int64_t w)
{
  double d;
  std::memcpy(&d, &w, sizeof d);
  return d;
}

inline int64_t word_of_cell(double cell)
{
  int64_t w;
  std::memcpy(&w, &cell, sizeof w);
  return w;
}

} // namespace tropical_port
