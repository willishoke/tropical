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
#include <cstdio>
#include <cstdlib>
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

// The word's range: [-2^(w-1), 2^(w-1)).
inline int64_t word_min(const Format & f)
{
  return f.elem == Elem::I32 ? INT32_MIN : INT64_MIN;
}

// "int32 Q16.16", as Lean's `PortFormat.describe` spells it.
inline std::string describe(const Format & f)
{
  if (!f.is_word()) return "float";
  const uint32_t w = word_bits(f.elem);
  return std::string(f.elem == Elem::I32 ? "int32" : "int") + " Q"
    + std::to_string(w - f.frac_bits) + "." + std::to_string(f.frac_bits);
}

// The real values a format holds, [-2^k, 2^k) with k = width - 1 - frac_bits
// (k >= -1, since frac_bits <= width), as Lean's `PortFormat.rangeLabel`.
inline std::string range_label(const Format & f)
{
  const int k = static_cast<int>(word_bits(f.elem)) - 1 - static_cast<int>(f.frac_bits);
  const std::string end = k >= 0 ? std::to_string(uint64_t{1} << k) : std::string("0.5");
  return "[-" + end + ", " + end + ")";
}

// The shortest plain decimal (no exponent) that reads back as `x`, as a
// JSON number prints it (`100000`, `0.5`); `%.17g` past 40 decimals.
inline std::string shortest(double x)
{
  char buf[400];
  if (std::isfinite(x) && std::fabs(x) < 1e21)
    for (int p = 0; p <= 40; ++p)
    {
      std::snprintf(buf, sizeof buf, "%.*f", p, x);
      if (std::strtod(buf, nullptr) == x) return buf;
    }
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

// Why control value `x` does not fit an integer format, or "" when it does:
// its word x * 2^frac must lie in the word's range (it is then truncated
// toward zero, `encode_word`). Past the range the word would saturate, so a
// write is refused instead, as Lean's `PortFormat.rangeError?` refuses it at
// ingest and at `set_param`.
inline std::string range_error(const Format & f, double x)
{
  if (!f.is_word()) return {};
  const double scaled = std::ldexp(x, static_cast<int>(f.frac_bits));
  const double hi = std::ldexp(1.0, static_cast<int>(word_bits(f.elem)) - 1);
  if (scaled >= -hi && scaled < hi) return {};
  return "value " + shortest(x) + " is outside the range of " + describe(f) + ", "
    + range_label(f);
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
