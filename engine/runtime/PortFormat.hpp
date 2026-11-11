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
//
// A binary32 artifact (`--float f32`, `Kernel.FloatFormat`) declares its
// float ports `f32`: the cell then holds the binary32's bits in its low
// word, written as `(float)value` (IEEE round to nearest even), and an `f32`
// output renders into its own buffer and widens exactly to f64.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace tropical_port
{

enum class Elem : uint8_t { F64, I32, I64, F32 };

// The binary32 overflow threshold, 2^128 - 2^103: halfway between the
// largest finite binary32, (2 - 2^-23) * 2^127, and 2^128. A double whose
// magnitude reaches it rounds (to nearest even) to an infinity.
inline constexpr double kBinary32OverflowThreshold = 0x1.ffffffp+127;

struct Format
{
  Elem elem = Elem::F64;
  uint32_t frac_bits = 0;

  // An integer word with a fixed-point format.
  bool is_word() const { return elem == Elem::I32 || elem == Elem::I64; }
  // Not an f64: the cell holds the port's own bits (a word, or a binary32).
  bool is_encoded() const { return elem != Elem::F64; }
};

inline Elem elem_of_wire(const std::string & wire)
{
  if (wire == "f64") return Elem::F64;
  if (wire == "i32") return Elem::I32;
  if (wire == "i64") return Elem::I64;
  if (wire == "f32") return Elem::F32;
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
  if (f.elem == Elem::F32) return "binary32";
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
  // binary32 rounds every value to nearest; one at or past the overflow
  // threshold would round to an infinity, which is refused, not written.
  // Compared, not cast: the cast of an out-of-range double is not
  // something to lean on.
  if (f.elem == Elem::F32)
  {
    if (!std::isfinite(x) || std::fabs(x) < kBinary32OverflowThreshold) return {};
    return "value " + shortest(x) + " is outside the range of binary32, |x| <= 3.4028234663852886e38";
  }
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

// The binary32 nearest a double: round to nearest even, with a value at or
// past the overflow threshold made the infinity it rounds to explicitly, so
// no out-of-range double is ever cast.
inline float binary32_nearest(double x)
{
  if (std::isfinite(x) && std::fabs(x) >= kBinary32OverflowThreshold)
    return x > 0 ? std::numeric_limits<float>::infinity()
                 : -std::numeric_limits<float>::infinity();
  return static_cast<float>(x);
}

// An 8-byte slot cell holding binary32 `v`'s bits in its low word (a
// little-endian `float` load of the cell reads it), and back.
inline double cell_of_binary32(float v)
{
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  return cell_of_word(static_cast<int64_t>(static_cast<uint64_t>(bits)));
}

inline float binary32_of_cell(double cell)
{
  const uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(word_of_cell(cell)));
  float v;
  std::memcpy(&v, &bits, sizeof v);
  return v;
}

// The cell a control write of `x` stores at a port of format `f`: the value
// itself (f64), the binary32 nearest it (f32), or the word (`encode_word`).
inline double cell_of_value(const Format & f, double x)
{
  if (f.elem == Elem::F32) return cell_of_binary32(binary32_nearest(x));
  if (f.is_word()) return cell_of_word(encode_word(f, x));
  return x;
}

// The real value a cell of format `f` holds.
inline double value_of_cell(const Format & f, double cell)
{
  if (f.elem == Elem::F32) return static_cast<double>(binary32_of_cell(cell));
  if (f.is_word()) return word_value(f, word_of_cell(cell));
  return cell;
}

} // namespace tropical_port
