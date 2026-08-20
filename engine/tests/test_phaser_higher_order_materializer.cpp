#include "metal/PhaserHigherOrderMaterializer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

constexpr uint32_t kPartials = 6;
constexpr uint32_t kSections = 6;
constexpr uint32_t kInterval = 32;
constexpr uint32_t kHeader = 8;
constexpr uint32_t kSourceBase = kHeader;
constexpr uint32_t kTailBase = kSourceBase + 16 * kPartials;
constexpr uint32_t kPrimitiveBase = kTailBase + kInterval;
constexpr uint32_t kSourcePrimitiveBase = kPrimitiveBase;
constexpr uint32_t kSupportBase = kSourcePrimitiveBase + 4 * kPartials;
constexpr uint32_t kTimeBase = kSupportBase + 8 * (kSections + 1);
constexpr uint32_t kImageSize = kTimeBase + kInterval;
constexpr double kSampleRate = 44100.0;
constexpr double kTwoPi = 6.2831853071795864769;

int failures = 0;

#define CHECK(condition) do {                                                 \
  if (!(condition)) {                                                        \
    std::printf("FAIL line %d: %s\n", __LINE__, #condition);                \
    ++failures;                                                              \
  }                                                                          \
} while (0)

double read(const std::vector<int64_t> & image, std::size_t index)
{
  return std::bit_cast<double>(image[index]);
}

void write(std::vector<int64_t> & image, std::size_t index, double value)
{
  image[index] = std::bit_cast<int64_t>(value);
}

double alpha_at(double sample)
{
  constexpr double modulus = 4294967296.0;
  const uint64_t increment = static_cast<uint64_t>(8.0 * modulus / kSampleRate);
  const double phase = std::fmod(increment * sample, modulus) / modulus;
  return kTwoPi * 4000.0
    * std::pow(2.0, 3.0 * std::sin(kTwoPi * phase));
}

double ratio(uint32_t section)
{
  return std::pow(
    2.0, -1.25 + 2.5 * section / static_cast<double>(kSections - 1));
}

std::complex<double> source_pole(uint32_t partial)
{
  const double harmonic = partial + 1.0;
  return {-4.0 * (1.0 + 0.4 * harmonic),
          kTwoPi * 220.0 * harmonic};
}

std::complex<double> source_amp(uint32_t partial)
{
  return {std::pow(partial + 1.0, -1.1), 0.0};
}

std::complex<double> exact_source(uint32_t partial, double sample)
{
  const auto pole = source_pole(partial);
  std::complex<double> transfer{1.0, 0.0};
  const double alpha = alpha_at(sample);
  for (uint32_t section = 0; section < kSections; ++section)
  {
    const double rate = alpha * ratio(section);
    transfer *= (pole - rate) / (pole + rate);
  }
  return source_amp(partial) * (0.5 + 0.5 * transfer);
}

double exact_tail(double sample)
{
  if (sample <= 0.0) return 0.0;
  const double alpha = alpha_at(sample);
  const double time = sample / kSampleRate;
  std::complex<double> total{};
  for (uint32_t section = 0; section < kSections; ++section)
  {
    const double rate = alpha * ratio(section);
    const std::complex<double> pole{-rate, 0.0};
    std::complex<double> allpass_residue{-2.0 * rate, 0.0};
    for (uint32_t other = 0; other < kSections; ++other)
      if (other != section)
      {
        const double other_rate = alpha * ratio(other);
        allpass_residue *= (pole - other_rate) / (pole + other_rate);
      }
    std::complex<double> source_at_pole{};
    for (uint32_t partial = 0; partial < kPartials; ++partial)
      source_at_pole += source_amp(partial) / (pole - source_pole(partial));
    total += 0.5 * allpass_residue * source_at_pole
      * std::exp(-rate * time);
  }
  return total.real();
}

std::complex<double> evaluate_source(
  const std::vector<int64_t> & image, uint32_t partial, double coordinate)
{
  std::complex<double> following{}, after_following{};
  for (uint32_t degree = 8; degree-- > 1;)
  {
    const std::size_t index = kSourceBase + 16 * partial + 2 * degree;
    const std::complex<double> coefficient{
      read(image, index), read(image, index + 1)};
    const auto current = 2.0 * coordinate * following - after_following
      + coefficient;
    after_following = following;
    following = current;
  }
  const std::complex<double> first{
    read(image, kSourceBase + 16 * partial),
    read(image, kSourceBase + 16 * partial + 1)};
  return coordinate * following - after_following + first;
}

std::complex<double> evaluate_source_shape(
  const std::vector<int64_t> & image, uint32_t partial, double coordinate,
  uint32_t source_support)
{
  const uint32_t stride = 2 * source_support;
  std::complex<double> following{}, after_following{};
  for (uint32_t degree = source_support; degree-- > 1;)
  {
    const std::size_t index = kHeader + stride * partial + 2 * degree;
    const std::complex<double> coefficient{
      read(image, index), read(image, index + 1)};
    const auto current = 2.0 * coordinate * following - after_following
      + coefficient;
    after_following = following;
    following = current;
  }
  const std::complex<double> first{
    read(image, kHeader + stride * partial),
    read(image, kHeader + stride * partial + 1)};
  return coordinate * following - after_following + first;
}

std::vector<int64_t> primitive_image_shape(
  uint32_t interval, uint32_t source_support, uint32_t weight_support,
  double start = 0.0)
{
  const uint32_t tail_base = kHeader + 2 * source_support * kPartials;
  const uint32_t primitive_base = tail_base + interval;
  const uint32_t source_primitive_base = primitive_base;
  const uint32_t support_base = source_primitive_base + 4 * kPartials;
  const uint32_t time_base = support_base
    + source_support * (kSections + 1);
  std::vector<int64_t> image(time_base + interval);
  for (uint32_t index = 0; index < image.size(); ++index)
    write(image, index, 0.0);
  for (const auto [index, value] : std::array<std::pair<uint32_t, double>, 8>{
      {{0, 8430261.0}, {1, 1.0}, {2, kPartials}, {3, kSections},
       {4, interval}, {5, source_support}, {6, weight_support},
       {7, primitive_base}}})
    write(image, index, value);
  for (uint32_t partial = 0; partial < kPartials; ++partial)
  {
    const auto pole = source_pole(partial);
    const auto amp = source_amp(partial);
    const uint32_t base = source_primitive_base + 4 * partial;
    write(image, base, -pole.real());
    write(image, base + 1, pole.imag());
    write(image, base + 2, amp.real());
    write(image, base + 3, amp.imag());
  }
  std::vector<int> offsets(source_support);
  for (uint32_t index = 0; index < source_support; ++index)
  {
    const double target = 0.5 * interval * (1.0 - std::cos(
      M_PI * index / static_cast<double>(source_support - 1)));
    const int lower = index ? offsets[index - 1] + 1 : 0;
    const int upper = interval - (source_support - 1 - index);
    offsets[index] = std::max(
      lower, std::min(upper, static_cast<int>(std::llround(target))));
  }
  for (uint32_t support = 0; support < source_support; ++support)
  {
    const double alpha = alpha_at(start + offsets[support]);
    const uint32_t base = support_base + support * (kSections + 1);
    write(image, base, 0.5);
    for (uint32_t section = 0; section < kSections; ++section)
      write(image, base + 1 + section, alpha * ratio(section));
  }
  for (uint32_t offset = 0; offset < interval; ++offset)
    write(image, time_base + offset, (start + offset) / kSampleRate);
  return image;
}

std::vector<int64_t> primitive_image(double start = 0.0)
{
  std::vector<int64_t> image(kImageSize);
  for (uint32_t index = 0; index < kImageSize; ++index) write(image, index, 0.0);
  for (const auto [index, value] : std::array<std::pair<uint32_t, double>, 8>{
      {{0, 8430261.0}, {1, 1.0}, {2, kPartials}, {3, kSections},
       {4, kInterval}, {5, 8.0}, {6, 6.0}, {7, kPrimitiveBase}}})
    write(image, index, value);
  for (uint32_t partial = 0; partial < kPartials; ++partial)
  {
    const auto pole = source_pole(partial);
    const auto amp = source_amp(partial);
    const uint32_t base = kSourcePrimitiveBase + 4 * partial;
    write(image, base, -pole.real());
    write(image, base + 1, pole.imag());
    write(image, base + 2, amp.real());
    write(image, base + 3, amp.imag());
  }
  constexpr std::array<int, 8> offsets{0, 2, 6, 12, 20, 26, 30, 32};
  for (uint32_t support = 0; support < offsets.size(); ++support)
  {
    const double alpha = alpha_at(start + offsets[support]);
    const uint32_t base = kSupportBase + support * (kSections + 1);
    write(image, base, 0.5);
    for (uint32_t section = 0; section < kSections; ++section)
      write(image, base + 1 + section, alpha * ratio(section));
  }
  for (uint32_t offset = 0; offset < kInterval; ++offset)
    write(image, kTimeBase + offset, (start + offset) / kSampleRate);
  return image;
}

void run()
{
  auto image = primitive_image();
  auto repeat = image;
  std::string error;
  CHECK(tropical_metal::materialize_higher_order_phaser_image(
    image, kInterval, &error));
  CHECK(tropical_metal::materialize_higher_order_phaser_image(
    repeat, kInterval, &error));
  CHECK(image == repeat);
  constexpr std::array<double, 5> starts{0.0, 256.0, 1376.0, 4096.0, 8192.0};
  std::array<std::vector<int64_t>, starts.size()> forward;
  for (std::size_t index = 0; index < starts.size(); ++index)
  {
    forward[index] = primitive_image(starts[index]);
    CHECK(tropical_metal::materialize_higher_order_phaser_image(
      forward[index], kInterval, &error));
  }
  for (std::size_t reverse = starts.size(); reverse-- > 0;)
  {
    auto shuffled = primitive_image(starts[reverse]);
    CHECK(tropical_metal::materialize_higher_order_phaser_image(
      shuffled, kInterval, &error));
    CHECK(shuffled == forward[reverse]);
  }
  double worst_source = 0.0;
  double worst_tail = 0.0;
  for (uint32_t offset = 0; offset < kInterval; ++offset)
  {
    const double coordinate = 2.0 * offset / kInterval - 1.0;
    for (uint32_t partial = 0; partial < kPartials; ++partial)
      worst_source = std::max(worst_source,
        std::abs(evaluate_source(image, partial, coordinate)
                 - exact_source(partial, offset)));
    worst_tail = std::max(worst_tail,
      std::abs(read(image, kTailBase + offset) - exact_tail(offset)));
  }
  CHECK(worst_source < 1.0e-6);
  CHECK(worst_tail < 1.0e-6);
  for (uint32_t index = kPrimitiveBase; index < image.size(); ++index)
    CHECK(read(image, index) == 0.0);
  auto malformed = primitive_image();
  write(malformed, 0, 0.0);
  CHECK(!tropical_metal::materialize_higher_order_phaser_image(
    malformed, kInterval, &error));
  auto nonuniform = primitive_image();
  write(nonuniform, kSupportBase + (kSections + 1) + 2,
        1.01 * read(nonuniform, kSupportBase + (kSections + 1) + 2));
  CHECK(!tropical_metal::materialize_higher_order_phaser_image(
    nonuniform, kInterval, &error));
  constexpr uint32_t wide_interval = 128;
  constexpr uint32_t wide_source_support = 10;
  auto wide = primitive_image_shape(
    wide_interval, wide_source_support, 8);
  CHECK(tropical_metal::materialize_higher_order_phaser_image(
    wide, wide_interval, &error));
  const uint32_t wide_tail_base = kHeader
    + 2 * wide_source_support * kPartials;
  double wide_source = 0.0;
  double wide_tail = 0.0;
  for (uint32_t offset = 0; offset < wide_interval; ++offset)
  {
    const double coordinate = 2.0 * offset / wide_interval - 1.0;
    for (uint32_t partial = 0; partial < kPartials; ++partial)
      wide_source = std::max(wide_source,
        std::abs(evaluate_source_shape(
          wide, partial, coordinate, wide_source_support)
          - exact_source(partial, offset)));
    wide_tail = std::max(wide_tail,
      std::abs(read(wide, wide_tail_base + offset) - exact_tail(offset)));
  }
  CHECK(wide_source < 1.0e-6);
  CHECK(wide_tail < 1.0e-6);
  std::printf("source max abs %.3e, tail max abs %.3e\n",
              worst_source, worst_tail);
  std::printf("wide source max abs %.3e, wide tail max abs %.3e\n",
              wide_source, wide_tail);
}

} // namespace

int main()
{
  run();
  return failures == 0 ? 0 : 1;
}
