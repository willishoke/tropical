#include "metal/PhaserHigherOrderMaterializer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

namespace tropical_metal
{
namespace
{

constexpr uint32_t kHeaderCount = 8;
constexpr uint32_t kBaseSourceSupport = 8;
constexpr uint32_t kWideSourceSupport = 10;
constexpr uint32_t kBaseWeightSupport = 6;
constexpr uint32_t kWideWeightSupport = 8;
constexpr uint32_t kMaxSections = 18;
constexpr double kImageMagic = 8430261.0;
constexpr double kImageVersion = 1.0;
constexpr double kTailCutoff = 1.0e-12;
constexpr double kPi = 3.14159265358979323846;

struct DD
{
  double hi = 0.0;
  double lo = 0.0;
};

struct CDD
{
  DD real;
  DD imag;
};

DD quick_two_sum(double left, double right)
{
  const double total = left + right;
  return {total, right - (total - left)};
}

DD two_sum(double left, double right)
{
  const double total = left + right;
  const double virtual_right = total - left;
  return {total,
          (left - (total - virtual_right)) + (right - virtual_right)};
}

DD two_product(double left, double right)
{
  const double product = left * right;
  return {product, std::fma(left, right, -product)};
}

DD add(DD left, DD right)
{
  const DD pair = two_sum(left.hi, right.hi);
  return quick_two_sum(pair.hi, pair.lo + left.lo + right.lo);
}

DD negate(DD value) { return {-value.hi, -value.lo}; }
DD subtract(DD left, DD right) { return add(left, negate(right)); }

DD multiply(DD left, DD right)
{
  const DD product = two_product(left.hi, right.hi);
  return quick_two_sum(
    product.hi,
    product.lo + left.hi * right.lo + left.lo * right.hi);
}

DD divide(DD left, DD right)
{
  const double first = left.hi / right.hi;
  const DD remainder = subtract(left, multiply(right, {first, 0.0}));
  return add({first, 0.0},
             {(remainder.hi + remainder.lo) / right.hi, 0.0});
}

double value(DD number) { return number.hi + number.lo; }

CDD cdd(std::complex<double> number)
{
  return {{number.real(), 0.0}, {number.imag(), 0.0}};
}

CDD cadd(CDD left, CDD right)
{
  return {add(left.real, right.real), add(left.imag, right.imag)};
}

CDD csubtract(CDD left, CDD right)
{
  return {subtract(left.real, right.real),
          subtract(left.imag, right.imag)};
}

CDD cdivide_corrected(std::complex<double> left,
                      std::complex<double> right)
{
  const double inverse_denominator = 1.0 /
    (right.real() * right.real() + right.imag() * right.imag());
  const auto divide_fast = [&](std::complex<double> numerator) {
    return std::complex<double>{
      (numerator.real() * right.real()
       + numerator.imag() * right.imag()) * inverse_denominator,
      (numerator.imag() * right.real()
       - numerator.real() * right.imag()) * inverse_denominator};
  };
  const std::complex<double> first = divide_fast(left);
  const CDD product{
    subtract(multiply({first.real(), 0.0}, {right.real(), 0.0}),
             multiply({first.imag(), 0.0}, {right.imag(), 0.0})),
    add(multiply({first.real(), 0.0}, {right.imag(), 0.0}),
        multiply({first.imag(), 0.0}, {right.real(), 0.0}))};
  const std::complex<double> residual{
    value(subtract({left.real(), 0.0}, product.real)),
    value(subtract({left.imag(), 0.0}, product.imag))};
  return cadd(cdd(first), cdd(divide_fast(residual)));
}

CDD cdivide_real(CDD left, DD right)
{
  return {divide(left.real, right), divide(left.imag, right)};
}

CDD cscale_real(CDD number, DD scale)
{
  return {multiply(number.real, scale), multiply(number.imag, scale)};
}

std::complex<double> value(CDD number)
{
  return {value(number.real), value(number.imag)};
}

bool fail(std::string * error, const char * message)
{
  if (error) *error = message;
  return false;
}

double load(const std::vector<int64_t> & image, std::size_t index)
{
  return std::bit_cast<double>(image[index]);
}

void store(std::vector<int64_t> & image, std::size_t index, double number)
{
  image[index] = std::bit_cast<int64_t>(number);
}

bool finite_complex(std::complex<double> value)
{
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

template <std::size_t Count>
std::array<int, Count> support_offsets(uint32_t interval)
{
  std::array<int, Count> result{};
  for (uint32_t index = 0; index < Count; ++index)
  {
    const double target = 0.5 * interval * (
      1.0 - std::cos(kPi * index
                     / static_cast<double>(Count - 1)));
    const int lower = index ? result[index - 1] + 1 : 0;
    const int upper = static_cast<int>(interval - (Count - 1 - index));
    result[index] = std::max(
      lower, std::min(upper, static_cast<int>(std::llround(target))));
  }
  return result;
}

template <std::size_t Count>
std::array<std::array<double, Count>, Count> inverse_transform(
  const std::array<int, Count> & offsets, uint32_t interval)
{
  std::array<std::array<double, Count * 2>, Count> rows{};
  for (std::size_t row = 0; row < Count; ++row)
  {
    const double coordinate =
      2.0 * static_cast<double>(offsets[row]) / interval - 1.0;
    rows[row][0] = 1.0;
    if constexpr (Count > 1) rows[row][1] = coordinate;
    for (std::size_t degree = 2; degree < Count; ++degree)
      rows[row][degree] =
        2.0 * coordinate * rows[row][degree - 1]
        - rows[row][degree - 2];
    rows[row][Count + row] = 1.0;
  }
  for (std::size_t column = 0; column < Count; ++column)
  {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < Count; ++row)
      if (std::abs(rows[row][column]) > std::abs(rows[pivot][column]))
        pivot = row;
    std::swap(rows[column], rows[pivot]);
    const double divisor = rows[column][column];
    for (std::size_t index = 0; index < Count * 2; ++index)
      rows[column][index] /= divisor;
    for (std::size_t row = 0; row < Count; ++row)
    {
      if (row == column) continue;
      const double scale = rows[row][column];
      for (std::size_t index = 0; index < Count * 2; ++index)
        rows[row][index] -= scale * rows[column][index];
    }
  }
  std::array<std::array<double, Count>, Count> inverse{};
  for (std::size_t row = 0; row < Count; ++row)
    for (std::size_t column = 0; column < Count; ++column)
      inverse[row][column] = rows[row][Count + column];
  return inverse;
}

template <typename Value, std::size_t Count>
std::array<Value, Count> transform(
  const std::array<Value, Count> & values,
  const std::array<std::array<double, Count>, Count> & inverse)
{
  std::array<Value, Count> result{};
  for (std::size_t degree = 0; degree < Count; ++degree)
    for (std::size_t node = 0; node < Count; ++node)
      result[degree] += inverse[degree][node] * values[node];
  return result;
}

template <typename Value, std::size_t Count>
Value evaluate(const std::array<Value, Count> & coefficients,
               double coordinate)
{
  Value following{};
  Value after_following{};
  for (std::size_t degree = Count - 1; degree >= 1; --degree)
  {
    const Value current = 2.0 * coordinate * following - after_following
      + coefficients[degree];
    after_following = following;
    following = current;
    if (degree == 1) break;
  }
  return coordinate * following - after_following + coefficients[0];
}

bool normalize_rates(const std::vector<double> & rates,
                     double & scale, std::vector<double> & normalized)
{
  double logarithm = 0.0;
  for (double rate : rates)
  {
    if (!std::isfinite(rate) || rate <= 0.0) return false;
    logarithm += std::log(rate);
  }
  scale = std::exp(logarithm / static_cast<double>(rates.size()));
  if (!std::isfinite(scale) || scale <= 0.0) return false;
  normalized.resize(rates.size());
  for (std::size_t index = 0; index < rates.size(); ++index)
  {
    normalized[index] = rates[index] / scale;
    if (!std::isfinite(normalized[index]) || normalized[index] <= 0.0)
      return false;
  }
  return true;
}

struct WeightGeometry
{
  std::size_t sections = 0;
  std::array<DD, kMaxSections> nodes{};
  std::array<DD, kMaxSections> zeros{};
  std::array<DD, kMaxSections> numerators{};
  std::array<DD, kMaxSections> transitions{};
  std::array<std::array<DD, kMaxSections>, kMaxSections> differences{};
  std::array<double, kMaxSections> node_values{};
  std::vector<double> normalized_rates;
};

bool build_weight_geometry(const std::vector<double> & rates,
                           double & scale,
                           WeightGeometry & geometry)
{
  if (!normalize_rates(rates, scale, geometry.normalized_rates)
      || rates.empty() || rates.size() > kMaxSections)
    return false;
  geometry.sections = rates.size();
  for (std::size_t index = 0; index < rates.size(); ++index)
  {
    geometry.nodes[index] = {-geometry.normalized_rates[index], 0.0};
    geometry.zeros[index] = negate(geometry.nodes[index]);
    geometry.node_values[index] = value(geometry.nodes[index]);
  }
  for (std::size_t index = 0; index < rates.size(); ++index)
  {
    DD numerator{1.0, 0.0};
    for (std::size_t zero = 0; zero < rates.size(); ++zero)
      numerator = multiply(
        numerator, subtract(geometry.nodes[index], geometry.zeros[zero]));
    geometry.numerators[index] = numerator;
    DD transitions{1.0, 0.0};
    for (std::size_t node = index; node + 1 < rates.size(); ++node)
      transitions = multiply(transitions, negate(geometry.nodes[node]));
    geometry.transitions[index] = transitions;
    for (std::size_t order = 1; index + order < rates.size(); ++order)
      geometry.differences[order][index] = subtract(
        geometry.nodes[index + order], geometry.nodes[index]);
  }
  return true;
}

bool rate_scale(const std::vector<double> & rates,
                const WeightGeometry & geometry,
                double & scale)
{
  if (rates.size() != geometry.sections || rates.empty()
      || geometry.normalized_rates[0] <= 0.0)
    return false;
  scale = rates[0] / geometry.normalized_rates[0];
  if (!std::isfinite(scale) || scale <= 0.0) return false;
  for (std::size_t index = 0; index < rates.size(); ++index)
  {
    const double expected = scale * geometry.normalized_rates[index];
    const double tolerance = 1.0e-8 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(rates[index]) || rates[index] <= 0.0
        || std::abs(rates[index] - expected) > tolerance)
      return false;
  }
  return true;
}

bool build_weights(
  const std::vector<std::complex<double>> & physical_poles,
  const std::vector<std::complex<double>> & amplitudes, double scale,
  const WeightGeometry & geometry, double mix,
  std::array<std::complex<double>, kMaxSections> & weights_out)
{
  if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(mix))
    return false;
  const std::size_t sections = geometry.sections;
  std::array<CDD, kMaxSections> divided{}, next{}, coefficients{};
  for (std::size_t node_index = 0; node_index < sections; ++node_index)
  {
    CDD source{};
    for (std::size_t partial = 0; partial < physical_poles.size(); ++partial)
    {
      const std::complex<double> pole = physical_poles[partial] / scale;
      const std::complex<double> denominator =
        std::complex<double>{geometry.node_values[node_index], 0.0} - pole;
      source = cadd(source, cdivide_corrected(
        amplitudes[partial], denominator));
    }
    divided[node_index] = cscale_real(
      source, geometry.numerators[node_index]);
  }
  auto * current = &divided;
  auto * scratch = &next;
  coefficients[0] = (*current)[0];
  for (std::size_t order = 1; order < sections; ++order)
  {
    for (std::size_t index = 0; index < sections - order; ++index)
      (*scratch)[index] = cdivide_real(
        csubtract((*current)[index + 1], (*current)[index]),
        geometry.differences[order][index]);
    std::swap(current, scratch);
    coefficients[order] = (*current)[0];
  }
  for (std::size_t index = 0; index < sections; ++index)
  {
    weights_out[index] = value(cscale_real(
      cdivide_real(coefficients[index], geometry.transitions[index]),
      {mix, 0.0}));
    if (!finite_complex(weights_out[index])) return false;
  }
  return true;
}

std::complex<double> source_amplitude(
  std::complex<double> physical_pole,
  std::complex<double> amplitude,
  double scale, const std::vector<double> & normalized_rates,
  double mix)
{
  if (!std::isfinite(scale) || scale <= 0.0)
    return {std::numeric_limits<double>::quiet_NaN(), 0.0};
  const std::complex<double> pole = physical_pole / scale;
  std::complex<double> transfer{1.0, 0.0};
  for (double rate : normalized_rates)
    transfer *= (pole - rate) / (pole + rate);
  return amplitude * ((1.0 - mix) + mix * transfer);
}

std::complex<double> uniform_tail(
  const WeightGeometry & geometry,
  const std::array<std::complex<double>, kMaxSections> & weights,
  double coordinate)
{
  const std::size_t sections = geometry.sections;
  double rho = 0.0;
  for (std::size_t section = 0; section < sections; ++section)
    rho = std::max(rho, -geometry.node_values[section]);
  const double mean = rho * coordinate;
  if (!std::isfinite(mean) || mean < 0.0 || mean > 700.0)
    return {};
  double poisson = std::exp(-mean);
  std::array<double, kMaxSections> column{}, next{}, result{};
  column[sections - 1] = 1.0;
  auto * current = &column;
  auto * scratch = &next;
  const int limit = std::max(32, static_cast<int>(
    mean + 14.0 * std::sqrt(mean + 1.0) + 32.0));
  for (int step = 0; step < limit; ++step)
  {
    for (std::size_t section = 0; section < sections; ++section)
      result[section] += poisson * (*current)[section];
    scratch->fill(0.0);
    (*scratch)[sections - 1] =
      (1.0 + geometry.node_values[sections - 1] / rho)
      * (*current)[sections - 1];
    for (std::size_t section = sections - 1; section-- > 0;)
      (*scratch)[section] =
        (1.0 + geometry.node_values[section] / rho) * (*current)[section]
        + (-geometry.node_values[section] / rho) * (*current)[section + 1];
    std::swap(current, scratch);
    poisson *= mean / static_cast<double>(step + 1);
  }
  std::complex<double> tail{};
  for (std::size_t section = 0; section < sections; ++section)
    tail += weights[section] * result[section];
  return tail;
}

template <uint32_t SourceSupport, uint32_t WeightSupport>
bool materialize_shape(
  std::vector<int64_t> & image,
  uint32_t interval_frames,
  std::string * error)
{
  if (image.size() < kHeaderCount)
    return fail(error, "higher-order phaser image is shorter than its header");
  const auto header = [&](std::size_t index) { return load(image, index); };
  if (header(0) != kImageMagic || header(1) != kImageVersion)
    return fail(error, "higher-order phaser image magic/version mismatch");
  const uint32_t partials = static_cast<uint32_t>(header(2));
  const uint32_t sections = static_cast<uint32_t>(header(3));
  const uint32_t interval = static_cast<uint32_t>(header(4));
  const uint32_t source_support = static_cast<uint32_t>(header(5));
  const uint32_t weight_support = static_cast<uint32_t>(header(6));
  const std::size_t primitive_base = static_cast<std::size_t>(header(7));
  const bool base_support = source_support == kBaseSourceSupport
    && weight_support == kBaseWeightSupport;
  const bool refined_weight_support = interval == 64
    && source_support == kBaseSourceSupport
    && weight_support == kWideWeightSupport;
  const bool wide_support = interval == 128
    && source_support == kWideSourceSupport
    && weight_support == kWideWeightSupport;
  if ((partials != 6 && partials != 32)
      || (sections != 6 && sections != 12 && sections != 18)
      || interval != interval_frames
      || (interval != 32 && interval != 64 && interval != 128)
      || source_support != SourceSupport
      || weight_support != WeightSupport
      || (!base_support && !refined_weight_support && !wide_support))
    return fail(error, "higher-order phaser image shape is not admitted");

  const std::size_t source_coefficient_base = kHeaderCount;
  const std::size_t tail_base = source_coefficient_base
    + 2 * SourceSupport * partials;
  const std::size_t source_primitive_base = primitive_base;
  const std::size_t support_base = source_primitive_base + 4 * partials;
  const std::size_t time_base = support_base
    + SourceSupport * (sections + 1);
  const std::size_t expected_size = time_base + interval;
  if (primitive_base != tail_base + interval || image.size() != expected_size)
    return fail(error, "higher-order phaser image layout mismatch");

  std::vector<std::complex<double>> poles(partials), amplitudes(partials);
  for (uint32_t partial = 0; partial < partials; ++partial)
  {
    const std::size_t base = source_primitive_base + 4 * partial;
    const double sigma = load(image, base);
    const double omega = load(image, base + 1);
    const double real = load(image, base + 2);
    const double imag = load(image, base + 3);
    if (!std::isfinite(sigma) || sigma <= 0.0 || !std::isfinite(omega)
        || !std::isfinite(real) || !std::isfinite(imag))
      return fail(error, "higher-order phaser source primitive is invalid");
    poles[partial] = {-sigma, omega};
    amplitudes[partial] = {real, imag};
  }

  const auto source_nodes = support_offsets<SourceSupport>(interval);
  const auto weight_nodes = support_offsets<WeightSupport>(interval);
  const auto source_inverse = inverse_transform(source_nodes, interval);
  const auto weight_inverse = inverse_transform(weight_nodes, interval);
  std::array<double, SourceSupport> mix_values{};
  std::vector<std::array<double, SourceSupport>> rate_values(sections);
  for (uint32_t support = 0; support < SourceSupport; ++support)
  {
    const std::size_t base = support_base + support * (sections + 1);
    mix_values[support] = load(image, base);
    if (!std::isfinite(mix_values[support]))
      return fail(error, "higher-order phaser mix support is invalid");
    for (uint32_t section = 0; section < sections; ++section)
    {
      rate_values[section][support] = load(image, base + 1 + section);
      if (!std::isfinite(rate_values[section][support])
          || rate_values[section][support] <= 0.0)
        return fail(error, "higher-order phaser rate support is invalid");
    }
  }
  const auto mix_coefficients = transform(mix_values, source_inverse);
  std::vector<double> initial_rates(sections);
  for (uint32_t section = 0; section < sections; ++section)
    initial_rates[section] = rate_values[section][0];
  WeightGeometry geometry;
  double initial_scale = 0.0;
  if (!build_weight_geometry(initial_rates, initial_scale, geometry))
    return fail(error, "higher-order phaser rate geometry is invalid");
  std::array<double, SourceSupport> scale_values{};
  for (uint32_t support = 0; support < SourceSupport; ++support)
  {
    std::vector<double> rates(sections);
    for (uint32_t section = 0; section < sections; ++section)
      rates[section] = rate_values[section][support];
    if (!rate_scale(rates, geometry, scale_values[support]))
      return fail(error, "higher-order phaser rates do not share one sweep");
  }
  const auto scale_coefficients = transform(scale_values, source_inverse);

  std::vector<std::array<std::complex<double>, SourceSupport>>
    source_values(partials);
  for (uint32_t support = 0; support < SourceSupport; ++support)
  {
    for (uint32_t partial = 0; partial < partials; ++partial)
    {
      source_values[partial][support] = source_amplitude(
        poles[partial], amplitudes[partial], scale_values[support],
        geometry.normalized_rates, mix_values[support]);
      if (!finite_complex(source_values[partial][support]))
        return fail(error, "higher-order phaser source gain is invalid");
    }
  }
  for (uint32_t partial = 0; partial < partials; ++partial)
  {
    const auto coefficients = transform(source_values[partial], source_inverse);
    for (uint32_t degree = 0; degree < SourceSupport; ++degree)
    {
      store(image, source_coefficient_base
            + 2 * SourceSupport * partial + 2 * degree,
            coefficients[degree].real());
      store(image, source_coefficient_base
            + 2 * SourceSupport * partial + 2 * degree + 1,
            coefficients[degree].imag());
    }
  }

  std::vector<std::array<std::complex<double>, WeightSupport>>
    weight_values(sections);
  for (uint32_t support = 0; support < WeightSupport; ++support)
  {
    const double coordinate =
      2.0 * static_cast<double>(weight_nodes[support]) / interval - 1.0;
    const double scale = evaluate(scale_coefficients, coordinate);
    const double mix = evaluate(mix_coefficients, coordinate);
    std::array<std::complex<double>, kMaxSections> weights{};
    if (!build_weights(
          poles, amplitudes, scale, geometry, mix, weights))
      return fail(error, "higher-order phaser Newton image is invalid");
    for (uint32_t section = 0; section < sections; ++section)
      weight_values[section][support] = weights[section];
  }
  std::vector<std::array<std::complex<double>, WeightSupport>>
    weight_coefficients(sections);
  for (uint32_t section = 0; section < sections; ++section)
    weight_coefficients[section] = transform(
      weight_values[section], weight_inverse);

  for (uint32_t offset = 0; offset < interval; ++offset)
  {
    const double time = load(image, time_base + offset);
    if (!std::isfinite(time))
      return fail(error, "higher-order phaser response time is invalid");
    if (time <= 0.0)
    {
      store(image, tail_base + offset, 0.0);
      continue;
    }
    const double coordinate = 2.0 * static_cast<double>(offset) / interval - 1.0;
    const double scale = evaluate(scale_coefficients, coordinate);
    if (!std::isfinite(scale) || scale <= 0.0)
      return fail(error, "higher-order phaser interpolated rates are invalid");
    std::array<std::complex<double>, kMaxSections> weights{};
    double survival_bound = 0.0;
    const double phase_coordinate = scale * time;
    for (uint32_t section = 0; section < sections; ++section)
    {
      weights[section] = evaluate(
        weight_coefficients[section], coordinate);
      survival_bound += std::abs(weights[section])
        * std::exp(geometry.node_values[section] * phase_coordinate);
    }
    if (!std::isfinite(survival_bound))
      return fail(error, "higher-order phaser survival bound is invalid");
    const std::complex<double> tail = survival_bound < kTailCutoff
      ? std::complex<double>{}
      : uniform_tail(geometry, weights, phase_coordinate);
    if (!finite_complex(tail))
      return fail(error, "higher-order phaser whole tail is invalid");
    store(image, tail_base + offset, tail.real());
  }

  // Primitive and exact-fallback cells never cross semantically.  Zeroing them
  // makes the published image bounded and deterministic without leaking stale
  // high-dynamic-range worker inputs into Metal's immutable column payload.
  for (std::size_t index = primitive_base; index < image.size(); ++index)
    store(image, index, 0.0);
  return true;
}

} // namespace

bool materialize_higher_order_phaser_image(
  std::vector<int64_t> & image,
  uint32_t interval_frames,
  std::string * error)
{
  if (image.size() < kHeaderCount)
    return fail(error, "higher-order phaser image is shorter than its header");
  const double source_support = load(image, 5);
  const double weight_support = load(image, 6);
  if (source_support == kBaseSourceSupport
      && weight_support == kBaseWeightSupport)
    return materialize_shape<kBaseSourceSupport, kBaseWeightSupport>(
      image, interval_frames, error);
  if (source_support == kBaseSourceSupport
      && weight_support == kWideWeightSupport)
    return materialize_shape<kBaseSourceSupport, kWideWeightSupport>(
      image, interval_frames, error);
  if (source_support == kWideSourceSupport
      && weight_support == kWideWeightSupport)
    return materialize_shape<kWideSourceSupport, kWideWeightSupport>(
      image, interval_frames, error);
  return fail(error, "higher-order phaser image shape is not admitted");
}

} // namespace tropical_metal
