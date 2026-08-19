#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPartials = 32;
constexpr int kMaxSections = 18;
constexpr int kTailSupport = 24;
constexpr int kSourceSupport = 8;
constexpr int kWeightSupport = 6;
constexpr int kMaxInterval = 128;
constexpr double kSampleRate = 44100.0;
constexpr double kSplitter = 134217729.0;

struct DD { double hi = 0.0, lo = 0.0; };
struct CDD { DD re, im; };

inline DD quick_two_sum(double a, double b) {
  const double s = a + b;
  return {s, b - (s - a)};
}

inline DD two_sum(double a, double b) {
  const double s = a + b;
  const double bv = s - a;
  return {s, (a - (s - bv)) + (b - bv)};
}

inline DD two_product(double a, double b) {
  const double p = a * b;
  const double ca = kSplitter * a;
  const double ah = ca - (ca - a);
  const double al = a - ah;
  const double cb = kSplitter * b;
  const double bh = cb - (cb - b);
  const double bl = b - bh;
  return {p, ((ah * bh - p) + ah * bl + al * bh) + al * bl};
}

inline DD add(DD a, DD b) {
  const DD s = two_sum(a.hi, b.hi);
  return quick_two_sum(s.hi, s.lo + a.lo + b.lo);
}
inline DD neg(DD a) { return {-a.hi, -a.lo}; }
inline DD sub(DD a, DD b) { return add(a, neg(b)); }
inline DD mul(DD a, DD b) {
  const DD p = two_product(a.hi, b.hi);
  return quick_two_sum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}
inline DD div(DD a, DD b) {
  const double q1 = a.hi / b.hi;
  const DD r = sub(a, mul(b, {q1, 0.0}));
  return add({q1, 0.0}, {(r.hi + r.lo) / b.hi, 0.0});
}
inline double value(DD a) { return a.hi + a.lo; }

inline CDD cadd(CDD a, CDD b) { return {add(a.re, b.re), add(a.im, b.im)}; }
inline CDD csub(CDD a, CDD b) { return {sub(a.re, b.re), sub(a.im, b.im)}; }
inline CDD cdiv(CDD a, CDD b) {
  const DD den = add(mul(b.re, b.re), mul(b.im, b.im));
  return {div(add(mul(a.re, b.re), mul(a.im, b.im)), den),
          div(sub(mul(a.im, b.re), mul(a.re, b.im)), den)};
}
inline CDD cdiv_real(CDD a, DD b) { return {div(a.re, b), div(a.im, b)}; }
inline std::complex<double> value(CDD a) { return {value(a.re), value(a.im)}; }

constexpr std::array<std::complex<double>, 8> kTheta {{
  {-10.843917078696988026, 19.277446167181652284},
  {-5.2649713434426468895, 16.220221473167927305},
  {5.9481522689511774808, 3.5874573620183222829},
  {3.5091036084149180974, 8.4361989858843750826},
  {6.4161776990994341923, 1.1941223933701386874},
  {1.4193758971856659786, 10.925363484496722585},
  {4.9931747377179963991, 5.9968817136039422260},
  {-1.4139284624888862114, 13.497725698892745389},
}};
constexpr std::array<std::complex<double>, 8> kAlpha {{
  {-5.0901521865224915650e-7, -2.4220017652852287970e-5},
  {2.1151742182466030907e-4, 4.3892969647380673918e-3},
  {1.1339775178483930527e2, 1.0194721704215856450e2},
  {1.5059585270023467528e1, -5.7514052776421819979},
  {-6.4500878025539646595e1, -2.2459440762652096056e2},
  {-1.4793007113557999718, 1.7686588323782937906},
  {-6.2518392463207918892e1, -1.1190391094283228480e1},
  {4.1023136835410021273e-2, -1.5743466173455468191e-1},
}};
constexpr double kAlpha0 = 2.1248537104952237488e-16;

struct BenchCase {
  const char *name;
  int partials;
  int sections;
  int interval;
  double rate;
  double center;
  double sweep;
  double start;
  std::array<double, kMaxSections> ratios {};
  std::array<double, kMaxPartials> sigma {};
  std::array<double, kMaxPartials> omega {};
  std::array<double, kMaxPartials> amp {};
  std::array<int, kTailSupport> tail_offsets {};
  std::array<int, kSourceSupport> source_offsets {};
  std::array<int, kWeightSupport> weight_offsets {};
  std::array<std::array<double, kTailSupport>, kTailSupport> tail_transform {};
  std::array<std::array<double, kSourceSupport>, kSourceSupport> source_transform {};
  std::array<std::array<double, kWeightSupport>, kWeightSupport> weight_transform {};
};

template<int Count>
std::array<int, Count> integer_nodes(int width) {
  std::array<int, Count> out {};
  for (int i = 0; i < Count; ++i) {
    const double target = 0.5 * width *
      (1.0 - std::cos(M_PI * i / static_cast<double>(Count - 1)));
    const int lower = i ? out[i - 1] + 1 : 0;
    const int upper = width - (Count - 1 - i);
    out[i] = std::max(lower, std::min(upper, static_cast<int>(std::llround(target))));
  }
  return out;
}

BenchCase make_case(const char *name, int p, int s, int interval,
                    double rate, double center, double sweep, double start) {
  BenchCase c {name, p, s, interval, rate, center, sweep, start};
  for (int i = 0; i < s; ++i)
    c.ratios[i] = std::pow(2.0, -1.25 + 2.5 * i / static_cast<double>(s - 1));
  for (int i = 0; i < p; ++i) {
    const double harmonic = i + 1.0;
    c.sigma[i] = 4.0 * (1.0 + 0.4 * harmonic);
    c.omega[i] = 2.0 * M_PI * 220.0 * harmonic;
    c.amp[i] = std::pow(harmonic, -1.1);
  }
  c.tail_offsets = integer_nodes<kTailSupport>(interval);
  c.source_offsets = integer_nodes<kSourceSupport>(interval);
  c.weight_offsets = integer_nodes<kWeightSupport>(interval);
  // The timed work is the dense precomputed inverse-transform multiply. The
  // exact constants do not affect its cost, so bounded cosine rows suffice.
  for (int i = 0; i < kTailSupport; ++i)
    for (int j = 0; j < kTailSupport; ++j)
      c.tail_transform[i][j] = std::cos(M_PI * (i + 0.5) * j / kTailSupport)
        / kTailSupport;
  for (int i = 0; i < kSourceSupport; ++i)
    for (int j = 0; j < kSourceSupport; ++j)
      c.source_transform[i][j] = std::cos(M_PI * (i + 0.5) * j / kSourceSupport)
        / kSourceSupport;
  for (int i = 0; i < kWeightSupport; ++i)
    for (int j = 0; j < kWeightSupport; ++j)
      c.weight_transform[i][j] = std::cos(M_PI * (i + 0.5) * j / kWeightSupport)
        / kWeightSupport;
  return c;
}

inline double phaser_alpha(const BenchCase &c, double sample) {
  constexpr double modulus = 4294967296.0;
  const auto increment = static_cast<uint64_t>(c.rate * modulus / kSampleRate);
  const double phase = std::fmod(increment * sample, modulus) / modulus;
  return 2.0 * M_PI * c.center *
    std::pow(2.0, c.sweep * std::sin(2.0 * M_PI * phase));
}

void build_weights(const BenchCase &c, double alpha,
                   std::array<double, kMaxSections> &nodes_out,
                   std::array<std::complex<double>, kMaxSections> &weights_out) {
  std::array<DD, kMaxSections> nodes, zeros;
  for (int i = 0; i < c.sections; ++i) {
    nodes[i] = {-c.ratios[i], 0.0};
    zeros[i] = neg(nodes[i]);
    nodes_out[i] = value(nodes[i]);
  }
  std::array<CDD, kMaxSections> divided, next;
  for (int n = 0; n < c.sections; ++n) {
    DD numerator {1.0, 0.0};
    for (int j = 0; j < c.sections; ++j)
      numerator = mul(numerator, sub(nodes[n], zeros[j]));
    CDD source {};
    for (int p = 0; p < c.partials; ++p) {
      const CDD amplitude {{c.amp[p], 0.0}, {0.0, 0.0}};
      const CDD denominator {
        sub(nodes[n], {-c.sigma[p] / alpha, 0.0}),
        {-c.omega[p] / alpha, 0.0}};
      source = cadd(source, cdiv(amplitude, denominator));
    }
    divided[n] = {mul(source.re, numerator), mul(source.im, numerator)};
  }
  std::array<CDD, kMaxSections> coefficients;
  coefficients[0] = divided[0];
  for (int order = 1; order < c.sections; ++order) {
    for (int i = 0; i < c.sections - order; ++i)
      next[i] = cdiv_real(csub(divided[i + 1], divided[i]),
                          sub(nodes[i + order], nodes[i]));
    divided = next;
    coefficients[order] = divided[0];
  }
  for (int i = 0; i < c.sections; ++i) {
    DD transitions {1.0, 0.0};
    for (int j = i; j < c.sections - 1; ++j)
      transitions = mul(transitions, neg(nodes[j]));
    weights_out[i] = value(cdiv_real(coefficients[i], transitions));
  }
}

std::complex<double> cram_tail(const BenchCase &c, double u,
    const std::array<double, kMaxSections> &nodes,
    const std::array<std::complex<double>, kMaxSections> &weights) {
  std::array<double, kMaxSections> result {};
  result[c.sections - 1] = kAlpha0;
  for (int pole = 0; pole < 8; ++pole) {
    std::array<std::complex<double>, kMaxSections> solution {};
    for (int i = c.sections - 1; i >= 0; --i) {
      std::complex<double> rhs = i == c.sections - 1 ? 1.0 : 0.0;
      if (i + 1 < c.sections)
        rhs -= u * (-nodes[i]) * solution[i + 1];
      solution[i] = rhs / (u * nodes[i] - kTheta[pole]);
    }
    for (int i = 0; i < c.sections; ++i)
      result[i] += 2.0 * std::real(kAlpha[pole] * solution[i]);
  }
  std::complex<double> tail {};
  for (int i = 0; i < c.sections; ++i) tail += weights[i] * result[i];
  return tail;
}

// Cost surrogate for order-48 IPF CRAM.  Both forms perform one upper-
// bidiagonal complex solve per pole; three distinct order-16 calls preserve
// the exact 24-solve count without duplicating a second coefficient table in
// this timing-only program.
[[gnu::noinline]] std::complex<double> cram48_cost_tail(const BenchCase &c,
    double u, const std::array<double, kMaxSections> &nodes,
    const std::array<std::complex<double>, kMaxSections> &weights) {
  const auto first = cram_tail(c, u, nodes, weights);
  const auto second = cram_tail(c, u + 1.0e-12, nodes, weights);
  const auto third = cram_tail(c, u + 2.0e-12, nodes, weights);
  return first + 1.0e-30 * (second + third);
}

std::complex<double> uniform_tail(const BenchCase &c, double u,
    const std::array<double, kMaxSections> &nodes,
    const std::array<std::complex<double>, kMaxSections> &weights) {
  const double rho = -nodes[c.sections - 1];
  const double mean = rho * u;
  if (mean > 700.0) return 0.0;
  double poisson = std::exp(-mean);
  std::array<double, kMaxSections> column {}, next {}, result {};
  column[c.sections - 1] = 1.0;
  const int limit = std::max(32, static_cast<int>(
    mean + 14.0 * std::sqrt(mean + 1.0) + 32.0));
  for (int step = 0; step < limit; ++step) {
    for (int section = 0; section < c.sections; ++section)
      result[section] += poisson * column[section];
    next.fill(0.0);
    next[c.sections - 1] =
      (1.0 + nodes[c.sections - 1] / rho) * column[c.sections - 1];
    for (int section = c.sections - 2; section >= 0; --section)
      next[section] = (1.0 + nodes[section] / rho) * column[section]
        + (-nodes[section] / rho) * column[section + 1];
    column = next;
    poisson *= mean / (step + 1.0);
  }
  std::complex<double> tail {};
  for (int section = 0; section < c.sections; ++section)
    tail += weights[section] * result[section];
  return tail;
}

void wet_source(const BenchCase &c, double alpha,
                std::array<std::complex<double>, kMaxPartials> &out) {
  for (int p = 0; p < c.partials; ++p) {
    const std::complex<double> pole(-c.sigma[p] / alpha, c.omega[p] / alpha);
    std::complex<double> gain = c.amp[p];
    for (int s = 0; s < c.sections; ++s)
      gain *= (pole - c.ratios[s]) / (pole + c.ratios[s]);
    out[p] = gain;
  }
}

[[gnu::noinline]] double materialize_segment_direct(const BenchCase &c, double start) {
  std::array<std::complex<double>, kTailSupport> tail_values, tail_coefficients;
  std::array<std::array<std::complex<double>, kMaxPartials>, kSourceSupport>
    source_values {};
  for (int i = 0; i < kTailSupport; ++i) {
    const double sample = start + c.tail_offsets[i];
    const double alpha = phaser_alpha(c, sample);
    std::array<double, kMaxSections> nodes {};
    std::array<std::complex<double>, kMaxSections> weights {};
    build_weights(c, alpha, nodes, weights);
    const double u = alpha * sample / kSampleRate;
    double bound = 0.0;
    for (int s = 0; s < c.sections; ++s)
      bound += std::abs(weights[s]) * std::exp(nodes[s] * u);
    tail_values[i] = bound < 1.0e-12 ? 0.0 : cram_tail(c, u, nodes, weights);
  }
  for (int i = 0; i < kSourceSupport; ++i) {
    const double sample = start + c.source_offsets[i];
    wet_source(c, phaser_alpha(c, sample), source_values[i]);
  }
  for (int degree = 0; degree < kTailSupport; ++degree)
    for (int node = 0; node < kTailSupport; ++node)
      tail_coefficients[degree] += c.tail_transform[degree][node] * tail_values[node];
  double checksum = 0.0;
  for (int p = 0; p < c.partials; ++p) {
    for (int degree = 0; degree < kSourceSupport; ++degree) {
      std::complex<double> coefficient {};
      for (int node = 0; node < kSourceSupport; ++node)
        coefficient += c.source_transform[degree][node] * source_values[node][p];
      checksum += coefficient.real() * (1.0 + p + degree);
    }
  }
  for (int degree = 0; degree < kTailSupport; ++degree)
    checksum += tail_coefficients[degree].real() * (1.0 + degree);
  return checksum;
}

std::complex<double> eval_weight_polynomial(
    const std::array<std::complex<double>, kWeightSupport> &coefficients,
    double coordinate) {
  std::complex<double> following {}, after_following {};
  for (int degree = kWeightSupport - 1; degree >= 1; --degree) {
    const auto current = 2.0 * coordinate * following - after_following
      + coefficients[degree];
    after_following = following;
    following = current;
  }
  return coordinate * following - after_following + coefficients[0];
}

double materialize_segment_shared_impl(const BenchCase &c, double start,
                                       bool use_uniformization) {
  std::array<std::array<std::complex<double>, kMaxPartials>, kSourceSupport>
    source_values {};
  std::array<std::array<std::complex<double>, kMaxSections>, kWeightSupport>
    weight_values {};
  std::array<std::array<std::complex<double>, kWeightSupport>, kMaxSections>
    weight_coefficients {};
  for (int i = 0; i < kWeightSupport; ++i) {
    const double sample = start + c.weight_offsets[i];
    const double alpha = phaser_alpha(c, sample);
    std::array<double, kMaxSections> ignored_nodes {};
    build_weights(c, alpha, ignored_nodes, weight_values[i]);
  }
  for (int i = 0; i < kSourceSupport; ++i) {
    const double sample = start + c.source_offsets[i];
    const double alpha = phaser_alpha(c, sample);
    wet_source(c, alpha, source_values[i]);
  }
  for (int section = 0; section < c.sections; ++section)
    for (int degree = 0; degree < kWeightSupport; ++degree)
      for (int node = 0; node < kWeightSupport; ++node)
        weight_coefficients[section][degree] +=
          c.weight_transform[degree][node] * weight_values[node][section];

  std::array<double, kMaxSections> nodes {};
  for (int section = 0; section < c.sections; ++section)
    nodes[section] = -c.ratios[section];
  std::array<std::complex<double>, kMaxInterval> tail_values {};
  for (int i = 0; i < c.interval; ++i) {
    const double sample = start + i;
    const double coordinate = 2.0 * i / c.interval - 1.0;
    std::array<std::complex<double>, kMaxSections> weights {};
    for (int section = 0; section < c.sections; ++section)
      weights[section] = eval_weight_polynomial(
        weight_coefficients[section], coordinate);
    const double alpha = phaser_alpha(c, sample);
    const double u = alpha * sample / kSampleRate;
    double bound = 0.0;
    for (int section = 0; section < c.sections; ++section)
      bound += std::abs(weights[section]) * std::exp(nodes[section] * u);
    tail_values[i] = bound < 1.0e-12 ? 0.0
      : (use_uniformization ? uniform_tail(c, u, nodes, weights)
                            : cram48_cost_tail(c, u, nodes, weights));
  }
  double checksum = 0.0;
  for (int partial = 0; partial < c.partials; ++partial)
    for (int degree = 0; degree < kSourceSupport; ++degree) {
      std::complex<double> coefficient {};
      for (int node = 0; node < kSourceSupport; ++node)
        coefficient += c.source_transform[degree][node]
          * source_values[node][partial];
      checksum += coefficient.real() * (1.0 + partial + degree);
    }
  for (int sample = 0; sample < c.interval; ++sample)
    checksum += tail_values[sample].real() * (1.0 + sample);
  return checksum;
}

[[gnu::noinline]] double materialize_segment_shared_cram48(
    const BenchCase &c, double start) {
  return materialize_segment_shared_impl(c, start, false);
}

[[gnu::noinline]] double materialize_segment_shared_uniform(
    const BenchCase &c, double start) {
  return materialize_segment_shared_impl(c, start, true);
}

using Materializer = double (*)(const BenchCase &, double);

void run(const BenchCase &c, const char *variant, Materializer materialize) {
  volatile double sink = 0.0;
  for (int i = 0; i < 20; ++i) sink += materialize(c, c.start + (i & 1));
  std::vector<double> timings;
  constexpr int repeats = 300;
  for (int batch = 0; batch < 9; ++batch) {
    const auto before = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i)
      sink += materialize(c, c.start + (i & 1));
    const auto after = std::chrono::steady_clock::now();
    timings.push_back(std::chrono::duration<double, std::micro>(after - before).count()
                      / repeats);
  }
  std::sort(timings.begin(), timings.end());
  const double us = timings[timings.size() / 2];
  const double budget = c.interval * 1.0e6 / kSampleRate;
  std::cout << std::left << std::setw(22) << c.name << " " << std::setw(7) << variant
            << " median_us=" << std::right << std::setw(9) << std::fixed
            << std::setprecision(3) << us
            << " interval_budget_us=" << std::setw(9) << budget
            << " one_core_load=" << std::setw(7) << std::setprecision(2)
            << 100.0 * us / budget << "%\n";
  if (sink == 1.23456789) std::cerr << sink;
}

} // namespace

int main() {
  const std::array cases {
    make_case("p6_s6_i128_attack", 6, 6, 128, 0.2, 700.0, 1.5, 0),
    make_case("p6_s6_i64_low_late", 6, 6, 64, 8.0, 40.0, 3.0, 2048),
    make_case("p32_s12_i64_attack", 32, 12, 64, 0.2, 700.0, 1.5, 0),
    make_case("p32_s18_i32_attack", 32, 18, 32, 8.0, 700.0, 1.5, 0),
    make_case("p32_s18_i32_high", 32, 18, 32, 8.0, 4000.0, 3.0, 0),
    make_case("p32_s18_i128_high", 32, 18, 128, 8.0, 4000.0, 3.0, 0),
    make_case("p32_s18_i128_low", 32, 18, 128, 8.0, 40.0, 3.0, 256),
  };
  for (const auto &c : cases) {
    run(c, "direct", materialize_segment_direct);
    run(c, "cram48", materialize_segment_shared_cram48);
    run(c, "uniform", materialize_segment_shared_uniform);
  }
}
