#include "runtime/FlatRuntime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using tropical_runtime::FlatRuntime;

namespace
{

constexpr uint32_t kBuffer = 512;
constexpr uint64_t kJumpIndex = uint64_t{1} << 40;
constexpr uint32_t kReferenceSlot = 13;
constexpr uint64_t kFirstEventBlock = 2666;
constexpr uint64_t kEventStrideBlocks = 86;
constexpr uint64_t kJumpAppliedBlock = 1292;
constexpr uint64_t kCaptureOffsetBlocks = 3878;

std::string read_file(const std::string & path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return {
    std::istreambuf_iterator<char>(in),
    std::istreambuf_iterator<char>()
  };
}

struct Artifacts
{
  std::string ir;
  std::string coeff;
  std::string msl;
  std::string manifest;
};

void load(FlatRuntime & rt, const Artifacts & a, bool metal)
{
  if (!rt.load_ir_staged(
        read_file(a.ir), metal ? read_file(a.msl) : std::string{},
        read_file(a.coeff), read_file(a.manifest)))
    throw std::runtime_error("load failed");
}

struct Result
{
  double snr = 0.0;
  double max_error = 0.0;
  double signal = 0.0;
  double reference_max = 0.0;
  double actual_max = 0.0;
  double tau_base = 0.0;
  float tau_base_f32 = 0.0f;
};

struct Comparison
{
  double snr = 0.0;
  double max_error = 0.0;
};

Comparison compare(
  const std::vector<double> & reference,
  const std::vector<double> & actual)
{
  double signal = 0.0;
  double error = 0.0;
  double max_error = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i)
  {
    signal += reference[i] * reference[i];
    const double delta = reference[i] - actual[i];
    error += delta * delta;
    max_error = std::max(max_error, std::fabs(delta));
  }
  return {
    error == 0.0 ? 999.0 : 10.0 * std::log10(signal / error),
    max_error,
  };
}

uint64_t event_sample_index(uint64_t round)
{
  const uint64_t block = kFirstEventBlock + round * kEventStrideBlocks;
  return kJumpIndex + (block - kJumpAppliedBlock) * kBuffer;
}

Result run_scenario(
  const Artifacts & initial, const Artifacts & replacement,
  bool toggle_velocity)
{
  FlatRuntime metal(kBuffer);
  FlatRuntime jit(kBuffer);
  load(metal, initial, true);
  load(jit, initial, false);
  // The actual row hot-swapped before this 15-write tail. Loading the
  // replacement here deliberately starts from its fresh post-swap defaults.
  load(metal, replacement, true);
  load(jit, replacement, false);

  const uint32_t tau_slot = metal.slot_index("param:master.tau_base");
  if (tau_slot == UINT32_MAX
      || jit.slot_index("param:master.tau_base") != tau_slot)
    throw std::runtime_error("tau slot missing");

  for (uint64_t round = 0; round < 15; ++round)
  {
    const uint64_t now = event_sample_index(round);
    metal.set_sample_index(now - kBuffer);
    jit.set_sample_index(now - kBuffer);
    metal.process();
    jit.process();
    if (metal.current_sample_index() != now
        || jit.current_sample_index() != now)
      throw std::runtime_error("failed to publish exact event index");

    const bool high = (round & 1U) == 0;
    const double values[] = {
      high ? 260.0 : 180.0,
      high ? 0.5 : 0.0,
      high ? 65.0 : 55.0,
      toggle_velocity ? (high ? 0.75 : 1.0) : 1.0,
    };
    const char * names[] = {
      "bank.freq", "canary.morph", "canary.freq", "master.velocity"
    };
    for (std::size_t i = 0; i < 4; ++i)
    {
      const auto metal_dispatch =
        metal.dispatch_param_sync(names[i], values[i]);
      const auto jit_dispatch =
        jit.dispatch_param_sync(names[i], values[i]);
      if (!metal_dispatch.ok || !jit_dispatch.ok
          || metal_dispatch.applied_sample_index != now
          || jit_dispatch.applied_sample_index != now)
        throw std::runtime_error("event dispatch mismatch");
    }
  }

  constexpr uint64_t capture_start =
    kJumpIndex + kCaptureOffsetBlocks * kBuffer;
  metal.set_sample_index(capture_start);
  metal.process();
  std::vector<double> reference(kBuffer, 0.0);
  if (!jit.render_window(
        capture_start, kBuffer, &kReferenceSlot, 1, reference.data()))
    throw std::runtime_error("reference render failed");

  double signal = 0.0;
  double error = 0.0;
  double max_error = 0.0;
  double reference_max = 0.0;
  double actual_max = 0.0;
  for (uint32_t i = 0; i < kBuffer; ++i)
  {
    const double actual = metal.outputBuffer[i];
    reference_max = std::max(reference_max, std::fabs(reference[i]));
    actual_max = std::max(actual_max, std::fabs(actual));
    signal += reference[i] * reference[i];
    const double delta = reference[i] - actual;
    error += delta * delta;
    max_error = std::max(max_error, std::fabs(delta));
  }
  const double tau = jit.get_slot(tau_slot);
  return {
    error == 0.0 ? 999.0 : 10.0 * std::log10(signal / error),
    max_error,
    signal,
    reference_max,
    actual_max,
    tau,
    static_cast<float>(tau),
  };
}

std::pair<Comparison, Comparison> run_delayed_oracle(
  const Artifacts & initial, const Artifacts & replacement)
{
  FlatRuntime metal(kBuffer);
  FlatRuntime exact(kBuffer);
  FlatRuntime stale(kBuffer);
  load(metal, initial, true);
  load(exact, initial, false);
  load(stale, initial, false);
  load(metal, replacement, true);
  load(exact, replacement, false);
  load(stale, replacement, false);

  for (uint64_t round = 0; round < 15; ++round)
  {
    const uint64_t batch_start = event_sample_index(round);
    metal.set_sample_index(batch_start - kBuffer);
    metal.process();
    const bool high = (round & 1U) == 0;
    const double values[] = {
      high ? 260.0 : 180.0,
      high ? 0.5 : 0.0,
      high ? 65.0 : 55.0,
      high ? 0.75 : 1.0,
    };
    const char * names[] = {
      "bank.freq", "canary.morph", "canary.freq", "master.velocity"
    };
    for (std::size_t i = 0; i < 4; ++i)
    {
      // A batch takes up to 10.264 ms against an 11.610 ms callback period.
      // Force one possible crossing before the final anchor. Production and
      // exact replay use the later boundary; the legacy oracle incorrectly
      // holds batch_start for both anchor and the following velocity write.
      if (i == 2 && round == 14)
      {
        metal.set_sample_index(batch_start);
        metal.process();
      }
      const auto production =
        metal.dispatch_param_sync(names[i], values[i]);
      const auto replay = exact.dispatch_param_sync_at_sample_index(
        names[i], values[i], production.applied_sample_index);
      const auto legacy = stale.dispatch_param_sync_at_sample_index(
        names[i], values[i], batch_start);
      if (!production.ok || !replay.ok || !legacy.ok)
        throw std::runtime_error("delayed oracle dispatch failed");
    }
  }

  constexpr uint64_t capture_start =
    kJumpIndex + kCaptureOffsetBlocks * kBuffer;
  metal.set_sample_index(capture_start);
  metal.process();
  const std::vector<double> actual = metal.outputBuffer;
  std::vector<double> exact_reference(kBuffer, 0.0);
  std::vector<double> stale_reference(kBuffer, 0.0);
  if (!exact.render_window(
        capture_start, kBuffer, &kReferenceSlot, 1, exact_reference.data())
      || !stale.render_window(
        capture_start, kBuffer, &kReferenceSlot, 1, stale_reference.data()))
    throw std::runtime_error("delayed oracle reference render failed");
  return {
    compare(exact_reference, actual),
    compare(stale_reference, actual),
  };
}

} // namespace

int main(int argc, char ** argv)
try
{
  if (argc != 9)
    throw std::runtime_error(
      "expected initial/replacement ir coeff msl manifest paths");
  const Artifacts initial{argv[1], argv[2], argv[3], argv[4]};
  const Artifacts replacement{argv[5], argv[6], argv[7], argv[8]};
  const Result a = run_scenario(initial, replacement, true);
  const Result b = run_scenario(initial, replacement, false);
  const auto [exact, stale] = run_delayed_oracle(initial, replacement);
  const bool passed =
    a.snr > 140.0 && b.snr > 140.0
    && exact.snr > 140.0 && stale.snr < 100.0;

  std::cout << std::setprecision(17)
            << "{\"schema\":\"tropical_metal_velocity_oracle_discriminator_1\""
            << ",\"passed\":" << (passed ? "true" : "false")
            << ",\"buffer_length\":" << kBuffer
            << ",\"jump_index\":" << kJumpIndex
            << ",\"event_sample_indices\":[";
  for (uint64_t round = 0; round < 15; ++round)
  {
    if (round) std::cout << ',';
    std::cout << event_sample_index(round);
  }
  std::cout << "],\"A_velocity_toggle\":{\"snr_db\":" << a.snr
            << ",\"max_error\":" << a.max_error
            << ",\"signal\":" << a.signal
            << ",\"reference_max\":" << a.reference_max
            << ",\"actual_max\":" << a.actual_max
            << ",\"tau_base_f64\":" << a.tau_base
            << ",\"tau_base_f32\":" << a.tau_base_f32
            << "},\"B_velocity_noop\":{\"snr_db\":" << b.snr
            << ",\"max_error\":" << b.max_error
            << ",\"signal\":" << b.signal
            << ",\"reference_max\":" << b.reference_max
            << ",\"actual_max\":" << b.actual_max
            << ",\"tau_base_f64\":" << b.tau_base
            << ",\"tau_base_f32\":" << b.tau_base_f32
            << "},\"C_forced_stale_oracle\":{"
            << "\"exact_replay_snr_db\":" << exact.snr
            << ",\"exact_replay_max_error\":" << exact.max_error
            << ",\"legacy_batch_start_snr_db\":" << stale.snr
            << ",\"legacy_batch_start_max_error\":" << stale.max_error
            << "}}\n";
  return passed ? 0 : 1;
}
catch (const std::exception & error)
{
  std::cerr << "metal_velocity_oracle_discriminator: "
            << error.what() << '\n';
  return 1;
}
