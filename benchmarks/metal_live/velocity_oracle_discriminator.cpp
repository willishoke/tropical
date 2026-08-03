#include "runtime/FlatRuntime.hpp"

#include <algorithm>
#include <array>
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
constexpr uint64_t kReverseDistance = 1536;
constexpr uint64_t kBatchLeadFrames = 8 * kBuffer;
constexpr uint64_t kFirstEventBlock = 2666;
constexpr uint64_t kEventStrideBlocks = 86;
constexpr uint64_t kJumpAppliedBlock = 1292;
constexpr uint64_t kCaptureOffsetBlocks = 3878;

// The retained 60-second row's 20 post-swap velocity effective coordinates.
// The no-DAC probe steers the source far enough ahead of each batch that the
// four production dispatches land the velocity transaction at this exact E.
constexpr std::array<uint64_t, 20> kBatchCoordinates = {
  1369600, 1413120, 1457152, 1501696, 1545216,
  1099511654400, 1099511698432, 1099511743488, 1099511785984,
  1099511830016,
  1809408, 1853952, 1899520, 1942016, 1986560,
  1099511654400, 1099511698432, 1099511743488, 1099511787008,
  1099511831040,
};

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

struct Event
{
  uint64_t jump_coordinate = 0;
  uint64_t observed = 0;
  uint64_t effective = 0;
  double velocity = 0.0;
};

struct Comparison
{
  double snr = 0.0;
  double max_error = 0.0;
  double signal = 0.0;
  double reference_checksum = 0.0;
  double actual_checksum = 0.0;
};

struct Result
{
  Comparison exact;
  Comparison f32_origin;
  std::vector<Event> velocity_events;
  uint64_t final_velocity_e = 0;
  uint64_t capture_start = 0;
  double final_velocity = 0.0;
  double tau_base = 0.0;
  float tau_base_f32 = 0.0f;
};

struct DelayedOracleResult
{
  Comparison exact;
  Comparison stale;
};

uint64_t delayed_event_sample_index(uint64_t round)
{
  const uint64_t block = kFirstEventBlock + round * kEventStrideBlocks;
  return kJumpIndex + (block - kJumpAppliedBlock) * kBuffer;
}

Comparison compare(
  const std::vector<double> & reference,
  const std::vector<double> & actual)
{
  if (reference.size() != actual.size())
    throw std::runtime_error("comparison length mismatch");
  double signal = 0.0;
  double error = 0.0;
  double max_error = 0.0;
  double reference_checksum = 0.0;
  double actual_checksum = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i)
  {
    signal += reference[i] * reference[i];
    const double delta = reference[i] - actual[i];
    error += delta * delta;
    max_error = std::max(max_error, std::fabs(delta));
    const double weight = static_cast<double>(i + 1);
    reference_checksum += weight * reference[i];
    actual_checksum += weight * actual[i];
  }
  return {
    error == 0.0
      ? (signal > 0.0 ? 999.0 : 0.0)
      : (signal > 0.0 ? 10.0 * std::log10(signal / error) : -999.0),
    max_error,
    signal,
    reference_checksum,
    actual_checksum,
  };
}

void acknowledge_latest(FlatRuntime & metal)
{
  const uint64_t requested = metal.metal_published_activation_epoch();
  for (uint32_t attempt = 0;
       attempt < 16
       && metal.metal_acknowledged_activation_epoch() < requested;
       ++attempt)
    metal.process_offline();
  if (requested == 0
      || metal.metal_acknowledged_activation_epoch() < requested)
    throw std::runtime_error("Metal activation did not acknowledge");
}

std::vector<double> render_reference(
  FlatRuntime & jit, uint32_t reference_slot, uint64_t capture_start)
{
  std::vector<double> output(kBuffer, 0.0);
  if (!jit.render_window(
        capture_start, kBuffer, &reference_slot, 1, output.data()))
    throw std::runtime_error("reference render failed");
  return output;
}

Result run_scenario(
  const Artifacts & initial, const Artifacts & replacement,
  bool toggle_velocity, bool toggle_glide, bool toggle_anchor,
  bool hot_swap, bool reverse_crossing,
  bool diagnostic_f32_origin)
{
  FlatRuntime metal(kBuffer);
  FlatRuntime exact(kBuffer);
  FlatRuntime f32_origin(kBuffer);
  load(metal, initial, true);
  load(exact, initial, false);
  load(f32_origin, initial, false);
  if (hot_swap)
  {
    load(metal, replacement, true);
    load(exact, replacement, false);
    load(f32_origin, replacement, false);
  }
  acknowledge_latest(metal);

  const uint32_t tau_slot = metal.slot_index("param:master.tau_base");
  const uint32_t velocity_slot =
    metal.slot_index("param:master.velocity");
  const uint32_t reference_slot = metal.slot_index("__root__.out");
  if (tau_slot == UINT32_MAX || velocity_slot == UINT32_MAX
      || reference_slot == UINT32_MAX
      || exact.slot_index("param:master.tau_base") != tau_slot
      || exact.slot_index("param:master.velocity") != velocity_slot
      || exact.slot_index("__root__.out") != reference_slot)
    throw std::runtime_error("master clock slots missing");

  std::vector<Event> velocity_events;
  uint64_t final_velocity_e = 0;
  for (uint64_t round = 0; round < kBatchCoordinates.size(); ++round)
  {
    const uint64_t jump_coordinate =
      kBatchCoordinates[round] - kBatchLeadFrames;
    metal.set_sample_index(jump_coordinate);
    acknowledge_latest(metal);

    const bool high = (round & 1U) == 0;
    const double values[] = {
      high ? 260.0 : 180.0,
      toggle_glide ? (high ? 0.5 : 0.0) : 0.0,
      toggle_anchor ? (high ? 65.0 : 55.0) : 55.0,
      toggle_velocity ? (high ? 0.75 : 1.0) : 1.0,
    };
    const char * names[] = {
      "bank.freq", "canary.morph", "canary.freq", "master.velocity"
    };
    for (std::size_t i = 0; i < 4; ++i)
    {
      const auto production =
        metal.dispatch_param_sync(names[i], values[i]);
      const auto replay =
        exact.dispatch_param_sync_at_sample_index(
          names[i], values[i], production.effective_sample_index);
      const auto f32_replay =
        f32_origin.dispatch_param_sync_at_sample_index(
          names[i], values[i], production.effective_sample_index);
      if (!production.ok || !replay.ok || !f32_replay.ok
          || replay.effective_sample_index
               != production.effective_sample_index)
        throw std::runtime_error("event dispatch mismatch");
      acknowledge_latest(metal);
      if (i == 3)
      {
        final_velocity_e = production.effective_sample_index;
        velocity_events.push_back({
          jump_coordinate,
          production.observed_sample_index,
          production.effective_sample_index,
          values[i],
        });
      }
    }
  }

  if (final_velocity_e < kReverseDistance)
    throw std::runtime_error("final velocity E is too small");
  const uint64_t capture_start = reverse_crossing
    ? final_velocity_e - kReverseDistance
    : final_velocity_e + kReverseDistance;
  metal.set_sample_index(capture_start);
  acknowledge_latest(metal);
  const std::vector<double> actual = metal.outputBuffer;
  const std::vector<double> exact_reference =
    render_reference(exact, reference_slot, capture_start);

  Result result;
  result.exact = compare(exact_reference, actual);
  if (diagnostic_f32_origin)
  {
    f32_origin.set_slot(
      tau_slot,
      static_cast<double>(
        static_cast<float>(f32_origin.get_slot(tau_slot))));
    result.f32_origin =
      compare(
        render_reference(f32_origin, reference_slot, capture_start), actual);
  }
  result.velocity_events = std::move(velocity_events);
  result.final_velocity_e = final_velocity_e;
  result.capture_start = capture_start;
  result.final_velocity = exact.get_slot(velocity_slot);
  result.tau_base = exact.get_slot(tau_slot);
  result.tau_base_f32 = static_cast<float>(result.tau_base);
  return result;
}

DelayedOracleResult run_delayed_oracle(
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
  acknowledge_latest(metal);

  const uint32_t reference_slot = metal.slot_index("__root__.out");
  if (reference_slot == UINT32_MAX
      || exact.slot_index("__root__.out") != reference_slot
      || stale.slot_index("__root__.out") != reference_slot)
    throw std::runtime_error("delayed oracle reference slot missing");

  for (uint64_t round = 0; round < 15; ++round)
  {
    const uint64_t batch_start = delayed_event_sample_index(round);
    metal.set_sample_index(batch_start - kBuffer);
    acknowledge_latest(metal);
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
      // Force a callback boundary inside the final control batch.
      // Production and exact replay observe the later effective epoch; the
      // obsolete oracle assigns every write to batch_start.
      if (i == 2 && round == 14)
        metal.process_offline();
      const auto production =
        metal.dispatch_param_sync(names[i], values[i]);
      const auto replay = exact.dispatch_param_sync_at_sample_index(
        names[i], values[i], production.effective_sample_index);
      const auto legacy = stale.dispatch_param_sync_at_sample_index(
        names[i], values[i], batch_start);
      if (!production.ok || !replay.ok || !legacy.ok)
        throw std::runtime_error("delayed oracle dispatch failed");
      acknowledge_latest(metal);
    }
  }

  constexpr uint64_t capture_start =
    kJumpIndex + kCaptureOffsetBlocks * kBuffer;
  metal.set_sample_index(capture_start);
  acknowledge_latest(metal);
  const std::vector<double> actual = metal.outputBuffer;
  return {
    compare(
      render_reference(exact, reference_slot, capture_start), actual),
    compare(
      render_reference(stale, reference_slot, capture_start), actual),
  };
}

void print_comparison(const char * name, const Comparison & c)
{
  std::cout << '"' << name << "\":{"
            << "\"snr_db\":" << c.snr
            << ",\"max_error\":" << c.max_error
            << ",\"signal_energy\":" << c.signal
            << ",\"reference_checksum\":" << c.reference_checksum
            << ",\"actual_checksum\":" << c.actual_checksum
            << '}';
}

void print_result(const char * name, const Result & result)
{
  std::cout << '"' << name << "\":{"
            << "\"final_velocity_e\":" << result.final_velocity_e
            << ",\"capture_start\":" << result.capture_start
            << ",\"final_velocity\":" << result.final_velocity
            << ",\"tau_base_f64\":" << result.tau_base
            << ",\"tau_base_f32\":" << result.tau_base_f32
            << ',';
  print_comparison("comparison", result.exact);
  std::cout << '}';
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

  const Result no_glide = run_scenario(
    initial, replacement, true, false, true, true, true, false);
  const Result no_anchor = run_scenario(
    initial, replacement, true, true, false, true, true, false);
  const Result reverse = run_scenario(
    initial, replacement, true, true, true, true, true, true);
  const Result forward = run_scenario(
    initial, replacement, true, true, true, true, false, false);
  const Result no_velocity = run_scenario(
    initial, replacement, false, true, true, true, true, false);
  const Result no_swap = run_scenario(
    initial, replacement, true, true, true, false, true, false);
  const DelayedOracleResult delayed_oracle =
    run_delayed_oracle(initial, replacement);

  const bool passed =
    reverse.exact.snr > 140.0
    && forward.exact.snr > 140.0
    && no_velocity.exact.snr > 140.0
    && no_swap.exact.snr > 140.0
    && no_glide.exact.snr > 140.0
    && no_anchor.exact.snr > 140.0
    && delayed_oracle.exact.snr > 140.0
    && delayed_oracle.stale.snr < 100.0;

  std::cout << std::setprecision(17)
            << "{\"schema\":"
               "\"tropical_metal_velocity_oracle_discriminator_2\""
            << ",\"passed\":" << (passed ? "true" : "false")
            << ",\"buffer_length\":" << kBuffer
            << ",\"far_jump_index\":" << kJumpIndex
            << ",\"reverse_distance\":" << kReverseDistance
            << ",\"jump_coordinates\":[";
  for (std::size_t i = 0; i < kBatchCoordinates.size(); ++i)
  {
    if (i) std::cout << ',';
    std::cout << kBatchCoordinates[i] - kBatchLeadFrames;
  }
  std::cout << "],\"velocity_events\":[";
  for (std::size_t i = 0; i < reverse.velocity_events.size(); ++i)
  {
    if (i) std::cout << ',';
    const auto & event = reverse.velocity_events[i];
    std::cout << "{\"jump_coordinate\":" << event.jump_coordinate
              << ",\"observed_sample_index\":" << event.observed
              << ",\"effective_sample_index\":" << event.effective
              << ",\"value\":" << event.velocity << '}';
  }
  std::cout << "],";
  print_result("diagnostic_no_glide", no_glide);
  std::cout << ',';
  print_result("diagnostic_no_anchor", no_anchor);
  std::cout << ',';
  print_result("A_full_reverse_crossing", reverse);
  std::cout << ',';
  print_result("B_capture_after_last_E", forward);
  std::cout << ',';
  print_result("C_no_velocity_toggles", no_velocity);
  std::cout << ',';
  print_result("D_no_hot_swap", no_swap);
  std::cout << ",\"E_f32_origin_jit_oracle\":{";
  print_comparison("comparison", reverse.f32_origin);
  std::cout << "},\"F_stale_E_oracle\":{";
  print_comparison("exact_replay", delayed_oracle.exact);
  std::cout << ',';
  print_comparison("stale_replay", delayed_oracle.stale);
  std::cout << "}}\n";
  return passed ? 0 : 1;
}
catch (const std::exception & error)
{
  std::cerr << "metal_velocity_oracle_discriminator: "
            << error.what() << '\n';
  return 1;
}
