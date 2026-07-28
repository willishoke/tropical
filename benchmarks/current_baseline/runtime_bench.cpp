// Reproducible runtime timing probe for the current-baseline and Metal-live
// qualification harnesses. It consumes already-emitted artifacts so frontend,
// LLVM/MSL load, parameter writes, and block execution remain distinct walls.

#include "c_api/tropical_c.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace
{
using Clock = std::chrono::steady_clock;

struct Options
{
  std::string ir;
  std::string manifest;
  std::string coeff;
  std::string msl;
  uint32_t buffer = 512;
  uint64_t blocks = 200;
  uint64_t warmup = 8;
  double duration_seconds = 0.0;
  double rate = 44100.0;
  bool realtime = false;
  uint64_t rss_every = 0;
  uint64_t reference_every = 0;
  uint64_t write_every = 0;
  uint32_t write_count = 1;
  std::optional<uint32_t> slot;
  std::optional<uint32_t> reference_slot;
  double slot_value = 0.75;
  std::optional<uint64_t> reload_at;
  std::optional<uint64_t> jump_at;
  uint64_t jump_index = (uint64_t{1} << 40);
  bool latency = false;
  bool dac = false;
};

std::string read_file(const std::string & path)
{
  if (path.empty()) return {};
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read '" + path + "'");
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

uint64_t parse_u64(const char * raw, const char * flag)
{
  try
  {
    std::size_t used = 0;
    const auto value = std::stoull(raw, &used);
    if (used != std::string(raw).size())
      throw std::invalid_argument("trailing input");
    return value;
  }
  catch (...)
  {
    throw std::runtime_error(std::string(flag) + " expects an unsigned integer");
  }
}

double parse_double(const char * raw, const char * flag)
{
  try
  {
    std::size_t used = 0;
    const auto value = std::stod(raw, &used);
    if (used != std::string(raw).size() || !std::isfinite(value))
      throw std::invalid_argument("invalid");
    return value;
  }
  catch (...)
  {
    throw std::runtime_error(std::string(flag) + " expects a finite number");
  }
}

Options parse_options(int argc, char ** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i)
  {
    const std::string flag = argv[i];
    auto value = [&]() -> const char * {
      if (++i >= argc) throw std::runtime_error(flag + " requires a value");
      return argv[i];
    };
    if (flag == "--ir") o.ir = value();
    else if (flag == "--manifest") o.manifest = value();
    else if (flag == "--coeff") o.coeff = value();
    else if (flag == "--msl") o.msl = value();
    else if (flag == "--buffer") o.buffer = static_cast<uint32_t>(parse_u64(value(), "--buffer"));
    else if (flag == "--blocks") o.blocks = parse_u64(value(), "--blocks");
    else if (flag == "--warmup") o.warmup = parse_u64(value(), "--warmup");
    else if (flag == "--duration-seconds")
      o.duration_seconds = parse_double(value(), "--duration-seconds");
    else if (flag == "--rate") o.rate = parse_double(value(), "--rate");
    else if (flag == "--rss-every") o.rss_every = parse_u64(value(), "--rss-every");
    else if (flag == "--reference-every")
      o.reference_every = parse_u64(value(), "--reference-every");
    else if (flag == "--write-every") o.write_every = parse_u64(value(), "--write-every");
    else if (flag == "--write-count")
      o.write_count = static_cast<uint32_t>(parse_u64(value(), "--write-count"));
    else if (flag == "--slot")
      o.slot = static_cast<uint32_t>(parse_u64(value(), "--slot"));
    else if (flag == "--reference-slot")
      o.reference_slot = static_cast<uint32_t>(parse_u64(value(), "--reference-slot"));
    else if (flag == "--slot-value") o.slot_value = parse_double(value(), "--slot-value");
    else if (flag == "--reload-at") o.reload_at = parse_u64(value(), "--reload-at");
    else if (flag == "--jump-at") o.jump_at = parse_u64(value(), "--jump-at");
    else if (flag == "--jump-index") o.jump_index = parse_u64(value(), "--jump-index");
    else if (flag == "--latency") o.latency = true;
    else if (flag == "--realtime") o.realtime = true;
    else if (flag == "--dac") o.dac = true;
    else throw std::runtime_error("unknown argument '" + flag + "'");
  }
  if (o.ir.empty() || o.manifest.empty())
    throw std::runtime_error("--ir and --manifest are required");
  if (o.buffer == 0 || o.rate <= 0.0 || o.duration_seconds < 0.0)
    throw std::runtime_error("buffer/rate/duration values are out of range");
  if ((o.latency || o.write_every > 0) && !o.slot)
    throw std::runtime_error("--latency/--write-every requires --slot");
  if (o.write_count == 0)
    throw std::runtime_error("--write-count must be positive");
  if (o.reference_every > 0 && !o.reference_slot)
    throw std::runtime_error("--reference-every requires --reference-slot");
  if (o.dac && o.duration_seconds <= 0.0)
    throw std::runtime_error("--dac requires --duration-seconds");
  return o;
}

uint64_t rss_bytes()
{
#if defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return 0;
  return static_cast<uint64_t>(info.resident_size);
#elif defined(__linux__)
  std::ifstream in("/proc/self/statm");
  uint64_t total = 0, resident = 0;
  if (!(in >> total >> resident)) return 0;
  return resident * static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end)
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

bool load(tropical_runtime_t rt, const Options & o,
          const std::string & ir, const std::string & manifest,
          const std::string & coeff, const std::string & msl)
{
  return tropical_runtime_load_ir_staged(
    rt, ir.data(), ir.size(),
    msl.empty() ? nullptr : msl.data(), msl.size(),
    coeff.empty() ? nullptr : coeff.data(), coeff.size(),
    manifest.data(), manifest.size());
}

template <typename T>
void print_array(const std::vector<T> & values)
{
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}
} // namespace

int main(int argc, char ** argv)
{
  try
  {
    const Options o = parse_options(argc, argv);
    const std::string ir = read_file(o.ir);
    const std::string manifest = read_file(o.manifest);
    const std::string coeff = read_file(o.coeff);
    const std::string msl = read_file(o.msl);

    tropical_runtime_t rt = tropical_runtime_new(o.buffer);
    if (!rt) throw std::runtime_error(tropical_last_error());
    const auto load_start = Clock::now();
    if (!load(rt, o, ir, manifest, coeff, msl))
    {
      const std::string error = tropical_last_error();
      tropical_runtime_free(rt);
      throw std::runtime_error(error);
    }
    const uint64_t load_ns = elapsed_ns(load_start, Clock::now());
    const uint32_t depth = tropical_runtime_metal_pipeline_depth(rt);

    for (uint64_t i = 0; i < o.warmup; ++i)
      tropical_runtime_process(rt);

    const double before =
      tropical_runtime_output_buffer(rt) ? tropical_runtime_output_buffer(rt)[0] : 0.0;
    std::vector<uint64_t> write_ns;
    std::vector<uint64_t> process_ns;
    std::vector<uint64_t> rss_block;
    std::vector<uint64_t> rss_value;
    std::vector<uint64_t> reference_block;
    std::vector<double> reference_snr_db;
    std::vector<double> reference_max_error;
    std::optional<uint64_t> latency_blocks;
    std::optional<uint64_t> reload_ns;
    uint64_t overrun_count = 0;
    uint64_t nonfinite_count = 0;
    double checksum = 0.0;
    double write_value = o.slot_value;
    const uint64_t deadline_ns =
      static_cast<uint64_t>(static_cast<double>(o.buffer) * 1e9 / o.rate);

    if (o.latency)
    {
      const auto start = Clock::now();
      for (uint32_t i = 0; i < o.write_count; ++i)
        tropical_runtime_set_slot(rt, *o.slot + i, write_value);
      write_ns.push_back(elapsed_ns(start, Clock::now()));
    }

    const auto run_start = Clock::now();
    uint64_t block = 0;
    uint64_t run_ns = 0;
    tropical_dac_stats_t dac_stats{};
    std::vector<std::string> dac_snapshot_labels;
    std::vector<tropical_dac_stats_t> dac_snapshots;
    bool dac_aborted = false;
    std::string dac_abort_reason;
    if (o.dac)
    {
      tropical_dac_t dac =
        tropical_dac_new_runtime(rt, static_cast<unsigned int>(o.rate), 2);
      if (!dac) throw std::runtime_error(tropical_last_error());
      tropical_dac_start(dac);
      if (!tropical_dac_is_running(dac))
      {
        const std::string error = tropical_last_error();
        tropical_dac_free(dac);
        throw std::runtime_error(
          error.empty() ? "DAC did not start on the default device" : error);
      }

      auto snapshot = [&](const std::string & label) {
        tropical_dac_stats_t stats{};
        tropical_dac_get_stats(dac, &stats);
        dac_snapshot_labels.push_back(label);
        dac_snapshots.push_back(stats);
      };

      // Startup/device-prime status is useful evidence but is not part of the
      // measured steady-state window. Preserve it, then reset at one fixed
      // two-second boundary before any qualification event is scheduled.
      std::this_thread::sleep_for(std::chrono::seconds(2));
      snapshot("startup_warmup_before_reset");
      tropical_dac_reset_stats(dac);
      const auto measured_start = Clock::now();
      const uint64_t baseline_blocks =
        std::max<uint64_t>(1, static_cast<uint64_t>(2.0 * o.rate / o.buffer));
      uint64_t next_write = std::numeric_limits<uint64_t>::max();
      uint64_t next_rss = 0;
      bool baseline_recorded = false;
      bool writes_recorded = false;
      bool jump_recorded = false;
      bool jumped = false;
      bool reloaded = false;
      while (std::chrono::duration<double>(Clock::now() - measured_start).count()
               < o.duration_seconds)
      {
        tropical_dac_get_stats(dac, &dac_stats);
        block = dac_stats.callback_count;
        if (dac_stats.underrun_count > 0 || dac_stats.overrun_count > 0)
        {
          dac_aborted = true;
          dac_abort_reason = dac_stats.underrun_count > 0
            ? "post_reset_underrun" : "post_reset_callback_overrun";
          snapshot("measured_failure_abort");
          break;
        }
        if (!baseline_recorded && block >= baseline_blocks)
        {
          snapshot("clean_baseline_after_reset");
          baseline_recorded = true;
          next_write = o.write_every > 0 ? block + o.write_every
                                         : std::numeric_limits<uint64_t>::max();
        }
        if (baseline_recorded && o.write_every > 0 && block >= next_write)
        {
          write_value =
            write_value == o.slot_value ? o.slot_value * 0.5 : o.slot_value;
          const auto start = Clock::now();
          for (uint32_t i = 0; i < o.write_count; ++i)
            tropical_runtime_set_slot(rt, *o.slot + i, write_value);
          write_ns.push_back(elapsed_ns(start, Clock::now()));
          next_write += o.write_every;
        }
        if (!jumped && o.jump_at && block >= *o.jump_at)
        {
          snapshot("after_periodic_writes");
          writes_recorded = true;
          tropical_runtime_set_sample_index(rt, o.jump_index);
          jumped = true;
        }
        if (!reloaded && o.reload_at && block >= *o.reload_at)
        {
          snapshot("after_clock_jump");
          jump_recorded = true;
          const auto start = Clock::now();
          if (!load(rt, o, ir, manifest, coeff, msl))
            throw std::runtime_error(tropical_last_error());
          reload_ns = elapsed_ns(start, Clock::now());
          reloaded = true;
        }
        if (o.rss_every > 0 && block >= next_rss)
        {
          rss_block.push_back(block);
          rss_value.push_back(rss_bytes());
          next_rss = block + o.rss_every;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      if (!dac_aborted)
      {
        if (!baseline_recorded) snapshot("clean_baseline_after_reset");
        if (!writes_recorded) snapshot("after_periodic_writes");
        if (!jump_recorded) snapshot("after_clock_jump");
        snapshot("after_hot_swap");
      }
      tropical_dac_get_stats(dac, &dac_stats);
      overrun_count = dac_stats.overrun_count;
      tropical_dac_stop(dac);
      snapshot("after_stop");
      tropical_dac_free(dac);
      run_ns = elapsed_ns(measured_start, Clock::now());
    }
    else
    {
      auto next_deadline = run_start;
      while (o.duration_seconds > 0.0
               ? std::chrono::duration<double>(Clock::now() - run_start).count()
                   < o.duration_seconds
               : block < o.blocks)
      {
        if (o.write_every > 0 && block > 0 && block % o.write_every == 0)
        {
          write_value =
            write_value == o.slot_value ? o.slot_value * 0.5 : o.slot_value;
          const auto start = Clock::now();
          for (uint32_t i = 0; i < o.write_count; ++i)
            tropical_runtime_set_slot(rt, *o.slot + i, write_value);
          write_ns.push_back(elapsed_ns(start, Clock::now()));
        }
        if (o.jump_at && block == *o.jump_at)
          tropical_runtime_set_sample_index(rt, o.jump_index);
        if (o.reload_at && block == *o.reload_at)
        {
          const auto start = Clock::now();
          if (!load(rt, o, ir, manifest, coeff, msl))
            throw std::runtime_error(tropical_last_error());
          reload_ns = elapsed_ns(start, Clock::now());
        }

        const auto start = Clock::now();
        tropical_runtime_process(rt);
        const uint64_t ns = elapsed_ns(start, Clock::now());
        process_ns.push_back(ns);
        if (ns > deadline_ns) ++overrun_count;

        const double * out = tropical_runtime_output_buffer(rt);
        if (out)
        {
          const double first = out[0];
          checksum += first;
          if (!std::isfinite(first)) ++nonfinite_count;
          if (o.latency && !latency_blocks
              && std::fabs(first - write_value) <= 1e-6
              && std::fabs(first - before) > 1e-6)
            latency_blocks = block;
        }
        if (out && o.reference_every > 0 && block % o.reference_every == 0)
        {
          std::vector<double> reference(o.buffer, 0.0);
          const uint64_t next =
            static_cast<uint64_t>(tropical_runtime_current_sample_index(rt));
          const uint64_t start_index = next >= o.buffer ? next - o.buffer : 0;
          const uint32_t slot_id = *o.reference_slot;
          if (!tropical_runtime_render_window(
                rt, start_index, o.buffer, &slot_id, 1, reference.data()))
            throw std::runtime_error("render_window reference failed");
          double signal = 0.0;
          double error = 0.0;
          double max_error = 0.0;
          for (uint32_t i = 0; i < o.buffer; ++i)
          {
            signal += reference[i] * reference[i];
            const double delta = reference[i] - out[i];
            error += delta * delta;
            max_error = std::max(max_error, std::fabs(delta));
          }
          reference_block.push_back(block);
          reference_snr_db.push_back(
            error == 0.0 ? 999.0
                         : 10.0 * std::log10(signal / error));
          reference_max_error.push_back(max_error);
        }
        if (o.rss_every > 0 && block % o.rss_every == 0)
        {
          rss_block.push_back(block);
          rss_value.push_back(rss_bytes());
        }

        ++block;
        if (o.realtime)
        {
          next_deadline += std::chrono::nanoseconds(deadline_ns);
          std::this_thread::sleep_until(next_deadline);
        }
      }
      run_ns = elapsed_ns(run_start, Clock::now());
    }
    const uint64_t final_rss = rss_bytes();
    tropical_runtime_free(rt);

    std::cout << "{\"schema\":\"tropical_runtime_bench_1\""
              << ",\"buffer_length\":" << o.buffer
              << ",\"sample_rate\":" << std::setprecision(17) << o.rate
              << ",\"backend\":\"" << (msl.empty() ? "jit" : "metal") << "\""
              << ",\"pipeline_depth\":" << depth
              << ",\"load_ns\":" << load_ns
              << ",\"run_ns\":" << run_ns
              << ",\"blocks\":" << block
              << ",\"dac_mode\":" << (o.dac ? "true" : "false")
              << ",\"dac_aborted\":" << (dac_aborted ? "true" : "false")
              << ",\"dac_abort_reason\":";
    if (dac_abort_reason.empty()) std::cout << "null";
    else std::cout << '"' << dac_abort_reason << '"';
    std::cout
              << ",\"deadline_ns\":" << deadline_ns
              << ",\"overrun_count\":" << overrun_count
              << ",\"nonfinite_count\":" << nonfinite_count
              << ",\"checksum\":" << checksum
              << ",\"final_rss_bytes\":" << final_rss
              << ",\"latency_blocks\":";
    if (latency_blocks) std::cout << *latency_blocks; else std::cout << "null";
    std::cout << ",\"predicted_latency_blocks\":" << depth
              << ",\"reload_ns\":";
    if (reload_ns) std::cout << *reload_ns; else std::cout << "null";
    std::cout << ",\"process_ns\":";
    print_array(process_ns);
    std::cout << ",\"write_ns\":";
    print_array(write_ns);
    std::cout << ",\"rss_blocks\":";
    print_array(rss_block);
    std::cout << ",\"rss_bytes\":";
    print_array(rss_value);
    std::cout << ",\"reference_blocks\":";
    print_array(reference_block);
    std::cout << ",\"reference_snr_db\":";
    print_array(reference_snr_db);
    std::cout << ",\"reference_max_error\":";
    print_array(reference_max_error);
    std::cout << ",\"dac_stats\":{\"callback_count\":"
              << dac_stats.callback_count
              << ",\"avg_callback_ms\":" << dac_stats.avg_callback_ms
              << ",\"max_callback_ms\":" << dac_stats.max_callback_ms
              << ",\"underrun_count\":" << dac_stats.underrun_count
              << ",\"overrun_count\":" << dac_stats.overrun_count << '}';
    std::cout << ",\"dac_snapshots\":[";
    for (std::size_t i = 0; i < dac_snapshots.size(); ++i)
    {
      if (i) std::cout << ',';
      const auto & stats = dac_snapshots[i];
      std::cout << "{\"label\":\"" << dac_snapshot_labels[i]
                << "\",\"callback_count\":" << stats.callback_count
                << ",\"avg_callback_ms\":" << stats.avg_callback_ms
                << ",\"max_callback_ms\":" << stats.max_callback_ms
                << ",\"underrun_count\":" << stats.underrun_count
                << ",\"overrun_count\":" << stats.overrun_count << '}';
    }
    std::cout << ']';
    std::cout << "}\n";
    return 0;
  }
  catch (const std::exception & e)
  {
    std::cerr << "tropical_runtime_bench: " << e.what() << '\n';
    return 1;
  }
}
