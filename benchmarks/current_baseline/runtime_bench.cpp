// Reproducible runtime timing probe for the current-baseline and Metal-live
// qualification harnesses. It consumes already-emitted artifacts so frontend,
// LLVM/MSL load, parameter writes, and block execution remain distinct walls.

#include "c_api/tropical_c.h"
#include "runtime/FlatRuntime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace
{
using Clock = std::chrono::steady_clock;

struct ParamEvent
{
  std::string discipline;
  std::string name;
  double low = 0.0;
  double high = 0.0;
};

struct Options
{
  std::string ir;
  std::string manifest;
  std::string coeff;
  std::string msl;
  std::string reload_ir;
  std::string reload_manifest;
  std::string reload_coeff;
  std::string reload_msl;
  uint32_t buffer = 512;
  uint64_t blocks = 200;
  uint64_t warmup = 8;
  double duration_seconds = 0.0;
  double rate = 44100.0;
  bool realtime = false;
  uint64_t rss_every = 0;
  uint64_t reference_every = 0;
  uint64_t write_every = 0;
  std::optional<uint64_t> write_stop_at;
  uint32_t write_count = 1;
  std::vector<ParamEvent> param_events;
  std::optional<uint32_t> slot;
  std::optional<uint32_t> reference_slot;
  double slot_value = 0.75;
  std::optional<uint64_t> reload_at;
  std::optional<uint64_t> jump_at;
  uint64_t jump_index = (uint64_t{1} << 40);
  bool latency = false;
  bool dac = false;
};

struct RuntimeDeleter
{
  void operator()(void * value) const
  {
    tropical_runtime_free(value);
  }
};
using RuntimeOwner = std::unique_ptr<void, RuntimeDeleter>;

struct DacDeleter
{
  void operator()(void * value) const
  {
    tropical_dac_free(value);
  }
};
using DacOwner = std::unique_ptr<void, DacDeleter>;

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

ParamEvent parse_param_event(const char * raw)
{
  const std::string text = raw;
  std::array<std::string, 4> fields;
  std::size_t begin = 0;
  for (std::size_t i = 0; i < fields.size(); ++i)
  {
    const std::size_t end =
      i + 1 == fields.size() ? std::string::npos : text.find(',', begin);
    if (end == std::string::npos && i + 1 != fields.size())
      throw std::runtime_error(
        "--param-event expects discipline,name,low,high");
    fields[i] = text.substr(
      begin, end == std::string::npos ? end : end - begin);
    begin = end == std::string::npos ? text.size() : end + 1;
  }
  if (begin != text.size() || fields[0].empty() || fields[1].empty())
    throw std::runtime_error(
      "--param-event expects discipline,name,low,high");
  return {
    fields[0], fields[1],
    parse_double(fields[2].c_str(), "--param-event low"),
    parse_double(fields[3].c_str(), "--param-event high"),
  };
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
    else if (flag == "--reload-ir") o.reload_ir = value();
    else if (flag == "--reload-manifest") o.reload_manifest = value();
    else if (flag == "--reload-coeff") o.reload_coeff = value();
    else if (flag == "--reload-msl") o.reload_msl = value();
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
    else if (flag == "--write-stop-at")
      o.write_stop_at = parse_u64(value(), "--write-stop-at");
    else if (flag == "--param-event")
      o.param_events.push_back(parse_param_event(value()));
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
  if (o.latency && !o.slot && o.param_events.empty())
    throw std::runtime_error("--latency requires --slot or --param-event");
  if (o.latency && !o.param_events.empty() && o.param_events.size() != 1)
    throw std::runtime_error("--latency accepts exactly one --param-event");
  if (o.write_every > 0 && !o.slot && o.param_events.empty())
    throw std::runtime_error(
      "--write-every requires --slot or --param-event");
  if (o.write_count == 0)
    throw std::runtime_error("--write-count must be positive");
  if (o.reference_every > 0 && !o.reference_slot)
    throw std::runtime_error("--reference-every requires --reference-slot");
  if (o.dac && o.duration_seconds <= 0.0)
    throw std::runtime_error("--dac requires --duration-seconds");
  if (o.dac && o.write_every > 0 && !o.write_stop_at)
    throw std::runtime_error(
      "--dac writes require --write-stop-at before the final checkpoint");
  const std::array<std::string, 4> required_disciplines = {
    "raw", "glide", "anchor", "velocity"
  };
  for (const auto & event : o.param_events)
    if (std::find(required_disciplines.begin(), required_disciplines.end(),
                  event.discipline) == required_disciplines.end())
      throw std::runtime_error(
        "--param-event discipline must be raw/glide/anchor/velocity");
  if (o.dac)
  {
    if (!o.reference_slot || o.write_every == 0 || !o.write_stop_at
        || !o.jump_at || !o.reload_at)
      throw std::runtime_error(
        "--dac qualification requires --reference-slot, --write-every, "
        "--write-stop-at, --jump-at, and --reload-at");
    if (o.param_events.size() != required_disciplines.size())
      throw std::runtime_error(
        "--dac qualification requires exactly one raw, glide, anchor, "
        "and velocity --param-event");
    for (const auto & required : required_disciplines)
      if (std::count_if(
            o.param_events.begin(), o.param_events.end(),
            [&](const ParamEvent & event) {
              return event.discipline == required;
            }) != 1)
        throw std::runtime_error(
          "--dac qualification requires exactly one " + required
          + " --param-event");
    const uint64_t expected_blocks = std::max<uint64_t>(
      1, static_cast<uint64_t>(o.duration_seconds * o.rate / o.buffer));
    const uint64_t baseline_blocks = std::max<uint64_t>(
      1, static_cast<uint64_t>(2.0 * o.rate / o.buffer));
    if (!(baseline_blocks < *o.jump_at
          && *o.jump_at < *o.reload_at
          && *o.reload_at < *o.write_stop_at
          && *o.write_stop_at < expected_blocks))
      throw std::runtime_error(
        "--dac event order must be baseline < jump < reload < write-stop "
        "< duration");
  }
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

uint64_t required_rss_bytes()
{
  const uint64_t value = rss_bytes();
  if (value == 0)
    throw std::runtime_error("RSS sampling failed");
  return value;
}

uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end)
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

uint64_t process_cpu_ns()
{
#if defined(__APPLE__) || defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  const auto timeval_ns = [](const timeval & value) {
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL
      + static_cast<uint64_t>(value.tv_usec) * 1000ULL;
  };
  return timeval_ns(usage.ru_utime) + timeval_ns(usage.ru_stime);
#else
  return 0;
#endif
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
    const std::string reload_ir =
      o.reload_ir.empty() ? ir : read_file(o.reload_ir);
    const std::string reload_manifest =
      o.reload_manifest.empty() ? manifest : read_file(o.reload_manifest);
    const std::string reload_coeff =
      o.reload_coeff.empty() ? coeff : read_file(o.reload_coeff);
    const std::string reload_msl =
      o.reload_msl.empty() ? msl : read_file(o.reload_msl);
    const bool reload_artifacts_distinct =
      reload_ir != ir
      && reload_manifest != manifest
      && reload_coeff != coeff
      && reload_msl != msl;
    if (o.dac && msl.empty())
      throw std::runtime_error(
        "--dac qualification requires a Metal artifact");
    if (o.dac && !reload_artifacts_distinct)
      throw std::runtime_error(
        "--dac qualification requires distinguishable replacement "
        "IR, manifest, coefficient IR, and MSL artifacts");

    RuntimeOwner rt_owner(tropical_runtime_new(o.buffer));
    tropical_runtime_t rt = rt_owner.get();
    if (!rt) throw std::runtime_error(tropical_last_error());
    const auto load_start = Clock::now();
    if (!load(rt, o, ir, manifest, coeff, msl))
      throw std::runtime_error(tropical_last_error());
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
    std::vector<std::string> reference_label;
    std::vector<uint64_t> reference_start_index;
    std::vector<double> reference_snr_db;
    std::vector<double> reference_max_error;
    std::vector<double> reference_signal_energy;
    std::vector<double> reference_checksum;
    std::vector<uint64_t> param_event_block;
    std::vector<std::string> param_event_discipline;
    std::vector<std::string> param_event_name;
    std::vector<double> param_event_value;
    std::vector<uint64_t> callback_histogram;
    uint64_t callback_histogram_epoch = 0;
    uint64_t measured_stats_epoch = 0;
    unsigned int negotiated_buffer_frames = 0;
    bool callback_progress_stalled = false;
    bool baseline_completed = false;
    bool writes_completed = false;
    bool writes_stopped = false;
    bool jump_completed = false;
    bool reload_completed = false;
    bool start_reference_completed = false;
    bool post_jump_reference_completed = false;
    bool midpoint_reference_completed = false;
    bool end_reference_completed = false;
    std::optional<uint64_t> baseline_block;
    std::optional<uint64_t> first_write_block;
    std::optional<uint64_t> last_write_block;
    std::optional<uint64_t> writes_stopped_block;
    std::optional<uint64_t> jump_request_block;
    std::optional<uint64_t> jump_progress_block;
    std::optional<uint64_t> reload_request_block;
    std::optional<uint64_t> reload_publication_block;
    std::optional<uint64_t> reload_progress_block;
    std::optional<uint64_t> start_reference_block;
    std::optional<uint64_t> post_jump_reference_block;
    std::optional<uint64_t> midpoint_reference_block;
    std::optional<uint64_t> end_reference_block;
    std::optional<uint64_t> duration_elapsed_block;
    std::optional<uint64_t> post_jump_preceding_write_block;
    std::optional<uint64_t> midpoint_preceding_write_block;
    std::optional<uint64_t> end_preceding_write_block;
    std::optional<uint64_t> latency_blocks;
    std::optional<uint64_t> reload_ns;
    uint64_t overrun_count = 0;
    uint64_t measured_loop_ns = 0;
    uint64_t measured_loop_callback_count = 0;
    uint64_t disconnect_count = 0;
    uint64_t reconnect_success_count = 0;
    uint64_t reconnect_failure_count = 0;
    uint64_t ownership_failure_count = 0;
    uint64_t reference_ownership_failure_count = 0;
    uint64_t nonfinite_count = 0;
    double checksum = 0.0;
    double write_value = o.slot_value;
    const uint64_t deadline_ns =
      static_cast<uint64_t>(static_cast<double>(o.buffer) * 1e9 / o.rate);

    if (o.latency)
    {
      const auto start = Clock::now();
      if (!o.param_events.empty())
      {
        const auto & event = o.param_events.front();
        write_value = event.high;
        const auto result =
          static_cast<tropical_runtime::FlatRuntime *>(rt)
            ->dispatch_param_sync(event.name, write_value);
        if (!result.ok)
          throw std::runtime_error(result.error);
        if (result.discipline != event.discipline)
          throw std::runtime_error(
            "latency param discipline mismatch for " + event.name
            + ": expected " + event.discipline
            + ", applied " + result.discipline);
        param_event_block.push_back(0);
        param_event_discipline.push_back(result.discipline);
        param_event_name.push_back(event.name);
        param_event_value.push_back(write_value);
      }
      else
        for (uint32_t i = 0; i < o.write_count; ++i)
          tropical_runtime_set_slot(rt, *o.slot + i, write_value);
      write_ns.push_back(elapsed_ns(start, Clock::now()));
    }

    const auto run_start = Clock::now();
    uint64_t cpu_start_ns = process_cpu_ns();
    uint64_t cpu_delta_ns = 0;
    uint64_t block = 0;
    uint64_t run_ns = 0;
    tropical_dac_stats_t dac_stats{};
    std::vector<std::string> dac_snapshot_labels;
    std::vector<tropical_dac_stats_t> dac_snapshots;
    std::vector<uint64_t> dac_snapshot_epochs;
    bool dac_aborted = false;
    std::string dac_abort_reason;
    std::string reference_failure_reason;
    if (o.dac)
    {
      if (!o.reference_slot)
        throw std::runtime_error(
          "--dac qualification requires --reference-slot");

      // A completely separate JIT runtime is the reference oracle. It is
      // never processed by the DAC callback and is updated only by this
      // control thread, avoiding races with the live Metal runtime.
      RuntimeOwner reference_owner(tropical_runtime_new(o.buffer));
      tropical_runtime_t reference_rt = reference_owner.get();
      if (!reference_rt) throw std::runtime_error(tropical_last_error());
      if (!load(reference_rt, o, ir, manifest, coeff, {}))
        throw std::runtime_error(tropical_last_error());

      DacOwner dac_owner(tropical_dac_new_runtime(
        rt, static_cast<unsigned int>(o.rate), 2));
      tropical_dac_t dac = dac_owner.get();
      if (!dac) throw std::runtime_error(tropical_last_error());
      tropical_dac_start(dac);
      if (!tropical_dac_is_running(dac))
      {
        const std::string error = tropical_last_error();
        throw std::runtime_error(
          error.empty() ? "DAC did not start on the default device" : error);
      }

      auto snapshot = [&](const std::string & label) {
        tropical_dac_stats_t stats{};
        tropical_dac_get_stats(dac, &stats);
        dac_snapshot_labels.push_back(label);
        dac_snapshots.push_back(stats);
        dac_snapshot_epochs.push_back(tropical_dac_get_stats_epoch(dac));
      };

      // Startup/device-prime status is useful evidence but is not part of the
      // measured steady-state window. Preserve it, then reset at one fixed
      // two-second boundary before any qualification event is scheduled.
      std::this_thread::sleep_for(std::chrono::seconds(2));
      snapshot("startup_warmup_before_reset");
      negotiated_buffer_frames = tropical_dac_get_buffer_frames(dac);
      if (negotiated_buffer_frames != o.buffer)
      {
        dac_aborted = true;
        dac_abort_reason = "negotiated_buffer_length_mismatch";
      }
      measured_stats_epoch = tropical_dac_reset_stats_epoch(dac);
      if (measured_stats_epoch == 0)
      {
        dac_aborted = true;
        dac_abort_reason = "stats_epoch_reset_timeout";
      }
      const auto measured_start = Clock::now();
      cpu_start_ns = process_cpu_ns();
      const uint64_t baseline_blocks =
        std::max<uint64_t>(1, static_cast<uint64_t>(2.0 * o.rate / o.buffer));
      uint64_t next_write = std::numeric_limits<uint64_t>::max();
      uint64_t next_rss = 0;
      uint64_t last_progress_block = 0;
      auto last_progress_at = Clock::now();

      auto capture_reference =
        [&](const std::string & label, std::optional<uint64_t> & event_block)
          -> bool {
        const uint64_t sequence = tropical_dac_request_output_capture(dac);
        if (sequence == 0)
        {
          reference_failure_reason = label + "_capture_not_idle";
          return false;
        }
        std::vector<double> actual(o.buffer, 0.0);
        uint64_t start_index = 0;
        const auto capture_deadline =
          Clock::now() + std::chrono::seconds(2);
        bool captured = false;
        while (Clock::now() < capture_deadline)
        {
          captured = tropical_dac_read_output_capture(
            dac, sequence, &start_index, actual.data(), actual.size());
          if (captured) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!captured)
        {
          reference_failure_reason = label + "_capture_timeout";
          return false;
        }

        std::vector<double> reference(o.buffer, 0.0);
        const uint32_t slot_id = *o.reference_slot;
        if (!tropical_runtime_render_window(
              reference_rt, start_index, o.buffer,
              &slot_id, 1, reference.data()))
          return false;

        double signal = 0.0;
        double error = 0.0;
        double max_error = 0.0;
        double reference_sum = 0.0;
        for (uint32_t i = 0; i < o.buffer; ++i)
        {
          signal += reference[i] * reference[i];
          reference_sum += reference[i];
          const double delta = reference[i] - actual[i];
          error += delta * delta;
          max_error = std::max(max_error, std::fabs(delta));
        }
        tropical_dac_stats_t stats{};
        tropical_dac_get_stats(dac, &stats);
        event_block = stats.callback_count;
        reference_label.push_back(label);
        reference_block.push_back(stats.callback_count);
        reference_start_index.push_back(start_index);
        reference_snr_db.push_back(
          error == 0.0 ? (signal > 0.0 ? 999.0 : 0.0)
                       : (signal > 0.0
                            ? 10.0 * std::log10(signal / error) : -999.0));
        reference_max_error.push_back(max_error);
        reference_signal_energy.push_back(signal);
        reference_checksum.push_back(reference_sum);
        if (signal <= 0.0)
          reference_failure_reason = label + "_reference_signal_zero";
        else if (!std::isfinite(reference_snr_db.back()))
          reference_failure_reason = label + "_reference_snr_nonfinite";
        else if (reference_snr_db.back() <= 100.0)
          reference_failure_reason = label + "_reference_snr_not_above_100db";
        return reference_failure_reason.empty();
      };

      uint64_t write_round = 0;
      auto apply_control_events = [&](uint64_t event_block) -> bool {
        const auto start = Clock::now();
        if (!o.param_events.empty())
        {
          // Keep the isolated JIT oracle's published "now" aligned with the
          // live runtime before invoking the exact shared host dispatch.
          const uint64_t live_now = static_cast<uint64_t>(
            tropical_runtime_current_sample_index(rt));
          tropical_runtime_set_sample_index(
            reference_rt, live_now >= o.buffer ? live_now - o.buffer : 0);
          tropical_runtime_process(reference_rt);

          for (const auto & event : o.param_events)
          {
            const double event_value =
              (write_round & 1U) == 0 ? event.high : event.low;
            auto live_result =
              static_cast<tropical_runtime::FlatRuntime *>(rt)
                ->dispatch_param_sync(event.name, event_value);
            auto reference_result =
              static_cast<tropical_runtime::FlatRuntime *>(reference_rt)
                ->dispatch_param_sync(event.name, event_value);
            if (!live_result.ok || !reference_result.ok)
            {
              dac_abort_reason = !live_result.ok
                ? "live_param_dispatch_failed:" + live_result.error
                : "reference_param_dispatch_failed:" + reference_result.error;
              return false;
            }
            if (live_result.discipline != event.discipline
                || reference_result.discipline != event.discipline)
            {
              dac_abort_reason =
                "param_discipline_mismatch:" + event.name
                + ":expected=" + event.discipline
                + ":live=" + live_result.discipline
                + ":reference=" + reference_result.discipline;
              return false;
            }
            param_event_block.push_back(event_block);
            param_event_discipline.push_back(live_result.discipline);
            param_event_name.push_back(event.name);
            param_event_value.push_back(event_value);
          }
          ++write_round;
        }
        else
        {
          write_value =
            write_value == o.slot_value ? o.slot_value * 0.5 : o.slot_value;
          for (uint32_t i = 0; i < o.write_count; ++i)
          {
            tropical_runtime_set_slot(rt, *o.slot + i, write_value);
            tropical_runtime_set_slot(
              reference_rt, *o.slot + i, write_value);
          }
        }
        write_ns.push_back(elapsed_ns(start, Clock::now()));
        return true;
      };

      bool jumped = false;
      bool jump_progress_seen = false;
      bool reloaded = false;
      bool reload_progress_seen = false;
      while (!dac_aborted
             && std::chrono::duration<double>(
                  Clock::now() - measured_start).count()
               < o.duration_seconds)
      {
        tropical_dac_get_stats(dac, &dac_stats);
        block = dac_stats.callback_count;
        disconnect_count = tropical_dac_disconnect_count(dac);
        reconnect_success_count =
          tropical_dac_reconnect_success_count(dac);
        reconnect_failure_count =
          tropical_dac_reconnect_failure_count(dac);
        ownership_failure_count =
          tropical_runtime_ownership_failure_count(rt);
        reference_ownership_failure_count =
          tropical_runtime_ownership_failure_count(reference_rt);
        if (tropical_dac_is_reconnecting(dac) || disconnect_count > 0
            || reconnect_success_count > 0 || reconnect_failure_count > 0)
        {
          dac_aborted = true;
          dac_abort_reason = "device_continuity_fault";
          snapshot("measured_failure_abort");
          break;
        }
        if (ownership_failure_count > 0
            || reference_ownership_failure_count > 0)
        {
          dac_aborted = true;
          dac_abort_reason = ownership_failure_count > 0
            ? "live_runtime_ownership_failure"
            : "reference_runtime_ownership_failure";
          snapshot("measured_failure_abort");
          break;
        }
        if (tropical_dac_get_stats_epoch(dac) != measured_stats_epoch)
        {
          dac_aborted = true;
          dac_abort_reason = "stats_epoch_changed";
          snapshot("measured_failure_abort");
          break;
        }
        if (dac_stats.underrun_count > 0 || dac_stats.overrun_count > 0)
        {
          dac_aborted = true;
          dac_abort_reason = dac_stats.underrun_count > 0
            ? "post_reset_underrun" : "post_reset_callback_overrun";
          snapshot("measured_failure_abort");
          break;
        }
        if (block != last_progress_block)
        {
          last_progress_block = block;
          last_progress_at = Clock::now();
        }
        else if (Clock::now() - last_progress_at > std::chrono::seconds(2))
        {
          callback_progress_stalled = true;
          dac_aborted = true;
          dac_abort_reason = "callback_progress_stalled";
          snapshot("measured_failure_abort");
          break;
        }

        if (!baseline_completed && block >= baseline_blocks)
        {
          snapshot("clean_baseline_after_reset");
          baseline_completed = true;
          baseline_block = block;
          start_reference_completed =
            capture_reference("start", start_reference_block);
          if (!start_reference_completed)
          {
            dac_aborted = true;
            dac_abort_reason = reference_failure_reason.empty()
              ? "start_reference_capture_failed" : reference_failure_reason;
            break;
          }
          next_write = o.write_every > 0 ? block + o.write_every
                                         : std::numeric_limits<uint64_t>::max();
        }
        if (!writes_stopped && o.write_stop_at
            && block >= *o.write_stop_at)
        {
          writes_stopped = true;
          writes_stopped_block = block;
          next_write = std::numeric_limits<uint64_t>::max();
        }
        if (baseline_completed && !writes_stopped
            && o.write_every > 0 && block >= next_write)
        {
          if (!apply_control_events(block))
          {
            dac_aborted = true;
            snapshot("measured_failure_abort");
            break;
          }
          if (!first_write_block) first_write_block = block;
          last_write_block = block;
          writes_completed = true;
          next_write += o.write_every;
        }
        if (!jumped && baseline_completed && writes_completed
            && last_write_block && block > *last_write_block
            && o.jump_at && block >= *o.jump_at)
        {
          snapshot("before_clock_jump");
          jump_request_block = block;
          tropical_runtime_set_sample_index(rt, o.jump_index);
          tropical_runtime_set_sample_index(reference_rt, o.jump_index);
          jumped = true;
        }
        if (jumped && !jump_progress_seen
            && block >= *jump_request_block + depth + 1)
        {
          jump_progress_seen = true;
          jump_completed = true;
          jump_progress_block = block;
          snapshot("after_clock_jump_progress");
          post_jump_preceding_write_block = last_write_block;
          post_jump_reference_completed =
            capture_reference("post_2p40_clock_jump",
                              post_jump_reference_block);
          if (!post_jump_reference_completed)
          {
            dac_aborted = true;
            dac_abort_reason = reference_failure_reason.empty()
              ? "post_jump_reference_capture_failed"
              : reference_failure_reason;
            break;
          }
        }
        if (!reloaded && jump_completed && last_write_block
            && block > *last_write_block && o.reload_at
            && block >= *o.reload_at)
        {
          snapshot("before_hot_swap");
          reload_request_block = block;
          const auto start = Clock::now();
          if (!load(rt, o, reload_ir, reload_manifest,
                    reload_coeff, reload_msl))
            throw std::runtime_error(tropical_last_error());
          reload_ns = elapsed_ns(start, Clock::now());
          tropical_dac_stats_t published{};
          tropical_dac_get_stats(dac, &published);
          reload_publication_block = published.callback_count;
          if (!load(reference_rt, o, reload_ir, reload_manifest,
                    reload_coeff, {}))
            throw std::runtime_error(tropical_last_error());
          reloaded = true;
        }
        if (reloaded && !reload_progress_seen
            && block >= *reload_publication_block + depth + 1)
        {
          reload_progress_seen = true;
          reload_completed = true;
          reload_progress_block = block;
          snapshot("after_hot_swap_progress");
          midpoint_preceding_write_block = last_write_block;
          midpoint_reference_completed =
            capture_reference("midpoint_after_hot_swap",
                              midpoint_reference_block);
          if (!midpoint_reference_completed)
          {
            dac_aborted = true;
            dac_abort_reason = reference_failure_reason.empty()
              ? "midpoint_reference_capture_failed"
              : reference_failure_reason;
            break;
          }
        }
        if (o.rss_every > 0 && block >= next_rss)
        {
          rss_block.push_back(block);
          rss_value.push_back(required_rss_bytes());
          next_rss = block + o.rss_every;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }

      // "end" is a true post-duration checkpoint, not a sample from the last
      // tenth of the timed loop. Keep the DAC live, record the exact callback
      // boundary at duration expiry, then capture the next completed block.
      // measured_loop_ns/count are taken only after this capture so both cover
      // precisely the same interval.
      tropical_dac_stats_t duration_elapsed_stats{};
      tropical_dac_get_stats(dac, &duration_elapsed_stats);
      duration_elapsed_block = duration_elapsed_stats.callback_count;
      block = duration_elapsed_stats.callback_count;
      if (!dac_aborted)
      {
        const bool future_queue_clear =
          writes_stopped_block && last_write_block
          && block >= *writes_stopped_block + depth + 1
          && block >= *last_write_block + depth + 1;
        if (!reload_completed || !writes_stopped || !future_queue_clear)
        {
          dac_aborted = true;
          dac_abort_reason = "final_reference_precondition_missing";
          snapshot("measured_failure_abort");
        }
        else
        {
          end_preceding_write_block = last_write_block;
          end_reference_completed =
            capture_reference("end", end_reference_block);
          if (!end_reference_completed)
          {
            dac_aborted = true;
            dac_abort_reason = reference_failure_reason.empty()
              ? "end_reference_capture_failed" : reference_failure_reason;
            snapshot("measured_failure_abort");
          }
          else
            snapshot("final_reference_after_duration");
        }
      }
      measured_loop_ns = elapsed_ns(measured_start, Clock::now());
      tropical_dac_stats_t loop_end_stats{};
      tropical_dac_get_stats(dac, &loop_end_stats);
      measured_loop_callback_count = loop_end_stats.callback_count;

      if (!dac_aborted
          && !(baseline_completed && writes_completed && jump_completed
               && writes_stopped && reload_completed && start_reference_completed
               && post_jump_reference_completed
               && midpoint_reference_completed && end_reference_completed))
      {
        dac_aborted = true;
        dac_abort_reason = "required_event_missing";
        snapshot("measured_failure_abort");
      }
      tropical_dac_get_stats(dac, &dac_stats);
      block = dac_stats.callback_count;
      overrun_count = dac_stats.overrun_count;
      disconnect_count = tropical_dac_disconnect_count(dac);
      reconnect_success_count = tropical_dac_reconnect_success_count(dac);
      reconnect_failure_count = tropical_dac_reconnect_failure_count(dac);
      ownership_failure_count =
        tropical_runtime_ownership_failure_count(rt);
      reference_ownership_failure_count =
        tropical_runtime_ownership_failure_count(reference_rt);

      callback_histogram.resize(
        tropical_dac_callback_histogram_bin_count(), 0);
      bool histogram_consistent = false;
      for (unsigned int attempt = 0; attempt < 100; ++attempt)
      {
        if (tropical_dac_copy_callback_histogram(
              dac, callback_histogram.data(), callback_histogram.size(),
              &callback_histogram_epoch) == callback_histogram.size())
        {
          tropical_dac_stats_t candidate{};
          tropical_dac_get_stats(dac, &candidate);
          uint64_t histogram_count = 0;
          for (const uint64_t count : callback_histogram)
            histogram_count += count;
          if (histogram_count == candidate.callback_count)
          {
            dac_stats = candidate;
            block = candidate.callback_count;
            overrun_count = candidate.overrun_count;
            histogram_consistent = true;
            break;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (!histogram_consistent
          || callback_histogram_epoch != measured_stats_epoch)
      {
        dac_aborted = true;
        if (dac_abort_reason.empty())
          dac_abort_reason = "callback_histogram_snapshot_failed";
      }

      run_ns = elapsed_ns(measured_start, Clock::now());
      const uint64_t cpu_end_ns = process_cpu_ns();
      cpu_delta_ns =
        cpu_end_ns >= cpu_start_ns ? cpu_end_ns - cpu_start_ns : 0;
      tropical_dac_stop(dac);
      snapshot("after_stop");
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
          rss_value.push_back(required_rss_bytes());
        }

        ++block;
        if (o.realtime)
        {
          next_deadline += std::chrono::nanoseconds(deadline_ns);
          std::this_thread::sleep_until(next_deadline);
        }
      }
      run_ns = elapsed_ns(run_start, Clock::now());
      const uint64_t cpu_end_ns = process_cpu_ns();
      cpu_delta_ns =
        cpu_end_ns >= cpu_start_ns ? cpu_end_ns - cpu_start_ns : 0;
    }
    ownership_failure_count =
      tropical_runtime_ownership_failure_count(rt);
    const uint64_t final_rss = required_rss_bytes();

    auto print_optional = [](const std::optional<uint64_t> & value) {
      if (value) std::cout << *value; else std::cout << "null";
    };

    std::cout << "{\"schema\":\"tropical_runtime_bench_1\""
              << ",\"buffer_length\":" << o.buffer
              << ",\"sample_rate\":" << std::setprecision(17) << o.rate
              << ",\"backend\":\"" << (msl.empty() ? "jit" : "metal") << "\""
              << ",\"pipeline_depth\":" << depth
              << ",\"load_ns\":" << load_ns
              << ",\"run_ns\":" << run_ns
              << ",\"measured_loop_ns\":" << measured_loop_ns
              << ",\"measured_loop_callback_count\":"
              << measured_loop_callback_count
              << ",\"process_cpu_ns\":" << cpu_delta_ns
              << ",\"process_cpu_seconds\":"
              << static_cast<double>(cpu_delta_ns) / 1e9
              << ",\"process_cpu_fraction\":"
              << (run_ns > 0
                    ? static_cast<double>(cpu_delta_ns)
                      / static_cast<double>(run_ns) : 0.0)
              << ",\"blocks\":" << block
              << ",\"dac_mode\":" << (o.dac ? "true" : "false")
              << ",\"dac_aborted\":" << (dac_aborted ? "true" : "false")
              << ",\"dac_abort_reason\":";
    if (dac_abort_reason.empty()) std::cout << "null";
    else std::cout << '"' << dac_abort_reason << '"';
    std::cout
              << ",\"deadline_ns\":" << deadline_ns
              << ",\"negotiated_buffer_frames\":"
              << negotiated_buffer_frames
              << ",\"measured_stats_epoch\":" << measured_stats_epoch
              << ",\"disconnect_count\":" << disconnect_count
              << ",\"reconnect_success_count\":" << reconnect_success_count
              << ",\"reconnect_failure_count\":" << reconnect_failure_count
              << ",\"ownership_failure_count\":"
              << ownership_failure_count
              << ",\"reference_ownership_failure_count\":"
              << reference_ownership_failure_count
              << ",\"reload_artifacts_distinct\":"
              << (reload_artifacts_distinct ? "true" : "false")
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
    std::cout << ",\"reference_labels\":[";
    for (std::size_t i = 0; i < reference_label.size(); ++i)
    {
      if (i) std::cout << ',';
      std::cout << '"' << reference_label[i] << '"';
    }
    std::cout << ']';
    std::cout << ",\"reference_start_indices\":";
    print_array(reference_start_index);
    std::cout << ",\"reference_snr_db\":";
    print_array(reference_snr_db);
    std::cout << ",\"reference_max_error\":";
    print_array(reference_max_error);
    std::cout << ",\"reference_signal_energy\":";
    print_array(reference_signal_energy);
    std::cout << ",\"reference_checksum\":";
    print_array(reference_checksum);
    std::cout << ",\"param_events\":[";
    for (std::size_t i = 0; i < param_event_block.size(); ++i)
    {
      if (i) std::cout << ',';
      std::cout << "{\"block\":" << param_event_block[i]
                << ",\"discipline\":\"" << param_event_discipline[i]
                << "\",\"name\":\"" << param_event_name[i]
                << "\",\"value\":" << param_event_value[i] << '}';
    }
    std::cout << ']';
    std::cout << ",\"dac_stats\":{\"callback_count\":"
              << dac_stats.callback_count
              << ",\"avg_callback_ms\":" << dac_stats.avg_callback_ms
              << ",\"max_callback_ms\":" << dac_stats.max_callback_ms
              << ",\"underrun_count\":" << dac_stats.underrun_count
              << ",\"overrun_count\":" << dac_stats.overrun_count << '}';
    std::cout << ",\"callback_histogram\":{"
              << "\"epoch\":" << callback_histogram_epoch
              << ",\"bin_width_ns\":"
              << tropical_dac_callback_histogram_bin_width_ns()
              << ",\"overflow_floor_ns\":"
              << tropical_dac_callback_histogram_overflow_floor_ns()
              << ",\"counts\":";
    print_array(callback_histogram);
    std::cout << '}';
    std::cout << ",\"event_completion\":{"
              << "\"baseline\":" << (baseline_completed ? "true" : "false")
              << ",\"writes\":" << (writes_completed ? "true" : "false")
              << ",\"writes_stopped\":"
              << (writes_stopped ? "true" : "false")
              << ",\"clock_jump\":" << (jump_completed ? "true" : "false")
              << ",\"hot_swap\":" << (reload_completed ? "true" : "false")
              << ",\"start_reference\":"
              << (start_reference_completed ? "true" : "false")
              << ",\"post_jump_reference\":"
              << (post_jump_reference_completed ? "true" : "false")
              << ",\"midpoint_reference\":"
              << (midpoint_reference_completed ? "true" : "false")
              << ",\"end_reference\":"
              << (end_reference_completed ? "true" : "false")
              << ",\"callback_progress_stalled\":"
              << (callback_progress_stalled ? "true" : "false")
              << '}';
    std::cout << ",\"event_blocks\":{";
    std::cout << "\"baseline\":"; print_optional(baseline_block);
    std::cout << ",\"first_write\":"; print_optional(first_write_block);
    std::cout << ",\"last_write\":"; print_optional(last_write_block);
    std::cout << ",\"writes_stopped\":";
    print_optional(writes_stopped_block);
    std::cout << ",\"clock_jump_request\":";
    print_optional(jump_request_block);
    std::cout << ",\"clock_jump_progress\":";
    print_optional(jump_progress_block);
    std::cout << ",\"hot_swap_request\":";
    print_optional(reload_request_block);
    std::cout << ",\"hot_swap_publication\":";
    print_optional(reload_publication_block);
    std::cout << ",\"hot_swap_progress\":";
    print_optional(reload_progress_block);
    std::cout << ",\"start_reference\":";
    print_optional(start_reference_block);
    std::cout << ",\"post_jump_reference\":";
    print_optional(post_jump_reference_block);
    std::cout << ",\"midpoint_reference\":";
    print_optional(midpoint_reference_block);
    std::cout << ",\"end_reference\":";
    print_optional(end_reference_block);
    std::cout << ",\"duration_elapsed\":";
    print_optional(duration_elapsed_block);
    std::cout << ",\"post_jump_preceding_write\":";
    print_optional(post_jump_preceding_write_block);
    std::cout << ",\"midpoint_preceding_write\":";
    print_optional(midpoint_preceding_write_block);
    std::cout << ",\"end_preceding_write\":";
    print_optional(end_preceding_write_block);
    std::cout << '}';
    std::cout << ",\"dac_snapshots\":[";
    for (std::size_t i = 0; i < dac_snapshots.size(); ++i)
    {
      if (i) std::cout << ',';
      const auto & stats = dac_snapshots[i];
      std::cout << "{\"label\":\"" << dac_snapshot_labels[i]
                << "\",\"callback_count\":" << stats.callback_count
                << ",\"epoch\":" << dac_snapshot_epochs[i]
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
    std::cerr << "tropical_runtime_bench[classified_failure=exception]: "
              << e.what() << '\n';
    return 1;
  }
}
