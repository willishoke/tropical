#pragma once

#include "metal/MetalKernel.hpp"
#include "runtime/EpochTileQueue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tropical_metal
{

// A candidate that cannot stay ahead of the callback must fail closed instead
// of occupying the compiler/control lane forever. The elapsed bound covers
// unusually slow device work; the attempt bound keeps deterministic failures
// prompt even when a fake/test renderer advances the device instantaneously.
inline constexpr uint32_t kActivationRetargetLimit = 8;
inline constexpr uint64_t kActivationRetargetTimeoutNs = 2000000000ULL;

enum class EpochTransitionKind : uint8_t
{
  Fresh,
  Continuous,
  ClockJump,
  HotSwap,
};

struct RenderEpochRequest
{
  uint64_t epoch_id = 0;
  MetalKernelPtr kernel;
  double sample_rate = 44100.0;
  std::vector<float> slots;
  std::vector<float> columns;
  uint64_t source_origin = 0;
  EpochTransitionKind transition = EpochTransitionKind::Continuous;
  bool fixed_activation = false;
  uint64_t activation_frame = 0;
  // Device coordinate captured by reserve(). Fixed live epochs use it to
  // learn how much of their original horizon elapsed before the candidate was
  // ready. UINT64_MAX distinguishes an unreserved direct worker request from
  // the valid initial device coordinate zero.
  uint64_t reservation_device_frame = UINT64_MAX;
};

struct EpochScheduleResult
{
  bool ok = false;
  std::string error;
  uint64_t epoch_id = 0;
  uint64_t activation_frame = 0;
  uint64_t effective_sample_index = 0;
  bool retargeted = false;
};

struct EpochReservation
{
  uint64_t device_frame = 0;
  uint64_t activation_frame = 0;
  uint64_t effective_sample_index = 0;
};

struct MetalWorkerStageTimes
{
  uint64_t request_received = 0;
  uint64_t activation_target_reserved = 0;
  uint64_t candidate_render_submitted = 0;
  uint64_t candidate_gpu_completion = 0;
  uint64_t candidate_window_ready = 0;
  uint64_t activation_published = 0;
  uint64_t activation_acknowledged = 0;
  uint64_t old_epoch_retired = 0;
};

struct MetalActivationLatencyStats
{
  uint64_t count = 0;
  uint64_t total_ns = 0;
  uint64_t min_ns = 0;
  uint64_t max_ns = 0;
};

struct MetalRenderWorkerTestSeam
{
  std::atomic<bool> pause_after_target_reserved{false};
  std::atomic<bool> target_reserved{false};
  std::atomic<bool> release_target{false};
};

class MetalRenderWorker
{
public:
  using RenderFunction = std::function<bool(
    const RenderEpochRequest &, uint64_t, uint32_t, double *)>;

  explicit MetalRenderWorker(
    tropical_runtime::EpochTileQueue & queue,
    RenderFunction render = {},
    std::atomic<bool> * realtime_running = nullptr);
  ~MetalRenderWorker();

  MetalRenderWorker(const MetalRenderWorker &) = delete;
  MetalRenderWorker & operator=(const MetalRenderWorker &) = delete;

  // Control-thread only. The caller may block; the callback never enters this
  // mailbox or touches the worker thread lifecycle.
  EpochScheduleResult schedule(RenderEpochRequest request);
  EpochReservation reserve(
    EpochTransitionKind transition,
    uint64_t requested_source = 0) const noexcept;

  uint64_t dispatch_failure_count() const noexcept;
  uint64_t activation_retarget_count() const noexcept;
  uint64_t activation_failure_count() const noexcept;
  uint64_t stale_completion_count() const noexcept;
  MetalWorkerStageTimes stage_times() const noexcept;
  MetalActivationLatencyStats activation_latency_stats() const noexcept;
  uint64_t worker_cpu_time_ns() const noexcept;
  uint64_t worker_wall_time_ns() const noexcept;
  void set_realtime_running(bool running) noexcept;
  void set_test_seam(MetalRenderWorkerTestSeam * seam) noexcept;

private:
  struct PendingRequest
  {
    explicit PendingRequest(RenderEpochRequest value)
      : request(std::move(value)) {}
    RenderEpochRequest request;
    EpochScheduleResult result;
    bool done = false;
    std::condition_variable completed;
  };

  struct BankRenderCursor
  {
    bool valid = false;
    RenderEpochRequest request;
    uint64_t next_device = 0;
    uint64_t next_source = 0;
    uint32_t next_slot = 0;
  };

  void ensure_started();
  void run();
  bool observe_activation_acknowledgement();
  bool refill_pending_bank();
  bool refill_active_bank();
  EpochScheduleResult prepare_activation(RenderEpochRequest request);
  bool render_one(
    uint32_t bank_index, BankRenderCursor & cursor,
    bool record_candidate_stage);
  bool render_tiles(
    uint32_t bank_index, BankRenderCursor & cursor, uint32_t tile_count);
  uint64_t activation_lead_frames(
    EpochTransitionKind transition) const noexcept;
  void learn_live_activation_lead(
    const RenderEpochRequest & request,
    uint64_t device_after_render,
    bool candidate_was_late) noexcept;
  static uint32_t prime_tile_count(
    EpochTransitionKind transition) noexcept;
  static uint64_t monotonic_time_ns();

  tropical_runtime::EpochTileQueue & queue_;
  RenderFunction render_;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<PendingRequest>> requests_;
  std::thread thread_;
  bool started_ = false;
  bool stopping_ = false;

  std::array<BankRenderCursor, tropical_runtime::EpochTileQueue::kBankCount>
    bank_cursors_;
  uint32_t active_bank_ = tropical_runtime::EpochTileQueue::kNoBank;
  uint64_t active_epoch_ = 0;
  uint64_t pending_activation_epoch_ = 0;

  std::atomic<uint64_t> dispatch_failures_{0};
  std::atomic<uint64_t> activation_retargets_{0};
  std::atomic<uint64_t> activation_failures_{0};
  std::atomic<uint64_t> stale_completions_{0};

  std::atomic<uint64_t> request_received_time_{0};
  std::atomic<uint64_t> target_reserved_time_{0};
  std::atomic<uint64_t> render_submitted_time_{0};
  std::atomic<uint64_t> gpu_completion_time_{0};
  std::atomic<uint64_t> window_ready_time_{0};
  std::atomic<uint64_t> activation_published_time_{0};
  std::atomic<uint64_t> activation_acknowledged_time_{0};
  std::atomic<uint64_t> old_epoch_retired_time_{0};
  std::atomic<uint64_t> activation_latency_count_{0};
  std::atomic<uint64_t> activation_latency_total_ns_{0};
  std::atomic<uint64_t> activation_latency_min_ns_{UINT64_MAX};
  std::atomic<uint64_t> activation_latency_max_ns_{0};
  std::atomic<uint64_t> worker_start_time_ns_{0};
  std::atomic<uint64_t> worker_cpu_baseline_ns_{0};
  std::atomic<uint64_t> worker_cpu_latest_ns_{0};
  // Continuous parameter writes and clock scrubs normally activate one tile
  // ahead. A real graph that misses that boundary teaches the next retry its
  // measured preparation horizon instead of repeating the same late target.
  std::atomic<uint64_t> live_activation_lead_frames_{0};
  std::atomic<bool> realtime_running_{false};
  std::atomic<bool> * realtime_running_source_ = &realtime_running_;
  std::atomic<MetalRenderWorkerTestSeam *> test_seam_{nullptr};
};

} // namespace tropical_metal
