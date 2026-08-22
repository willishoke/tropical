#include "metal/MetalRenderWorker.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using tropical_metal::EpochTransitionKind;
using tropical_metal::MetalRenderWorker;
using tropical_metal::RenderEpochRequest;
using tropical_runtime::EpochTileQueue;
using tropical_runtime::TileConsumeStatus;
using tropical_runtime::TileState;

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond)                                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::printf("FAIL\n    assertion failed: %s (line %d)\n",              \
                  #cond, __LINE__);                                           \
      ++g_fail;                                                               \
      return;                                                                 \
    }                                                                         \
  } while (0)

static void run_test(const char * name, const std::function<void()> & fn)
{
  const int failures_before = g_fail;
  std::printf("  %-58s", name);
  std::fflush(stdout);
  fn();
  if (failures_before == g_fail)
  {
    ++g_pass;
    std::printf("PASS\n");
  }
}

static bool wait_for(
  const std::function<bool()> & predicate,
  std::chrono::milliseconds timeout = std::chrono::seconds(2))
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return predicate();
}

static RenderEpochRequest request(
  uint64_t epoch, EpochTransitionKind kind, uint64_t source = 0)
{
  RenderEpochRequest value;
  value.epoch_id = epoch;
  value.source_origin = source;
  value.transition = kind;
  value.slots = {static_cast<float>(epoch)};
  return value;
}

static void absolute_endpoint_materializer(
  const int64_t *, int64_t *, int64_t * const * arrays,
  const uint64_t *, int64_t *, double, uint64_t start_sample_index,
  const uint64_t *, double *, uint64_t, double * slots)
{
  const double bias = slots ? slots[0] : 0.0;
  const uint64_t interval = slots
    ? static_cast<uint64_t>(slots[1]) : 128;
  arrays[0][0] = std::bit_cast<int64_t>(
    static_cast<double>(start_sample_index) + bias);
  arrays[0][1] = std::bit_cast<int64_t>(
    static_cast<double>(start_sample_index + interval) + bias);
}

static void unsafe_endpoint_materializer(
  const int64_t *, int64_t *, int64_t * const * arrays,
  const uint64_t *, int64_t *, double, uint64_t,
  const uint64_t *, double *, uint64_t, double *)
{
  arrays[0][0] = std::bit_cast<int64_t>(
    std::numeric_limits<double>::quiet_NaN());
}

static void exact_fallback_kernel(
  const int64_t *, int64_t *, int64_t * const *,
  const uint64_t *, int64_t *, double, uint64_t start_sample_index,
  const uint64_t *, double * output, uint64_t frames, double * slots)
{
  const double bias = slots ? slots[0] : 0.0;
  for (uint64_t i = 0; i < frames; ++i)
    output[i] = static_cast<double>(start_sample_index + i) + bias;
}

static void exact_fallback_stereo_kernel(
  const int64_t *, int64_t *, int64_t * const *,
  const uint64_t *, int64_t *, double, uint64_t start_sample_index,
  const uint64_t *, double * output, uint64_t frames, double *)
{
  for (uint64_t frame = 0; frame < frames; ++frame)
  {
    output[frame * 2] = static_cast<double>(start_sample_index + frame);
    output[frame * 2 + 1] = -static_cast<double>(start_sample_index + frame);
  }
}

static void test_interleaved_stereo_rendering()
{
  EpochTileQueue queue(2, 4, 2);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest &, uint64_t source, uint32_t frames,
       double * destination) {
      for (uint32_t frame = 0; frame < frames; ++frame)
      {
        destination[frame * 2] = static_cast<double>(source + frame);
        destination[frame * 2 + 1] = -static_cast<double>(source + frame);
      }
      return true;
    });

  auto mismatched = request(1, EpochTransitionKind::Fresh);
  const auto refused = worker.schedule(std::move(mismatched));
  ASSERT(!refused.ok);
  ASSERT(refused.error.find("request/queue output channel mismatch")
         != std::string::npos);

  auto stereo = request(2, EpochTransitionKind::Fresh);
  stereo.output_channels = 2;
  ASSERT(worker.schedule(std::move(stereo)).ok);
  std::array<double, 4> output{};
  ASSERT(queue.consume(output.data(), 2).status == TileConsumeStatus::Audio);
  const std::array<double, 4> first{{0.0, -0.0, 1.0, -1.0}};
  ASSERT(output == first);
  ASSERT(queue.consume(output.data(), 2).status == TileConsumeStatus::Audio);
  const std::array<double, 4> second{{2.0, -2.0, 3.0, -3.0}};
  ASSERT(output == second);
}

static void test_absolute_time_materialization_and_clock_jump()
{
  EpochTileQueue queue(128, 128);
  std::atomic<bool> mismatch{false};
  std::mutex seen_mutex;
  std::vector<uint64_t> seen_sources;
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest & value, uint64_t source, uint32_t frames,
        double * destination) {
      if (value.columns.size() != 4
          || value.columns[0] != static_cast<float>(source + 7)
          || value.columns[1] != static_cast<float>(source + 64 + 7)
          || value.columns[2] != static_cast<float>(source + 64 + 7)
          || value.columns[3] != static_cast<float>(source + 128 + 7))
        mismatch.store(true, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(seen_mutex);
        seen_sources.push_back(source);
      }
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = static_cast<double>(source + i);
      return true;
    });

  auto program = std::make_shared<tropical_metal::TileMaterializerProgram>();
  program->kernel = absolute_endpoint_materializer;
  program->array_sizes = {2};
  program->tile_array_slots = {0};
  program->interval_frames = 64;
  auto initial = request(1, EpochTransitionKind::Fresh);
  initial.tile_materializer = program;
  initial.materializer_slots = {7.0, 64.0};
  initial.materializer_seed_arrays = {{0, 0}};
  ASSERT(worker.schedule(std::move(initial)).ok);
  std::weak_ptr<const tropical_metal::TileMaterializerProgram> retained =
    program;
  program.reset();
  ASSERT(!retained.expired());
  ASSERT(!mismatch.load(std::memory_order_acquire));
  {
    std::lock_guard<std::mutex> lock(seen_mutex);
    ASSERT(seen_sources.size() == EpochTileQueue::kTilesPerBank);
    for (std::size_t i = 0; i < seen_sources.size(); ++i)
      ASSERT(seen_sources[i] == i * 128);
  }

  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  auto jump_program =
    std::make_shared<tropical_metal::TileMaterializerProgram>();
  jump_program->kernel = absolute_endpoint_materializer;
  jump_program->array_sizes = {2};
  jump_program->tile_array_slots = {0};
  jump_program->interval_frames = 64;
  auto jump = request(2, EpochTransitionKind::ClockJump, 17);
  jump.tile_materializer = std::move(jump_program);
  jump.materializer_slots = {7.0, 64.0};
  jump.materializer_seed_arrays = {{0, 0}};
  const auto jumped = worker.schedule(std::move(jump));
  ASSERT(jumped.ok);
  ASSERT(jumped.effective_sample_index == 17);
  ASSERT(!mismatch.load(std::memory_order_acquire));
  {
    std::lock_guard<std::mutex> lock(seen_mutex);
    ASSERT(std::find(seen_sources.begin(), seen_sources.end(), 17)
           != seen_sources.end());
  }
  ASSERT(worker.materialization_failure_count() == 0);
  ASSERT(worker.exact_fallback_count() == 0);
}

static void test_unsafe_materialization_uses_exact_fallback()
{
  EpochTileQueue queue(128, 128);
  std::atomic<uint32_t> staged_dispatches{0};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest &, uint64_t, uint32_t frames,
        double * destination) {
      staged_dispatches.fetch_add(1, std::memory_order_relaxed);
      std::fill_n(destination, frames, -1.0);
      return true;
    });

  auto materializer =
    std::make_shared<tropical_metal::TileMaterializerProgram>();
  materializer->kernel = unsafe_endpoint_materializer;
  materializer->array_sizes = {1};
  materializer->tile_array_slots = {0};
  materializer->interval_frames = 128;
  auto exact =
    std::make_shared<tropical_metal::ExactTileFallbackProgram>();
  exact->kernel = exact_fallback_kernel;
  exact->array_storage = {{0}};
  exact->array_sizes = {1};
  exact->initial_slots = {0.0};

  auto staged = request(1, EpochTransitionKind::Fresh);
  staged.tile_materializer = std::move(materializer);
  staged.materializer_slots = {0.0};
  staged.materializer_seed_arrays = {{0}};
  staged.exact_fallback = std::move(exact);
  staged.exact_fallback_slots = {3.0};
  ASSERT(worker.schedule(std::move(staged)).ok);
  ASSERT(staged_dispatches.load(std::memory_order_acquire) == 0);
  ASSERT(worker.materialization_failure_count()
         == EpochTileQueue::kTilesPerBank);
  ASSERT(worker.exact_fallback_count() == EpochTileQueue::kTilesPerBank);

  std::array<double, 128> output{};
  const auto consumed = queue.consume(output.data(), output.size());
  ASSERT(consumed.status == TileConsumeStatus::Audio);
  for (uint32_t i = 0; i < output.size(); ++i)
    ASSERT(output[i] == static_cast<double>(i) + 3.0);
}

static void test_stereo_exact_fallback_uses_full_sample_width()
{
  EpochTileQueue queue(2, 4, 2);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest &, uint64_t, uint32_t, double *) {
      return false;
    });

  auto materializer =
    std::make_shared<tropical_metal::TileMaterializerProgram>();
  materializer->kernel = unsafe_endpoint_materializer;
  materializer->array_sizes = {1};
  materializer->tile_array_slots = {0};
  materializer->interval_frames = 4;
  auto exact =
    std::make_shared<tropical_metal::ExactTileFallbackProgram>();
  exact->kernel = exact_fallback_stereo_kernel;
  exact->array_storage = {{0}};
  exact->array_sizes = {1};

  auto staged = request(1, EpochTransitionKind::Fresh);
  staged.output_channels = 2;
  staged.tile_materializer = std::move(materializer);
  staged.materializer_slots = {0.0};
  staged.materializer_seed_arrays = {{0}};
  staged.exact_fallback = std::move(exact);
  ASSERT(worker.schedule(std::move(staged)).ok);

  std::array<double, 4> output{};
  ASSERT(queue.consume(output.data(), 2).status == TileConsumeStatus::Audio);
  const std::array<double, 4> expected{{0.0, -0.0, 1.0, -1.0}};
  ASSERT(output == expected);
}

static void test_worker_activation_and_refill()
{
  EpochTileQueue queue(128, 512);
  const std::thread::id caller = std::this_thread::get_id();
  std::atomic<bool> wrong_thread{false};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest &, uint64_t source, uint32_t frames,
        double * destination) {
      if (std::this_thread::get_id() == caller)
        wrong_thread.store(true, std::memory_order_release);
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = static_cast<double>(source + i);
      return true;
    });

  const auto initial =
    worker.schedule(request(1, EpochTransitionKind::Fresh, 0));
  ASSERT(initial.ok);
  ASSERT(initial.activation_frame == 0);
  ASSERT(initial.effective_sample_index == 0);

  std::array<double, 128> output{};
  for (uint32_t callback = 0; callback < 12; ++callback)
  {
    const auto consumed = queue.consume(output.data(), 128);
    ASSERT(consumed.status == TileConsumeStatus::Audio);
    for (uint32_t i = 0; i < output.size(); ++i)
      ASSERT(output[i] == static_cast<double>(callback * 128 + i));
  }
  ASSERT(wait_for([&] {
    return worker.stage_times().activation_acknowledged != 0;
  }));
  ASSERT(!wrong_thread.load(std::memory_order_acquire));

  const auto next =
    worker.schedule(request(2, EpochTransitionKind::Continuous));
  ASSERT(next.ok);
  ASSERT(next.activation_frame > queue.published_device_frame());
  while (queue.published_device_frame() < next.activation_frame)
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  const auto switched = queue.consume(output.data(), 128);
  ASSERT(switched.status == TileConsumeStatus::Audio);
  ASSERT(switched.activated);
  ASSERT(switched.source_start == next.effective_sample_index);
  ASSERT(wait_for([&] {
    return worker.stage_times().old_epoch_retired != 0;
  }));
  const auto candidate_stages = worker.stage_times();
  ASSERT(candidate_stages.request_received
         <= candidate_stages.activation_target_reserved);
  ASSERT(candidate_stages.activation_target_reserved
         <= candidate_stages.candidate_render_submitted);
  ASSERT(candidate_stages.candidate_render_submitted
         <= candidate_stages.candidate_gpu_completion);
  ASSERT(candidate_stages.candidate_gpu_completion
         <= candidate_stages.candidate_window_ready);
  ASSERT(candidate_stages.candidate_window_ready
         <= candidate_stages.activation_published);
  ASSERT(candidate_stages.activation_published
         <= candidate_stages.activation_acknowledged);
  ASSERT(candidate_stages.activation_acknowledged
         <= candidate_stages.old_epoch_retired);

  // Releasing active-bank tiles causes steady-state refills. Those renders
  // must not overwrite the retained candidate-window timeline.
  for (uint32_t callback = 0; callback < 8; ++callback)
  {
    ASSERT(wait_for([&] { return queue.next_callback_ready_for_offline(); }));
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  }
  ASSERT(wait_for([&] {
    return queue.tile_state(queue.audio_active_bank(), 0)
           == TileState::Ready;
  }));
  const auto after_refill = worker.stage_times();
  ASSERT(after_refill.candidate_render_submitted
         == candidate_stages.candidate_render_submitted);
  ASSERT(after_refill.candidate_gpu_completion
         == candidate_stages.candidate_gpu_completion);
}

static void test_candidate_within_bank_horizon_does_not_retarget()
{
  EpochTileQueue queue(128, 512);
  std::atomic<bool> candidate_started{false};
  std::atomic<bool> release_candidate{false};
  std::atomic<bool> blocked_once{false};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest & request, uint64_t source, uint32_t frames,
        double * destination) {
      if (request.epoch_id == 2
          && !blocked_once.exchange(true, std::memory_order_acq_rel))
      {
        candidate_started.store(true, std::memory_order_release);
        while (!release_candidate.load(std::memory_order_acquire))
          std::this_thread::yield();
      }
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = static_cast<double>(source + i);
      return true;
    });
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), 128).status
         == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  tropical_metal::EpochScheduleResult result;
  std::thread control([&] {
    result = worker.schedule(
      request(2, EpochTransitionKind::HotSwap));
  });
  ASSERT(wait_for([&] {
    return candidate_started.load(std::memory_order_acquire);
  }));
  for (unsigned i = 0; i < 4; ++i)
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  release_candidate.store(true, std::memory_order_release);
  control.join();
  ASSERT(result.ok);
  ASSERT(worker.activation_retarget_count() == 0);
  ASSERT(result.effective_sample_index == result.activation_frame);

  while (queue.published_device_frame() < result.activation_frame)
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  const auto activated = queue.consume(output.data(), 128);
  ASSERT(activated.activated);
  ASSERT(activated.source_start == result.effective_sample_index);
}

static void test_perpetually_late_candidate_fails_bounded()
{
  EpochTileQueue queue(128, 512);
  std::array<double, 128> callback_output{};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest & request, uint64_t source, uint32_t frames,
        double * destination) {
      std::fill_n(destination, frames, static_cast<double>(source));
      if (request.epoch_id == 2)
      {
        // Advance one complete bank while every candidate bank renders. This
        // deterministic adversary used to make prepare_activation loop
        // forever; it must now return a diagnostic after a bounded number of
        // discarded windows.
        for (uint32_t callback = 0;
             callback < queue.capacity_frames() / callback_output.size();
             ++callback)
          (void)queue.consume(
            callback_output.data(), callback_output.size());
      }
      return true;
    });
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  ASSERT(queue.consume(
    callback_output.data(), callback_output.size()).status
    == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto started = std::chrono::steady_clock::now();
  const auto failed =
    worker.schedule(request(2, EpochTransitionKind::Continuous));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT(!failed.ok);
  ASSERT(failed.error.find("candidate remained late") != std::string::npos);
  ASSERT(worker.activation_retarget_count()
         == tropical_metal::kActivationRetargetLimit);
  ASSERT(worker.activation_failure_count() == 1);
  ASSERT(elapsed < std::chrono::seconds(2));
}

static void test_failed_candidate_keeps_active_epoch()
{
  EpochTileQueue queue(128, 512);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t source, uint32_t frames,
       double * destination) {
      if (request.epoch_id == 2) return false;
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = static_cast<double>(source + i);
      return true;
    });
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), 128).status
         == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto failed =
    worker.schedule(request(2, EpochTransitionKind::Continuous));
  ASSERT(!failed.ok);
  ASSERT(worker.activation_failure_count() == 1);
  ASSERT(queue.consume(output.data(), 128).status
         == TileConsumeStatus::Audio);
  ASSERT(queue.activation_acknowledged() == 1);
}

static void test_teardown_waits_for_inflight_tile()
{
  EpochTileQueue queue(128, 512);
  std::atomic<uint32_t> renders{0};
  std::atomic<bool> refill_started{false};
  std::atomic<bool> release_refill{false};
  auto worker = std::make_unique<MetalRenderWorker>(
    queue,
    [&](const RenderEpochRequest &, uint64_t source, uint32_t frames,
        double * destination) {
      const uint32_t number =
        renders.fetch_add(1, std::memory_order_acq_rel) + 1;
      if (number == 5)
      {
        refill_started.store(true, std::memory_order_release);
        while (!release_refill.load(std::memory_order_acquire))
          std::this_thread::yield();
      }
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = static_cast<double>(source + i);
      return true;
    });
  ASSERT(worker->schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  for (unsigned i = 0; i < 4; ++i)
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] {
    return refill_started.load(std::memory_order_acquire);
  }));

  std::atomic<bool> destroyed{false};
  std::thread control([&] {
    worker.reset();
    destroyed.store(true, std::memory_order_release);
  });
  std::this_thread::yield();
  ASSERT(!destroyed.load(std::memory_order_acquire));
  release_refill.store(true, std::memory_order_release);
  control.join();
  ASSERT(destroyed.load(std::memory_order_acquire));
}

static void test_running_controls_serialize_acknowledgements()
{
  EpochTileQueue queue(512, 512);
  std::atomic<bool> realtime_running{true};
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      std::fill_n(
        destination, frames, static_cast<double>(request.epoch_id));
      return true;
    }, &realtime_running);
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 512> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto second = worker.schedule(
    request(2, EpochTransitionKind::Continuous));
  ASSERT(second.ok);
  ASSERT(queue.published_activation_epoch() == 2);

  tropical_metal::EpochScheduleResult third;
  std::atomic<bool> third_returned{false};
  std::jthread control([&] {
    third = worker.schedule(request(3, EpochTransitionKind::Continuous));
    third_returned.store(true, std::memory_order_release);
  });
  // If running-audio cancellation regresses, epoch 2 is replaced and this
  // call returns immediately. It must instead remain pending until callback
  // acknowledgement makes epoch 2's projected glide state real.
  ASSERT(!wait_for(
    [&] { return third_returned.load(std::memory_order_acquire); },
    std::chrono::milliseconds(20)));
  ASSERT(queue.published_activation_epoch() == 2);

  while (queue.published_device_frame() < second.activation_frame)
    ASSERT(queue.consume(output.data(), output.size()).status
           == TileConsumeStatus::Audio);
  const auto activated_two = queue.consume(output.data(), output.size());
  ASSERT(activated_two.activated);
  ASSERT(activated_two.epoch_id == 2);
  ASSERT(wait_for(
    [&] { return third_returned.load(std::memory_order_acquire); }));
  control.join();
  ASSERT(third.ok);
  ASSERT(third.activation_frame > second.activation_frame);
  while (queue.published_device_frame() < third.activation_frame)
    ASSERT(queue.consume(output.data(), output.size()).status
           == TileConsumeStatus::Audio);
  const auto activated_three = queue.consume(output.data(), output.size());
  ASSERT(activated_three.activated);
  ASSERT(activated_three.epoch_id == 3);
  ASSERT(worker.activation_failure_count() == 0);
  ASSERT(worker.stale_completion_count() == 0);
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
}

static void test_live_control_primes_one_tile_before_publication()
{
  EpochTileQueue queue(128, 512);
  std::atomic<uint32_t> epoch_two_renders{0};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest & request, uint64_t, uint32_t frames,
        double * destination) {
      if (request.epoch_id == 2)
        epoch_two_renders.fetch_add(1, std::memory_order_relaxed);
      std::fill_n(destination, frames, static_cast<double>(request.epoch_id));
      return true;
    });
  ASSERT(worker.schedule(request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const uint64_t before = queue.published_device_frame();
  const auto live = worker.schedule(
    request(2, EpochTransitionKind::Continuous));
  ASSERT(live.ok);
  ASSERT(live.activation_frame == before + queue.render_frames());
  // schedule() returns as soon as one live-control tile is ready. Depending
  // on scheduling, background refill may already have progressed, but the
  // candidate publication itself records exactly one submitted prime tile.
  ASSERT(worker.stage_times().candidate_window_ready != 0);
  ASSERT(epoch_two_renders.load(std::memory_order_acquire) >= 1);
  ASSERT(wait_for([&] {
    return epoch_two_renders.load(std::memory_order_acquire)
           == EpochTileQueue::kTilesPerBank;
  }));
}

static void test_fixed_live_control_learns_late_render_horizon()
{
  EpochTileQueue queue(128, 512);
  std::array<double, 128> output{};
  std::atomic<bool> callback_failed{false};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest & request, uint64_t, uint32_t frames,
        double * destination) {
      std::fill_n(destination, frames, static_cast<double>(request.epoch_id));
      if (request.epoch_id == 2)
      {
        // A real modal graph can spend a complete render tile between the
        // control reservation and candidate readiness. Repeating the same
        // one-tile reservation can therefore never catch the device clock.
        for (uint32_t callback = 0; callback < 4; ++callback)
          if (queue.consume(output.data(), output.size()).status
              != TileConsumeStatus::Audio)
            callback_failed.store(true, std::memory_order_release);
      }
      return true;
    });
  ASSERT(worker.schedule(request(1, EpochTransitionKind::Fresh, 0)).ok);
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  tropical_metal::EpochScheduleResult scheduled;
  uint32_t attempts = 0;
  for (; attempts < tropical_metal::kActivationRetargetLimit; ++attempts)
  {
    const auto reservation = worker.reserve(EpochTransitionKind::Continuous);
    auto candidate = request(
      2, EpochTransitionKind::Continuous,
      reservation.effective_sample_index);
    candidate.fixed_activation = true;
    candidate.activation_frame = reservation.activation_frame;
    candidate.reservation_device_frame = reservation.device_frame;
    candidate.enqueue_device_frame = reservation.device_frame;
    scheduled = worker.schedule(std::move(candidate));
    if (!scheduled.retargeted) break;
  }

  ASSERT(!callback_failed.load(std::memory_order_acquire));
  ASSERT(scheduled.ok);
  ASSERT(attempts < 3);
  ASSERT(worker.activation_retarget_count() == 1);
}

static void test_queued_control_wait_does_not_inflate_lead()
{
  EpochTileQueue queue(128, 512);
  std::atomic<bool> realtime_running{true};
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      std::fill_n(destination, frames, static_cast<double>(request.epoch_id));
      return true;
    }, &realtime_running);
  ASSERT(worker.schedule(request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto second = worker.schedule(
    request(2, EpochTransitionKind::Continuous));
  ASSERT(second.ok);
  const auto reservation = worker.reserve(EpochTransitionKind::Continuous);
  auto candidate = request(
    3, EpochTransitionKind::Continuous,
    reservation.effective_sample_index);
  candidate.fixed_activation = true;
  candidate.activation_frame = reservation.activation_frame;
  candidate.reservation_device_frame = reservation.device_frame;
  candidate.enqueue_device_frame = reservation.device_frame;

  tropical_metal::EpochScheduleResult third;
  std::atomic<bool> returned{false};
  std::jthread control([&] {
    third = worker.schedule(std::move(candidate));
    returned.store(true, std::memory_order_release);
  });
  ASSERT(!wait_for(
    [&] { return returned.load(std::memory_order_acquire); },
    std::chrono::milliseconds(20)));
  while (queue.published_device_frame() < second.activation_frame)
    ASSERT(queue.consume(output.data(), output.size()).status
           == TileConsumeStatus::Audio);
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return returned.load(std::memory_order_acquire); }));
  control.join();
  ASSERT(third.retargeted);

  const uint64_t current = queue.published_device_frame();
  const auto next = worker.reserve(EpochTransitionKind::Continuous);
  ASSERT(next.activation_frame == current + queue.render_frames());
}

static void test_running_control_waits_for_audible_acknowledgement()
{
  EpochTileQueue queue(128, 512);
  std::atomic<bool> realtime_running{true};
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      std::fill_n(destination, frames, static_cast<double>(request.epoch_id));
      return true;
    }, &realtime_running);
  ASSERT(worker.schedule(request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto control = worker.schedule(
    request(2, EpochTransitionKind::Continuous));
  ASSERT(control.ok);
  std::atomic<bool> returned{false};
  bool acknowledged = false;
  std::jthread waiter([&] {
    acknowledged = worker.wait_for_activation_acknowledgement(2);
    returned.store(true, std::memory_order_release);
  });
  ASSERT(!wait_for(
    [&] { return returned.load(std::memory_order_acquire); },
    std::chrono::milliseconds(20)));
  while (queue.published_device_frame() < control.activation_frame)
    ASSERT(queue.consume(output.data(), output.size()).status
           == TileConsumeStatus::Audio);
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return returned.load(std::memory_order_acquire); }));
  waiter.join();
  ASSERT(acknowledged);
}

static void test_unclaimed_epochs_coalesce_without_audio_callback()
{
  EpochTileQueue queue(128, 512);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      std::fill_n(
        destination, frames, static_cast<double>(request.epoch_id));
      return true;
    });

  // There is deliberately no consume() between these publications. Before
  // coalescing applied to every unclaimed descriptor, the third schedule
  // waited forever for a callback that cannot exist while the DAC is stopped.
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  ASSERT(worker.schedule(
    request(2, EpochTransitionKind::Continuous)).ok);
  const auto started = std::chrono::steady_clock::now();
  const auto latest = worker.schedule(
    request(3, EpochTransitionKind::Continuous));
  ASSERT(latest.ok);
  ASSERT(std::chrono::steady_clock::now() - started
         < std::chrono::seconds(1));
  ASSERT(queue.published_activation_epoch() == 3);

  std::array<double, 128> output{};
  const auto activated = queue.consume(output.data(), output.size());
  ASSERT(activated.status == TileConsumeStatus::Audio);
  ASSERT(activated.activated);
  ASSERT(activated.epoch_id == 3);
  for (double sample : output) ASSERT(sample == 3.0);
}

static void test_unclaimed_epochs_coalesce_after_audio_stops()
{
  EpochTileQueue queue(128, 512);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      std::fill_n(
        destination, frames, static_cast<double>(request.epoch_id));
      return true;
    });
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  // Model a stopped DAC by ceasing callbacks after an active epoch exists.
  ASSERT(worker.schedule(
    request(2, EpochTransitionKind::Continuous)).ok);
  const auto latest = worker.schedule(
    request(3, EpochTransitionKind::Continuous));
  ASSERT(latest.ok);
  ASSERT(queue.published_activation_epoch() == 3);

  while (queue.published_device_frame() < latest.activation_frame)
  {
    ASSERT(wait_for([&] { return queue.next_callback_ready_for_offline(); }));
    ASSERT(queue.consume(output.data(), output.size()).status
           == TileConsumeStatus::Audio);
  }
  ASSERT(wait_for([&] { return queue.next_callback_ready_for_offline(); }));
  const auto activated = queue.consume(output.data(), output.size());
  ASSERT(activated.status == TileConsumeStatus::Audio);
  ASSERT(activated.activated);
  ASSERT(activated.epoch_id == 3);
  for (double sample : output) ASSERT(sample == 3.0);
}

int main()
{
  std::printf("test_metal_render_worker (no DAC)\n");
  run_test("stereo render requests preserve interleaved tile width",
           test_interleaved_stereo_rendering);
  run_test("absolute tile images rematerialize across a backward clock jump",
           test_absolute_time_materialization_and_clock_jump);
  run_test("unsafe tile images fall back to the exact CPU kernel",
           test_unsafe_materialization_uses_exact_fallback);
  run_test("stereo exact fallback validates the full interleaved tile",
           test_stereo_exact_fallback_uses_full_sample_width);
  run_test("lazy worker activation, refill, and continuous handoff",
           test_worker_activation_and_refill);
  run_test("candidate inside full-bank horizon avoids a retarget",
           test_candidate_within_bank_horizon_does_not_retarget);
  run_test("perpetually late candidate fails after bounded retargets",
           test_perpetually_late_candidate_fails_bounded);
  run_test("failed candidate is not published over active epoch",
           test_failed_candidate_keeps_active_epoch);
  run_test("control-thread teardown joins an in-flight tile safely",
           test_teardown_waits_for_inflight_tile);
  run_test("running controls serialize through acknowledgements",
           test_running_controls_serialize_acknowledgements);
  run_test("live controls publish with a one-tile activation lead",
           test_live_control_primes_one_tile_before_publication);
  run_test("fixed controls learn enough lead after one late candidate",
           test_fixed_live_control_learns_late_render_horizon);
  run_test("queued wait does not inflate activation lead",
           test_queued_control_wait_does_not_inflate_lead);
  run_test("running control returns at audible acknowledgement",
           test_running_control_waits_for_audible_acknowledgement);
  run_test("unclaimed controls coalesce before audio starts",
           test_unclaimed_epochs_coalesce_without_audio_callback);
  run_test("unclaimed controls coalesce after audio stops",
           test_unclaimed_epochs_coalesce_after_audio_stops);
  std::printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
