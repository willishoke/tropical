#include "metal/MetalRenderWorker.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <thread>

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

static void test_boot_zero_target_retargets_after_callback_entry()
{
  EpochTileQueue queue(128, 512);
  std::atomic<bool> render_started{false};
  std::atomic<bool> release_render{false};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest &, uint64_t, uint32_t frames,
        double * destination) {
      render_started.store(true, std::memory_order_release);
      while (!release_render.load(std::memory_order_acquire))
        std::this_thread::yield();
      std::fill_n(destination, frames, 1.0);
      return true;
    });
  auto candidate = request(1, EpochTransitionKind::Continuous, 0);
  candidate.fixed_activation = true;
  candidate.activation_frame = 0;
  tropical_metal::EpochScheduleResult result;
  std::thread control([&] { result = worker.schedule(candidate); });
  ASSERT(wait_for([&] {
    return render_started.load(std::memory_order_acquire);
  }));
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).status
         == TileConsumeStatus::InitialSilence);
  release_render.store(true, std::memory_order_release);
  control.join();
  ASSERT(!result.ok);
  ASSERT(result.retargeted);
  ASSERT(worker.activation_retarget_count() == 1);
  ASSERT(queue.published_activation_epoch() == 0);
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
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
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
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

static void test_candidate_late_retargets_off_audio()
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
      request(2, EpochTransitionKind::Continuous));
  });
  ASSERT(wait_for([&] {
    return candidate_started.load(std::memory_order_acquire);
  }));
  for (unsigned i = 0; i < 9; ++i)
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  release_candidate.store(true, std::memory_order_release);
  control.join();
  ASSERT(result.ok);
  ASSERT(worker.activation_retarget_count() >= 1);
  ASSERT(result.effective_sample_index == result.activation_frame);

  while (queue.published_device_frame() < result.activation_frame)
    ASSERT(queue.consume(output.data(), 128).status
           == TileConsumeStatus::Audio);
  const auto activated = queue.consume(output.data(), 128);
  ASSERT(activated.activated);
  ASSERT(activated.source_start == result.effective_sample_index);
}

static void test_publication_at_exact_target_retargets()
{
  EpochTileQueue queue(128, 512);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = static_cast<double>(request.epoch_id);
      return true;
    });
  ASSERT(worker.schedule(
    request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).status
         == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto reservation =
    worker.reserve(EpochTransitionKind::Continuous);
  auto candidate = request(2, EpochTransitionKind::Continuous);
  candidate.fixed_activation = true;
  candidate.activation_frame = reservation.activation_frame;
  candidate.source_origin = reservation.effective_sample_index;

  tropical_runtime::EpochTileQueueTestSeam seam;
  seam.pause_activation_publication.store(true, std::memory_order_release);
  queue.set_test_seam(&seam);
  tropical_metal::EpochScheduleResult result;
  std::thread control([&] { result = worker.schedule(candidate); });
  ASSERT(wait_for([&] {
    return seam.activation_publication_started.load(
      std::memory_order_acquire);
  }));

  // Model the callback publishing boundary E after its descriptor read. The
  // worker's descriptor then appears at E but remains unclaimed: it cannot be
  // observed until the E+Bdev callback and must be retargeted, not stranded.
  queue.synchronize_audio_coordinates(
    reservation.activation_frame, reservation.effective_sample_index);
  seam.release_activation_publication.store(true, std::memory_order_release);
  control.join();
  queue.set_test_seam(nullptr);

  ASSERT(!result.ok);
  ASSERT(result.retargeted);
  ASSERT(worker.activation_retarget_count() == 1);
  ASSERT(queue.published_activation_epoch() == 0);
  ASSERT(queue.activation_acknowledged() == 1);
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
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
  ASSERT(worker.morph_failure_count() == 1);
  ASSERT(worker.morph_count() == 0);
  ASSERT(queue.consume(output.data(), 128).status
         == TileConsumeStatus::Audio);
  ASSERT(queue.activation_acknowledged() == 1);
}

static void test_continuous_publish_requires_two_tiles_not_full_bank()
{
  EpochTileQueue queue(128, 512);
  std::atomic<uint32_t> epoch_two_renders{0};
  std::atomic<bool> third_render_started{false};
  std::atomic<bool> release_third_render{false};
  MetalRenderWorker worker(
    queue,
    [&](const RenderEpochRequest & request, uint64_t, uint32_t frames,
        double * destination) {
      if (request.epoch_id == 2)
      {
        const uint32_t count =
          epoch_two_renders.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (count == 3)
        {
          third_render_started.store(true, std::memory_order_release);
          while (!release_third_render.load(std::memory_order_acquire))
            std::this_thread::yield();
        }
      }
      std::fill_n(destination, frames, static_cast<double>(request.epoch_id));
      return true;
    });
  ASSERT(worker.schedule(request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).status
         == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  std::atomic<bool> schedule_returned{false};
  tropical_metal::EpochScheduleResult result;
  std::thread control([&] {
    result = worker.schedule(request(2, EpochTransitionKind::Continuous));
    schedule_returned.store(true, std::memory_order_release);
  });
  ASSERT(wait_for([&] {
    return third_render_started.load(std::memory_order_acquire);
  }));
  ASSERT(schedule_returned.load(std::memory_order_acquire));
  ASSERT(queue.published_activation_epoch() == 2);
  ASSERT(epoch_two_renders.load(std::memory_order_acquire) == 3);
  release_third_render.store(true, std::memory_order_release);
  control.join();
  ASSERT(result.ok);
}

static void test_admission_waits_for_prior_acknowledgement()
{
  EpochTileQueue queue(128, 512);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t, uint32_t frames,
       double * destination) {
      std::fill_n(destination, frames, static_cast<double>(request.epoch_id));
      return true;
    });
  ASSERT(worker.schedule(request(1, EpochTransitionKind::Fresh, 0)).ok);
  std::array<double, 128> output{};
  ASSERT(queue.consume(output.data(), output.size()).status
         == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  const auto second =
    worker.schedule(request(2, EpochTransitionKind::Continuous));
  ASSERT(second.ok);
  std::atomic<bool> admitted{false};
  std::jthread waiter([&] {
    admitted.store(worker.wait_for_admission(), std::memory_order_release);
  });
  std::this_thread::yield();
  ASSERT(!admitted.load(std::memory_order_acquire));
  while (queue.published_device_frame() < second.activation_frame)
    ASSERT(queue.consume(output.data(), output.size()).status
           == TileConsumeStatus::Audio);
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] {
    return admitted.load(std::memory_order_acquire);
  }));
  waiter.join();
}

static void test_worker_morph_matches_whole_window_oracle()
{
  EpochTileQueue queue(128, 512);
  MetalRenderWorker worker(
    queue,
    [](const RenderEpochRequest & request, uint64_t source, uint32_t frames,
       double * destination) {
      const double offset = request.slots.empty() ? 0.0 : request.slots[0];
      for (uint32_t i = 0; i < frames; ++i)
        destination[i] = offset
          + 0.25 * std::sin(static_cast<double>(source + i) * 0.013);
      return true;
    });
  auto old_request = request(1, EpochTransitionKind::Fresh, 0);
  old_request.slots = {1.0F};
  ASSERT(worker.schedule(std::move(old_request)).ok);
  std::array<double, 128> block{};
  ASSERT(queue.consume(block.data(), block.size()).status
         == TileConsumeStatus::Audio);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  auto verify_transition = [&](uint64_t epoch, double old_offset,
                               double new_offset) {
    auto next_request = request(epoch, EpochTransitionKind::Continuous);
    next_request.slots = {static_cast<float>(new_offset)};
    const auto scheduled = worker.schedule(std::move(next_request));
    if (!scheduled.ok) return false;
    while (queue.published_device_frame() < scheduled.activation_frame)
      if (queue.consume(block.data(), block.size()).status
          != TileConsumeStatus::Audio)
        return false;

    std::array<double, 512> transition{};
    for (uint32_t callback = 0; callback < 4; ++callback)
    {
      const auto consumed = queue.consume(block.data(), block.size());
      if (consumed.status != TileConsumeStatus::Audio) return false;
      if (callback == 0 && !consumed.activated) return false;
      std::copy_n(
        block.begin(), block.size(),
        transition.begin() + callback * block.size());
    }
    for (uint32_t k = 0; k < transition.size(); ++k)
    {
      const double wave = 0.25 * std::sin(
        static_cast<double>(scheduled.effective_sample_index + k) * 0.013);
      const double old_value = old_offset + wave;
      const double new_value = new_offset + wave;
      const double x = static_cast<double>(k)
        / static_cast<double>(transition.size() - 1);
      const double weight = x * x * (3.0 - 2.0 * x);
      const double expected =
        (1.0 - weight) * old_value + weight * new_value;
      if (std::abs(transition[k] - expected) > 1e-12) return false;
      if (std::abs(transition[k])
          > std::max(std::abs(old_value), std::abs(new_value)) + 1e-12)
        return false;
    }

    std::array<double, 512> pure_new{};
    for (uint32_t callback = 0; callback < 4; ++callback)
    {
      if (queue.consume(block.data(), block.size()).status
          != TileConsumeStatus::Audio)
        return false;
      std::copy_n(
        block.begin(), block.size(),
        pure_new.begin() + callback * block.size());
    }
    for (uint32_t k = 0; k < pure_new.size(); ++k)
    {
      const double expected = new_offset + 0.25 * std::sin(
        static_cast<double>(scheduled.effective_sample_index
          + transition.size() + k) * 0.013);
      if (std::abs(pure_new[k] - expected) > 1e-12) return false;
    }
    return wait_for([&] {
      return queue.activation_acknowledged() == epoch;
    });
  };

  ASSERT(verify_transition(2, 1.0, 2.0));
  ASSERT(verify_transition(3, 2.0, 1.0));
  ASSERT(worker.morph_count() == 2);
  ASSERT(worker.morph_failure_count() == 0);
  ASSERT(worker.morph_render_time_ns() > 0);
  ASSERT(worker.activation_retarget_count() == 0);
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
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

static void test_rapid_a_b_a_serializes_acknowledgements()
{
  EpochTileQueue queue(512, 512);
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
  std::array<double, 512> output{};
  ASSERT(queue.consume(output.data(), output.size()).activated);
  ASSERT(wait_for([&] { return queue.activation_acknowledged() == 1; }));

  std::array<tropical_metal::EpochScheduleResult, 3> results{};
  for (uint32_t index = 0; index < results.size(); ++index)
  {
    const uint64_t epoch = index + 2;
    const uint64_t expected_activation =
      worker.reserve(EpochTransitionKind::HotSwap).activation_frame;
    std::jthread control([&, index, epoch] {
      results[index] = worker.schedule(
        request(epoch, EpochTransitionKind::HotSwap));
    });
    ASSERT(wait_for([&] {
      return queue.published_activation_epoch() == epoch;
    }));
    while (queue.published_device_frame()
           < expected_activation)
      ASSERT(queue.consume(output.data(), output.size()).status
             == TileConsumeStatus::Audio);
    const auto activated = queue.consume(output.data(), output.size());
    ASSERT(activated.status == TileConsumeStatus::Audio);
    ASSERT(activated.activated);
    ASSERT(activated.epoch_id == epoch);
    for (double sample : output)
      ASSERT(sample == static_cast<double>(epoch));
    control.join();
    ASSERT(results[index].ok);
    ASSERT(results[index].activation_frame == expected_activation);
    ASSERT(queue.activation_acknowledged() == epoch);
  }
  ASSERT(results[0].activation_frame < results[1].activation_frame);
  ASSERT(results[1].activation_frame < results[2].activation_frame);
  ASSERT(worker.activation_failure_count() == 0);
  ASSERT(worker.stale_completion_count() == 0);
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
}

int main()
{
  std::printf("test_metal_render_worker (no DAC)\n");
  run_test("boot frame zero retargets after callback entry",
           test_boot_zero_target_retargets_after_callback_entry);
  run_test("lazy worker activation, refill, and continuous handoff",
           test_worker_activation_and_refill);
  run_test("late candidate retargets while old coverage stays audible",
           test_candidate_late_retargets_off_audio);
  run_test("publication at exact target retargets before expiry",
           test_publication_at_exact_target_retargets);
  run_test("failed candidate is not published over active epoch",
           test_failed_candidate_keeps_active_epoch);
  run_test("continuous publish requires two tiles, not a full bank",
           test_continuous_publish_requires_two_tiles_not_full_bank);
  run_test("admission waits for the prior audible acknowledgement",
           test_admission_waits_for_prior_acknowledgement);
  run_test("whole-window worker morph matches the offline oracle",
           test_worker_morph_matches_whole_window_oracle);
  run_test("control-thread teardown joins an in-flight tile safely",
           test_teardown_waits_for_inflight_tile);
  run_test("rapid A/B/A promises serialize through acknowledgements",
           test_rapid_a_b_a_serializes_acknowledgements);
  std::printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
