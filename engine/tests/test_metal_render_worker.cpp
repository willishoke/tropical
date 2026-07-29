#include "metal/MetalRenderWorker.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <thread>

using tropical_metal::EpochTransitionKind;
using tropical_metal::MetalRenderWorker;
using tropical_metal::RenderEpochRequest;
using tropical_runtime::EpochTileQueue;
using tropical_runtime::TileConsumeStatus;

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
  for (unsigned i = 0; i < 4; ++i)
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
      worker.reserve(EpochTransitionKind::Continuous).activation_frame;
    std::jthread control([&, index, epoch] {
      results[index] = worker.schedule(
        request(epoch, EpochTransitionKind::Continuous));
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
  run_test("lazy worker activation, refill, and continuous handoff",
           test_worker_activation_and_refill);
  run_test("late candidate retargets while old coverage stays audible",
           test_candidate_late_retargets_off_audio);
  run_test("failed candidate is not published over active epoch",
           test_failed_candidate_keeps_active_epoch);
  run_test("control-thread teardown joins an in-flight tile safely",
           test_teardown_waits_for_inflight_tile);
  run_test("rapid A/B/A promises serialize through acknowledgements",
           test_rapid_a_b_a_serializes_acknowledgements);
  std::printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
