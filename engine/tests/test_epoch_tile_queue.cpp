#include "runtime/EpochTileQueue.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>

using tropical_runtime::EpochActivation;
using tropical_runtime::EpochTileQueue;
using tropical_runtime::EpochTileStarvationSnapshot;
using tropical_runtime::EpochTileQueueTestSeam;
using tropical_runtime::TileConsumeStatus;
using tropical_runtime::TileState;
using tropical_runtime::TileTag;

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
  std::printf("  %-56s", name);
  std::fflush(stdout);
  const int failures_before = g_fail;
  fn();
  if (g_fail == failures_before)
  {
    std::printf("PASS\n");
    ++g_pass;
  }
}

static bool publish_ramp(EpochTileQueue & queue, uint32_t bank,
                         uint32_t slot, const TileTag & tag)
{
  EpochTileQueue::RenderClaim claim;
  if (!queue.claim_tile(bank, slot, tag, claim)) return false;
  for (uint32_t i = 0; i < tag.frame_count; ++i)
    claim.destination[i] = static_cast<double>(tag.source_start + i);
  return queue.publish_tile(claim);
}

static bool publish_constant(EpochTileQueue & queue, uint32_t bank,
                             uint32_t slot, const TileTag & tag,
                             double value)
{
  EpochTileQueue::RenderClaim claim;
  if (!queue.claim_tile(bank, slot, tag, claim)) return false;
  std::fill_n(claim.destination, tag.frame_count, value);
  return queue.publish_tile(claim);
}

static void prepare_epoch(EpochTileQueue & queue, uint32_t bank,
                          uint64_t epoch, uint64_t device_start,
                          uint64_t source_start, uint32_t count = 4)
{
  ASSERT(queue.begin_epoch(bank, epoch));
  for (uint32_t i = 0; i < count; ++i)
    ASSERT(publish_ramp(
      queue, bank, i,
      {epoch,
       device_start + static_cast<uint64_t>(i) * queue.render_frames(),
       source_start + static_cast<uint64_t>(i) * queue.render_frames(),
       queue.render_frames()}));
}

static void test_independent_quanta()
{
  for (uint32_t device : {128U, 256U, 512U})
  {
    EpochTileQueue queue(device, 512);
    prepare_epoch(queue, 0, 1, 0, 0);
    ASSERT(queue.publish_activation({1, 0, 0, 0}));
    std::array<double, 512> out{};
    for (uint32_t callback = 0; callback < 2048 / device; ++callback)
    {
      const auto result = queue.consume(out.data(), device);
      ASSERT(result.status == TileConsumeStatus::Audio);
      for (uint32_t i = 0; i < device; ++i)
        ASSERT(out[i] == static_cast<double>(callback * device + i));
    }
  }
}

static void test_interleaved_stereo_tiles()
{
  EpochTileQueue queue(2, 4, 2);
  ASSERT(queue.output_channels() == 2);

  std::array<double, 4> out{{9.0, 9.0, 9.0, 9.0}};
  ASSERT(queue.consume(out.data(), 2).status
         == TileConsumeStatus::InitialSilence);
  for (double sample : out) ASSERT(sample == 0.0);

  ASSERT(queue.begin_epoch(0, 1));
  EpochTileQueue::RenderClaim claim;
  ASSERT(queue.claim_tile(0, 0, {1, 2, 10, 4}, claim));
  for (uint32_t frame = 0; frame < 4; ++frame)
  {
    claim.destination[frame * 2] = static_cast<double>(10 + frame);
    claim.destination[frame * 2 + 1] = -static_cast<double>(10 + frame);
  }
  ASSERT(queue.publish_tile(claim));
  ASSERT(queue.publish_activation({1, 2, 10, 0}));

  ASSERT(queue.consume(out.data(), 2).status == TileConsumeStatus::Audio);
  const std::array<double, 4> first{{10.0, -10.0, 11.0, -11.0}};
  ASSERT(out == first);
  ASSERT(queue.consume(out.data(), 2).status == TileConsumeStatus::Audio);
  const std::array<double, 4> second{{12.0, -12.0, 13.0, -13.0}};
  ASSERT(out == second);
}

static void test_activation_and_jump()
{
  EpochTileQueue queue(128, 512);
  prepare_epoch(queue, 0, 1, 0, 0);
  ASSERT(queue.publish_activation({1, 0, 0, 0}));
  std::array<double, 128> out{};
  for (unsigned i = 0; i < 4; ++i)
    ASSERT(queue.consume(out.data(), 128).status == TileConsumeStatus::Audio);

  prepare_epoch(queue, 1, 2, 1024, 1ULL << 40);
  ASSERT(queue.publish_activation({2, 1024, 1ULL << 40, 1}));
  for (unsigned i = 0; i < 4; ++i)
    ASSERT(queue.consume(out.data(), 128).status == TileConsumeStatus::Audio);
  const auto activation = queue.consume(out.data(), 128);
  ASSERT(activation.status == TileConsumeStatus::Audio);
  ASSERT(activation.activated);
  ASSERT(out.front() == static_cast<double>(1ULL << 40));
  ASSERT(queue.activation_acknowledged() == 2);
}

static void test_unstable_activation_defers()
{
  EpochTileQueue queue(128, 512);
  prepare_epoch(queue, 0, 1, 128, 128);
  EpochTileQueueTestSeam seam;
  seam.pause_activation_publication.store(true, std::memory_order_release);
  queue.set_test_seam(&seam);

  std::thread publisher([&] {
    ASSERT(queue.publish_activation({1, 128, 128, 0}));
  });
  while (!seam.activation_publication_started.load(std::memory_order_acquire))
    std::this_thread::yield();

  std::array<double, 128> out{};
  const auto deferred = queue.consume(out.data(), 128);
  ASSERT(deferred.activation_deferred);
  ASSERT(deferred.status == TileConsumeStatus::InitialSilence);
  seam.release_activation_publication.store(true, std::memory_order_release);
  publisher.join();
  const auto activated = queue.consume(out.data(), 128);
  ASSERT(activated.activated);
  ASSERT(activated.status == TileConsumeStatus::Audio);
}

static void test_bad_tags_are_refused()
{
  {
    EpochTileQueue queue(128, 512);
    ASSERT(queue.begin_epoch(0, 1));
    EpochTileQueue::RenderClaim claim;
    ASSERT(!queue.claim_tile(0, 0, {9, 0, 0, 512}, claim));
  }
  const std::array<TileTag, 3> bad{{
    {1, 0, 64, 512},
    {1, 128, 0, 512},
    {1, 0, 0, 64},
  }};
  for (const TileTag & tag : bad)
  {
    EpochTileQueue queue(128, 512);
    ASSERT(queue.begin_epoch(0, 1));
    ASSERT(publish_ramp(queue, 0, 0, tag));
    ASSERT(queue.publish_activation({1, 0, 0, 0}));
    std::array<double, 128> out{};
    const auto result = queue.consume(out.data(), 128);
    ASSERT(result.status == TileConsumeStatus::FaultSilence);
    ASSERT(queue.tag_mismatch_count() == 1);
    ASSERT(queue.consume(out.data(), 128).status
           == TileConsumeStatus::FaultSilence);
    ASSERT(queue.tag_mismatch_count() == 1);
  }
}

static void test_starvation_latches_once()
{
  EpochTileQueue queue(128, 512);
  prepare_epoch(queue, 0, 1, 0, 0, 1);
  ASSERT(queue.publish_activation({1, 0, 0, 0}));
  std::array<double, 128> out{};
  for (unsigned i = 0; i < 4; ++i)
    ASSERT(queue.consume(out.data(), 128).status == TileConsumeStatus::Audio);
  ASSERT(queue.tile_state(0, 0) == TileState::Free);
  ASSERT(queue.consume(out.data(), 128).status
         == TileConsumeStatus::FaultSilence);
  ASSERT(queue.starvation_count() == 1);
  const EpochTileStarvationSnapshot snapshot =
    queue.first_starvation_snapshot();
  ASSERT(snapshot.valid);
  ASSERT(snapshot.epoch_id == 1);
  ASSERT(snapshot.device_frame == 512);
  ASSERT(snapshot.source_sample == 512);
  ASSERT(snapshot.expected_tile_device == 512);
  ASSERT(snapshot.expected_tile_source == 512);
  ASSERT(snapshot.last_published_epoch == 1);
  ASSERT(snapshot.last_published_device_end == 512);
  ASSERT(snapshot.last_published_source_end == 512);
  ASSERT(snapshot.active_bank == 0);
  ASSERT(snapshot.expected_tile_index == 1);
  ASSERT(snapshot.observed_tile_state
         == static_cast<uint32_t>(TileState::Free));
  ASSERT(snapshot.ready_mask == 0);
  ASSERT(snapshot.free_mask == 0xf);
  ASSERT(snapshot.rendering_mask == 0);
  ASSERT(snapshot.reading_mask == 0);
  ASSERT(queue.consume(out.data(), 128).status
         == TileConsumeStatus::FaultSilence);
  ASSERT(queue.starvation_count() == 1);
  ASSERT(queue.first_starvation_snapshot().device_frame == 512);
  for (double sample : out) ASSERT(sample == 0.0);
}

static void test_wrap_and_bank_reuse()
{
  EpochTileQueue queue(128, 512);
  prepare_epoch(queue, 0, 1, 0, 0);
  ASSERT(queue.publish_activation({1, 0, 0, 0}));
  std::array<double, 128> out{};
  for (unsigned i = 0; i < 4; ++i)
    ASSERT(queue.consume(out.data(), 128).status == TileConsumeStatus::Audio);
  ASSERT(queue.tile_state(0, 0) == TileState::Free);
  ASSERT(publish_ramp(queue, 0, 0, {1, 2048, 2048, 512}));
  for (unsigned callback = 4; callback < 20; ++callback)
  {
    ASSERT(queue.consume(out.data(), 128).status == TileConsumeStatus::Audio);
    for (unsigned i = 0; i < 128; ++i)
      ASSERT(out[i] == static_cast<double>(callback * 128 + i));
  }

  prepare_epoch(queue, 1, 2, 2560, 2560);
  ASSERT(queue.publish_activation({2, 2560, 2560, 1}));
  const auto switched = queue.consume(out.data(), 128);
  ASSERT(switched.activated);
  ASSERT(queue.audio_active_bank() == 1);
  ASSERT(queue.begin_epoch(0, 3));
}

static void test_ten_thousand_epoch_bank_reuses()
{
  constexpr uint32_t frames = 512;
  constexpr uint64_t epochs = 10000;
  EpochTileQueue queue(frames, frames);
  std::array<double, frames> out{};
  uint32_t bank = 0;
  for (uint64_t epoch = 1; epoch <= epochs; ++epoch)
  {
    ASSERT(queue.begin_epoch(bank, epoch));
    for (uint32_t tile = 0; tile < EpochTileQueue::kTilesPerBank; ++tile)
      ASSERT(publish_constant(
        queue, bank, tile,
        {epoch, epoch * frames + tile * frames,
         epoch * frames + tile * frames, frames},
        static_cast<double>(epoch)));
    queue.synchronize_audio_coordinates(epoch * frames, epoch * frames);
    ASSERT(queue.publish_activation(
      {epoch, epoch * frames, epoch * frames, bank}));
    const auto consumed = queue.consume(out.data(), frames);
    ASSERT(consumed.status == TileConsumeStatus::Audio);
    ASSERT(consumed.activated);
    ASSERT(consumed.epoch_id == epoch);
    ASSERT(queue.activation_acknowledged() == epoch);
    for (double sample : out)
      ASSERT(sample == static_cast<double>(epoch));
    bank = 1U - bank;
  }
  ASSERT(queue.starvation_count() == 0);
  ASSERT(queue.tag_mismatch_count() == 0);
}

int main()
{
  std::printf("test_epoch_tile_queue\n");
  run_test("Bdev 128/256/512 slices Rgpu 512", test_independent_quanta);
  run_test("stereo tiles slice on frames and copy interleaved samples",
           test_interleaved_stereo_tiles);
  run_test("continuous activation and exact clock jump", test_activation_and_jump);
  run_test("unstable activation read defers without spinning",
           test_unstable_activation_defers);
  run_test("stale/source/device/short tags are refused", test_bad_tags_are_refused);
  run_test("post-activation starvation is whole-buffer and sticky",
           test_starvation_latches_once);
  run_test("tile wrap and acknowledged bank reuse", test_wrap_and_bank_reuse);
  run_test("10,000 acknowledged epoch/bank reuse cycles",
           test_ten_thousand_epoch_bank_reuses);
  std::printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
