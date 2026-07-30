#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tropical_runtime
{

struct TileTag
{
  uint64_t epoch_id = 0;
  uint64_t device_start = 0;
  uint64_t source_start = 0;
  uint32_t frame_count = 0;
};

enum class TileState : uint8_t
{
  Free,
  Rendering,
  Ready,
  Reading,
};

enum class TileConsumeStatus : uint8_t
{
  Audio,
  InitialSilence,
  FaultSilence,
};

struct EpochActivation
{
  uint64_t epoch_id = 0;
  uint64_t activation_frame = 0;
  uint64_t source_start = 0;
  uint32_t bank_index = 0;
};

struct TileConsumeResult
{
  TileConsumeStatus status = TileConsumeStatus::InitialSilence;
  uint64_t epoch_id = 0;
  uint64_t device_start = 0;
  uint64_t source_start = 0;
  bool activated = false;
  bool activation_deferred = false;
};

struct EpochTileStarvationSnapshot
{
  bool valid = false;
  uint64_t epoch_id = 0;
  uint64_t device_frame = 0;
  uint64_t source_sample = 0;
  uint64_t expected_tile_device = 0;
  uint64_t expected_tile_source = 0;
  uint64_t last_published_epoch = 0;
  uint64_t last_published_device_end = 0;
  uint64_t last_published_source_end = 0;
  uint32_t active_bank = 0;
  uint32_t expected_tile_index = 0;
  uint32_t observed_tile_state = 0;
  uint32_t ready_mask = 0;
  uint32_t free_mask = 0;
  uint32_t rendering_mask = 0;
  uint32_t reading_mask = 0;
};

// Test-only control-side pause. It makes the activation publication sequence
// observably odd so a callback can prove that it defers after one read rather
// than spinning.
struct EpochTileQueueTestSeam
{
  std::atomic<bool> pause_activation_publication{false};
  std::atomic<bool> activation_publication_started{false};
  std::atomic<bool> release_activation_publication{false};
};

/**
 * A fixed SPSC handoff between one render worker and one audio callback.
 *
 * The worker owns Free -> Rendering -> Ready. The callback owns
 * Ready -> Reading -> Free. Tile storage is allocated by the constructor and
 * remains at a stable address for the queue's lifetime.
 */
class EpochTileQueue
{
public:
  static constexpr uint32_t kBankCount = 2;
  static constexpr uint32_t kTilesPerBank = 4;
  static constexpr uint32_t kNoBank = UINT32_MAX;
  static constexpr uint32_t kNoTile = UINT32_MAX;

  struct RenderClaim
  {
    uint32_t bank_index = kNoBank;
    uint32_t tile_index = kNoTile;
    double * destination = nullptr;
  };

  EpochTileQueue(uint32_t device_frames, uint32_t render_frames)
    : device_frames_(device_frames),
      render_frames_(render_frames)
  {
    if (device_frames == 0 || render_frames < device_frames
        || render_frames % device_frames != 0)
      throw std::invalid_argument(
        "EpochTileQueue: render frames must be a positive multiple of device frames");
    for (auto & bank : banks_)
      for (auto & tile : bank.tiles)
        tile.samples.assign(render_frames_, 0.0);
  }

  EpochTileQueue(const EpochTileQueue &) = delete;
  EpochTileQueue & operator=(const EpochTileQueue &) = delete;

  uint32_t device_frames() const noexcept { return device_frames_; }
  uint32_t render_frames() const noexcept { return render_frames_; }
  uint32_t capacity_frames() const noexcept
  {
    return kTilesPerBank * render_frames_;
  }

  // Audio callback only. Used when an explicit JIT epoch hands the device
  // boundary back to Metal; the two coordinates remain independent.
  void synchronize_audio_coordinates(
    uint64_t device_frame, uint64_t source_sample) noexcept
  {
    audio_device_frame_ = device_frame;
    audio_source_sample_ = source_sample;
    publish_audio_boundary();
  }

  // Worker only. A bank may be staged only when no callback owns one of its
  // tiles and the callback no longer names it as active.
  bool begin_epoch(uint32_t bank_index, uint64_t epoch_id)
  {
    if (bank_index >= kBankCount || epoch_id == 0
        || audio_active_bank_.load(std::memory_order_acquire) == bank_index)
      return false;
    Bank & bank = banks_[bank_index];
    for (auto & tile : bank.tiles)
    {
      const TileState state = tile.state.load(std::memory_order_acquire);
      if (state == TileState::Reading || state == TileState::Rendering)
        return false;
    }
    for (auto & tile : bank.tiles)
      tile.state.store(TileState::Free, std::memory_order_release);
    bank.epoch_id = epoch_id;
    return true;
  }

  // Worker only. Sequential epochs start in slot zero and wrap in coordinate
  // order; this makes callback acquisition a single bounded state transition.
  bool claim_tile(uint32_t bank_index, uint32_t tile_index,
                  const TileTag & tag, RenderClaim & claim)
  {
    claim = {};
    if (bank_index >= kBankCount || tile_index >= kTilesPerBank
        || tag.epoch_id == 0 || tag.frame_count == 0
        || tag.frame_count > render_frames_
        || banks_[bank_index].epoch_id != tag.epoch_id)
      return false;
    Tile & tile = banks_[bank_index].tiles[tile_index];
    TileState expected = TileState::Free;
    if (!tile.state.compare_exchange_strong(
          expected, TileState::Rendering,
          std::memory_order_acquire, std::memory_order_relaxed))
      return false;
    tile.tag = tag;
    claim = {bank_index, tile_index, tile.samples.data()};
    return true;
  }

  // Worker only. The tag and samples become immutable at this release.
  bool publish_tile(const RenderClaim & claim)
  {
    Tile * tile = claimed_tile(claim);
    if (!tile) return false;
    TileState expected = TileState::Rendering;
    if (!tile->state.compare_exchange_strong(
      expected, TileState::Ready,
      std::memory_order_release, std::memory_order_relaxed))
      return false;
    last_published_epoch_.store(
      tile->tag.epoch_id, std::memory_order_relaxed);
    last_published_device_end_.store(
      tile->tag.device_start + tile->tag.frame_count,
      std::memory_order_relaxed);
    last_published_source_end_.store(
      tile->tag.source_start + tile->tag.frame_count,
      std::memory_order_release);
    return true;
  }

  // Worker only. Used for obsolete or failed render completions.
  bool discard_tile(const RenderClaim & claim)
  {
    Tile * tile = claimed_tile(claim);
    if (!tile) return false;
    TileState expected = TileState::Rendering;
    return tile->state.compare_exchange_strong(
      expected, TileState::Free,
      std::memory_order_release, std::memory_order_relaxed);
  }

  // Worker only. At most one descriptor may remain unacknowledged. The
  // descriptor fields are atomic so even an intentionally interrupted
  // publication contains no C++ data race.
  bool publish_activation(const EpochActivation & activation)
  {
    if (activation.epoch_id == 0 || activation.bank_index >= kBankCount
        || banks_[activation.bank_index].epoch_id != activation.epoch_id)
      return false;
    const uint64_t published =
      activation_epoch_.load(std::memory_order_acquire);
    if (published != 0
        && activation_acknowledged_.load(std::memory_order_acquire) != published)
      return false;

    activation_sequence_.fetch_add(1, std::memory_order_acq_rel);
    if (EpochTileQueueTestSeam * seam =
          test_seam_.load(std::memory_order_acquire);
        seam
        && seam->pause_activation_publication.load(std::memory_order_acquire))
    {
      seam->activation_publication_started.store(
        true, std::memory_order_release);
      while (!seam->release_activation_publication.load(
        std::memory_order_acquire))
        std::this_thread::yield();
    }
    activation_frame_.store(
      activation.activation_frame, std::memory_order_relaxed);
    activation_source_.store(
      activation.source_start, std::memory_order_relaxed);
    activation_bank_.store(
      activation.bank_index, std::memory_order_relaxed);
    activation_claim_.store(0, std::memory_order_relaxed);
    activation_epoch_.store(
      activation.epoch_id, std::memory_order_relaxed);
    activation_sequence_.fetch_add(1, std::memory_order_release);
    return true;
  }

  // Worker only. Coalesces a fresh prepared epoch before any callback has
  // claimed it. The claim CAS prevents overwriting a descriptor while audio
  // may be acting on a stable read.
  bool cancel_unclaimed_activation(uint64_t epoch_id)
  {
    if (epoch_id == 0
        || activation_epoch_.load(std::memory_order_acquire) != epoch_id)
      return false;
    uint64_t expected = 0;
    if (!activation_claim_.compare_exchange_strong(
          expected, UINT64_MAX,
          std::memory_order_acq_rel, std::memory_order_relaxed))
      return false;
    activation_sequence_.fetch_add(1, std::memory_order_acq_rel);
    activation_epoch_.store(0, std::memory_order_relaxed);
    activation_sequence_.fetch_add(1, std::memory_order_release);
    return true;
  }

  /**
   * Audio callback only. Makes one activation read and one tile acquisition
   * attempt, copies exactly one device quantum, then advances both coordinates.
   * Every failure writes a whole quantum of silence.
   */
  TileConsumeResult consume(double * destination, uint32_t frames)
  {
    TileConsumeResult result;
    result.device_start = audio_device_frame_;
    result.source_start = audio_source_sample_;
    result.epoch_id = audio_epoch_id_;

    if (!destination || frames != device_frames_)
    {
      if (destination)
        std::fill_n(destination, frames, 0.0);
      advance_coordinates(frames);
      result.status = TileConsumeStatus::FaultSilence;
      return result;
    }

    EpochActivation activation;
    bool has_activation = false;
    bool unstable_activation = false;
    read_activation_once(activation, has_activation, unstable_activation);
    if (unstable_activation)
      result.activation_deferred = true;
    else if (has_activation && activation.epoch_id != audio_epoch_id_
             && activation.activation_frame <= audio_device_frame_)
    {
      uint64_t unclaimed = 0;
      if (!activation_claim_.compare_exchange_strong(
            unclaimed, activation.epoch_id,
            std::memory_order_acquire, std::memory_order_relaxed))
      {
        result.activation_deferred = true;
        has_activation = false;
      }
    }
    if (has_activation && activation.epoch_id != audio_epoch_id_
        && activation.activation_frame <= audio_device_frame_)
    {
      release_reading_tile();
      audio_epoch_id_ = activation.epoch_id;
      audio_active_bank_local_ = activation.bank_index;
      audio_active_bank_.store(
        activation.bank_index, std::memory_order_release);
      audio_source_sample_ = activation.source_start;
      expected_tile_device_ = audio_device_frame_;
      expected_tile_source_ = activation.source_start;
      expected_tile_index_ = 0;
      active_faulted_ = false;
      result.activated = true;
      result.epoch_id = audio_epoch_id_;
      result.source_start = audio_source_sample_;
      activation_acknowledged_.store(
        activation.epoch_id, std::memory_order_release);
    }

    if (audio_epoch_id_ == 0)
    {
      std::fill_n(destination, frames, 0.0);
      advance_coordinates(frames);
      result.status = TileConsumeStatus::InitialSilence;
      publish_audio_boundary();
      return result;
    }

    if (active_faulted_)
    {
      std::fill_n(destination, frames, 0.0);
      advance_coordinates(frames);
      result.status = TileConsumeStatus::FaultSilence;
      publish_audio_boundary();
      return result;
    }

    Tile * tile = nullptr;
    if (reading_tile_index_ != kNoTile)
      tile = &banks_[audio_active_bank_local_].tiles[reading_tile_index_];
    else
    {
      tile = &banks_[audio_active_bank_local_].tiles[expected_tile_index_];
      TileState expected = TileState::Ready;
      if (!tile->state.compare_exchange_strong(
            expected, TileState::Reading,
            std::memory_order_acquire, std::memory_order_relaxed))
      {
        latch_starvation(expected);
        std::fill_n(destination, frames, 0.0);
        advance_coordinates(frames);
        result.status = TileConsumeStatus::FaultSilence;
        publish_audio_boundary();
        return result;
      }
      reading_tile_index_ = expected_tile_index_;
      reading_offset_ = 0;
    }

    const TileTag & tag = tile->tag;
    const bool tag_matches =
      tag.epoch_id == audio_epoch_id_
      && tag.device_start == expected_tile_device_
      && tag.source_start == expected_tile_source_
      && tag.frame_count > 0
      && tag.frame_count <= render_frames_
      && reading_offset_ <= tag.frame_count
      && frames <= tag.frame_count - reading_offset_
      && tag.device_start + reading_offset_ == audio_device_frame_
      && tag.source_start + reading_offset_ == audio_source_sample_;
    if (!tag_matches)
    {
      latch_tag_mismatch();
      std::fill_n(destination, frames, 0.0);
      advance_coordinates(frames);
      result.status = TileConsumeStatus::FaultSilence;
      publish_audio_boundary();
      return result;
    }

    std::copy_n(
      tile->samples.data() + reading_offset_, frames, destination);
    reading_offset_ += frames;
    advance_coordinates(frames);
    if (reading_offset_ == tag.frame_count)
    {
      expected_tile_device_ += tag.frame_count;
      expected_tile_source_ += tag.frame_count;
      expected_tile_index_ =
        (expected_tile_index_ + 1) % kTilesPerBank;
      release_reading_tile();
    }
    result.status = TileConsumeStatus::Audio;
    publish_audio_boundary();
    return result;
  }

  /**
   * Offline-render thread only. Reports whether the next call to consume()
   * can acquire its exact tile without faulting. This is deliberately not a
   * callback wait primitive: deterministic render tools may poll it off-RT
   * when they advance faster than the GPU worker's real-time watermark.
   */
  bool next_callback_ready_for_offline() const
  {
    if (active_faulted_) return false;
    if (reading_tile_index_ != kNoTile) return true;

    uint64_t epoch = audio_epoch_id_;
    uint32_t bank_index = audio_active_bank_local_;
    uint32_t tile_index = expected_tile_index_;
    uint64_t device_start = expected_tile_device_;
    uint64_t source_start = expected_tile_source_;

    EpochActivation activation;
    bool has_activation = false;
    bool unstable_activation = false;
    read_activation_once(
      activation, has_activation, unstable_activation);
    if (unstable_activation) return false;
    if (has_activation && activation.epoch_id != audio_epoch_id_
        && activation.activation_frame <= audio_device_frame_)
    {
      if (activation_claim_.load(std::memory_order_acquire) != 0)
        return false;
      epoch = activation.epoch_id;
      bank_index = activation.bank_index;
      tile_index = 0;
      device_start = audio_device_frame_;
      source_start = activation.source_start;
    }

    if (epoch == 0 || bank_index >= kBankCount
        || tile_index >= kTilesPerBank)
      return false;
    const Tile & tile = banks_[bank_index].tiles[tile_index];
    if (tile.state.load(std::memory_order_acquire) != TileState::Ready)
      return false;
    const TileTag & tag = tile.tag;
    return tag.epoch_id == epoch
        && tag.device_start == device_start
        && tag.source_start == source_start
        && tag.frame_count > 0
        && tag.frame_count <= render_frames_
        && device_frames_ <= tag.frame_count;
  }

  uint64_t activation_acknowledged() const noexcept
  {
    return activation_acknowledged_.load(std::memory_order_acquire);
  }
  uint64_t published_activation_epoch() const noexcept
  {
    return activation_epoch_.load(std::memory_order_acquire);
  }
  uint32_t audio_active_bank() const noexcept
  {
    return audio_active_bank_.load(std::memory_order_acquire);
  }
  uint64_t published_device_frame() const noexcept
  {
    return published_device_frame_.load(std::memory_order_acquire);
  }
  uint64_t published_source_sample() const noexcept
  {
    return published_source_sample_.load(std::memory_order_acquire);
  }
  uint64_t starvation_count() const noexcept
  {
    return starvation_count_.load(std::memory_order_acquire);
  }
  EpochTileStarvationSnapshot first_starvation_snapshot() const noexcept
  {
    EpochTileStarvationSnapshot snapshot;
    if (first_starvation_valid_.load(std::memory_order_acquire) == 0)
      return snapshot;
    snapshot.valid = true;
    snapshot.epoch_id =
      first_starvation_epoch_.load(std::memory_order_relaxed);
    snapshot.device_frame =
      first_starvation_device_.load(std::memory_order_relaxed);
    snapshot.source_sample =
      first_starvation_source_.load(std::memory_order_relaxed);
    snapshot.expected_tile_device =
      first_starvation_expected_device_.load(std::memory_order_relaxed);
    snapshot.expected_tile_source =
      first_starvation_expected_source_.load(std::memory_order_relaxed);
    snapshot.last_published_epoch =
      first_starvation_last_published_epoch_.load(
        std::memory_order_relaxed);
    snapshot.last_published_device_end =
      first_starvation_last_published_device_end_.load(
        std::memory_order_relaxed);
    snapshot.last_published_source_end =
      first_starvation_last_published_source_end_.load(
        std::memory_order_relaxed);
    snapshot.active_bank =
      first_starvation_bank_.load(std::memory_order_relaxed);
    snapshot.expected_tile_index =
      first_starvation_tile_.load(std::memory_order_relaxed);
    snapshot.observed_tile_state =
      first_starvation_state_.load(std::memory_order_relaxed);
    snapshot.ready_mask =
      first_starvation_ready_mask_.load(std::memory_order_relaxed);
    snapshot.free_mask =
      first_starvation_free_mask_.load(std::memory_order_relaxed);
    snapshot.rendering_mask =
      first_starvation_rendering_mask_.load(std::memory_order_relaxed);
    snapshot.reading_mask =
      first_starvation_reading_mask_.load(std::memory_order_relaxed);
    return snapshot;
  }
  uint64_t tag_mismatch_count() const noexcept
  {
    return tag_mismatch_count_.load(std::memory_order_acquire);
  }
  bool active_faulted() const noexcept { return active_faulted_; }

  TileState tile_state(uint32_t bank_index, uint32_t tile_index) const
  {
    if (bank_index >= kBankCount || tile_index >= kTilesPerBank)
      return TileState::Free;
    return banks_[bank_index].tiles[tile_index].state.load(
      std::memory_order_acquire);
  }

  void set_test_seam(EpochTileQueueTestSeam * seam)
  {
    test_seam_.store(seam, std::memory_order_release);
  }

private:
  struct Tile
  {
    std::atomic<TileState> state{TileState::Free};
    TileTag tag{};
    std::vector<double> samples;
  };
  struct Bank
  {
    uint64_t epoch_id = 0;
    std::array<Tile, kTilesPerBank> tiles;
  };

  static_assert(std::atomic<TileState>::is_always_lock_free);
  static_assert(std::atomic<uint8_t>::is_always_lock_free);
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  static_assert(std::atomic<uint64_t>::is_always_lock_free);

  Tile * claimed_tile(const RenderClaim & claim)
  {
    if (claim.bank_index >= kBankCount || claim.tile_index >= kTilesPerBank)
      return nullptr;
    Tile & tile = banks_[claim.bank_index].tiles[claim.tile_index];
    return claim.destination == tile.samples.data() ? &tile : nullptr;
  }

  void read_activation_once(EpochActivation & activation, bool & available,
                            bool & unstable) const
  {
    available = false;
    unstable = false;
    const uint64_t before =
      activation_sequence_.load(std::memory_order_acquire);
    if ((before & 1U) != 0)
    {
      unstable = true;
      return;
    }
    activation.epoch_id =
      activation_epoch_.load(std::memory_order_relaxed);
    activation.activation_frame =
      activation_frame_.load(std::memory_order_relaxed);
    activation.source_start =
      activation_source_.load(std::memory_order_relaxed);
    activation.bank_index =
      activation_bank_.load(std::memory_order_relaxed);
    const uint64_t after =
      activation_sequence_.load(std::memory_order_acquire);
    if (before != after || (after & 1U) != 0)
    {
      unstable = true;
      return;
    }
    available = activation.epoch_id != 0;
  }

  void release_reading_tile()
  {
    if (reading_tile_index_ == kNoTile) return;
    Tile & tile =
      banks_[audio_active_bank_local_].tiles[reading_tile_index_];
    tile.state.store(TileState::Free, std::memory_order_release);
    reading_tile_index_ = kNoTile;
    reading_offset_ = 0;
  }

  void latch_starvation(TileState observed_state)
  {
    if (!active_faulted_)
    {
      active_faulted_ = true;
      if (first_starvation_valid_.load(std::memory_order_relaxed) == 0)
      {
        uint32_t ready_mask = 0;
        uint32_t free_mask = 0;
        uint32_t rendering_mask = 0;
        uint32_t reading_mask = 0;
        if (audio_active_bank_local_ < kBankCount)
        {
          for (uint32_t i = 0; i < kTilesPerBank; ++i)
          {
            const TileState state =
              banks_[audio_active_bank_local_].tiles[i].state.load(
                std::memory_order_relaxed);
            const uint32_t bit = 1U << i;
            if (state == TileState::Ready) ready_mask |= bit;
            else if (state == TileState::Free) free_mask |= bit;
            else if (state == TileState::Rendering) rendering_mask |= bit;
            else if (state == TileState::Reading) reading_mask |= bit;
          }
        }
        first_starvation_epoch_.store(
          audio_epoch_id_, std::memory_order_relaxed);
        first_starvation_device_.store(
          audio_device_frame_, std::memory_order_relaxed);
        first_starvation_source_.store(
          audio_source_sample_, std::memory_order_relaxed);
        first_starvation_expected_device_.store(
          expected_tile_device_, std::memory_order_relaxed);
        first_starvation_expected_source_.store(
          expected_tile_source_, std::memory_order_relaxed);
        first_starvation_last_published_epoch_.store(
          last_published_epoch_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
        first_starvation_last_published_device_end_.store(
          last_published_device_end_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
        first_starvation_last_published_source_end_.store(
          last_published_source_end_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
        first_starvation_bank_.store(
          audio_active_bank_local_, std::memory_order_relaxed);
        first_starvation_tile_.store(
          expected_tile_index_, std::memory_order_relaxed);
        first_starvation_state_.store(
          static_cast<uint32_t>(observed_state),
          std::memory_order_relaxed);
        first_starvation_ready_mask_.store(
          ready_mask, std::memory_order_relaxed);
        first_starvation_free_mask_.store(
          free_mask, std::memory_order_relaxed);
        first_starvation_rendering_mask_.store(
          rendering_mask, std::memory_order_relaxed);
        first_starvation_reading_mask_.store(
          reading_mask, std::memory_order_relaxed);
        first_starvation_valid_.store(1, std::memory_order_release);
      }
      starvation_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void latch_tag_mismatch()
  {
    if (!active_faulted_)
    {
      active_faulted_ = true;
      tag_mismatch_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void advance_coordinates(uint32_t frames)
  {
    audio_device_frame_ += frames;
    audio_source_sample_ += frames;
  }

  void publish_audio_boundary()
  {
    published_device_frame_.store(
      audio_device_frame_, std::memory_order_release);
    published_source_sample_.store(
      audio_source_sample_, std::memory_order_release);
  }

  const uint32_t device_frames_;
  const uint32_t render_frames_;
  std::array<Bank, kBankCount> banks_;

  // One immutable activation descriptor while epoch != acknowledgement.
  std::atomic<uint64_t> activation_sequence_{0};
  std::atomic<uint64_t> activation_epoch_{0};
  std::atomic<uint64_t> activation_frame_{0};
  std::atomic<uint64_t> activation_source_{0};
  std::atomic<uint32_t> activation_bank_{kNoBank};
  std::atomic<uint64_t> activation_acknowledged_{0};
  std::atomic<uint64_t> activation_claim_{0};

  // Fixed callback facts observed by the worker/control side.
  std::atomic<uint32_t> audio_active_bank_{kNoBank};
  std::atomic<uint64_t> published_device_frame_{0};
  std::atomic<uint64_t> published_source_sample_{0};
  std::atomic<uint64_t> starvation_count_{0};
  std::atomic<uint64_t> tag_mismatch_count_{0};
  std::atomic<uint64_t> last_published_epoch_{0};
  std::atomic<uint64_t> last_published_device_end_{0};
  std::atomic<uint64_t> last_published_source_end_{0};
  std::atomic<uint64_t> first_starvation_valid_{0};
  std::atomic<uint64_t> first_starvation_epoch_{0};
  std::atomic<uint64_t> first_starvation_device_{0};
  std::atomic<uint64_t> first_starvation_source_{0};
  std::atomic<uint64_t> first_starvation_expected_device_{0};
  std::atomic<uint64_t> first_starvation_expected_source_{0};
  std::atomic<uint64_t> first_starvation_last_published_epoch_{0};
  std::atomic<uint64_t> first_starvation_last_published_device_end_{0};
  std::atomic<uint64_t> first_starvation_last_published_source_end_{0};
  std::atomic<uint32_t> first_starvation_bank_{0};
  std::atomic<uint32_t> first_starvation_tile_{0};
  std::atomic<uint32_t> first_starvation_state_{0};
  std::atomic<uint32_t> first_starvation_ready_mask_{0};
  std::atomic<uint32_t> first_starvation_free_mask_{0};
  std::atomic<uint32_t> first_starvation_rendering_mask_{0};
  std::atomic<uint32_t> first_starvation_reading_mask_{0};
  std::atomic<EpochTileQueueTestSeam *> test_seam_{nullptr};

  // Audio-thread-owned cursor state.
  uint64_t audio_epoch_id_ = 0;
  uint32_t audio_active_bank_local_ = kNoBank;
  uint64_t audio_device_frame_ = 0;
  uint64_t audio_source_sample_ = 0;
  uint64_t expected_tile_device_ = 0;
  uint64_t expected_tile_source_ = 0;
  uint32_t expected_tile_index_ = 0;
  uint32_t reading_tile_index_ = kNoTile;
  uint32_t reading_offset_ = 0;
  bool active_faulted_ = false;
};

} // namespace tropical_runtime
