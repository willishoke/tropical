#include "metal/MetalRenderWorker.hpp"

#include <chrono>
#include <utility>

namespace tropical_metal
{

using tropical_runtime::EpochActivation;
using tropical_runtime::EpochTileQueue;
using tropical_runtime::TileTag;

MetalRenderWorker::MetalRenderWorker(
  EpochTileQueue & queue, RenderFunction render)
  : queue_(queue),
    render_(std::move(render))
{
  if (!render_)
  {
    render_ = [](const RenderEpochRequest & request, uint64_t source_start,
                 uint32_t frames, double * destination) {
      return request.kernel
          && tropical_metal::render_tile(
            *request.kernel,
            request.slots.data(),
            static_cast<uint32_t>(request.slots.size()),
            request.columns.data(),
            static_cast<uint32_t>(request.columns.size()),
            request.sample_rate, source_start, frames, destination);
    };
  }
}

MetalRenderWorker::~MetalRenderWorker()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    for (auto & pending : requests_)
    {
      pending->result = {
        false, "MetalRenderWorker: shutting down",
        pending->request.epoch_id, 0, 0
      };
      pending->done = true;
      pending->completed.notify_one();
    }
    requests_.clear();
  }
  wake_.notify_one();
  if (thread_.joinable())
    thread_.join();
}

void MetalRenderWorker::ensure_started()
{
  if (started_) return;
  started_ = true;
  thread_ = std::thread(&MetalRenderWorker::run, this);
}

EpochScheduleResult
MetalRenderWorker::schedule(RenderEpochRequest request)
{
  if (request.epoch_id == 0)
    return {false, "MetalRenderWorker: epoch id must be nonzero", 0, 0, 0};
  if (!request.kernel && !render_)
    return {
      false, "MetalRenderWorker: request has no Metal kernel",
      request.epoch_id, 0, 0
    };

  auto pending = std::make_shared<PendingRequest>(std::move(request));
  std::unique_lock<std::mutex> lock(mutex_);
  if (stopping_)
    return {
      false, "MetalRenderWorker: shutting down",
      pending->request.epoch_id, 0, 0
    };
  ensure_started();
  requests_.push_back(pending);
  wake_.notify_one();
  pending->completed.wait(lock, [&] { return pending->done; });
  return pending->result;
}

void MetalRenderWorker::run()
{
  for (;;)
  {
    observe_activation_acknowledgement();
    bool worked = refill_active_bank();

    std::shared_ptr<PendingRequest> pending;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stopping_) return;
      if (pending_activation_epoch_ != 0 && active_epoch_ == 0
          && !requests_.empty()
          && queue_.cancel_unclaimed_activation(
            pending_activation_epoch_))
      {
        bank_cursors_[0].valid = false;
        pending_activation_epoch_ = 0;
      }
      if (pending_activation_epoch_ == 0 && !requests_.empty())
      {
        pending = requests_.front();
        requests_.pop_front();
      }
      else if (!worked)
      {
        wake_.wait_for(lock, std::chrono::microseconds(100));
        if (stopping_) return;
      }
    }
    if (!pending) continue;

    pending->result = prepare_activation(std::move(pending->request));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending->done = true;
    }
    pending->completed.notify_one();
  }
}

bool MetalRenderWorker::observe_activation_acknowledgement()
{
  if (pending_activation_epoch_ == 0
      || queue_.activation_acknowledged() != pending_activation_epoch_)
    return false;

  activation_acknowledged_time_.store(
    monotonic_time_ns(), std::memory_order_release);
  const uint32_t previous = active_bank_;
  active_bank_ = queue_.audio_active_bank();
  active_epoch_ = pending_activation_epoch_;
  pending_activation_epoch_ = 0;
  if (previous != EpochTileQueue::kNoBank && previous != active_bank_)
  {
    bank_cursors_[previous].valid = false;
    old_epoch_retired_time_.store(
      monotonic_time_ns(), std::memory_order_release);
  }
  return true;
}

bool MetalRenderWorker::refill_active_bank()
{
  if (active_bank_ == EpochTileQueue::kNoBank) return false;
  BankRenderCursor & cursor = bank_cursors_[active_bank_];
  if (!cursor.valid || cursor.request.epoch_id != active_epoch_) return false;
  if (queue_.tile_state(active_bank_, cursor.next_slot)
      != tropical_runtime::TileState::Free)
    return false;
  return render_one(active_bank_, cursor);
}

EpochScheduleResult
MetalRenderWorker::prepare_activation(RenderEpochRequest request)
{
  request_received_time_.store(
    monotonic_time_ns(), std::memory_order_release);
  const uint32_t staging_bank =
    active_bank_ == EpochTileQueue::kNoBank ? 0U : 1U - active_bank_;

  for (;;)
  {
    while (refill_active_bank()) {}

    const uint64_t observed_device = queue_.published_device_frame();
    const uint64_t observed_source = queue_.published_source_sample();
    const bool fresh = active_bank_ == EpochTileQueue::kNoBank;
    const uint64_t activation_frame = request.fixed_activation
      ? request.activation_frame
      : (fresh && observed_device == 0
          ? 0
          : observed_device + queue_.render_frames());
    const uint64_t source_start = request.fixed_activation
      ? request.source_origin
      : (request.transition == EpochTransitionKind::ClockJump
          ? request.source_origin
          : observed_source + (activation_frame - observed_device));
    target_reserved_time_.store(
      monotonic_time_ns(), std::memory_order_release);
    if (MetalRenderWorkerTestSeam * seam =
          test_seam_.load(std::memory_order_acquire);
        seam
        && seam->pause_after_target_reserved.load(
          std::memory_order_acquire))
    {
      seam->target_reserved.store(true, std::memory_order_release);
      while (!seam->release_target.load(std::memory_order_acquire))
        std::this_thread::yield();
    }

    if (!queue_.begin_epoch(staging_bank, request.epoch_id))
    {
      activation_failures_.fetch_add(1, std::memory_order_relaxed);
      return {
        false, "MetalRenderWorker: staging bank is not reusable",
        request.epoch_id, activation_frame, source_start
      };
    }

    BankRenderCursor candidate;
    candidate.valid = true;
    candidate.request = request;
    candidate.next_device = activation_frame;
    candidate.next_source = source_start;
    candidate.next_slot = 0;
    if (!render_window(staging_bank, candidate))
    {
      activation_failures_.fetch_add(1, std::memory_order_relaxed);
      return {
        false, "MetalRenderWorker: candidate tile render failed",
        request.epoch_id, activation_frame, source_start
      };
    }
    window_ready_time_.store(
      monotonic_time_ns(), std::memory_order_release);

    const uint64_t device_after_render = queue_.published_device_frame();
    if (activation_frame != 0
        && activation_frame < device_after_render + queue_.device_frames())
    {
      activation_retargets_.fetch_add(1, std::memory_order_relaxed);
      if (request.fixed_activation)
        return {
          false, "MetalRenderWorker: activation target became inadmissible",
          request.epoch_id, activation_frame, source_start, true
        };
      continue;
    }

    if (!queue_.publish_activation(
          EpochActivation{
            request.epoch_id, activation_frame, source_start, staging_bank
          }))
    {
      activation_failures_.fetch_add(1, std::memory_order_relaxed);
      return {
        false, "MetalRenderWorker: activation publication refused",
        request.epoch_id, activation_frame, source_start
      };
    }
    activation_published_time_.store(
      monotonic_time_ns(), std::memory_order_release);
    bank_cursors_[staging_bank] = std::move(candidate);
    pending_activation_epoch_ = request.epoch_id;
    return {
      true, {}, request.epoch_id, activation_frame, source_start
    };
  }
}

EpochReservation MetalRenderWorker::reserve(
  EpochTransitionKind transition,
  uint64_t requested_source) const noexcept
{
  const uint64_t device = queue_.published_device_frame();
  const uint64_t source = queue_.published_source_sample();
  const bool no_active =
    queue_.audio_active_bank() == EpochTileQueue::kNoBank;
  const uint64_t frame =
    no_active && device == 0 ? 0 : device + queue_.render_frames();
  return {
    frame,
    transition == EpochTransitionKind::ClockJump
      ? requested_source
      : source + (frame - device)
  };
}

bool MetalRenderWorker::render_one(
  uint32_t bank_index, BankRenderCursor & cursor)
{
  EpochTileQueue::RenderClaim claim;
  const TileTag tag{
    cursor.request.epoch_id,
    cursor.next_device,
    cursor.next_source,
    queue_.render_frames()
  };
  if (!queue_.claim_tile(
        bank_index, cursor.next_slot, tag, claim))
    return false;
  render_submitted_time_.store(
    monotonic_time_ns(), std::memory_order_release);
  const bool rendered = render_(
    cursor.request, cursor.next_source,
    queue_.render_frames(), claim.destination);
  gpu_completion_time_.store(
    monotonic_time_ns(), std::memory_order_release);
  if (!rendered)
  {
    queue_.discard_tile(claim);
    if (cursor.request.kernel
        && tropical_metal::take_dispatch_failure(*cursor.request.kernel))
      dispatch_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (cursor.request.epoch_id != tag.epoch_id)
  {
    queue_.discard_tile(claim);
    stale_completions_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!queue_.publish_tile(claim))
  {
    stale_completions_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  cursor.next_device += tag.frame_count;
  cursor.next_source += tag.frame_count;
  cursor.next_slot =
    (cursor.next_slot + 1) % EpochTileQueue::kTilesPerBank;
  return true;
}

bool MetalRenderWorker::render_window(
  uint32_t bank_index, BankRenderCursor & cursor)
{
  for (uint32_t i = 0; i < EpochTileQueue::kTilesPerBank; ++i)
    if (!render_one(bank_index, cursor))
      return false;
  return true;
}

uint64_t MetalRenderWorker::monotonic_time_ns()
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t MetalRenderWorker::dispatch_failure_count() const noexcept
{
  return dispatch_failures_.load(std::memory_order_acquire);
}
uint64_t MetalRenderWorker::activation_retarget_count() const noexcept
{
  return activation_retargets_.load(std::memory_order_acquire);
}
uint64_t MetalRenderWorker::activation_failure_count() const noexcept
{
  return activation_failures_.load(std::memory_order_acquire);
}
uint64_t MetalRenderWorker::stale_completion_count() const noexcept
{
  return stale_completions_.load(std::memory_order_acquire);
}

MetalWorkerStageTimes MetalRenderWorker::stage_times() const noexcept
{
  return {
    request_received_time_.load(std::memory_order_acquire),
    target_reserved_time_.load(std::memory_order_acquire),
    render_submitted_time_.load(std::memory_order_acquire),
    gpu_completion_time_.load(std::memory_order_acquire),
    window_ready_time_.load(std::memory_order_acquire),
    activation_published_time_.load(std::memory_order_acquire),
    activation_acknowledged_time_.load(std::memory_order_acquire),
    old_epoch_retired_time_.load(std::memory_order_acquire),
  };
}

void MetalRenderWorker::set_test_seam(
  MetalRenderWorkerTestSeam * seam) noexcept
{
  test_seam_.store(seam, std::memory_order_release);
}

} // namespace tropical_metal
