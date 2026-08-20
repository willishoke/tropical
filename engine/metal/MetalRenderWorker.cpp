#include "metal/MetalRenderWorker.hpp"
#include "metal/PhaserHigherOrderMaterializer.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <ctime>
#include <utility>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_info.h>
#endif

namespace tropical_metal
{

using tropical_runtime::EpochActivation;
using tropical_runtime::EpochTileQueue;
using tropical_runtime::TileTag;

namespace
{
uint64_t current_thread_cpu_time_ns()
{
#ifdef __APPLE__
  thread_basic_info_data_t info{};
  mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
  const thread_t thread = mach_thread_self();
  const kern_return_t status = thread_info(
    thread, THREAD_BASIC_INFO,
    reinterpret_cast<thread_info_t>(&info), &count);
  mach_port_deallocate(mach_task_self(), thread);
  if (status != KERN_SUCCESS) return 0;
  return (
    static_cast<uint64_t>(info.user_time.seconds)
    + static_cast<uint64_t>(info.system_time.seconds)) * 1000000000ULL
    + (static_cast<uint64_t>(info.user_time.microseconds)
       + static_cast<uint64_t>(info.system_time.microseconds)) * 1000ULL;
#elif defined(CLOCK_THREAD_CPUTIME_ID)
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return 0;
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL
       + static_cast<uint64_t>(value.tv_nsec);
#else
  return 0;
#endif
}

void update_min(std::atomic<uint64_t> & target, uint64_t value)
{
  uint64_t observed = target.load(std::memory_order_relaxed);
  while (value < observed
         && !target.compare_exchange_weak(
           observed, value, std::memory_order_relaxed))
  {}
}

void update_max(std::atomic<uint64_t> & target, uint64_t value)
{
  uint64_t observed = target.load(std::memory_order_relaxed);
  while (value > observed
         && !target.compare_exchange_weak(
           observed, value, std::memory_order_relaxed))
  {}
}
} // namespace

MetalRenderWorker::MetalRenderWorker(
  EpochTileQueue & queue, RenderFunction render,
  std::atomic<bool> * realtime_running)
  : queue_(queue),
    render_(std::move(render)),
    realtime_running_source_(
      realtime_running ? realtime_running : &realtime_running_)
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

bool MetalRenderWorker::materialize_columns(
  BankRenderCursor & cursor, uint64_t source_start)
{
  const auto & program = cursor.request.tile_materializer;
  if (!program) return true;
  if (!program->kernel || program->interval_frames == 0
      || queue_.render_frames() % program->interval_frames != 0
      || program->tile_array_slots.empty())
    return false;

  if (!cursor.materializer_ready)
  {
    cursor.materializer_registers.assign(program->register_count, 0);
    cursor.materializer_temps.assign(program->temp_count, 0);
    cursor.materializer_arrays = cursor.request.materializer_seed_arrays;
    cursor.materializer_arrays.resize(program->array_sizes.size());
    cursor.materializer_array_ptrs.resize(program->array_sizes.size());
    cursor.base_columns = cursor.request.columns;
    for (std::size_t i = 0; i < program->array_sizes.size(); ++i)
    {
      cursor.materializer_arrays[i].resize(
        static_cast<std::size_t>(program->array_sizes[i]), 0);
      cursor.materializer_array_ptrs[i] =
        cursor.materializer_arrays[i].data();
    }
    const std::size_t segment_count =
      queue_.render_frames() / program->interval_frames;
    std::size_t prepared_size = cursor.base_columns.size();
    for (uint32_t slot : program->tile_array_slots)
    {
      if (slot >= program->array_sizes.size()) return false;
      prepared_size += segment_count
        * static_cast<std::size_t>(program->array_sizes[slot]);
    }
    cursor.prepared_columns.reserve(prepared_size);
    cursor.materializer_ready = true;
  }

  if (cursor.request.materializer_seed_arrays.size()
      != cursor.materializer_arrays.size())
    return false;
  cursor.prepared_columns = cursor.base_columns;
  const uint32_t segment_count =
    queue_.render_frames() / program->interval_frames;
  for (uint32_t segment = 0; segment < segment_count; ++segment)
  {
    for (std::size_t i = 0; i < cursor.materializer_arrays.size(); ++i)
    {
      const auto & seed = cursor.request.materializer_seed_arrays[i];
      auto & work = cursor.materializer_arrays[i];
      if (seed.size() != work.size()) return false;
      std::copy(seed.begin(), seed.end(), work.begin());
    }
    std::fill(cursor.materializer_registers.begin(),
              cursor.materializer_registers.end(), 0);
    std::fill(cursor.materializer_temps.begin(),
              cursor.materializer_temps.end(), 0);
    cursor.materializer_slots = cursor.request.materializer_slots;
    double scratch_output = 0.0;
    program->kernel(
      nullptr,
      cursor.materializer_registers.data(),
      cursor.materializer_array_ptrs.data(),
      program->array_sizes.data(),
      cursor.materializer_temps.data(),
      cursor.request.sample_rate,
      source_start + static_cast<uint64_t>(segment)
        * program->interval_frames,
      program->param_ptrs.data(),
      &scratch_output,
      1,
      cursor.materializer_slots.data());

    if (program->higher_order_phaser)
    {
      if (program->tile_array_slots.size() != 1)
        return false;
      const uint32_t image_slot = program->tile_array_slots.front();
      if (image_slot >= cursor.materializer_arrays.size()
          || !materialize_higher_order_phaser_image(
            cursor.materializer_arrays[image_slot],
            program->interval_frames))
        return false;
    }

    for (uint32_t slot : program->tile_array_slots)
    {
      if (slot >= cursor.materializer_arrays.size()) return false;
      for (int64_t bits : cursor.materializer_arrays[slot])
      {
        const double value = std::bit_cast<double>(bits);
        if (!std::isfinite(value)
            || std::abs(value) > program->max_abs_coefficient)
          return false;
        cursor.prepared_columns.push_back(static_cast<float>(value));
      }
    }
  }
  return true;
}

bool MetalRenderWorker::render_exact_fallback(
  BankRenderCursor & cursor, uint64_t source_start, uint32_t frames,
  double * destination)
{
  const auto & program = cursor.request.exact_fallback;
  if (!program || !program->kernel || !destination || frames == 0)
    return false;

  if (!cursor.exact_fallback_ready)
  {
    cursor.exact_registers.assign(program->register_count, 0);
    cursor.exact_temps.assign(program->temp_count, 0);
    cursor.exact_coeff_registers.assign(program->coeff_register_count, 0);
    cursor.exact_coeff_temps.assign(program->coeff_temp_count, 0);
    cursor.exact_arrays = program->array_storage;
    cursor.exact_arrays.resize(program->array_sizes.size());
    cursor.exact_array_ptrs.resize(program->array_sizes.size());
    for (std::size_t i = 0; i < program->array_sizes.size(); ++i)
    {
      cursor.exact_arrays[i].resize(
        static_cast<std::size_t>(program->array_sizes[i]), 0);
      cursor.exact_array_ptrs[i] = cursor.exact_arrays[i].data();
    }
    cursor.exact_fallback_ready = true;
  }

  if (program->array_storage.size() != cursor.exact_arrays.size())
    return false;
  for (std::size_t i = 0; i < cursor.exact_arrays.size(); ++i)
  {
    if (program->array_storage[i].size() != cursor.exact_arrays[i].size())
      return false;
    std::copy(program->array_storage[i].begin(),
              program->array_storage[i].end(), cursor.exact_arrays[i].begin());
  }
  std::fill(cursor.exact_registers.begin(), cursor.exact_registers.end(), 0);
  std::fill(cursor.exact_temps.begin(), cursor.exact_temps.end(), 0);
  std::fill(cursor.exact_coeff_registers.begin(),
            cursor.exact_coeff_registers.end(), 0);
  std::fill(cursor.exact_coeff_temps.begin(),
            cursor.exact_coeff_temps.end(), 0);
  cursor.exact_slots = cursor.request.exact_fallback_slots;
  if (cursor.exact_slots.size() != program->initial_slots.size())
    return false;

  double coefficient_output = 0.0;
  if (program->coeff_kernel)
    program->coeff_kernel(
      nullptr,
      cursor.exact_coeff_registers.data(),
      cursor.exact_array_ptrs.data(),
      program->array_sizes.data(),
      cursor.exact_coeff_temps.data(),
      cursor.request.sample_rate,
      0,
      program->param_ptrs.data(),
      &coefficient_output,
      1,
      cursor.exact_slots.data());

  program->kernel(
    nullptr,
    cursor.exact_registers.data(),
    cursor.exact_array_ptrs.data(),
    program->array_sizes.data(),
    cursor.exact_temps.data(),
    cursor.request.sample_rate,
    source_start,
    program->param_ptrs.data(),
    destination,
    frames,
    cursor.exact_slots.data());
  for (uint32_t i = 0; i < frames; ++i)
    if (!std::isfinite(destination[i])) return false;
  return true;
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
  worker_start_time_ns_.store(
    monotonic_time_ns(), std::memory_order_release);
  worker_cpu_baseline_ns_.store(
    current_thread_cpu_time_ns(), std::memory_order_release);
  for (;;)
  {
    observe_activation_acknowledgement();
    // Finish the newly published bank before servicing the old active bank.
    // Live controls publish after one prime tile; the remaining tiles are
    // rendered during the one-tile lead instead of delaying publication by a
    // complete four-tile window.
    bool worked = refill_pending_bank();
    if (!worked) worked = refill_active_bank();

    std::shared_ptr<PendingRequest> pending;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stopping_)
      {
        worker_cpu_latest_ns_.store(
          current_thread_cpu_time_ns(), std::memory_order_release);
        return;
      }
      // schedule() returns once a complete activation has been published; it
      // does not wait for the audio callback to reach that future boundary.
      // With no device callback, an intermediate epoch can never be
      // acknowledged, so stopped-DAC edits coalesce to the newest complete
      // image. While audio is running we must preserve activation order:
      // FlatRuntime's glide companions project the state at each activation,
      // and reusing a projection from a cancelled epoch causes a discontinuity.
      // The queue's claim CAS still makes stopped-DAC cancellation lose safely
      // to a callback that has already committed to the activation.
      if (pending_activation_epoch_ != 0 && !requests_.empty()
          && !realtime_running_source_->load(std::memory_order_acquire)
          && queue_.cancel_unclaimed_activation(
            pending_activation_epoch_))
      {
        for (BankRenderCursor & cursor : bank_cursors_)
          if (cursor.valid
              && cursor.request.epoch_id == pending_activation_epoch_)
            cursor.valid = false;
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
        if (stopping_)
        {
          worker_cpu_latest_ns_.store(
            current_thread_cpu_time_ns(), std::memory_order_release);
          return;
        }
      }
    }
    if (!pending)
    {
      worker_cpu_latest_ns_.store(
        current_thread_cpu_time_ns(), std::memory_order_release);
      continue;
    }

    pending->result = prepare_activation(std::move(pending->request));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending->done = true;
    }
    pending->completed.notify_one();
    worker_cpu_latest_ns_.store(
      current_thread_cpu_time_ns(), std::memory_order_release);
  }
}

bool MetalRenderWorker::observe_activation_acknowledgement()
{
  if (pending_activation_epoch_ == 0
      || queue_.activation_acknowledged() != pending_activation_epoch_)
    return false;

  const uint64_t acknowledged = monotonic_time_ns();
  activation_acknowledged_time_.store(
    acknowledged, std::memory_order_release);
  const uint64_t requested =
    request_received_time_.load(std::memory_order_acquire);
  if (requested != 0 && acknowledged >= requested)
  {
    const uint64_t latency = acknowledged - requested;
    activation_latency_count_.fetch_add(1, std::memory_order_relaxed);
    activation_latency_total_ns_.fetch_add(latency, std::memory_order_relaxed);
    update_min(activation_latency_min_ns_, latency);
    update_max(activation_latency_max_ns_, latency);
  }
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
  return render_one(active_bank_, cursor, false);
}

bool MetalRenderWorker::refill_pending_bank()
{
  if (pending_activation_epoch_ == 0) return false;
  for (uint32_t bank = 0; bank < EpochTileQueue::kBankCount; ++bank)
  {
    BankRenderCursor & cursor = bank_cursors_[bank];
    if (!cursor.valid
        || cursor.request.epoch_id != pending_activation_epoch_)
      continue;
    if (queue_.tile_state(bank, cursor.next_slot)
        != tropical_runtime::TileState::Free)
      return false;
    return render_one(bank, cursor, false);
  }
  return false;
}

EpochScheduleResult
MetalRenderWorker::prepare_activation(RenderEpochRequest request)
{
  const uint64_t preparation_device_frame =
    queue_.published_device_frame();
  // Publish the new request identity only after clearing every later stage.
  // Readers may observe an incomplete in-flight request, but never a prefix
  // from this request combined with a suffix from the preceding activation.
  target_reserved_time_.store(0, std::memory_order_relaxed);
  render_submitted_time_.store(0, std::memory_order_relaxed);
  gpu_completion_time_.store(0, std::memory_order_relaxed);
  window_ready_time_.store(0, std::memory_order_relaxed);
  activation_published_time_.store(0, std::memory_order_relaxed);
  activation_acknowledged_time_.store(0, std::memory_order_relaxed);
  old_epoch_retired_time_.store(0, std::memory_order_relaxed);
  request_received_time_.store(
    monotonic_time_ns(), std::memory_order_release);
  const uint64_t retarget_started = monotonic_time_ns();
  uint32_t retargets = 0;
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
          : observed_device + activation_lead_frames(request.transition));
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

    // A fixed request can become stale while it waits behind a previously
    // published activation. Reject that coordinate before spending another
    // GPU tile on a candidate FlatRuntime must recompute at a fresh E.
    const uint64_t admissible_before_render =
      observed_device > UINT64_MAX - queue_.device_frames()
        ? UINT64_MAX
        : observed_device + queue_.device_frames();
    if (request.fixed_activation && activation_frame != 0
        && activation_frame < admissible_before_render)
    {
      activation_retargets_.fetch_add(1, std::memory_order_relaxed);
      learn_live_activation_lead(
        request, preparation_device_frame, observed_device, true);
      return {
        false, "MetalRenderWorker: activation target became inadmissible",
        request.epoch_id, activation_frame, source_start, true
      };
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
    if (!render_tiles(
          staging_bank, candidate, prime_tile_count(request.transition)))
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
      {
        learn_live_activation_lead(
          request, preparation_device_frame, device_after_render, true);
        return {
          false, "MetalRenderWorker: activation target became inadmissible",
          request.epoch_id, activation_frame, source_start, true
        };
      }
      ++retargets;
      const uint64_t now = monotonic_time_ns();
      if (retargets >= kActivationRetargetLimit
          || now - retarget_started >= kActivationRetargetTimeoutNs)
      {
        activation_failures_.fetch_add(1, std::memory_order_relaxed);
        return {
          false,
          "MetalRenderWorker: candidate remained late after "
            + std::to_string(retargets) + " retargets",
          request.epoch_id, activation_frame, source_start
        };
      }
      continue;
    }

    if (request.fixed_activation)
      learn_live_activation_lead(
        request, preparation_device_frame, device_after_render, false);

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
    no_active && device == 0 ? 0 : device + activation_lead_frames(transition);
  return {
    device,
    frame,
    transition == EpochTransitionKind::ClockJump
      ? requested_source
      : source + (frame - device)
  };
}

uint64_t MetalRenderWorker::device_frame() const noexcept
{
  return queue_.published_device_frame();
}

bool MetalRenderWorker::wait_for_activation_acknowledgement(
  uint64_t epoch_id) const noexcept
{
  if (epoch_id == 0
      || !realtime_running_source_->load(std::memory_order_acquire))
    return true;
  const uint64_t started = monotonic_time_ns();
  while (queue_.activation_acknowledged() != epoch_id)
  {
    // A DAC stop removes the only consumer that can acknowledge the future
    // boundary. Stopped-audio controls deliberately coalesce instead.
    if (!realtime_running_source_->load(std::memory_order_acquire))
      return true;
    if (monotonic_time_ns() - started >= kActivationRetargetTimeoutNs)
      return false;
    std::this_thread::yield();
  }
  return true;
}

bool MetalRenderWorker::render_one(
  uint32_t bank_index, BankRenderCursor & cursor,
  bool record_candidate_stage)
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
  if (record_candidate_stage)
    render_submitted_time_.store(
      monotonic_time_ns(), std::memory_order_release);
  const uint64_t render_started = monotonic_time_ns();
  bool rendered = false;
  if (!materialize_columns(cursor, cursor.next_source))
  {
    materialization_failures_.fetch_add(1, std::memory_order_relaxed);
    rendered = render_exact_fallback(
      cursor, cursor.next_source, queue_.render_frames(), claim.destination);
    if (rendered)
      exact_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  }
  else
  {
    if (cursor.request.tile_materializer)
      cursor.request.columns = cursor.prepared_columns;
    rendered = render_(
      cursor.request, cursor.next_source,
      queue_.render_frames(), claim.destination);
  }
  const uint64_t render_finished = monotonic_time_ns();
  render_time_ns_.fetch_add(
    render_finished - render_started, std::memory_order_relaxed);
  rendered_frame_count_.fetch_add(
    queue_.render_frames(), std::memory_order_relaxed);
  if (record_candidate_stage)
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

bool MetalRenderWorker::render_tiles(
  uint32_t bank_index, BankRenderCursor & cursor, uint32_t tile_count)
{
  for (uint32_t i = 0; i < tile_count; ++i)
    if (!render_one(bank_index, cursor, true))
      return false;
  return true;
}

uint64_t MetalRenderWorker::activation_lead_frames(
  EpochTransitionKind transition) const noexcept
{
  switch (transition)
  {
    case EpochTransitionKind::Continuous:
    case EpochTransitionKind::ClockJump:
      return std::max<uint64_t>(
        queue_.render_frames(),
        live_activation_lead_frames_.load(std::memory_order_acquire));
    case EpochTransitionKind::Fresh:
    case EpochTransitionKind::HotSwap:
      return queue_.capacity_frames();
  }
  return queue_.capacity_frames();
}

void MetalRenderWorker::learn_live_activation_lead(
  const RenderEpochRequest & request,
  uint64_t preparation_device_frame,
  uint64_t device_after_render,
  bool candidate_was_late) noexcept
{
  if ((request.transition != EpochTransitionKind::Continuous
       && request.transition != EpochTransitionKind::ClockJump)
      || request.reservation_device_frame == UINT64_MAX)
    return;

  const uint64_t control_progress =
    request.enqueue_device_frame != UINT64_MAX
      && request.enqueue_device_frame > request.reservation_device_frame
      ? request.enqueue_device_frame - request.reservation_device_frame
      : 0;
  const uint64_t worker_progress =
    device_after_render > preparation_device_frame
      ? device_after_render - preparation_device_frame
      : 0;
  const uint64_t preparation_progress =
    control_progress > UINT64_MAX - worker_progress
      ? UINT64_MAX
      : control_progress + worker_progress;
  const uint64_t elapsed_horizon =
    preparation_progress > UINT64_MAX - queue_.device_frames()
      ? UINT64_MAX
      : preparation_progress + queue_.device_frames();
  const uint64_t learned = std::max<uint64_t>(
    queue_.render_frames(), elapsed_horizon);
  if (candidate_was_late)
    update_max(live_activation_lead_frames_, learned);
  else
    live_activation_lead_frames_.store(learned, std::memory_order_release);
}

uint32_t MetalRenderWorker::prime_tile_count(
  EpochTransitionKind transition) noexcept
{
  switch (transition)
  {
    case EpochTransitionKind::Continuous:
    case EpochTransitionKind::ClockJump:
      return 1;
    case EpochTransitionKind::Fresh:
    case EpochTransitionKind::HotSwap:
      return EpochTileQueue::kTilesPerBank;
  }
  return EpochTileQueue::kTilesPerBank;
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
uint64_t MetalRenderWorker::materialization_failure_count() const noexcept
{
  return materialization_failures_.load(std::memory_order_acquire);
}
uint64_t MetalRenderWorker::exact_fallback_count() const noexcept
{
  return exact_fallbacks_.load(std::memory_order_acquire);
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

MetalActivationLatencyStats
MetalRenderWorker::activation_latency_stats() const noexcept
{
  const uint64_t count =
    activation_latency_count_.load(std::memory_order_acquire);
  return {
    count,
    activation_latency_total_ns_.load(std::memory_order_acquire),
    count == 0 ? 0
      : activation_latency_min_ns_.load(std::memory_order_acquire),
    activation_latency_max_ns_.load(std::memory_order_acquire),
  };
}

uint64_t MetalRenderWorker::worker_cpu_time_ns() const noexcept
{
  const uint64_t baseline =
    worker_cpu_baseline_ns_.load(std::memory_order_acquire);
  const uint64_t latest =
    worker_cpu_latest_ns_.load(std::memory_order_acquire);
  return latest >= baseline ? latest - baseline : 0;
}

uint64_t MetalRenderWorker::worker_wall_time_ns() const noexcept
{
  const uint64_t start =
    worker_start_time_ns_.load(std::memory_order_acquire);
  return start == 0 ? 0 : monotonic_time_ns() - start;
}

uint64_t MetalRenderWorker::render_time_ns() const noexcept
{
  return render_time_ns_.load(std::memory_order_acquire);
}

uint64_t MetalRenderWorker::rendered_frame_count() const noexcept
{
  return rendered_frame_count_.load(std::memory_order_acquire);
}

void MetalRenderWorker::set_realtime_running(bool running) noexcept
{
  realtime_running_source_->store(running, std::memory_order_release);
  wake_.notify_one();
}

void MetalRenderWorker::set_test_seam(
  MetalRenderWorkerTestSeam * seam) noexcept
{
  test_seam_.store(seam, std::memory_order_release);
}

} // namespace tropical_metal
