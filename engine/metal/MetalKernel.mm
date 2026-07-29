// MetalKernel — ObjC++ implementation. See MetalKernel.hpp for the contract
// and lean/Tropical/Ir/EmitMsl.lean for the kernel ABI this must match:
//
//   buffer(0) device float*                  output_buffer
//   buffer(1) constant float*                slots
//   buffer(2) constant TropicalKernelConsts& k
//   buffer(3) constant float*                coeff_columns   (ONLY when the
//             plan advertises hoisted coefficient columns — column_count > 0;
//             ordinary kernels keep the exact 3-binding signature)
//   thread_position_in_grid                  = sample index
//
// with TropicalKernelConsts = { ulong start_sample_index; float sample_rate;
// uint buffer_length; } (16 bytes, no padding).

#include "MetalKernel.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace tropical_metal
{

namespace
{
thread_local bool current_thread_is_audio_callback = false;
std::atomic<uint64_t> callback_thread_violations{0};

bool reject_audio_callback_entry()
{
  if (!current_thread_is_audio_callback) return false;
  callback_thread_violations.fetch_add(1, std::memory_order_relaxed);
  return true;
}

// Matches the MSL struct exactly (8 + 4 + 4).
struct KernelConsts
{
  uint64_t start_sample_index;
  float    sample_rate;
  uint32_t buffer_length;
};
static_assert(sizeof(KernelConsts) == 16, "KernelConsts must match the MSL layout");

// Process-wide device + queue (one GPU, many kernels; queues are cheap but
// one is enough — dispatches are serialized per block anyway).
id<MTLDevice> shared_device()
{
  static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  return dev;
}

id<MTLCommandQueue> shared_queue()
{
  static id<MTLCommandQueue> q = [shared_device() newCommandQueue];
  return q;
}
} // namespace

// Maximum qualification depth. TROPICAL_METAL_PIPELINE_DEPTH=1..3 is the
// explicit opt-in sweep seam.
// Blocks are pre-rendered AHEAD of the playhead. Legal because kernels are
// closed-form — block S+kB is a pure function of its sample index and the slot
// snapshot at enqueue time. Audio-position latency is ZERO; param changes lag
// up to D blocks.
static constexpr uint32_t kMaxPipelineDepth = 3;

enum class RingDispatchState : uint8_t
{
  NeverSubmitted,
  Pending,
  Succeeded,
  Failed,
};

struct MetalKernel
{
  id<MTLComputePipelineState> pso    = nil;
  id<MTLBuffer>               out    = nil;  // f32 × buffer_length, shared (sync path)
  id<MTLBuffer>               slots  = nil;  // f32 × slot_count, shared (sync path)
  // Packed coefficient-column buffer (buffer(3), banks-as-data) — nil when
  // the plan hoists no columns. Sized at load from the advertised slots'
  // capacities; a hot-swap creates a fresh MetalKernel, so the size follows
  // the plan across swaps like every other buffer here.
  id<MTLBuffer>               columns = nil; // f32 × column_count, shared (sync path)
  uint32_t capacity     = 0;
  uint32_t slot_count   = 0;
  uint32_t column_count = 0;
  // f64→f32 staging (audio-thread use; preallocated — no alloc in process).
  std::vector<float> slot_staging;

  // ── Pipelined mode (TROPICAL_METAL_PIPELINE_DEPTH=1..3) ────────────────
  uint32_t depth = 0;
  id<MTLBuffer>        ring_out[kMaxPipelineDepth]     = { nil, nil, nil };
  id<MTLBuffer>        ring_slots[kMaxPipelineDepth]   = { nil, nil, nil };
  // Per-ring-entry column buffers, like slots: an in-flight dispatch must
  // never see a later upload, so columns get a buffer per ring entry, not
  // a shared one.
  id<MTLBuffer>        ring_columns[kMaxPipelineDepth] = { nil, nil, nil };
  dispatch_semaphore_t ring_done[kMaxPipelineDepth]    = { nullptr, nullptr, nullptr };
  // Retaining the command buffer until its ring entry is consumed lets the
  // audio thread classify the completed dispatch from Metal's authoritative
  // status/error before it reads shared output. The completion handler only
  // signals a preallocated semaphore, so teardown remains safe even when a
  // superseded kernel still has futures in flight.
  id<MTLCommandBuffer> ring_command[kMaxPipelineDepth] = { nil, nil, nil };
  RingDispatchState ring_state[kMaxPipelineDepth] = {
    RingDispatchState::NeverSubmitted,
    RingDispatchState::NeverSubmitted,
    RingDispatchState::NeverSubmitted,
  };
  bool     primed       = false;
  uint32_t read_pos     = 0;
  uint64_t next_enqueue = 0;   // sample index of the next block to submit

  // Deterministic no-DAC test seam, captured once at construction so the
  // realtime path never reads the environment. 0 disables it; N classifies
  // the Nth otherwise-successful completed command as failed.
  uint64_t test_fail_dispatch_at = 0;
  uint64_t completed_dispatches  = 0;
  bool dispatch_failure_latched  = false;
  bool dispatch_failure_unreported = false;
};

static bool configure_pipeline_depth(uint32_t & depth, std::string & err)
{
  depth = 0;
  const char * raw = getenv("TROPICAL_METAL_PIPELINE_DEPTH");
  if (!raw || !*raw)
    return true;

  errno = 0;
  char * end = nullptr;
  const unsigned long parsed = std::strtoul(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0'
      || parsed < 1 || parsed > kMaxPipelineDepth)
  {
    err = "MetalKernel: TROPICAL_METAL_PIPELINE_DEPTH must be an integer in [1,3]";
    return false;
  }
  depth = static_cast<uint32_t>(parsed);
  return true;
}

static bool configure_test_failure(MetalKernel & k, std::string & err)
{
  const char * raw = getenv("TROPICAL_METAL_TEST_FAIL_DISPATCH_AT");
  if (!raw || !*raw)
    return true;

  errno = 0;
  char * end = nullptr;
  const unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || parsed == 0)
  {
    err = "MetalKernel: TROPICAL_METAL_TEST_FAIL_DISPATCH_AT must be a positive integer";
    return false;
  }
  k.test_fail_dispatch_at = static_cast<uint64_t>(parsed);
  return true;
}

static bool completed_successfully(MetalKernel & k,
                                   id<MTLCommandBuffer> command)
{
  if (!command)
    return false;

  ++k.completed_dispatches;
  const bool injected =
    k.test_fail_dispatch_at != 0
    && k.completed_dispatches == k.test_fail_dispatch_at;
  return !injected
      && command.status == MTLCommandBufferStatusCompleted
      && command.error == nil;
}

static void latch_dispatch_failure(MetalKernel & k)
{
  if (!k.dispatch_failure_latched)
  {
    k.dispatch_failure_latched = true;
    k.dispatch_failure_unreported = true;
  }
}

static bool render_tile_impl(MetalKernel & k,
                             const float * slots, uint32_t n_slots,
                             const float * columns, uint32_t n_columns,
                             double sample_rate, uint64_t source_start,
                             uint32_t frames, double * destination)
{
  if (!destination || frames == 0 || frames > k.capacity
      || k.dispatch_failure_latched)
    return false;

  @autoreleasepool
  {
    if (!k.pso || !k.out || !k.slots
        || (k.column_count > 0 && !k.columns))
    {
      latch_dispatch_failure(k);
      return false;
    }

    const uint32_t n = std::min<uint32_t>(n_slots, k.slot_count);
    float * slot_destination = static_cast<float *>(k.slots.contents);
    if (!slot_destination)
    {
      latch_dispatch_failure(k);
      return false;
    }
    if (slots && n > 0)
      std::memcpy(slot_destination, slots, n * sizeof(float));

    id<MTLCommandQueue> queue = shared_queue();
    if (!queue)
    {
      latch_dispatch_failure(k);
      return false;
    }
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command)
    {
      latch_dispatch_failure(k);
      return false;
    }
    id<MTLComputeCommandEncoder> encoder =
      [command computeCommandEncoder];
    if (!encoder)
    {
      latch_dispatch_failure(k);
      return false;
    }
    [encoder setComputePipelineState:k.pso];
    [encoder setBuffer:k.out offset:0 atIndex:0];
    [encoder setBuffer:k.slots offset:0 atIndex:1];
    KernelConsts constants{
      source_start, static_cast<float>(sample_rate), frames
    };
    [encoder setBytes:&constants length:sizeof(constants) atIndex:2];
    if (k.column_count > 0)
    {
      const uint32_t count =
        std::min<uint32_t>(n_columns, k.column_count);
      void * column_destination = k.columns.contents;
      if (!column_destination)
      {
        [encoder endEncoding];
        latch_dispatch_failure(k);
        return false;
      }
      if (columns && count > 0)
        std::memcpy(
          column_destination, columns, count * sizeof(float));
      [encoder setBuffer:k.columns offset:0 atIndex:3];
    }
    const NSUInteger threads =
      std::min<NSUInteger>(k.pso.maxTotalThreadsPerThreadgroup, frames);
    [encoder dispatchThreads:MTLSizeMake(frames, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (!completed_successfully(k, command))
    {
      latch_dispatch_failure(k);
      return false;
    }

    const float * source = static_cast<const float *>(k.out.contents);
    if (!source)
    {
      latch_dispatch_failure(k);
      return false;
    }
    for (uint32_t i = 0; i < frames; ++i)
      destination[i] = static_cast<double>(source[i]);
    return true;
  }
}

static bool fail_ring_submission(MetalKernel & k, uint32_t j)
{
  k.ring_command[j] = nil;
  k.ring_state[j] = RingDispatchState::Failed;
  if (k.ring_done[j])
    dispatch_semaphore_signal(k.ring_done[j]);
  return false;
}

// Submit block `start` into ring slot j (no wait). The completion handler
// signals ring_done[j]; Metal retains the buffers until completion, and ARC
// keeps the captured semaphore alive, so teardown mid-flight is safe (the
// semaphores start at 0 and only ever get signaled — never destroyed below
// their initial value).
static bool enqueue_block(MetalKernel & k, uint32_t j,
                          const double * slots, uint32_t n_slots,
                          const float * columns, uint32_t n_columns,
                          double sample_rate, uint64_t start, uint32_t len,
                          bool commit_immediately)
{
  k.ring_state[j] = RingDispatchState::Pending;
  k.ring_command[j] = nil;
  if (!k.ring_done[j] || !k.pso || !k.ring_out[j] || !k.ring_slots[j]
      || (k.column_count > 0 && !k.ring_columns[j]))
  {
    // create() rejects these conditions. Keeping this path terminal makes a
    // future platform allocation anomaly fail closed instead of waiting on a
    // semaphore that no command can signal.
    return fail_ring_submission(k, j);
  }

  const uint32_t n = std::min<uint32_t>(n_slots, k.slot_count);
  float * sdst = static_cast<float *>(k.ring_slots[j].contents);
  if (!sdst)
    return fail_ring_submission(k, j);
  for (uint32_t i = 0; i < n; ++i)
    sdst[i] = static_cast<float>(slots[i]);

  id<MTLCommandQueue> queue = shared_queue();
  if (!queue)
    return fail_ring_submission(k, j);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  if (!cb)
    return fail_ring_submission(k, j);
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc)
    return fail_ring_submission(k, j);
  [enc setComputePipelineState:k.pso];
  [enc setBuffer:k.ring_out[j] offset:0 atIndex:0];
  [enc setBuffer:k.ring_slots[j] offset:0 atIndex:1];
  KernelConsts consts{ start, static_cast<float>(sample_rate), len };
  [enc setBytes:&consts length:sizeof(consts) atIndex:2];
  if (k.column_count > 0)
  {
    // Coefficient columns, copied at ENQUEUE like the slot snapshot —
    // pipelined columns inherit the documented D-block param lag, no new
    // invalidation machinery. `columns` is the f32 image of the ONE
    // generation this process() call captured (before audio_processing_
    // went true), so this copy carries the whole-generation no-tear
    // guarantee; the per-ring buffer keeps it from racing an in-flight
    // dispatch.
    const uint32_t nc = std::min<uint32_t>(n_columns, k.column_count);
    void * cdst = k.ring_columns[j].contents;
    if (!cdst)
    {
      [enc endEncoding];
      return fail_ring_submission(k, j);
    }
    if (columns && nc > 0)
      std::memcpy(cdst, columns, nc * sizeof(float));
    [enc setBuffer:k.ring_columns[j] offset:0 atIndex:3];
  }
  const NSUInteger tg =
    std::min<NSUInteger>(k.pso.maxTotalThreadsPerThreadgroup, len);
  [enc dispatchThreads:MTLSizeMake(len, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  dispatch_semaphore_t done = k.ring_done[j];
  [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
    dispatch_semaphore_signal(done);
  }];
  // Retain the exact command whose completion signals this entry. The consumer
  // reads status/error after the acquire wait and before touching ring_out.
  k.ring_command[j] = cb;
  if (commit_immediately)
    [cb commit];
  return true;
}

static bool wait_for_ring(MetalKernel & k, uint32_t j)
{
  if (!k.ring_done[j])
  {
    k.ring_state[j] = RingDispatchState::Failed;
    return false;
  }
  dispatch_semaphore_wait(k.ring_done[j], DISPATCH_TIME_FOREVER);
  if (k.ring_state[j] == RingDispatchState::Pending)
    k.ring_state[j] = completed_successfully(k, k.ring_command[j])
      ? RingDispatchState::Succeeded
      : RingDispatchState::Failed;
  k.ring_command[j] = nil;
  return k.ring_state[j] == RingDispatchState::Succeeded;
}

MetalKernelPtr create(const std::string & msl_source,
                      uint32_t buffer_length,
                      uint32_t slot_count,
                      uint32_t column_count,
                      std::string & err)
{
  @autoreleasepool
  {
    id<MTLDevice> dev = shared_device();
    if (!dev) { err = "MetalKernel: no Metal device"; return nullptr; }

    MTLCompileOptions * opts = [MTLCompileOptions new];
    // The numeric contract (design/fixed-carrier.md): IEEE-ordered f32 —
    // no fast-math reassociation, no fma contraction — so the SNR gates
    // are stable across driver versions.
    opts.mathMode = MTLMathModeSafe;

    NSError * nserr = nil;
    id<MTLLibrary> lib = [dev
      newLibraryWithSource:[NSString stringWithUTF8String:msl_source.c_str()]
                   options:opts
                     error:&nserr];
    if (!lib)
    {
      err = std::string("MetalKernel: MSL compile failed: ")
          + [[nserr localizedDescription] UTF8String];
      return nullptr;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"tropical_kernel"];
    if (!fn) { err = "MetalKernel: no 'tropical_kernel' in library"; return nullptr; }

    NSError * perr = nil;
    id<MTLComputePipelineState> pso =
      [dev newComputePipelineStateWithFunction:fn error:&perr];
    if (!pso)
    {
      err = std::string("MetalKernel: pipeline state failed: ")
          + [[perr localizedDescription] UTF8String];
      return nullptr;
    }

    auto k = std::make_shared<MetalKernel>();
    k->pso          = pso;
    k->capacity     = buffer_length;
    k->slot_count   = slot_count;
    k->column_count = column_count;
    k->out = [dev newBufferWithLength:(NSUInteger)buffer_length * sizeof(float)
                              options:MTLResourceStorageModeShared];
    k->slots = [dev newBufferWithLength:(NSUInteger)std::max<uint32_t>(slot_count, 1) * sizeof(float)
                                options:MTLResourceStorageModeShared];
    k->slot_staging.assign(slot_count, 0.0f);
    if (!k->out || !k->slots) { err = "MetalKernel: buffer allocation failed"; return nullptr; }
    if (column_count > 0)
    {
      k->columns = [dev newBufferWithLength:(NSUInteger)column_count * sizeof(float)
                                    options:MTLResourceStorageModeShared];
      if (!k->columns) { err = "MetalKernel: column buffer allocation failed"; return nullptr; }
    }

    if (!configure_pipeline_depth(k->depth, err))
      return nullptr;
    if (!configure_test_failure(*k, err))
      return nullptr;
    if (k->depth > 0)
    {
      for (uint32_t j = 0; j < k->depth; ++j)
      {
        k->ring_out[j] = [dev newBufferWithLength:(NSUInteger)buffer_length * sizeof(float)
                                          options:MTLResourceStorageModeShared];
        k->ring_slots[j] = [dev newBufferWithLength:(NSUInteger)std::max<uint32_t>(slot_count, 1) * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        if (column_count > 0)
        {
          k->ring_columns[j] = [dev newBufferWithLength:(NSUInteger)column_count * sizeof(float)
                                                options:MTLResourceStorageModeShared];
          if (!k->ring_columns[j])
          { err = "MetalKernel: ring column allocation failed"; return nullptr; }
        }
        k->ring_done[j] = dispatch_semaphore_create(0);
        if (!k->ring_out[j] || !k->ring_slots[j] || !k->ring_done[j])
        { err = "MetalKernel: ring allocation failed"; return nullptr; }
      }
    }
    return k;
  }
}

bool process_block(MetalKernel & k,
                   const double * slots, uint32_t n_slots,
                   const float * columns, uint32_t n_columns,
                   double sample_rate, uint64_t start_sample_index,
                   double * out_f64, uint32_t len)
{
  if (reject_audio_callback_entry()) return false;
  if (len == 0 || len > k.capacity) return false;
  // A failed command invalidates this kernel's dispatch stream. Do not submit,
  // wait, or expose any later ring contents until control publishes a fresh
  // MetalKernel; FlatRuntime continues advancing time with silent blocks.
  if (k.dispatch_failure_latched) return false;

  if (k.depth > 0)
  {
    @autoreleasepool
    {
      // Fresh kernel (or a clock jump — scrub/set_sample_index): (re)prime
      // by submitting D future blocks starting at the requested position.
      // Hot-swap stays seamless: the carried sample_index is where the ring
      // primes, and the first read waits only one dispatch (~sub-ms).
      const uint64_t queued = (uint64_t)k.depth * len;
      const uint64_t expected =
        k.next_enqueue >= queued ? k.next_enqueue - queued
                                 : std::numeric_limits<uint64_t>::max();
      bool discarded_failure = false;
      if (!k.primed || expected != start_sample_index)
      {
        if (k.primed)  // drain stale futures before re-priming
          for (uint32_t j = 0; j < k.depth; ++j)
            discarded_failure = !wait_for_ring(k, j) || discarded_failure;
        if (discarded_failure)
        {
          latch_dispatch_failure(k);
          return false;
        }
        k.next_enqueue = start_sample_index;
        k.read_pos = 0;
        for (uint32_t j = 0; j < k.depth; ++j)
        {
          if (!enqueue_block(k, j, slots, n_slots, columns, n_columns,
                             sample_rate, k.next_enqueue, len, true))
          {
            latch_dispatch_failure(k);
            return false;
          }
          k.next_enqueue += len;
        }
        k.primed = true;
      }
      const uint32_t j = k.read_pos;
      const bool current_success = wait_for_ring(k, j);
      const float * src =
        static_cast<const float *>(k.ring_out[j].contents);
      if (!current_success || !src)
      {
        latch_dispatch_failure(k);
        return false;
      }
      // Refill the consumed entry before accepting even the successful current
      // output. A nil queue/buffer/encoder path therefore fails this callback
      // closed and cannot expose a block from a now-invalid dispatch stream.
      if (!enqueue_block(k, j, slots, n_slots, columns, n_columns,
                         sample_rate, k.next_enqueue, len, false))
      {
        latch_dispatch_failure(k);
        return false;
      }
      // The replacement command is fully encoded but cannot overwrite the
      // shared ring output until commit. Copy the completed command first,
      // then start the replacement.
      for (uint32_t i = 0; i < len; ++i)
        out_f64[i] = static_cast<double>(src[i]);
      [k.ring_command[j] commit];
      k.next_enqueue += len;
      k.read_pos = (j + 1) % k.depth;
      return true;
    }
  }

  const uint32_t count = std::min<uint32_t>(n_slots, k.slot_count);
  for (uint32_t i = 0; i < count; ++i)
    k.slot_staging[i] = static_cast<float>(slots[i]);
  return render_tile_impl(
    k, k.slot_staging.data(), count, columns, n_columns,
    sample_rate, start_sample_index, len, out_f64);
}

bool render_tile(MetalKernel & k,
                 const float * slots, uint32_t n_slots,
                 const float * columns, uint32_t n_columns,
                 double sample_rate, uint64_t source_start,
                 uint32_t frames, double * destination)
{
  if (reject_audio_callback_entry()) return false;
  return render_tile_impl(
    k, slots, n_slots, columns, n_columns,
    sample_rate, source_start, frames, destination);
}

bool take_dispatch_failure(MetalKernel & k)
{
  const bool value = k.dispatch_failure_unreported;
  k.dispatch_failure_unreported = false;
  return value;
}

uint32_t pipeline_depth(const MetalKernel & k)
{
  return k.depth;
}

void set_audio_callback_thread(bool value)
{
  current_thread_is_audio_callback = value;
}

uint64_t callback_thread_violation_count()
{
  return callback_thread_violations.load(std::memory_order_acquire);
}

void reset_callback_thread_violation_count()
{
  callback_thread_violations.store(0, std::memory_order_release);
}

} // namespace tropical_metal
