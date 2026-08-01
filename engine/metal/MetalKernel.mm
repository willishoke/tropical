// MetalKernel — ObjC++ implementation. See MetalKernel.hpp for the contract
// and lean/Tropical/Ir/EmitMsl.lean for the kernel ABI this must match:
//
//   buffer(0) device float*                  output_buffer
//   buffer(1) constant float*                slots
//   buffer(2) constant TropicalKernelConsts& k
//   buffer(3) constant float*                coeff_columns   (ONLY when the
//             plan advertises hoisted coefficient columns — column_count > 0;
//             ordinary kernels keep the exact 3-binding signature)
//   buffer(4) const device float*            immutable_asset (ONLY for Plan-6;
//             uploaded once when the kernel is created)
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

struct MetalKernel
{
  id<MTLComputePipelineState> pso    = nil;
  id<MTLBuffer>               out    = nil;  // f32 × render quantum, worker-owned
  id<MTLBuffer>               slots  = nil;  // immutable epoch snapshot upload
  // Packed coefficient-column buffer (buffer(3), banks-as-data) — nil when
  // the plan hoists no columns. Sized at load from the advertised slots'
  // capacities; a hot-swap creates a fresh MetalKernel, so the size follows
  // the plan across swaps like every other buffer here.
  id<MTLBuffer>               columns = nil; // packed immutable epoch columns
  id<MTLBuffer>               immutable_asset = nil; // one-time Plan-6 upload
  uint32_t capacity     = 0;
  uint32_t slot_count   = 0;
  uint32_t column_count = 0;
  std::size_t immutable_asset_bytes = 0;

  // Deterministic no-DAC test seam, captured once at construction so the
  // realtime path never reads the environment. 0 disables it; N classifies
  // the Nth otherwise-successful completed command as failed.
  uint64_t test_fail_dispatch_at = 0;
  uint64_t completed_dispatches  = 0;
  bool dispatch_failure_latched  = false;
  bool dispatch_failure_unreported = false;
};

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
        || (k.column_count > 0 && !k.columns)
        || (k.immutable_asset_bytes > 0 && !k.immutable_asset))
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
    if (k.immutable_asset_bytes > 0)
      [encoder setBuffer:k.immutable_asset offset:0 atIndex:4];
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

MetalKernelPtr create(const std::string & msl_source,
                      uint32_t buffer_length,
                      uint32_t slot_count,
                      uint32_t column_count,
                      const uint8_t * asset_bytes,
                      std::size_t asset_byte_count,
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
    k->immutable_asset_bytes = asset_byte_count;
    k->out = [dev newBufferWithLength:(NSUInteger)buffer_length * sizeof(float)
                              options:MTLResourceStorageModeShared];
    k->slots = [dev newBufferWithLength:(NSUInteger)std::max<uint32_t>(slot_count, 1) * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (!k->out || !k->slots) { err = "MetalKernel: buffer allocation failed"; return nullptr; }
    if (column_count > 0)
    {
      k->columns = [dev newBufferWithLength:(NSUInteger)column_count * sizeof(float)
                                    options:MTLResourceStorageModeShared];
      if (!k->columns) { err = "MetalKernel: column buffer allocation failed"; return nullptr; }
    }
    if (asset_byte_count > 0)
    {
      if (!asset_bytes)
      {
        err = "MetalKernel: immutable asset bytes are null";
        return nullptr;
      }
      k->immutable_asset = [dev newBufferWithLength:(NSUInteger)asset_byte_count
                                            options:MTLResourceStorageModeShared];
      if (!k->immutable_asset || !k->immutable_asset.contents)
      {
        err = "MetalKernel: immutable asset buffer allocation failed";
        return nullptr;
      }
      std::memcpy(k->immutable_asset.contents, asset_bytes, asset_byte_count);
    }

    if (!configure_test_failure(*k, err))
      return nullptr;
    return k;
  }
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
