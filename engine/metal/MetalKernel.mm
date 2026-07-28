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
#include <cerrno>
#include <cstring>
#include <limits>
#include <vector>

namespace tropical_metal
{

namespace
{
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

// Maximum qualification depth.  TROPICAL_METAL_PIPELINE=1 keeps its existing
// D=3 behavior; TROPICAL_METAL_PIPELINE_DEPTH=1..3 is the opt-in sweep seam.
// Blocks are pre-rendered AHEAD of the playhead. Legal because kernels are
// closed-form — block S+kB is a pure function of its sample index and the slot
// snapshot at enqueue time. Audio-position latency is ZERO; param changes lag
// up to D blocks.
static constexpr uint32_t kMaxPipelineDepth = 3;

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

  // ── Pipelined mode (TROPICAL_METAL_PIPELINE=1) ─────────────────────────
  uint32_t depth = 0;
  id<MTLBuffer>        ring_out[kMaxPipelineDepth]     = { nil, nil, nil };
  id<MTLBuffer>        ring_slots[kMaxPipelineDepth]   = { nil, nil, nil };
  // Per-ring-entry column buffers, like slots: an in-flight dispatch must
  // never see a later upload, so columns get a buffer per ring entry, not
  // a shared one.
  id<MTLBuffer>        ring_columns[kMaxPipelineDepth] = { nil, nil, nil };
  dispatch_semaphore_t ring_done[kMaxPipelineDepth]    = { nullptr, nullptr, nullptr };
  bool     primed       = false;
  uint32_t read_pos     = 0;
  uint64_t next_enqueue = 0;   // sample index of the next block to submit
};

static bool configure_pipeline_depth(uint32_t & depth, std::string & err)
{
  depth = 0;
  const char * pipe = getenv("TROPICAL_METAL_PIPELINE");
  if (pipe && pipe[0] == '1' && pipe[1] == '\0')
    depth = 3; // preserve the original opt-in behavior

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

// Submit block `start` into ring slot j (no wait). The completion handler
// signals ring_done[j]; Metal retains the buffers until completion, and ARC
// keeps the captured semaphore alive, so teardown mid-flight is safe (the
// semaphores start at 0 and only ever get signaled — never destroyed below
// their initial value).
static void enqueue_block(MetalKernel & k, uint32_t j,
                          const double * slots, uint32_t n_slots,
                          const float * columns, uint32_t n_columns,
                          double sample_rate, uint64_t start, uint32_t len)
{
  const uint32_t n = std::min<uint32_t>(n_slots, k.slot_count);
  float * sdst = static_cast<float *>(k.ring_slots[j].contents);
  for (uint32_t i = 0; i < n; ++i)
    sdst[i] = static_cast<float>(slots[i]);

  id<MTLCommandBuffer> cb = [shared_queue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
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
    if (columns && nc > 0)
      std::memcpy(k.ring_columns[j].contents, columns, nc * sizeof(float));
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
  [cb commit];
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
        if (!k->ring_out[j] || !k->ring_slots[j])
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
  if (len == 0 || len > k.capacity) return false;

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
      if (!k.primed || expected != start_sample_index)
      {
        if (k.primed)  // drain stale futures before re-priming
          for (uint32_t j = 0; j < k.depth; ++j)
            dispatch_semaphore_wait(k.ring_done[j], DISPATCH_TIME_FOREVER);
        k.next_enqueue = start_sample_index;
        k.read_pos = 0;
        for (uint32_t j = 0; j < k.depth; ++j)
        {
          enqueue_block(k, j, slots, n_slots, columns, n_columns,
                        sample_rate, k.next_enqueue, len);
          k.next_enqueue += len;
        }
        k.primed = true;
      }
      const uint32_t j = k.read_pos;
      dispatch_semaphore_wait(k.ring_done[j], DISPATCH_TIME_FOREVER);
      const float * src = static_cast<const float *>(k.ring_out[j].contents);
      for (uint32_t i = 0; i < len; ++i)
        out_f64[i] = static_cast<double>(src[i]);
      enqueue_block(k, j, slots, n_slots, columns, n_columns,
                    sample_rate, k.next_enqueue, len);
      k.next_enqueue += len;
      k.read_pos = (j + 1) % k.depth;
      return true;
    }
  }

  @autoreleasepool
  {
    // Slot snapshot: f64 host truth → f32 shared buffer, once per block.
    const uint32_t n = std::min<uint32_t>(n_slots, k.slot_count);
    float * sdst = static_cast<float *>(k.slots.contents);
    for (uint32_t i = 0; i < n; ++i)
      sdst[i] = static_cast<float>(slots[i]);

    id<MTLCommandBuffer> cb = [shared_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:k.pso];
    [enc setBuffer:k.out offset:0 atIndex:0];
    [enc setBuffer:k.slots offset:0 atIndex:1];
    KernelConsts consts{ start_sample_index,
                         static_cast<float>(sample_rate),
                         len };
    [enc setBytes:&consts length:sizeof(consts) atIndex:2];
    if (k.column_count > 0)
    {
      // Coefficient columns, copied at ENCODE next to the slot snapshot.
      // `columns` is the f32 image of the ONE generation this process()
      // call explicitly owns and revalidated, so
      // the copy inherits the whole-generation guarantee: the GPU reads
      // one consistent generation of columns, no cross-column tear on a
      // live knob move.
      const uint32_t nc = std::min<uint32_t>(n_columns, k.column_count);
      if (columns && nc > 0)
        std::memcpy(k.columns.contents, columns, nc * sizeof(float));
      [enc setBuffer:k.columns offset:0 atIndex:3];
    }
    const NSUInteger tg =
      std::min<NSUInteger>(k.pso.maxTotalThreadsPerThreadgroup, len);
    [enc dispatchThreads:MTLSizeMake(len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) return false;

    const float * src = static_cast<const float *>(k.out.contents);
    for (uint32_t i = 0; i < len; ++i)
      out_f64[i] = static_cast<double>(src[i]);
    return true;
  }
}

uint32_t pipeline_depth(const MetalKernel & k)
{
  return k.depth;
}

} // namespace tropical_metal
