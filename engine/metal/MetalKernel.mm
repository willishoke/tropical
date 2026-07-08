// MetalKernel — ObjC++ implementation. See MetalKernel.hpp for the contract
// and lean/Tropical/Ir/EmitMsl.lean for the kernel ABI this must match:
//
//   buffer(0) device float*                  output_buffer
//   buffer(1) constant float*                slots
//   buffer(2) constant TropicalKernelConsts& k
//   thread_position_in_grid                  = sample index
//
// with TropicalKernelConsts = { ulong start_sample_index; float sample_rate;
// uint buffer_length; } (16 bytes, no padding).

#include "MetalKernel.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
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

struct MetalKernel
{
  id<MTLComputePipelineState> pso    = nil;
  id<MTLBuffer>               out    = nil;  // f32 × buffer_length, shared
  id<MTLBuffer>               slots  = nil;  // f32 × slot_count, shared
  uint32_t capacity   = 0;
  uint32_t slot_count = 0;
  // f64→f32 staging (audio-thread use; preallocated — no alloc in process).
  std::vector<float> slot_staging;
};

MetalKernelPtr create(const std::string & msl_source,
                      uint32_t buffer_length,
                      uint32_t slot_count,
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
    k->pso        = pso;
    k->capacity   = buffer_length;
    k->slot_count = slot_count;
    k->out = [dev newBufferWithLength:(NSUInteger)buffer_length * sizeof(float)
                              options:MTLResourceStorageModeShared];
    k->slots = [dev newBufferWithLength:(NSUInteger)std::max<uint32_t>(slot_count, 1) * sizeof(float)
                                options:MTLResourceStorageModeShared];
    k->slot_staging.assign(slot_count, 0.0f);
    if (!k->out || !k->slots) { err = "MetalKernel: buffer allocation failed"; return nullptr; }
    return k;
  }
}

bool process_block(MetalKernel & k,
                   const double * slots, uint32_t n_slots,
                   double sample_rate, uint64_t start_sample_index,
                   double * out_f64, uint32_t len)
{
  if (len == 0 || len > k.capacity) return false;

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

} // namespace tropical_metal
