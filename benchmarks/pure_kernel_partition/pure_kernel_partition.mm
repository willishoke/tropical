// pure_kernel_partition.mm
//
// Proof that a pure Tropical graph can cross Metal's per-kernel threadgroup-memory ceiling
// without a CPU handoff. Two independent modal branches and a final mix are encoded into one
// command buffer. Device-private intermediate buffers keep the whole chain on the GPU.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

constexpr uint32_t kWorkspaceFloats = 5600;
constexpr uint32_t kThreadsPerGroup = 128;
constexpr uint32_t kWarmups = 16;
constexpr uint32_t kIterations = 300;
constexpr double kSampleRate = 44100.0;

struct KernelConstants {
  uint32_t frames;
  uint32_t start_sample;
  float sample_rate;
  float pad;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  const size_t index = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

const char* kValidSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint kWorkspaceFloats = 5600;
constant uint kThreadsPerGroup = 128;

struct KernelConstants {
  uint frames;
  uint start_sample;
  float sample_rate;
  float pad;
};

inline float modal_term(uint index, float time, float detune) {
  const float mode = float(index + 1);
  const float frequency = (34.0f + 0.177f * mode) * detune;
  const float decay = 0.08f + 0.000031f * mode;
  const float amplitude = 1.0f / (1.0f + 0.0025f * mode);
  return amplitude * exp(-decay * time) * sin(6.28318530718f * frequency * time);
}

inline void fill_branch(threadgroup float* workspace,
                        uint lane,
                        float time,
                        float detune) {
  for (uint index = lane; index < kWorkspaceFloats; index += kThreadsPerGroup) {
    workspace[index] = modal_term(index, time, detune);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float reduce_branch(threadgroup float* workspace, uint lane) {
  float partial = 0.0f;
  for (uint index = lane; index < kWorkspaceFloats; index += kThreadsPerGroup) {
    partial += workspace[index];
  }
  workspace[lane] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = kThreadsPerGroup / 2; stride > 0; stride >>= 1) {
    if (lane < stride) workspace[lane] += workspace[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return workspace[0] / float(kWorkspaceFloats);
}

kernel void branch_a(device float* output [[buffer(0)]],
                     constant KernelConstants& constants [[buffer(1)]],
                     uint sample [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
  if (sample >= constants.frames) return;
  threadgroup float workspace[kWorkspaceFloats];
  const float time = float(constants.start_sample + sample) / constants.sample_rate;
  fill_branch(workspace, lane, time, 0.997f);
  const float value = reduce_branch(workspace, lane);
  if (lane == 0) output[sample] = value;
}

kernel void branch_b(device float* output [[buffer(0)]],
                     constant KernelConstants& constants [[buffer(1)]],
                     uint sample [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
  if (sample >= constants.frames) return;
  threadgroup float workspace[kWorkspaceFloats];
  const float time = float(constants.start_sample + sample) / constants.sample_rate;
  fill_branch(workspace, lane, time, 1.003f);
  const float value = reduce_branch(workspace, lane);
  if (lane == 0) output[sample] = value;
}

kernel void mix_branches(device float* output [[buffer(0)]],
                         device const float* branch_a_output [[buffer(1)]],
                         device const float* branch_b_output [[buffer(2)]],
                         constant KernelConstants& constants [[buffer(3)]],
                         uint sample [[thread_position_in_grid]]) {
  if (sample < constants.frames) {
    output[sample] = branch_a_output[sample] + branch_b_output[sample];
  }
}

kernel void fused_reuse(device float* output [[buffer(0)]],
                        constant KernelConstants& constants [[buffer(1)]],
                        uint sample [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
  if (sample >= constants.frames) return;
  threadgroup float workspace[kWorkspaceFloats];
  threadgroup float first_branch;
  const float time = float(constants.start_sample + sample) / constants.sample_rate;

  fill_branch(workspace, lane, time, 0.997f);
  const float first = reduce_branch(workspace, lane);
  if (lane == 0) first_branch = first;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  fill_branch(workspace, lane, time, 1.003f);
  const float second = reduce_branch(workspace, lane);
  if (lane == 0) output[sample] = first_branch + second;
}
)METAL";

// Both arrays remain live until both fills finish. On a 32 KiB device the pipeline must be
// rejected: its 44,800-byte static threadgroup allocation cannot be scheduled.
const char* kOversizedSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void monolithic_overlapping(device float* output [[buffer(0)]],
                                   uint sample [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float first[5600];
  threadgroup float second[5600];
  for (uint index = lane; index < 5600; index += 128) {
    first[index] = sin(float(index + sample) * 0.001f);
    second[index] = cos(float(index + sample) * 0.001f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) output[sample] = first[sample % 5600] + second[sample % 5600];
}
)METAL";

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          NSString* name,
                                          NSError** error) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) {
    if (error) {
      *error = [NSError errorWithDomain:@"pure-kernel-partition"
                                   code:1
                               userInfo:@{NSLocalizedDescriptionKey:
                                              [NSString stringWithFormat:@"missing %@", name]}];
    }
    return nil;
  }
  return [device newComputePipelineStateWithFunction:function error:error];
}

void dispatch_branch(id<MTLCommandBuffer> command_buffer,
                     id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> output,
                     const KernelConstants& constants) {
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder setBytes:&constants length:sizeof(constants) atIndex:1];
  [encoder dispatchThreadgroups:MTLSizeMake(constants.frames, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
  [encoder endEncoding];
}

void dispatch_mix(id<MTLCommandBuffer> command_buffer,
                  id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> output,
                  id<MTLBuffer> first,
                  id<MTLBuffer> second,
                  const KernelConstants& constants) {
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:0 atIndex:0];
  [encoder setBuffer:first offset:0 atIndex:1];
  [encoder setBuffer:second offset:0 atIndex:2];
  [encoder setBytes:&constants length:sizeof(constants) atIndex:3];
  const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(constants.frames, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
}

struct Timings {
  double median;
  double p95;
  double p99;
  double maximum;
};

template <typename Encode>
Timings measure(id<MTLCommandQueue> queue, Encode encode) {
  std::vector<double> samples;
  samples.reserve(kIterations);
  for (uint32_t iteration = 0; iteration < kWarmups + kIterations; ++iteration) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      const auto start = Clock::now();
      encode(command_buffer);
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        std::fprintf(stderr, "command buffer failed: %s\n",
                     command_buffer.error.localizedDescription.UTF8String);
        std::exit(2);
      }
      if (iteration >= kWarmups) samples.push_back(elapsed_ms(start, Clock::now()));
    }
  }
  std::sort(samples.begin(), samples.end());
  return Timings{percentile(samples, 0.50), percentile(samples, 0.95),
                 percentile(samples, 0.99), samples.back()};
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      std::fprintf(stderr, "failed to create Metal command queue\n");
      return 1;
    }

    MTLCompileOptions* options = [MTLCompileOptions new];
    // Match the production backend's numerical contract: no fast-math
    // reassociation or FMA contraction.
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kValidSource]
                            options:options
                              error:&error];
    if (!library) {
      std::fprintf(stderr, "valid shader compile failed: %s\n",
                   error.localizedDescription.UTF8String);
      return 1;
    }

    id<MTLComputePipelineState> branch_a = make_pipeline(device, library, @"branch_a", &error);
    id<MTLComputePipelineState> branch_b = make_pipeline(device, library, @"branch_b", &error);
    id<MTLComputePipelineState> mix = make_pipeline(device, library, @"mix_branches", &error);
    id<MTLComputePipelineState> fused = make_pipeline(device, library, @"fused_reuse", &error);
    if (!branch_a || !branch_b || !mix || !fused) {
      std::fprintf(stderr, "valid pipeline creation failed: %s\n",
                   error.localizedDescription.UTF8String);
      return 1;
    }

    bool oversized_rejected = false;
    std::string oversized_error;
    error = nil;
    id<MTLLibrary> oversized_library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kOversizedSource]
                            options:options
                              error:&error];
    if (!oversized_library) {
      oversized_rejected = true;
      oversized_error = error.localizedDescription.UTF8String;
    } else {
      id<MTLComputePipelineState> oversized =
          make_pipeline(device, oversized_library, @"monolithic_overlapping", &error);
      oversized_rejected = oversized == nil;
      if (error) oversized_error = error.localizedDescription.UTF8String;
    }

    std::printf("device: %s\n", device.name.UTF8String);
    std::printf("unified memory: %s\n", device.hasUnifiedMemory ? "yes" : "no");
    std::printf("device threadgroup limit: %lu bytes\n",
                static_cast<unsigned long>(device.maxThreadgroupMemoryLength));
    std::printf("branch static threadgroup memory: %lu bytes\n",
                static_cast<unsigned long>(branch_a.staticThreadgroupMemoryLength));
    std::printf("fused-reuse static threadgroup memory: %lu bytes\n",
                static_cast<unsigned long>(fused.staticThreadgroupMemoryLength));
    std::printf("overlapping monolith request: %u bytes; rejected: %s\n",
                2 * kWorkspaceFloats * static_cast<uint32_t>(sizeof(float)),
                oversized_rejected ? "yes" : "NO");
    if (!oversized_error.empty()) std::printf("rejection: %s\n", oversized_error.c_str());
    if (!oversized_rejected && device.maxThreadgroupMemoryLength <
                                   2 * kWorkspaceFloats * sizeof(float)) {
      std::fprintf(stderr, "oversized pipeline unexpectedly compiled\n");
      return 2;
    }

    std::printf("\n%-6s %10s %10s %10s %10s %10s %9s\n", "frames", "path", "median_ms",
                "p95_ms", "p99_ms", "max_ms", "deadline");

    for (const uint32_t frames : {128u, 256u, 512u}) {
      KernelConstants constants{frames, 0, static_cast<float>(kSampleRate), 0.0f};
      const NSUInteger byte_count = frames * sizeof(float);
      id<MTLBuffer> intermediate_a =
          [device newBufferWithLength:byte_count options:MTLResourceStorageModePrivate];
      id<MTLBuffer> intermediate_b =
          [device newBufferWithLength:byte_count options:MTLResourceStorageModePrivate];
      id<MTLBuffer> partitioned_output =
          [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
      id<MTLBuffer> fused_output =
          [device newBufferWithLength:byte_count options:MTLResourceStorageModeShared];

      const auto encode_partitioned = [&](id<MTLCommandBuffer> command_buffer) {
        dispatch_branch(command_buffer, branch_a, intermediate_a, constants);
        dispatch_branch(command_buffer, branch_b, intermediate_b, constants);
        dispatch_mix(command_buffer, mix, partitioned_output, intermediate_a, intermediate_b,
                     constants);
      };
      const auto encode_fused = [&](id<MTLCommandBuffer> command_buffer) {
        dispatch_branch(command_buffer, fused, fused_output, constants);
      };

      // Correctness check before timing. Separate encoders in one command buffer establish
      // the producer/consumer ordering for the device-private intermediates.
      id<MTLCommandBuffer> partitioned_check = [queue commandBuffer];
      encode_partitioned(partitioned_check);
      [partitioned_check commit];
      [partitioned_check waitUntilCompleted];
      id<MTLCommandBuffer> fused_check = [queue commandBuffer];
      encode_fused(fused_check);
      [fused_check commit];
      [fused_check waitUntilCompleted];
      if (partitioned_check.status == MTLCommandBufferStatusError ||
          fused_check.status == MTLCommandBufferStatusError) {
        std::fprintf(stderr, "correctness dispatch failed\n");
        return 2;
      }

      const float* partitioned_values =
          static_cast<const float*>(partitioned_output.contents);
      const float* fused_values = static_cast<const float*>(fused_output.contents);
      float max_abs_error = 0.0f;
      for (uint32_t sample = 0; sample < frames; ++sample) {
        max_abs_error =
            std::max(max_abs_error, std::abs(partitioned_values[sample] - fused_values[sample]));
      }
      if (!std::isfinite(max_abs_error) || max_abs_error > 1.0e-5f) {
        std::fprintf(stderr, "partition correctness failed at %u frames: max abs error %.9g\n",
                     frames, max_abs_error);
        return 2;
      }

      const Timings fused_times = measure(queue, encode_fused);
      const Timings partitioned_times = measure(queue, encode_partitioned);
      const double deadline_ms = 1000.0 * frames / kSampleRate;
      std::printf("%-6u %10s %10.3f %10.3f %10.3f %10.3f %9.3f\n", frames,
                  "fused", fused_times.median, fused_times.p95, fused_times.p99,
                  fused_times.maximum, deadline_ms);
      std::printf("%-6u %10s %10.3f %10.3f %10.3f %10.3f %9.3f  error=%.3g\n", frames,
                  "partition", partitioned_times.median, partitioned_times.p95,
                  partitioned_times.p99, partitioned_times.maximum, deadline_ms,
                  max_abs_error);
    }
  }
  return 0;
}
