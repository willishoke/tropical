// gpu_time_partition.mm — GPU sibling of ../simd_time_partition.cpp
//
// The CPU benchmark showed: for a closed-form-in-τ kernel every output sample is an
// independent function of τ, so W consecutive samples fill W SIMD lanes at once. That
// tops out at NEON's 4 lanes. This benchmark takes the SAME time-partition axis to the
// GPU, where a realtime block of B samples becomes one parallel dispatch of B threads —
// one thread per time index, no cross-thread state, pure f(t, params).
//
// The only question that gates a realtime GPU backend on unified memory (UMA): does that
// per-block dispatch complete inside the audio deadline (B / sampleRate)? We measure three
// submission strategies, because the naive one sets a misleading floor:
//
//   sync   commit + waitUntilCompleted            single-shot round trip, CPU sleeps
//   spin   commit + busy-wait on cb.status        single-shot round trip, no sleep/wake
//   pipe   ring of D buffers + async completion    critical-path cost (pipe_call) and
//          handler signalling a semaphore          sustained throughput (pipe_sust)
//
//   pipe_call = encode+commit only (GPU overlaps; what an audio callback actually pays)
//   pipe_sust = wall/ITERS with the pipeline full (GPU-bound sustained block interval)
//
// rt column: PASS if the sustained interval clears the deadline.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
static double pct(const std::vector<double>& v, double p) {
  if (v.empty()) return 0.0;
  return v[(size_t)(p * (v.size() - 1))];
}

static const char* kShaderSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;
// One thread per time index n. t = t0 + n*dt is this sample's time coordinate — the
// parallel axis. Modal residue sum: K damped sines. No dependence on any other sample.
kernel void modal(device   float* out    [[buffer(0)]],
                  constant float* freqs  [[buffer(1)]],
                  constant float* decays [[buffer(2)]],
                  constant float* amps   [[buffer(3)]],
                  constant uint&  K      [[buffer(4)]],
                  constant float& t0     [[buffer(5)]],
                  constant float& dt     [[buffer(6)]],
                  uint            n      [[thread_position_in_grid]]) {
    float t = t0 + float(n) * dt;
    float acc = 0.0f;
    for (uint m = 0; m < K; ++m)
        acc += amps[m] * exp(-decays[m] * t) * sin(6.2831853071795864f * freqs[m] * t);
    out[n] = acc;
}
)METAL";

int main() {
  const double SR = 48000.0;
  const int blocks[] = {64, 128, 256, 512, 1024, 2048};
  const int Ks[] = {16, 64, 256};
  const int WARM = 64, ITERS = 2000, D = 3;  // D = pipeline depth

  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc] options:nil error:&err];
    if (!lib) { fprintf(stderr, "shader compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"modal"] error:&err];
    if (!pso) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }

    printf("device: %s   maxThreads/tg: %lu   UMA: %s\n", dev.name.UTF8String,
           (unsigned long)pso.maxTotalThreadsPerThreadgroup, dev.hasUnifiedMemory ? "yes" : "no");
    printf("kernel: modal residue sum, K damped sines/sample   sampleRate: %.0f Hz\n", SR);
    printf("all times us. sync/spin = single-shot round trip; pipe_call = encode+commit "
           "(GPU overlapped); pipe_sust = sustained block interval.\n\n");
    printf("%-4s %-5s %9s %9s %9s %9s %9s %10s   %-4s\n", "K", "B", "sync_med", "spin_med",
           "spin_p99", "pipe_call", "pipe_sust", "deadline", "rt");

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uf(0.5f, 1.0f);

    for (int K : Ks) {
      std::vector<float> fr(K), de(K), am(K);
      for (int m = 0; m < K; ++m) { fr[m] = 110.f * (m + 1); de[m] = 0.5f + uf(rng); am[m] = uf(rng) / K; }
      id<MTLBuffer> fB = [dev newBufferWithBytes:fr.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> dB = [dev newBufferWithBytes:de.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> aB = [dev newBufferWithBytes:am.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];

      for (int B : blocks) {
        float t0 = 0.f, dt = 1.f / (float)SR;
        uint Ku = (uint)K;
        NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)B);

        // Encode the (identical) modal dispatch writing into outBuf; return the committed-ready cb.
        auto encode = [&](id<MTLBuffer> outBuf) -> id<MTLCommandBuffer> {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pso];
          [e setBuffer:outBuf offset:0 atIndex:0];
          [e setBuffer:fB offset:0 atIndex:1];
          [e setBuffer:dB offset:0 atIndex:2];
          [e setBuffer:aB offset:0 atIndex:3];
          [e setBytes:&Ku length:sizeof(Ku) atIndex:4];
          [e setBytes:&t0 length:sizeof(t0) atIndex:5];
          [e setBytes:&dt length:sizeof(dt) atIndex:6];
          [e dispatchThreads:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
          [e endEncoding];
          return cb;
        };

        id<MTLBuffer> oB = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];

        // ---- sync: commit + waitUntilCompleted (CPU sleeps until done) ----
        std::vector<double> sync_lat;
        sync_lat.reserve(ITERS);
        for (int i = 0; i < WARM + ITERS; ++i) {
          @autoreleasepool {
            auto s = clk::now();
            id<MTLCommandBuffer> cb = encode(oB);
            [cb commit];
            [cb waitUntilCompleted];
            if (i >= WARM) sync_lat.push_back(us_since(s, clk::now()));
          }
        }
        std::sort(sync_lat.begin(), sync_lat.end());

        // ---- spin: commit + busy-wait on status (no sleep/wake) ----
        std::vector<double> spin_lat;
        spin_lat.reserve(ITERS);
        for (int i = 0; i < WARM + ITERS; ++i) {
          @autoreleasepool {
            auto s = clk::now();
            id<MTLCommandBuffer> cb = encode(oB);
            [cb commit];
            while (cb.status < MTLCommandBufferStatusCompleted) { /* spin */ }
            if (i >= WARM) spin_lat.push_back(us_since(s, clk::now()));
          }
        }
        std::sort(spin_lat.begin(), spin_lat.end());

        // ---- pipe: ring of D buffers, async completion handler signals a semaphore ----
        std::vector<id<MTLBuffer>> ring(D);
        for (int k = 0; k < D; ++k)
          ring[k] = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
        dispatch_semaphore_t sem = dispatch_semaphore_create(D);
        for (int i = 0; i < WARM; ++i) {
          @autoreleasepool {
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            id<MTLCommandBuffer> cb = encode(ring[i % D]);
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(sem); }];
            [cb commit];
          }
        }
        for (int k = 0; k < D; ++k) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);  // quiesce
        for (int k = 0; k < D; ++k) dispatch_semaphore_signal(sem);                       // reset to D

        std::vector<double> call;
        call.reserve(ITERS);
        auto pipe_start = clk::now();
        for (int i = 0; i < ITERS; ++i) {
          @autoreleasepool {
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);  // backpressure — not in call cost
            auto s = clk::now();
            id<MTLCommandBuffer> cb = encode(ring[i % D]);
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(sem); }];
            [cb commit];
            call.push_back(us_since(s, clk::now()));  // encode+commit only; GPU work overlaps
          }
        }
        double pipe_total = us_since(pipe_start, clk::now());
        for (int k = 0; k < D; ++k) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);    // drain in-flight
        for (int k = 0; k < D; ++k) dispatch_semaphore_signal(sem);                         // restore to D before release
        std::sort(call.begin(), call.end());

        double dl = 1e6 * (double)B / SR;
        double pipe_call = pct(call, 0.5), pipe_sust = pipe_total / ITERS;
        const char* rt = (pipe_sust < dl) ? "PASS" : "FAIL";
        printf("%-4d %-5d %8.1fu %8.1fu %8.1fu %8.1fu %8.1fu %9.1fu   %-4s\n", K, B,
               pct(sync_lat, 0.5), pct(spin_lat, 0.5), pct(spin_lat, 0.99), pipe_call, pipe_sust, dl, rt);
      }
      printf("\n");
    }
  }
  return 0;
}
