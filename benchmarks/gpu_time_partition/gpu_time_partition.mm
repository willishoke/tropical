// gpu_time_partition.mm — GPU sibling of ../simd_time_partition.cpp
//
// The CPU benchmark showed: for a closed-form-in-τ kernel every output sample is an
// independent function of τ, so W consecutive samples fill W SIMD lanes at once. That
// tops out at NEON's 4 lanes. This benchmark takes the SAME time-partition axis to the
// GPU, where a realtime block of B samples becomes one parallel dispatch of B threads —
// one thread per time index, no cross-thread state, pure f(t, params).
//
// The only question that gates a realtime GPU backend on unified memory (UMA): does that
// per-block dispatch complete inside the audio deadline (B / sampleRate)? We measure the
// full round trip a realtime callback would pay — encode + commit + waitUntilCompleted —
// and report median / p99 / max against the deadline, with a single-thread CPU baseline
// for the crossover. Metal is reached at runtime (newLibraryWithSource); no .metal step.
//
// verdict:  PASS   p99 < deadline           (robustly realtime, jitter included)
//           p99>dl median < deadline ≤ p99  (fits typically; scheduling jitter overruns)
//           FAIL   median ≥ deadline

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

// CPU baseline: identical modal sum, one thread, one block. -O3 auto-vectorizes across n.
static float cpu_block(int B, int K, const float* fr, const float* de, const float* am,
                       float t0, float dt, float* out) {
  for (int n = 0; n < B; ++n) {
    float t = t0 + (float)n * dt, acc = 0.0f;
    for (int m = 0; m < K; ++m)
      acc += am[m] * std::exp(-de[m] * t) * std::sin(6.2831853071795864f * fr[m] * t);
    out[n] = acc;
  }
  return out[B - 1];  // defeat DCE
}

int main() {
  const double SR = 48000.0;
  const int blocks[] = {64, 128, 256, 512, 1024, 2048};
  const int Ks[] = {16, 64, 256};
  const int WARM = 64, ITERS = 2000;

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
    printf("metric: per-block round-trip (encode+commit+wait), %d iters after %d warm\n\n", ITERS, WARM);
    printf("%-4s %-5s %10s %10s %10s %10s %10s   %-7s\n", "K", "B", "gpu_med", "gpu_p99",
           "gpu_max", "cpu_med", "deadline", "verdict");

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uf(0.5f, 1.0f);

    for (int K : Ks) {
      std::vector<float> fr(K), de(K), am(K);
      for (int m = 0; m < K; ++m) { fr[m] = 110.f * (m + 1); de[m] = 0.5f + uf(rng); am[m] = uf(rng) / K; }
      id<MTLBuffer> fB = [dev newBufferWithBytes:fr.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> dB = [dev newBufferWithBytes:de.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> aB = [dev newBufferWithBytes:am.data() length:K * sizeof(float) options:MTLResourceStorageModeShared];

      for (int B : blocks) {
        id<MTLBuffer> oB = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
        float t0 = 0.f, dt = 1.f / (float)SR;
        uint Ku = (uint)K;
        NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)B);

        auto dispatch = [&] {
          @autoreleasepool {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:oB offset:0 atIndex:0];
            [e setBuffer:fB offset:0 atIndex:1];
            [e setBuffer:dB offset:0 atIndex:2];
            [e setBuffer:aB offset:0 atIndex:3];
            [e setBytes:&Ku length:sizeof(Ku) atIndex:4];
            [e setBytes:&t0 length:sizeof(t0) atIndex:5];
            [e setBytes:&dt length:sizeof(dt) atIndex:6];
            [e dispatchThreads:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
          }
        };

        for (int i = 0; i < WARM; ++i) dispatch();
        std::vector<double> lat;
        lat.reserve(ITERS);
        for (int i = 0; i < ITERS; ++i) { auto s = clk::now(); dispatch(); lat.push_back(us_since(s, clk::now())); }
        std::sort(lat.begin(), lat.end());

        std::vector<float> cout(B);
        std::vector<double> clat;
        clat.reserve(256);
        for (int i = 0; i < 32; ++i) cpu_block(B, K, fr.data(), de.data(), am.data(), 0, dt, cout.data());
        for (int i = 0; i < 256; ++i) {
          auto s = clk::now();
          volatile float sink = cpu_block(B, K, fr.data(), de.data(), am.data(), 0, dt, cout.data());
          (void)sink;
          clat.push_back(us_since(s, clk::now()));
        }
        std::sort(clat.begin(), clat.end());

        double dl = 1e6 * (double)B / SR;
        double gmed = pct(lat, 0.5), gp99 = pct(lat, 0.99), gmax = lat.back(), cmed = pct(clat, 0.5);
        const char* verdict = (gp99 < dl) ? "PASS" : (gmed < dl ? "p99>dl" : "FAIL");
        printf("%-4d %-5d %9.1fu %9.1fu %9.1fu %9.1fu %9.1fu   %-7s\n", K, B, gmed, gp99, gmax, cmed, dl, verdict);
      }
      printf("\n");
    }
  }
  return 0;
}
