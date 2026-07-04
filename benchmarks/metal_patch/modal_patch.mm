// modal_patch.mm — a real tropical patch (ModalHeavy64) realized on Metal in f32,
// scored against tropical's own f64 render (mh64_ref.bin) for fitness, and benchmarked
// for realtime latency vs a CPU-f64 baseline of the same patch.
//
// The patch: out(t) = gain · Σ_i w_i · sin(2π · freq_i · t),  t = sampleIndex / sampleRate.
// 64 partials — authored in tropical_program_2, compiled by `diffcli compile`, rendered by
// the real engine. modal_params.h is generated from that patch's coefficients.
//
// Fitness criterion (its own goldens, NOT bit-exactness): SNR / max-abs / correlation of
// the f32 GPU output vs the f64 CPU reference over the reference window.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include "modal_params.h"

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
static double pct(const std::vector<double>& v, double p) { return v[(size_t)(p * (v.size() - 1))]; }

static const char* kShaderSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;
// One thread per time index. t = (start + n)/SR — the parallel axis. Modal sum, f32.
kernel void modal_patch(device   float* out   [[buffer(0)]],
                        constant float* freq  [[buffer(1)]],
                        constant float* w     [[buffer(2)]],
                        constant uint&  N     [[buffer(3)]],
                        constant float& gain  [[buffer(4)]],
                        constant float& invSR [[buffer(5)]],
                        constant uint&  start [[buffer(6)]],
                        uint            n     [[thread_position_in_grid]]) {
    float t = float(start + n) * invSR;
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i)
        acc += w[i] * sin(6.2831853071795864f * freq[i] * t);
    out[n] = gain * acc;
}
)METAL";

// CPU f64 baseline of the identical patch (the reference realization, and the perf baseline).
static double cpu_block(int B, uint start, double invSR, double* out) {
  for (int n = 0; n < B; ++n) {
    double t = (double)(start + n) * invSR, acc = 0.0;
    for (int i = 0; i < MODAL_N; ++i)
      acc += MODAL_W_D[i] * std::sin(6.2831853071795864 * MODAL_FREQ_D[i] * t);
    out[n] = MODAL_GAIN * acc;
  }
  return out[B - 1];
}

int main() {
  const double SR = MODAL_SR;
  const float invSR = (float)(1.0 / SR);

  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc] options:nil error:&err];
    if (!lib) { fprintf(stderr, "shader: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"modal_patch"] error:&err];
    if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }

    uint N = MODAL_N;
    float gain = MODAL_GAIN;
    id<MTLBuffer> fB = [dev newBufferWithBytes:MODAL_FREQ length:N * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> wB = [dev newBufferWithBytes:MODAL_W    length:N * sizeof(float) options:MTLResourceStorageModeShared];

    printf("device: %s   patch: ModalHeavy%u (%u partials, fmax %.0f Hz)   SR %.0f\n",
           dev.name.UTF8String, N, N, MODAL_FREQ[N - 1], SR);

    auto encode = [&](id<MTLBuffer> oB, int B, uint start) -> id<MTLCommandBuffer> {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pso];
      [e setBuffer:oB offset:0 atIndex:0];
      [e setBuffer:fB offset:0 atIndex:1];
      [e setBuffer:wB offset:0 atIndex:2];
      [e setBytes:&N length:sizeof(N) atIndex:3];
      [e setBytes:&gain length:sizeof(gain) atIndex:4];
      [e setBytes:&invSR length:sizeof(invSR) atIndex:5];
      [e setBytes:&start length:sizeof(start) atIndex:6];
      NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)B);
      [e dispatchThreads:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [e endEncoding];
      return cb;
    };

    // ---------------- FITNESS: GPU f32 vs tropical f64 reference ----------------
    FILE* rf = fopen("mh64_ref.bin", "rb");
    if (!rf) { fprintf(stderr, "missing mh64_ref.bin (render it first)\n"); return 1; }
    std::vector<double> ref;
    { double d; while (fread(&d, sizeof(double), 1, rf) == 1) ref.push_back(d); }
    fclose(rf);
    int total = (int)ref.size();
    const int RB = 256;  // reference was rendered in 256-frame blocks
    std::vector<float> gpu(total);
    id<MTLBuffer> oB = [dev newBufferWithLength:RB * sizeof(float) options:MTLResourceStorageModeShared];
    for (int s = 0; s < total; s += RB) {
      int B = std::min(RB, total - s);
      @autoreleasepool {
        id<MTLCommandBuffer> cb = encode(oB, B, (uint)s);
        [cb commit];
        [cb waitUntilCompleted];
      }
      const float* o = (const float*)oB.contents;
      for (int n = 0; n < B; ++n) gpu[s + n] = o[n];
    }
    double sigE = 0, errE = 0, maxAbs = 0, sxy = 0, sxx = 0, syy = 0, mx = 0, my = 0;
    for (int i = 0; i < total; ++i) { mx += ref[i]; my += gpu[i]; }
    mx /= total; my /= total;
    for (int i = 0; i < total; ++i) {
      double r = ref[i], g = gpu[i], e = r - g;
      sigE += r * r; errE += e * e; maxAbs = std::max(maxAbs, std::abs(e));
      sxy += (r - mx) * (g - my); sxx += (r - mx) * (r - mx); syy += (g - my) * (g - my);
    }
    double snr = 10.0 * std::log10(sigE / errE);
    double rmsErr = std::sqrt(errE / total), rmsSig = std::sqrt(sigE / total);
    double corr = sxy / std::sqrt(sxx * syy);
    double bits = -std::log2(rmsErr / rmsSig);
    printf("\nFITNESS vs tropical f64 render  (%d samples, %.3f s)\n", total, total / SR);
    printf("  peak signal   %.5f\n", rmsSig * std::sqrt(2.0));
    printf("  max abs err   %.3e\n", maxAbs);
    printf("  rms err       %.3e   (signal rms %.3e)\n", rmsErr, rmsSig);
    printf("  SNR           %.1f dB   (~%.1f effective bits)\n", snr, bits);
    printf("  correlation   %.6f\n", corr);
    auto winSNR = [&](int lo, int hi) {
      double s = 0, e = 0;
      for (int i = lo; i < hi; ++i) { double r = ref[i], er = r - gpu[i]; s += r * r; e += er * er; }
      return 10.0 * std::log10(s / e);
    };
    if (total >= 8192) {
      int W = 4096;
      printf("  SNR by window first %.1f dB @ t=0   last %.1f dB @ t=%.1fs   (f32 phase drift)\n",
             winSNR(0, W), winSNR(total - W, total), (total - W) / SR);
    }

    // ---------------- PERFORMANCE: latency vs deadline & CPU f64 ----------------
    const int blocks[] = {64, 128, 256, 512, 1024};
    const int WARM = 64, ITERS = 2000, D = 3;
    printf("\nPERFORMANCE  (all us; sync=commit+wait, pipe_call=encode+commit, pipe_sust=block interval)\n");
    printf("%-6s %9s %9s %9s %10s %9s   %-6s %-5s\n", "B", "sync_med", "pipe_call", "pipe_sust",
           "cpu_f64", "deadline", "rt", "spdup");
    for (int B : blocks) {
      id<MTLBuffer> ob = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
      std::vector<double> s_lat;
      s_lat.reserve(ITERS);
      for (int i = 0; i < WARM + ITERS; ++i)
        @autoreleasepool {
          auto s = clk::now();
          id<MTLCommandBuffer> cb = encode(ob, B, 0);
          [cb commit];
          [cb waitUntilCompleted];
          if (i >= WARM) s_lat.push_back(us_since(s, clk::now()));
        }
      std::sort(s_lat.begin(), s_lat.end());

      std::vector<id<MTLBuffer>> ring(D);
      for (int k = 0; k < D; ++k) ring[k] = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
      dispatch_semaphore_t sem = dispatch_semaphore_create(D);
      for (int i = 0; i < WARM; ++i)
        @autoreleasepool {
          dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
          id<MTLCommandBuffer> cb = encode(ring[i % D], B, 0);
          [cb addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(sem); }];
          [cb commit];
        }
      for (int k = 0; k < D; ++k) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
      for (int k = 0; k < D; ++k) dispatch_semaphore_signal(sem);
      std::vector<double> call;
      call.reserve(ITERS);
      auto p0 = clk::now();
      for (int i = 0; i < ITERS; ++i)
        @autoreleasepool {
          dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
          auto s = clk::now();
          id<MTLCommandBuffer> cb = encode(ring[i % D], B, 0);
          [cb addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(sem); }];
          [cb commit];
          call.push_back(us_since(s, clk::now()));
        }
      double pipe_sust = us_since(p0, clk::now()) / ITERS;
      for (int k = 0; k < D; ++k) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
      for (int k = 0; k < D; ++k) dispatch_semaphore_signal(sem);
      std::sort(call.begin(), call.end());

      std::vector<double> co(B), clat;
      clat.reserve(256);
      for (int i = 0; i < 32; ++i) cpu_block(B, 0, 1.0 / SR, co.data());
      for (int i = 0; i < 256; ++i) {
        auto s = clk::now();
        volatile double sink = cpu_block(B, 0, 1.0 / SR, co.data());
        (void)sink;
        clat.push_back(us_since(s, clk::now()));
      }
      std::sort(clat.begin(), clat.end());

      double dl = 1e6 * (double)B / SR, cpu = pct(clat, 0.5);
      const char* rt = (pipe_sust < dl) ? "PASS" : "FAIL";
      printf("%-6d %8.1fu %8.1fu %8.1fu %9.1fu %8.1fu   %-6s %.2fx\n", B, pct(s_lat, 0.5),
             pct(call, 0.5), pipe_sust, cpu, dl, rt, cpu / pipe_sust);
    }
  }
  return 0;
}
