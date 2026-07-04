// modal_patch.mm — a real tropical patch (ModalHeavy64) realized on Metal, scored against
// tropical's own f64 render (mh64_ref.bin) for fitness, and benchmarked for realtime latency.
//
// Two phase substrates, same patch, same f32 value arithmetic:
//   naive : t = (start+n)/SR in f32, sin(2π·freq·t)          — phase on the line (f32)
//   fixed : φ = (nabs · incr) mod 2^64, sin(2π·φ/2^64)        — phase on the circle (fixed point)
//           incr = round((freq/SR)·2^64); φ is a pure function of the integer sample index
//           (stateless, no accumulator), reduced to a bounded [0,1) f32 only at the sine leaf.
//
// The fixed path is the "time is a different object than value" thesis: keep the monotone
// unbounded coordinate in a uniform-precision integer, drop to f32 only for the bounded value.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
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

kernel void modal_naive(device   float* out   [[buffer(0)]],
                        constant float* w     [[buffer(1)]],
                        constant float* freq  [[buffer(2)]],
                        constant uint&  N     [[buffer(3)]],
                        constant float& gain  [[buffer(4)]],
                        constant float& invSR [[buffer(5)]],
                        constant uint&  start [[buffer(6)]],
                        uint            n     [[thread_position_in_grid]]) {
    float t = float(start + n) * invSR;                 // phase argument on the line, f32
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i)
        acc += w[i] * sin(6.2831853071795864f * freq[i] * t);
    out[n] = gain * acc;
}

kernel void modal_fixed(device   float* out   [[buffer(0)]],
                        constant float* w     [[buffer(1)]],
                        constant ulong* incr  [[buffer(2)]],
                        constant uint&  N     [[buffer(3)]],
                        constant float& gain  [[buffer(4)]],
                        constant uint&  start [[buffer(6)]],
                        uint            n     [[thread_position_in_grid]]) {
    ulong nabs = (ulong)(start + n);
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
        ulong ph = nabs * incr[i];                      // (nabs·incr) mod 2^64 — exact, uniform
        float phf = float(uint(ph >> 40)) * (1.0f / 16777216.0f);  // top 24 bits -> [0,1), f32
        acc += w[i] * sin(6.2831853071795864f * phf);
    }
    out[n] = gain * acc;
}
)METAL";

static double cpu_block(int B, uint start, double invSR, double* out) {
  for (int n = 0; n < B; ++n) {
    double t = (double)(start + n) * invSR, acc = 0.0;
    for (int i = 0; i < MODAL_N; ++i)
      acc += MODAL_W_D[i] * std::sin(6.2831853071795864 * MODAL_FREQ_D[i] * t);
    out[n] = MODAL_GAIN * acc;
  }
  return out[B - 1];
}

struct Metrics { double snr, maxAbs, corr, snrFirst, snrLast, tLast; };
static Metrics score(const std::vector<double>& ref, const std::vector<float>& g, double SR) {
  int T = (int)ref.size();
  double sig = 0, e = 0, ma = 0, sxy = 0, sxx = 0, syy = 0, mx = 0, my = 0;
  for (int i = 0; i < T; ++i) { mx += ref[i]; my += g[i]; }
  mx /= T; my /= T;
  for (int i = 0; i < T; ++i) {
    double r = ref[i], er = r - g[i];
    sig += r * r; e += er * er; ma = std::max(ma, std::abs(er));
    sxy += (r - mx) * (g[i] - my); sxx += (r - mx) * (r - mx); syy += (g[i] - my) * (g[i] - my);
  }
  auto win = [&](int lo, int hi) {
    double s = 0, ee = 0;
    for (int i = lo; i < hi; ++i) { double r = ref[i], er = r - g[i]; s += r * r; ee += er * er; }
    return 10.0 * std::log10(s / ee);
  };
  int W = std::min(4096, T);
  return {10.0 * std::log10(sig / e), ma, sxy / std::sqrt(sxx * syy),
          win(0, W), win(T - W, T), (T - W) / SR};
}

int main() {
  const double SR = MODAL_SR;
  const float invSR = (float)(1.0 / SR);
  uint N = MODAL_N;
  float gain = MODAL_GAIN;

  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc] options:nil error:&err];
    if (!lib) { fprintf(stderr, "shader: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> naive =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"modal_naive"] error:&err];
    id<MTLComputePipelineState> fixed =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"modal_fixed"] error:&err];
    if (!naive || !fixed) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }

    // host: freq buffer (naive) + fixed-point increment buffer (fixed)
    id<MTLBuffer> wB = [dev newBufferWithBytes:MODAL_W length:N * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fB = [dev newBufferWithBytes:MODAL_FREQ length:N * sizeof(float) options:MTLResourceStorageModeShared];
    std::vector<uint64_t> incr(N);
    for (uint i = 0; i < N; ++i) {
      double cps = MODAL_FREQ_D[i] / SR;
      cps -= std::floor(cps);
      incr[i] = (uint64_t)(cps * 18446744073709551616.0);  // round((freq/SR)·2^64)
    }
    id<MTLBuffer> iB = [dev newBufferWithBytes:incr.data() length:N * sizeof(uint64_t) options:MTLResourceStorageModeShared];

    printf("device: %s   patch: ModalHeavy%u (%u partials, fmax %.0f Hz)   SR %.0f\n",
           dev.name.UTF8String, N, N, MODAL_FREQ[N - 1], SR);

    auto encode = [&](id<MTLComputePipelineState> pso, bool fx, id<MTLBuffer> oB, int B, uint start) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pso];
      [e setBuffer:oB offset:0 atIndex:0];
      [e setBuffer:wB offset:0 atIndex:1];
      [e setBuffer:(fx ? iB : fB) offset:0 atIndex:2];
      [e setBytes:&N length:sizeof(N) atIndex:3];
      [e setBytes:&gain length:sizeof(gain) atIndex:4];
      if (!fx) [e setBytes:&invSR length:sizeof(invSR) atIndex:5];
      [e setBytes:&start length:sizeof(start) atIndex:6];
      NSUInteger tg = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)B);
      [e dispatchThreads:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [e endEncoding];
      return cb;
    };

    // ---------------- FITNESS: both phase substrates vs tropical f64 ----------------
    FILE* rf = fopen("mh64_ref.bin", "rb");
    if (!rf) { fprintf(stderr, "missing mh64_ref.bin\n"); return 1; }
    std::vector<double> ref;
    { double d; while (fread(&d, sizeof(double), 1, rf) == 1) ref.push_back(d); }
    fclose(rf);
    int total = (int)ref.size();
    const int RB = 256;
    auto render = [&](id<MTLComputePipelineState> pso, bool fx) {
      std::vector<float> g(total);
      id<MTLBuffer> oB = [dev newBufferWithLength:RB * sizeof(float) options:MTLResourceStorageModeShared];
      for (int s = 0; s < total; s += RB) {
        int B = std::min(RB, total - s);
        @autoreleasepool {
          id<MTLCommandBuffer> cb = encode(pso, fx, oB, B, (uint)s);
          [cb commit];
          [cb waitUntilCompleted];
        }
        const float* o = (const float*)oB.contents;
        for (int n = 0; n < B; ++n) g[s + n] = o[n];
      }
      return g;
    };
    Metrics mn = score(ref, render(naive, false), SR);
    Metrics mf = score(ref, render(fixed, true), SR);
    printf("\nFITNESS vs tropical f64 render  (%d samples, %.1f s)   [phase substrate comparison]\n", total, total / SR);
    printf("  %-8s %10s %10s %12s %12s %10s\n", "phase", "SNR", "max_abs", "SNR@t=0", "SNR@t=end", "corr");
    printf("  %-8s %8.1f dB %10.2e %9.1f dB %9.1f dB %10.6f\n", "naive f32", mn.snr, mn.maxAbs, mn.snrFirst, mn.snrLast, mn.corr);
    printf("  %-8s %8.1f dB %10.2e %9.1f dB %9.1f dB %10.6f\n", "fixed", mf.snr, mf.maxAbs, mf.snrFirst, mf.snrLast, mf.corr);
    printf("  (@t=end is t=%.1fs; naive degrades with t, fixed holds — uniform-precision phase)\n", mn.tLast);

    // ---------------- PERFORMANCE: the fixed kernel, latency vs deadline & CPU f64 --------
    const int blocks[] = {64, 128, 256, 512, 1024};
    const int WARM = 64, ITERS = 2000, D = 3;
    printf("\nPERFORMANCE (fixed kernel; us; pipe_call=encode+commit, pipe_sust=block interval)\n");
    printf("%-6s %9s %9s %9s %10s %9s   %-6s %-5s\n", "B", "sync_med", "pipe_call", "pipe_sust", "cpu_f64", "deadline", "rt", "spdup");
    for (int B : blocks) {
      id<MTLBuffer> ob = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
      std::vector<double> s_lat;
      s_lat.reserve(ITERS);
      for (int i = 0; i < WARM + ITERS; ++i)
        @autoreleasepool {
          auto s = clk::now();
          id<MTLCommandBuffer> cb = encode(fixed, true, ob, B, 0);
          [cb commit];
          [cb waitUntilCompleted];
          if (i >= WARM) s_lat.push_back(us_since(s, clk::now()));
        }
      std::sort(s_lat.begin(), s_lat.end());

      std::vector<id<MTLBuffer>> ring(D);
      for (int k = 0; k < D; ++k) ring[k] = [dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
      dispatch_semaphore_t sem = dispatch_semaphore_create(D);
      auto pump = [&](int count, std::vector<double>* call) {
        for (int i = 0; i < count; ++i)
          @autoreleasepool {
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            auto s = clk::now();
            id<MTLCommandBuffer> cb = encode(fixed, true, ring[i % D], B, 0);
            [cb addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(sem); }];
            [cb commit];
            if (call) call->push_back(us_since(s, clk::now()));
          }
      };
      pump(WARM, nullptr);
      for (int k = 0; k < D; ++k) dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
      for (int k = 0; k < D; ++k) dispatch_semaphore_signal(sem);
      std::vector<double> call;
      call.reserve(ITERS);
      auto p0 = clk::now();
      pump(ITERS, &call);
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
      printf("%-6d %8.1fu %8.1fu %8.1fu %9.1fu %8.1fu   %-6s %.2fx\n", B, pct(s_lat, 0.5),
             pct(call, 0.5), pipe_sust, cpu, dl, (pipe_sust < dl) ? "PASS" : "FAIL", cpu / pipe_sust);
    }
  }
  return 0;
}
