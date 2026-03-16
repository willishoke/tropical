/*
 * Standalone fade-logic test for Graph::begin_fade_in / begin_fade_out.
 *
 * Does NOT open any audio device — drives Graph::process() directly
 * and inspects outputBuffer to verify the smoothstep envelope math.
 *
 * Build (from repo root):
 *   cmake --build build-jit --target test_fade
 *   ./build-jit/test_fade
 */

#include "graph/Graph.hpp"
#include "expr/Expr.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// ---- tiny assertion helpers ------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char * desc)
{
  if (cond)
  {
    std::printf("  PASS  %s\n", desc);
    ++g_pass;
  }
  else
  {
    std::printf("  FAIL  %s\n", desc);
    ++g_fail;
  }
}

static bool near(double a, double b, double tol = 1e-9)
{
  return std::fabs(a - b) <= tol;
}

// ---- helpers ---------------------------------------------------------------

// Populate a Graph so its sole output is the constant 1.0.
static void setup_constant_graph(Graph & g)
{
  using namespace egress_expr;
  // The output mixer divides each source by 20.0, so use 20.0 here so that
  // outputBuffer reads as 1.0 per sample when no fade is active.
  bool ok = g.addOutputExpr(literal_expr(20.0));
  (void)ok;
  assert(ok && "addOutputExpr failed for literal 20.0");
}

// Run process() n times and collect the mean of |outputBuffer| each call.
// Returns a vector of per-call means.
static std::vector<double> run_n(Graph & g, int n)
{
  std::vector<double> means;
  means.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
  {
    g.process();
    const auto & buf = g.outputBuffer;
    double sum = 0.0;
    for (double v : buf) sum += std::fabs(v);
    means.push_back(sum / static_cast<double>(buf.size()));
  }
  return means;
}

// ---- individual tests ------------------------------------------------------

// With no fade active, outputBuffer should be 1.0 for every sample.
static void test_no_fade()
{
  std::puts("\n[test_no_fade]");
  Graph g(512);
  setup_constant_graph(g);
  g.process();
  const auto & buf = g.outputBuffer;
  check(near(buf.front(), 1.0) && near(buf.back(), 1.0),
        "all samples == 1.0 when no fade is active");
}

// After begin_fade_in(2048), the first buffer should be near-silent (t≈0)
// and after 4 full buffer periods (4*512 = 2048 samples) the output should
// be at or very near full amplitude.
static void test_fade_in()
{
  std::puts("\n[test_fade_in]");
  const unsigned int buf = 512;
  Graph g(buf);
  setup_constant_graph(g);

  g.begin_fade_in(2048);

  auto means = run_n(g, 5);

  // Buffer 0: fade_in_remaining starts at 2048.
  // First sample: t = 1 - 2048/2048 = 0 → scale = 0.
  // Last sample of buf 0: t = 1 - (2048-511)/2048 ≈ 0.2495 → scale ≈ 0.108
  // So mean of buf 0 should be well below 0.5.
  check(means[0] < 0.5,
        "fade-in buf 0: mean amplitude < 0.5 (ramp starting from silence)");

  // Buffer 1 mean should be higher than buffer 0.
  check(means[1] > means[0],
        "fade-in buf 1: mean amplitude increases");

  // Buffer 2 mean should still be higher.
  check(means[2] > means[1],
        "fade-in buf 2: mean amplitude still increasing");

  // Buffer 3 mean should still be higher.
  check(means[3] > means[2],
        "fade-in buf 3: mean amplitude still increasing");

  // After 4 buffers the fade should be complete; buf 4 is post-fade — full amplitude.
  check(near(means[4], 1.0, 1e-6),
        "fade-in buf 4 (post-fade): mean amplitude == 1.0");
}

// After begin_fade_out(2048), after 4 full buffers the output should be
// completely silent, and stay silent on subsequent calls.
static void test_fade_out()
{
  std::puts("\n[test_fade_out]");
  const unsigned int buf = 512;
  Graph g(buf);
  setup_constant_graph(g);

  // Warm up first so outputBuffer is at full amplitude.
  run_n(g, 2);

  g.begin_fade_out(2048);

  auto means = run_n(g, 6);

  // Buffer 0 starts at fo=2048, decrementing down.
  // First sample: t = 2048/2048 = 1 → scale = 1.
  // Last sample: t ≈ (2048-511)/2048 ≈ 0.75 → scale ≈ 0.844
  // Mean should be high but not quite 1.
  check(means[0] > 0.5,
        "fade-out buf 0: amplitude still mostly up (ramp starting from full)");

  // Each successive buffer the mean should drop.
  check(means[1] < means[0], "fade-out buf 1: amplitude decreasing");
  check(means[2] < means[1], "fade-out buf 2: amplitude still decreasing");
  check(means[3] < means[2], "fade-out buf 3: amplitude still decreasing");

  // Buffer 3 is the last fading buffer (samples 1537–2048 of the ramp).
  // Buffer 4: fo stored as 0 → each sample is zeroed and held at silence.
  check(near(means[4], 0.0, 1e-9),
        "fade-out buf 4: held at silence");
  check(near(means[5], 0.0, 1e-9),
        "fade-out buf 5: still held at silence");
}

// Verify that fade-in followed immediately by fade-out works:
// begin_fade_in resets the fade-out sentinel to -1 (inactive), so
// a subsequent begin_fade_out should re-enable it without interference.
static void test_fade_in_then_out()
{
  std::puts("\n[test_fade_in_then_out]");
  const unsigned int buf = 512;
  Graph g(buf);
  setup_constant_graph(g);

  // Complete a full fade-in.
  g.begin_fade_in(2048);
  run_n(g, 4);  // consumes all 2048 fade-in samples

  // Confirm we're at full amplitude.
  {
    g.process();
    const auto & b = g.outputBuffer;
    check(near(b.front(), 1.0, 1e-6) && near(b.back(), 1.0, 1e-6),
          "post-fade-in: amplitude == 1.0");
  }

  // Now fade out.
  g.begin_fade_out(2048);
  auto means = run_n(g, 5);
  check(near(means[4], 0.0, 1e-9),
        "after full fade-out: amplitude held at 0.0");
}

// Verify the smoothstep shape: the midpoint (t=0.5) gives exactly 0.5
// (smoothstep(0.5) = 0.5^2 * (3 - 2*0.5) = 0.25 * 2 = 0.5).
//
// With begin_fade_in(1024), fi starts at 1024.
// The FIRST sample processed has fi=1024 (before decrement):
//   t = 1 - 1024/2048 = 0.5  →  scale = 0.25 * 2.0 = 0.5
// So outputBuffer.front() should be exactly 0.5 * base (= 0.5 with base=1.0).
static void test_smoothstep_midpoint()
{
  std::puts("\n[test_smoothstep_midpoint]");
  const unsigned int buf = 512;
  Graph g(buf);
  setup_constant_graph(g);

  g.begin_fade_in(1024);
  g.process();

  const double front = g.outputBuffer.front();
  check(near(front, 0.5, 1e-9),
        "smoothstep midpoint (t=0.5): outputBuffer.front() == 0.5");
}

// ---- main ------------------------------------------------------------------

int main()
{
  std::puts("=== egress fade logic unit tests ===");

  test_no_fade();
  test_fade_in();
  test_fade_out();
  test_fade_in_then_out();
  test_smoothstep_midpoint();

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
