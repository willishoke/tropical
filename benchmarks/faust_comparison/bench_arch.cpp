// Faust architecture file: wraps the generated DSP in the same block loop
// tropical_runtime_bench uses, so the two sets of numbers are comparable.
//
//   saturation = median per-block wall / (buffer_frames / rate)
//
// Same buffer, rate, warmup discipline and order statistics; the process is
// run in isolation and reports median/p95/p99/max rather than a mean, because
// adjacent runs contend and inflate tails without moving the median.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif

struct Meta { void declare(const char *, const char *) {} };
struct UI {
  void openVerticalBox(const char *) {}
  void openHorizontalBox(const char *) {}
  void openTabBox(const char *) {}
  void closeBox() {}
  void declare(FAUSTFLOAT *, const char *, const char *) {}
  void addButton(const char *, FAUSTFLOAT *) {}
  void addCheckButton(const char *, FAUSTFLOAT *) {}
  void addVerticalSlider(const char *, FAUSTFLOAT *, FAUSTFLOAT, FAUSTFLOAT, FAUSTFLOAT, FAUSTFLOAT) {}
  void addHorizontalSlider(const char *, FAUSTFLOAT *, FAUSTFLOAT, FAUSTFLOAT, FAUSTFLOAT, FAUSTFLOAT) {}
  void addNumEntry(const char *, FAUSTFLOAT *, FAUSTFLOAT, FAUSTFLOAT, FAUSTFLOAT, FAUSTFLOAT) {}
  void addHorizontalBargraph(const char *, FAUSTFLOAT *, FAUSTFLOAT, FAUSTFLOAT) {}
  void addVerticalBargraph(const char *, FAUSTFLOAT *, FAUSTFLOAT, FAUSTFLOAT) {}
  void addSoundfile(const char *, const char *, void **) {}
};
struct dsp {
  virtual ~dsp() {}
};

<<includeIntrinsic>>
<<includeclass>>

using Clock = std::chrono::steady_clock;

static double pct(std::vector<double> & v, double f)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t i = (size_t)std::min<double>(v.size() - 1, std::max(0.0, f * v.size() - 1));
  return v[i];
}

int main(int argc, char ** argv)
{
  int buffer = 512, blocks = 300, warmup = 32, rate = 44100;
  const char * label = "faust";
  for (int i = 1; i < argc - 1; ++i) {
    if (!strcmp(argv[i], "--buffer")) buffer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--blocks")) blocks = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rate")) rate = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--label")) label = argv[++i];
  }

  mydsp dsp_instance;
  dsp_instance.init(rate);
  const int nout = dsp_instance.getNumOutputs();
  const int nin = dsp_instance.getNumInputs();

  std::vector<std::vector<FAUSTFLOAT>> outbuf(std::max(nout, 1),
                                              std::vector<FAUSTFLOAT>(buffer, 0));
  std::vector<std::vector<FAUSTFLOAT>> inbuf(std::max(nin, 1),
                                             std::vector<FAUSTFLOAT>(buffer, 0));
  std::vector<FAUSTFLOAT *> outs(std::max(nout, 1)), ins(std::max(nin, 1));
  for (int c = 0; c < std::max(nout, 1); ++c) outs[c] = outbuf[c].data();
  for (int c = 0; c < std::max(nin, 1); ++c) ins[c] = inbuf[c].data();

  for (int i = 0; i < warmup; ++i)
    dsp_instance.compute(buffer, ins.data(), outs.data());

  std::vector<double> ns;
  ns.reserve(blocks);
  double checksum = 0.0;
  for (int i = 0; i < blocks; ++i) {
    const auto t0 = Clock::now();
    dsp_instance.compute(buffer, ins.data(), outs.data());
    const auto t1 = Clock::now();
    ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    checksum += (double)outs[0][0];
  }

  const double deadline_ns = 1e9 * (double)buffer / (double)rate;
  std::vector<double> copy = ns;
  const double med = pct(copy, 0.50);
  printf("{\"schema\":\"faust_bench_1\",\"label\":\"%s\",\"buffer\":%d,\"rate\":%d,"
         "\"blocks\":%d,\"outputs\":%d,\"deadline_ns\":%.1f,"
         "\"median_ns\":%.1f,\"p95_ns\":%.1f,\"p99_ns\":%.1f,\"max_ns\":%.1f,"
         "\"saturation_median\":%.6f,\"saturation_p99\":%.6f,\"checksum\":%.6f}\n",
         label, buffer, rate, blocks, nout, deadline_ns,
         med, pct(copy, 0.95), pct(copy, 0.99), pct(copy, 1.0),
         med / deadline_ns, pct(copy, 0.99) / deadline_ns, checksum);
  return 0;
}
