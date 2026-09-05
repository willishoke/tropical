// Faust architecture file: render output ch0 as raw little-endian f64 on
// stdout, --frames blocks of --buffer at --rate. The metric-3 twin of
// ../faust_comparison/bench_arch.cpp (which only times); an oversampled run
// of the same dsp is the sweep-quality reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef FAUSTFLOAT
#define FAUSTFLOAT double
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
struct dsp { virtual ~dsp() {} };

<<includeIntrinsic>>
<<includeclass>>

int main(int argc, char ** argv)
{
  int buffer = 512, frames = 16, rate = 44100;
  for (int i = 1; i < argc - 1; ++i) {
    if (!strcmp(argv[i], "--buffer")) buffer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--frames")) frames = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rate")) rate = atoi(argv[++i]);
  }
  mydsp d;
  d.init(rate);
  const int nout = d.getNumOutputs();
  std::vector<std::vector<FAUSTFLOAT>> outs(nout, std::vector<FAUSTFLOAT>(buffer, 0));
  std::vector<FAUSTFLOAT *> optr(nout);
  for (int c = 0; c < nout; ++c) optr[c] = outs[c].data();
  std::vector<double> ch0(buffer);
  for (int b = 0; b < frames; ++b) {
    d.compute(buffer, nullptr, optr.data());
    for (int i = 0; i < buffer; ++i) ch0[i] = (double)outs[0][i];
    fwrite(ch0.data(), sizeof(double), buffer, stdout);
  }
  return 0;
}
