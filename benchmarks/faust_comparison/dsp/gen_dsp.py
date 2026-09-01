#!/usr/bin/env python3
"""Emit the three Faust variants at a given voice count.

The variants separate two axes that a naive "N sines" race would conflate:

  F1 wavetable + phase accumulator  — what a Faust user actually ships
  F2 libm sin  + phase accumulator  — recurrent, sine held constant vs F3
  F3 libm sin  + absolute counter   — CLOSED FORM: tropical's semantics

F2 vs F3 isolates recurrence-vs-closed-form with the sine implementation
fixed, which is the scientific comparison. F1 is the real-world number.
tropical (fixed-point polynomial, closed form) is a fourth point.

Frequencies are spread exactly as the tropical fixture spreads them
(55 Hz * (1 + 0.017*i)) so no voice can be folded into another.
"""
import sys, pathlib

def freqs(n):
    return [55.0 * (1.0 + 0.017 * i) for i in range(n)]

def f1(n):  # wavetable oscillator, phase accumulator (idiomatic Faust)
    body = " + ".join(f"os.osc({f:.6f})" for f in freqs(n))
    return f'import("stdfaust.lib");\nprocess = {body};\n'

def f2(n):  # libm sin, phase accumulator
    head = ('import("stdfaust.lib");\n'
            'phasor(f) = f/ma.SR : (+ : ma.frac) ~ _;\n'
            'sinosc(f) = sin(2.0*ma.PI*phasor(f));\n')
    body = " + ".join(f"sinosc({f:.6f})" for f in freqs(n))
    return head + f"process = {body};\n"

def f3(n):  # libm sin, absolute sample counter -> closed form in t
    head = ('import("stdfaust.lib");\n'
            'counter = (+(1) ~ _) - 1;\n'
            'tsec = counter / ma.SR;\n'
            'cfosc(f) = sin(2.0*ma.PI*f*tsec);\n')
    body = " + ".join(f"cfosc({f:.6f})" for f in freqs(n))
    return head + f"process = {body};\n"

if __name__ == "__main__":
    n = int(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("f1_wavetable_recurrent", f1),
                     ("f2_libmsin_recurrent", f2),
                     ("f3_libmsin_closedform", f3)):
        (out / f"{name}_{n}.dsp").write_text(fn(n))
    print(f"wrote 3 variants at N={n} into {out}")
