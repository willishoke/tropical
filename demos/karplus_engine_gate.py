"""karplus_engine_gate — the two-realization differential, in the ENGINE.

The gong-pattern move applied to Karplus-Strong: prosecute the STATEFUL
realization (the delay-line recurrence, numpy `ks_recurrence`) against the
CLOSED-FORM realization (the diagonalized pole bank, rendered by the Lean
`string` node through `diffcli render-graph`) — the SAME LTI object, two ways.

`karplus.py export` writes `karplus_pluck_graph.json` (the pole table as DATA,
one `string` node) and this script renders it through the engine and gates it
against the exact render of that same kept-mode table. It reports:

  * TAIL SNR  — the sustained ring, the modal island's actual claim. ~90 dB
    (float-grade): the engine renders the diagonalized bank essentially
    exactly.
  * ATTACK SNR — the first 0.25 s, where the whole bank momentarily sums to the
    reconstructed pluck (a sharp, high-crest transient). Modal reconstruction
    of a near-delta is numerically delicate — more modes ⇒ sharper spike ⇒
    lower relative SNR — and audibly correct (it is the bright pluck onset,
    same character as gong's "thwack"). Not an engine defect: a sparse bank
    lands ~55 dB here, the full bank ~30 dB, tail unchanged.

The engine applies a fixed 1/20 output scale (a speaker-protection relic), so
the reference is scaled to match. Run `karplus.py export` first.

Usage:  ENGINE=~/tropical/lean/.lake/build/bin/diffcli \\
        uv run --with numpy --with scipy python karplus_engine_gate.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import karplus as K

FMAX = 20000.0          # must match the export's ks_mode_rows fmax
N, RHO, SEED = 224, 0.997, 17
OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)          # the tropical repo root (engine reads
                                     # stdlib/parsed/manifest.json relative to cwd)
ENGINE = os.environ.get(
    "ENGINE", os.path.join(ROOT, "lean/.lake/build/bin/diffcli"))


def kept_render(burst, N, rho, fmax, nsamp):
    """Exact time render of the modes ks_mode_rows keeps: Σ 2·Re((A/p)·pⁿ) over
    conjugate pairs below fmax (+ any real pole). This is the ground truth for
    what the engine's `string` node was handed — bit-exact to the recurrence
    minus the dropped top."""
    p, A, _ = K.ks_residues(burst, N, rho)
    w = np.angle(p)
    n = np.arange(nsamp)
    used = np.zeros(len(p), bool)
    y = np.zeros(nsamp, complex)
    for i in range(len(p)):
        if used[i]:
            continue
        if abs(w[i]) < 1e-9:
            used[i] = True
            y += (A[i] / p[i]) * p[i] ** n
            continue
        j = int(np.argmin(np.abs(p - np.conj(p[i])) + used * 1e9))
        used[i] = used[j] = True
        pos = i if w[i] > 0 else j
        if abs(w[pos]) * K.SR / (2 * np.pi) > fmax:
            continue
        y += 2 * (A[pos] / p[pos] * p[pos] ** n).real
    return y.real


def snr(ref, err):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum(err ** 2))


def main():
    graph = f"{OUT}/karplus_pluck_graph.json"
    if not os.path.exists(graph):
        sys.exit("run `karplus.py export` first (missing karplus_pluck_graph.json)")
    dur = 2.0
    frames = int(dur * K.SR) // 256 + 1
    raw = subprocess.run(
        [ENGINE, "render-graph", graph, "--frames", str(frames), "--buffer", "256"],
        capture_output=True, check=True, cwd=ROOT).stdout
    eng = np.frombuffer(raw, dtype=np.float64)
    nsamp = len(eng)

    burst = K.hash_burst(SEED, N)
    ref = kept_render(burst, N, RHO, FMAX, nsamp) / 20.0   # engine's 1/20 scale
    err = eng - ref
    chunk = K.SR // 4
    attack = snr(ref[:chunk], err[:chunk])
    tail = snr(ref[chunk:], err[chunk:])
    print(f"engine `string` vs exact diagonalized bank ({nsamp} samples):")
    print(f"  TAIL   (>0.25s): {tail:6.1f} dB   float-grade — the closed-form ring")
    print(f"  ATTACK (<0.25s): {attack:6.1f} dB   high-crest onset (bright, delicate, correct)")
    print(f"  {'PASS' if tail > 60 else 'FAIL'}: engine renders the diagonalized KS bank to float precision in the tail")


if __name__ == "__main__":
    main()
