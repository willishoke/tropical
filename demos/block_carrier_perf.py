"""
Performance-side experiments for modal banks.
E4  two-level table factorisation e^{lam d} = A[i]*B[j] in f32 vs f64 reference over a tail.
E5  exact live-mode culling: fraction of modes above a floor as a function of d, synthetic rooms.
E6  harmonic lattice: cos(k theta) / envelope by index recurrence in f32; restart period.
E7  ops-per-mode-per-sample bookkeeping for the candidate kernels.
"""
import numpy as np
rng = np.random.default_rng(11)
SR = 44100

print("=" * 78)
print("E4  two-level tables, f32 storage + f32 complex multiply, vs f64 direct; per-mode SNR over the tail")
T = 2.0; n = int(T * SR); C = int(np.ceil(np.sqrt(n)))          # coarse period C samples
print(f"    tail {T}s = {n} samples, coarse period C={C}, table sizes A={n//C+1} B={C} per mode")
def snr_db(x, ref):
    return 10*np.log10(np.sum(np.abs(ref)**2) / max(np.sum(np.abs(x-ref)**2), 1e-300))
for sigma, freq in [(2.0, 110.0), (20.0, 1000.0), (60.0, 5000.0), (5.0, 12000.0)]:
    lam = complex(-sigma, 2*np.pi*freq) / SR                         # per-sample pole
    j = np.arange(n); ref = np.exp(lam * j)                          # f64 direct
    A = np.exp(lam * C * np.arange(n//C + 1)).astype(np.complex64)   # built in f64, stored f32
    B = np.exp(lam * np.arange(C)).astype(np.complex64)
    tab = (A[j // C] * B[j % C])                                     # f32 complex multiply
    # what the sample actually hears: Re(a * e^{lam d}) with a unit amp
    print(f"    sigma={sigma:>5} f={freq:>7}: SNR(f32 tables vs f64) = {snr_db(tab.real.astype(np.float64), ref.real):6.1f} dB"
          f"   direct f32 exp: {snr_db(np.exp(np.complex64(lam) * j.astype(np.float32)).real.astype(np.float64), ref.real):6.1f} dB")

print("=" * 78)
print("E5  exact culling: modes sorted by death time ln(|a|/floor)/sigma; live fraction over a 2 s tail")
def room(N, rt60_lo, rt60_hi, fmax):
    # RT60 in seconds, mapped to sigma = 6.91/rt60; typical rooms: high modes die faster
    f = np.sort(rng.uniform(40, fmax, N))
    rt60 = rt60_hi * (1 - 0.85 * f / fmax) + rt60_lo               # long at LF, short at HF
    sigma = 6.91 / rt60
    amp = 1.0 / np.sqrt(1 + (f / 800.0)**2)                          # gentle HF rolloff
    return f, sigma, amp
for N, lo, hi, fmax, label in [(64, 0.05, 2.0, 8000, "hall-ish"), (64, 0.2, 0.6, 8000, "room-ish"),
                               (256, 0.05, 3.0, 12000, "big hall, 256 modes"), (32, 1.0, 4.0, 4000, "plate-ish")]:
    f, sigma, amp = room(N, lo, hi, fmax)
    for floor_db in (-96, -120):
        floor = 10**(floor_db/20)
        death = np.log(np.maximum(amp / floor, 1.0)) / sigma          # seconds
        d = np.linspace(0, 2.0, 2001)
        live = (death[None, :] > d[:, None]).sum(axis=1)
        print(f"    {label:>22} floor {floor_db} dB: mean live modes over 2 s = {live.mean():6.1f} / {N}"
              f"  (x{N/live.mean():4.1f});  live at 0.1s={live[100]:>3}  0.5s={live[500]:>3}  1s={live[1000]:>3}")

print("=" * 78)
print("E6  harmonic lattice in f32: cos(k*theta) by Chebyshev recurrence across k; max rel error vs f64")
for K in (64, 128, 256):
    errs = []
    for theta in rng.uniform(0, 2*np.pi, 50):
        ref = np.cos(np.arange(1, K+1) * theta)
        c1 = np.float32(np.cos(theta)); two_c = np.float32(2)*c1
        cur, prev = c1, np.float32(1.0); out = [cur]
        for k in range(2, K+1):
            cur, prev = np.float32(two_c*cur - prev), cur; out.append(cur)
        errs.append(np.max(np.abs(np.array(out, dtype=np.float64) - ref)))
    # with restart every 16
    errs_r = []
    for theta in rng.uniform(0, 2*np.pi, 50):
        ref = np.cos(np.arange(1, K+1) * theta); out = []
        for start in range(0, K, 16):
            c0 = np.float32(np.cos(start*theta)); cm = np.float32(np.cos((start-1)*theta))
            two_c = np.float32(2*np.cos(theta)); cur, prev = c0, cm
            for k in range(start+1, min(start+17, K+1)):
                cur, prev = np.float32(two_c*cur - prev), cur; out.append(cur)
        errs_r.append(np.max(np.abs(np.array(out[:K], dtype=np.float64) - ref)))
    print(f"    K={K:>3}: no restart max abs err {max(errs):.1e}   restart/16 {max(errs_r):.1e}   (f32 eps 6e-8)")

print("=" * 78)
print("E7  ops per mode per sample (rough, from the emitted shapes on main vs candidates)")
rows = [
    ("today: Q32 phase + fixed cos/sin Horner + expSig", "1 imul + 2×(~8 int ops) + ~12 flops + 6 mul", "~40"),
    ("two-level tables (const poles)", "2 loads + 1 complex mul (4 mul 2 add) + 1 add", "~8, memory-bound"),
    ("block of size 2 (paired atom today)", "2 exp + cexpm1 lane (~20 flops)", "~50 per pair"),
    ("block of size k via tables", "k×(2 loads + cmul) + k nested DD terms", "~8k + k²"),
    ("harmonic lattice member", "1 mul + 1 sub (cos) + 1 mul (env) + 1 mul (amp)", "~4"),
]
for a, b, c in rows: print(f"    {a:<48} {b:<48} {c}")
