"""Three 10 s demos of stateless (closed-form, random-access) modal synthesis.

Engine: y(tau) = Re sum_i e^{mu_i tau} * C_i(tau),
        C_i(tau) = sum_{ev_j < tau} a_ij e^{-mu_i ev_j}   (prefix sum over events)

No per-sample state anywhere: every sample is a pure function of
(tau, mode bank, event list). tau may be any grid — nonmonotonic included.
"""
import numpy as np
import wave
import os

SR = 44100
DUR = 10.0
N = int(SR * DUR)
OUT = "/Users/willishoke/tropical/demos"


# ---------------------------------------------------------------- engine

def render(modes, ev_t, coup, taus, chunk=8192):
    """modes: (M,) complex mu = -sigma + i 2pi f
       ev_t:  (E,) sorted event times (scene time)
       coup:  (E, M, 2) complex per-event, per-mode, per-channel couplings
       taus:  (n,) evaluation times, arbitrary order/spacing
       returns (n, 2) float
    """
    growth = np.exp(-np.outer(ev_t, modes))                 # (E, M)
    pref = np.zeros((len(ev_t) + 1, len(modes), 2), complex)
    pref[1:] = np.cumsum(coup * growth[:, :, None], axis=0)  # C after j events
    seg = np.searchsorted(ev_t, taus, side="right")
    out = np.zeros((len(taus), 2))
    for s in np.unique(seg):
        if s == 0:
            continue                                        # before first event
        idx = np.where(seg == s)[0]
        C = pref[s]                                         # (M, 2)
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            E = np.exp(np.outer(taus[ii], modes))           # (c, M)
            out[ii] = (E @ C).real
    return out


def selftest():
    r = np.random.default_rng(0)
    M, E = 7, 4
    mu = -r.uniform(0.5, 3, M) + 2j * np.pi * r.uniform(50, 300, M)
    ev = np.sort(r.uniform(0, 1, E))
    coup = r.normal(size=(E, M, 2)) + 1j * r.normal(size=(E, M, 2))
    taus = r.uniform(0, 2, 50)
    got = render(mu, ev, coup, taus)
    want = np.zeros((50, 2))
    for j in range(E):
        dt = taus - ev[j]
        on = dt >= 0
        for c in range(2):
            direct = (np.exp(np.outer(dt, mu)) @ coup[j, :, c]).real
            want[:, c] += np.where(on, direct, 0.0)
    err = np.abs(got - want).max()
    assert err < 1e-8, f"selftest failed, err={err}"
    print(f"engine selftest: prefix-sum == direct per-event sum, max err {err:.2e}")


# ---------------------------------------------------------------- helpers

def write_wav(path, y):
    assert np.all(np.isfinite(y)), "non-finite samples"
    y = y / np.max(np.abs(y)) * 0.9
    f = int(0.01 * SR)
    env = np.ones(len(y))
    env[:f] = np.linspace(0, 1, f)
    env[-f:] = np.linspace(1, 0, f)
    y = y * env[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    rms = np.sqrt(np.mean(y ** 2))
    print(f"wrote {path}  ({len(y)/SR:.2f} s, rms {rms:.3f})")


def pluck_bank(f0, npart, rng):
    k = np.arange(1, npart + 1)
    mu = -(1.8 + 0.35 * k) + 2j * np.pi * (f0 * k)
    amp = (1.0 / k) * np.exp(2j * np.pi * rng.random(npart))
    return mu, amp


def room_bank(M, flo, fhi, rt_lo, rt_hi, rng):
    f = np.sqrt(rng.uniform(flo ** 2, fhi ** 2, M))         # density ~ f
    x = np.log(f / flo) / np.log(fhi / flo)
    rt = rt_lo + (rt_hi - rt_lo) * x                        # freq-dependent RT60
    sigma = 6.91 / rt
    return -sigma + 2j * np.pi * f, f


def cplx_randn(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def mix_measured(dry, wet, wet_over_dry):
    g = wet_over_dry * np.sqrt(np.mean(dry ** 2) / np.mean(wet ** 2))
    return dry + g * wet


# ---------------------------------------------------------------- demo 1
# Addressability: one closed-form reverberant scene, evaluated along a
# nonmonotonic master-clock trajectory. Forward -> brake -> reverse (true
# zero-latency reverse reverb: tails become swells) -> fast-forward 2x
# (octave up, varispeed reverb). No buffers, no interpolation: y(w(t)).

def demo1():
    rng = np.random.default_rng(11)
    # the 3.3 s pluck is past the forward pass's turnaround — it is only
    # ever heard during the 2x fast-forward
    notes = [(0.5, 220.0), (1.5, 293.66), (2.5, 349.23), (3.3, 440.0)]
    npart = 12
    E = len(notes)

    mu_p, coup_p = [], np.zeros((E, E * npart, 2), complex)
    for j, (t0, f0) in enumerate(notes):
        mu, amp = pluck_bank(f0, npart, rng)
        mu_p.append(mu)
        for c in range(2):
            coup_p[j, j * npart:(j + 1) * npart, c] = amp * np.exp(
                2j * np.pi * 0.02 * rng.random(npart))
    mu_p = np.concatenate(mu_p)

    mu_r, f_r = room_bank(400, 100, 6000, 2.8, 1.2, rng)
    coup_r = np.zeros((E, 400, 2), complex)
    for j, (t0, f0) in enumerate(notes):
        env = 1.0 / (1.0 + (f_r / (6 * f0)) ** 2)
        coup_r[j] = env[:, None] * cplx_randn(rng, 400, 2)

    ev = np.array([t for t, _ in notes])

    # master-clock velocity profile (cosine ramps between setpoints)
    segs = [(0.0, 3.2, 1, 1), (3.2, 3.8, 1, -1), (3.8, 6.2, -1, -1),
            (6.2, 6.6, -1, 2), (6.6, 8.8, 2, 2), (8.8, 10.01, 2, 0.35)]
    t = np.arange(N) / SR
    v = np.zeros(N)
    for (t0, t1, v0, v1) in segs:
        m = (t >= t0) & (t < t1)
        x = (t[m] - t0) / (t1 - t0)
        v[m] = v0 + (v1 - v0) * (1 - np.cos(np.pi * x)) / 2
    w = np.cumsum(v) / SR
    print(f"demo1 scene-time range along warp: [{w.min():.2f}, {w.max():.2f}] s")

    dry = render(mu_p, ev, coup_p, w)
    wet = render(mu_r, ev, coup_r, w)
    write_wav(f"{OUT}/demo1_timewarp_scrub.wav", mix_measured(dry, wet, 0.7))


# ---------------------------------------------------------------- demo 2
# Lush stateless IIR: 18 plucked notes into a 900-mode reverb with
# frequency-dependent RT60. Modes are shared across all events — the
# prefix sum keeps per-sample cost O(modes), not O(modes x events),
# and the tail is a closed form, not a truncated FIR.

def demo2():
    rng = np.random.default_rng(23)
    scale = [146.83, 174.61, 196.0, 220.0, 261.63, 293.66, 349.23, 392.0]
    order = [0, 2, 4, 5, 4, 2, 1, 3, 5, 7, 5, 3, 0, 4, 6, 7, 4, 0]
    durs = [0.35, 0.35, 0.35, 0.55, 0.35, 0.35, 0.35, 0.55,
            0.35, 0.35, 0.35, 0.55, 0.35, 0.35, 0.35, 0.55, 0.35, 0.35]
    times = 0.3 + np.concatenate([[0], np.cumsum(durs[:-1])])
    freqs = [scale[i] for i in order]
    npart = 8
    E, Mp = len(freqs), len(freqs) * 8

    mu_p, coup_p = [], np.zeros((E, Mp, 2), complex)
    for j, f0 in enumerate(freqs):
        mu, amp = pluck_bank(f0, npart, rng)
        mu_p.append(mu)
        vel = 0.6 + 0.4 * rng.random()
        for c in range(2):
            coup_p[j, j * npart:(j + 1) * npart, c] = vel * amp * np.exp(
                2j * np.pi * 0.02 * rng.random(npart))
    mu_p = np.concatenate(mu_p)

    mu_r, f_r = room_bank(900, 60, 9000, 4.5, 1.2, rng)
    coup_r = np.zeros((E, 900, 2), complex)
    for j, f0 in enumerate(freqs):
        env = 1.0 / (1.0 + (f_r / (8 * f0)) ** 2)
        coup_r[j] = env[:, None] * cplx_randn(rng, 900, 2)

    taus = np.arange(N) / SR
    dry = render(mu_p, times, coup_p, taus)
    wet = render(mu_r, times, coup_r, taus)
    write_wav(f"{OUT}/demo2_stateless_modal_reverb.wav",
              mix_measured(dry, wet, 0.8))


# ---------------------------------------------------------------- demo 3
# Spectral geometry: two mode banks with IDENTICAL smooth density, band,
# decay law, and coupling statistics — differing only in spacing
# statistics. Bank A: near-square membrane (integrable: clustered,
# near-degenerate pairs -> beating, metallic coloration). Bank B: GOE
# (chaotic: level repulsion -> even, smooth wash). Two hits through each.

def demo3():
    rng = np.random.default_rng(37)

    ks = np.arange(1, 40)
    K, L = np.meshgrid(ks, ks)
    fa = 170.0 * np.sqrt(K ** 2 + (L / 1.02) ** 2).ravel()
    fa = np.sort(fa[(fa >= 200) & (fa <= 3000)])
    n = len(fa)

    m = 6 * n
    H = rng.normal(size=(m, m))
    H = (H + H.T) / np.sqrt(2 * m)
    e = np.linalg.eigvalsh(H)[(m - n) // 2:(m - n) // 2 + n]
    p = np.polynomial.Polynomial.fit(e, np.arange(n), 7)
    u = p(e)
    u = (u - u.min()) / (u.max() - u.min())
    q = np.polynomial.Polynomial.fit(np.linspace(0, 1, n), fa, 7)
    fb = np.sort(q(u))

    sa = np.diff(fa) / np.mean(np.diff(fa))
    sb = np.diff(fb) / np.mean(np.diff(fb))
    print(f"demo3: {n} modes/bank; frac of spacings < 0.25*mean — "
          f"membrane {np.mean(sa < 0.25):.2f} vs GOE {np.mean(sb < 0.25):.2f}")

    def bank(f):
        x = np.log(f / 200) / np.log(3000 / 200)
        rt = 3.2 + (1.4 - 3.2) * x
        return -6.91 / rt + 2j * np.pi * f

    mu = np.concatenate([bank(fa), bank(fb)])
    ev = np.array([0.4, 2.3, 5.4, 7.3])
    amp = [1.0, 0.7, 1.0, 0.7]
    coup = np.zeros((4, 2 * n, 2), complex)
    for j in range(4):
        half = slice(0, n) if j < 2 else slice(n, 2 * n)
        coup[j, half] = amp[j] * cplx_randn(rng, n, 2)

    taus = np.arange(N) / SR
    write_wav(f"{OUT}/demo3_spectral_geometry.wav",
              render(mu, ev, coup, taus))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    selftest()
    demo1()
    demo2()
    demo3()
