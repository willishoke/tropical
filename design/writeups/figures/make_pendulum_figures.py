#!/usr/bin/env python
"""Figures for the pendulum writeup (oscillation-without-accumulation.md).

    venv/bin/python design/writeups/figures/make_pendulum_figures.py

See _style.py for the shared house aesthetic. Three beats:
  euler-spiral   -- forward Euler injects energy and spirals off the true
                    orbit; a "state" is just a point on that orbit.
  read-anywhere  -- the exact cosine is read at any coordinate directly;
                    the stepped simulation must walk there in order.
  three-regimes  -- the boundary: small angle (cosine), large angle (still
                    closed form, elliptic), driven+damped (chaos, no f(t)).
"""
import numpy as np
from _style import apply_style, figdir_of, finish, tidy, K, G, L, FILL
import matplotlib.pyplot as plt

apply_style()
FIGDIR = figdir_of(__file__)


def _break(x, y, thresh=1.0):
    """Insert NaN at any large jump so a wrapped curve draws without a seam."""
    y = np.asarray(y, float)
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(y)):
        if abs(y[i] - y[i - 1]) > thresh:
            xs.append(x[i]); ys.append(np.nan)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)


# =========================================================================
# Figure 1 -- the simulation drifts.  Forward (explicit) Euler on the
# harmonic oscillator theta'' = -theta injects energy every step, so the
# simulated orbit spirals outward off the true closed orbit.  A "state"
# (theta, theta-dot) is just a point on that true orbit -- reachable by
# evaluating the closed form at t, without travelling there.
# =========================================================================
def fig_euler_spiral():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))

    # true orbit: the unit circle (theta = cos t, theta-dot = -sin t)
    a = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(a), np.sin(a), color=L, linewidth=0.9)
    ax.text(0.72, 0.72, "true orbit", color="0.45", fontsize=8.5,
            ha="left", va="bottom", rotation=-45)

    # forward Euler: x_{n+1} = x + dt v ; v_{n+1} = v - dt x  (energy grows)
    dt = 0.1
    N = 132
    x, v = 1.0, 0.0
    xs, vs = [x], [v]
    for _ in range(N):
        x, v = x + dt * v, v - dt * x
        xs.append(x); vs.append(v)
    xs, vs = np.array(xs), np.array(vs)
    ax.plot(xs, vs, color=K, linewidth=0.9)
    ax.plot(xs[:6], vs[:6], linestyle="none", marker="o", color=K, ms=2.6)
    ax.annotate("forward Euler:\nenergy added each step",
                xy=(xs[-1], vs[-1]), xytext=(-1.72, 1.28), fontsize=8.5,
                color=K, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.3", linewidth=0.7,
                                connectionstyle="arc3,rad=-0.2"))

    # a state = a point on the true orbit, reached by evaluation
    tstar = np.pi / 3
    px, pv = np.cos(tstar), -np.sin(tstar)
    ax.plot(px, pv, marker="o", color=K, ms=4.5)
    ax.annotate(r"a state $(\theta,\dot\theta)$ at time $t$"
                "\nread by evaluating,\nnot travelled to",
                xy=(px, pv), xytext=(0.62, -1.42), fontsize=8.5, color=K,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.3", linewidth=0.7,
                                connectionstyle="arc3,rad=0.25"))

    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)
    ax.axhline(0, color="0.8", linewidth=0.5, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.5, zorder=0)
    ax.set_xlabel(r"position  $\theta$")
    ax.set_ylabel(r"velocity  $\dot\theta$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    tidy(ax)
    finish(fig, "euler-spiral", FIGDIR)


# =========================================================================
# Figure 2 -- read anywhere.  theta(t) = cos(omega t) is a function of t
# alone, so it is read at any coordinate in one step: backward, far ahead,
# or between the samples.  A stepped simulation produces the light dots in
# order, left to right, and can reach a coordinate only by walking to it.
# =========================================================================
def fig_read_anywhere():
    fig, ax = plt.subplots(figsize=(5.9, 2.8))
    t = np.linspace(-3.4, 30, 3000)
    ax.plot(t, np.cos(t), color=K, linewidth=0.9)

    # the stream's samples: computed in order, one after another
    ts = np.arange(0.0, 12.01, 0.6)
    ax.plot(ts, np.cos(ts), linestyle="none", marker="o", color=L, ms=2.6)
    ax.annotate("a stream computes these in order,\nleft to right",
                xy=(ts[9], np.cos(ts[9])), xytext=(3.2, -1.72),
                fontsize=8.5, color="0.45", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.6", linewidth=0.7,
                                connectionstyle="arc3,rad=0.15"))

    # three direct reads, reachable without stepping
    reads = [(-2.3, "backward"), (14.4, "fractional"), (27.0, "jumped")]
    for tau, lab in reads:
        v = np.cos(tau)
        ax.axvline(tau, color="0.6", linestyle=(0, (1, 2.2)), linewidth=0.6)
        ax.plot(tau, v, marker="o", color=K, ms=4.2)
        ax.annotate(lab, xy=(tau, v), xytext=(tau, 1.5), ha="center",
                    va="bottom", fontsize=8.5, color="0.25")

    ax.set_xlim(-3.4, 30)
    ax.set_ylim(-1.9, 1.9)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"time  $t$")
    ax.set_ylabel(r"$\theta(t)=\cos\omega t$")
    ax.axhline(0, color="0.8", linewidth=0.5, zorder=0)
    tidy(ax)
    finish(fig, "read-anywhere", FIGDIR)


# =========================================================================
# Figure 3 -- the boundary, in three phase portraits.  Small angle: a clean
# closed orbit, theta = A cos(omega t).  Large angle: still a closed orbit
# and still a closed form (Jacobi sn), only distorted and slower.  Driven +
# damped: a chaotic tangle -- no f(t), only integration.  The undamped
# orbits come straight from energy conservation (closed form); the chaotic
# one must be integrated, which is exactly the point.
# =========================================================================
def _orbit(theta_max):
    """Undamped pendulum orbit theta-dot = +/- sqrt(2(cos th - cos th_max))
    from energy conservation -- a closed form, no time-stepping."""
    th = np.linspace(-theta_max, theta_max, 600)
    w = np.sqrt(np.maximum(2.0 * (np.cos(th) - np.cos(theta_max)), 0.0))
    x = np.concatenate([th, th[::-1]])
    y = np.concatenate([w, -w[::-1]])
    return x, y


def _driven_damped(keep_every=4):
    """theta'' = -(1/q) theta' - sin theta + g cos(wd t), a chaotic set.
    RK4; drop the transient; wrap theta into (-pi, pi].  Subsampled so the
    trajectory stays a single thin vector path (no dense scatter blowup)."""
    q, g, wd = 2.0, 1.5, 2.0 / 3.0
    dt = 0.02
    n = 60000
    th, om, tt = 0.2, 0.0, 0.0

    def deriv(th, om, tt):
        return om, -(1.0 / q) * om - np.sin(th) + g * np.cos(wd * tt)

    THS, OMS = [], []
    for i in range(n):
        k1t, k1o = deriv(th, om, tt)
        k2t, k2o = deriv(th + 0.5 * dt * k1t, om + 0.5 * dt * k1o, tt + 0.5 * dt)
        k3t, k3o = deriv(th + 0.5 * dt * k2t, om + 0.5 * dt * k2o, tt + 0.5 * dt)
        k4t, k4o = deriv(th + dt * k3t, om + dt * k3o, tt + dt)
        th += dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += dt / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
        tt += dt
        if i > 12000 and i % keep_every == 0:    # drop transient, subsample
            THS.append((th + np.pi) % (2 * np.pi) - np.pi)
            OMS.append(om)
    return np.array(THS), np.array(OMS)


def fig_three_regimes():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    SUB_Y = -0.46                            # status line, just under xlabel

    # small angle -- a clean closed orbit
    x, y = _orbit(0.5)
    axes[0].plot(x, y, color=K, linewidth=0.9)
    axes[0].set_title("small angle", fontsize=9.5)
    axes[0].text(0.5, SUB_Y, r"$\theta = A\cos\omega t$",
                 transform=axes[0].transAxes, ha="center", fontsize=8.5,
                 color="0.3")
    axes[0].set_xlim(-1.0, 1.0); axes[0].set_ylim(-1.0, 1.0)
    axes[0].set_xticks([-0.5, 0, 0.5]); axes[0].set_yticks([-0.5, 0, 0.5])

    # large angle -- still closed, only distorted and slower
    x, y = _orbit(0.94 * np.pi)
    axes[1].plot(x, y, color=K, linewidth=0.9)
    axes[1].set_title("large angle", fontsize=9.5)
    axes[1].text(0.5, SUB_Y, r"still closed form (elliptic)",
                 transform=axes[1].transAxes, ha="center", fontsize=8.5,
                 color="0.3")
    axes[1].set_xlim(-np.pi, np.pi); axes[1].set_ylim(-2.5, 2.5)
    axes[1].set_xticks([-2, 0, 2]); axes[1].set_yticks([-2, 0, 2])

    # driven + damped -- chaos, no closed form.  A single thin trajectory
    # line, wrap-broken at +/-pi so no horizontal seam draws.
    th, om = _driven_damped()
    thb, omb = _break(th, om, thresh=np.pi)
    axes[2].plot(thb, omb, color=K, linewidth=0.3, alpha=0.7)
    axes[2].set_title("driven $+$ damped", fontsize=9.5)
    axes[2].text(0.5, SUB_Y, r"no closed form",
                 transform=axes[2].transAxes, ha="center", fontsize=8.5,
                 color="0.3")
    axes[2].set_xlim(-np.pi, np.pi); axes[2].set_ylim(-3.2, 3.2)
    axes[2].set_xticks([-2, 0, 2]); axes[2].set_yticks([-2, 0, 2])

    for ax in axes:
        ax.set_xlabel(r"$\theta$", labelpad=1.5)
        tidy(ax)
    axes[0].set_ylabel(r"$\dot\theta$")
    fig.subplots_adjust(wspace=0.34, bottom=0.30, top=0.88)
    finish(fig, "three-regimes", FIGDIR)


if __name__ == "__main__":
    fig_euler_spiral()
    fig_read_anywhere()
    fig_three_regimes()
    print("wrote euler-spiral.svg, read-anywhere.svg, three-regimes.svg to", FIGDIR)
