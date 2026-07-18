"""modal_arrow — a prototype of the Modal island as a typed arrow EDSL.

The claim from the architecture note, made concrete and executable:

  - objects are one of three kinds: Events, Modal, Signal
  - a voice is an arrow   Events ⇝ Modal   (poles + event-anchored coeffs)
  - a filter/reverb is    Modal  ⇝ Modal   (an EXACT residue-calculus law,
                                            not a heuristic coupling table)
  - eval is               Modal  ⇝ Signal  (the forgetful inclusion into the
                                            pointwise world; warp-equivariant)

`>>>` (spelled `>>` in Python) is arrow composition. It is typed: the cod of
the left must equal the dom of the right, else it raises — the admission rule
lives in the object kinds, not in a runtime warning. Composition of two LTI
filters is itself an LTI filter (transfer-function product, re-residue'd), so
`>>` is associative *by construction* and we test that at the output.

The Modal object is a finite formal sum of exp-poly atoms

    atom(t0, λ, a, k)(τ) = a · (τ−t0)^k · e^{λ(τ−t0)} · [τ > t0]

closed under LTI filtering: the k grows by one exactly when an input pole
lands on a filter pole — that degeneracy *is* the resonance blow-up (τ·e^{μτ}).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import factorial
import numpy as np


# ---------------------------------------------------------------- objects

class Obj(Enum):
    EVENTS = "Events"
    MODAL = "Modal"
    SIGNAL = "Signal"


@dataclass(frozen=True)
class Atom:
    t0: float           # onset (scene time)
    pole: complex       # λ = -σ + iω
    amp: complex        # coefficient a
    deg: int = 0        # k in (τ-t0)^k  (exp-POLY; k>0 is resonance/repeat)


Modal = list          # a Modal signal is just list[Atom]


# ---------------------------------------------------------------- arrows

class Arrow:
    dom: Obj
    cod: Obj

    def __call__(self, x):
        raise NotImplementedError

    def __rshift__(self, other: "Arrow") -> "Arrow":
        if self.cod is not other.dom:
            raise TypeError(
                f"cannot compose {type(self).__name__}:{self.dom.value}⇝"
                f"{self.cod.value} with {type(other).__name__}:"
                f"{other.dom.value}⇝{other.cod.value} — "
                f"{self.cod.value} ≠ {other.dom.value}")
        return Composed(self, other)


class Composed(Arrow):
    def __init__(self, f: Arrow, g: Arrow):
        self.f, self.g = f, g
        self.dom, self.cod = f.dom, g.cod

    def __call__(self, x):
        return self.g(self.f(x))


# ---------------------------------------------------------------- Events ⇝ Modal

class Voice(Arrow):
    """A patch of plucked/struck events → a Modal signal.

    spec(f0) -> list of (partial_freq_ratio, decay_sigma, amp) for one hit.
    Called on a list of (t, f0, vel) events.
    """
    dom, cod = Obj.EVENTS, Obj.MODAL

    def __init__(self, spec):
        self.spec = spec

    def __call__(self, events) -> Modal:
        atoms = []
        for (t, f0, vel) in events:
            for (ratio, sigma, a) in self.spec(f0):
                atoms.append(Atom(t, -sigma + 2j * np.pi * f0 * ratio,
                                  vel * a, 0))
        return atoms


# ---------------------------------------------------------------- Modal ⇝ Modal

@dataclass(frozen=True)
class LTI:
    """A linear time-invariant filter as a partial-fraction transfer function
    H(s) = Σ_q r[q]/(s − ν[q]).  Impulse response h(t) = Σ_q r[q] e^{ν[q] t}.
    A modal reverb is exactly this with thousands of ν on the jω axis."""
    nu: np.ndarray      # filter poles ν_q          (complex, shape M)
    r: np.ndarray       # residues r_q              (complex, shape M)

    def H(self, s):
        return np.sum(self.r / (s - self.nu))

    def compose_after(self, first: "LTI") -> "LTI":
        """self ∘ first  as an LTI:  H = H_self · H_first, re-residue'd.
        Assumes all poles distinct (generic); residue at a simple pole of a
        product is that factor's residue times the *other* factor there."""
        nu = np.concatenate([first.nu, self.nu])
        r = np.concatenate([first.r * np.array([self.H(p) for p in first.nu]),
                            self.r * np.array([first.H(p) for p in self.nu])])
        return LTI(nu, r)


class Filter(Arrow):
    """Modal ⇝ Modal by EXACT residue calculus. For each input atom
    a·(τ-t0)^k e^{λ(τ-t0)} and each filter pole ν_q with residue r_q:

        forced (at λ):   Σ_{i=0}^k  a r_q (-1)^i (k!/(k-i)!) / Δ^{i+1}   deg k-i
        ringing (at ν_q): -a r_q (-1)^k k! / Δ^{k+1}                     deg k
        with Δ = λ - ν_q ;  and when Δ = 0 (input pole ON a filter pole):
        resonance:        a r_q / (k+1)                                  deg k+1
    """
    dom, cod = Obj.MODAL, Obj.MODAL

    def __init__(self, lti: LTI, tol: float = 1e-9):
        self.lti, self.tol = lti, tol

    def __call__(self, atoms: Modal) -> Modal:
        out = []
        nu, r = self.lti.nu, self.lti.r
        for at in atoms:
            k, a, lam, t0 = at.deg, at.amp, at.pole, at.t0
            kfac = factorial(k)
            for q in range(len(nu)):
                D = lam - nu[q]
                rq = r[q]
                if abs(D) > self.tol:
                    poch = 1.0
                    for i in range(k + 1):
                        # poch = k!/(k-i)!  built incrementally
                        out.append(Atom(t0, lam,
                                        a * rq * ((-1) ** i) * poch / D ** (i + 1),
                                        k - i))
                        poch *= (k - i)
                    out.append(Atom(t0, nu[q],
                                    -a * rq * ((-1) ** k) * kfac / D ** (k + 1),
                                    k))
                else:
                    out.append(Atom(t0, nu[q], a * rq / (k + 1), k + 1))
        return out


def modal_reverb(nu, r, tol=1e-9) -> Filter:
    return Filter(LTI(np.asarray(nu, complex), np.asarray(r, complex)), tol)


# ---------------------------------------------------------------- Modal ⇝ Signal

class Eval(Arrow):
    """The forgetful inclusion Modal ⇝ Signal: sample the closed form on a
    time grid `taus` (ANY grid — nonmonotonic/warped included). Optionally a
    diagonal, time-varying gain surface g(ω, τ) — the *swept* filter, which
    lives at the eval boundary because a time-varying gain on e^{λτ} is only
    Modal-closed when the gain is itself exp-poly. Warp-equivariant by
    construction: g and the atoms are both read at the same scene-time τ."""
    dom, cod = Obj.MODAL, Obj.SIGNAL

    def __init__(self, taus: np.ndarray, gain=None):
        self.taus = np.asarray(taus, float)
        self.gain = gain

    def __call__(self, atoms: Modal) -> np.ndarray:
        taus = self.taus
        y = np.zeros(len(taus))
        for at in atoms:
            d = taus - at.t0
            on = d > 0
            dd = np.where(on, d, 0.0)
            val = at.amp * (dd ** at.deg) * np.exp(at.pole * dd)
            if self.gain is not None:
                val = val * self.gain(at.pole.imag / (2 * np.pi), dd + at.t0)
            y += np.where(on, val.real, 0.0)
        return y


def eval_complex(atoms: Modal, taus: np.ndarray) -> np.ndarray:
    """Analytic (complex) realisation Σ a (τ-t0)^k e^{λ(τ-t0)} — no Re. The
    residue law is a statement about complex exponentials, so this is the
    domain in which it is exact. The real waveform is Re of this; a physically
    real filter is modelled by conjugate-closing the pole bank."""
    taus = np.asarray(taus, float)
    z = np.zeros(len(taus), complex)
    for at in atoms:
        d = taus - at.t0
        on = d > 0
        dd = np.where(on, d, 0.0)
        z += np.where(on, at.amp * dd ** at.deg * np.exp(at.pole * dd), 0.0)
    return z


# fast degree-0 evaluator via the event/pole prefix-sum (for long renders)
def eval_fast(atoms: Modal, taus: np.ndarray) -> np.ndarray:
    assert all(a.deg == 0 for a in atoms), "fast path is degree-0 only"
    poles = sorted({a.pole for a in atoms}, key=lambda z: (z.real, z.imag))
    t0s = sorted({a.t0 for a in atoms})
    pi = {p: i for i, p in enumerate(poles)}
    ti = {t: i for i, t in enumerate(t0s)}
    modes = np.array(poles, complex)
    ev = np.array(t0s, float)
    coup = np.zeros((len(ev), len(modes)), complex)
    for a in atoms:
        coup[ti[a.t0], pi[a.pole]] += a.amp
    growth = np.exp(-np.outer(ev, modes))
    pref = np.zeros((len(ev) + 1, len(modes)), complex)
    pref[1:] = np.cumsum(coup * growth, axis=0)
    seg = np.searchsorted(ev, taus, side="right")
    out = np.zeros(len(taus))
    for s in np.unique(seg):
        if s == 0:
            continue
        idx = np.where(seg == s)[0]
        for a0 in range(0, len(idx), 8192):
            ii = idx[a0:a0 + 8192]
            E = np.exp(np.outer(taus[ii], modes))
            out[ii] = (E @ pref[s]).real
    return out
