"""High-precision settling of the bloomComposedSig 'semantically invalid' claim.

The docstring (Modal.lean:1212) says: at Re(a) ~ -1 (a near a negative integer)
the Gamma-star bridge is 'semantically invalid, the two lane-totals disagree by
~1e8'. The recon claims: the bridge is EXACT; only the float64 series-M Horner is
garbage (catastrophic cancellation). Settle it with mpmath at 80 digits.

Formulas transcribed from Modal.lean crossing bake (1121-1138) + bloomComposedSig
(1216-1319, non-coincident crossing lanes 1302-1317) and the CplxB numerics.
"""
import mpmath as mp
mp.mp.dps = 80

TWO_PI = 2 * mp.pi
g = mp.mpf('1.8')

def cf_hp(a, z):
    """CF(z) = Gamma(a,z)*e^z*z^{-a} at high precision (the code's bloomCF twin)."""
    return mp.gammainc(a, z) * mp.e**z * z**(-a)

def M_hp(a, z):
    """M(1,a+1,z) = hyp1f1(1, a+1, z) at high precision."""
    return mp.hyp1f1(1, a + 1, z)

def gamma_star_hp(a, kappa):
    """Gamma* = Gamma(a)*kappa^{-a}*e^kappa / g, via exp(lgamma - a log k + k)/g."""
    return mp.e**(mp.loggamma(a) - a * mp.log(kappa) + kappa) / g

def M_float_horner(a, z, nDepth):
    """The code's FIXED-DEPTH float64 series Horner (bloomComposedSig:1303-1304):
       mser = foldr (fun ik h => 1 + z*ik*h) 1  over invA = [1/(a+k+1)]."""
    import cmath
    a = complex(a); z = complex(z)
    invA = [1.0 / (a + (k + 1)) for k in range(nDepth)]
    h = complex(1, 0)
    for ik in reversed(invA):
        h = 1 + z * ik * h
    return h

def lanes(mu, nu, B, an_re, an_im, dprobe, nDepth, kDepth):
    a = (nu - mu) / g
    kappa = mu * B
    # crossing bake
    cfK = cf_hp(a, kappa)
    gs = gamma_star_hp(a, kappa)
    k1Ser = gs - cfK / g
    k1Cf = -cfK / g
    fSer = -1 / (nu - mu)
    # at dprobe
    d = mp.mpf(dprobe)
    eg = mp.e**(-g * d)
    z = kappa * eg
    off = B * (1 - eg)
    Env_nu = mp.e**(nu * d)
    Env_mu = mp.e**(mu * (d + off))
    # series lane (high precision M)
    Mhp = M_hp(a, z)
    k2ser_hp = Mhp * fSer
    ser_hp = k1Ser * Env_nu + k2ser_hp * Env_mu
    # CF lane (high precision)
    k2cf = cf_hp(a, z) / g
    cf_hp_total = k1Cf * Env_nu + k2cf * Env_mu
    # series lane with the code's float64 Horner M
    Mf = M_float_horner(a, z, nDepth)
    k2ser_f = complex(Mf) * complex(fSer)
    ser_f = complex(k1Ser) * complex(Env_nu) + k2ser_f * complex(Env_mu)
    return dict(a=a, kappa=kappa, z=z, Mhp=Mhp, Mf=Mf,
                ser_hp=ser_hp, cf_hp=cf_hp_total, ser_f=ser_f,
                dSwitch=mp.log(abs(kappa) / abs(a + 1)) / g)

def show(title, mu, nu, B, dprobe, nDepth, kDepth):
    r = lanes(mu, nu, B, None, None, dprobe, nDepth, kDepth)
    a, kappa, z = r['a'], r['kappa'], r['z']
    ser_hp, cf_hp_v, ser_f = r['ser_hp'], r['cf_hp'], r['ser_f']
    gap_hp = abs(ser_hp - cf_hp_v) / max(abs(ser_hp), abs(cf_hp_v))
    print(f"\n== {title} ==")
    print(f"   a = {mp.nstr(a,6)}   |a|={mp.nstr(abs(a),5)}  |kappa|={mp.nstr(abs(kappa),5)}  dSwitch={mp.nstr(r['dSwitch'],5)}  probe d={dprobe}")
    print(f"   |z| at probe = {mp.nstr(abs(z),6)}")
    print(f"   HP  M(1,a+1,z) = {mp.nstr(r['Mhp'],6)}   |M|={mp.nstr(abs(r['Mhp']),6)}")
    print(f"   f64 M (Horner) = {r['Mf']}   |M|={abs(r['Mf']):.6e}")
    print(f"   RELATIVE M error (f64 vs HP) = {mp.nstr(abs(r['Mf']-r['Mhp'])/abs(r['Mhp']),4)}")
    print(f"   BRIDGE at d: |ser_hp|={mp.nstr(abs(ser_hp),6)}  |cf_hp|={mp.nstr(abs(cf_hp_v),6)}  RELGAP(HP)={mp.nstr(gap_hp,4)}")
    print(f"   float64 series lane |ser_f|={abs(ser_f):.6e}  vs HP cf |cf_hp|={mp.nstr(abs(cf_hp_v),6)}")
    fgap = abs(complex(ser_f) - complex(cf_hp_v)) / max(abs(complex(ser_f)), abs(complex(cf_hp_v)))
    print(f"   float64-ser vs HP-cf RELGAP = {fgap:.4e}   <- what the render would 'see' at the switch")

# ---- a benign moderate crossing (control): |a| ~ 7, Im a large ----
mu = mp.mpc(-2.0, TWO_PI * 110)
nu = mp.mpc(-5.0, TWO_PI * 112)
show("benign crossing (|a|~7, Im a large)", mu, nu, mp.mpf('0.05')/g, 0.3, 40, 20)

# ---- the hazard: a near -1, Im a small, but |kappa| SMALL enough to not depth-explode ----
# choose mu with small |mu| so |kappa|=|mu|B is small; a.re ~ -0.98
vSig = mp.mpf('0.5'); vOm = TWO_PI * 40    # a LOW partial -> small |kappa|
mu2 = mp.mpc(-vSig, vOm)
# a.re = (vSig - rSig)/g = -0.98 -> rSig = vSig + 0.98*g
rSig = vSig + mp.mpf('0.98') * g
rOm = vOm + mp.mpf('0.05') * g             # a.im ~ 0.05
nu2 = mp.mpc(-rSig, rOm)
a2 = (nu2 - mu2)/g
k2 = mu2 * (mp.mpf('0.05')/g)
print(f"\n[hazard config] a={mp.nstr(a2,6)} |kappa|={mp.nstr(abs(k2),5)} |a+1|={mp.nstr(abs(a2+1),5)}")
show("hazard: a near -1, small |kappa|", mu2, nu2, mp.mpf('0.05')/g, 0.2, 60, 20)

# ---- the hazard with LARGE |kappa| (the 'filter on its own partial' config) ----
vSig3 = mp.mpf('0.5'); vOm3 = TWO_PI * 440
mu3 = mp.mpc(-vSig3, vOm3)
rSig3 = vSig3 + mp.mpf('0.98') * g
rOm3 = vOm3 + mp.mpf('0.05') * g
nu3 = mp.mpc(-rSig3, rOm3)
a3 = (nu3 - mu3)/g; k3 = mu3 * (mp.mpf('0.05')/g)
print(f"\n[hazard-largek config] a={mp.nstr(a3,6)} |kappa|={mp.nstr(abs(k3),5)} |a+1|={mp.nstr(abs(a3+1),5)}")
show("hazard: a near -1, LARGE |kappa| (filter-on-partial)", mu3, nu3, mp.mpf('0.05')/g, 0.2, 120, 40)
