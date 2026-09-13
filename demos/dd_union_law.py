# Claim: causal convolution of exponential divided differences is multiset union:
#   conv(exp[L], exp[M])(d) = exp[L ∪ M](d),  exp[z1..zn] = divided difference of z -> e^{z d}.
# Check 1 (exact identity): conv(exp[l,n], exp[r]) == exp[l,n,r] by quadrature.
# Check 2 (stability): Lagrange/partial-fraction form of exp[l,n,r] in float64 vs a 50-digit
#   reference as n -> l, versus the Hermite-Genocchi (simplex-integral) form.
import numpy as np
from decimal import Decimal, getcontext
getcontext().prec = 50
d = 0.8
l, r = -1.3, -4.1
def dd2(a, b, x):            # exp[a,b](x)
    return (np.exp(a*x) - np.exp(b*x)) / (a - b)
def dd3_lagrange(z, x, E=np.exp):
    z1, z2, z3 = z
    return (E(z1*x)/((z1-z2)*(z1-z3)) + E(z2*x)/((z2-z1)*(z2-z3)) + E(z3*x)/((z3-z1)*(z3-z2)))
def dd3_hg(z, x, N=4000):    # Hermite-Genocchi: ∫_{simplex} e^{x (z1 + t1(z2-z1) + t2(z3-z1))} , 0<=t2<=t1<=1
    z1, z2, z3 = z
    t = (np.arange(N) + 0.5) / N
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    mask = T2 <= T1
    f = np.exp(x*(z1 + T1*(z2-z1) + T2*(z3-z2))) * mask
    return x*x * f.sum() / N**2
# Check 1
n = -2.2
s = np.linspace(0, d, 200001); ds = s[1]-s[0]
conv = np.trapezoid(dd2(l, n, s) * np.exp(r*(d - s)), dx=ds)
print(f"check1  conv(exp[l,n],exp[r]) = {conv:.10f}   exp[l,n,r] = {dd3_lagrange((l,n,r), d):.10f}")
# Check 2
print("check2  eps        lagrange f64 relerr   hermite-genocchi relerr")
for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    z = (l, l+eps, r)
    D = lambda v: Decimal(v)
    zd = tuple(D(v) for v in z)
    ref = float(dd3_lagrange(zd, D(d), E=lambda q: q.exp()))
    lag = dd3_lagrange(z, d); hg = dd3_hg(z, d)
    print(f"        {eps:.0e}   {abs(lag-ref)/abs(ref):.1e}              {abs(hg-ref)/abs(ref):.1e}")
