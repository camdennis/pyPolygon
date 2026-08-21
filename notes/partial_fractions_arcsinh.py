"""
Partial-fraction data (alpha_k, beta_k) for the arcsinh dilog integral.

definitions.tex eq (7.30):

    R(y) = [ nu (y^2 - 1) - 2 m y ]
           -----------------------------------------------------
           [ A (y^4 + 1) + 2 B (y^3 - y) + (4C - 2A) y^2 ]
         = sum_{k=1}^{4}  alpha_k / (y - beta_k),

with  A = 1 + m^2,  B = 2 m nu,  C = 1 + nu^2.

The denominator came from  A sinh^2 t + B sinh t + C  under sinh t = (y-1/y)/2,
so it factors as

    D(y) = A (y^2 - 2 s1 y - 1)(y^2 - 2 s2 y - 1),

with s1, s2 the roots of  A s^2 + B s + C = 0.  Each quadratic (y^2 - 2 s_i y - 1)
gives two poles  beta = s_i +/- r_i,  r_i = sqrt(s_i^2 + 1).  Working the residue
by hand on this factored form (using beta^2 = 2 s_i beta + 1) collapses everything:

    N(beta)  = 2 beta (nu s_i - m)
    D'(beta) = 4 A (+/- r_i)(s_i - s_j) beta
    alpha    = N(beta)/D'(beta) = (nu s_i - m) / ( 2 A (+/- r_i)(s_i - s_j) ).

Since 4AC - B^2 = 4(1 + m^2 + nu^2) > 0, s1,s2 are a complex-conjugate pair, so
the four beta_k are two conjugate pairs and the assembled sum is real.

RESULT: every residue is  alpha_k = +/- i/4  (magnitude exactly 1/4, purely
imaginary), so nothing about alpha needs computing -- only the poles beta_k and
their signs sigma_k = -4 i alpha_k in {+1,-1}.  Then

    J_arcsinh = (i/4) sum_k sigma_k [ Li2(y/beta_k) + ln y ln(1 - y/beta_k) ]_{Y0}^{Y1}.

(alpha = (nu s_i - m)/(2A (+/- r_i)(s_i - s_j)); with s_i^2+1 = (m w - i nu)^2/(1+m^2)^2,
w = sqrt(1+m^2+nu^2), the (m w - i nu) factors cancel and alpha = -/+ 1/(4 i) = +/- i/4.)
"""
import sympy as sp

m, nu, y = sp.symbols('m nu y', real=True)

A = 1 + m**2
B = 2*m*nu
C = 1 + nu**2
N = nu*(y**2 - 1) - 2*m*y
D = A*(y**4 + 1) + 2*B*(y**3 - y) + (4*C - 2*A)*y**2
R = N / D

# roots of A s^2 + B s + C = 0 are (-m nu +/- i w)/(1+m^2),  w = sqrt(1+m^2+nu^2)
w = sp.sqrt(1 + m**2 + nu**2)


def poles_and_residues():
    """Return (betas, alphas) as symbolic lists in m, nu (four entries each).

    Closed form (upper root s = (-m nu + i w)/(1+m^2), w = sqrt(1+m^2+nu^2), and
    sqrt(s^2+1) = (m w - i nu)/(1+m^2)) collapses beta = s +/- sqrt(s^2+1) to

        beta_+ = (w - nu)(m + i)/(1+m^2),   alpha = +i/4   (sigma = +1)
        beta_- = (w + nu)(-m + i)/(1+m^2),  alpha = -i/4   (sigma = -1),

    plus their conjugates (lower half-plane), residues -/+ i/4.  No complex sqrt.
    """
    I, d = sp.I, 1 + m**2
    bp = (w - nu)*(m + I)/d
    bm = (w + nu)*(-m + I)/d
    bpc = (w - nu)*(m - I)/d
    bmc = (w + nu)*(-m - I)/d
    return [bp, bm, bpc, bmc], [I/4, -I/4, -I/4, I/4]


def numeric(mval, nuval):
    """Numeric (alphas, betas) at given (m, nu) as complex lists."""
    sub = {m: sp.Float(mval), nu: sp.Float(nuval)}
    betas, alphas = poles_and_residues()
    b = [complex(bi.subs(sub).evalf()) for bi in betas]
    a = [complex(ai.subs(sub).evalf()) for ai in alphas]
    return a, b


def _numeric_checks(mval, nuval):
    """Independent numeric checks: residues vs N/D', poles vs D, sum vs R."""
    import numpy as np
    a, b = numeric(mval, nuval)
    Dc = [A, 2*B, 4*C - 2*A, -2*B, A]                 # D coeffs, high->low power
    Dc = [complex(sp.Float(c.subs({m: mval, nu: nuval}))) for c in Dc]
    Np = np.poly1d([nuval, -2*mval, -nuval])          # N(y) = nu y^2 - 2m y - nu
    Dp = np.poly1d(Dc)
    Dpder = Dp.deriv()
    roots = np.roots(Dc)
    # match each beta to nearest denominator root; residue should be N/D'
    res = []
    for bk in b:
        rr = roots[np.argmin(abs(roots - bk))]
        res.append(Np(rr) / Dpder(rr))
    pole_err = max(min(abs(roots - bk)) for bk in b)
    resid_err = max(abs(ak - rk) for ak, rk in zip(a, res))
    yt = 1.37
    Rnum = Np(yt) / Dp(yt)
    sum_err = abs(sum(ak/(yt - bk) for ak, bk in zip(a, b)) - Rnum)
    return pole_err, resid_err, sum_err


def J_arcsinh(mval, nuval, X0, X1):
    """Real (i-free) value of J_arcsinh: sum over the two upper-half-plane poles

        J = -1/2 sum_{Im beta_k > 0} sigma_k Im[ Li2(y/beta_k)
                                                 + ln y ln(1 - y/beta_k) ]_{Y0}^{Y1},

    sigma_k = -4 i alpha_k in {+1,-1}, Y0 = exp(arcsinh(X0)), Y1 = exp(arcsinh(X1)).
    Needs mpmath for the complex dilogarithm.
    """
    import math, cmath, mpmath as mp
    a, b = numeric(mval, nuval)
    Y0, Y1 = math.exp(math.asinh(X0)), math.exp(math.asinh(X1))
    G = lambda beta, y: complex(mp.polylog(2, y/beta)) + cmath.log(y)*cmath.log(1 - y/beta)
    tot = 0.0
    for ak, bk in zip(a, b):
        if bk.imag > 0:
            sigma = (-4j*ak).real
            tot += -0.5 * sigma * (G(bk, Y1) - G(bk, Y0)).imag
    return tot


def blochWigner(z):
    """Bloch-Wigner dilogarithm D(z) = Im Li2(z) + arg(1-z) ln|z| (real, single-valued)."""
    import cmath, math, mpmath as mp
    return complex(mp.polylog(2, z)).imag + cmath.phase(1 - z)*math.log(abs(z))


def J_arcsinh_real(mval, nuval, X0, X1):
    """J_arcsinh in the manifestly real Bloch-Wigner form,

        J = -1/2 sum_{Im beta_k>0} sigma_k [ D(y/beta_k)
                                             + ln|beta_k| arg(1 - y/beta_k) ]_{Y0}^{Y1},

    using  Im[ Li2(y/b) + ln y ln(1 - y/b) ] = D(y/b) + ln|b| arg(1 - y/b).
    """
    import math, cmath
    a, b = numeric(mval, nuval)
    Y0, Y1 = math.exp(math.asinh(X0)), math.exp(math.asinh(X1))
    imG = lambda beta, y: blochWigner(y/beta) + math.log(abs(beta))*cmath.phase(1 - y/beta)
    tot = 0.0
    for ak, bk in zip(a, b):
        if bk.imag > 0:
            sigma = (-4j*ak).real
            tot += -0.5 * sigma * (imG(bk, Y1) - imG(bk, Y0))
    return tot


def J_arcsinh_clausen(mval, nuval, X0, X1):
    """Fully real J_arcsinh: no complex arithmetic anywhere (Clausen form).

    Lewin's reduction of Bloch-Wigner,  D(r e^{i th}) = ( Cl2(2 th) + Cl2(2 om)
    - Cl2(2 th + 2 om) )/2  with om = -arg(1 - z),  specialized to beta_+/-:

        psi  = atan2(1, m)                       (= arg beta_+, arg beta_- = pi - psi)
        L    = ln(w - nu) - ln(1 + m^2)/2        (= ln|beta_+| = -ln|beta_-|)
        phiP = atan2(y, w - nu - m y)            (= arg(1 - y/beta_+))
        phiM = atan2(y, w + nu + m y)            (= arg(1 - y/beta_-))
        DP   = ( Cl2(2 psi + 2 phiP) - Cl2(2 psi) - Cl2(2 phiP) )/2
        DM   = ( Cl2(2 psi) - Cl2(2 phiM) - Cl2(2 psi - 2 phiM) )/2

        J = -1/2 [ (DP + L phiP) - (DM - L phiM) ]_{Y0}^{Y1}.

    Cl2 (real, odd, 2 pi periodic) is the only special function used.
    """
    import math, mpmath as mp
    Cl2 = lambda x: float(mp.clsin(2, x))
    w = math.sqrt(1 + mval**2 + nuval**2)
    psi = math.atan2(1.0, mval)
    L = math.log(w - nuval) - 0.5*math.log(1 + mval**2)

    def imG(y):
        phiP = math.atan2(y, w - nuval - mval*y)
        phiM = math.atan2(y, w + nuval + mval*y)
        DP = 0.5*(Cl2(2*psi + 2*phiP) - Cl2(2*psi) - Cl2(2*phiP))
        DM = 0.5*(Cl2(2*psi) - Cl2(2*phiM) - Cl2(2*psi - 2*phiM))
        return (DP + L*phiP) - (DM - L*phiM)

    Y0, Y1 = math.exp(math.asinh(X0)), math.exp(math.asinh(X1))
    return -0.5*(imG(Y1) - imG(Y0))


def _check_J(mval, nuval, X0, X1):
    """Real-form J_arcsinh vs the direct integral (definitions.tex eq 7.27)."""
    import math
    A_, B_, C_ = 1 + mval**2, 2*mval*nuval, 1 + nuval**2
    f = lambda x: (math.asinh(x)/math.sqrt(x*x+1))*(nuval*x - mval)/(A_*x*x + B_*x + C_)
    n = 20000
    h = (X1 - X0)/n
    direct = 0.5 * (f(X0) + f(X1) + sum((4 if i % 2 else 2)*f(X0 + i*h) for i in range(1, n)))*h/3
    return abs(J_arcsinh(mval, nuval, X0, X1) - direct)


if __name__ == "__main__":
    betas, alphas = poles_and_residues()
    print("Symbolic poles beta_k:")
    for k, bk in enumerate(betas, 1):
        print(f"  beta_{k} =", bk)
    print("\nSymbolic residues alpha_k:")
    for k, ak in enumerate(alphas, 1):
        print(f"  alpha_{k} =", ak)

    mv, nuv = 0.7, -0.4
    a, b = numeric(mv, nuv)
    print(f"\nNumeric at (m, nu) = ({mv}, {nuv}):")
    for k, (ak, bk) in enumerate(zip(a, b), 1):
        sigma = -4j * ak                     # alpha_k = sigma_k * i/4,  sigma_k = +/-1
        print(f"  beta_{k} = {bk:+.6f}   alpha_{k} = {ak:+.6f}   sigma_{k} = {sigma.real:+.0f}")

    pe, re_, se = _numeric_checks(mv, nuv)
    print(f"\nchecks:  max|beta - root(D)| = {pe:.1e}   "
          f"max|alpha - N/D'| = {re_:.1e}   |sum - R| = {se:.1e}")

    # alpha_k = +/- i/4 for all m, nu
    err = max(abs(abs(ak) - 0.25) + abs(ak.real) for ak in a)
    print(f"max| |alpha_k| - 1/4 | + |Re alpha_k| = {err:.1e}   (alpha_k = +/- i/4)")

    # real (i-free) J_arcsinh vs direct integral
    print(f"J_arcsinh(real form) = {J_arcsinh(mv, nuv, 0.3, 1.5):+.8f}   "
          f"|J - direct| = {_check_J(mv, nuv, 0.3, 1.5):.1e}")

    # Bloch-Wigner form  Im[G] = D(y/b) + ln|b| arg(1 - y/b)  vs the Im[Li2] form
    print(f"J_arcsinh(Bloch-Wigner) = {J_arcsinh_real(mv, nuv, 0.3, 1.5):+.8f}   "
          f"|BW - Im-form| = {abs(J_arcsinh_real(mv, nuv, 0.3, 1.5) - J_arcsinh(mv, nuv, 0.3, 1.5)):.1e}")

    # fully real Clausen form (atan2/log/sqrt + Cl2 only) vs the Bloch-Wigner form
    print(f"J_arcsinh(Clausen)     = {J_arcsinh_clausen(mv, nuv, 0.3, 1.5):+.8f}   "
          f"|Cl2 - BW| = {abs(J_arcsinh_clausen(mv, nuv, 0.3, 1.5) - J_arcsinh_real(mv, nuv, 0.3, 1.5)):.1e}")
