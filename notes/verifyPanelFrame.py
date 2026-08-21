"""Check the restructured edge-pair panel of notes/mollifiedDerivation.tex sec 3.

The section was rewritten to introduce the normalized frame (X, m, nu, w) first and
to drop the intermediate a, b, c, D, J1, J2, p, I_a..I_d. This verifies the two
equations that replaced them, against direct quadrature of the original integrals:

  eq:panel   int_0^1 ln(|x-y|^2 + sigma^2) dt
             = (sigma/L) [ (nu+mX) ln(sigma^2 F) - 2(nu+mX) + 2 sqrt(X^2+1) Theta ]_{nu0}^{nu1}

  eq:Isplit  I2/|e_beta| = -(sigma^2/(L e_beta^y)) { [nu dX + (m/2) dX^2](ln sigma^2 - 2)
                            + int (nu+mX) ln F dX + 2 int sqrt(X^2+1) Theta dX }_{nu0}^{nu1}
"""

import numpy as np
from scipy.integrate import quad, dblquad

rng = np.random.default_rng(20260722)


def frame(vA, eA, vB, eB):
    """Rotate so eA = (L, 0), then return the normalized-frame parameters of eq:Xsub."""
    length = np.hypot(*eA)
    ct, st = eA[0] / length, eA[1] / length
    rot = np.array([[ct, st], [-st, ct]])
    delta = rot @ (vA - vB)
    eBr = rot @ eB
    return length, delta, eBr


def panelClosed(sVal, length, delta, eBr, sigma):
    m = eBr[0] / eBr[1]
    nu0 = (delta[0] - m * delta[1]) / sigma
    nu1 = nu0 + length / sigma
    X = (delta[1] - eBr[1] * sVal) / sigma
    root = np.sqrt(X * X + 1.0)

    def branch(nu):
        f = (nu + m * X) ** 2 + X * X + 1.0
        return ((nu + m * X) * np.log(sigma ** 2 * f) - 2.0 * (nu + m * X)
                + 2.0 * root * np.arctan((nu + m * X) / root))

    return (sigma / length) * (branch(nu1) - branch(nu0))


def panelDirect(sVal, vA, eA, vB, eB, sigma):
    y = vB + sVal * eB
    f = lambda t: np.log(np.sum((vA + t * eA - y) ** 2) + sigma ** 2)
    return quad(f, 0.0, 1.0, epsabs = 1e-13, epsrel = 1e-13)[0]


def outerClosed(length, delta, eBr, sigma):
    m = eBr[0] / eBr[1]
    nu0 = (delta[0] - m * delta[1]) / sigma
    nu1 = nu0 + length / sigma
    X0 = delta[1] / sigma
    X1 = (delta[1] - eBr[1]) / sigma

    def branch(nu):
        poly = (nu * (X1 - X0) + 0.5 * m * (X1 ** 2 - X0 ** 2)) * (np.log(sigma ** 2) - 2.0)
        logPiece = quad(lambda X: (nu + m * X) * np.log((nu + m * X) ** 2 + X * X + 1.0),
                        X0, X1, epsabs = 1e-13, epsrel = 1e-13)[0]
        tanPiece = quad(lambda X: np.sqrt(X * X + 1.0)
                        * np.arctan((nu + m * X) / np.sqrt(X * X + 1.0)),
                        X0, X1, epsabs = 1e-13, epsrel = 1e-13)[0]
        return poly + logPiece + 2.0 * tanPiece

    return -(sigma ** 2 / (length * eBr[1])) * (branch(nu1) - branch(nu0))


def outerDirect(vA, eA, vB, eB, sigma):
    f = lambda t, s: np.log(np.sum((vA + t * eA - vB - s * eB) ** 2) + sigma ** 2)
    return dblquad(f, 0.0, 1.0, 0.0, 1.0, epsabs = 1e-13, epsrel = 1e-13)[0]


worstPanel = 0.0
worstOuter = 0.0
for trial in range(300):
    vA, eA, vB, eB = (rng.normal(size = 2) for _ in range(4))
    sigma = 10.0 ** rng.uniform(-3, -0.5)
    length, delta, eBr = frame(vA, eA, vB, eB)
    if abs(eBr[1]) < 1e-3:
        continue

    sVal = rng.random()
    worstPanel = max(worstPanel, abs(panelClosed(sVal, length, delta, eBr, sigma)
                                     - panelDirect(sVal, vA, eA, vB, eB, sigma)))
    worstOuter = max(worstOuter, abs(outerClosed(length, delta, eBr, sigma)
                                     - outerDirect(vA, eA, vB, eB, sigma)))

print(f"eq:panel   vs 1D quadrature : max abs err {worstPanel:.3e}")
print(f"eq:Isplit  vs 2D quadrature : max abs err {worstOuter:.3e}")
