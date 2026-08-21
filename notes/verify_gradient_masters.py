"""Back the analytic-gradient section of definitions.tex.

Checks, over random (alpha, beta, sigma) and endpoints, that the two gradient master integrals are

    M1(xi)  = int xi   / sqrt(xi^2+sigma^2) * Theta(xi) dxi   (elementary, _m1)
    M1'(xi) = int xi^2 / sqrt(xi^2+sigma^2) * Theta(xi) dxi   (_m1Prime)

with Theta = arctan((alpha xi + beta)/sqrt(xi^2+sigma^2)), and that the only transcendental in M1' is

    T(xi) = int arcsinh(xi/sigma)(alpha sigma^2 - beta xi)/(sqrt(xi^2+sigma^2) Q2) dxi = -2 J_arcsinh,

J_arcsinh the definitions.tex sec 7 integral at m = alpha, nu = beta/sigma (eq Jarcsinh).
"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/rdennis/Documents/Code/pyPolygon")
import energies as E
from partial_fractions_arcsinh import J_arcsinh


def theta(xi, al, be, sg):
    return np.arctan((al * xi + be) / np.sqrt(xi * xi + sg * sg))


def simpson(f, x0, x1, n = 40000):
    h = (x1 - x0) / n
    xs = x0 + h * np.arange(n + 1)
    w = np.ones(n + 1); w[1:-1:2] = 4; w[2:-1:2] = 2
    return h / 3 * np.sum(w * f(xs))


rng = np.random.default_rng(1)
eM1 = eM1p = eT = 0.0
for _ in range(300):
    al = rng.uniform(-3, 3); be = rng.uniform(-3, 3); sg = rng.uniform(0.1, 2.0)
    x0 = rng.uniform(-3, 3); x1 = x0 + rng.uniform(0.2, 4.0)

    dM1 = E._m1(x1, al, be, sg) - E._m1(x0, al, be, sg)
    iM1 = simpson(lambda x: x / np.sqrt(x * x + sg * sg) * theta(x, al, be, sg), x0, x1)
    eM1 = max(eM1, abs(dM1 - iM1))

    dM1p = E._m1Prime(x1, al, be, sg) - E._m1Prime(x0, al, be, sg)
    iM1p = simpson(lambda x: x * x / np.sqrt(x * x + sg * sg) * theta(x, al, be, sg), x0, x1)
    eM1p = max(eM1p, abs(dM1p - iM1p))

    dT = E._tCoreReal(x1, al, be, sg) - E._tCoreReal(x0, al, be, sg)
    m, nu = al, be / sg
    minus2J = -2.0 * (J_arcsinh(m, nu, x0 / sg, x1 / sg))
    eT = max(eT, abs(dT - minus2J))

print(f"max |M1  - int xi/sqrt * Theta|   = {eM1:.2e}")
print(f"max |M1' - int xi^2/sqrt * Theta| = {eM1p:.2e}")
print(f"max |T - (-2 J_arcsinh(alpha, beta/sigma))| = {eT:.2e}")
