"""Proof of concept: the mollified overlap energy minimizes to numerical precision.

    python pocPrecision.py

Builds a small periodic packing, relaxes it with FIRE, then polishes with Newton's method. The sharp
(unmollified) overlap floors at max|F| ~ 3.6e-3 because its gradient is only C0 across contacts; the
Plummer-mollified energy is C-infinity, so a second-order method drives max|F| down to ~1e-12 (near
the float64 floor) -- which is the whole point of mollifying.

Newton uses the SEMI-analytic force (the default): it is self-consistent (force = grad energy to the
~1e-9 quadrature accuracy), which is what Newton needs. The fully-analytic force is nominally more
accurate but numerically fragile for near-parallel edge pairs, so it stalls Newton -- see
Model.minimizeNewton. N is kept small so the finite-difference Hessian is cheap; scaling to large N
needs the analytic Hessian.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyPolygon as pp


def maxForce(model):
    """Largest per-vertex force magnitude in the model's cached force."""
    return float(np.max(np.abs(model.getForces())))


def main():
    model = pp.Model(N = 6, n = 6, seed = 7)
    model.generateEquilateralPolygons(phi = 1.0, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants(0, area = 1, perimeter = 1, edge = 1)
    model.setSofteningFraction(0.10)

    model.minimizeFIRE(maxSteps = 4000, fThreshold = 1e-6)
    print(f"after FIRE:   max|F| = {maxForce(model):.2e}")

    model.minimizeNewton(maxSteps = 20, fThreshold = 1e-13)
    print(f"after Newton: max|F| = {maxForce(model):.2e}   (sharp overlap floors at ~3.6e-3)")


if __name__ == "__main__":
    main()
