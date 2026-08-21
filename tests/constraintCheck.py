"""Validation for constraints.ShapeConstraints and the constrained minimizers.

Checks, in order:

  1. Jacobian vs central finite differences of the residual (every family combination).
  2. projectVector really lands in the tangent space (J w_tan = 0) and is idempotent.
  3. SHAKE pulls a deliberately perturbed configuration back to max|C| ~ 1e-15, quadratically.
  4. A short constrained FIRE run holds the shapes to the SHAKE tolerance throughout, where the
     equivalent spring run leaves a finite shape error.

Run:  python tests/constraintCheck.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import ShapeConstraints
from model import Model


MODES = [
    ("area+edge", {"area": True, "edge": True}),
    ("area+perim", {"area": True, "perimeter": True}),
    ("edge", {"edge": True}),
    ("area", {"area": True}),
    ("all", {"area": True, "perimeter": True, "edge": True}),
]


def fdJacobian(constraints, packing, h = 1e-6):
    """Central-difference dC/dr, assembled block by block into (P, m, 2n)."""
    numPolygons = constraints.numPolygons
    m = constraints.numConstraints
    stride = 2 * constraints.n
    J = np.zeros((numPolygons, m, stride))
    x = packing.positions
    x0 = x.copy()
    for g in range(x.size):
        p, j = divmod(g, stride)
        x[g] = x0[g] + h
        cPlus = constraints.residual(packing)[p]
        x[g] = x0[g] - h
        cMinus = constraints.residual(packing)[p]
        x[g] = x0[g]
        J[p, :, j] = (cPlus - cMinus) / (2.0 * h)
    return J


def buildModel(numPolygons = 6, numVertices = 6, phi = 0.6, seed = 42):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    return model


def checkJacobian(model):
    print("\n[1] Jacobian vs finite differences")
    ok = True
    for mode, kw in MODES:
        constraints = ShapeConstraints(model.packing, **kw)
        analytic = constraints.jacobian(model.packing)
        numeric = fdJacobian(constraints, model.packing)
        err = np.abs(analytic - numeric).max() / max(1.0, np.abs(numeric).max())
        ok &= err < 1e-7
        print(f"    mode={mode:<10} rows/polygon={constraints.numConstraints:<3} "
              f"max relative error {err:.2e}")
    return ok


def checkProjection(model):
    print("\n[2] projectVector lands in the tangent space")
    ok = True
    rng = np.random.default_rng(0)
    for mode, kw in MODES:
        constraints = ShapeConstraints(model.packing, **kw)
        J = constraints.jacobian(model.packing)
        w = rng.standard_normal(model.packing.positions.size)
        tangent = constraints.projectVector(model.packing, w)
        residualNorm = np.abs(
            np.einsum("pmd,pd->pm", J, tangent.reshape(constraints.numPolygons, -1))
        ).max()
        scale = np.abs(np.einsum("pmd,pd->pm", J, w.reshape(constraints.numPolygons, -1))).max()
        twice = constraints.projectVector(model.packing, tangent)
        idem = np.abs(twice - tangent).max()
        ok &= (residualNorm / scale < 1e-12) and (idem < 1e-12)
        print(f"    mode={mode:<10} |J w_tan| / |J w| = {residualNorm / scale:.2e}   "
              f"idempotency {idem:.2e}")
    return ok


def checkShake(model):
    print("\n[3] SHAKE retraction")
    ok = True
    rng = np.random.default_rng(1)
    for mode, kw in MODES:
        constraints = ShapeConstraints(model.packing, **kw)
        saved = model.packing.positions.copy()
        constraints.projectPositions(model.packing)
        edge = float(np.mean(model.packing.targetEdgeLength))
        model.packing.positions += 0.02 * edge * rng.standard_normal(saved.size)
        before = constraints.maxResidual(model.packing)
        iterations, after = constraints.projectPositions(model.packing)
        ok &= after < 1e-13
        print(f"    mode={mode:<10} max|C| {before:.2e} -> {after:.2e} in {iterations} iterations")
        model.packing.positions[:] = saved
    return ok


def checkConstrainedRun(maxSteps = 400, phi = 1.0):
    """Relax the SAME dense seed both ways and compare. Dense enough that the overlap is genuinely
    active, otherwise the constrained run has nothing to do (no contacts -> zero force at step 0)."""
    print(f"\n[4] constrained FIRE vs spring FIRE (sharp overlap, phi = {phi})")
    springModel = buildModel(phi = phi)
    springModel.setSpringConstants(adhesion = 0.0, area = 1.0, perimeter = 0.0, edge = 1.0)
    springModel.minimizeFIRE(maxSteps = maxSteps, fThreshold = 1e-12, dtMax = 0.03)
    springDrift = ShapeConstraints(springModel.packing, area = True,
                                   edge = True).maxResidual(springModel.packing)
    print(f"    springs      max|F| {springModel.getMaxUnbalancedForce():.3e}   "
          f"E {springModel.getEnergy():.6e}   shape error max|C| {springDrift:.2e}")

    model = buildModel(phi = phi)
    model.setConstraints()
    model.minimizeFIRE(maxSteps = maxSteps, fThreshold = 1e-12, dtMax = 0.10)
    drift = model.constraintResidual()
    print(f"    constrained  max|F| {model.getMaxUnbalancedForce():.3e}   "
          f"E {model.getEnergy():.6e}   shape error max|C| {drift:.2e}")
    print(f"                 (tangential residual, dtMax = 0.10, springs dropped from E)")

    ok = drift < 1e-12
    if not ok:
        print("    !! constraint drift exceeded the SHAKE tolerance")
    return ok


def checkRagged():
    """Polygons with DIFFERENT vertex counts must work -- never assume a uniform count."""
    print("\n[5] ragged vertex counts")
    model = Model(N = 6, n = 6, seed = 42)
    model.generateEquilateralPolygons(phi = 0.5, kappa = 4.0)
    for sides, radius, center in ((5, 0.08, 0.5), (3, 0.06, 0.2)):
        angles = np.linspace(0.0, 2.0 * np.pi, sides + 1)[:-1]
        model.addShape(np.column_stack([center + radius * np.cos(angles),
                                        center + radius * np.sin(angles)]))
    counts = np.diff(model.packing.startIndices)
    constraints = ShapeConstraints(model.packing, area = True, edge = True)
    iterations, residual = constraints.projectPositions(model.packing)
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(model.packing.positions.size)
    tangent = constraints.projectVector(model.packing, vector)
    J = constraints.jacobian(model.packing)
    w = tangent.reshape(-1, 2)[constraints.index].reshape(constraints.numPolygons, -1)
    normal = np.abs(np.einsum("pmd,pd->pm", J, w)).max()
    scale = np.abs(np.einsum("pmd,pd->pm", J,
                             vector.reshape(-1, 2)[constraints.index].reshape(
                                 constraints.numPolygons, -1))).max()
    ok = constraints.ragged and residual < 1e-12 and normal / scale < 1e-12
    print(f"      {'OK ' if ok else 'FAIL'} counts {np.unique(counts)} (ragged={constraints.ragged}), "
          f"maxN={constraints.n}, SHAKE -> {residual:.1e}, |J w_tan|/|J w| = {normal / scale:.1e}")
    return ok


def main():
    model = buildModel()
    results = [
        checkJacobian(model),
        checkProjection(model),
        checkShake(model),
        checkConstrainedRun(),
        checkRagged(),
    ]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
