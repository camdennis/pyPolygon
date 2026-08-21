"""Verification for the SHAPE BUDGET constraint -- the annealing handle that replaces the one-shot
rigid projection.

The budget is ``Phi = sum_i d_i`` with ``d_i = P_i / (sqrt(A_i) g_i) - 1 >= 0`` and
``g_i = sqrt(4 n_i tan(pi/n_i))``. Because every term is nonnegative, holding the SUM is already a
hard squeeze: it can only reach zero with every polygon regular. That is what lets one row do the job
the rigid handoff was doing discontinuously.

Six checks, in the order they would catch a mistake:

  1. the distortion agrees with an INDEPENDENT construction (Model's bincount perimeter / shoelace
     area path, not the constraint's padded-block path);
  2. a regular n-gon reads exactly zero for n = 3 .. 8 -- the floor is where it is claimed to be;
  3. the analytic Jacobian row matches central differences of the constraint VALUE;
  4. lowering the budget and retracting lands on the requested value, with the hard areas still met;
  5. the shape row's own norm decays as the budget does, which is why the ramp must hand off
     rather than finish;
  6. driving the budget down drives the WORST polygon regular, not just the average.

Check 3 is the weakest of the six on its own and is here for bookkeeping only: finite differences of a
quantity against its own gradient prove the derivative, never the definition. Check 1 is what makes
the definition trustworthy, by testing it against code that shares none of its indexing.

Run: python tests/shapeBudgetCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from constraints import DistributionConstraints


def buildPacking(n = 4, N = 6, seed = 7, distort = 0.12):
    """A small free packing of n-gons, deliberately bent away from regular.

    ``kappa`` is the shape index, whose floor depends on n, so it is built from n rather than fixed at
    a square's 4 -- a triangle's floor is 4.559 and would be rejected."""
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = 0.30, kappa = float(np.sqrt(4.0 * n * np.tan(np.pi / n))))
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    if distort > 0.0:
        model.spreadShapes(distort)
    return model


def checkAgainstIndependentConstruction():
    """1. The constraint's distortion vs Model's, which shares no indexing with it."""
    model = buildPacking()
    constraints = DistributionConstraints(model.packing, [1], shape = True)
    fromConstraint = constraints.distortions(model.packing)
    fromModel = model.getShapeDistortions()
    error = float(np.abs(fromConstraint - fromModel).max())
    print(f"  1. distortion vs independent construction   max |diff| = {error:.3e}")
    assert error < 1e-13, f"constraint and Model disagree on the distortion by {error:.3e}"
    assert fromConstraint.min() > 0.0, "a deliberately distorted packing read as regular"


def checkRegularIsZero():
    """2. A regular n-gon is the floor: d = 0 to roundoff, for every n tried.

    The polygons are written EXACTLY here, vertices placed on a circle in closed form, rather than
    taken from the builder. ``generateEquilateralPolygons`` seeds random stars and relaxes them to
    equilateral with FIRE, so its output is regular only to the minimizer's tolerance -- measured
    1.5e-06 in shape index for n = 4, from a 4e-08 spread in edge length. That is the builder's
    precision, not the formula's, and testing the formula against it would measure the wrong thing.
    The builder's own floor is reported below, since it is the smallest budget a fresh packing can
    honestly be annealed to."""
    worst = 0.0
    for n in (3, 4, 5, 6, 7, 8):
        model = buildPacking(n = n, N = 4, distort = 0.0)
        packing = model.packing
        r = packing.positions.reshape(-1, 2)
        for polygon in range(packing.numPolygons):
            a, b = int(packing.startIndices[polygon]), int(packing.startIndices[polygon + 1])
            center = r[a:b].mean(axis = 0)
            radius = float(np.linalg.norm(r[a] - center))
            angles = 2.0 * np.pi * np.arange(b - a) / (b - a)
            r[a:b] = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis = -1)
        constraints = DistributionConstraints(packing, [1], shape = True)
        worst = max(worst, float(np.abs(constraints.distortions(packing)).max()))
    print(f"  2. exact regular n-gon (n = 3..8)           max |d| = {worst:.3e}")
    assert worst < 1e-14, f"an exactly regular polygon reported distortion {worst:.3e}"

    fromBuilder = max(float(np.abs(buildPacking(n = n, N = 4, distort = 0.0)
                                   .getShapeDistortions()).max()) for n in (4, 6, 8))
    print(f"     (builder's own regularity floor           max |d| = {fromBuilder:.3e})")


def checkJacobian():
    """3. Analytic row vs central differences of the constraint value."""
    model = buildPacking()
    packing = model.packing
    constraints = DistributionConstraints(packing, [1], shape = True)
    analytic = constraints.jacobian(packing)[0]
    scale = float(constraints._familyScale("shape")[0])

    step = 1e-7
    numerical = np.zeros_like(analytic)
    saved = packing.positions.copy()
    for i in range(packing.positions.size):
        packing.positions[i] = saved[i] + step
        plus = float(constraints.values(packing)[0])
        packing.positions[i] = saved[i] - step
        minus = float(constraints.values(packing)[0])
        packing.positions[i] = saved[i]
        numerical[i] = (plus - minus) / (2.0 * step) / scale
    packing.positions[:] = saved

    error = float(np.abs(analytic - numerical).max())
    magnitude = float(np.abs(numerical).max())
    print(f"  3. shape Jacobian vs central differences    max |diff| = {error:.3e}"
          f"  (rows ~ {magnitude:.3e})")
    assert error < 1e-6 * max(magnitude, 1.0), f"shape Jacobian off by {error:.3e}"


def checkRetraction():
    """4. Ask for a smaller budget, SHAKE, and land on it with the areas still exact."""
    model = buildPacking()
    model.setConstraints(area = True, edge = False, shape = True)
    targetAreas = np.array(model.getTargetAreas(), dtype = float)

    start = model.getShapeBudget()
    asked = 0.5 * start
    model.setShapeBudget(asked)
    realized = model.getShapeBudget()
    areaError = float(np.abs(model.getAreas()[:model.getNumPolygons()]
                             / targetAreas[:model.getNumPolygons()] - 1.0).max())
    print(f"  4. retract {start:.4f} -> {asked:.4f}                 landed {realized:.6f}"
          f"   area error {areaError:.3e}")
    assert abs(realized - asked) < 1e-9, f"budget landed at {realized:.6e}, asked {asked:.6e}"
    assert areaError < 1e-9, f"the hard areas drifted by {areaError:.3e} during the retraction"


def checkRowDegenerates():
    """5. The row degenerates as the budget vanishes -- the reason the ramp hands off.

    Measured by the row's own NORM, not by ``conditioning``: with a single moment row the ratio of
    singular values is identically 1 and would report perfect health at every budget, right up to the
    point where the retraction stops working."""
    model = buildPacking()
    model.setConstraints(area = True, edge = False, shape = True)
    distribution = model.constraints.distribution
    budget = model.getShapeBudget()
    norms = []
    for factor in (1.0, 0.3, 0.1, 0.03):
        model.setShapeBudget(budget * factor)
        norms.append((budget * factor, distribution.rowNorms(model.packing)["shape 1"]))
    for value, norm in norms:
        print(f"  5. budget {value:.5f}  ->  shape row norm {norm:.3e}")
    assert norms[-1][1] < 0.5 * norms[0][1], (
        f"the shape row did not degenerate as the budget fell: {norms[0][1]:.3e} -> {norms[-1][1]:.3e}")


def checkWorstPolygonFollows():
    """6. The SUM going down takes the worst polygon with it -- nonnegativity doing its job."""
    model = buildPacking()
    model.setConstraints(area = True, edge = False, shape = True)
    budget = model.getShapeBudget()
    before = model.getMaxShapeDistortion()
    model.setShapeBudget(budget * 0.02)
    after = model.getMaxShapeDistortion()
    print(f"  6. worst distortion {before:.3e} -> {after:.3e}")
    assert after < 0.1 * before, (f"the budget fell 50x but the worst polygon only went "
                                  f"{before:.3e} -> {after:.3e}")


def main():
    print("shape budget constraint")
    checkAgainstIndependentConstruction()
    checkRegularIsZero()
    checkJacobian()
    checkRetraction()
    checkRowDegenerates()
    checkWorstPolygonFollows()
    print("all checks passed")


if __name__ == "__main__":
    main()
