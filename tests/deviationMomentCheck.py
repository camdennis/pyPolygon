"""Verification for DEVIATION moment constraints -- moments of the distance from ideal, with a
``k = -1`` barrier keeping every deviation strictly positive.

    shape   delta_i = P_i - g_i sqrt(A_i)     g_i = sqrt(4 n_i tan(pi/n_i))   isoperimetric deficit
    area    alpha_i = A0_i - A_i                                              shrink-only
    edge    eps_ik  = |l_ik - l0_i|                                           about the polygon's ideal

The point of the reformulation is conditioning. The direct shape budget holds ``sum_i d_i`` with
``d_i >= 0`` minimized at the regular polygon, so its gradient VANISHES exactly where the ramp is
headed -- measured row norm 1.55e-01 -> 2.24e-02 as the budget fell 0.030 -> 0.0009, after which the
retraction had to be backtracked to stop it hurling the packing away. The ``k = -1`` row's weight is
``-delta^-2``, which GROWS without bound as the deviation shrinks: best conditioned exactly where the
direct form dies. Check 4 is the one that proves it, and it is the check that would have caught the
old degeneracy.

Six checks:

  1. deviations are nonnegative on distorted packings, and the shape deficit is zero (to 1e-14) on
     analytically placed regular polygons -- the floor is where the theorem says it is;
  2. the shape deficit agrees with an INDEPENDENT construction (Model's bincount perimeter / shoelace
     area path, not the constraint's padded-block path);
  3. every deviation Jacobian row against central differences, for k = +1 and k = -1;
  4. the k = -1 row's norm GROWS as the deviation shrinks, where the direct budget's SHRANK;
  5. a retraction lands on a requested pair of (+1, -1) targets with the hard areas still exact;
  6. no deviation crosses zero during a full ramp -- the barrier does its job.

Check 3 is bookkeeping only: finite differences against a quantity's own gradient prove the derivative,
never the definition. Checks 1 and 2 are what make the definition trustworthy.

Run: python tests/deviationMomentCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from constraints import DistributionConstraints


def buildPacking(n = 4, N = 6, seed = 7, distort = 0.15):
    """A small free packing of n-gons, bent away from regular so the deviations are nonzero."""
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = 0.30, kappa = float(np.sqrt(4.0 * n * np.tan(np.pi / n))))
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    if distort > 0.0:
        model.spreadShapes(distort)
    return model


def makeRegular(model):
    """Overwrite every polygon with an exactly regular one, vertices placed in closed form."""
    packing = model.packing
    r = packing.positions.reshape(-1, 2)
    for polygon in range(packing.numPolygons):
        a, b = int(packing.startIndices[polygon]), int(packing.startIndices[polygon + 1])
        center = r[a:b].mean(axis = 0)
        radius = float(np.linalg.norm(r[a] - center))
        angles = 2.0 * np.pi * np.arange(b - a) / (b - a)
        r[a:b] = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis = -1)
    return model


def seedAreaSlack(model, fraction = 0.01):
    """Shrink every polygon about its centroid so ``alpha = A0 - A`` starts strictly POSITIVE.

    An area-deviation family cannot be built on a fresh packing: ``setPackingFraction`` scales the
    geometry and the targets together, so ``alpha`` sits at exactly zero (measured -2.8e-17, i.e.
    roundoff of either sign) and a ``k = -1`` barrier is singular there. The slack is a deliberate
    annealing freedom -- how much a polygon may shrink to squeeze past an obstruction -- so it has to be
    opened before it can be closed, exactly as ``spreadShapes`` seeds shape spread before the moments
    narrow it. Targets are NOT touched, only the geometry."""
    packing = model.packing
    r = packing.positions.reshape(-1, 2)
    factor = np.sqrt(1.0 - float(fraction))
    for polygon in range(model.getNumPolygons()):
        a, b = int(packing.startIndices[polygon]), int(packing.startIndices[polygon + 1])
        centroid = r[a:b].mean(axis = 0)
        r[a:b] = centroid + factor * (r[a:b] - centroid)
    model._forces = None
    model._energy = None
    return model


def checkNonnegativeAndFloor():
    """1. Nonnegative where distorted; the shape deficit exactly zero where regular."""
    model = seedAreaSlack(buildPacking())
    constraints = DistributionConstraints(model.packing, [1], area = True, edge = True, shape = True,
                                          deviation = True)
    worst = {}
    for name in ("shape", "area", "edge"):
        values = constraints.deviations(model.packing, name)
        worst[name] = float(values.min())
    print(f"  1. minimum deviation   shape {worst['shape']:.3e}   area {worst['area']:.3e}   "
          f"edge {worst['edge']:.3e}")
    for name, value in worst.items():
        assert value > 0.0, f"{name} deviation is not strictly positive ({value:.3e})"

    regular = makeRegular(buildPacking(distort = 0.0))
    deficit = DistributionConstraints(regular.packing, [1], shape = True,
                                      deviation = True).deviations(regular.packing, "shape")
    scale = float(np.abs(regular.getPerimeters()).max())
    print(f"     exact regular n-gon deficit   max |delta| = {np.abs(deficit).max():.3e}"
          f"   (perimeter ~ {scale:.3f})")
    assert np.abs(deficit).max() < 1e-14 * scale, "a regular polygon has a nonzero isoperimetric deficit"


def checkAgainstIndependentConstruction():
    """2. The shape deficit vs Model's perimeter/area path, which shares no indexing with it."""
    model = buildPacking()
    constraints = DistributionConstraints(model.packing, [1], shape = True, deviation = True)
    fromConstraint = constraints.deviations(model.packing, "shape")

    counts = np.diff(np.asarray(model.packing.startIndices, dtype = int))[:model.getNumPolygons()]
    g = np.sqrt(4.0 * counts * np.tan(np.pi / counts))
    fromModel = model.getPerimeters() - g * np.sqrt(np.abs(model.getAreas()))

    error = float(np.abs(fromConstraint - fromModel).max())
    print(f"  2. deficit vs independent construction   max |diff| = {error:.3e}")
    assert error < 1e-13, f"constraint and Model disagree on the deficit by {error:.3e}"


def checkJacobians():
    """3. Analytic rows vs central differences, for k = +1 and k = -1."""
    worst = 0.0
    for name in ("shape", "area", "edge"):
        model = seedAreaSlack(buildPacking())
        packing = model.packing
        constraints = DistributionConstraints(packing, [1, -1], deviation = True,
                                              **{name: True})
        analytic = constraints.jacobian(packing)
        scale = constraints._familyScale(name)

        step = 1e-8
        numerical = np.zeros_like(analytic)
        saved = packing.positions.copy()
        for i in range(packing.positions.size):
            packing.positions[i] = saved[i] + step
            plus = constraints.values(packing).copy()
            packing.positions[i] = saved[i] - step
            minus = constraints.values(packing).copy()
            packing.positions[i] = saved[i]
            numerical[:, i] = (plus - minus) / (2.0 * step) / scale
        packing.positions[:] = saved

        for j, k in enumerate([1, -1]):
            error = float(np.abs(analytic[j] - numerical[j]).max())
            magnitude = float(np.abs(numerical[j]).max())
            worst = max(worst, error / max(magnitude, 1e-300))
            print(f"  3. {name:5s} k = {k:+d}   max |diff| = {error:.3e}   (rows ~ {magnitude:.3e})")
            assert error < 1e-5 * max(magnitude, 1.0), f"{name} k={k} Jacobian off by {error:.3e}"


def checkBarrierRowGrows():
    """4. THE point of the reformulation: the k = -1 row stiffens as the deviation shrinks."""
    model = buildPacking()
    model.setConstraints(area = True, edge = False)
    packing = model.packing
    constraints = DistributionConstraints(packing, [1, -1], shape = True, deviation = True)

    saved = packing.positions.copy()
    norms = []
    for factor in (1.0, 0.5, 0.25, 0.125):
        packing.positions[:] = saved
        # Shrink the deviations geometrically by interpolating toward the regular configuration,
        # which is a direct geometric move rather than a retraction -- the retraction is check 5.
        regular = makeRegular(buildPacking(distort = 0.0)).packing.positions
        packing.positions[:] = regular + factor * (saved - regular)
        deficit = float(np.sum(constraints.deviations(packing, "shape")))
        rows = constraints.rowNorms(packing)
        norms.append((deficit, rows["shape 1"], rows["shape -1"]))
    packing.positions[:] = saved

    for deficit, plus, minus in norms:
        print(f"  4. deficit {deficit:.4e}   ||row k=+1|| {plus:.3e}   ||row k=-1|| {minus:.3e}")
    assert norms[-1][2] > norms[0][2], (
        f"the k = -1 row did not stiffen as the deviation shrank: "
        f"{norms[0][2]:.3e} -> {norms[-1][2]:.3e}. That is the whole reason it exists.")


def checkRetraction():
    """5. Move both targets and SHAKE onto them, with the hard areas untouched.

    Composed through ``setConstraints`` rather than by projecting a bare ``DistributionConstraints``:
    the moment rows alone know nothing about the per-object areas, so projecting them directly moves
    the areas freely (measured 1.6e-02 that way). Holding both at once is what ``CompositeConstraints``
    is for, and it is how the anneal actually runs."""
    model = buildPacking()
    model.setConstraints(area = True, edge = False, shape = [1, -1], deviation = True)
    packing = model.packing
    constraints = model.constraints.distribution
    targetAreas = np.array(model.getTargetAreas(), dtype = float)[:model.getNumPolygons()]

    start = constraints.reference["shape"].copy()
    asked = np.array([start[0] * 0.7, start[1] / 0.7])
    constraints.setReference("shape", asked)
    model.constraints.projectPositions(packing)
    landed = constraints.values(packing)
    areaError = float(np.abs(model.getAreas()[:model.getNumPolygons()] / targetAreas - 1.0).max())
    print(f"  5. asked ({asked[0]:.5f}, {asked[1]:.2f})   landed "
          f"({landed[0]:.5f}, {landed[1]:.2f})   area error {areaError:.3e}")
    assert np.abs((landed - asked) / asked).max() < 1e-6, "the retraction missed the targets"
    assert areaError < 1e-9, f"the hard areas drifted by {areaError:.3e} during the retraction"


def checkNoZeroCrossing():
    """6. The barrier keeps every deviation strictly positive through a full ramp."""
    model = buildPacking()
    model.setConstraints(area = True, edge = False)
    packing = model.packing
    constraints = DistributionConstraints(packing, [1, -1], shape = True, deviation = True)

    smallest = np.inf
    reference = constraints.reference["shape"].copy()
    for round in range(1, 9):
        factor = 0.75 ** round
        constraints.setReference("shape", [reference[0] * factor, reference[1] / factor])
        constraints.projectPositions(packing)
        smallest = min(smallest, float(constraints.deviations(packing, "shape").min()))
    print(f"  6. smallest deviation over an 8-round ramp: {smallest:.3e}")
    assert smallest > 0.0, f"a deviation reached {smallest:.3e} -- the barrier failed"


def main():
    print("deviation moment constraints")
    checkNonnegativeAndFloor()
    checkAgainstIndependentConstruction()
    checkJacobians()
    checkBarrierRowGrows()
    checkRetraction()
    checkNoZeroCrossing()
    print("all checks passed")


if __name__ == "__main__":
    main()
