"""A moment constraint must RIDE a uniform rescaling of the packing, not fight it.

``DistributionConstraints`` holds absolute sums -- ``sum_i l_i`` and ``sum_i l_i^2`` under
``edge = [1, 2]`` -- so scaling every polygon about its own centroid moves them by ``factor^k`` even
though the distribution's SHAPE is completely unchanged. That is exactly what every density controller
does, on every step: ``holdExcessEnergy``, ``energySweep`` and ``bisectJamming`` all reach
``Model.setPackingFraction``.

Before ``DistributionConstraints.rescale`` existed the constraint read a pure size change as a
violation. Measured at N = 5, n = 8 under ``area = True, perimeter = True, edge = [1, 2]``:

    x1.02   residual 1.30e-02   retraction pulled phi back 0.083%
    x1.10   residual 4.06e-01   retraction DID NOT CONVERGE, phi back 0.117%

against 3.55e-15 and no drift for the same move with the edges held per object. The residual came out
exactly ``factor - 1`` on the first moment, which is the signature of the whole family being carried
along by the geometry while its reference stayed behind.

  0  a pure rescale leaves EVERY family's residual at zero, up and down
  1  the retraction does not drag the packing fraction back
  2  the distribution itself is untouched -- the CV is what a rescale must not change
  3  the dimensionless families do not move, and the dimensional ones move by the right power

Check 3 is the one that would catch a plausible wrong fix: scaling every family by ``factor^k`` is
arithmetically tidy and wrong, because an area is not a length and the direct shape distortion is a
ratio.

    python tests/momentRescaleCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp

_FACTORS = (1.02, 1.10, 0.85, 0.5)

_SETS = (("edge = [1, 2]", dict(area = True, perimeter = True, edge = [1, 2])),
         ("area = [1, 2]", dict(area = [1, 2], edge = True)),
         ("both families", dict(area = [1, 2], edge = [1, 2])),
         ("shape budget", dict(area = True, edge = True, shape = True)),
         ("distortion", dict(area = True, edge = False, distortion = [1])),
         ("deviation mode", dict(area = True, edge = [1, 2], deviation = True)))


def build(constraints, N = 5, n = 8, seed = 42):
    model = pp.Model(N = N, n = n, seed = seed)
    model.generatePolygons(phi = 0.3, kappa = 4.0, edgePolydispersity = 0.15)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    model.setLogNormalScale(polydispersity = 0.25)
    model.setConstraints(**constraints)
    return model


def checkRescale():
    """CHECKS 0-2, over every constraint set that carries moments."""
    ok = True
    for label, constraints in _SETS:
        model = build(constraints)
        base = model.getPackingFraction()
        width = model.getEdgePolydispersity()["pooled"]
        residual, drift, spread = 0.0, 0.0, 0.0
        for factor in _FACTORS:
            model.setPackingFraction(base * factor)
            asked = model.getPackingFraction()
            residual = max(residual, model.constraintResidual())
            spread = max(spread, abs(model.getEdgePolydispersity()["pooled"] / width - 1.0))
            model.constraints.projectPositions(model.packing)
            drift = max(drift, abs(model.getPackingFraction() / asked - 1.0))
            model.setPackingFraction(base)
            model.constraints.projectPositions(model.packing)
        good = residual < 1e-12 and drift < 1e-12 and spread < 1e-12
        ok = ok and good
        print(f"  {label:16s} residual {residual:.2e}   phi drift {drift:.2e}   "
              f"CV moved {spread:.2e}   {'ok' if good else 'FAIL'}")
    print(f"  CHECKS 0-2 rescale is transparent to the moments {'PASS' if ok else 'FAIL'}")
    return ok


def checkPowers():
    """CHECK 3: each family moves by its own LENGTH DIMENSION, read off the stored reference.

    A length goes as ``factor``, an area as ``factor^2``, and the direct shape distortion is a ratio so
    it does not move at all. Compared against the reference recomputed from scratch on the rescaled
    geometry, which is the definition rather than a restatement of the formula being tested."""
    ok = True
    for label, constraints in _SETS:
        model = build(constraints)
        distribution = model._distributionConstraints()
        base = model.getPackingFraction()
        model.setPackingFraction(base * 1.30)
        worst = 0.0
        for name in distribution.families():
            carried = np.asarray(distribution.reference[name], dtype = float)
            fresh = distribution.momentValues(distribution.quantity(model.packing, name),
                                              distribution.familyMoments(name))
            scale = np.maximum(np.abs(fresh), 1e-300)
            worst = max(worst, float(np.abs(carried - fresh).max() / scale.max()))
        good = worst < 1e-12
        ok = ok and good
        print(f"  {label:16s} carried vs recomputed reference: {worst:.2e}   "
              f"{'ok' if good else 'FAIL'}")
    print(f"  CHECK 3 the exponents are the length dimensions {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("moment constraints under a density move", flush = True)
    warnings.filterwarnings("ignore")
    results = []
    for name, check in (("rescale is transparent", checkRescale),
                        ("length dimensions", checkPowers)):
        print(f"\n{name}:", flush = True)
        results.append(check())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
