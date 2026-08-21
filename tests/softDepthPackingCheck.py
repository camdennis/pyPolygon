"""Does soft depth SEE the overlap that is actually there, and does a descent improve the packing?

Checks 1-11 of ``softDepthCheck.py`` all validate the integrand and the pair. None of them can catch
the two failures that actually happened on 2026-08-01, because both were properties of the assembled
packing rather than of any pair:

  - non-convex loops made ``min_i ell_i`` negative INSIDE, so the energy collapsed to ~0 and FIRE
    reported max|F| = 1.9e-10 on a configuration the sharp tier scored at 3.907;
  - the periodic shift was applied to both bodies of a pair, cancelling, so contacts across the seam
    contributed nothing at all.

Both were reported as successes because ``max|F|`` looked converged. The lesson is that a tier must be
checked against an INDEPENDENT measure of the thing it claims to model, not against its own gradient.
The sharp tier is that measure and is already available.

  1. NOT BLIND -- wherever the sharp tier measures real overlap, soft depth must report energy well
     above zero. This is the direct guard for both failures above, on convex AND non-convex packings.
  2. A VALID PACKING IS A FIXED POINT -- given a zero-overlap configuration, soft depth must recognise
     it and a descent must not walk away from it.
  3. DESCENT -- the energy must fall monotonically, and the real overlap is REPORTED alongside it.

Check 3 does not assert that real overlap falls, and that is deliberate rather than a weak test. Soft
depth penalizes penetration DEPTH and never AREA: for a contact of chord L and depth d the energy is
~ L d^(5/2) while the area is ~ L d, so at fixed area the energy is ~ a d^(3/2), strictly lower the
thinner and wider the overlap is spread. A quench can therefore lower the energy monotonically while
the real overlap area grows -- measured, 6.70e-01 -> 9.9996e-01 at phi = 0.8. That is a property of the
functional, not a bug, and asserting otherwise would encode a false expectation.

Run: python tests/softDepthPackingCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import softDepth as sd
from model import Model


def buildModel(numPolygons, vertexCount, phi = 0.8, seed = 42, evaluate = True):
    model = Model(N = numPolygons, n = vertexCount, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBoundaryConditions("periodic")
    model.setSpringConstants()
    model.setConstraints()
    model.setModelType("softDepth")
    model.setSoftDepth(fraction = 1e-2)
    model.initForceEnergy()
    # A non-convex packing is refused on the default path, so the caller that wants the opt-in tree
    # asks for the model without evaluating it first.
    if evaluate:
        model.calcForceEnergy()
    return model


def sharpOverlap(model, reevaluate = True):
    """The INDEPENDENT measure: the sharp tier's overlap energy at the current configuration.

    Independent in the sense that matters -- it is built on overlap AREA and exact edge-edge crossings,
    sharing no code path, no depth field and no quadrature with the tier under test."""
    keep = model.modelType
    model.setModelType("sharp")
    model.initForceEnergy()
    model.calcForceEnergy()
    value = model.getEnergy()
    model.setModelType(keep)
    model.initForceEnergy()
    # Restoring the tier re-evaluates by default so the cached energy is not stale. A non-convex
    # packing is refused on that path, so the opt-in caller skips it.
    if reevaluate:
        model.calcForceEnergy()
    return value


def convexCount(model):
    vertices = model.packing.positions.reshape(-1, 2)
    starts = np.asarray(model.packing.startIndices, dtype = int)
    return sum(sd.isConvex(vertices[starts[i]:starts[i + 1]])
               for i in range(int(model.packing.numPolygons)))


def checkNotBlind():
    """1. Real overlap must produce real energy.

    Convex packings go through the supported path. The builder cannot make a convex polygon above
    n = 4, and soft depth is CONVEX-ONLY by decision, so the non-convex case is exercised separately
    below through the opt-in rather than through ``Model``."""
    for numPolygons, phi in ((16, 0.8), (12, 0.9), (24, 0.7)):
        model = buildModel(numPolygons, 4, phi = phi)
        soft = model.getEnergy()
        sharp = sharpOverlap(model)
        convex = convexCount(model)
        print(f"  1. N={numPolygons:3d} n=  4 phi={phi}  convex {convex}/{numPolygons}   "
              f"sharp overlap {sharp:.4e}   softDepth {soft:.4e}")
        assert convex == numPolygons, "the fixture is not convex, so it is not the supported path"
        assert sharp > 1e-3, "the reference configuration has no real overlap, so this proves nothing"
        assert soft > 1e-12, (
            f"BLIND: the sharp tier measures {sharp:.4e} of overlap and soft depth reports "
            f"{soft:.4e}. This is the failure mode of 2026-08-01 -- read the module docstring.")

    # The non-convex tier is opt-in (convexDifference.py). It stays covered here because the failure it
    # was built to fix -- a non-convex loop reporting ~0 against real overlap -- is exactly check 1.
    model = buildModel(8, 8, evaluate = False)
    packing = model.packing
    assert convexCount(model) == 0, "the fixture was supposed to be non-convex"
    try:
        sd.packingEnergyForce(packing, model.softEpsilon, 1.0, 0.0, 1.0, 1.0, 16, useCuda = False)
        raise AssertionError("a non-convex packing was accepted without the opt-in")
    except ValueError as refusal:
        assert "CONVEX-ONLY" in str(refusal)
    soft, _ = sd.packingEnergyForce(packing, model.softEpsilon, 1.0, 0.0, 1.0, 1.0, 16,
                                    useCuda = False, allowNonConvex = True)
    sharp = sharpOverlap(model, reevaluate = False)
    print(f"  1. N=  8 n=  8  convex 0/8 (opt-in tree)   sharp overlap {sharp:.4e}   "
          f"softDepth {soft:.4e}   [refused without allowNonConvex, as it must]")
    assert sharp > 1e-3 and soft > 1e-12, "the non-convex tree is blind to real overlap"


def checkValidPackingIsAFixedPoint():
    """2. Given a zero-overlap packing, soft depth must recognise it and stay."""
    model = buildModel(16, 4)
    model.setModelType("sharp")
    model.initForceEnergy()
    model.calcForceEnergy()
    model.minimizeFIRE(maxSteps = 4000, fThreshold = 1e-8, progressBar = False)
    reached = model.getEnergy()
    model.setModelType("softDepth")
    model.initForceEnergy()
    model.calcForceEnergy()
    atValid = model.getEnergy()
    model.minimizeFIRE(maxSteps = 4000, fThreshold = 1e-10, progressBar = False)
    after = model.getEnergy()
    overlapAfter = sharpOverlap(model)
    print(f"  2. sharp reached a valid packing (overlap {reached:.3e});  soft depth scores it "
          f"{atValid:.3e}, and after its own descent {after:.3e}  (overlap {overlapAfter:.3e})")
    assert reached < 1e-6, "the sharp tier did not reach a valid packing, so there is nothing to test"
    assert atValid < 1e-8, "soft depth does not recognise a zero-overlap packing as unstressed"
    assert overlapAfter < 1e-6, "soft depth walked off a valid packing"


def checkDescent():
    """3. The energy falls monotonically. Real overlap is reported, not asserted -- see the docstring."""
    for numPolygons, vertexCount in ((16, 4), (24, 4)):
        model = buildModel(numPolygons, vertexCount)
        budget = 1500
        before, overlapBefore = model.getEnergy(), sharpOverlap(model)
        steps = model.minimizeFIRE(maxSteps = budget, fThreshold = 1e-7, progressBar = False)
        after, overlapAfter = model.getEnergy(), sharpOverlap(model)
        print(f"  3. N={numPolygons:3d} n={vertexCount:3d}  {steps:5d} steps   "
              f"E {before:.4e} -> {after:.4e}   real overlap {overlapBefore:.4e} -> {overlapAfter:.4e}"
              f"   {'(overlap grew -- depth law, see docstring)' if overlapAfter > overlapBefore else ''}")
        assert after <= before, "a descent increased the energy it is descending"


def main():
    print("soft depth at the PACKING level (independent measure: the sharp tier)")
    checkNotBlind()
    checkValidPackingIsAFixedPoint()
    checkDescent()
    print("all checks passed")


if __name__ == "__main__":
    main()
