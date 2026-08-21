"""Can a move set find a packing that descent cannot?

The decisive test, because the answer is KNOWN. Five unit squares fit in a square of side
``2 + 1/sqrt(2)``, i.e. at ``phi = 0.68227`` -- `tests/knownOptimumCheck.py` constructs that packing in
closed form and shows it is a valid, attracting minimum of this energy. So at that density a
zero-overlap arrangement provably exists. If a search starting from a random configuration cannot find
one, the failure is the SEARCH and nothing else; there is no appeal to the model.

Four checks:

  1. the moves preserve what they claim to -- rotate and translate leave every constraint residual
     untouched (rigid motions), and swap exchanges two sizes with the hard constraints still exact;
  2. a quench alone fails at the optimal density, establishing the baseline the search has to beat;
  3. basin hopping is measured against that baseline at the optimal density;
  4. swap moves are NEGLIGIBLE on a nominally monodisperse packing and substantial on a polydisperse
     one, so a result that credits them can be believed. Note "monodisperse" is never exact here:
     ``syncTargetAreas`` reads the targets off the geometry, and the builder relaxes random stars with
     FIRE, so sizes differ by ~1e-06 and a swap is always formally possible -- just pointless.

Run: python tests/searchCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
import search


SIDE = 2.0 + 1.0 / np.sqrt(2.0)
OPTIMUM = 5.0 / SIDE ** 2
WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


def buildRandom(seed, phi, polydispersity = 0.0):
    """Five squares from a random start, walled and rigid, at the requested density."""
    model = Model(N = 5, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = 0.45, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    if polydispersity > 0.0:
        model.setLogNormalScale(polydispersity = polydispersity)
    model.addShape(WALL)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setSpringConstants(area = 0, edge = 1, perimeter = 0)
    model.setConstraints(area = True, edge = True)
    model.setModelType("sharp")
    model.setPackingFraction(phi)
    return model


def checkMovesPreserveConstraints():
    """1. The moves land on the constraint manifold rather than being projected onto it."""
    model = buildRandom(3, 0.55)
    packing = model.packing
    before = model.constraintResidual()

    search.rotatePolygon(packing, 0, 0.4)
    afterRotate = model.constraintResidual()
    search.translatePolygon(packing, 1, np.array([0.01, -0.02]))
    afterTranslate = model.constraintResidual()
    print(f"  1. constraint residual   start {before:.3e}   after rotate {afterRotate:.3e}   "
          f"after translate {afterTranslate:.3e}")
    assert afterRotate <= max(before, 1e-14) * 1.01 + 1e-15, "rotation broke the constraints"
    assert afterTranslate <= max(before, 1e-14) * 1.01 + 1e-15, "translation broke the constraints"

    poly = buildRandom(3, 0.55, polydispersity = 0.15)
    areas = np.array(poly.getTargetAreas()[:5])
    swapped = search.swapSizes(poly.packing, 0, 1)
    after = np.array(poly.getTargetAreas()[:5])
    residual = poly.constraintResidual()
    actual = poly.getAreas()[:5]
    error = float(np.abs(actual / after[:5] - 1.0).max())
    print(f"  1. swap   sizes {areas[0]:.5f},{areas[1]:.5f} -> {after[0]:.5f},{after[1]:.5f}   "
          f"max|C| {residual:.3e}   area vs target {error:.3e}")
    assert swapped, "the swap was a no-op on a polydisperse packing"
    assert abs(after[0] - areas[1]) < 1e-12 and abs(after[1] - areas[0]) < 1e-12, \
        "swap did not exchange the two target areas"
    assert error < 1e-12, f"swap left the geometry {error:.3e} off its targets"


def quenchBaseline(seed, phi):
    """2. What a plain relaxation achieves at this density -- the bar the search must clear."""
    model = buildRandom(seed, phi)
    model.minimizeFIRE(maxUnbalancedForce = 1e-10, maxSteps = 20000, progressBar = False)
    return search.objective(model)


def checkSearchAtOptimum():
    """3. At the optimal density, where a solution is KNOWN to exist, does the search find one?"""
    print(f"  3. target density {OPTIMUM:.6f} (the known optimum; a valid packing exists)")
    for seed in (1, 2, 3):
        baseline = quenchBaseline(seed, OPTIMUM)
        model = buildRandom(seed, OPTIMUM)
        rng = np.random.default_rng(seed)
        result = search.basinHop(model, rounds = 400, temperature = 2e-4,
                                 relaxSteps = 2000, rng = rng)
        print(f"     seed {seed}   quench {baseline:.3e}   ->   search {result.objective:.3e}   "
              f"{'SOLVED' if result.solved else 'unsolved'}   "
              f"rot {result.rate('rotate'):.2f} tra {result.rate('translate'):.2f} "
              f"swp {result.rate('swap'):.2f}   ({result.rounds} rounds)", flush = True)


def swapMagnitude(model):
    """How much a swap actually changes the two sizes, as |ratio - 1|."""
    areas = np.asarray(model.getTargetAreas(), dtype = float)[:5]
    return float(abs(np.sqrt(areas[1] / areas[0]) - 1.0))


def checkSwapActivity():
    """4. Swap is negligible on a nominally monodisperse packing, substantial on a polydisperse one."""
    mono = swapMagnitude(buildRandom(5, 0.55))
    poly = swapMagnitude(buildRandom(5, 0.55, polydispersity = 0.15))
    print(f"  4. swap magnitude |ratio-1|   nominally monodisperse {mono:.3e}   "
          f"polydisperse {poly:.3e}   ({poly / max(mono, 1e-300):.0f}x)")
    assert mono < 1e-4, f"a nominally monodisperse packing has a {mono:.3e} size spread"
    assert poly > 1e-2, f"a polydisperse packing should swap substantially, got {poly:.3e}"


def main():
    print("search: can a move set find what descent cannot?")
    checkMovesPreserveConstraints()
    checkSwapActivity()
    checkSearchAtOptimum()
    print("done")


if __name__ == "__main__":
    main()
