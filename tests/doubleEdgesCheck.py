"""Validation for Model.doubleNumEdges.

Inserting edge midpoints must change RESOLUTION and nothing else. A midpoint lies on the edge it
splits, so area, perimeter, shape index and the overlap energy have to be preserved to roundoff --
that invariance is the whole test, because it is what makes refine-then-relax a valid strategy rather
than a perturbation.

Also checks that the derived state follows: vertex count doubles, targetEdgeLength halves, shape
constraints are rebuilt for the new block size, pins propagate correctly, and self-repulsion stays
silent (it is a fraction of the edge length, so it halves too -- with the old delta = sigma it would
have lit up).

Run:  python tests/doubleEdgesCheck.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from softBody import backboneArea

warnings.filterwarnings("ignore")


def buildModel(numPolygons = 16, numVertices = 8, constrained = True):
    model = Model(N = numPolygons, n = numVertices, seed = 42)
    model.generateEquilateralPolygons(phi = 1.0, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    model.setModelType("mollified")
    model.setMollification(sigma = 1e-2)
    if constrained:
        model.setConstraints()
    return model


def checkInvariance():
    print("\n[1] geometry and energy are preserved exactly")
    model = buildModel()
    model.minimizeFIRE(maxSteps = 200, fThreshold = 1e-12, dtMax = 0.10)
    model.calcForceEnergy()
    areaBefore = backboneArea(model.packing).copy()
    kappaBefore = model.shapeIndices().copy()
    energyBefore = model.getEnergy()
    vertsBefore = model.getNumVertices()

    model.doubleNumEdges()
    model.calcForceEnergy()
    areaError = np.abs(backboneArea(model.packing) - areaBefore).max()
    kappaError = np.abs(model.shapeIndices() - kappaBefore).max()
    energyError = abs(model.getEnergy() - energyBefore) / max(abs(energyBefore), 1e-300)

    ok = areaError < 1e-14 and kappaError < 1e-12 and energyError < 1e-11
    print(f"      {'OK ' if ok else 'FAIL'} dArea {areaError:.2e}  dKappa {kappaError:.2e}  "
          f"dE/E {energyError:.2e}   ({vertsBefore} -> {model.getNumVertices()} vertices)")
    return ok


def checkDerivedState():
    print("\n[2] derived state follows the refinement")
    model = buildModel()
    edgeBefore = model.packing.targetEdgeLength.copy()
    model.doubleNumEdges()
    # targetEdgeLength is per VERTEX (the edge leaving it), so refinement DOUBLES its length: each
    # original edge becomes two half-length ones, interleaved v0, m0, v1, m1, ... Comparing against the
    # un-repeated array compares arrays of different length.
    expected = np.repeat(edgeBefore, 2) / 2.0
    halved = np.abs(model.packing.targetEdgeLength - expected).max()
    residual = model.constraintResidual()
    rows = model.constraints.numConstraints
    selfRep = model.energyBreakdown()['selfRep']
    ok = halved < 1e-15 and residual < 1e-12 and rows == model.n + 1 and selfRep < 1e-20
    print(f"      {'OK ' if ok else 'FAIL'} targetEdgeLength halved ({halved:.1e}), "
          f"constraints rebuilt to {rows} rows (max|C| {residual:.1e}), selfRep {selfRep:.1e}")
    return ok


def checkPins():
    print("\n[3] pins propagate: a midpoint is pinned only if both endpoints were")
    model = buildModel(constrained = False)
    model.pinPolygons([0])
    starts = model.packing.startIndices
    pinnedPolygonSize = int(starts[1] - starts[0])
    model.pinVertices(np.concatenate([model.getPinnedVertices(), [int(starts[1])]]))
    before = model.getPinnedVertices().size
    model.doubleNumEdges()
    after = model.getPinnedVertices()
    starts = model.packing.startIndices
    wholePolygon = np.arange(int(starts[0]), int(starts[1]))
    # Polygon 0 was fully pinned, so every vertex AND every midpoint of it must be pinned.
    fullyPinned = np.all(np.isin(wholePolygon, after))
    # The lone extra pinned vertex had unpinned neighbors, so it must gain no pinned midpoints.
    isolatedGrew = after.size - int(starts[1]) == 1
    ok = fullyPinned and isolatedGrew
    print(f"      {'OK ' if ok else 'FAIL'} fully-pinned polygon stayed fully pinned ({fullyPinned}); "
          f"isolated pin gained no midpoints ({isolatedGrew})   {before} -> {after.size} pins")
    return ok


def checkRefineThenRelax():
    print("\n[4] refine-then-relax lowers the energy")
    model = buildModel()
    model.minimizeFIRE(maxSteps = 400, fThreshold = 1e-12, dtMax = 0.10)
    coarse = model.getEnergy()
    model.doubleNumEdges()
    model.minimizeFIRE(maxSteps = 400, fThreshold = 1e-12, dtMax = 0.10)
    fine = model.getEnergy()
    ok = fine <= coarse * (1.0 + 1e-9)
    print(f"      {'OK ' if ok else 'FAIL'} E {coarse:.6e} -> {fine:.6e} after refining "
          f"(more resolution can only help)")
    return ok


def checkPowerOfTwo():
    """``powerOfTwo = k`` must multiply the vertex count by 2**k, still without moving anything."""
    print("\n[5] powerOfTwo repeats the refinement")
    ok = True
    for k in (1, 2, 3):
        model = buildModel(numPolygons = 8, numVertices = 8)
        model.calcForceEnergy()
        areaBefore = backboneArea(model.packing).copy()
        energyBefore = model.getEnergy()
        vertsBefore = model.getNumVertices()
        model.doubleNumEdges(powerOfTwo = k)
        model.calcForceEnergy()
        factor = model.getNumVertices() // vertsBefore
        areaError = np.abs(backboneArea(model.packing) - areaBefore).max()
        energyError = abs(model.getEnergy() - energyBefore) / max(abs(energyBefore), 1e-300)
        good = factor == 2 ** k and areaError < 1e-14 and energyError < 1e-10
        ok &= good
        print(f"      {'OK ' if good else 'FAIL'} powerOfTwo={k}: x{factor} vertices "
              f"({vertsBefore} -> {model.getNumVertices()}), dArea {areaError:.1e}, "
              f"dE/E {energyError:.1e}")
    model = buildModel(numPolygons = 4, numVertices = 6, constrained = False)
    try:
        model.doubleNumEdges(powerOfTwo = 0)
        print("      FAIL powerOfTwo=0 did not raise")
        ok = False
    except ValueError:
        print("      OK  powerOfTwo=0 rejected")
    return ok


def main():
    print("\ndoubleNumEdges")
    results = [checkInvariance(), checkDerivedState(), checkPins(), checkRefineThenRelax(),
               checkPowerOfTwo()]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
