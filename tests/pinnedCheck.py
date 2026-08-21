"""Validation for PINNED vertices (Model.pinVertices / pinPolygons).

A pin must hold a vertex EXACTLY fixed while leaving the physics otherwise untouched: the pinned
vertex still pushes on its neighbors and still enters its polygon's area / edge terms.

Checks:
  1. Pinned vertices do not move under FIRE, in the spring model and under constraints.
  2. Free vertices DO move, and the pinned ones still exert force (a pinned polygon is felt).
  3. Pinning composes with the shape constraints: SHAKE satisfies them using only free vertices, so
     the constraint residual stays at its floor while pinned coordinates are bit-identical.
  4. pinPolygons pins exactly the right vertex set, and releasing restores ordinary motion.
  5. A wrapped periodic box never relocates a polygon holding a pin.

Run:  python tests/pinnedCheck.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model

warnings.filterwarnings("ignore")


def buildModel(numPolygons = 12, numVertices = 8, phi = 1.0, seed = 42, mollified = True):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    if mollified:
        model.setModelType("mollified")
        model.setSofteningFraction(0.08)
    return model


def checkHeldFixed():
    print("\n[1] pinned vertices do not move")
    ok = True
    for label, constrained in (("springs", False), ("constrained", True)):
        model = buildModel()
        if constrained:
            model.setConstraints()
        pins = np.array([0, 1, 2, 40, 41])
        model.pinVertices(pins)
        before = model.packing.positions.reshape(-1, 2)[pins].copy()
        model.minimizeFIRE(maxSteps = 150, fThreshold = 1e-12, dtMax = 0.05)
        after = model.packing.positions.reshape(-1, 2)[pins]
        moved = np.abs(after - before).max()
        free = np.abs(model.packing.positions.reshape(-1, 2)[20] - 0.0).max()
        ok &= moved == 0.0
        print(f"      {'OK ' if moved == 0.0 else 'FAIL'} {label:<13} pinned motion {moved:.3e}"
              f"   max|C| {model.constraintResidual():.1e}")
    return ok


def checkFreeStillMove():
    print("\n[2] free vertices still move, pinned ones still push")
    model = buildModel()
    model.pinPolygons([0])
    packing = model.packing
    pinned = model.getPinnedVertices()
    start = packing.positions.copy()
    model.minimizeFIRE(maxSteps = 150, fThreshold = 1e-12, dtMax = 0.05)
    delta = np.abs(packing.positions - start).reshape(-1, 2)
    freeMask = np.ones(packing.numVertices, dtype = bool)
    freeMask[pinned] = False
    movedFree = delta[freeMask].max()
    movedPinned = delta[pinned].max()

    # The pinned polygon must still be FELT: its neighbors should carry force from it.
    model.calcForceEnergy()
    forces = model.getForces()
    neighborForce = np.abs(forces[freeMask]).max()
    ok = movedPinned == 0.0 and movedFree > 1e-6 and neighborForce > 0.0
    print(f"      {'OK ' if ok else 'FAIL'} pinned moved {movedPinned:.3e}, free moved "
          f"{movedFree:.3e}, force on free vertices {neighborForce:.3e}")
    return ok


def checkConstraintsWithPins():
    print("\n[3] constraints satisfied using only free vertices")
    model = buildModel()
    model.setConstraints()
    model.pinVertices([0, 1, 2, 3])
    packing = model.packing
    pins = model.getPinnedVertices()
    before = packing.positions.reshape(-1, 2)[pins].copy()
    iterations, residual = model.constraints.projectPositions(packing)
    after = packing.positions.reshape(-1, 2)[pins]
    moved = np.abs(after - before).max()
    ok = moved == 0.0 and residual < 1e-12
    print(f"      {'OK ' if ok else 'FAIL'} SHAKE {iterations} iterations -> max|C| {residual:.2e}, "
          f"pinned motion {moved:.3e}")
    return ok


def checkPinPolygonsAndRelease():
    print("\n[4] pinPolygons selection and release")
    model = buildModel()
    model.pinPolygons([2, 5])
    starts = model.packing.startIndices
    expected = np.concatenate([np.arange(starts[2], starts[3]), np.arange(starts[5], starts[6])])
    got = model.getPinnedVertices()
    selectionOk = np.array_equal(np.sort(got), np.sort(expected))

    model.pinVertices(None)
    released = model.getPinnedVertices().size == 0
    start = model.packing.positions.copy()
    model.minimizeFIRE(maxSteps = 60, fThreshold = 1e-12, dtMax = 0.05)
    everythingMoves = np.abs(model.packing.positions - start).reshape(-1, 2).max(axis = 1).min() > 0.0
    ok = selectionOk and released and everythingMoves
    print(f"      {'OK ' if ok else 'FAIL'} selection {selectionOk}, released {released}, "
          f"all vertices free to move {everythingMoves}")
    return ok


def checkNoWrapOfPinnedPolygon():
    print("\n[5] periodic wrap never relocates a pinned polygon")
    from packing import wrapPolygonsIntoCell
    model = buildModel()
    model.pinPolygons([0])
    packing = model.packing
    starts = packing.startIndices
    # Push polygon 0 well outside the cell; without the guard the wrap would translate it back.
    packing.positions.reshape(-1, 2)[starts[0]:starts[1]] += 3.0
    before = packing.positions.reshape(-1, 2)[starts[0]:starts[1]].copy()
    wrapPolygonsIntoCell(packing)
    after = packing.positions.reshape(-1, 2)[starts[0]:starts[1]]
    moved = np.abs(after - before).max()
    ok = moved == 0.0
    print(f"      {'OK ' if ok else 'FAIL'} pinned polygon displaced by {moved:.3e} under wrap")
    return ok


def main():
    print("\nPinned vertices")
    results = [checkHeldFixed(), checkFreeStillMove(), checkConstraintsWithPins(),
               checkPinPolygonsAndRelease(), checkNoWrapOfPinnedPolygon()]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
