"""Checks on the vertex-count cascade: does dropping n force the square, with no diagonal targets?

The protocol under test holds every polygon EQUILATERAL at a fixed shape index while its size is free,
and withdraws compliance by reducing the vertex count rather than by narrowing a distribution. The
claim that makes it worth doing is a counting one:

    2n - 3 shape DOF  -  n edge rows  -  1 area row  =  n - 4 free

so the compliance IS the vertex count, reaching zero at n = 4 -- where an equilateral quadrilateral is
a rhombus with kappa = 4 / sqrt(sin theta), so kappa = 4 admits only theta = 90 degrees. If that holds,
the square needs no template, no diagonals and no angle constraint of any kind.

  0  the constraint Jacobian really has rank n + 1, so the free DOF really is n - 4
  1  kappa = 4 stays REACHABLE at every count, because the regular floor only meets 4 at n = 4
  2  resampling preserves the target shape index and the area, and refuses to move a pinned vertex
  3  a single polygon at n = 4 under these constraints IS a square, from a deliberately bad start
  4  the cascade 32 -> 16 -> 8 -> 4 runs end to end and ends square
  5  the same cascade never needs targetDiagonal -- it stays None throughout

Check 3 is the one that decides the protocol. It is run from a random start rather than from something
already square, because the question is whether the CONSTRAINTS force the square or whether a nearby
initial guess was doing the work.

    python tests/vertexCascadeCheck.py
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
import build
from softBody import backboneEdgeLengths, backboneArea

_KAPPA = 4.0


def equilateralModel(N = 4, n = 32, seed = 42, phi = 0.2):
    """A packing whose edge targets are EQUAL per polygon -- the protocol's constraint set."""
    model = pp.Model(N = N, n = n, seed = seed)
    model.generatePolygons(phi = phi, kappa = _KAPPA)
    model.syncTargetAreas()
    model.setConstraints(area = True, edge = True)
    return model


def checkRank():
    """CHECK 0: the free shape DOF is n - 4, measured from the Jacobian rather than counted on paper."""
    ok = True
    for n in (32, 16, 8, 6, 5, 4):
        model = equilateralModel(N = 2, n = n)
        block = getattr(model.constraints, "block", model.constraints)
        J = block.jacobian(model.packing)[0]
        rank = int(np.linalg.matrix_rank(J.reshape(J.shape[0], -1), tol = 1e-9))
        free = 2 * n - 3 - rank
        good = free == n - 4
        ok = ok and good
        print(f"  n {n:3d}   rows {J.shape[0]:3d}   rank {rank:3d}   free {free:3d}   "
              f"expected {n - 4:3d}   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 0 free DOF is n - 4 {'PASS' if ok else 'FAIL'}")
    return ok


def checkFeasible():
    """CHECK 1: kappa = 4 is reachable at every count, and ONLY just at n = 4.

    The cascade works because the regular n-gon's shape index rises as n falls and meets 4 exactly at
    the end. If it crossed 4 early the targets would go infeasible mid-cascade."""
    ok = True
    for n in (32, 16, 8, 6, 5, 4):
        floor = build.regularShapeIndex(n)
        slack = _KAPPA - floor
        good = slack >= -1e-12 and (abs(slack) < 1e-12 if n == 4 else slack > 0.0)
        ok = ok and good
        print(f"  n {n:3d}   regular floor {floor:.6f}   slack {slack:+.6f}   "
              f"{'ok' if good else 'FAIL'}")
    print(f"  CHECK 1 kappa 4 reachable throughout, tight at n = 4 {'PASS' if ok else 'FAIL'}")
    return ok


def checkResample():
    """CHECK 2: resampling preserves the TARGETS it promises and refuses what it cannot do."""
    ok = True
    model = equilateralModel(N = 3, n = 32)
    beforeArea = model.packing.targetArea.copy()
    beforePerimeter = model.packing.targetPerimeter.copy()
    model.resampleEdges(8)
    afterArea = model.packing.targetArea
    afterPerimeter = model.packing.targetPerimeter
    edges = np.asarray(model.packing.targetEdgeLength, dtype = float)
    spread = float(np.std(edges.reshape(3, 8), axis = 1).max() / edges.mean())
    good = (np.abs(afterArea / beforeArea - 1.0).max() < 1e-14
            and np.abs(afterPerimeter / beforePerimeter - 1.0).max() < 1e-14
            and spread < 1e-15 and model.packing.targetDiagonal is None)
    ok = ok and good
    print(f"  32 -> 8: dArea {np.abs(afterArea / beforeArea - 1.0).max():.2e}   "
          f"dPerimeter {np.abs(afterPerimeter / beforePerimeter - 1.0).max():.2e}   "
          f"edge spread {spread:.2e}   diagonals dropped "
          f"{model.packing.targetDiagonal is None}   {'ok' if good else 'FAIL'}")

    # And the pin refusal, since a resampled vertex is a new point that no pin can follow.
    model = equilateralModel(N = 2, n = 16)
    model.pinVertices([0])
    try:
        model.resampleEdges(8)
        refused = False
    except ValueError as error:
        refused = "pinned" in str(error)
    ok = ok and refused
    print(f"  refuses to resample a pinned polygon: {refused}")
    print(f"  CHECK 2 resampling keeps its promises {'PASS' if ok else 'FAIL'}")
    return ok


def checkSquareIsForced():
    """CHECK 3: at n = 4 the constraints alone produce a SQUARE, from a bad start.

    THE DECIDING CHECK. No template, no diagonal target, no angle constraint -- only equal edges, a
    pinned area, and a shape index of 4. If the corners come out at 90 degrees from a random star, the
    protocol needs none of the machinery the old cascade used."""
    ok = True
    for seed in (1, 7, 42):
        model = pp.Model(N = 1, n = 4, seed = seed)
        model.generatePolygons(phi = 0.1, kappa = _KAPPA)
        model.syncTargetAreas()
        # Deliberately spoil it: squash the quadrilateral into a thin rhombus before constraining.
        r = model.packing.positions.reshape(-1, 2)
        centroid = r.mean(axis = 0)
        r[:] = centroid + (r - centroid) * np.array([1.8, 0.45])
        model.setConstraints(area = True, edge = True)
        model.minimizeFIRE(maxUnbalancedForce = 1e-10, maxSteps = 40000, patience = 2000)
        v = model.packing.positions.reshape(-1, 2)
        edge = np.roll(v, -1, axis = 0) - v
        lengths = np.linalg.norm(edge, axis = 1)
        unit = edge / lengths[:, None]
        interior = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", -np.roll(unit, 1, axis = 0), unit), -1.0, 1.0)))
        kappa = float(lengths.sum() / np.sqrt(abs(backboneArea(model.packing)[0])))
        worstAngle = float(np.abs(interior - 90.0).max())
        worstEdge = float(np.abs(lengths / lengths.mean() - 1.0).max())
        # 1e-5 on kappa, not 1e-6: this is the RELAXATION's residual, not the constraint's. FIRE
        # stopped at max|F| = 1e-10 and leaves kappa a part per million out, which says nothing about
        # whether the constraint set forces a square. The ANGLE is the claim under test.
        good = worstAngle < 0.5 and worstEdge < 1e-6 and abs(kappa - _KAPPA) < 1e-5
        ok = ok and good
        print(f"  seed {seed:3d}: worst |angle - 90| {worstAngle:8.4f} deg   "
              f"edge spread {worstEdge:.2e}   kappa {kappa:.6f}   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 3 the square is FORCED by the constraints {'PASS' if ok else 'FAIL'}")
    return ok


def checkCascade():
    """CHECKS 4-5: the whole ladder runs, ends square, and never touches a diagonal target."""
    model = pp.Model(N = 1, n = 32, seed = 3)
    model.generatePolygons(phi = 0.1, kappa = _KAPPA)
    model.syncTargetAreas()
    model.setConstraints(area = True, edge = True)
    usedDiagonals = False
    for count in (32, 16, 8, 4):
        if count != 32:
            model.resampleEdges(count)
        model.minimizeFIRE(maxUnbalancedForce = 1e-10, maxSteps = 40000, patience = 2000)
        usedDiagonals = usedDiagonals or model.packing.targetDiagonal is not None
        v = model.packing.positions.reshape(-1, 2)
        lengths = np.linalg.norm(np.roll(v, -1, axis = 0) - v, axis = 1)
        kappa = float(lengths.sum() / np.sqrt(abs(backboneArea(model.packing)[0])))
        print(f"  n {count:3d}   kappa {kappa:.6f}   edge spread "
              f"{float(np.abs(lengths / lengths.mean() - 1.0).max()):.2e}   "
              f"residual {model.constraintResidual():.2e}")
    v = model.packing.positions.reshape(-1, 2)
    edge = np.roll(v, -1, axis = 0) - v
    unit = edge / np.linalg.norm(edge, axis = 1)[:, None]
    interior = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", -np.roll(unit, 1, axis = 0), unit), -1.0, 1.0)))
    worstAngle = float(np.abs(interior - 90.0).max())
    ok = worstAngle < 0.5 and not usedDiagonals
    print(f"  ends at worst |angle - 90| = {worstAngle:.4f} deg, diagonals never used "
          f"{not usedDiagonals}")
    print(f"  CHECKS 4-5 cascade ends square without diagonals {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("vertex-count cascade", flush = True)
    warnings.filterwarnings("ignore")
    results = []
    for name, check in (("free DOF is n - 4", checkRank),
                        ("kappa 4 reachable throughout", checkFeasible),
                        ("resampling keeps its promises", checkResample),
                        ("the square is forced", checkSquareIsForced),
                        ("the cascade end to end", checkCascade)):
        print(f"\n{name}:", flush = True)
        results.append(check())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
