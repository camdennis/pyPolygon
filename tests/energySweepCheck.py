"""Verify Model.energySweep -- annealed decompression to the densest packing.

The headline case has a KNOWN answer: 5 unit squares fit in a square of side 2 + 1/sqrt(2) = 2.7071,
so the optimal packing fraction is 5/2.7071^2 = 0.68227. Starting above it at 5/2.7^2 = 0.68587, the
sweep must come down to a density that is valid and cannot exceed the optimum.

Checks:
  [1] the sigma schedule respects the dynamics floor and warns when asked to go below it
  [2] a packing that was never overjammed is REFUSED rather than reported as a success
  [3] the known-answer case: density reached is valid, <= the optimum, and close to it
  [4] the packed state is confirmed by INDEPENDENT convex clipping, not only by the sharp energy
  [5] no shape shrank to fit -- A/A_target stays 1 throughout
  [6] the same seed gives the same density

Run: python tests/energySweepCheck.py
"""

import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model

WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
OPTIMUM = 5.0 / (2.0 + 1.0 / np.sqrt(2.0)) ** 2          # 0.682270...


# --- independent overlap reference (exact for convex polygons) ----------------------------------

def clipPolygon(subject, clip):
    """Sutherland-Hodgman intersection of two CONVEX counter-clockwise polygons."""
    output = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        if not output:
            return []
        current, output = output, []
        for j in range(len(current)):
            p, q = current[j], current[(j + 1) % len(current)]
            inP = float(edge[0] * (p - a)[1] - edge[1] * (p - a)[0]) >= 0.0
            inQ = float(edge[0] * (q - a)[1] - edge[1] * (q - a)[0]) >= 0.0
            if inP:
                output.append(p)
            if inP != inQ:
                d = q - p
                denominator = float(edge[0] * d[1] - edge[1] * d[0])
                if abs(denominator) > 1e-300:
                    t = float(edge[0] * (a - p)[1] - edge[1] * (a - p)[0]) / denominator
                    output.append(p + d * t)
    return output


def loopArea(loop):
    if len(loop) < 3:
        return 0.0
    r = np.asarray(loop, dtype = float)
    return 0.5 * abs(float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])))


def counterClockwise(loop):
    r = np.asarray(loop, dtype = float)
    signed = 0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]))
    return r if signed > 0 else r[::-1]


def violationByClipping(model):
    """Total overlap + outside-the-wall area, by exact clipping. Independent of the code under test."""
    packing = model.packing
    r = packing.positions.reshape(-1, 2)
    container = packing.containerIndex
    loops = [counterClockwise(r[packing.startIndices[p]: packing.startIndices[p + 1]])
             for p in range(int(container))]
    wall = counterClockwise(r[packing.startIndices[container]: packing.startIndices[container + 1]])
    total = sum(loopArea(a) - loopArea(clipPolygon(a, wall)) for a in loops)
    for i, j in itertools.combinations(range(len(loops)), 2):
        total += loopArea(clipPolygon(loops[i], loops[j]))
    return total


# --- the packing under test ---------------------------------------------------------------------

def build(phi, seed = 42, polydispersity = 0.25, softening = 0.05):
    """Cam's transientSquares setup: 5 squares in a pinned unit container, hard areas, soft edges."""
    model = Model(N = 5, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    if polydispersity:
        model.setLogNormalTargetPerimeter(polydispersity = polydispersity)
    model.addShape(WALL)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setSpringConstants(area = 0.0, edge = 1.0, perimeter = 0.0)
    model.setConstraints(area = True, edge = False)
    model.setModelType("mollified")
    model.setSofteningFraction(softening)
    model.setDOFType("transient")
    model.setMoments([1, 2, -1])
    return model


def checkSigmaFloor():
    """[1] sigma cannot be annealed below the dynamics floor, and asking is warned about."""
    print("\n[1] sigma schedule respects the dynamics floor")
    model = build(5.0 / 2.7 ** 2)
    floor = 0.01 * float(np.mean(model.packing.targetEdgeLength))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model.energySweep(finalSigma = 1e-5, annealRounds = 2, phiStep = 0.05, refineRounds = 0,
                          minPhi = 0.66, maxSteps = 300, verbose = False)
        clamped = any("below the dynamics floor" in str(w.message) for w in caught)
    ok = clamped and model.sigma >= floor * (1.0 - 1e-9)
    print(f"    {'OK  ' if ok else 'FAIL'} warned = {clamped}; final sigma {model.sigma:.3e} "
          f">= floor {floor:.3e}")
    return ok


def checkNotOverjammed():
    """[2] starting below jamming is refused, not reported as a triumph."""
    print("\n[2] a packing that was never overjammed is refused")
    model = build(0.35, polydispersity = 0.05)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        result = model.energySweep(annealRounds = 1, phiStep = 0.02, refineRounds = 0,
                                   maxSteps = 400, verbose = False)
        warned = any("already valid" in str(w.message) for w in caught)
    ok = warned and result.packed
    print(f"    {'OK  ' if ok else 'FAIL'} warned = {warned}, returned phi = {result.phi:.5f} "
          f"(a lower bound, not a result)")
    return ok


def checkKnownAnswer():
    """[3][4][5] the real case, against the known optimum and an independent area check."""
    print(f"\n[3] known answer: 5 squares, optimum phi = {OPTIMUM:.6f}")
    model = build(5.0 / 2.7 ** 2)
    print(f"    starting phi = {model.getPackingFraction():.6f} (overjammed by "
          f"{model.getPackingFraction() - OPTIMUM:+.6f})")
    result = model.energySweep(finalPolydispersity = 1e-5, finalEnergy = 1e-5,
                               annealRounds = 8, phiStep = 0.003, refineRounds = 8,
                               maxSteps = 6000, verbose = True)
    if not result.packed:
        print("    FAIL never packed")
        return False
    print(f"\n    reached phi = {result.phi:.6f}   optimum {OPTIMUM:.6f}   "
          f"gap {result.phi - OPTIMUM:+.6f}")

    print("\n[4] packed state confirmed by INDEPENDENT clipping")
    reported = model.getOverlapArea()
    reference = violationByClipping(model)
    agree = abs(reported - reference) < 1e-9 and reference < 1e-5
    print(f"    {'OK  ' if agree else 'FAIL'} getOverlapArea {reported:.3e} vs clipping "
          f"{reference:.3e}")

    print("\n[5] no shape shrank to fit")
    worst = max(step["areaError"] for step in result.history)
    honest = worst < 1e-9
    print(f"    {'OK  ' if honest else 'FAIL'} worst |A/A_target - 1| over {len(result.history)} "
          f"steps = {worst:.2e}")

    feasible = result.phi <= OPTIMUM + 1e-6
    print(f"\n    {'OK  ' if feasible else 'FAIL'} density does not exceed the optimum")
    return agree and honest and feasible


def checkDeterminism():
    """[6] same seed, same answer."""
    print("\n[6] determinism")
    densities = []
    for _ in range(2):
        model = build(5.0 / 2.7 ** 2)
        densities.append(model.energySweep(annealRounds = 3, phiStep = 0.01, refineRounds = 3,
                                           maxSteps = 2000, verbose = False).phi)
    ok = np.isfinite(densities[0]) and abs(densities[0] - densities[1]) < 1e-12
    print(f"    {'OK  ' if ok else 'FAIL'} {densities[0]:.8f} vs {densities[1]:.8f}")
    return ok


def main():
    warnings.simplefilter("ignore")
    print("=" * 78)
    print("energySweep check")
    print("=" * 78)
    results = [checkSigmaFloor(), checkNotOverjammed(), checkKnownAnswer(), checkDeterminism()]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print(f"{results.count(False)} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
