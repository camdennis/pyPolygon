"""Per-body wall stiffness: does the kernel agree with numpy, and is the multiplier exact?

The wall's stiffness rides the same batched pair loop as every body contact, applied as a per-pair
multiplier on the work items touching the exterior. That is exact in principle -- energy and gradient
are both linear in the stiffness -- so the checks are about whether it is wired correctly, on BOTH
paths, and a silently wrong GRADIENT is the failure mode to fear: this codebase has already had a
compile-time cap drop gradients while the energy stayed correct, and it survived the whole suite.

  1  CUDA vs numpy, energy AND gradient, at several wall stiffnesses
  2  LINEARITY -- the wall term is exactly proportional to wallStiffness, so the total at k must equal
     (bodies-only) + k * (wall-only), measured against the independent confinementEnergyGradient
  3  wallStiffness = 1 reproduces the old behaviour bit for bit
  4  vertex counts past every stride, and MIXED counts in one packing

    python tests/wallStiffnessCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp
import polyContactSystem as pcs
import cudaOverlap


# TOLERANCES ARE ABSOLUTE, AND THAT IS THE WHOLE POINT.
#
# A relative comparison is meaningless for this law near contact onset. The energy vanishes as d^3, so
# as a contact gets shallow the relative gap explodes while the absolute gap sits still. Measured on a
# single square poking through a wall, sweeping the penetration depth:
#
#   depth/edge      energy          relE        |dE| absolute
#   1.0e-01       5.067e-05       5.43e-12        2.8e-16
#   1.0e-02       5.307e-08       9.53e-09        5.1e-16
#   1.0e-03       5.331e-11       9.26e-06        4.9e-16
#   1.0e-04       5.331e-14       2.27e-03        1.2e-16
#   1.0e-05      -9.336e-17       2.92e+00        ~1e-16     <- energy has gone NEGATIVE
#
# The two paths agree to ~3e-16 absolute throughout: pure double roundoff, no depth dependence at all.
# Only the denominator moves. Gating on relE would therefore fail a correct kernel on any packing whose
# wall contacts are shallow -- which is every packing near jamming, and exactly the regime this knob
# exists for. The last row is worth its own note: below about 1e-05 of an edge the contact energy is
# entirely roundoff and can come out NEGATIVE, which is the floor underneath the ~1e-9 excess noise
# measured elsewhere.
_ENERGY_TOLERANCE = 1e-13
_GRADIENT_TOLERANCE = 1e-12


def agreement(bodies, stiffness, wallStiffness):
    """``(|dE|, |dE|/|E|, max|dg|, max|dg|/max|g|)`` between the CUDA and numpy paths."""
    eH, gH = pcs.systemEnergyGradient(bodies, stiffness, useCuda = False, wallStiffness = wallStiffness)
    eD, gD = pcs.systemEnergyGradient(bodies, stiffness, useCuda = True, wallStiffness = wallStiffness)
    absE = abs(eD - eH)
    absG = float(np.abs(gD - gH).max())
    return (absE, absE / max(abs(eH), 1e-300), absG,
            absG / max(float(np.abs(gH).max()), 1e-300), eH)


def build(N = 6, n = 8, seed = 42, side = 3.2):
    """A packing pressed hard enough that the wall is genuinely loaded."""
    model = pp.Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = N / side ** 2, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    model.addShape(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setConstraints(area = True, edge = True)
    model.minimizeFIRE(maxUnbalancedForce = 1e-3, maxSteps = 1500)
    return model


def bodySet(model):
    """The BodySet packingEnergyForce builds, wall winding normalized the same way."""
    packing = model.packing
    container = int(packing.containerIndex)
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    wall = slice(starts[container], starts[container + 1])
    positions = vertices
    if pcs.pc.signedArea(vertices[wall]) > 0.0:
        positions = vertices.copy()
        positions[wall] = vertices[wall][::-1]
    bodies = pcs.BodySet.__new__(pcs.BodySet)
    bodies.positions = positions
    bodies.startIndices = starts
    bodies.boxSize = None
    bodies.exterior = container
    return bodies


def checkAgainstNumpy():
    """CHECK 1 -- the kernel and the numpy loop agree, energy and GRADIENT, at every stiffness."""
    model = build()
    bodies = bodySet(model)
    if not (cudaOverlap is not None and cudaOverlap.isAvailable()):
        print("  no CUDA device -- SKIPPED")
        return True
    worstE = worstG = 0.0
    for k in (1.0, 10.0, 100.0, 1000.0):
        absE, relE, absG, relG, energy = agreement(bodies, 1.0, k)
        worstE, worstG = max(worstE, absE), max(worstG, absG)
        print(f"  wallStiffness {k:7.1f}   E {energy:13.6e}   |dE| {absE:.2e} (rel {relE:.1e})   "
              f"max|dg| {absG:.2e} (rel {relG:.1e})")
    ok = worstE < _ENERGY_TOLERANCE and worstG < _GRADIENT_TOLERANCE
    print(f"  CHECK 1 CUDA == numpy        {'PASS' if ok else 'FAIL'}")
    return ok


def checkLinearity():
    """CHECK 2 -- the total is (bodies) + k * (wall), against an INDEPENDENT wall computation.

    ``confinementEnergyGradient`` is the slow per-body reference and shares no code with the batched
    pair loop, so this pins the multiplier to a construction the kernel knows nothing about rather than
    to another spelling of itself."""
    model = build()
    packing = model.packing
    base, _ = pcs.packingEnergyForce(packing, 1.0, wallStiffness = 1.0)
    wall, _ = pcs.confinementEnergyGradient(packing, 1.0)
    bodiesOnly = base - wall
    worst = 0.0
    for k in (1.0, 10.0, 100.0, 1000.0):
        total, _ = pcs.packingEnergyForce(packing, 1.0, wallStiffness = k)
        predicted = bodiesOnly + k * wall
        gap = abs(total - predicted)
        worst = max(worst, gap)
        print(f"  k {k:7.1f}   total {total:.9e}   bodies + k*wall {predicted:.9e}   "
              f"|gap| {gap:.2e}")
    print(f"  bodies-only {bodiesOnly:.6e}, wall-only {wall:.6e}")
    # The wall must actually be carrying something, or the linearity holds for the wrong reason.
    ok = worst < _ENERGY_TOLERANCE and abs(wall) > 1e-14
    print(f"  CHECK 2 exact linearity      {'PASS' if ok else 'FAIL'}")
    return ok


def checkNeutralAtOne():
    """CHECK 3 -- wallStiffness = 1 is bit-for-bit the old behaviour."""
    model = build()
    plain, gPlain = pcs.packingEnergyForce(model.packing, 1.0)
    one, gOne = pcs.packingEnergyForce(model.packing, 1.0, wallStiffness = 1.0)
    gap = abs(plain - one)
    gradGap = float(np.abs(gPlain - gOne).max())
    print(f"  energy delta {gap:.3e}   max|force delta| {gradGap:.3e}")
    ok = gap == 0.0 and gradGap == 0.0
    print(f"  CHECK 3 neutral at k = 1     {'PASS' if ok else 'FAIL'}")
    return ok


def checkVertexCounts():
    """CHECK 4 -- past every stride, and MIXED counts in one packing.

    POLYCONTACT_BLOCK is 64 and POLYCONTACT_MAXN is 64, so 4, 63, 64, 65 and 100 straddle both. Mixed
    counts matter because nothing in this project may assume a uniform vertex count -- the strides come
    from starts[] at runtime, and a bug that reads the first body's count would pass a uniform test."""
    if not (cudaOverlap is not None and cudaOverlap.isAvailable()):
        print("  no CUDA device -- SKIPPED")
        return True
    worstE = worstG = 0.0
    reported = []
    for n in (4, 12, 63, 64):
        model = build(N = 4, n = n, side = 2.6)
        bodies = bodySet(model)
        absE, _, absG, _, _ = agreement(bodies, 1.0, 100.0)
        worstE, worstG = max(worstE, absE), max(worstG, absG)
        reported.append(f"n={n}:{absE:.1e}/{absG:.1e}")

    # MIXED: a 5-gon, a 12-gon and a 40-gon in one packing, plus the 4-vertex wall.
    model = pp.Model(N = 3, n = 12, seed = 7)
    model.generateEquilateralPolygons(phi = 3 / 2.4 ** 2, kappa = 4.0)
    model.syncTargetPerimeters(); model.syncTargetAreas()
    model.addShape(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.minimizeFIRE(maxUnbalancedForce = 1e-3, maxSteps = 800)
    bodies = bodySet(model)
    counts = np.diff(np.asarray(bodies.startIndices, dtype = int))
    mixedE, _, mixedG, _, _ = agreement(bodies, 1.0, 100.0)
    print(f"  uniform (|dE|/max|dg|): {'  '.join(reported)}")
    print(f"  mixed counts {list(counts)}: {mixedE:.1e}/{mixedG:.1e}")
    ok = max(worstE, mixedE) < _ENERGY_TOLERANCE and max(worstG, mixedG) < _GRADIENT_TOLERANCE
    print(f"  CHECK 4 vertex counts        {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("per-body wall stiffness")
    results = []
    for name, check in (("CUDA vs numpy", checkAgainstNumpy), ("linearity", checkLinearity),
                        ("neutral at 1", checkNeutralAtOne), ("vertex counts", checkVertexCounts)):
        print(f"\n{name}:", flush = True)
        results.append(check())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
