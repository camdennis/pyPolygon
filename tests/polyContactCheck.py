"""Contract for ``polyContact.py`` -- the vectorized port of the polygon-contact reference.

Every test corresponds to a numbered validation item in ``notes/polygonContact/contact.pdf`` sec 13,
ported from the handoff's ``tests/test_reference.py``. Several are DELIBERATELY ADVERSARIAL: they exist
because a plausible wrong implementation passes the obvious test and fails these. Marked [TRAP].
DO NOT WEAKEN THEM.

Ground truth is ``tests/polyContactReference.py``, vendored verbatim. Every numerical check here runs
against BOTH the reference and, where it is the stronger test, a finite difference or an independent
construction. The handoff's warning is worth repeating: conservation is nearly worthless as a test --
net force and torque vanish STRUCTURALLY, and passed on every buggy intermediate including one with a
48% error. Only finite differencing localizes anything.

ONE PORT BUG WAS FOUND BY THIS SUITE and is worth recording, because it is invisible to everything
except the finite difference: ``march`` returns exact breakpoints but an unreliable WINNER list -- it
identifies the winner at ``t + 1e-13``, which does not separate two candidates that cross shallowly.
Trusting it reported (E3, E3) where the truth was (E3, E1) on crossed bars perturbed by 1e-5, inflating
the energy EIGHTFOLD while the breakpoints stayed correct to 1e-9. The features are now re-identified at
sub-stretch midpoints, which is what the reference does.

Run: python tests/polyContactCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polyContact as pc
import polyContactReference as ref


FAILURES = []


def check(name, got, want, tolerance):
    ok = abs(got - want) <= tolerance
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:54s} got {got:.8g}  want {want:.8g}")
    if not ok:
        FAILURES.append(name)


def checkTrue(name, condition, detail = ""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name:54s} {detail}")
    if not condition:
        FAILURES.append(name)


def rotation(degrees):
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


GRADIENT_CASES = {
    "parallel faces, |e|=1": (ref.rect(0, 0, 1, 1), ref.rect(-1, 0.85, 2, 1.9)),
    "parallel faces, |e|=2  [TRAP]": (ref.rect(0, 0, 2, 1), ref.rect(-1, 0.85, 3, 1.9)),
    "rotated 30 deg": (ref.rect(0, 0, 1, 1) @ rotation(30), ref.rect(-2, 0.85, 2, 3)),
    "no vertex of A in B": (ref.rect(0, 0, 1, 1), ref.rect(0.3, 0.85, 0.75, 1.9)),
    "vertex-on-face": (ref.rect(0, 0, 1, 1), ref.rect(0.9, 0.88, 2.0, 2.0)),
    "crossed bars": (ref.rect(0, 0.4, 1, 0.6), ref.rect(0.4, 0, 0.6, 1.0)),
    "L vs cross [nonconvex]": (ref.L_shape(), ref.place(ref.cross_shape(), 1.5, 1.50, 0.35)),
    "L vs cross rotated [nonconvex]": (ref.L_shape(), ref.place(ref.cross_shape(), 1.6, 1.49, 0.70)),
}


def checkVertexNearestGradient():
    """4c. The gradient on sub-stretches whose nearest feature is a VERTEX. [COVERAGE HOLE, 2026-08-09]

    Every case in GRADIENT_CASES has ZERO vertex-nearest sub-stretches -- including the two labelled
    nonconvex -- so that branch of ``pairGradient`` went undifferenced, and a missing factor of ``1/3``
    lived in it (and in the vendored reference, and in the CUDA port). It survived because the branch
    fires only when the LOOP body presents a reflex vertex to the boundary body, which needs the
    obstacle to be nonconvex WHERE CONTACT HAPPENS, not merely nonconvex somewhere.

    A confining wall is the natural such case: the exterior of a box has four reflex corners. This
    checks the closed-form gradient against a finite difference of the closed-form energy, and asserts
    that the branch was actually reached -- an FD check that silently exercises nothing is the failure
    being guarded against."""
    print("\n4c. gradient where the nearest feature is a VERTEX   [was never covered]")
    exterior = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    cases = {
        "square into a box corner": np.array([[-0.035, -0.035], [0.265, -0.035],
                                              [0.265, 0.265], [-0.035, 0.265]]),
        "square into a box corner, deeper": np.array([[-0.09, -0.09], [0.21, -0.09],
                                                      [0.21, 0.21], [-0.09, 0.21]]),
    }
    for name, loop in cases.items():
        _, _, _, kind, _, _ = pc._substretches(loop, exterior,
                                               pc.edgeFrame(loop), pc.edgeFrame(exterior))
        hits = int((kind == 1).sum())
        checkTrue(f"branch reached  {name}", hits > 0, f"{hits} vertex-nearest sub-stretches")

        energy, gradientA, _ = pc.pairGradient(loop.copy(), exterior.copy())
        direct = pc.pairEnergy(loop.copy(), exterior.copy())
        checkTrue(f"energy self-consistent  {name}",
                  abs(energy - direct) <= 1e-14 * max(abs(direct), 1e-300),
                  f"pairGradient {energy:.9e} vs pairEnergy {direct:.9e}")
        step, worst = 1e-6, 0.0
        for vertex in range(len(loop)):
            for component in range(2):
                moved = loop.copy()
                original = moved[vertex, component]
                moved[vertex, component] = original + step
                plus = pc.pairEnergy(moved, exterior)
                moved[vertex, component] = original - step
                minus = pc.pairEnergy(moved, exterior)
                worst = max(worst, abs(gradientA[vertex, component]
                                       - (plus - minus) / (2.0 * step)))
        checkTrue(f"grad==FD  {name}", worst < 1e-7, f"max|dg|={worst:.2e}")


def checkBuriedArclength():
    """1. Span arclength on a pair with NO vertex of A inside B. [TRAP]

    Any vertex-sampled rule returns ~0 here while the truth is O(1). This is the same failure found
    independently in softDepth on 2026-07-31, and the handoff lists it among the rejected alternatives:
    vertex sampling is blind to shallow face-on-face contact and INVERTS the face/vertex contrast."""
    print("\n1. span arclength with NO vertex of A inside B   [TRAP]")
    loopA, loopB = ref.rect(0, 0, 1, 1), ref.rect(0.3, 0.9, 0.7, 1.9)
    vectors, _, _, _ = pc.edgeFrame(loopA)
    edges, low, high = pc.spans(loopA, loopB)
    chord = float(np.sum((high - low) * np.linalg.norm(vectors[edges], axis = 1)))
    _, _, _, inside = pc.nearestFeature(loopA, loopB)
    checkTrue("no vertex of A lies inside B", int(inside.sum()) == 0, f"count={int(inside.sum())}")
    check("buried arclength of dA in B", chord, 0.40, 1e-12)


def checkFaceVertexContrast():
    """2. Face-on-face against vertex-on-face at equal depth."""
    print("\n2. face-on-face vs vertex-on-face contrast at equal depth")
    face = pc.contactEnergy(ref.rect(0, 0, 1, 1), ref.rect(0.2, 0.9, 0.8, 1.9))
    vertex = pc.contactEnergy(ref.rect(0, 0, 1, 1), ref.rect(0.9, 0.9, 2.0, 2.0))
    checkTrue("face/vertex ratio is O(1) and > 1", 2.0 < face / vertex < 40.0,
              f"ratio={face / vertex:.3f}")


def checkClosedFormAgainstQuadrature():
    """3. Closed form against an INDEPENDENT quadrature, convex and nonconvex."""
    print("\n3. closed form vs independent quadrature, convex and NONCONVEX")
    pairs = {
        "face-on-face": (ref.rect(0, 0, 1, 1), ref.rect(0.3, 0.9, 0.7, 1.9)),
        "crossed bars": (ref.rect(0, 0.4, 1, 0.6), ref.rect(0.4, 0, 0.6, 1.0)),
        "vertex-on-face": (ref.rect(0, 0, 1, 1), ref.rect(0.9, 0.9, 2.0, 2.0)),
        "deep overlap": (ref.rect(0, 0, 1, 1), ref.rect(0.45, 0.45, 1.45, 1.45)),
        "L vs cross [nonconvex]": (ref.L_shape(), ref.place(ref.cross_shape(), 1.5, 1.50, 0.35)),
        "L vs cross rotated [nonconvex]": (ref.L_shape(),
                                           ref.place(ref.cross_shape(), 1.6, 1.49, 0.70)),
    }
    for name, (first, second) in pairs.items():
        mine = pc.pairEnergy(first, second)
        quadrature = ref.E_pair_quad(first, second, ng = 48)
        reference = ref.E_pair_closed(first, second)
        againstQuad = abs(mine - quadrature) / max(quadrature, 1e-30)
        againstRef = abs(mine - reference) / max(reference, 1e-30)
        checkTrue(f"closed==quad  {name}", againstQuad < 2e-6 and againstRef < 1e-10,
                  f"quad {againstQuad:.2e}  ref {againstRef:.2e}")


def checkGradientAgainstFiniteDifference():
    """4. Analytic gradient against central finite differences of THIS implementation's energy. [TRAP]

    T4: whatever is evaluated for the energy is what must be differentiated -- so the difference is
    taken of ``pairEnergy``, not of the reference's. The case list must keep a non-unit contacting edge
    (a missing arclength factor is EXACT when |e|=1, hiding a 48% error) and an exactly face-parallel
    pair (a 1/m antiderivative is exact there and cancels catastrophically nearby)."""
    print("\n4. analytic gradient vs central finite differences   [TRAP]")
    step = 1e-5
    for name, (first, second) in GRADIENT_CASES.items():
        first, second = first.copy(), second.copy()
        _, gradientA, gradientB = pc.pairGradient(first, second)
        worst = 0.0
        for array, gradient in ((first, gradientA), (second, gradientB)):
            for vertex in range(len(array)):
                for component in range(2):
                    original = array[vertex, component]
                    array[vertex, component] = original + step
                    plus = pc.pairEnergy(first, second)
                    array[vertex, component] = original - step
                    minus = pc.pairEnergy(first, second)
                    array[vertex, component] = original
                    worst = max(worst, abs(gradient[vertex, component]
                                           - (plus - minus) / (2.0 * step)))
        checkTrue(f"grad==FD  {name}", worst < 1e-9, f"max|dg|={worst:.2e}")


def checkGradientMatchesReference():
    """4a. The port's gradient equals the reference's, entry by entry."""
    print("\n4a. gradient equals the reference gradient, entry by entry")
    worstOverall = 0.0
    for name, (first, second) in GRADIENT_CASES.items():
        energy, gradientA, gradientB = pc.pairGradient(first.copy(), second.copy())
        refEnergy, refA, refB = ref.grad_pair(first.copy(), second.copy())
        worst = max(float(np.abs(gradientA - refA).max()), float(np.abs(gradientB - refB).max()),
                    abs(energy - refEnergy))
        worstOverall = max(worstOverall, worst)
    checkTrue("port == reference on every gradient case", worstOverall < 1e-14,
              f"max|difference|={worstOverall:.2e}")


def checkMedialAxisBreaksTheGradient():
    """4b. The CONTACT MUST go attractive past the medial axis. [negative control]

    WHAT THIS USED TO TEST, AND WHY THAT WAS WRONG. It differenced the analytic gradient against finite
    differences of the energy and demanded a MISMATCH, on the theory that an invalid state "has no
    gradient". That conflates two things. Finite differences of an energy can only ever expose an error
    in the DERIVATIVE of that energy -- they are silent about whether the energy itself is the right
    quantity. Past the ridge the derivative is fine; it is the ENERGY that is wrong, because the depth
    it integrates starts to fall as the bodies are pushed further together.

    So the old control was measuring bookkeeping and calling it physics, and it only ever "passed"
    because the gradient had a bug: the vertex-nearest branch was 3x too large (the missing /3, fixed
    2026-08-09). Once the gradient became self-consistent with its own energy, the mismatch vanished --
    max|dg| = 2.38e-11 -- and this control failed while reporting nothing real. Its passing state had
    been evidence of a defect elsewhere.

    WHAT IT TESTS NOW is the actual pathology, which needs no reference solution: push the bodies
    FURTHER into each other and the energy must RISE. Where it falls, the law is pulling them through,
    and that is exactly what ``dMax / rIn << 1`` exists to forbid. Measured along the approach:

        offset   dMax/rIn      energy      dE/d(approach)
          2.00     2.60      6.343e-04       3.377e-03     repulsive
          1.80     2.60      3.300e-04      -4.925e-04     ATTRACTIVE
          1.40     3.12      4.448e-04       1.626e-15     a turning point

    Note the 1.40 configuration the old control used sits ON the stationary point between the two,
    which is the least informative place on the curve to have been sampling."""
    print("\n4b. contact MUST go attractive past the medial axis   [negative control]")
    first = ref.L_shape()
    approach, step = 1.8, 1e-4
    second = ref.place(ref.cross_shape(), approach, 0.9, 0.3)
    ratio = pc.maximumDepth(first, second, 400) / 0.16
    closer = ref.place(ref.cross_shape(), approach - step, 0.9, 0.3)
    farther = ref.place(ref.cross_shape(), approach + step, 0.9, 0.3)
    slope = (pc.pairEnergy(first.copy(), closer)
             - pc.pairEnergy(first.copy(), farther)) / (2.0 * step)
    checkTrue("invalid state is flagged by dMax/rIn", ratio > 1.0, f"dMax/rIn={ratio:.2f}")
    checkTrue("repulsion REVERSES there (as designed)", slope < 0.0,
              f"dE/d(approach)={slope:.2e}")

    # And the complement: a VALID contact must repel, or the control above would be satisfied by a law
    # that is simply broken everywhere.
    valid = ref.place(ref.cross_shape(), 2.4, 0.9, 0.3)
    validRatio = pc.maximumDepth(first, valid, 400) / 0.16
    nearer = ref.place(ref.cross_shape(), 2.4 - step, 0.9, 0.3)
    further = ref.place(ref.cross_shape(), 2.4 + step, 0.9, 0.3)
    validSlope = (pc.pairEnergy(first.copy(), nearer)
                  - pc.pairEnergy(first.copy(), further)) / (2.0 * step)
    checkTrue("a shallow contact still REPELS", validSlope > 0.0,
              f"dMax/rIn={validRatio:.2f}, dE/d(approach)={validSlope:.2e}")


def checkConservation():
    """5. Conservation -- necessary but WEAK. It passes on buggy gradients too."""
    print("\n5. conservation (necessary but WEAK -- passes on buggy gradients too)")
    for name, (first, second) in list(GRADIENT_CASES.items())[-2:]:
        _, gradientA, gradientB = pc.pairGradient(first.copy(), second.copy())
        scale = max(float(np.abs(gradientA).max()), float(np.abs(gradientB).max()))
        force = float(np.abs(gradientA.sum(axis = 0) + gradientB.sum(axis = 0)).max()) / scale
        torque = abs(float(np.cross(first, gradientA).sum() + np.cross(second, gradientB).sum())) / scale
        checkTrue(f"net force ~ 0   {name}", force < 1e-13, f"{force:.2e}")
        checkTrue(f"net torque ~ 0  {name}", torque < 1e-13, f"{torque:.2e}")


def checkMembership():
    """6. The membership rule against ray-cast parity. [TRAP]

    The REFLEX clause is the whole content of the rule; a convex-only test passes it trivially."""
    print("\n6. membership rule vs ray-cast parity   [TRAP]")
    generator = np.random.default_rng(1)
    shapes = [("square (convex)", ref.rect(0, 0, 1, 1)),
              ("L-shape (1 reflex)", ref.L_shape()),
              ("cross (4 reflex)", ref.cross_shape()),
              ("flower(40) [many reflex]", ref.flower(40, 1.0, 0.35, 5))]
    for name, loop in shapes:
        low, high = loop.min(axis = 0) - 0.2, loop.max(axis = 0) + 0.2
        points = generator.uniform(low, high, size = (1500, 2))
        kind, _, distance, inside = pc.nearestFeature(points, loop)
        parity = pc.insideParity(points, loop)
        away = distance >= 1e-9
        mismatches = int(np.sum(inside[away] != parity[away]))
        checkTrue(f"membership  {name}", mismatches == 0,
                  f"mismatches={mismatches}, vertex-nearest hits={int(kind[away].sum())}")


def checkMarchAgainstBisection():
    """7. The output-sensitive march against the reference's sample-and-bisect partition.

    Only the BREAKPOINTS are compared. march's winner list is not reliable near a shallow crossing --
    see the module docstring -- which is why the production path re-identifies features at midpoints."""
    print("\n7. march == bisection for the feature partition")
    cross = ref.cross_shape()
    chords = [("shallow chord", np.array([-0.40, 0.10]), np.array([0.80, 0.0])),
              ("deep chord", np.array([-0.40, 0.00]), np.array([0.80, 0.0]))]
    for name, start, vector in chords:
        breakpoints, _, _ = pc.march(start, vector, cross, 0.0, 1.0)
        keep = [breakpoints[0]]
        for m in range(1, len(breakpoints) - 1):
            leftKind, leftIndex, _, _ = pc.nearestFeature(
                start + 0.5 * (breakpoints[m - 1] + breakpoints[m]) * vector, cross)
            rightKind, rightIndex, _, _ = pc.nearestFeature(
                start + 0.5 * (breakpoints[m] + breakpoints[m + 1]) * vector, cross)
            if (leftKind[0], leftIndex[0]) != (rightKind[0], rightIndex[0]):
                keep.append(breakpoints[m])
        keep.append(breakpoints[-1])
        expected = ref.feature_partition(None, cross, 0, start, vector, 0.0, 1.0)
        ok = len(keep) == len(expected) and np.abs(np.asarray(keep) - expected).max() < 1e-10
        checkTrue(f"march==bisect  {name}", ok, f"{len(keep) - 2} switches")


def checkMarchFindsEverySwitch():
    """7b. The march must find EVERY genuine feature switch, against a dense scan.   [TRAP]

    The reference's suite cannot catch a missed switch: ``E_pair_closed`` partitions by bisection, not
    by ``march``, and its march test compares two hand-picked chords of one shape. The reference's
    ``march`` DOES miss switches -- it shrinks the candidate array after prefiltering and then looks
    for crossings only among survivors, but a pruned candidate can still win later in the interval.
    Measured on a 9-body 12-gon packing: 5 genuine switches missed across 107 spans, moving the total
    energy 0.7%.

    A spurious EXTRA breakpoint is harmless (subdividing a stretch of constant nearest feature changes
    nothing), so this only asserts that nothing is MISSED."""
    print("\n7b. march finds every genuine feature switch, vs a dense scan   [TRAP]")
    generator = np.random.default_rng(4)
    shapes = [ref.cross_shape(), ref.flower(16, 0.6, 0.30, 5), ref.L_shape()]
    missed = 0
    chords = 0
    for shape in shapes:
        low, high = shape.min(axis = 0), shape.max(axis = 0)
        frame = pc.edgeFrame(shape)
        for _ in range(40):
            start = generator.uniform(low - 0.1, high + 0.1)
            vector = generator.uniform(-1.0, 1.0, 2) * (high - low)
            chords += 1
            breakpoints, _, _ = pc.march(start, vector, shape, 0.0, 1.0, frame = frame)
            scan = np.linspace(0.0, 1.0, 2001)
            kinds, indices, _, _ = pc.nearestFeature(start + scan[:, None] * vector, shape,
                                                     frame = frame)
            switches = np.nonzero((kinds[1:] != kinds[:-1]) | (indices[1:] != indices[:-1]))[0]
            for position in switches:
                middle = 0.5 * (scan[position] + scan[position + 1])
                if np.abs(breakpoints - middle).min() > 2.0 / 2000:
                    missed += 1
    checkTrue("no genuine switch is missed", missed == 0,
              f"{missed} missed over {chords} random chords")


def checkRealizingPointCriterion():
    """8. Realizing-point separation, not tie-counting, separates C1 from the medial axis. [TRAP]"""
    print("\n8. realizing-point criterion: C1 switch vs medial axis   [TRAP]")
    loop = ref.L_shape()
    point = np.array([1.0, 0.9])
    _, lengths, tangents, normals = pc.edgeFrame(loop)
    candidates = sorted(
        [("E", j, abs(normals[j] @ (loop[j] - point))) for j in range(len(loop))
         if 0 <= tangents[j] @ (point - loop[j]) <= lengths[j]]
        + [("V", j, np.linalg.norm(point - loop[j])) for j in range(len(loop))],
        key = lambda item: item[2])[:2]
    separation = np.linalg.norm(
        pc.realizingPoint(loop, 0 if candidates[0][0] == "E" else 1, candidates[0][1], point)
        - pc.realizingPoint(loop, 0 if candidates[1][0] == "E" else 1, candidates[1][1], point))
    check("benign switch: realizing points coincide", float(separation), 0.0, 1e-12)

    cross = ref.place(ref.cross_shape(), 0.40, 0.40, 1.2)
    onRidge = np.array([0.127603, 0.505912])
    _, crossLengths, crossTangents, crossNormals = pc.edgeFrame(cross)
    ridgePair = sorted(
        [("E", j, abs(crossNormals[j] @ (cross[j] - onRidge))) for j in range(len(cross))
         if 0 <= crossTangents[j] @ (onRidge - cross[j]) <= crossLengths[j]],
        key = lambda item: item[2])[:2]
    ridgeSeparation = np.linalg.norm(
        pc.realizingPoint(cross, 0, ridgePair[0][1], onRidge)
        - pc.realizingPoint(cross, 0, ridgePair[1][1], onRidge))
    check("medial axis: realizing points 2*rIn apart", float(ridgeSeparation), 0.32, 2e-4)


def checkOverlapArea():
    """9. Overlap area by Green's theorem against a grid count."""
    print("\n9. overlap area via Green's theorem vs grid count")
    cases = [("square/square", ref.rect(0, 0, 1, 1), ref.rect(0.4, 0.4, 1.4, 1.4)),
             ("crossed bars", ref.rect(0, 0.4, 1, 0.6), ref.rect(0.4, 0, 0.6, 1.0))]
    for name, first, second in cases:
        green = pc.overlapArea(first, second)
        low = np.minimum(first.min(axis = 0), second.min(axis = 0))
        high = np.maximum(first.max(axis = 0), second.max(axis = 0))
        size = 500
        grid = np.stack(np.meshgrid(np.linspace(low[0], high[0], size),
                                    np.linspace(low[1], high[1], size)), -1).reshape(-1, 2)
        counted = int((pc.insideParity(grid, first) & pc.insideParity(grid, second)).sum())
        estimate = counted / size ** 2 * (high[0] - low[0]) * (high[1] - low[1])
        checkTrue(f"area  {name}", abs(green - estimate) < 3e-3,
                  f"green={green:.6f} grid={estimate:.6f}")


def checkCornerScaling():
    """10. Corner scaling exponents: 3 face-on-face, 4 sharp corner."""
    print("\n10. corner scaling exponents 3 / 4")
    wall = ref.rect(-8, -8, 8, 0.0)
    depths = np.array([1e-3, 3e-3, 1e-2, 3e-2])
    face = [pc.contactEnergy(ref.rect(-0.5, -d, 0.5, 1.0), wall) for d in depths]
    sharp = [pc.contactEnergy(
        ref.make_ccw(np.array([[0, -d], [3, -d + 3], [-3, -d + 3]], float)), wall) for d in depths]
    check("face-on-face exponent", float(np.polyfit(np.log(depths), np.log(face), 1)[0]), 3.0, 0.02)
    check("sharp-corner exponent", float(np.polyfit(np.log(depths), np.log(sharp), 1)[0]), 4.0, 0.10)


def checkValidityMonitor():
    """11. dMax/rIn flags interpenetration, PERSISTENTLY rather than transiently."""
    print("\n11. validity monitor: dMax/rIn flags interpenetration PERSISTENTLY")
    halfWidth = 0.16
    crossed = (ref.rect(-1, -halfWidth, 1, halfWidth), ref.rect(-halfWidth, -1, halfWidth, 1))
    shallow = (ref.rect(-0.5, -0.1 * halfWidth, 0.5, 1.0), ref.rect(-1, -2 * halfWidth, 1, 0.0))
    check("crossed limbs dMax/rIn", pc.maximumDepth(*crossed, samples = 800) / halfWidth, 1.0, 2e-2)
    check("shallow contact dMax/rIn", pc.maximumDepth(*shallow) / halfWidth, 0.1, 5e-3)


def main():
    print("polygon contact -- port of notes/polygonContact (ground truth: polyContactReference.py)")
    checkBuriedArclength()
    checkFaceVertexContrast()
    checkClosedFormAgainstQuadrature()
    checkGradientAgainstFiniteDifference()
    checkGradientMatchesReference()
    checkVertexNearestGradient()
    checkMedialAxisBreaksTheGradient()
    checkConservation()
    checkMembership()
    checkMarchAgainstBisection()
    checkMarchFindsEverySwitch()
    checkRealizingPointCriterion()
    checkOverlapArea()
    checkCornerScaling()
    checkValidityMonitor()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for name in FAILURES:
            print("   -", name)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
