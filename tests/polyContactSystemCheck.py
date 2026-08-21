"""Contract for ``polyContactSystem.py`` -- the many-body layer around the depth-contact law.

``tests/polyContactCheck.py`` covers the pair law against its reference. NONE of it can catch the
failures that live above the pair, and three of those were hit while building this layer on
2026-08-08. Every one produced a result that looked like success:

  1. AN UNCONSTRAINED RELAXER COLLAPSES THE BODIES. The law is purely repulsive, so with every vertex
     free its global minimum is every body shrunk to a POINT. 16 hexagons at phi 0.9755 reached E = 0
     in ONE L-BFGS iteration by shrinking to 68% of their area, reporting dMax/rIn = 0.000 and a
     perfect force balance throughout.
  2. A BODY LARGER THAN HALF THE BOX breaks the single minimum-image shift, which then flips
     discontinuously at the half-box. Four hexagons in a box of 1.4688 broke the assembled gradient's
     finite difference at 50% while net force sat at 3e-18 and reported nothing.
  3. A PERFECT SQUARE LATTICE IS SYMMETRIC, so every body's rigid gradient vanishes by symmetry
     (1.16e-17) and any relaxation test built on it is vacuous -- it "passes" having done nothing.

Checks 4, 5 and 6 exist for exactly those. Check 4 carries a NEGATIVE CONTROL: it asserts that the
unconstrained relaxer really does collapse, so that if someone later "fixes" it by adding a shape term
the test says so rather than passing silently.

Run: python tests/polyContactSystemCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polyContact as pc
import polyContactSystem as sysm
import polyContactReference as ref


FAILURES = []


def checkTrue(name, condition, detail = ""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name:56s} {detail}")
    if not condition:
        FAILURES.append(name)


def jitteredLattice(count = 9, radius = 0.5, squeeze = 0.16, seed = 7):
    """A lattice with per-body rotation and offset, then compressed into contact.

    Jittered DELIBERATELY: on a perfect lattice every body sits in a symmetric environment, its rigid
    gradient vanishes identically, and any test of the relaxer passes without doing anything."""
    bodies = sysm.certifiedLattice(ref.regular(6, radius), count)
    generator = np.random.default_rng(seed)
    centroids = sysm.bodyCentroids(bodies)
    for body in range(bodies.count):
        block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
        angle = generator.uniform(0.0, np.pi / 3.0)
        cosine, sine = np.cos(angle), np.sin(angle)
        turned = (bodies.positions[block] - centroids[body]) @ np.array([[cosine, sine],
                                                                         [-sine, cosine]])
        bodies.positions[block] = centroids[body] + turned + generator.normal(0.0, 0.03, 2)
    centroids = sysm.bodyCentroids(bodies)
    for body in range(bodies.count):
        block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
        bodies.positions[block] -= squeeze * centroids[body]
    bodies.boxSize *= (1.0 - squeeze)
    return bodies


def bodyAreas(bodies):
    return np.array([abs(pc.signedArea(bodies.loop(b))) for b in range(bodies.count)])


def checkCertificate():
    """1. A lattice at spacing > 2 x circumradius has EXACTLY zero energy and gradient."""
    print("\n1. certified-disjoint lattice (protocol step 1-2)")
    bodies = sysm.certifiedLattice(ref.regular(6, 0.5), 16)
    energy, gradient = sysm.systemEnergyGradient(bodies, useCuda = False)
    checkTrue("energy is exactly zero", energy == 0.0, f"E={energy!r}")
    checkTrue("gradient is exactly zero", float(np.abs(gradient).max()) == 0.0,
              f"max|g|={float(np.abs(gradient).max())!r}")


def checkBroadPhaseIsExact():
    """2. The broad phase drops nothing -- against an all-pairs sum with no culling."""
    print("\n2. broad phase vs all pairs, no culling")
    bodies = jitteredLattice()
    culled, _ = sysm.systemEnergyGradient(bodies, useCuda = False)
    # All pairs, culling NOTHING, but still applying the same minimum-image shift -- the cull and the
    # wrap are separate things and only the cull is under test. Comparing against an UNWRAPPED sum made
    # the culled energy look larger than "brute force", because it was the unwrapped sum that was
    # missing the pairs interacting across the periodic seam.
    brute = 0.0
    centroids = sysm.bodyCentroids(bodies)
    for i in range(bodies.count):
        for j in range(i + 1, bodies.count):
            shift = np.zeros(2)
            if bodies.boxSize is not None:
                offset = centroids[j] - centroids[i]
                shift = -bodies.boxSize * np.round(offset / bodies.boxSize)
            brute += pc.contactEnergy(bodies.loop(i), bodies.loop(j) + shift)
    checkTrue("culled energy == all-pairs energy", abs(culled - brute) < 1e-14 * max(brute, 1.0),
              f"culled={culled:.9e} brute={brute:.9e}")


def checkAssembledGradient():
    """3. The assembled gradient against central finite differences, free and periodic."""
    print("\n3. assembled gradient vs finite differences")
    for label, periodic in (("free    ", False), ("periodic", True)):
        bodies = jitteredLattice()
        if not periodic:
            bodies.boxSize = None
        energy, gradient = sysm.systemEnergyGradient(bodies, useCuda = False)
        generator = np.random.default_rng(0)
        step, worst = 1e-6, 0.0
        for index in generator.choice(len(bodies.positions), 12, replace = False):
            for component in range(2):
                original = bodies.positions[index, component]
                bodies.positions[index, component] = original + step
                plus = sysm.systemEnergyGradient(bodies, useCuda = False)[0]
                bodies.positions[index, component] = original - step
                minus = sysm.systemEnergyGradient(bodies, useCuda = False)[0]
                bodies.positions[index, component] = original
                worst = max(worst, abs(gradient[index, component] - (plus - minus) / (2.0 * step)))
        scale = float(np.abs(gradient).max())
        checkTrue(f"grad==FD  {label}", worst < 1e-8 * max(scale, 1.0),
                  f"max|dg|={worst:.2e}  (|g| ~ {scale:.2e})")


def checkRigidGradientAndCollapse():
    """4. Rigid-body gradient, area preservation, and the COLLAPSE negative control.

    The negative control is the point: it asserts that the vertex-level relaxer really does shrink the
    bodies. If someone later gives it a shape term, this fails and says so, rather than the suite
    quietly losing its only record of why ``relaxRigid`` exists."""
    print("\n4. rigid gradient, area preservation, and the collapse trap   [TRAP]")
    bodies = jitteredLattice()
    energy, perBody = sysm.rigidGradient(bodies)
    checkTrue("the fixture is NOT symmetric (else this is vacuous)",
              float(np.abs(perBody).max()) > 1e-6, f"|g|={float(np.abs(perBody).max()):.2e}")

    reference = bodies.positions.copy()
    centroids = sysm.bodyCentroids(bodies)
    step, worst = 1e-7, 0.0
    for body in range(bodies.count):
        for degree in range(3):
            values = []
            for sign in (+1, -1):
                bodies.positions[:] = reference
                block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
                if degree < 2:
                    bodies.positions[block, degree] += sign * step
                else:
                    angle = sign * step
                    cosine, sine = np.cos(angle), np.sin(angle)
                    offset = bodies.positions[block] - centroids[body]
                    bodies.positions[block] = centroids[body] + offset @ np.array(
                        [[cosine, sine], [-sine, cosine]])
                values.append(sysm.systemEnergyGradient(bodies, useCuda = False)[0])
            bodies.positions[:] = reference
            worst = max(worst, abs(perBody[body, degree] - (values[0] - values[1]) / (2.0 * step)))
    checkTrue("rigid gradient == FD", worst < 1e-8 * float(np.abs(perBody).max()),
              f"max|dg|={worst:.2e}  (|g| ~ {float(np.abs(perBody).max()):.2e})")

    before = bodyAreas(bodies)
    startEnergy = sysm.systemEnergyGradient(bodies, useCuda = False)[0]
    finalEnergy, worstGradient, iterations = sysm.relaxRigid(bodies, 3000)
    after = bodyAreas(bodies)
    drift = float(np.abs(after / before - 1.0).max())
    print(f"       relaxRigid: E {startEnergy:.4e} -> {finalEnergy:.4e}  "
          f"max|g| {worstGradient:.2e}  {iterations} iterations")
    checkTrue("relaxRigid PRESERVES body area", drift < 1e-12, f"max area drift={drift:.2e}")
    checkTrue("relaxRigid lowers the energy", finalEnergy < startEnergy,
              f"{startEnergy:.4e} -> {finalEnergy:.4e}")

    # NEGATIVE CONTROL: the vertex-level relaxer must be seen to collapse.
    collapsing = jitteredLattice()
    areaBefore = bodyAreas(collapsing)
    collapsedEnergy, _, _ = sysm.relax(collapsing, 500)
    areaAfter = bodyAreas(collapsing)
    shrink = float((areaAfter / areaBefore).min())
    checkTrue("[negative control] unconstrained relax COLLAPSES bodies", shrink < 0.95,
              f"smallest area ratio={shrink:.4f}, E -> {collapsedEnergy:.2e}")


def checkMinimumImageGuard():
    """5. A body larger than half the box must WARN, not silently break the gradient."""
    print("\n5. minimum-image guard: a body larger than half the box   [TRAP]")
    bodies = sysm.certifiedLattice(ref.regular(6, 0.5), 4)
    bodies.boxSize *= 0.72
    centroids = sysm.bodyCentroids(bodies)
    for body in range(bodies.count):
        block = slice(bodies.startIndices[body], bodies.startIndices[body + 1])
        bodies.positions[block] -= 0.28 * centroids[body]
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sysm.systemEnergyGradient(bodies, useCuda = False)
    fired = any("half the box" in str(w.message) or "spans" in str(w.message) for w in caught)
    checkTrue("a body spanning > 50% of the box warns", fired, f"{len(caught)} warning(s)")


def checkValidityMonitor():
    """6. dMax/rIn flags interpenetration, and is small on a valid relaxed state."""
    print("\n6. validity monitor")
    crossed = sysm.BodySet([ref.rect(-1, -0.16, 1, 0.16), ref.rect(-0.16, -1, 0.16, 1)])
    ratio, _ = sysm.systemValidity(crossed, samples = 400)
    checkTrue("crossed limbs flagged", ratio > 0.9, f"dMax/rIn={ratio:.3f}")

    bodies = jitteredLattice()
    sysm.relaxRigid(bodies, 3000)
    relaxedRatio, where = sysm.systemValidity(bodies)
    checkTrue("relaxed state is well inside validity", relaxedRatio < 0.35,
              f"dMax/rIn={relaxedRatio:.4f} at {where}")


def checkContainerConfines():
    """7. A container CONFINES: zero inside, positive outside, force pointing back in.

    This replaces the refusal that stood here until 2026-08-09. The three properties below are what
    "confines" means, and the sign of the second is the whole risk: a counter-clockwise wall inverts the
    membership test into an attractive well that pulls bodies INTO the boundary, and the energy alone
    cannot tell the two apart -- both are positive and both grow. Only the direction of the force
    distinguishes them, so it is asserted directly."""
    print("\n7. container confines (zero inside, repulsive outside)")

    class Fake:
        pass

    def boxed(offset):
        square = np.array([[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]]) + offset
        wall = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        packing = Fake()
        packing.positions = np.concatenate([square, wall]).ravel()
        packing.startIndices = np.array([0, 4, 8])
        packing.numPolygons = 2
        packing.box = None
        packing.containerIndex = 1
        return packing

    inside = boxed(np.array([0.0, 0.0]))
    energy, _ = sysm.packingEnergyForce(inside)
    checkTrue("strictly inside costs nothing", energy == 0.0, f"E = {energy:.3e}")

    poking = boxed(np.array([-0.35, 0.0]))
    energy, force = sysm.packingEnergyForce(poking)
    checkTrue("poking out costs energy", energy > 0.0, f"E = {energy:.3e}")
    push = force.reshape(-1, 2)[:4].sum(axis = 0)
    checkTrue("force points back INTO the box", push[0] > 0.0,
              f"net force on the body = [{push[0]:+.3e}, {push[1]:+.3e}]")

    step, worst = 1e-7, 0.0
    vertices = poking.positions.reshape(-1, 2)
    for vertex in range(len(vertices)):
        for component in range(2):
            original = vertices[vertex, component]
            vertices[vertex, component] = original + step
            plus, _ = sysm.packingEnergyForce(poking)
            vertices[vertex, component] = original - step
            minus, _ = sysm.packingEnergyForce(poking)
            vertices[vertex, component] = original
            worst = max(worst, abs(-force[vertex, component] - (plus - minus) / (2.0 * step)))
    checkTrue("gradient == FD, wall vertices included", worst < 1e-7, f"max|dg| = {worst:.2e}")

    # The batched path (wall inside the BodySet) against the explicit per-body reference.
    reference, _ = sysm.confinementEnergyGradient(poking)
    checkTrue("batched path == explicit reference",
              abs(energy - reference) < 1e-12 * max(reference, 1e-30),
              f"relative difference {abs(energy - reference) / max(reference, 1e-300):.2e}")


def checkCuda():
    """8. The CUDA kernel against the numpy assembly.

    THE GRADIENT IS THE THING TO CHECK, and vertex counts must be pushed past every stride in the
    build. That is not general caution, it is this codebase's own history: a hard-coded cap once
    dropped the gradient on every vertex past the 24th of a pair while the ENERGY stayed correct, so
    the failure survived the entire test suite.

    MIXED vertex counts matter most. A uniform-n packing cannot detect a kernel that assumes uniform n,
    because reading the stride from the wrong place still gives the right answer."""
    print("\n8. CUDA kernel vs the numpy assembly")
    try:
        import cudaOverlap
    except ImportError:
        print("       cudaOverlap unavailable -- skipped")
        return
    if not cudaOverlap.isAvailable():
        print("       no usable GPU -- skipped")
        return

    for vertexCount in (4, 6, 12, 13, 32, 33, 64):
        bodies = jitteredLattice(count = 4, radius = 0.5)
        bodies = sysm.BodySet([ref.regular(vertexCount, 0.5) + centre
                               for centre in sysm.bodyCentroids(bodies)],
                              boxSize = bodies.boxSize)
        hostEnergy, hostGradient = sysm.systemEnergyGradient(bodies, useCuda = False)
        deviceEnergy, deviceGradient = cudaOverlap.polyContactCuda(bodies)
        scale = max(float(np.abs(hostGradient).max()), 1e-300)
        energyError = abs(deviceEnergy / hostEnergy - 1.0) if hostEnergy else abs(deviceEnergy)
        gradientError = float(np.abs(hostGradient - deviceGradient).max())
        checkTrue(f"n={vertexCount:3d}", hostEnergy > 1e-12 and energyError < 1e-12
                  and gradientError < 1e-12 * scale,
                  f"E {hostEnergy:.3e}  relE {energyError:.1e}  max|dg| {gradientError:.1e}")

    # RAGGED: three different vertex counts in one system.
    ragged = sysm.BodySet([ref.rect(0.10, 0.10, 0.95, 0.95),
                           ref.regular(7, 0.45) + np.array([1.35, 0.50]),
                           ref.regular(13, 0.50) + np.array([0.80, 1.30])])
    hostEnergy, hostGradient = sysm.systemEnergyGradient(ragged, useCuda = False)
    deviceEnergy, deviceGradient = cudaOverlap.polyContactCuda(ragged)
    scale = max(float(np.abs(hostGradient).max()), 1e-300)
    checkTrue("MIXED n = 4, 7, 13 in one system",
              hostEnergy > 1e-12
              and abs(deviceEnergy / hostEnergy - 1.0) < 1e-12
              and float(np.abs(hostGradient - deviceGradient).max()) < 1e-12 * scale,
              f"E {hostEnergy:.3e}  relE {abs(deviceEnergy / hostEnergy - 1.0):.1e}  "
              f"max|dg| {float(np.abs(hostGradient - deviceGradient).max()):.1e}")

    # Past the cap: REPORTED, never truncated.
    huge = 200
    angles = np.linspace(0.0, 2.0 * np.pi, huge, endpoint = False)
    oversized = sysm.BodySet([np.stack([0.35 + 0.2 * np.cos(angles), 0.5 + 0.2 * np.sin(angles)], -1),
                              np.stack([0.60 + 0.2 * np.cos(angles), 0.5 + 0.2 * np.sin(angles)], -1)])
    try:
        cudaOverlap.polyContactCuda(oversized)
        checkTrue("n past POLYCONTACT_MAXN raises", False, "it was accepted silently")
    except ValueError as error:
        checkTrue("n past POLYCONTACT_MAXN raises", str(huge) in str(error), str(error)[:56] + "...")


def main():
    print("polygon contact -- many-body layer (polyContactSystem.py)")
    checkCertificate()
    checkBroadPhaseIsExact()
    checkAssembledGradient()
    checkRigidGradientAndCollapse()
    checkMinimumImageGuard()
    checkValidityMonitor()
    checkContainerConfines()
    checkCuda()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for name in FAILURES:
            print("   -", name)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
