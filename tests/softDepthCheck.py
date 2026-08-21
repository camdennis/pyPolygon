"""Verification for the smooth penetration depth of ``notes/softDepth-1.pdf``.

Follows the note's own validation list (sec 17), in its order of increasing cost. Equation numbers
refer to that document.

  1. IDENTITY (12): ``|grad h|^2 - eps*lap(h) - 1 == 0`` at machine precision. The strongest and
     cheapest check -- it tests the gradient and the Hessian trace together against an exact identity,
     with no reference solution and no finite differencing.
  2. INVARIANCE (sec 14): forces sum to zero and torques sum to zero, in floating point rather than to
     within truncation, because ``(1-s)n + s n - n = 0`` holds identically per edge.
  3. BOUNDS (Prop 1): ``0 <= min_i ell_i - h <= eps log N`` everywhere, and the deficit decays as
     ``eps (N-1) exp(-Delta/eps)`` away from the medial axis.
  4. EQUILIBRIUM CUBIC (24)/(25): the Cardano root satisfies ``eta^3 + eta = mu`` and ``phi'(h*) = 0``.
  5. WORK OF ADHESION (21)/(26): ``int W chi' dh = W`` exactly, and ``min_h phi''`` occurs at
     ``h = -lam/2`` with the value (26).
  6. CORNER GEOMETRY (14)/(15): the zero set crosses the bisector at ``eps log 2`` independent of
     angle, and the level-set radius matches ``R = eps sin(t/2)/cos^2(t/2)``.
  7. VERTEX DERIVATIVES (38)/(40): finite differences of ``dE/dx`` and ``dE/dv_j``. The note flags this
     as the only check exercising the vertex second derivatives and the only one needing care with the
     ``h_+^(1/2)`` factor near contact onset -- so it is run away from ``h = 0``.

Checks 8-10 cover the BOUNDARY INTEGRAL that turns the point law of 1-7 into the actual energy,
``E = int_{dA} phi(h_eps^B) dl``. They are not in the note's list because the note assumes the integral
rather than a discretization of it.

  8. BOUNDARY INTEGRAL: against an independent uniform walk along the boundary, plus the contact
     interval against a dense sign scan of ``h``. This is the check that distinguishes the integral
     from a vertex rule, which reports exactly zero on face-to-face contact.
  9. QUADRATURE FORCES: conservation and finite differences for the two terms check 7 cannot see -- the
     node's barycentric split onto its edge, and the tangential force from the moving measure.
 10. CONVERGENCE: relative error against order, per contact type, to CHOOSE an order for a given
     epsilon rather than to pass a threshold.
 11. PERIODIC ASSEMBLY: the pair loop, the minimum-image shift and the force scatter, which nothing
     above this line exercises.
 12. ORIENTATION AND CONFINEMENT: winding must not change the answer, and a container must push INWARD.

Run: python tests/softDepthCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import softDepth as sd


def regularLoop(n, radius = 1.0, center = (0.0, 0.0), phase = 0.0):
    """A CCW regular n-gon -- convex, so Lemma 1 applies."""
    angles = phase + 2.0 * np.pi * np.arange(n) / n
    return np.stack([center[0] + radius * np.cos(angles),
                     center[1] + radius * np.sin(angles)], axis = -1)


def square(cx, cy, side = 1.0, angle = 0.0):
    """A CCW square, rotated about its own center."""
    half = side / 2.0
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    cosine, sine = np.cos(angle), np.sin(angle)
    return corners @ np.array([[cosine, sine], [-sine, cosine]]) + np.array([cx, cy])


# The four contacts the boundary integral has to get right. "face-to-face" is the case a vertex rule
# misses entirely, and the sharp variant is the one that forced the envelope split.
CONTACTS = [("face-to-face", square(0.0, 0.0), square(0.95, 0.0), 1e-2),
            ("face-to-face sharp", square(0.0, 0.0), square(0.95, 0.0), 1e-3),
            ("corner-into-face", square(0.0, 0.0), square(0.62, 0.62, angle = 0.6), 1e-2),
            ("deep overlap", square(0.0, 0.0), square(0.4, 0.15, angle = 0.3), 1e-2)]


def checkIdentity():
    """1. |grad h|^2 - eps lap(h) = 1 exactly, eq (12)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (3, 4, 6, 12):
        loop = regularLoop(n, phase = 0.3)
        for epsilon in (1e-1, 1e-2, 1e-3):
            points = rng.uniform(-2.0, 2.0, size = (400, 2))
            _, weights, normals, _ = sd.softDepth(points, loop, epsilon)
            defect = sd.eikonalDefect(weights, normals, epsilon)
            worst = max(worst, float(np.abs(defect).max()))
    print(f"  1. eikonal identity (12)        max |defect| = {worst:.3e}")
    assert worst < 1e-10, f"identity (12) violated by {worst:.3e}"


def checkInvariance():
    """2. Forces and torques sum to zero, sec 14."""
    rng = np.random.default_rng(1)
    loop = regularLoop(5, phase = 0.2)
    points = rng.uniform(-0.6, 0.6, size = (7, 2))
    energy, pointForces, loopForces = sd.pointLoopEnergyForce(
        points, loop, epsilon = 0.05, stiffness = 3.0)
    net = pointForces.sum(axis = 0) + loopForces.sum(axis = 0)
    torque = (np.cross(points, pointForces).sum() + np.cross(loop, loopForces).sum())
    scale = max(float(np.abs(pointForces).max()), 1e-300)
    print(f"  2. sum of forces  {np.abs(net).max():.3e}   sum of torques  {abs(torque):.3e}   "
          f"(forces ~ {scale:.3e})")
    assert np.abs(net).max() < 1e-12 * max(scale, 1.0), "translational invariance broken"
    assert abs(torque) < 1e-11 * max(scale, 1.0), "rotational invariance broken"


def checkBounds():
    """3. Proposition 1: 0 <= min ell - h <= eps log N, and the deficit decays exponentially."""
    rng = np.random.default_rng(2)
    for n in (4, 8):
        loop = regularLoop(n)
        epsilon = 0.02
        points = rng.uniform(-1.5, 1.5, size = (2000, 2))
        h, _, normals, _ = sd.softDepth(points, loop, epsilon)
        _, _, normals2, lengths, offsets = sd.loopFrame(loop)
        ell = offsets[None, :] - points @ normals2.T
        deficit = ell.min(axis = 1) - h
        upper = epsilon * np.log(n)
        print(f"  3. n={n:2d}  deficit in [{deficit.min():.3e}, {deficit.max():.3e}]   "
              f"bound eps log N = {upper:.3e}")
        assert deficit.min() >= -1e-14, "h exceeded the exact depth"
        assert deficit.max() <= upper + 1e-12, "global deficit exceeded eps log N"

    # Local bound: away from the medial axis a single face dominates and the deficit is exponential.
    loop = regularLoop(4)
    epsilon = 0.02
    along = np.stack([np.zeros(6), np.linspace(-0.6, -0.05, 6)], axis = -1)
    h, _, _, _ = sd.softDepth(along, loop, epsilon)
    _, _, normals, _, offsets = sd.loopFrame(loop)
    ell = offsets[None, :] - along @ normals.T
    sortedEll = np.sort(ell, axis = 1)
    gap = sortedEll[:, 1] - sortedEll[:, 0]
    deficit = sortedEll[:, 0] - h
    predicted = epsilon * (len(loop) - 1) * np.exp(-gap / epsilon)
    print(f"  3. local bound: deficit <= eps(N-1)exp(-Delta/eps) satisfied at "
          f"{int(np.sum(deficit <= predicted + 1e-15))}/{len(gap)} sample points")
    assert np.all(deficit <= predicted + 1e-15), "local deficit bound violated"


def checkEquilibriumCubic():
    """4. The Cardano root solves eta^3 + eta = mu and phi'(h*) = 0, eqs (24)/(25)."""
    worstCubic, worstForce = 0.0, 0.0
    for stiffness in (1.0, 10.0):
        for adhesionWork in (1e-4, 1e-2, 1.0):
            for adhesionRange in (1e-3, 1e-2):
                star = sd.equilibriumIndentation(stiffness, adhesionWork, adhesionRange)
                eta = star / adhesionRange
                mu = (adhesionWork / (2.0 * stiffness * adhesionRange ** 2.5)) ** (2.0 / 3.0)
                worstCubic = max(worstCubic, abs(eta ** 3 + eta - mu) / max(mu, 1.0))
                _, first, _ = sd.contactLaw(np.array([star]), stiffness,
                                            adhesionWork, adhesionRange)
                scale = stiffness * max(star, 1e-300) ** 1.5
                worstForce = max(worstForce, abs(float(first[0])) / max(scale, 1e-300))
    print(f"  4. cubic residual {worstCubic:.3e}   |phi'(h*)| / scale {worstForce:.3e}")
    assert worstCubic < 1e-12, "the Cardano root does not solve the cubic"
    assert worstForce < 1e-9, "phi' does not vanish at the equilibrium indentation"


def checkAdhesionWork():
    """5. Work of adhesion (21) and the stiffness minimum (26)."""
    adhesionWork, adhesionRange = 0.7, 0.02
    grid = np.linspace(-400.0 * adhesionRange, 400.0 * adhesionRange, 2000001)
    _, first, _ = sd.plummerStep(grid, adhesionRange)
    integral = adhesionWork * np.trapezoid(first, grid) if hasattr(np, "trapezoid") \
        else adhesionWork * np.trapz(first, grid)
    print(f"  5. work of adhesion  {integral:.9f}   expected {adhesionWork:.9f}   "
          f"error {abs(integral - adhesionWork):.2e}")
    assert abs(integral - adhesionWork) < 1e-4, "the adhesive work is not W"

    # min_h phi'' at h = -lam/2 with value -(3/4)(4/5)^(5/2) W/lam^2, eq (26).
    scan = np.linspace(-6.0 * adhesionRange, 2.0 * adhesionRange, 400001)
    _, _, second = sd.contactLaw(scan, stiffness = 1.0,
                                 adhesionWork = adhesionWork, adhesionRange = adhesionRange)
    where = scan[int(np.argmin(second))]
    predicted = -0.75 * (0.8 ** 2.5) * adhesionWork / adhesionRange ** 2
    print(f"  5. argmin phi'' = {where:.6f}  (expected {-0.5 * adhesionRange:.6f});  "
          f"min phi'' = {second.min():.6f}  (expected {predicted:.6f})")
    assert abs(where + 0.5 * adhesionRange) < 2e-5, "phi'' minimum is not at h = -lam/2"
    assert abs(second.min() - predicted) / abs(predicted) < 1e-6, "min phi'' disagrees with (26)"


def checkCornerGeometry():
    """6. Apex offset eps log 2 (14) and corner radius (15)."""
    epsilon = 0.01
    for n in (4, 6, 8):
        loop = regularLoop(n, radius = 1.0)
        interior = np.pi * (n - 2) / n
        # On the bisector through vertex 0, the zero set sits at ell = eps log 2 from the two faces.
        vertex = loop[0]
        inward = -vertex / np.linalg.norm(vertex)
        offsets = np.linspace(0.0, 0.2, 200001)
        probes = vertex[None, :] + offsets[:, None] * inward[None, :]
        h, _, _, _ = sd.softDepth(probes, loop, epsilon)
        crossing = offsets[int(np.argmin(np.abs(h)))]
        # ell along the bisector is the inward offset times cos(alpha), where alpha = (pi - theta)/2 is
        # the half-angle between the two OUTWARD NORMALS (the note's alpha), not half the interior
        # angle. cos((pi - theta)/2) = sin(theta/2). The two coincide at theta = pi/2, so a square
        # cannot distinguish them -- which is exactly how the wrong factor survived until the hexagon.
        apex = crossing * np.sin(0.5 * interior)
        predicted = epsilon * np.log(2.0)
        radius = sd.cornerRadius(epsilon, interior)
        print(f"  6. n={n}  apex ell = {apex:.6e}  (eps log 2 = {predicted:.6e})   "
              f"R_corner = {radius:.4e}")
        assert abs(apex - predicted) / predicted < 5e-3, "apex offset disagrees with (14)"
    assert abs(sd.cornerRadius(1.0, np.pi / 2) - np.sqrt(2.0)) < 1e-12, \
        "a right angle should give R = sqrt(2) eps"


def checkVertexDerivatives():
    """7. Finite differences of dE/dx and dE/dv_j against (38)/(40).

    Run at a depth well away from contact onset: the note warns this check is the one that needs care
    with the ``h_+^(1/2)`` factor near ``h = 0``, where phi'' is continuous but its own derivative is
    not."""
    rng = np.random.default_rng(3)
    loop = regularLoop(5, phase = 0.11)
    points = rng.uniform(-0.35, 0.35, size = (4, 2))
    epsilon, stiffness = 0.04, 2.0
    adhesionWork, adhesionRange = 0.05, 0.01

    def energyOf(pts, lp):
        e, _, _ = sd.pointLoopEnergyForce(pts, lp, epsilon, stiffness, adhesionWork, adhesionRange)
        return e

    _, pointForces, loopForces = sd.pointLoopEnergyForce(
        points, loop, epsilon, stiffness, adhesionWork, adhesionRange)

    step = 1e-6
    worstPoint = 0.0
    for p in range(len(points)):
        for d in range(2):
            up, down = points.copy(), points.copy()
            up[p, d] += step; down[p, d] -= step
            numerical = -(energyOf(up, loop) - energyOf(down, loop)) / (2.0 * step)
            worstPoint = max(worstPoint, abs(numerical - pointForces[p, d]))
    worstLoop = 0.0
    for j in range(len(loop)):
        for d in range(2):
            up, down = loop.copy(), loop.copy()
            up[j, d] += step; down[j, d] -= step
            numerical = -(energyOf(points, up) - energyOf(points, down)) / (2.0 * step)
            worstLoop = max(worstLoop, abs(numerical - loopForces[j, d]))
    scale = max(float(np.abs(pointForces).max()), float(np.abs(loopForces).max()))
    print(f"  7. dE/dx   max |analytic - FD| = {worstPoint:.3e}")
    print(f"  7. dE/dv_j max |analytic - FD| = {worstLoop:.3e}   (forces ~ {scale:.3e})")
    assert worstPoint < 1e-6 * max(scale, 1.0), "point force disagrees with finite differences"
    assert worstLoop < 1e-6 * max(scale, 1.0), "vertex force disagrees with finite differences"


def midpointReference(loopA, loopB, epsilon, stiffness, samples = 100000):
    """``int_{dA} phi(h^B) dl`` by composite midpoint over the WHOLE boundary.

    Deliberately independent of everything it is used to check: no ``contactIntervals``, no
    ``envelopeCuts``, no Gauss-Legendre. It just walks the boundary uniformly. Slow, but converged --
    the value is stable to 1e-16 from 1e5 samples to 8e5, and the Gauss rule reproduces it digit for
    digit at order 128 -- so a disagreement is the fast rule's, not the reference's."""
    start = np.asarray(loopA, dtype = float)
    end = np.roll(start, -1, axis = 0)
    lengths = np.hypot(*(end - start).T)
    fraction = (np.arange(samples) + 0.5) / samples
    total = 0.0
    for edge in range(len(start)):
        points = start[edge] + fraction[:, None] * (end[edge] - start[edge])
        depth, _, _, _ = sd.softDepth(points, loopB, epsilon)
        density, _, _ = sd.contactLaw(depth, stiffness)
        total += float(density.sum()) * lengths[edge] / samples
    return total


def checkBoundaryIntegral():
    """8. The boundary integral against an INDEPENDENT reference, and the contact interval it rests on.

    This is the check that pins the model down. A vertex rule is not a low-order version of this
    integral, it is a different and wrong law: two squares meeting face to face have no vertex of
    either inside the other, so it reports exactly zero against real overlap. The reference here is a
    plain uniform walk along the boundary, sharing no machinery with the rule being tested."""
    stiffness = 1.0
    for label, loopA, loopB, epsilon in CONTACTS:
        reference = midpointReference(loopA, loopB, epsilon, stiffness)
        integral, _, _ = sd.edgeLoopEnergyForce(loopA, loopB, epsilon, stiffness, order = 96)
        vertexRule, _, _ = sd.pointLoopEnergyForce(loopA, loopB, epsilon, stiffness)
        error = abs(integral / reference - 1.0)
        print(f"  8. {label:20s} E = {integral:.9e}   vs reference {error:.2e}   "
              f"vertex rule {vertexRule:.3e}")
        assert error < 1e-7, f"{label}: boundary integral disagrees with the independent reference"
    assert sd.pointLoopEnergyForce(CONTACTS[0][1], CONTACTS[0][2], CONTACTS[0][3], stiffness)[0] == 0.0, \
        "the face-to-face case is supposed to be the one a vertex rule reports as zero"

    # The interval itself, against a dense sign scan -- concavity says {h >= 0} is ONE interval per
    # edge, so a disagreement here means either the claim or the root finder is wrong.
    for label, loopA, loopB, epsilon in CONTACTS:
        start = np.asarray(loopA, dtype = float)
        end = np.roll(start, -1, axis = 0)
        lower, upper = sd.contactIntervals(loopA, loopB, epsilon)
        scan = (np.arange(20000) + 0.5) / 20000
        worst = 0.0
        for edge in range(len(start)):
            points = start[edge] + scan[:, None] * (end[edge] - start[edge])
            depth, _, _, _ = sd.softDepth(points, loopB, epsilon)
            positive = depth > 0.0
            if positive.any():
                found = scan[positive]
                runs = int(np.count_nonzero(np.diff(positive.astype(int)) == 1)) + int(positive[0])
                assert runs == 1, f"{label} edge {edge}: {runs} contact runs, concavity says 1"
                worst = max(worst, abs(found[0] - lower[edge]), abs(found[-1] - upper[edge]))
            else:
                worst = max(worst, upper[edge] - lower[edge])
        print(f"  8. {label:20s} contact interval vs dense sign scan: {worst:.2e}")
        assert worst < 1e-4, f"{label}: contact interval disagrees with the sign of h"


def checkQuadratureForces():
    """9. Conservation and finite differences for the FULL boundary-integral force.

    Two terms exist here that the point law of check 7 never sees: the quadrature node's force split
    barycentrically back onto its edge's endpoints, and the TANGENTIAL force from ``d|e|/dv``, since
    the measure ``dl = |e| dt`` moves with the geometry. Both are torque-free by construction, so the
    conservation half is exact rather than approximate.

    The finite difference also settles the Leibniz claim. The integration limits ``t0``, ``t1`` and the
    envelope cuts all move as the vertices move, and none of them is differentiated -- justified
    because ``phi(h) = 0`` at a crossing. If that were wrong the FD would disagree at O(1)."""
    stiffness, order, step = 1.0, 48, 1e-6
    for label, loopA, loopB, epsilon in CONTACTS:
        _, forcesA, forcesB = sd.edgeLoopEnergyForce(loopA, loopB, epsilon, stiffness, order = order)
        net = forcesA.sum(axis = 0) + forcesB.sum(axis = 0)
        torque = float(np.cross(loopA, forcesA).sum() + np.cross(loopB, forcesB).sum())
        scale = max(float(np.abs(forcesA).max()), float(np.abs(forcesB).max()))

        def energyOf(a, b):
            return sd.edgeLoopEnergyForce(a, b, epsilon, stiffness, order = order)[0]

        worst = 0.0
        for moving, (loop, force) in enumerate(((loopA, forcesA), (loopB, forcesB))):
            for vertex in range(len(loop)):
                for axis in range(2):
                    up = [np.array(loopA), np.array(loopB)]
                    down = [np.array(loopA), np.array(loopB)]
                    up[moving][vertex, axis] += step
                    down[moving][vertex, axis] -= step
                    numerical = -(energyOf(*up) - energyOf(*down)) / (2.0 * step)
                    worst = max(worst, abs(numerical - force[vertex, axis]))
        print(f"  9. {label:20s} |sum F| {np.hypot(*net):.2e}   sum tau {torque:+.2e}   "
              f"max |analytic - FD| {worst:.2e}   (|F| ~ {scale:.2e})")
        assert np.hypot(*net) < 1e-12 * max(scale, 1.0), f"{label}: forces do not sum to zero"
        assert abs(torque) < 1e-12 * max(scale, 1.0), f"{label}: torques do not sum to zero"
        assert worst < 1e-6 * max(scale, 1.0), f"{label}: force disagrees with finite differences"


def checkQuadratureConvergence():
    """10. How fast the rule converges, and how that degrades as epsilon sharpens.

    Run this to CHOOSE an order rather than to pass a threshold. The integrand varies on the scale of
    epsilon, so the order needed grows as the shape sharpens: the same face-to-face contact that order
    16 resolves to 4e-06 at eps/edge = 1e-2 is only good to 8e-05 at 1e-3.

    Without the envelope split the sharp column is not merely slower, it is non-monotone in the order
    -- measured 2.1e-04 at order 12 but 6.4e-04 at 16 -- which is worse than being wrong, because the
    error gives no signal that more nodes are needed."""
    stiffness = 1.0
    orders = (8, 12, 16, 24, 32, 48, 64)
    print("  10. relative error in E vs order " + "".join(f"{o:>10d}" for o in orders))
    for label, loopA, loopB, epsilon in CONTACTS:
        reference, _, _ = sd.edgeLoopEnergyForce(loopA, loopB, epsilon, stiffness, order = 128)
        errors = []
        for order in orders:
            value, _, _ = sd.edgeLoopEnergyForce(loopA, loopB, epsilon, stiffness, order = order)
            errors.append(abs(value / reference - 1.0))
        print(f"      {label:20s}" + "".join(f"{e:>10.1e}" for e in errors))
        assert errors[-1] < 1e-7, f"{label}: order 48 has not converged"
        assert errors[-1] < errors[0], f"{label}: raising the order did not help"


def checkPeriodicAssembly():
    """11. The ASSEMBLY -- that `packingEnergyForce` places the pieces checks 8-10 validate.

    Checks 8-10 all drive `edgeLoopEnergyForce` on two loops directly, so nothing above this line
    exercises the pair loop, the minimum-image shift, or the force scatter. That gap is exactly how a
    real bug survived: the shift was applied to BOTH the boundary and the loop, which is a rigid
    translation of the pair and cancels, so periodicity was silently off while the cull still selected
    pairs by their minimum image. A pair overlapping only across the wrap measured 0.0 against 8.2e-05.

    The test is a translation invariance: the same two squares at the same separation, once across the
    periodic seam and once in the middle of the box, must give the same energy and the same forces."""
    from model import Model
    model = Model(N = 2, n = 4, seed = 1)
    model.generateEquilateralPolygons(phi = 0.2, kappa = 4.0)
    model.setBoundaryConditions("periodic")
    packing = model.packing
    epsilon, stiffness, order = 0.02, 1.0, 32

    def place(firstX, secondX):
        vertices = packing.positions.reshape(-1, 2)
        vertices[0:4] = square(firstX, 0.5, 0.2)
        vertices[4:8] = square(secondX, 0.5, 0.2)
        energy, force = sd.packingEnergyForce(packing, epsilon, stiffness, 0.0, 1.0, 1.0, order, useCuda = False)
        return energy, force.reshape(-1, 2)

    acrossSeam, seamForce = place(0.05, 0.97)
    inTheMiddle, middleForce = place(0.40, 0.32)
    energyError = abs(acrossSeam / inTheMiddle - 1.0)
    forceError = float(np.abs(np.sort(seamForce, axis = 0) - np.sort(middleForce, axis = 0)).max())
    print(f"  11. across the seam E = {acrossSeam:.9e}   in the middle E = {inTheMiddle:.9e}")
    print(f"  11. translation invariance: energy {energyError:.2e}   force {forceError:.2e}   "
          f"sum F {float(np.abs(seamForce.sum(axis = 0)).max()):.2e}")
    assert acrossSeam > 0.0, "a pair overlapping across the periodic seam measured zero energy"
    assert energyError < 1e-12, "the wrapped and unwrapped energies differ"
    assert forceError < 1e-12, "the wrapped and unwrapped forces differ"
    assert float(np.abs(seamForce.sum(axis = 0)).max()) < 1e-12, "forces do not sum to zero"

    # and the assembled force is still the gradient of the assembled energy
    vertices = packing.positions.reshape(-1, 2)
    vertices[0:4] = square(0.05, 0.5, 0.2)
    vertices[4:8] = square(0.97, 0.5, 0.2)
    reference = packing.positions.copy()
    _, analytic = sd.packingEnergyForce(packing, epsilon, stiffness, 0.0, 1.0, 1.0, order, useCuda = False)
    step, worst = 1e-6, 0.0
    for index in range(len(reference)):
        packing.positions[:] = reference; packing.positions[index] += step
        up = sd.packingEnergyForce(packing, epsilon, stiffness, 0.0, 1.0, 1.0, order, useCuda = False)[0]
        packing.positions[:] = reference; packing.positions[index] -= step
        down = sd.packingEnergyForce(packing, epsilon, stiffness, 0.0, 1.0, 1.0, order, useCuda = False)[0]
        worst = max(worst, abs(-(up - down) / (2.0 * step) - analytic[index]))
    packing.positions[:] = reference
    scale = float(np.abs(analytic).max())
    print(f"  11. assembled dE/dr across the seam: max |analytic - FD| {worst:.2e}   "
          f"(|F| ~ {scale:.2e})")
    assert worst < 1e-6 * max(scale, 1.0), "the assembled force is not the assembled energy's gradient"


def checkOrientationAndConfinement():
    """12. Winding must not change the answer, and a container must push INWARD.

    ``n_i = J t_i`` is outward only for a CCW loop. Given a clockwise one every normal points inward,
    so ``ell_i`` and ``h`` change sign and the model inverts. That is not hypothetical: the wall in
    ``tests/squaresInASquareArea-Boundary.ipynb`` is written ``[[0,0],[0,1],[1,1],[1,0]]``, whose signed
    area is -1. With inward normals ``h`` read -0.5139 at the box CENTRE and -1.5 outside, so ``-h`` was
    MINIMAL at the centre and confinement became an attractive well -- five squares collapsed onto a
    single point at [0.5, 0.5].

    ``loopFrame`` now normalizes the winding, so this checks the invariance rather than the convention.
    The second half checks the sign of the confinement force, which is the thing the collapse got
    backwards."""
    clockwise = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    counter = clockwise[::-1].copy()
    assert sd.signedAreaOf(clockwise) < 0.0 < sd.signedAreaOf(counter), "the fixture is not what it says"

    probes = np.array([[0.5, 0.5], [0.1, 0.5], [1.5, 0.5], [-0.3, 0.2]])
    depthClockwise = sd.softDepth(probes, clockwise, 1e-2)[0]
    depthCounter = sd.softDepth(probes, counter, 1e-2)[0]
    worst = float(np.abs(depthClockwise - depthCounter).max())
    print(f"  12. h agrees between windings to {worst:.2e};   h at box centre "
          f"{depthClockwise[0]:+.4f}, outside {depthClockwise[2]:+.4f}")
    assert worst < 1e-14, "winding changes the depth -- loopFrame is not normalizing"
    assert depthClockwise[0] > 0.0, "h is not positive INSIDE the loop"
    assert depthClockwise[2] < 0.0, "h is not negative OUTSIDE the loop"

    # A square poking out of the wall must be pushed back in, and harder the further out it is.
    previous = 0.0
    for center in (0.92, 1.00, 1.10):
        poking = square(center, 0.5, 0.2)
        energy, forces, wallForces = sd.edgeLoopEnergyForce(
            poking, clockwise, 1e-2, 1.0, order = 32, confine = True)
        inward = float(forces.sum(axis = 0)[0])
        net = float(np.abs(forces.sum(axis = 0) + wallForces.sum(axis = 0)).max())
        print(f"  12. square at x={center:.2f} pokes out {poking[:, 0].max() - 1.0:+.3f}   "
              f"E {energy:.3e}   net inward force {inward:+.3e}   sum F {net:.1e}")
        assert inward < previous, "the container is not pushing inward, or not harder further out"
        assert net < 1e-12, "container forces do not sum to zero"
        previous = inward

    inside = square(0.5, 0.5, 0.2)
    quiet, _, _ = sd.edgeLoopEnergyForce(inside, clockwise, 1e-2, 1.0, order = 32, confine = True)
    print(f"  12. square fully inside the wall: E {quiet:.3e}")
    assert quiet == 0.0, "a polygon wholly inside the container is being penalized"


def main():
    print("soft penetration depth (notes/softDepth-1.pdf)")
    checkIdentity()
    checkInvariance()
    checkBounds()
    checkEquilibriumCubic()
    checkAdhesionWork()
    checkCornerGeometry()
    checkVertexDerivatives()
    checkBoundaryIntegral()
    checkQuadratureForces()
    checkQuadratureConvergence()
    checkPeriodicAssembly()
    checkOrientationAndConfinement()
    print("all checks passed")


if __name__ == "__main__":
    main()
