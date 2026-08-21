"""Smooth penetration depth for point-loop contact -- the log-sum-exp softmin of half-plane distances,
with Hertzian repulsion and optional adhesion.

Implements Cam's ``notes/softDepth-2.pdf``. Equation numbers below refer to that document (they differ
from the -1 draft, which numbered the soft depth (5) rather than (7)).

WHAT THIS IS, AND HOW IT DIFFERS FROM THE OTHER TIERS. The sharp and mollified tiers both measure
contact by OVERLAP AREA between two loops. This one measures a PENETRATION DEPTH of a point into a
loop, and builds the energy on that depth instead. The consequences are the ones the note sets out:

  - the contact law is Hertzian in the depth, ``E = (2/5) k h_+^(5/2)``, so the force ``k h_+^(3/2)``
    is C2 at first contact rather than jumping;
  - it is real-analytic EVERYWHERE, with no branch-following and no measure-zero nonsmooth set, so the
    medial axis and the vertex Voronoi walls -- where an exact distance function is only C0 and C1
    respectively -- cost nothing;
  - gradient AND Hessian are closed-form, (8) and (10), with no quadrature.

THE OBSTRUCTION IT ACCEPTS (sec 2). No smooth function can be both exactly the distance and smooth: a
bounded region forces an interior maximum of the distance where the eikonal ``|grad phi| = 1`` must
fail. So the error is spent deliberately -- ``|grad h| = 1`` to exponential accuracy in the shallow
regime, all defect absorbed into an ``epsilon``-collar of the medial axis and an ``epsilon``
neighborhood of the vertices. Read the smooth object as THE model: ``h_eps`` defines a particle with
analytically rounded corners of radius (19), and the polygon is the sharp-epsilon limit that motivated
it. ``epsilon`` is a shape parameter, not a numerical regulator.

CONVEXITY IS REQUIRED. Lemma 1 (interior exactness, ``dist = min_i ell_i``) holds for convex loops.
For a nonconvex loop the line supporting an edge adjacent to a reflex vertex cuts the interior and
``min_i ell_i`` goes negative inside. Sec 15 gives the remedy -- decompose into convex pieces and
combine with a softMAX (46) -- which is NOT implemented here; ``softDepthEnergyForce`` checks convexity
and refuses rather than returning a wrong answer.
"""

# UNVERIFIED(Cam)

from functools import lru_cache

import numpy as np

from packing import minImageShift


# Exponent of the repulsive law. 5/2 is the Hertz value and the smallest that delivers C2 (sec 8): with
# E = (2/5) k h^(5/2), both phi' = k h^(3/2) and phi'' = (3/2) k h^(1/2) vanish as h -> 0+, so the
# extension by zero is C2. A harmonic law would leave phi'' discontinuous at contact, putting a jump in
# the stiffness matrix every time a contact forms or breaks.
_HERTZ = 2.5

# Safeguarded-Newton iterations used to locate a contact crossing along an edge. Fixed rather than
# tolerance-driven so the whole sweep stays vectorized and branch-free; see _bracketedRoot for why the
# accuracy requirement is far weaker than the count suggests.
# 16 rather than 24: the safeguard degrades to bisection in the worst case, so the guaranteed bracket
# is 2^-16 = 1.5e-05 -- inside check 8's 1e-04 tolerance and inside the 5e-05 resolution of the dense
# scan it is checked against. 12 was tried and fails that check at 2.4e-04, even though the ENERGY error
# is 1.2e-13: the energy is insensitive to the root (O(d^(7/2))), the interval location is not.
_ROOT_STEPS = 16

_TINY = 1e-300


@lru_cache(maxsize = 16)
def _gaussRule(order):
    """Gauss-Legendre nodes and weights on [-1, 1], cached.

    ``leggauss`` solves for the nodes every call -- it showed up as 270 ``legval`` calls and ~18% of a
    force evaluation. The rule depends only on the order, so it is computed once per order per run."""
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return nodes, weights


# UNVERIFIED(Cam)
def loopFrame(loop):
    """``(edges, tangents, normals, lengths, offsets)`` for one CCW loop, all (n, ...) arrays.

    ``normals`` are OUTWARD, ``n_i = J t_i`` with ``J = [[0, 1], [-1, 0]]`` (eq 1), and ``offsets`` are
    the ``c_i = n_i . v_i`` of (2), so the half-plane function is ``ell_i(x) = c_i - n_i . x``.

    ORIENTATION IS NORMALIZED HERE, and it is not a nicety. ``n_i = J t_i`` is outward only for a CCW
    loop; hand it a CLOCKWISE one and every normal points inward, so ``ell_i`` and hence ``h`` change
    sign and the model inverts. Measured on a clockwise unit box -- which is what
    ``[[0,0],[0,1],[1,1],[1,0]]`` is -- ``h`` read -0.5139 at the box CENTRE and -1.5 outside it, so
    ``-h`` was MINIMAL at the centre and the confinement term became an attractive well: five squares in
    a walled box collapsed onto a single point at [0.5, 0.5].

    Normalizing costs one shoelace sum and removes a whole class of caller error, so it happens once,
    here, rather than being a precondition every call site has to remember."""
    loop = np.asarray(loop, dtype = float)
    following = np.roll(loop, -1, axis = 0)
    edges = following - loop
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    tangents = edges / lengths[:, None]
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis = -1)
    if np.sum(loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1]) < 0.0:
        normals = -normals
    offsets = np.einsum("ij,ij->i", normals, loop)
    return edges, tangents, normals, lengths, offsets


# UNVERIFIED(Cam)
def signedAreaOf(loop):
    """Shoelace signed area; positive for a CCW loop. Exposed so callers can check a winding."""
    loop = np.asarray(loop, dtype = float)
    following = np.roll(loop, -1, axis = 0)
    return 0.5 * float(np.sum(loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1]))


# UNVERIFIED(Cam)
def isConvex(loop, tolerance = 1e-12):
    """Whether a CCW loop is convex -- the precondition of Lemma 1.

    Tested on the sign of the cross product of consecutive edges rather than on ``min_i ell_i``,
    because the latter is what fails and would be circular."""
    loop = np.asarray(loop, dtype = float)
    edges = np.roll(loop, -1, axis = 0) - loop
    following = np.roll(edges, -1, axis = 0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool(np.all(cross >= -tolerance * np.abs(cross).max(initial = 1.0)))


# Relative tolerance for calling two half-planes the same line, i.e. for detecting a collinear vertex
# triple. Compared against the normals' agreement and the offsets' difference scaled by the loop size.
_COLLINEAR = 1e-10


# UNVERIFIED(Cam)
def nonConvexPolygons(packing):
    """Indices of the packing's polygons that are NOT convex, and their reflex-corner counts.

    THE PRECONDITION IS NOT COSMETIC, AND ITS VIOLATION IS SILENT. Lemma 1 (``dist = min_i ell_i``)
    holds only for convex loops. With a reflex vertex the line supporting an adjacent edge cuts the
    interior, ``min_i ell_i`` goes negative INSIDE, and so does ``h``. Then ``[h]_+`` never fires over
    most of the interior and the boundary integral collects NOTHING no matter how deep the overlap.

    Measured on Cam's ``N=32, n=32, kappa=4`` packing: 197 reflex corners across 1024 vertices, ``h`` at
    one polygon's own CENTROID reading -4.29e-02, softDepth reporting E = 5.2e-13 and max|F| = 1.9e-10
    -- "converged" -- on a configuration the sharp tier scores at E = 3.907 with max|F| = 0.658. FIRE
    had nothing to minimize and returned the mess it started with.

    ``isConvex`` existed for this from the beginning and was never called on the energy path. That is
    the bug; this is the guard."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    container = getattr(packing, "containerIndex", None)
    stop = int(packing.numPolygons) if container is None else int(container)
    offenders = []
    for polygon in range(stop):
        loop = vertices[starts[polygon]:starts[polygon + 1]]
        if isConvex(loop):
            continue
        edges = np.roll(loop, -1, axis = 0) - loop
        following = np.roll(edges, -1, axis = 0)
        cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        offenders.append((polygon, int((cross < 0.0).sum())))
    return offenders


def requireConvex(packing):
    """Raise unless every polygon is convex. Called on the soft-depth energy path, every evaluation.

    Every evaluation rather than once, because a minimizer can drive a convex polygon non-convex
    mid-run. Failing loudly in the middle of a descent loses that descent; returning a near-zero energy
    instead loses the result AND hides it, which is strictly worse."""
    offenders = nonConvexPolygons(packing)
    if not offenders:
        return
    listed = ", ".join(f"#{polygon} ({reflex} reflex)" for polygon, reflex in offenders[:6])
    more = "" if len(offenders) <= 6 else f" and {len(offenders) - 6} more"
    raise ValueError(
        f"soft depth is CONVEX-ONLY by decision (2026-08-01) and {len(offenders)} of "
        f"{packing.numPolygons} polygons are not convex: {listed}{more}.\n"
        f"    On a non-convex loop min_i ell_i goes negative INSIDE, so h does too, [h]_+ never fires "
        f"and the energy silently collapses to ~0 however deep the overlap -- a minimizer then reports "
        f"'converged' on a mess. That is why this raises rather than warns.\n"
        f"    generateEquilateralPolygons cannot make convex shapes above n=4 at any kappa, so this is "
        f"expected there. Use n=4, or supply convex loops directly.\n"
        f"    A working non-convex tier EXISTS (convexDifference.py, the convex differences tree) and "
        f"is reachable with allowNonConvex = True. It is off by default because it is ~12.7 s per force "
        f"evaluation at N=32 n=32 with no CUDA path, and because it carries the reflex-corner bias in "
        f"TODO.md. See notes/penetrationDepthReview.md for why convex-only was chosen.")


# UNVERIFIED(Cam)
def meshWarnings(loop, adhesionRange = None):
    """The two mesh conditions of sec 17, which are properties of the LOOP rather than of the physics.
    Returns a list of complaint strings, empty when the loop is clean.

    REDUNDANT HALF-PLANES. A half-plane whose line does not touch the region is free in the sharp
    representation -- it leaves ``min_i ell_i`` unchanged -- but NOT in the smooth one: it adds a term
    to the sum in Proposition 1, widening the deficit bound from ``eps log N`` to ``eps log(N+1)`` and
    pulling the zero set further inward (sec 6). A collinear vertex triple produces exactly this,
    duplicating one line in the edge list.

    MINIMUM EDGE LENGTH. Sec 14 measures the hazard: a near-tied runner-up face can carry softmin
    weight approaching 1/2 while its foot ``s_i`` lies far outside its own segment, and the excursion
    scales as the polygon's diameter over its shortest edge -- reaching ``|s_i| ~ 1e2`` for a loop with
    one short edge. Conservation survives (the torque cancellation is proven identically in ``s_i``),
    but the individual vertex forces become large and nearly cancelling. It is a CONDITIONING hazard,
    not a correctness failure.

    Both are to be fixed when the shapes are GENERATED. The note is explicit that clamping ``s_i`` at
    run time is the wrong repair: it would break ``d/dv_i + d/dv_{i+1} + d/dx = 0`` and with it the
    exact conservation, trading a conditioning problem for a correctness one."""
    loop = np.asarray(loop, dtype = float)
    _, _, normals, lengths, offsets = loopFrame(loop)
    complaints = []

    scale = float(np.abs(loop).max(initial = 1.0))
    for i in range(len(loop)):
        j = (i + 1) % len(loop)
        if (abs(float(normals[i] @ normals[j]) - 1.0) < _COLLINEAR
                and abs(float(offsets[i] - offsets[j])) < _COLLINEAR * scale):
            complaints.append(
                f"edges {i} and {j} are collinear, so they contribute the SAME half-plane twice; "
                f"prune the redundant vertex (sec 6 / sec 17)")

    shortest, longest = float(lengths.min()), float(lengths.max())
    if shortest < 1e-3 * longest:
        complaints.append(
            f"shortest edge {shortest:.3e} is {longest / shortest:.0f}x below the longest; the "
            f"off-segment excursions |s_i| scale with that ratio and make the vertex forces large and "
            f"nearly cancelling (sec 14)")
    if adhesionRange is not None and shortest <= 10.0 * adhesionRange:
        complaints.append(
            f"shortest edge {shortest:.3e} is not >> adhesionRange {adhesionRange:.3e}; the contact "
            f"chord is smeared across several faces and the face/vertex distinction is lost (sec 17)")
    return complaints


# UNVERIFIED(Cam)
def exteriorFactor(interiorAngle):
    """``min_i ell_i / phi = sin(theta/2)`` beyond a vertex, eq (5) -- how much the softmin UNDERSTATES
    the exterior distance in the normal fan past a corner.

    1 for a flat boundary, 0.87 for a hexagon, 0.71 for a square, 0.5 for a triangle. It does not
    matter for a repulsive-only law, since the exterior magnitude never enters the energy, but it does
    once adhesion is added -- sec 13 records the resulting systematic inflation of the measured chord
    by ``1/sin(theta/2)``."""
    return float(np.sin(0.5 * float(interiorAngle)))


# UNVERIFIED(Cam)
def softDepth(points, loop, epsilon, frame = None):
    """Smooth penetration depth ``h_eps`` of each point into one convex loop, plus everything the
    derivatives need. Eq (7), evaluated in the shifted form (48).

    Returns ``(h, weights, normals, feet)`` where ``h`` is (P,), ``weights`` is (P, n) -- the softmin
    weights ``w_i`` of (9) -- ``normals`` is (n, 2) and ``feet`` is (P, n), the normalized
    foot-of-perpendicular parameters ``s_i`` of (41).

    THE SHIFT IS NOT OPTIONAL. Writing ``m = min_i ell_i``, (43) gives
    ``h = m - eps log sum_i exp(-(ell_i - m)/eps)`` in which every exponent is <= 0 and the sum lies in
    [1, N]. The unshifted form overflows as soon as ``ell_i/eps`` exceeds a few hundred, which at
    ``eps ~ 1e-3`` in units of the particle size is immediate.

    Note ``h`` is POSITIVE inside the loop and negative outside, and its sign is exact everywhere
    (``min_i ell_i > 0`` iff inside) even though the magnitude is not exact outside -- which is all a
    repulsive law needs, since the exterior magnitude never enters the energy."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    # The frame is CONSTANT for a given loop, so it is hoisted by callers that evaluate the same loop
    # many times -- the root finder does ~50 evaluations per edge. Recomputing it inside the loop was
    # 1419 loopFrame calls per force evaluation of five squares, most of the runtime.
    _, tangents, normals, lengths, offsets = loopFrame(loop) if frame is None else frame
    loop = np.asarray(loop, dtype = float)

    # ell_i(x) = c_i - n_i . x, one row per point.
    ell = offsets[None, :] - points @ normals.T
    lowest = ell.min(axis = 1, keepdims = True)
    shifted = np.exp(-(ell - lowest) / epsilon)
    total = shifted.sum(axis = 1, keepdims = True)
    h = (lowest - epsilon * np.log(total))[:, 0]
    weights = shifted / total

    # s_i = t_i . (x - v_i) / |e_i|, eq (41). Can fall outside [0, 1] when the foot lies beyond an
    # endpoint; the note is explicit that this is fine -- the lever rule extrapolates and stays exactly
    # conservative, and a large |s| is suppressed by a small w.
    delta = points[:, None, :] - loop[None, :, :]
    feet = np.einsum("pij,ij->pi", delta, tangents) / lengths[None, :]
    return h, weights, normals, feet


# UNVERIFIED(Cam)
def depthGradientHessian(weights, normals, epsilon):
    """``(gradient, hessian)`` of ``h_eps`` with respect to the POINT, eqs (10) and (12).

        grad h = -nbar,        nbar = sum_i w_i n_i
        hess h = -(1/eps) Cov_w(n)

    Both are O(N) with no transcendental inversions. The Hessian is negative semidefinite because
    ``h_eps`` is a softmin of affine functions and so concave -- a statement that the shape is convex,
    not a defect (sec 4)."""
    mean = np.einsum("pi,ij->pj", weights, normals)
    second = np.einsum("pi,ij,ik->pjk", weights, normals, normals)
    hessian = -(second - np.einsum("pj,pk->pjk", mean, mean)) / epsilon
    return -mean, hessian


# UNVERIFIED(Cam)
def eikonalDefect(weights, normals, epsilon):
    """``|grad h|^2 - eps * laplacian(h) - 1``, which is identically ZERO by (14).

    The cheapest and strongest validation in the note: it tests the gradient and the Hessian trace
    together against an exact identity, at machine precision, with no reference solution and no finite
    differencing. (14) is the viscosity regularization of the eikonal equation, which is what makes
    (5) the exact solution of a linear problem -- the screened Poisson equation (15) under the
    Cole-Hopf substitution -- rather than an ad hoc smoothing."""
    gradient, hessian = depthGradientHessian(weights, normals, epsilon)
    laplacian = np.einsum("pjj->p", hessian)
    return np.einsum("pj,pj->p", gradient, gradient) - epsilon * laplacian - 1.0


# UNVERIFIED(Cam)
def plummerStep(h, adhesionRange):
    """``(chi, chi', chi'')`` -- the Plummer-mollified Heaviside of (23) and its derivatives (24).

        chi(h) = (1/2) (1 + h / sqrt(h^2 + lam^2))

    Entire, so the adhesive term needs no positive part, no cutoff and no case analysis, and
    contributes no discontinuity to any derivative anywhere.

    THE TWO MOLLIFIERS ARE COMPLEMENTARY AND NOT INTERCHANGEABLE (sec 16). This one SATURATES, carrying
    no information beyond ``|h| ~ lam`` -- useless as a depth, exactly right as a membership function,
    because once contact is established the bond energy must stop growing. Log-sum-exp does not
    saturate, which is what a depth needs and what an adhesion must not have."""
    lam = float(adhesionRange)
    root = np.sqrt(h * h + lam * lam)
    chi = 0.5 * (1.0 + h / root)
    first = 0.5 * lam * lam / root ** 3
    second = -1.5 * lam * lam * h / root ** 5
    return chi, first, second


# UNVERIFIED(Cam)
def contactLaw(h, stiffness, adhesionWork = 0.0, adhesionRange = 1.0):
    """``(energy, phi', phi'')`` of the scalar contact law (23), evaluated per point.

        E = (2/5) k [h]_+^(5/2)  -  W chi_lam(h)

    With ``adhesionWork = 0`` this is the purely repulsive Hertzian law (20)."""
    positive = np.maximum(h, 0.0)
    energy = 0.4 * stiffness * positive ** _HERTZ
    first = stiffness * positive ** 1.5
    second = 1.5 * stiffness * np.sqrt(positive)
    if adhesionWork != 0.0:
        chi, chiFirst, chiSecond = plummerStep(h, adhesionRange)
        energy = energy - adhesionWork * chi
        first = first - adhesionWork * chiFirst
        second = second - adhesionWork * chiSecond
    return energy, first, second


# UNVERIFIED(Cam)
def equilibriumIndentation(stiffness, adhesionWork, adhesionRange):
    """The preferred overlap ``h*`` where ``phi'(h*) = 0``, from the depressed cubic (28) via Cardano
    (29).

        eta^3 + eta = mu,     mu = (W / (2 k lam^(5/2)))^(2/3),     h* = lam * eta

    The discriminant ``mu^2/4 + 1/27`` is strictly positive, so the root is unique for every ``mu > 0``
    -- the contact has exactly one equilibrium indentation, never several.

    ``mu`` is a Tabor parameter: the ratio of the elastic indentation produced by the peak adhesive
    force to the range of that force. ``mu << 1`` is DMT (weakly cohesive, ductile), ``mu >> 1`` is JKR
    (strongly cohesive, brittle), so sweeping it at fixed ``k`` is a physically interpretable
    brittle-ductile axis.

    EVALUATED IN HYPERBOLIC FORM, NOT AS WRITTEN IN (25). The two cube roots there differ by
    ``2 sqrt(mu^2/4 + 1/27)``, which tends to ``2/sqrt(27)`` as ``mu -> 0``: the expression subtracts
    two nearly-equal quantities and loses precision exactly in the DMT limit. Measured, the residual of
    ``eta^3 + eta - mu`` degraded to 5.9e-11. For a depressed cubic ``t^3 + p t + q`` with ``p > 0``
    there is one real root and it has the cancellation-free form

        eta = (2/sqrt 3) sinh( (1/3) asinh( (3 sqrt 3 / 2) mu ) ),

    identical analytically -- it reproduces ``eta ~ mu - mu^3`` as ``mu -> 0`` and ``eta ~ mu^(1/3)``
    as ``mu -> inf``, the two limits the note quotes -- and accurate to roundoff across the whole
    range."""
    if adhesionWork <= 0.0:
        return 0.0
    mu = (adhesionWork / (2.0 * stiffness * adhesionRange ** 2.5)) ** (2.0 / 3.0)
    eta = (2.0 / np.sqrt(3.0)) * np.sinh(np.arcsinh(1.5 * np.sqrt(3.0) * mu) / 3.0)
    return float(adhesionRange * eta)


# UNVERIFIED(Cam)
def cornerRadius(epsilon, interiorAngle):
    """Radius of the smoothed corner, eq (19): ``R = eps sin(theta/2) / cos^2(theta/2)``.

    ``sqrt(2) eps`` for a right angle. The zero set crosses the corner bisector at ``ell = eps log 2``
    (14), independent of the corner angle. Sharp spikes are the expensive feature: as ``theta -> 0``
    the radius collapses while the apex offset stays at ``eps log 2``."""
    half = 0.5 * float(interiorAngle)
    return float(epsilon) * np.sin(half) / np.cos(half) ** 2


# UNVERIFIED(Cam)
def pointLoopEnergyForce(points, loop, epsilon, stiffness, adhesionWork = 0.0, adhesionRange = 1.0):
    """Energy and forces for a set of POINTS against one convex LOOP.

    Returns ``(energy, pointForces, loopForces)``. Forces are ``-dE/dr``, and BOTH sets are returned
    because the energy depends on the loop's vertices too -- omitting the loop side is what would break
    conservation.

    The vertex derivative is (43)/(45): the normal load on edge ``i`` is distributed to its two
    endpoints with barycentric weights set by where the perpendicular from the point meets the edge
    line,

        d ell_i / d v_i = (1 - s_i) n_i,     d ell_i / d v_{i+1} = s_i n_i,     d ell_i / d x = -n_i

    so that summing over all three arguments gives ``(1 - s_i) n_i + s_i n_i - n_i = 0`` identically.
    Translating the point and the loop together therefore leaves every ``ell_i`` invariant and the
    forces sum to zero IN FLOATING POINT, not merely to within truncation error (requirement v). The
    torque balance (39) is exact edge by edge for the same reason."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    h, weights, normals, feet = softDepth(points, loop, epsilon)
    energy, first, _ = contactLaw(h, stiffness, adhesionWork, adhesionRange)

    # Force on each point: -dE/dx = -phi'(h) dh/dx = +phi'(h) nbar.
    mean = np.einsum("pi,ij->pj", weights, normals)
    pointForces = first[:, None] * mean

    # Force on loop vertex j: -phi'(h) [ w_{j-1} s_{j-1} n_{j-1} + w_j (1 - s_j) n_j ].
    # Edge i runs v_i -> v_{i+1} and hands (1 - s_i) of its load to v_i and s_i to v_{i+1}, so the
    # second term is the first rolled forward one vertex.
    toOwnEnd = np.einsum("p,pi,ij->ij", first, weights * (1.0 - feet), normals)
    toNextEnd = np.einsum("p,pi,ij->ij", first, weights * feet, normals)
    loopForces = -(toOwnEnd + np.roll(toNextEnd, 1, axis = 0))
    return float(energy.sum()), pointForces, loopForces


# UNVERIFIED(Cam)
def depthAlongEdges(vStart, vEnd, t, loop, epsilon, frame = None):
    """``(h, dh/dt)`` at ONE parameter per edge, for the segments ``vStart -> vEnd`` against ``loop``.

    The workhorse of the root finding below: both values come from a single ``softDepth`` call, since
    the gradient (10) is assembled from the same softmin weights as the depth itself."""
    direction = np.asarray(vEnd, dtype = float) - np.asarray(vStart, dtype = float)
    points = np.asarray(vStart, dtype = float) + np.asarray(t, dtype = float)[:, None] * direction
    h, weights, normals, _ = softDepth(points, loop, epsilon, frame)
    gradient, _ = depthGradientHessian(weights, normals, epsilon)
    return h, np.einsum("pj,pj->p", gradient, direction)


def _bracketedRoot(vStart, vEnd, loop, epsilon, lo, hi, positiveAtLo, frame = None):
    """Root of ``h`` along each edge inside the bracket ``[lo, hi]``, by Newton with a bisection
    safeguard: the Newton step is taken only when it lands inside the live bracket, so the iteration
    inherits bisection's guarantee while converging at Newton's rate in practice.

    The accuracy demanded here is very weak. Misplacing the crossing by ``d`` changes the energy by the
    integral over a sliver where ``h`` is itself O(d), i.e. by O(d^(7/2)) -- the same vanishing that
    kills the Leibniz boundary term. ``_ROOT_STEPS`` is far more than the energy needs; it is set for
    the finite-difference checks, which see the root through a much less forgiving lens."""
    t = 0.5 * (lo + hi)
    for _ in range(_ROOT_STEPS):
        value, slope = depthAlongEdges(vStart, vEnd, t, loop, epsilon, frame)
        atLoSide = (value > 0.0) == positiveAtLo
        lo = np.where(atLoSide, t, lo)
        hi = np.where(atLoSide, hi, t)
        step = np.divide(value, slope, out = np.full_like(value, np.inf),
                         where = np.abs(slope) > _TINY)
        newton = t - step
        t = np.where((newton > lo) & (newton < hi), newton, 0.5 * (lo + hi))
    return t


# UNVERIFIED(Cam)
def contactIntervals(loopA, loopB, epsilon, frame = None):
    """For each EDGE of ``loopA``, the sub-interval ``[t0, t1]`` of [0, 1] on which ``h_eps^B >= 0``.

    Returns ``(t0, t1)`` with ``t1 == t0`` on edges that never enter contact.

    THERE IS AT MOST ONE SUCH INTERVAL PER EDGE, and that is a theorem rather than an assumption.
    ``h_eps = -eps log sum_i exp(-ell_i/eps)`` is minus ``eps`` times a log-sum-exp of AFFINE functions;
    log-sum-exp is convex and nondecreasing in each argument, so the composition is convex and
    ``h_eps`` is CONCAVE. A concave function restricted to a line is concave, its superlevel set
    ``{h >= 0}`` is convex, and a convex subset of a line is an interval. So an edge cannot enter and
    leave contact twice, no candidate-crossing enumeration is needed, and the two roots are bracketed
    by the peak.

    Concavity also locates the peak for free: ``dh/dt`` is decreasing, so it sits at an endpoint unless
    the slope changes sign, and when it does a plain bisection on the slope finds it."""
    vStart = np.asarray(loopA, dtype = float)
    return contactIntervalsForEdges(vStart, np.roll(vStart, -1, axis = 0), loopB, epsilon, frame)


# UNVERIFIED(Cam)
def contactIntervalsForEdges(vStart, vEnd, loopB, epsilon, frame = None):
    """``contactIntervals`` on an EXPLICIT edge list rather than a closed loop.

    The loop form assumes only that consecutive vertices form the edges, which is one ``np.roll``.
    Taking the edges explicitly lets unrelated edges -- every polygon's, against a shared container --
    be solved in ONE vectorized sweep instead of one sweep per polygon. Measured on a walled 5-square
    packing, the per-polygon form spent 93% of the force evaluation here."""
    vStart = np.asarray(vStart, dtype = float)
    vEnd = np.asarray(vEnd, dtype = float)
    count = len(vStart)
    zeros, ones = np.zeros(count), np.ones(count)

    frame = loopFrame(loopB) if frame is None else frame
    hLo, slopeLo = depthAlongEdges(vStart, vEnd, zeros, loopB, epsilon, frame)
    hHi, slopeHi = depthAlongEdges(vStart, vEnd, ones, loopB, epsilon, frame)

    tPeak = np.where(slopeLo <= 0.0, 0.0, 1.0)
    interior = (slopeLo > 0.0) & (slopeHi < 0.0)
    if interior.any():
        lo, hi = zeros.copy(), ones.copy()
        for _ in range(_ROOT_STEPS):
            mid = 0.5 * (lo + hi)
            _, slope = depthAlongEdges(vStart, vEnd, mid, loopB, epsilon, frame)
            rising = slope > 0.0
            lo = np.where(rising, mid, lo)
            hi = np.where(rising, hi, mid)
        tPeak = np.where(interior, 0.5 * (lo + hi), tPeak)
    hPeak, _ = depthAlongEdges(vStart, vEnd, tPeak, loopB, epsilon, frame)

    active = hPeak > 0.0
    if not active.any():
        return zeros, zeros

    entering = _bracketedRoot(vStart, vEnd, loopB, epsilon, zeros, tPeak, False, frame)
    leaving = _bracketedRoot(vStart, vEnd, loopB, epsilon, tPeak, ones, True, frame)
    t0 = np.where(active, np.where(hLo > 0.0, 0.0, entering), 0.0)
    t1 = np.where(active, np.where(hHi > 0.0, 1.0, leaving), 0.0)
    return t0, t1


# UNVERIFIED(Cam)
def envelopeCuts(vStart, vEnd, loop, epsilon, t0, t1, frame = None):
    """Parameters inside each edge's contact interval where the softmin's ACTIVE half-plane switches.

    Returns ``(nEdges, maxCuts)``, padded with ``t1`` so every row is the same length; a padded entry
    makes a zero-length panel, which contributes exactly nothing.

    WHY THIS IS NEEDED ON TOP OF THE ``h = 0`` SPLIT. The crossing split removes the 5/2 branch point,
    but the integrand keeps a second, subtler feature: where the lower envelope ``min_i ell_i`` switches
    from one half-plane to another, ``h_eps`` turns over on the scale of ``epsilon``. On an edge of
    length 1 with ``epsilon = 1e-3`` that is a feature a thousand times smaller than the domain, and a
    single Gauss panel simply cannot see it -- measured, order 32 on one panel is wrong by 2.2e-03 and
    NON-MONOTONE in the order, while splitting here brings the same order to 8.9e-08.

    The cuts are exact. Each ``ell_i`` is AFFINE along the edge, so a switch is the crossing of two
    lines and comes from a linear solve, not a root find. They are located by probing the sharp argmin
    and solving exactly wherever it changes, which finds every envelope segment longer than the probe
    spacing -- and a segment shorter than that is shorter than ``epsilon``, where the softmin has
    smoothed the switch away in any case."""
    vStart = np.asarray(vStart, dtype = float)
    _, _, normals, _, offsets = loopFrame(loop) if frame is None else frame
    start = offsets[None, :] - vStart @ normals.T
    slope = (offsets[None, :] - np.asarray(vEnd, dtype = float) @ normals.T) - start

    probes = max(16, 2 * len(normals))
    fraction = (np.arange(probes) + 0.5) / probes
    t = t0[:, None] + (t1 - t0)[:, None] * fraction[None, :]
    ell = start[:, None, :] + slope[:, None, :] * t[:, :, None]
    active = ell.argmin(axis = 2)

    changed = active[:, 1:] != active[:, :-1]
    before, after = active[:, :-1], active[:, 1:]
    rows = np.arange(len(vStart))[:, None]
    gap = slope[rows, before] - slope[rows, after]
    crossing = np.divide(start[rows, after] - start[rows, before], gap,
                         out = np.full(before.shape, np.inf), where = np.abs(gap) > _TINY)
    inside = changed & (crossing > t0[:, None]) & (crossing < t1[:, None])

    ordered = np.sort(np.where(inside, crossing, np.inf), axis = 1)
    keep = int(inside.sum(axis = 1).max())
    if keep == 0:
        return np.empty((len(vStart), 0))
    cuts = ordered[:, :keep]
    return np.where(np.isfinite(cuts), cuts, t1[:, None])


# UNVERIFIED(Cam)
def edgeLoopEnergyForce(loopA, loopB, epsilon, stiffness, adhesionWork = 0.0, adhesionRange = 1.0,
                        order = 16, confine = False):
    """The boundary-area energy: the BOUNDARY of ``loopA`` integrated against the depth field of
    ``loopB``, eq (37) with the repulsive law in place of the adhesive one,

        E = int_{dA} phi(h_eps^B(x)) dl(x)

    by Gauss-Legendre of the given ``order`` on the contact sub-interval of each edge.

    SPLITTING AT THE CROSSING IS WHAT MAKES THE RULE ACCURATE. ``h_eps^B`` is real-analytic everywhere,
    but ``phi`` carries a 5/2 branch point at ``h = 0``, so a rule spanning the crossing integrates a
    function with a jump in its third derivative and converges algebraically. Restricted to the contact
    piece the integrand is smooth and the order buys the convergence it is supposed to.

    THE MOVING LIMITS CONTRIBUTE NOTHING. ``t0`` and ``t1`` depend on the vertices of BOTH loops, but by
    Leibniz their variation enters only through the integrand evaluated at them -- and ``phi(h) = 0``
    there, by the definition of the crossing. They are therefore held fixed when differentiating, which
    is why no derivative of the root finder is needed. Check 7's finite difference is what confirms it.

    ``confine = True`` FLIPS THE ROLE OF THE LOOP FROM OBSTACLE TO CONTAINER, penalizing ``[-h]_+``
    instead of ``[h]_+`` -- being OUTSIDE rather than inside. Two things change and nothing else:

      - the domain is the COMPLEMENT of the contact interval. ``h`` is concave, so ``{h >= 0}`` on an
        edge is one interval and ``{h <= 0}`` is up to TWO, namely ``[0, t0]`` and ``[t1, 1]``. Nothing
        new has to be solved for; the same roots bound both.
      - the depth handed to the contact law is ``-h``, so ``phi'`` enters with the opposite sign.

    This replaces an earlier attempt that reversed the loop's WINDING on the reasoning that this negates
    ``h``. Reversing negates every ``ell_i``, but ``h`` is their softMIN and ``min(-ell) = -MAX(ell)``,
    so the reversed loop's ``h`` came out negative both inside AND outside -- measured -0.51 at the
    center of the unit box and -2.0 a full unit outside it -- and ``[h]_+`` never fired. The container
    contributed exactly zero, always.

    WHAT ``-h`` MEASURES OUTSIDE, and its bias. ``-h = softmax_i(-ell_i)`` is the largest single
    half-plane violation. Through a FACE that is exactly the distance to the wall. Past a CORNER of
    interior angle ``theta`` it is short by ``sin(theta/2)`` -- 0.707 at a square's corner -- the same
    exterior inexactness recorded for reflex corners in TODO.md. Euclidean exterior distance would need
    vertex terms in the softmin, which have no working construction (see that entry).

    Returns ``(energy, forcesA, forcesB)``, each force being ``-dE/dr`` on that loop's vertices."""
    loopA = np.asarray(loopA, dtype = float)
    order0 = np.arange(len(loopA))
    return edgeSetAgainstLoop(loopA, np.roll(loopA, -1, axis = 0), order0,
                              (order0 + 1) % len(loopA), len(loopA), loopB, epsilon, stiffness,
                              adhesionWork, adhesionRange, order, confine)


# UNVERIFIED(Cam)
def edgeSetAgainstLoop(vStart, vEnd, firstIndex, secondIndex, vertexCount, loopB, epsilon, stiffness,
                       adhesionWork = 0.0, adhesionRange = 1.0, order = 16, confine = False):
    """``edgeLoopEnergyForce`` over an ARBITRARY edge set, scattering onto ``vertexCount`` vertices.

    ``firstIndex`` / ``secondIndex`` say which vertex slots each edge's endpoints occupy, so edges from
    many different polygons can be integrated against one shared loop in a single sweep. That is what
    the container does: every polygon's boundary against the same wall, one call instead of N."""
    vStart = np.asarray(vStart, dtype = float)
    vEnd = np.asarray(vEnd, dtype = float)
    firstIndex = np.asarray(firstIndex, dtype = int)
    secondIndex = np.asarray(secondIndex, dtype = int)
    count = len(vStart)
    edgeVectors = vEnd - vStart
    lengths = np.hypot(edgeVectors[:, 0], edgeVectors[:, 1])

    forcesA = np.zeros((vertexCount, 2), dtype = float)
    forcesB = np.zeros((len(loopB), 2), dtype = float)
    frame = loopFrame(loopB)
    t0, t1 = contactIntervalsForEdges(vStart, vEnd, loopB, epsilon, frame)
    if confine:
        # An edge with no contact interval at all lies wholly outside, so the whole edge is penalized.
        outside = t1 <= t0
        leaving = np.where(outside, 1.0, t0)
        entering = np.where(outside, 1.0, t1)
        t0, t1 = np.concatenate([np.zeros(count), entering]), np.concatenate([leaving, np.ones(count)])
        vStart = np.concatenate([vStart, vStart])
        vEnd = np.concatenate([vEnd, vEnd])
        edgeVectors = np.concatenate([edgeVectors, edgeVectors])
        lengths = np.concatenate([lengths, lengths])
        firstIndex = np.concatenate([firstIndex, firstIndex])
        secondIndex = np.concatenate([secondIndex, secondIndex])
        count = 2 * count
    span = t1 - t0
    if not np.any(span > 0.0):
        return 0.0, forcesA, forcesB

    order = int(order)
    nodes, gaussWeights = _gaussRule(order)
    bounds = np.concatenate([t0[:, None], envelopeCuts(vStart, vEnd, loopB, epsilon, t0, t1, frame),
                             t1[:, None]], axis = 1)
    lo, hi = bounds[:, :-1, None], bounds[:, 1:, None]
    local = 0.5 * (hi + lo) + 0.5 * (hi - lo) * nodes[None, None, :]
    weight = lengths[:, None, None] * 0.5 * (hi - lo) * gaussWeights[None, None, :]
    perEdge = local.shape[1] * order
    points = (vStart[:, None, None, :]
              + local[..., None] * edgeVectors[:, None, None, :]).reshape(-1, 2)
    local, weight = local.reshape(-1), weight.reshape(-1)
    # Under confinement the edge list was doubled to carry the two complement pieces, so both halves
    # scatter back onto the SAME original vertices.
    edgeIndex = np.repeat(np.arange(count), perEdge)

    h, weights, normals, feet = softDepth(points, loopB, epsilon, frame)
    # Confinement penalizes being OUTSIDE, so the law is evaluated at -h and phi' enters negated:
    # dE/dx = phi'(-h) d(-h)/dx = -phi'(-h) grad h, against dE/dx = phi'(h) grad h for repulsion.
    density, first, _ = contactLaw(-h if confine else h, stiffness, adhesionWork, adhesionRange)
    if confine:
        first = -first
    energy = float(np.dot(weight, density))

    # Force on a quadrature NODE is the point force of (43) carrying its weight, and the node rides on
    # the edge -- x_q = (1 - t_q) v_i + t_q v_{i+1} -- so it splits onto the two endpoints by the same
    # lever rule the loop side already uses. The split preserves the torque exactly, since
    # (1-t) v_i x F + t v_{i+1} x F = x_q x F.
    mean = np.einsum("pi,ij->pj", weights, normals)
    nodeForces = (weight * first)[:, None] * mean
    np.add.at(forcesA, firstIndex[edgeIndex], (1.0 - local)[:, None] * nodeForces)
    np.add.at(forcesA, secondIndex[edgeIndex], local[:, None] * nodeForces)

    # The MEASURE moves too: dl = |e| dt, so each edge carries a tangential force from d|e|/dv = -+ehat.
    # This term has no analogue in the vertex-sampled law, it is the one a finite difference catches,
    # and it is torque-free on its own -- equal and opposite along the edge it acts on.
    edgeEnergy = (weight * density).reshape(count, perEdge).sum(axis = 1)
    tangential = (edgeEnergy / lengths ** 2)[:, None] * edgeVectors
    np.add.at(forcesA, firstIndex, tangential)
    np.subtract.at(forcesA, secondIndex, tangential)

    scaled = weight * first
    toOwnEnd = np.einsum("p,pi,ij->ij", scaled, weights * (1.0 - feet), normals)
    toNextEnd = np.einsum("p,pi,ij->ij", scaled, weights * feet, normals)
    forcesB = -(toOwnEnd + np.roll(toNextEnd, 1, axis = 0))
    return energy, forcesA, forcesB


# UNVERIFIED(Cam)
def packingEnergyForce(packing, epsilon, stiffness = 1.0, adhesionWork = 0.0, adhesionRange = 1.0,
                       kContainer = 1.0, quadratureOrder = 16, useCuda = None,
                       allowNonConvex = False):
    """Whole-packing soft-depth energy and force. Returns ``(energy, force)`` with force flat (2N,).

    The whole BOUNDARY of each polygon is integrated against each nearby polygon's depth field,
    symmetrized over the pair so neither body is privileged -- ``E_AB`` from A's boundary against B plus
    ``E_BA`` from B's boundary against A. Both sides of every interaction receive their force (the
    quadrature node's and the loop's, via (43)), so momentum and angular momentum are conserved exactly.

    ``quadratureOrder`` is the Gauss-Legendre order PER CONTACTING EDGE. Sampling the vertices alone is
    not a low-order version of this, it is a different and wrong law: two squares meeting face to face
    have no vertex of either inside the other, so a vertex rule reports exactly zero energy against real
    overlap. ``edgeLoopEnergyForce`` carries the details.

    Pairs are culled on the covering-radius test, the same coarse level ``neighbors`` uses: two
    polygons whose centroids are farther apart than the sum of their covering radii cannot touch, and
    ``h_eps`` is negative there so the repulsive law contributes nothing anyway. A skin of
    ``epsilon`` is added because the soft depth's zero set sits an ``O(epsilon)`` offset inside the
    polygon.

    THE CONTAINER IS THE SAME LAW WITH THE SIGN FLIPPED. A wall confines rather than repels, so what is
    penalized is a vertex being OUTSIDE it -- ``[-h]_+`` rather than ``[h]_+`` -- and the wall's own
    vertices take the reaction, which is discarded only because a wall is normally pinned."""
    from energies import polygonCentroidsRadii
    import convexDifference

    # NONCONVEX LOOPS GO THROUGH THE CONVEX DIFFERENCES TREE (notes sec:nonconvex). Convex ones keep
    # the faster envelope-split path unchanged, so nothing about the common case changes. The trees are
    # rebuilt every call because a minimizer can change which vertices are on the hull; the build is
    # O(n log n) per polygon and is nowhere near the cost of the integration.
    trees = {}
    stopFor = int(packing.numPolygons)
    if getattr(packing, "containerIndex", None) is not None:
        stopFor = int(packing.containerIndex)
    allConvex = True
    verts = packing.positions.reshape(-1, 2)
    startsFor = np.asarray(packing.startIndices, dtype = int)
    for polygon in range(stopFor):
        loop = verts[startsFor[polygon]:startsFor[polygon + 1]]
        if isConvex(loop):
            continue
        allConvex = False
        if not allowNonConvex:
            requireConvex(packing)
        trees[polygon] = convexDifference.buildDifferenceTree(loop)
        convexDifference.warnOnSharpPockets(loop, trees[polygon])

    # The device tier carries PAIR interactions with the purely repulsive law. Adhesion still forces the
    # numpy path -- silently dropping it would be worse than being slow -- but a CONTAINER no longer
    # does. The kernel already excludes the container from its pair loop (its `stop` is the container
    # index), so the split is exact: pairs on the device, the confinement term added below on the host.
    # Only that term stays on numpy, and it is O(N) where the pairs are O(N^2).
    #
    # This gate used to require `containerIndex is None`, which sent every walled packing to numpy in
    # full -- measured 1242 ms per force evaluation for FIVE squares and a wall.
    deviceEnergy, deviceForce = None, None
    if useCuda is not False and allConvex and adhesionWork == 0.0:
        try:
            import cudaOverlap
        except ImportError:
            cudaOverlap = None
        if cudaOverlap is not None and cudaOverlap.isAvailable() and quadratureOrder in (16, 32):
            deviceEnergy, deviceForce = cudaOverlap.softDepthCuda(
                packing, epsilon, stiffness, quadratureOrder)
            if getattr(packing, "containerIndex", None) is None:
                return deviceEnergy, deviceForce

    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    box = packing.box
    container = getattr(packing, "containerIndex", None)
    numPoly = int(packing.numPolygons)
    stop = numPoly if container is None else int(container)
    centroids, radii = polygonCentroidsRadii(packing)

    energy = 0.0
    force = np.zeros_like(r)

    def loopOf(polygon, shift = 0.0):
        a, b = int(starts[polygon]), int(starts[polygon + 1])
        return a, b, r[a:b] + shift

    def interact(pointSlice, loopIndex, shift, sign, k):
        """The boundary of one polygon against the loop of another; ``sign = +1`` repels on overlap
        (polygon-polygon), ``sign = -1`` penalizes being outside (a container)."""
        nonlocal energy
        pa, pb = pointSlice
        la, lb, loop = loopOf(loopIndex, shift)
        # ONLY THE LOOP MOVES. Shifting the boundary as well is a rigid translation of the pair, which
        # cancels -- and silently switched periodicity off, because the cull still selected pairs by
        # their minimum image while the evaluation saw them unwrapped. A pair overlapping only across
        # the wrap then measured exactly zero against 8.2e-05 of real contact.
        boundary = r[pa:pb]
        if loopIndex in trees and sign > 0.0:
            e, boundaryForces, loopForces = convexDifference.treeEdgeLoopEnergyForce(
                boundary, loop, trees[loopIndex], epsilon, k, order = 8)
        else:
            e, boundaryForces, loopForces = edgeLoopEnergyForce(
                boundary, loop, epsilon, k, adhesionWork if sign > 0.0 else 0.0, adhesionRange,
                quadratureOrder, confine = sign < 0.0)
        energy += e
        force[pa:pb] += boundaryForces
        force[la:lb] += loopForces

    # The pair loop is skipped entirely when the device already did it; only the container remains.
    if deviceForce is None:
        for A in range(stop):
            for B in range(A + 1, stop):
                shift = minImageShift(r[int(starts[B])] - r[int(starts[A])], box)
                separation = np.hypot(*(centroids[B] + shift - centroids[A]))
                if separation >= radii[A] + radii[B] + epsilon:
                    continue
                aSlice = (int(starts[A]), int(starts[A + 1]))
                bSlice = (int(starts[B]), int(starts[B + 1]))
                interact(aSlice, B, shift, +1.0, stiffness)
                interact(bSlice, A, -shift, +1.0, stiffness)

    if container is not None:
        # EVERY polygon's boundary against the wall in ONE sweep. The per-polygon form ran the crossing
        # root finder separately for each, which was 93% of a walled force evaluation -- N independent
        # solves of ~50 numpy calls each, on four-element arrays. Batching makes the container O(1)
        # numpy calls instead of O(N); the arithmetic is identical, only the launch count changes.
        wallStart, wallEnd = int(starts[container]), int(starts[container + 1])
        firstParts, secondParts = [], []
        for A in range(stop):
            a, b = int(starts[A]), int(starts[A + 1])
            local = np.arange(a, b)
            firstParts.append(local)
            secondParts.append(a + (local - a + 1) % (b - a))
        if firstParts:
            firstIndex = np.concatenate(firstParts)
            secondIndex = np.concatenate(secondParts)
            wallEnergy, boundaryForces, wallForces = edgeSetAgainstLoop(
                r[firstIndex], r[secondIndex], firstIndex, secondIndex, len(r), r[wallStart:wallEnd],
                epsilon, kContainer, 0.0, adhesionRange, quadratureOrder, confine = True)
            energy += wallEnergy
            force += boundaryForces
            force[wallStart:wallEnd] += wallForces

    if deviceForce is not None:
        return float(energy) + deviceEnergy, force.reshape(-1) + deviceForce
    return float(energy), force.reshape(-1)
