"""EXACT arcs for the contact laws: a boundary of straight segments and circular arcs.

``roundedGeometry`` chords each corner so the existing polygon tiers can eat it. This module is the
replacement that does not: the corner IS an arc, and every distance, crossing and integral is taken
against it. Stages 1 to 8 of that build -- the representation, the exact distance field, the exact
boundary crossings, the nearest-feature partition (``substretches``), the ENERGY (``pairEnergy`` / ``contactEnergy``) and
its exact GRADIENT, the exact overlap AREA with its gradient (``overlapArea``), the packing-level
drivers that wire BOTH tiers into ``Model`` via ``setGeometryType("round", exact = True)``, and the
ANALYTIC gradients that replace the complex step in both.

BOTH GRADIENTS ARE NOW ANALYTIC (``pairEnergyBodyGradient``, ``overlapAreaBodyGradient``). They are
taken in the BODY arrays -- centre, radius, sweep, tail, head -- in a single pass over the partition,
and each body is converted to ``(loop, rho)`` exactly once per force evaluation. The complex-step
versions are kept as ``pairGradient`` and ``areaGradient``, which is what the analytic ones are tested
against: agreement is 1e-16..1e-13, i.e. round-off, not a tolerance.

WHY IT HAD TO BE DONE THIS WAY. Complex-stepping the whole energy costs one evaluation per degree of
freedom, so a pair of squares paid 24 of them. That is fine for a reference and wrong for a device.
Differentiating with respect to the BODY arrays instead makes the cost independent of the degree-of-
freedom count -- one thread per sub-stretch, fixed arithmetic, an atomic scatter -- and confines the
corner map's own derivative to a per-body pass that costs ``3n`` evaluations of a vectorized closed
form. Measured at N = 11: the depth tier went 1.39 s to 0.27 s per force evaluation and the area tier
0.39 s to 0.14 s. WHAT DOMINATES NOW IS ``substretches`` (82% of the depth driver, over half of it
inside ``numpy.roots``), which is the partition solve the ENERGY needs anyway -- so the next thing worth
doing is a closed-form quartic in ``_solveTrig``, not more gradient work.

THE TWO TIERS NEED DIFFERENT GRADIENT ARGUMENTS, which is the one thing here that cannot be guessed.
The energy's integrand vanishes at a crossing, so its moving breakpoints drop out and the FROZEN
partition gives the exact derivative. Green's integrand does NOT vanish there, so the same treatment is
40-60% wrong for the area; that one needs the shape derivative
``dA/dp = int (v . n) ds``, which has no boundary terms at all. Both were measured rather than
assumed.

THE MOVING BREAKPOINTS CONTRIBUTE NOTHING, which is what makes an exact gradient reachable: at a span
endpoint ``d_B = 0`` so the Leibniz boundary term vanishes, and at an interior feature switch ``d_B``
is continuous so adjacent sub-stretches cancel. The derivative is therefore the FROZEN-partition
derivative, and the argmin and root solving that build the partition never have to be differentiated.
Verified: the frozen energy tracks the true energy to 1.7e-14 under perturbation, and the gradient
agrees with a central difference of the TRUE energy (which re-partitions every step) to 1e-9..1e-11
relative over eight pairs.

MEASURED at stage 4: closed forms against a brute-force line integral of the true distance field agree
to 1e-16..1e-11 relative over nine pairs; ``rho -> 0`` reproduces ``polyContact.pairEnergy`` to 5.6e-15
relative, which is the real check since that is a separate implementation of the same law; and the one
quadrature branch converges geometrically -- 8.9e-07 at order 2, 2.8e-16 by order 8, flat at 4e-19
past order 12, so the default 24 is far inside the noise.

EVERY BUG FOUND SO FAR WAS A TANGENCY, and they will keep coming, because this geometry is BUILT from
tangencies -- each corner circle is tangent to both its edges by construction. Four in a row, all in
the same family and all silent (a mislabelled sub-stretch, never an exception):

  * a segment-versus-arc switch whose quadratic discriminant landed at -2.2e-16, discarded by a
    ``discriminant < 0 -> no roots`` test taken BEFORE the tangency test;
  * the same switch reached by two pair equations, coming back 3.8e-08 apart because
    ``sqrt(discriminant)`` near a double root loses half the digits;
  * a tangential switch along an ARC, returned by ``numpy.roots`` as a near-conjugate pair and thrown
    away by an ``|imag| < 1e-7`` filter, though its residual in the original equation was 3.9e-14;
  * line-circle and circle-circle tangencies returning two nearly-equal points instead of one.

The rules that follow: test tangency on the ABSOLUTE discriminant and BEFORE rejecting negatives; keep
a root by whether it SOLVES the equation, never by whether the solver called it real; and merge
breakpoints that agree to about ``sqrt(machine epsilon)``.

WHY IT IS WORTH THE TROUBLE. The chorded arc reverts to FACETED behavior once the penetration drops
below about ``rho / arcSegments^2``, which is precisely the shallow-contact limit that jamming lives
in. So the smoothness the rounding was chosen for -- capping the ``l/d`` stiffness ratio that makes a
sharp system relieve pressure by tilting out of alignment -- is exactly the thing the discretization
takes back where it matters most. Measured, ``arcSegments = 5`` puts that crossover at ``rho/25``.

THE BOUNDARY IS C1 AND HAS NO VERTICES AT ALL. Each corner arc meets its two incident segments
TANGENTIALLY, by construction: the circle is tangent to both edges at the kiss points. That is not a
detail, it is what makes the whole scheme tractable --

  * there is no vertex feature, so the nearest-feature partition has only two kinds of cell, and the
    ``arc versus vertex`` integral (which is elliptic) never arises;
  * ``grad d`` is continuous across every kiss point, so the medial axis is the only place the feature
    switches and the integrand stays smooth on each sub-stretch.

WHICH INTEGRALS ARE ELEMENTARY, worked out before committing to the design. For
``E = 1/2 sum int_{dP cap Q} (k/3) d_Q^3 dl``:

    integrate along | distance to | integrand                      | status
    ----------------|-------------|--------------------------------|------------
    segment         | segment     | (alpha - m s)^3                | elementary (polyContact has it)
    segment         | arc         | (sqrt((s-s0)^2 + h^2) - r)^3   | ELEMENTARY: expands to
                    |             |                                | int R^3, int R^2, int R ds
    arc             | segment     | (A - r cos phi)^3              | ELEMENTARY: polynomial in cos
    arc             | arc         | (sqrt(A + B cos phi) - r)^3    | ELLIPTIC: incomplete E and F,
                    |             |                                | so Gauss-Legendre instead

Only the last needs quadrature, and its integrand is analytic on each sub-stretch, so Gauss-Legendre
converges geometrically -- this is an exactness argument, not a tolerance to be tuned.

THE GPU IS DEFERRED, NOT LOST. ``polyContactCuda`` takes CSR polygons and cannot be handed an arc, so
this path starts numpy-only -- the project's usual order, a straightforward Python reference first and
the kernel after. Everything here is therefore written to PORT: flat arrays indexed by piece, no
per-body Python loops in the hot path, and branch-free min-reduces rather than early exits, which is
the shape the device wants. Sized before starting: the chorded depth tier runs 7.6 ms/evaluation at
N = 11, and this will be far slower until that port happens.

A DEGENERATE ARC IS A SHARP CORNER, not a special case. At ``rho_k = 0`` the arc collapses to a point,
its two neighbouring segments meet there, and the body is exactly the sharp backbone. So the sharp
polygon is the ``rho = 0`` member of this family and the same code covers both ends of the schedule.
"""

# UNVERIFIED(Cam)

import numpy as np

import roundedGeometry as rg


# UNVERIFIED(Cam)
class RoundedBody:
    """A closed C1 boundary: ``n`` circular arcs alternating with ``n`` straight segments.

    Arc ``k`` belongs to backbone vertex ``k`` and runs from its kiss point ``a^-`` to ``a^+``;
    segment ``k`` runs from ``a^+_k`` to ``a^-_{k+1}``. Both arrays are indexed by backbone vertex, so
    a body of ``n`` vertices has exactly ``n`` of each and the boundary is
    ``arc 0, segment 0, arc 1, segment 1, ...`` in counter-clockwise order.

    ``center``, ``radius``, ``start``, ``sweep`` describe the arcs; ``sweep`` is SIGNED, positive at a
    convex corner. ``tail`` and ``head`` are the segment endpoints. A zero-radius arc and a
    zero-length segment are both legal and both mean the obvious degenerate thing."""

    # UNVERIFIED(Cam)
    def __init__(self, center, radius, start, sweep, tail, head):
        # NOT forced to float: the gradient complex-steps this constructor, so a complex dtype has to
        # survive it. ``start`` stays real by construction (see ``bodyFromBackbone``).
        self.center = np.asarray(center).reshape(-1, 2)
        self.radius = np.asarray(radius).reshape(-1)
        self.start = np.asarray(np.real(start), dtype = float).reshape(-1)
        self.sweep = np.asarray(sweep).reshape(-1)
        self.tail = np.asarray(tail).reshape(-1, 2)
        self.head = np.asarray(head).reshape(-1, 2)

    @property
    def count(self):
        return len(self.radius)

    # UNVERIFIED(Cam)
    def sample(self, perArc = 64):
        """A dense counter-clockwise polyline of the true boundary -- for drawing and for tests that
        need an independent construction. NOT what the contact law uses."""
        points = []
        for k in range(self.count):
            if self.radius[k] > 0.0:
                angles = self.start[k] + self.sweep[k] * np.linspace(0.0, 1.0, perArc + 1)
                points.append(self.center[k] + self.radius[k]
                              * np.stack([np.cos(angles), np.sin(angles)], axis = 1))
            else:
                points.append(self.tail[k][None, :])
            points.append(self.head[k][None, :])
        return np.concatenate(points, axis = 0)


# UNVERIFIED(Cam)
def bodyFromBackbone(loop, rho):
    """Build a ``RoundedBody`` from one counter-clockwise backbone loop and its per-vertex radii.

    Uses ``roundedGeometry.cornerFrame``, so the kiss points, centres and sweeps are the same
    quantities the chorded path uses -- the two representations describe the SAME shape and can be
    compared directly, which is how this module is tested."""
    # NOT cast to float: a complex step through this constructor is how the gradient is taken, and
    # `dtype = float` would silently discard the imaginary part and return a gradient of zero.
    loop = np.asarray(loop).reshape(-1, 2)
    rho = np.asarray(rho).reshape(-1)
    count = len(loop)
    previousIndex = np.roll(np.arange(count), 1)
    nextIndex = np.roll(np.arange(count), -1)

    _, center, aMinus, aPlus, psi, _ = rg.cornerFrame(loop[previousIndex], loop, loop[nextIndex], rho)
    # COMBINATORIAL QUANTITIES ARE READ OFF THE REAL PART. Convexity is a sign and degeneracy is a
    # comparison; neither is differentiable, and neither changes under the infinitesimal imaginary
    # perturbation a complex step applies. Taking them from the real part is what lets this whole
    # constructor run on complex input, which the gradient in ``frozenPairEnergy`` depends on.
    realLoop = np.real(loop)
    realRho = np.real(rho)
    sign = rg.convexSign(realLoop, previousIndex, nextIndex)
    degenerate = realRho <= 0.0

    # ``start`` is an ANGLE and needs arctan2, which is not analytic, so it is computed on the real
    # part only. Nothing in the differentiated path reads it: the arc is parametrized there by ROTATING
    # ``a^- - z``, never by an absolute angle.
    toStart = np.real(aMinus - center)
    start = np.arctan2(toStart[:, 1], toStart[:, 0])
    start = np.where(degenerate, 0.0, start)
    sweep = np.where(degenerate, 0.0, sign * psi)
    return RoundedBody(center = np.where(degenerate[:, None], loop, center), radius = rho,
                       start = start, sweep = sweep,
                       tail = aPlus, head = aMinus[nextIndex])


# UNVERIFIED(Cam)
def _segmentDistance(points, tail, head):
    """``(distance, signedSide)`` from every point to every segment. Shapes ``(M, n)``.

    ``signedSide`` is the OUTWARD-normal component, positive outside, and is only meaningful where the
    segment is the nearest feature -- which is the only place the caller reads it."""
    vector = head - tail
    length2 = np.maximum(np.einsum("kc,kc->k", vector, vector), 1e-300)
    delta = points[:, None, :] - tail[None, :, :]
    t = np.clip(np.einsum("mkc,kc->mk", delta, vector) / length2, 0.0, 1.0)
    foot = tail[None, :, :] + t[:, :, None] * vector[None, :, :]
    offset = points[:, None, :] - foot
    distance = np.hypot(offset[:, :, 0], offset[:, :, 1])
    # OUTWARD normal of a counter-clockwise loop is (tangent.y, -tangent.x); see polyContact, whose
    # convention this deliberately matches so the two laws cannot disagree about inside.
    normal = np.stack([vector[:, 1], -vector[:, 0]], axis = 1) / np.sqrt(length2)[:, None]
    side = np.einsum("mkc,kc->mk", points[:, None, :] - tail[None, :, :], normal)
    return distance, side


# UNVERIFIED(Cam)
def _arcDistance(points, body):
    """``(distance, insideCircle)`` from every point to every arc, ``inf`` outside its angular wedge.

    Outside the wedge the nearest point of the arc is an ENDPOINT, and every endpoint is also an
    endpoint of a tangent segment, so the segment branch already covers it exactly. Returning ``inf``
    here is therefore not an approximation -- it removes a duplicate, and it is what keeps the
    arc-versus-vertex case (the elliptic one) from ever arising."""
    delta = points[:, None, :] - body.center[None, :, :]
    reach = np.hypot(delta[:, :, 0], delta[:, :, 1])
    angle = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    # Fraction of the way round the (signed) sweep, wrapped into [0, 1) -- inside the wedge iff <= 1.
    sweep = body.sweep[None, :]
    with np.errstate(divide = "ignore", invalid = "ignore"):
        fraction = np.where(np.abs(sweep) > 1e-300,
                            ((angle - body.start[None, :]) * np.sign(sweep)) % (2.0 * np.pi)
                            / np.abs(sweep), np.inf)
    within = (fraction <= 1.0) & (body.radius[None, :] > 0.0)
    distance = np.where(within, np.abs(reach - body.radius[None, :]), np.inf)
    return distance, reach < body.radius[None, :]


# UNVERIFIED(Cam)
def nearestFeature(points, body):
    """``(distance, inside, kind, index)`` against the exact boundary. ``kind`` 0 = segment, 1 = arc.

    MEMBERSHIP COMES FROM THE NEAREST FEATURE, no ray cast, exactly as in ``polyContact`` -- but the
    rule is simpler here because the boundary is C1 and has no vertices:

        nearest is segment k -> inside iff the point is on the inner side of it
        nearest is arc k     -> inside iff it lies within that corner circle

    The arc clause is right because the corner circle is pushed in from INSIDE, so its centre is
    interior and the material near the arc is the side toward the centre."""
    segment, side = _segmentDistance(points, body.tail, body.head)
    arc, insideCircle = _arcDistance(points, body)

    bestSegment = np.argmin(segment, axis = 1)
    bestArc = np.argmin(arc, axis = 1)
    rows = np.arange(len(points))
    segmentBest = segment[rows, bestSegment]
    arcBest = arc[rows, bestArc]

    useArc = arcBest < segmentBest
    distance = np.where(useArc, arcBest, segmentBest)
    index = np.where(useArc, bestArc, bestSegment)
    kind = useArc.astype(int)
    inside = np.where(useArc, insideCircle[rows, bestArc], side[rows, bestSegment] < 0.0)
    return distance, inside, kind, index


# UNVERIFIED(Cam)
def signedDistance(points, body):
    """Distance to the boundary, POSITIVE INSIDE -- the sign convention the contact law integrates."""
    distance, inside, _, _ = nearestFeature(np.atleast_2d(points), body)
    return np.where(inside, distance, -distance)


# UNVERIFIED(Cam)
def pieceCount(body):
    """Pieces per body: ``n`` arcs then ``n`` segments. Piece ``k`` is arc ``k``; piece ``n + k`` is
    segment ``k``. One flat index over both kinds is what the batched crossing scan wants."""
    return 2 * body.count


# UNVERIFIED(Cam)
def evaluatePiece(body, piece, parameter):
    """Point at ``parameter`` in [0, 1] along a piece. Arcs sweep ``start -> start + sweep``."""
    n = body.count
    parameter = np.asarray(parameter, dtype = float)
    if piece < n:
        angle = body.start[piece] + body.sweep[piece] * parameter
        return body.center[piece] + body.radius[piece] * np.stack(
            [np.cos(angle), np.sin(angle)], axis = -1)
    k = piece - n
    return body.tail[k] + parameter[..., None] * (body.head[k] - body.tail[k])


# UNVERIFIED(Cam)
def _arcParameter(body, k, points, tolerance = 1e-9):
    """Where ``points`` sit along arc ``k`` as a fraction of its sweep, ``nan`` if off the arc.

    Points are assumed to lie ON the circle already -- this only resolves the ANGULAR window, which is
    what a circle-circle or line-circle solve leaves undetermined."""
    if body.radius[k] <= 0.0 or abs(body.sweep[k]) <= 0.0:
        return np.full(len(points), np.nan)
    delta = points - body.center[k]
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    fraction = ((angle - body.start[k]) * np.sign(body.sweep[k])) % (2.0 * np.pi) \
        / abs(body.sweep[k])
    # A crossing exactly at an endpoint belongs to the arc; the wrap puts it at ~1 or ~2pi/|sweep|.
    fraction = np.where(fraction > 1.0 + tolerance,
                        np.where(fraction > (2.0 * np.pi / abs(body.sweep[k])) - tolerance,
                                 0.0, np.nan),
                        fraction)
    return np.clip(fraction, 0.0, 1.0)


# UNVERIFIED(Cam)
def _lineCircle(tail, head, center, radius):
    """Parameters in [0, 1] along the segment where it meets the circle. Up to two, unsorted."""
    vector = head - tail
    offset = tail - center
    a = float(vector @ vector)
    if a <= 1e-300:
        return np.zeros(0)
    b = 2.0 * float(offset @ vector)
    c = float(offset @ offset) - radius * radius
    discriminant = b * b - 4.0 * a * c
    # A TANGENTIAL crossing is ONE point, and must be returned as one. Near a double root
    # sqrt(discriminant) amplifies roundoff to about sqrt(machine epsilon), so the two roots come back
    # ~1e-8 apart: measured, a tangency at t = 0.125 gave 0.12499998097 and 0.12500001903. Kept as
    # two, they bracket a 3.8e-08 span whose midpoint sits exactly on the tangency, where the nearest
    # feature is ambiguous -- which mislabelled the sub-stretch. The relative test is on the
    # discriminant against the scale of the terms that cancelled to produce it.
    if abs(discriminant) <= 1e-13 * max(b * b + 4.0 * abs(a * c), 1e-300):
        t = np.array([-b / (2.0 * a)])
    elif discriminant < 0.0:
        return np.zeros(0)
    else:
        root = np.sqrt(discriminant)
        t = np.array([(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)])
    return t[(t >= -1e-12) & (t <= 1.0 + 1e-12)]


# UNVERIFIED(Cam)
def _circleCircle(centerA, radiusA, centerB, radiusB):
    """The (up to two) intersection points of two circles, or an empty array."""
    delta = centerB - centerA
    separation = float(np.hypot(delta[0], delta[1]))
    slack = 1e-9 * max(radiusA + radiusB, 1e-300)
    if separation <= 1e-300 or separation > radiusA + radiusB + slack or \
            separation < abs(radiusA - radiusB) - slack:
        return np.zeros((0, 2))
    along = (separation ** 2 + radiusA ** 2 - radiusB ** 2) / (2.0 * separation)
    height2 = radiusA ** 2 - along ** 2
    base = centerA + along * delta / separation
    perpendicular = np.array([-delta[1], delta[0]]) / separation
    # Tangent circles meet at ONE point, for the same reason as the line-circle case above.
    if abs(height2) <= 1e-13 * max(radiusA * radiusA + along * along, 1e-300) \
            or height2 < 0.0:
        return base[None, :]
    height = np.sqrt(height2)
    return np.stack([base + height * perpendicular, base - height * perpendicular])


# UNVERIFIED(Cam)
def crossings(bodyA, bodyB, tolerance = 1e-9):
    """Parameters along every piece of ``bodyA`` where its boundary meets ``bodyB``'s.

    Returns a list of sorted arrays, one per piece of A, each including the endpoints 0 and 1 so the
    caller can walk consecutive intervals. Three closed-form solves cover every combination, because
    the boundary has only two kinds of piece:

        segment x segment   the usual two-line solve
        segment x arc       line-circle, then BOTH parameter windows checked
        arc     x arc       circle-circle, then BOTH angular windows checked

    EVERY PAIR IS TESTED, never a prefiltered subset. ``polyContact.march`` records what pruning costs:
    a candidate dropped early can still become the winner later in the interval, and five genuine
    switches went missing across 107 spans of one packing, moving the energy 0.7%. A spurious crossing
    is harmless -- it subdivides an interval whose state does not change -- but a missed one is not.

    ``tolerance`` IS LOOSE BECAUSE OF THE KISS POINTS. Where a segment of B meets its arc they are
    TANGENT, so a crossing near that junction is found twice -- once by the line solve, once by the
    circle solve -- at values that differ only by roundoff. Measured, one such pair came back as
    0.12499998097 and 0.12500001903, and a 1e-12 dedup kept both: the 3.8e-08 interval between them
    survived as its own span, and its midpoint sat exactly ON the junction, where the nearest feature
    is genuinely ambiguous. Merging crossings this close costs nothing -- the interval between them
    cannot carry any integral."""
    n = bodyA.count
    out = []
    for piece in range(pieceCount(bodyA)):
        found = [0.0, 1.0]
        if piece < n:
            k = piece
            if bodyA.radius[k] > 0.0 and abs(bodyA.sweep[k]) > 0.0:
                for j in range(bodyB.count):
                    if bodyB.radius[j] > 0.0 and abs(bodyB.sweep[j]) > 0.0:
                        points = _circleCircle(bodyA.center[k], bodyA.radius[k],
                                               bodyB.center[j], bodyB.radius[j])
                        if len(points):
                            mine = _arcParameter(bodyA, k, points)
                            theirs = _arcParameter(bodyB, j, points)
                            found.extend(mine[np.isfinite(mine) & np.isfinite(theirs)])
                for j in range(bodyB.count):
                    hits = _lineCircle(bodyB.tail[j], bodyB.head[j],
                                       bodyA.center[k], bodyA.radius[k])
                    if len(hits):
                        points = bodyB.tail[j] + hits[:, None] * (bodyB.head[j] - bodyB.tail[j])
                        mine = _arcParameter(bodyA, k, points)
                        found.extend(mine[np.isfinite(mine)])
        else:
            k = piece - n
            tail, head = bodyA.tail[k], bodyA.head[k]
            vector = head - tail
            for j in range(bodyB.count):
                if bodyB.radius[j] > 0.0 and abs(bodyB.sweep[j]) > 0.0:
                    hits = _lineCircle(tail, head, bodyB.center[j], bodyB.radius[j])
                    if len(hits):
                        points = tail + hits[:, None] * vector
                        theirs = _arcParameter(bodyB, j, points)
                        found.extend(hits[np.isfinite(theirs)])
            for j in range(bodyB.count):
                other = bodyB.head[j] - bodyB.tail[j]
                denominator = vector[0] * other[1] - vector[1] * other[0]
                if abs(denominator) < 1e-300:
                    continue
                gap = bodyB.tail[j] - tail
                t = (gap[0] * other[1] - gap[1] * other[0]) / denominator
                u = (gap[0] * vector[1] - gap[1] * vector[0]) / denominator
                if -1e-12 <= t <= 1.0 + 1e-12 and -1e-12 <= u <= 1.0 + 1e-12:
                    found.append(t)
        values = np.clip(np.unique(np.asarray(found, dtype = float)), 0.0, 1.0)
        keep = np.concatenate([[True], np.diff(values) > tolerance])
        out.append(values[keep])
    return out


# UNVERIFIED(Cam)
def pieceLength(body, piece):
    """Arc length of a piece. Zero for a degenerate corner (rho = 0) or a vanished straight run."""
    n = body.count
    if piece < n:
        return float(body.radius[piece] * abs(body.sweep[piece]))
    k = piece - n
    return float(np.hypot(*(body.head[k] - body.tail[k])))


# UNVERIFIED(Cam)
def spans(bodyA, bodyB):
    """``(piece, low, high)`` for every interval of ``dA`` lying INSIDE ``bodyB``.

    The analogue of ``polyContact.spans``. Membership is decided at each interval's MIDPOINT rather
    than at a crossing, for the reason ``_substretches`` records about winners: a state read exactly at
    a transition is not reliably on either side of it."""
    pieces, lows, highs = [], [], []
    for piece, cuts in enumerate(crossings(bodyA, bodyB)):
        # A ZERO-LENGTH PIECE IS SKIPPED HERE, not later. A rho = 0 corner still has an arc entry, and
        # its "arc" is a single point: it can sit inside B and open a span that carries no integral but
        # does divide by a zero sweep when the angular candidates are mapped back to the parameter.
        if len(cuts) < 2 or pieceLength(bodyA, piece) <= 1e-15:
            continue
        low, high = cuts[:-1], cuts[1:]
        midpoint = evaluatePiece(bodyA, piece, 0.5 * (low + high))
        _, inside, _, _ = nearestFeature(np.atleast_2d(midpoint), bodyB)
        for a, b, isIn in zip(low, high, inside):
            if isIn and b - a > 1e-14:
                pieces.append(piece); lows.append(a); highs.append(b)
    return np.asarray(pieces, dtype = int), np.asarray(lows), np.asarray(highs)


# UNVERIFIED(Cam)
def _quadraticCandidates(a, b, c):
    """Real roots of ``a x^2 + b x + c``, PLUS the real part of a complex pair.

    Not the same contract as ``_realQuadratic``, and deliberately. This feeds the switch solver, whose
    standing rule is that a spurious breakpoint costs nothing and a missed one is a mislabelled
    sub-stretch. At a TANGENCY -- which this geometry is built from -- the discriminant lands a hair
    below zero and ``numpy.roots`` used to return a near-conjugate pair whose real part was the switch;
    the caller kept it and filtered on the RESIDUAL. Returning ``-b/2a`` when the discriminant is
    negative reproduces that exactly, and a genuinely complex root fails the residual test as before."""
    if abs(a) < 1e-300:
        return [] if abs(b) < 1e-300 else [-c / b]
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return [-b / (2.0 * a)]
    root = np.sqrt(discriminant)
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


# UNVERIFIED(Cam)
def _realCubicRoots(a2, a1, a0, candidates = False):
    """Real roots of the monic cubic ``t^3 + a2 t^2 + a1 t + a0``, closed form.

    Depress with ``t = y - a2/3`` and split on the discriminant: three real roots go through the
    trigonometric form (which is the numerically stable one in that branch -- Cardano there needs the
    cube root of a complex number and loses most of its digits), one real root through Cardano.

    ``candidates`` also returns the real part of the complex pair, for the same reason
    ``_quadraticCandidates`` does. It is OFF where the resolvent is solved, because there a genuine
    real root is wanted and a complex pair's real part is not one."""
    shift = a2 / 3.0
    p = a1 - a2 * a2 / 3.0
    q = 2.0 * a2 ** 3 / 27.0 - a2 * a1 / 3.0 + a0
    if abs(p) < 1e-300:
        return [np.cbrt(-q) - shift]
    discriminant = -4.0 * p ** 3 - 27.0 * q * q
    if discriminant > 0.0:
        # Three real roots. cos(3 theta) = 3q / (p m), m = 2 sqrt(-p/3), and -p is positive here.
        magnitude = 2.0 * np.sqrt(-p / 3.0)
        argument = np.clip(3.0 * q / (p * magnitude), -1.0, 1.0)
        angle = np.arccos(argument) / 3.0
        return [magnitude * np.cos(angle - 2.0 * np.pi * k / 3.0) - shift for k in (0, 1, 2)]
    inner = q * q / 4.0 + p ** 3 / 27.0
    root = np.sqrt(max(inner, 0.0))
    single = np.cbrt(-0.5 * q + root) + np.cbrt(-0.5 * q - root)
    if not candidates:
        return [single - shift]
    # The depressed roots sum to zero, so the conjugate pair sits at -single/2.
    return [single - shift, -0.5 * single - shift]


# UNVERIFIED(Cam)
def _realQuarticRoots(coefficients, polishSteps = 2):
    """Real roots of a polynomial of degree at most four, in CLOSED FORM -- no eigensolver.

    ``numpy.roots`` builds a companion matrix and calls LAPACK, which is both the single largest cost
    in the numpy path (12168 calls, 2.7 s of a 20-step minimization) and impossible on a device. This
    is the replacement, and it is written so the CUDA port can transcribe it directly: no allocation,
    no branching on anything but the degree, no library calls.

    Ferrari, in the factored form ``y^4 + p y^2 + q y + r = (y^2 + a y + b)(y^2 - a y + c)``. Matching
    coefficients gives ``a^2 = u`` where ``u`` solves the resolvent cubic
    ``u^3 + 2p u^2 + (p^2 - 4r) u - q^2 = 0``. Its constant term is ``-q^2 <= 0``, so the cubic is
    non-positive at the origin and rises without bound: a real root ``u >= 0`` ALWAYS exists, and the
    largest one is taken because a larger ``a`` divides ``q/a`` more safely.

    EVERY ROOT IS THEN POLISHED by Newton on the ORIGINAL quartic. Ferrari loses digits through the
    resolvent whenever the quartic is near-degenerate, and near-degenerate is the normal case here --
    this geometry is built from tangencies, so double roots are routine. Polishing costs a handful of
    flops and puts the residual back where the caller's filter expects it."""
    coefficients = np.asarray(coefficients, dtype = float)
    scale = np.max(np.abs(coefficients))
    if scale <= 0.0:
        return []
    coefficients = coefficients / scale
    # Drop leading zeros: the degree is data, and a quartic whose leading coefficient has vanished is a
    # cubic rather than a quartic with a root at infinity.
    start = 0
    while start < len(coefficients) - 1 and abs(coefficients[start]) < 1e-14:
        start += 1
    c = coefficients[start:]
    degree = len(c) - 1

    if degree <= 0:
        return []
    if degree == 1:
        return [-c[1] / c[0]]
    if degree == 2:
        return _quadraticCandidates(c[0], c[1], c[2])
    if degree == 3:
        roots = _realCubicRoots(c[1] / c[0], c[2] / c[0], c[3] / c[0], candidates = True)
    else:
        b, cc, d, e = c[1] / c[0], c[2] / c[0], c[3] / c[0], c[4] / c[0]
        shift = b / 4.0
        p = cc - 6.0 * shift * shift
        q = d - 2.0 * cc * shift + 8.0 * shift ** 3
        r = e - d * shift + cc * shift * shift - 3.0 * shift ** 4
        if abs(q) < 1e-14:
            # Biquadratic: y^4 + p y^2 + r, so y^2 solves a quadratic. Ferrari would divide by a -> 0.
            roots = []
            for square in _quadraticCandidates(1.0, p, r):
                root = np.sqrt(max(square, 0.0))
                roots.extend([root - shift, -root - shift])
        else:
            positive = [u for u in _realCubicRoots(2.0 * p, p * p - 4.0 * r, -q * q) if u > 0.0]
            if not positive:
                return []
            alpha = np.sqrt(max(positive))
            beta = 0.5 * (p + alpha * alpha - q / alpha)
            gamma = 0.5 * (p + alpha * alpha + q / alpha)
            roots = [y - shift for y in
                     _quadraticCandidates(1.0, alpha, beta) + _quadraticCandidates(1.0, -alpha, gamma)]

    # Newton on the original, by Horner in both value and derivative -- but ONLY WHERE IT HELPS.
    #
    # A DOUBLE ROOT MAKES PLAIN NEWTON DESTRUCTIVE, and double roots are the normal case here because
    # this geometry is built from tangencies. At one, p and p' vanish together and the step is 0/0:
    # measured on a switch quartic with two double roots, unguarded polishing moved -0.25287 to
    # -0.33893 and 2.35222 to -82.94, losing both -- and the sub-stretches that ended there with them.
    # Accepting a step only when it REDUCES the residual makes the polish monotone, so it can sharpen a
    # simple root and can never wreck a degenerate one.
    def horner(x):
        value, slope = c[0], 0.0
        for coefficient in c[1:]:
            slope = slope * x + value
            value = value * x + coefficient
        return value, slope

    polished = []
    for x in roots:
        if not np.isfinite(x):
            continue
        best = abs(horner(x)[0])
        for _ in range(polishSteps):
            value, slope = horner(x)
            if abs(slope) < 1e-13 * max(abs(value), 1e-300) or slope == 0.0:
                break
            candidate = x - value / slope
            if not np.isfinite(candidate):
                break
            residual = abs(horner(candidate)[0])
            if residual >= best:
                break
            x, best = candidate, residual
        polished.append(float(x))
    return polished


# UNVERIFIED(Cam)
def _solveTrig(coefficients, tolerance = 1e-11):
    """Real roots in ``(-pi, pi]`` of ``E0 + E1c cos p + E1s sin p + E2c cos 2p + E2s sin 2p = 0``.

    Every feature-switch equation in this module reduces to this one shape, so it is solved in one
    place. ``u = tan(p/2)`` turns it into a QUARTIC, whose roots come from ``numpy.roots``; the
    substitution cannot represent ``p = pi``, so that value is always offered as an extra candidate.

    SPURIOUS ROOTS ARE FINE, MISSED ONES ARE NOT. A root that is not really a switch just subdivides an
    interval whose nearest feature does not change, which alters nothing downstream -- see the note on
    ``crossings``. So the squaring steps that produce these equations are never un-done by checking the
    sign branch; the caller re-identifies the winner at midpoints regardless."""
    e0, e1c, e1s, e2c, e2s = coefficients
    # (1 + u^2)^2 * [ ... ], with cos p = (1-u^2)/(1+u^2), sin p = 2u/(1+u^2),
    # cos 2p = (1 - 6u^2 + u^4)/(1+u^2)^2, sin 2p = 4u(1-u^2)/(1+u^2)^2.
    quartic = np.array([
        e0 - e1c + e2c,                       # u^4
        2.0 * e1s - 4.0 * e2s,                # u^3
        2.0 * e0 - 6.0 * e2c,                 # u^2
        2.0 * e1s + 4.0 * e2s,                # u^1
        e0 + e1c + e2c,                       # u^0
    ])
    roots = []
    if np.max(np.abs(quartic)) > tolerance:
        scaled = quartic / np.max(np.abs(quartic))
        nonzero = np.nonzero(np.abs(scaled) > 1e-14)[0]
        if len(nonzero):
            solved = np.asarray(_realQuarticRoots(scaled[nonzero[0]:]), dtype = float)
            if not solved.size:
                roots.append(np.pi)
                return np.asarray(roots, dtype = float)
            # THE REAL PART OF A COMPLEX PAIR IS A CANDIDATE, which is why the solver returns them --
            # see ``_quadraticCandidates``. At a TANGENTIAL switch the quartic has a double root whose
            # discriminant lands a hair below zero; measured, discarding it mislabelled the
            # sub-stretch spanning a switch whose residual in the original equation was 3.9e-14. A
            # genuinely complex root contributes a spurious breakpoint, which by this module's rule
            # costs nothing: it subdivides an interval whose nearest feature does not change.
            candidate = 2.0 * np.arctan(solved)
            # Keep a root by whether it SOLVES the equation, not by whether the solver called it real.
            # This is the safe direction of the trade -- the test is on the original trigonometric
            # form, not on a proxy for it.
            residual = np.abs(e0 + e1c * np.cos(candidate) + e1s * np.sin(candidate)
                              + e2c * np.cos(2.0 * candidate) + e2s * np.sin(2.0 * candidate))
            scale = max(abs(e0) + abs(e1c) + abs(e1s) + abs(e2c) + abs(e2s), 1e-300)
            roots.extend(candidate[residual <= 1e-6 * scale])
    roots.append(np.pi)
    return np.asarray(roots, dtype = float)


# UNVERIFIED(Cam)
def _alongSegmentCoefficients(body, tail, vector):
    """Distance to every feature along ``tail + t vector``, INSIDE the body.

    Returns ``(linearA, linearM, arcRho, arcA, arcB, arcC)``: a segment contributes
    ``d = A - M t`` and an arc ``d = rho - sqrt(a t^2 + b t + c)``. Signed distances, positive inside,
    which is what keeps the degrees down -- see the module note."""
    edge = body.head - body.tail
    length = np.maximum(np.hypot(edge[:, 0], edge[:, 1]), 1e-300)
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis = 1) / length[:, None]
    linearA = -np.einsum("jc,c->j", normal, tail) + np.einsum("jc,jc->j", normal, body.tail)
    linearM = np.einsum("jc,c->j", normal, vector)

    offset = tail[None, :] - body.center
    arcA = np.full(body.count, float(vector @ vector))
    arcB = 2.0 * np.einsum("jc,c->j", offset, vector)
    arcC = np.einsum("jc,jc->j", offset, offset)
    return linearA, linearM, body.radius, arcA, arcB, arcC


# WHY THE SWITCH POSITIONS ARE ONLY GOOD TO sqrt(MACHINE EPSILON), AND WHY THAT IS NOT FIXABLE HERE.
# Tried and reverted, 2026-08-20 -- do not re-attempt without reading this.
#
# Every switch involving an arc is solved by SQUARING away a square root, so a transversal crossing
# becomes a DOUBLE root of the polynomial and the solve, however exact, places it only to about 1e-08.
# Measured against the CUDA port, whose partitions are otherwise structurally identical (52 intervals
# against 52, same features, same labels), two breakpoints sat 8.5e-09 and 1.7e-08 apart.
#
# THAT SHOWS UP IN THE GRADIENT AND NOT IN THE ENERGY. Moving a shared breakpoint m changes the energy
# by int (f_F - f_G), and d_B is CONTINUOUS across a feature switch, so the jump is zero and the energy
# is first-order insensitive -- measured 1e-15. The frozen gradient changes by int (d_p f_F - d_p f_G),
# and THAT jump is not zero, because the two features have different geometry and so different
# derivatives where their values agree. The error is linear in the displacement: 1e-08 of breakpoint
# bought 5e-07 of gradient.
#
# The obvious fix -- Newton on the ORIGINAL, unsquared equation, which should cross transversally --
# WAS IMPLEMENTED AND DOES NOT WORK, for a reason worth keeping. It sharpened 125 of 237 roots and
# moved no aggregate metric, because the switches that matter are between a segment of B and B's OWN
# TANGENT ARC. Those are adjacent features meeting at a kiss point, and the boundary is C1 there, so
# their distance functions agree to 1.7e-16 over a finite stretch rather than crossing: measured
# d(d_seg - d_arc)/dp = -8e-11, which is round-off. There is no crossing to sharpen and Newton is flat.
# The polish cost 59% more time (16.6 ms against 10.4 ms per substretches call) for nothing, so it was
# removed.
#
# THE C1 TANGENCY THAT MAKES THIS SCHEME TRACTABLE IS THE SAME PROPERTY THAT MAKES THE ASSIGNMENT
# AMBIGUOUS. No vertex feature and no elliptic integral come from the arc meeting its segments
# tangentially; so does a medial ray along which two features are indistinguishable. Expect ~1e-07
# relative agreement between any two implementations of this gradient, and judge a port against that
# floor rather than against the energy's 1e-15.


# UNVERIFIED(Cam)
def _switchesAlongSegment(body, tail, vector):
    """Candidate parameters where the nearest feature of ``body`` changes along a straight chord.

    ALL PAIRS, closed form throughout:

        segment vs segment   linear      (A1 - M1 t = A2 - M2 t)
        segment vs arc       quadratic   (square once)
        arc vs arc           quadratic   -- both distance-quadratics share the leading coefficient
                                            |v|^2, so their difference is LINEAR and one squaring is
                                            enough. This is why signed distances were used."""
    linearA, linearM, rho, arcA, arcB, arcC = _alongSegmentCoefficients(body, tail, vector)
    count = body.count
    found = []

    for i in range(count):
        for j in range(i + 1, count):
            slope = linearM[i] - linearM[j]
            if abs(slope) > 1e-300:
                found.append((linearA[i] - linearA[j]) / slope)

    for i in range(count):
        for j in range(count):
            if rho[j] <= 0.0:
                continue
            # A_i - M_i t = rho_j - sqrt(Q_j)  ->  sqrt(Q_j) = (rho_j - A_i) + M_i t
            g0, g1 = rho[j] - linearA[i], linearM[i]
            found.extend(_realQuadratic(arcA[j] - g1 * g1, arcB[j] - 2.0 * g0 * g1,
                                        arcC[j] - g0 * g0))

    for i in range(count):
        for j in range(i + 1, count):
            if rho[i] <= 0.0 or rho[j] <= 0.0:
                continue
            delta = rho[i] - rho[j]
            # sqrt(Q_i) - sqrt(Q_j) = delta.  Q_i - Q_j is LINEAR (same leading coefficient).
            db, dc = arcB[i] - arcB[j], arcC[i] - arcC[j]
            if abs(delta) < 1e-14:
                if abs(db) > 1e-300:
                    found.append(-dc / db)
                continue
            # sqrt(Q_j) = (Q_i - Q_j - delta^2) / (2 delta)
            h1, h0 = db / (2.0 * delta), (dc - delta * delta) / (2.0 * delta)
            found.extend(_realQuadratic(arcA[j] - h1 * h1, arcB[j] - 2.0 * h0 * h1,
                                        arcC[j] - h0 * h0))
    return np.asarray(found, dtype = float)


# UNVERIFIED(Cam)
def _realQuadratic(a, b, c):
    """Real roots of ``a x^2 + b x + c``, degrading to the linear, tangential and empty cases.

    A NEAR-DOUBLE ROOT IS RETURNED ONCE. Where a segment of B meets its arc they are TANGENT, so the
    two distance functions are tangent there as well and the switch equation has a genuine double
    root. ``sqrt(discriminant)`` then amplifies roundoff to about sqrt(machine epsilon): measured, a
    switch at t = 0.125 came back as 0.125 -/+ 1.9029e-08. Kept as two roots they bracket a 3.8e-08
    sub-stretch whose MIDPOINT lands exactly on the tangency, where the nearest feature is genuinely
    ambiguous -- which is what mislabelled it. One tangency is one breakpoint."""
    if abs(a) < 1e-300:
        return [] if abs(b) < 1e-300 else [-c / b]
    discriminant = b * b - 4.0 * a * c
    # THE TANGENCY TEST COMES FIRST, and on the ABSOLUTE value. At a true tangency the discriminant is
    # zero in exact arithmetic and lands on either side of it in floating point: measured, a real
    # segment-versus-arc switch at t = 0.794508 produced -2.22e-16, so a `< 0 -> no roots` test taken
    # first discarded the switch entirely and the sub-stretch spanning it was labelled with the wrong
    # feature. Rejecting negatives before testing for tangency loses exactly the roots that matter.
    if abs(discriminant) <= 1e-13 * max(b * b + 4.0 * abs(a * c), 1e-300):
        return [-b / (2.0 * a)]
    if discriminant < 0.0:
        return []
    root = np.sqrt(discriminant)
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


# UNVERIFIED(Cam)
def _switchesAlongArc(body, center, radius):
    """Candidate ANGLES where the nearest feature of ``body`` changes along a circle of radius
    ``radius`` about ``center``.

    Along an arc every distance becomes a first-harmonic trigonometric polynomial --
    ``d_segment = A + Bx cos p + By sin p`` and ``d_arc = rho - sqrt(P + Cx cos p + Cy sin p)`` -- so
    each pairwise equation is degree at most 2 in ``p`` after one squaring, and ``_solveTrig`` takes
    it from there."""
    segA, segX, segY, arcP, arcX, arcY = _alongArcCoefficients(body, center, radius)
    rho = body.radius
    count = body.count
    found = []

    for i in range(count):
        for j in range(i + 1, count):
            found.extend(_solveTrig((segA[i] - segA[j], segX[i] - segX[j],
                                     segY[i] - segY[j], 0.0, 0.0)))

    for i in range(count):
        for j in range(count):
            if rho[j] <= 0.0:
                continue
            # sqrt(P + Cx c + Cy s) = (rho - A) - Bx c - By s = g0 + gx c + gy s
            g0, gx, gy = rho[j] - segA[i], -segX[i], -segY[i]
            found.extend(_solveTrig(_squareFirstHarmonic(g0, gx, gy,
                                                         arcP[j], arcX[j], arcY[j])))

    for i in range(count):
        for j in range(i + 1, count):
            if rho[i] <= 0.0 or rho[j] <= 0.0:
                continue
            delta = rho[i] - rho[j]
            dP, dX, dY = arcP[i] - arcP[j], arcX[i] - arcX[j], arcY[i] - arcY[j]
            if abs(delta) < 1e-14:
                found.extend(_solveTrig((dP, dX, dY, 0.0, 0.0)))
                continue
            half = 1.0 / (2.0 * delta)
            found.extend(_solveTrig(_squareFirstHarmonic(
                (dP - delta * delta) * half, dX * half, dY * half,
                arcP[j], arcX[j], arcY[j])))
    return np.asarray(found, dtype = float)


# UNVERIFIED(Cam)
def _squareFirstHarmonic(g0, gx, gy, p, cx, cy):
    """Coefficients of ``(g0 + gx cos + gy sin)^2 - (P + Cx cos + Cy sin)`` in ``_solveTrig``'s basis.

    The squaring is where the second harmonic appears: ``cos^2 = (1 + cos 2p)/2``,
    ``sin^2 = (1 - cos 2p)/2``, ``cos sin = sin 2p / 2``."""
    return (g0 * g0 + 0.5 * (gx * gx + gy * gy) - p,
            2.0 * g0 * gx - cx,
            2.0 * g0 * gy - cy,
            0.5 * (gx * gx - gy * gy),
            gx * gy)


# UNVERIFIED(Cam)
def _alongArcCoefficients(body, center, radius):
    """Distance to every feature of ``body`` along a circle of ``radius`` about ``center``.

    ``d_segment = A + Bx cos p + By sin p`` and ``d_arc = rho - sqrt(P + Cx cos p + Cy sin p)``, both
    signed positive inside. Shared by the switch solver and the integrator ON PURPOSE: if the two ever
    disagreed about the integrand, the partition would be exact for a function nobody integrates."""
    edge = body.head - body.tail
    length = np.maximum(np.hypot(edge[:, 0], edge[:, 1]), 1e-300)
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis = 1) / length[:, None]
    segA = np.einsum("jc,jc->j", normal, body.tail - center[None, :])
    segX, segY = -radius * normal[:, 0], -radius * normal[:, 1]
    toCenter = center[None, :] - body.center
    arcP = np.einsum("jc,jc->j", toCenter, toCenter) + radius * radius
    arcX, arcY = 2.0 * radius * toCenter[:, 0], 2.0 * radius * toCenter[:, 1]
    return segA, segX, segY, arcP, arcX, arcY


# UNVERIFIED(Cam)
def _cubicLineIntegral(a, m, lo, hi):
    """``int (a - m t)^3 dt`` over ``[lo, hi]`` -- the segment-along, segment-to case.

    EXPANDED AS A POLYNOMIAL, not as ``-((a - m hi)^4 - (a - m lo)^4) / (4 m)``. The closed form
    divides by ``m``, which is ZERO whenever the two segments are PARALLEL -- routine between
    axis-aligned squares. Guarding that with ``if abs(m) < tiny`` looks safe and is not: under the
    complex step the gradient takes, ``m = 0`` becomes ``0 + 1e-30i``, whose modulus clears the guard,
    so the real evaluation took the safe branch while the differentiated one divided by ~0. Measured,
    two sub-stretches returned a derivative of exactly 0.0 against a true 8.4e-03, and only
    axis-aligned configurations were affected -- the random ones all passed.

    The expansion has no division and no branch, so it is analytic everywhere and the two paths cannot
    diverge."""
    return (a ** 3 * (hi - lo)
            - 1.5 * a * a * m * (hi ** 2 - lo ** 2)
            + a * m * m * (hi ** 3 - lo ** 3)
            - 0.25 * m ** 3 * (hi ** 4 - lo ** 4))


# UNVERIFIED(Cam)
def _rootAntiderivatives(a, b, c, t):
    """``(int R dt, int R^3 dt)`` at ``t`` for ``R = sqrt(a t^2 + b t + c)``, ``a > 0``.

    Completing the square gives ``R = sqrt(a) sqrt(u^2 + h^2)`` with ``u = t + b/(2a)``. The cubic
    antiderivative is the same ``w (2w^2 + 5h^2) sqrt(w^2+h^2)/8 + 3 h^4 asinh(w/h)/8`` that
    ``polyContact._vertexAntiderivative`` uses -- the vertex branch there is this integral."""
    # COMPLEX-SAFE THROUGHOUT: no hypot, no abs, no comparison on a possibly-complex value. Each of
    # those either raises or silently takes a modulus, which is not analytic and would return a wrong
    # derivative rather than an error.
    root = np.sqrt(a)
    u = t + b / (2.0 * a)
    h2 = (4.0 * a * c - b * b) / (4.0 * a * a)
    if np.real(h2) <= 0.0:
        # The degenerate case R = sqrt(a) |u|, where the arc centre lies ON the chord. The sign is
        # combinatorial, so it is read off the real part.
        side = np.sign(np.real(u))
        return root * 0.5 * side * u * u, root ** 3 * 0.25 * side * u ** 4
    h = np.sqrt(h2)
    s = np.sqrt(u * u + h2)
    first = 0.5 * u * s + 0.5 * h2 * np.arcsinh(u / h)
    third = u * (2.0 * u * u + 5.0 * h2) * s / 8.0 + 3.0 * h2 * h2 * np.arcsinh(u / h) / 8.0
    return root * first, root ** 3 * third


# UNVERIFIED(Cam)
def _cubicTrigIntegral(a, c, phi):
    """Antiderivative of ``(a + c cos psi)^3`` in ``psi``, evaluated at ``phi``.

    ``a^3 psi + 3a^2 c sin psi + 3 a c^2 (psi/2 + sin 2psi/4) + c^3 (sin psi - sin^3 psi / 3)``."""
    return (a ** 3 * phi + 3.0 * a * a * c * np.sin(phi)
            + 3.0 * a * c * c * (0.5 * phi + 0.25 * np.sin(2.0 * phi))
            + c ** 3 * (np.sin(phi) - np.sin(phi) ** 3 / 3.0))


# UNVERIFIED(Cam)
def substretches(bodyA, bodyB, tolerance = 1e-7):
    """Every nearest-feature-constant sub-stretch of ``dA`` inside ``bodyB``.

    Returns ``(piece, low, high, kind, feature, spanId)`` -- the arc/segment analogue of
    ``polyContact._substretches``, and deliberately the same shape, including ``spanId``, because the
    gradient's measure term is per SPAN rather than per sub-stretch.

    THE WINNER IS RE-IDENTIFIED AT EACH MIDPOINT, never inferred from which candidate produced the
    breakpoint. ``polyContact`` records the cost of trusting a marcher's winner list: two candidates
    crossing shallowly were mis-assigned, which inflated the energy EIGHTFOLD and surfaced only as a
    finite-difference failure."""
    n = bodyA.count
    pieces, lows, highs, spanIds = [], [], [], []
    spanPiece, spanLow, spanHigh = spans(bodyA, bodyB)

    for spanId, (piece, low, high) in enumerate(zip(spanPiece, spanLow, spanHigh)):
        if piece < n:
            angles = _switchesAlongArc(bodyB, bodyA.center[piece], bodyA.radius[piece])
            start, sweep = bodyA.start[piece], bodyA.sweep[piece]
            # Every angle has period 2pi; offer both wrappings so a switch just outside the principal
            # branch is not lost.
            candidates = []
            for shift in (-2.0 * np.pi, 0.0, 2.0 * np.pi):
                candidates.append((angles + shift - start) / sweep)
            cuts = np.concatenate(candidates)
        else:
            k = piece - n
            cuts = _switchesAlongSegment(bodyB, bodyA.tail[k], bodyA.head[k] - bodyA.tail[k])
        # NEAR-COINCIDENT ROOTS ARE MERGED, and the tolerance is loose ON PURPOSE. The same switch is
        # reached by several pair equations, each of which squares once or twice and so loses about
        # half its digits: measured, one switch at t = 0.125 came back as 0.12499998097 and
        # 0.12500001903 from two different pairs. A 1e-13 dedup kept both, leaving a 3.8e-08 sliver
        # straddling the true switch whose MIDPOINT sat exactly on it -- where the winner is genuinely
        # ambiguous, so the sub-stretch was labelled with the wrong feature.
        #
        # Merging two switches that really are this close is harmless: the interval between them is
        # narrower than the tolerance and contributes nothing to the integral, and the midpoint rule
        # then picks one of the two features, which is right everywhere except inside that sliver.
        inside = cuts[(cuts > low + tolerance) & (cuts < high - tolerance)]
        edges = np.unique(np.concatenate([[low], inside, [high]]))
        edges = edges[np.concatenate([[True], np.diff(edges) > tolerance])]
        for a, b in zip(edges[:-1], edges[1:]):
            pieces.append(piece); lows.append(a); highs.append(b); spanIds.append(spanId)

    if not pieces:
        empty = np.zeros(0, dtype = int)
        return empty, np.zeros(0), np.zeros(0), empty, empty, empty

    pieces = np.asarray(pieces, dtype = int)
    lows, highs = np.asarray(lows), np.asarray(highs)
    midpoint = np.stack([evaluatePiece(bodyA, p, 0.5 * (l + h))
                         for p, l, h in zip(pieces, lows, highs)])
    _, _, kind, feature = nearestFeature(midpoint, bodyB)
    return pieces, lows, highs, kind, feature, np.asarray(spanIds, dtype = int)


# UNVERIFIED(Cam)
def pairEnergy(bodyA, bodyB, stiffness = 1.0, quadratureOrder = 24):
    """``int_{dA cap B} (k/3) d_B^3 dl`` for the exact segment-and-arc boundary.

    Three of the four cases are closed form; only ARC-ALONG against ARC-TO needs quadrature, because
    ``int sqrt(A + B cos p) dp`` is an incomplete elliptic integral. Its integrand is analytic on a
    sub-stretch -- that is exactly what ``substretches`` guarantees by cutting at every feature switch
    -- so Gauss-Legendre converges geometrically and ``quadratureOrder`` is a convergence knob rather
    than an accuracy compromise. ``pairEnergyOrders`` demonstrates the convergence.

    The measure is ``dl``: ``|v| dt`` along a segment and ``r |sweep| ds`` along an arc. The closed-form
    arc case integrates in the ANGLE, so its Jacobian is ``r sign(sweep)`` rather than ``r |sweep|``."""
    piece, low, high, kind, feature, _ = substretches(bodyA, bodyB)
    if not len(piece):
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(int(quadratureOrder))
    n = bodyA.count
    total = 0.0

    for p, lo, hi, k, f in zip(piece, low, high, kind, feature):
        if hi - lo <= 0.0:
            continue
        if p >= n:
            j = p - n
            tail = bodyA.tail[j]
            vector = bodyA.head[j] - bodyA.tail[j]
            speed = float(np.hypot(vector[0], vector[1]))
            if speed <= 0.0:
                continue
            linearA, linearM, rho, arcA, arcB, arcC = _alongSegmentCoefficients(bodyB, tail, vector)
            if k == 0:
                value = _cubicLineIntegral(linearA[f], linearM[f], lo, hi)
            else:
                r = rho[f]
                firstHi, thirdHi = _rootAntiderivatives(arcA[f], arcB[f], arcC[f], hi)
                firstLo, thirdLo = _rootAntiderivatives(arcA[f], arcB[f], arcC[f], lo)
                second = (arcA[f] * (hi ** 3 - lo ** 3) / 3.0
                          + arcB[f] * (hi ** 2 - lo ** 2) / 2.0 + arcC[f] * (hi - lo))
                # (r - R)^3 = r^3 - 3 r^2 R + 3 r R^2 - R^3
                value = (r ** 3 * (hi - lo) - 3.0 * r * r * (firstHi - firstLo)
                         + 3.0 * r * second - (thirdHi - thirdLo))
            total += speed * value
        else:
            radius, sweep = bodyA.radius[p], bodyA.sweep[p]
            if radius <= 0.0 or sweep == 0.0:
                continue
            segA, segX, segY, arcP, arcX, arcY = _alongArcCoefficients(
                bodyB, bodyA.center[p], radius)
            start = bodyA.start[p]
            if k == 0:
                amplitude = float(np.hypot(segX[f], segY[f]))
                gamma = float(np.arctan2(segY[f], segX[f]))
                value = (_cubicTrigIntegral(segA[f], amplitude, start + sweep * hi - gamma)
                         - _cubicTrigIntegral(segA[f], amplitude, start + sweep * lo - gamma))
                total += radius * np.sign(sweep) * value
            else:
                s = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
                phi = start + sweep * s
                inner = arcP[f] + arcX[f] * np.cos(phi) + arcY[f] * np.sin(phi)
                d = bodyB.radius[f] - np.sqrt(np.maximum(inner, 0.0))
                total += (radius * abs(sweep) * 0.5 * (hi - lo)
                          * float(np.sum(weights * d ** 3)))
    return float(stiffness / 3.0 * total)


# UNVERIFIED(Cam)
def contactEnergy(bodyA, bodyB, stiffness = 1.0, quadratureOrder = 24):
    """Symmetrized contact energy for the ordered pair, as in ``polyContact.contactEnergy``."""
    return 0.5 * (pairEnergy(bodyA, bodyB, stiffness, quadratureOrder)
                  + pairEnergy(bodyB, bodyA, stiffness, quadratureOrder))


# UNVERIFIED(Cam)
def _cubicHarmonicIntegral(a, b, c, psi):
    """Antiderivative of ``(a + b cos psi + c sin psi)^3`` in ``psi``, evaluated at ``psi``.

    NO PHASE ANGLE ANYWHERE. The single-amplitude form ``a + C cos(psi - gamma)`` is tidier but needs
    ``arctan2``, which is not analytic and would break the complex step the gradient is taken with. The
    expanded form costs a few more terms and stays differentiable."""
    cosine, sine = np.cos(psi), np.sin(psi)
    linear = b * sine - c * cosine
    square = ((b * b + c * c) * psi / 2.0 + (b * b - c * c) * np.sin(2.0 * psi) / 4.0
              - b * c * np.cos(2.0 * psi) / 2.0)
    cube = (b ** 3 * (sine - sine ** 3 / 3.0) - b * b * c * cosine ** 3
            + b * c * c * sine ** 3 + c ** 3 * (-cosine + cosine ** 3 / 3.0))
    return a ** 3 * psi + 3.0 * a * a * linear + 3.0 * a * square + cube


# UNVERIFIED(Cam)
def frozenPairEnergy(loopA, rhoA, loopB, rhoB, partition, stiffness = 1.0, quadratureOrder = 24):
    """``pairEnergy`` with the PARTITION HELD FIXED -- analytic in every input, so it complex-steps.

    THE FROZEN PARTITION IS NOT AN APPROXIMATION. Differentiating the true energy would add Leibniz
    boundary terms wherever a breakpoint moves, and every one of them is zero:

      * at a SPAN endpoint ``dA`` crosses ``dB``, so ``d_B = 0`` there and the boundary term
        ``(k/3) d^3 dt/dp`` vanishes identically;
      * at an interior FEATURE SWITCH ``d_B`` is continuous (it is a min of continuous functions), so
        the term contributed by the sub-stretch ending there cancels the one contributed by the
        sub-stretch beginning there.

    So the derivative of the energy IS the derivative at fixed combinatorics, which is what makes an
    exact gradient reachable at all: the argmin and the root solving that produce the partition are not
    differentiable, and never have to be.

    The arc parametrization is by ROTATION of ``a^- - z`` rather than by absolute angle, for the same
    reason ``_cubicHarmonicIntegral`` avoids a phase: ``arctan2`` is not analytic."""
    piece, low, high, kind, feature = partition
    A = bodyFromBackbone(loopA, rhoA)
    B = bodyFromBackbone(loopB, rhoB)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadratureOrder))
    n = A.count
    total = 0.0 * (loopA[0, 0] + loopB[0, 0] + rhoA[0] + rhoB[0])

    edge = B.head - B.tail
    length = np.sqrt(edge[:, 0] ** 2 + edge[:, 1] ** 2)
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis = 1) / length[:, None]

    for p, lo, hi, k, f in zip(piece, low, high, kind, feature):
        if hi - lo <= 0.0:
            continue
        if p >= n:
            j = p - n
            tail = A.tail[j]
            vector = A.head[j] - A.tail[j]
            speed = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
            if k == 0:
                a = normal[f] @ (B.tail[f] - tail)
                m = normal[f] @ vector
                value = _cubicLineIntegral(a, m, lo, hi)
            else:
                offset = tail - B.center[f]
                qa = vector @ vector
                qb = 2.0 * (offset @ vector)
                qc = offset @ offset
                r = B.radius[f]
                firstHi, thirdHi = _rootAntiderivatives(qa, qb, qc, hi)
                firstLo, thirdLo = _rootAntiderivatives(qa, qb, qc, lo)
                second = (qa * (hi ** 3 - lo ** 3) / 3.0 + qb * (hi ** 2 - lo ** 2) / 2.0
                          + qc * (hi - lo))
                value = (r ** 3 * (hi - lo) - 3.0 * r * r * (firstHi - firstLo)
                         + 3.0 * r * second - (thirdHi - thirdLo))
            total = total + speed * value
        else:
            radius, sweep = A.radius[p], A.sweep[p]
            if np.real(A.radius[p]) == 0.0 or np.real(sweep) == 0.0:
                continue
            center = A.center[p]
            # The arc's start direction as a VECTOR, so no angle is ever formed. Arc p runs from
            # a^-_p to a^+_p, and a^-_p is stored as head[p - 1] (head[k] is a^- of corner k + 1).
            startVector = A.head[(p - 1) % n] - center
            turned = np.stack([-startVector[1], startVector[0]])
            if k == 0:
                a = normal[f] @ (B.tail[f] - center)
                b = -(normal[f] @ startVector)
                c = -(normal[f] @ turned)
                value = (_cubicHarmonicIntegral(a, b, c, sweep * hi)
                         - _cubicHarmonicIntegral(a, b, c, sweep * lo))
                # dl = radius |sweep| ds and dpsi = sweep ds, so integrating in psi carries
                # radius |sweep| / sweep = radius sign(sweep).
                total = total + radius * np.sign(np.real(sweep)) * value
            else:
                s = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
                psi = sweep * s
                delta = center - B.center[f]
                inner = (delta @ delta + radius * radius
                         + 2.0 * (np.cos(psi) * (delta @ startVector)
                                  + np.sin(psi) * (delta @ turned)))
                d = B.radius[f] - np.sqrt(inner)
                # |sweep| written as sweep * sign(Re sweep): abs() would take a modulus.
                total = total + (radius * sweep * np.sign(np.real(sweep)) * 0.5 * (hi - lo)
                                 * np.sum(weights * d ** 3))
    return stiffness / 3.0 * total


# UNVERIFIED(Cam)
def _rotate(vector, angle):
    """Rotate a 2-vector, complex-step safe."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.stack([cosine * vector[0] - sine * vector[1],
                     sine * vector[0] + cosine * vector[1]])


# UNVERIFIED(Cam)
def pairGradient(loopA, rhoA, loopB, rhoB, stiffness = 1.0, quadratureOrder = 24, step = 1e-30):
    """``(energy, dE/dloopA, dE/drhoA, dE/dloopB, dE/drhoB)`` for ``int_{dA cap B} (k/3) d_B^3 dl``.

    The partition is computed ONCE and frozen, then ``frozenPairEnergy`` is complex-stepped in every
    degree of freedom. Exact -- a complex step has no subtractive cancellation, so this is the true
    derivative rather than a difference quotient -- and self-consistent by construction, because the
    function differentiated IS the function evaluated.

    One evaluation per degree of freedom. KEPT AS THE REFERENCE ONLY -- ``pairGradientAnalytic`` is the
    same derivative in one pass and is what the drivers call. This one stays because it is
    self-consistent by construction (the function differentiated IS the function evaluated), which makes
    it the right thing to test the analytic form against."""
    loopA = np.asarray(loopA, dtype = float)
    loopB = np.asarray(loopB, dtype = float)
    rhoA = np.asarray(rhoA, dtype = float)
    rhoB = np.asarray(rhoB, dtype = float)
    bodyA, bodyB = bodyFromBackbone(loopA, rhoA), bodyFromBackbone(loopB, rhoB)
    piece, low, high, kind, feature, _ = substretches(bodyA, bodyB)
    partition = (piece, low, high, kind, feature)
    if not len(piece):
        return (0.0, np.zeros_like(loopA), np.zeros_like(rhoA),
                np.zeros_like(loopB), np.zeros_like(rhoB))

    energy = float(np.real(frozenPairEnergy(loopA, rhoA, loopB, rhoB, partition,
                                            stiffness, quadratureOrder)))

    def sensitivity(which, index):
        parts = [loopA.astype(complex), rhoA.astype(complex),
                 loopB.astype(complex), rhoB.astype(complex)]
        parts[which] = parts[which].copy()
        parts[which][index] += 1j * step
        value = frozenPairEnergy(parts[0], parts[1], parts[2], parts[3], partition,
                                 stiffness, quadratureOrder)
        return np.imag(value) / step

    gradients = []
    for which, template in ((0, loopA), (1, rhoA), (2, loopB), (3, rhoB)):
        out = np.zeros_like(template)
        for index in np.ndindex(template.shape):
            out[index] = sensitivity(which, index)
        gradients.append(out)
    return (energy, *gradients)


# UNVERIFIED(Cam)
def _lineIntegralPartials(a, m, lo, hi):
    """``(value, dv/da, dv/dm)`` for ``int_lo^hi (a - m t)^3 dt`` -- the derivative of
    ``_cubicLineIntegral`` with respect to its two coefficients.

    Differentiated in the EXPANDED form for the same reason the value is computed in it: the compact
    ``-((a - m hi)^4 - (a - m lo)^4) / (4 m)`` divides by ``m``, which is zero whenever the two segments
    are parallel."""
    d1, d2 = hi - lo, hi ** 2 - lo ** 2
    d3, d4 = hi ** 3 - lo ** 3, hi ** 4 - lo ** 4
    value = a ** 3 * d1 - 1.5 * a * a * m * d2 + a * m * m * d3 - 0.25 * m ** 3 * d4
    return (value,
            3.0 * a * a * d1 - 3.0 * a * m * d2 + m * m * d3,
            -1.5 * a * a * d2 + 2.0 * a * m * d3 - 0.75 * m * m * d4)


# UNVERIFIED(Cam)
def _arcToPartials(radius, height, lowW, highW):
    """``(value, d/dradius, d/dheight, d/dlowW, d/dhighW)`` for
    ``int_lowW^highW (radius - sqrt(w^2 + height^2))^3 dw``.

    THE SEGMENT-ALONG, ARC-TO CASE IN ITS NATURAL COORDINATE. ``pairEnergy`` writes that integral as a
    general quadratic ``sqrt(qa t^2 + qb t + qc)`` in the segment's own parameter and carries the
    ``speed`` outside; differentiating THAT form with respect to ``(qa, qb, qc)`` produces expressions
    with ``qa^(13/2)`` in the denominator. Rewriting in arc length first -- ``w`` measured from the foot
    of the perpendicular, ``height`` the perpendicular distance -- absorbs the speed into the measure
    and leaves four parameters whose partials are the SAME five antiderivatives the value already needs:

        int dw/R = asinh(w/h),  int dw = w,  int R dw,  int R^2 dw = w^3/3 + h^2 w,  int R^3 dw

    because ``d/dh (radius - R)^3 = -3h (radius - R)^2 / R`` and ``(radius - R)^2 / R`` expands into
    exactly those. The two endpoint partials are the fundamental theorem, so they cost nothing.

    ``height -> 0`` (the arc centre sitting on the chord's line) is the removable case: ``h asinh(w/h)``
    tends to zero, so the whole ``d/dheight`` does, and it is returned as zero."""
    squareH = height * height

    def antiderivatives(w):
        root = np.sqrt(w * w + squareH)
        arc = np.arcsinh(w / height) if height > 0.0 else 0.0
        return (arc, w,
                0.5 * w * root + 0.5 * squareH * arc,
                w ** 3 / 3.0 + squareH * w,
                w * (2.0 * w * w + 5.0 * squareH) * root / 8.0 + 0.375 * squareH * squareH * arc,
                root)

    arcLo, zeroLo, firstLo, secondLo, thirdLo, rootLo = antiderivatives(lowW)
    arcHi, zeroHi, firstHi, secondHi, thirdHi, rootHi = antiderivatives(highW)
    dArc, d0 = arcHi - arcLo, zeroHi - zeroLo
    d1, d2, d3 = firstHi - firstLo, secondHi - secondLo, thirdHi - thirdLo
    value = radius ** 3 * d0 - 3.0 * radius * radius * d1 + 3.0 * radius * d2 - d3
    return (value,
            3.0 * radius * radius * d0 - 6.0 * radius * d1 + 3.0 * d2,
            -3.0 * height * (radius * radius * dArc - 2.0 * radius * d0 + d1),
            -(radius - rootLo) ** 3,
            (radius - rootHi) ** 3)


# UNVERIFIED(Cam)
def _harmonicPartials(a, b, c, psi):
    """``(F, dF/da, dF/db, dF/dc, dF/dpsi)`` for ``F = int (a + b cos psi + c sin psi)^3 dpsi``.

    ``dF/dpsi`` is the integrand itself, by the fundamental theorem -- which is also the cheapest
    available check on the other three, since they were derived symbolically."""
    cosine, sine = np.cos(psi), np.sin(psi)
    doubleCos, doubleSin = np.cos(2.0 * psi), np.sin(2.0 * psi)
    da = (3.0 * a * a * psi + 6.0 * a * (b * sine - c * cosine) - 1.5 * b * c * doubleCos
          + 1.5 * psi * (b * b + c * c) + 0.75 * (b * b - c * c) * doubleSin)
    db = (3.0 * a * a * sine + 1.5 * a * (2.0 * b * psi + b * doubleSin - c * doubleCos)
          + b * b * (cosine ** 2 + 2.0) * sine - 2.0 * b * c * cosine ** 3 + c * c * sine ** 3)
    dc = (-3.0 * a * a * cosine - 1.5 * a * (b * doubleCos - 2.0 * c * psi + c * doubleSin)
          - b * b * cosine ** 3 + 2.0 * b * c * sine ** 3 + c * c * (cosine ** 2 - 3.0) * cosine)
    return (_cubicHarmonicIntegral(a, b, c, psi), da, db, dc,
            (a + b * cosine + c * sine) ** 3)


# UNVERIFIED(Cam)
class BodyGradient:
    """``dE/d(body array)`` for one ``RoundedBody``, in the same five arrays the energy reads.

    ``start`` is absent ON PURPOSE: nothing in the differentiated path uses it. The arc is parametrized
    by rotating ``a^- - z``, which is ``head[k - 1] - center[k]``, so the absolute angle (and its
    non-analytic ``arctan2``) never enters."""

    # UNVERIFIED(Cam)
    def __init__(self, count):
        self.center = np.zeros((count, 2))
        self.radius = np.zeros(count)
        self.sweep = np.zeros(count)
        self.tail = np.zeros((count, 2))
        self.head = np.zeros((count, 2))

    # UNVERIFIED(Cam)
    def flat(self):
        """The five arrays as one vector, in the order ``bodySensitivity`` uses."""
        return np.concatenate([self.center.ravel(), self.radius, self.sweep,
                               self.tail.ravel(), self.head.ravel()])

    # UNVERIFIED(Cam)
    def add(self, other, weight = 1.0):
        """Accumulate another body gradient in place -- one body collects from every pair it is in.

        ``weight`` is the outer chain rule the area tier needs: its energy is a function OF the overlap
        area, so each pair's contribution is scaled by ``dU/da`` before it joins the body's total."""
        self.center += weight * other.center
        self.radius += weight * other.radius
        self.sweep += weight * other.sweep
        self.tail += weight * other.tail
        self.head += weight * other.head
        return self


# UNVERIFIED(Cam)
def _normalPullback(gradient, edge, normal, length):
    """``d/d(edge vector)`` of ``gradient . normal`` for the inward unit normal ``normal = K e / |e|``.

    ``K e = (e_y, -e_x)``, so ``dn = K de / |e| - n (e . de) / |e|^2`` and the covector conjugate to
    ``de`` is ``[K^T g - (g . n) e / |e|] / |e|`` with ``K^T g = (-g_y, g_x)``. Both segment cases need
    it and both get it here rather than twice.

    THE TWO TERMS CARRY DIFFERENT POWERS OF ``|e|``. Factoring a single ``1 / |e|`` out of both is the
    obvious slip and it is nearly invisible: measured, it left one component 1.6% wrong and the other
    completely wrong, so a spot check on the wrong component reads as a pass."""
    return (np.array([-gradient[1], gradient[0]])
            - (gradient @ normal) * edge / length) / length


# UNVERIFIED(Cam)
def pairEnergyBodyGradient(bodyA, bodyB, partition, stiffness = 1.0, quadratureOrder = 24):
    """``(energy, gradA, gradB)`` -- the frozen-partition derivative of ``pairEnergy`` with respect to
    the BODY ARRAYS, analytically, in ONE pass over the partition.

    This is the piece that had to be derived. ``pairGradient`` gets the same numbers by complex-stepping
    the whole energy once per degree of freedom; here every sub-stretch is visited once and its
    contribution to all eight body arrays falls out of the chain rule, so the cost stops scaling with
    the degree-of-freedom count. That is the shape a kernel wants -- one thread per sub-stretch, a fixed
    amount of arithmetic, and an atomic scatter into the body arrays.

    THE BODY ARRAYS ARE TREATED AS INDEPENDENT here even though they are not: ``head[k] - center[k+1]``
    has length ``radius[k+1]`` by construction, and ``tail``/``head`` are the kiss points the arcs end
    at. Imposing those relations is exactly what ``bodySensitivity`` does, and keeping them out of this
    function is what makes it a plain chain rule with no geometry hidden in it.

    Four cases, matching ``pairEnergy`` term for term:

      * segment-along, segment-to: ``(alpha - m t)^3``, differentiated in ``(alpha, m)``;
      * segment-along, arc-to: re-expressed in arc length as ``(r - sqrt(w^2 + h^2))^3`` so the
        partials stay in the same five antiderivatives -- see ``_arcToPartials``;
      * arc-along, segment-to: ``(a + b cos + c sin)^3``, differentiated in ``(a, b, c)`` and in the
        endpoint angle, which is ``sweep`` times a frozen parameter;
      * arc-along, arc-to: quadrature, so the integrand is differentiated POINTWISE at the nodes and
        the same Gauss-Legendre weights carry the derivative. Exact for the same reason the value is.
    """
    piece, low, high, kind, feature = partition
    n = bodyA.count
    gradA, gradB = BodyGradient(n), BodyGradient(bodyB.count)
    if not len(piece):
        return 0.0, gradA, gradB

    nodes, weights = np.polynomial.legendre.leggauss(int(quadratureOrder))
    edge = bodyB.head - bodyB.tail
    length = np.maximum(np.hypot(edge[:, 0], edge[:, 1]), 1e-300)
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis = 1) / length[:, None]
    scale = stiffness / 3.0
    energy = 0.0

    for p, lo, hi, k, f in zip(piece, low, high, kind, feature):
        if hi - lo <= 0.0:
            continue

        if p >= n:
            j = p - n
            tail = bodyA.tail[j]
            vector = bodyA.head[j] - bodyA.tail[j]
            speed = float(np.hypot(vector[0], vector[1]))
            if speed <= 0.0:
                continue
            unit = vector / speed

            if k == 0:
                offset = bodyB.tail[f] - tail
                a = normal[f] @ offset
                m = normal[f] @ vector
                value, da, dm = _lineIntegralPartials(a, m, lo, hi)
                energy += scale * speed * value
                weight = scale * speed
                gradA.tail[j] += -unit * (scale * value) - normal[f] * (weight * (da + dm))
                gradA.head[j] += unit * (scale * value) + normal[f] * (weight * dm)
                toNormal = weight * (da * offset + dm * vector)
                conjugate = _normalPullback(toNormal, edge[f], normal[f], length[f])
                gradB.head[f] += conjugate
                gradB.tail[f] += -conjugate + normal[f] * (weight * da)
            else:
                delta = tail - bodyB.center[f]
                radius = bodyB.radius[f]
                # w is arc length from the foot of the perpendicular; the segment's speed is absorbed
                # into the measure, which is what makes the partials elementary.
                along = float(delta @ unit)
                perpendicular = delta - along * unit
                height = float(np.hypot(perpendicular[0], perpendicular[1]))
                value, dRadius, dHeight, dLow, dHigh = _arcToPartials(
                    radius, height, speed * lo + along, speed * hi + along)
                energy += scale * value
                shift = dLow + dHigh
                stretch = lo * dLow + hi * dHigh
                if height > 0.0:
                    direction = perpendicular / height
                else:
                    direction, dHeight = np.zeros(2), 0.0
                toDelta = scale * (shift * unit + dHeight * direction)
                toVector = scale * (stretch * unit
                                    + (shift * perpendicular - along * dHeight * direction) / speed)
                gradA.tail[j] += toDelta - toVector
                gradA.head[j] += toVector
                gradB.center[f] += -toDelta
                gradB.radius[f] += scale * dRadius
        else:
            radius, sweep = float(bodyA.radius[p]), float(bodyA.sweep[p])
            if radius <= 0.0 or sweep == 0.0:
                continue
            center = bodyA.center[p]
            previous = (p - 1) % n
            startVector = bodyA.head[previous] - center
            turned = np.array([-startVector[1], startVector[0]])
            orientation = np.sign(sweep)

            if k == 0:
                offset = bodyB.tail[f] - center
                a = normal[f] @ offset
                b = -(normal[f] @ startVector)
                c = -(normal[f] @ turned)
                valueHi, daHi, dbHi, dcHi, dPsiHi = _harmonicPartials(a, b, c, sweep * hi)
                valueLo, daLo, dbLo, dcLo, dPsiLo = _harmonicPartials(a, b, c, sweep * lo)
                value = valueHi - valueLo
                da, db, dc = daHi - daLo, dbHi - dbLo, dcHi - dcLo
                energy += scale * radius * orientation * value
                weight = scale * radius * orientation
                gradA.radius[p] += scale * orientation * value
                gradA.sweep[p] += weight * (hi * dPsiHi - lo * dPsiLo)
                turnedNormal = np.array([-normal[f][1], normal[f][0]])
                toStart = weight * (-db * normal[f] + dc * turnedNormal)
                gradA.head[previous] += toStart
                gradA.center[p] += -weight * da * normal[f] - toStart
                toNormal = weight * (da * offset - db * startVector - dc * turned)
                conjugate = _normalPullback(toNormal, edge[f], normal[f], length[f])
                gradB.head[f] += conjugate
                gradB.tail[f] += -conjugate + normal[f] * (weight * da)
            else:
                gap = center - bodyB.center[f]
                parameter = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
                psi = sweep * parameter
                cosine, sine = np.cos(psi), np.sin(psi)
                alongStart = float(gap @ startVector)
                alongTurned = float(gap @ turned)
                inner = (gap @ gap + radius * radius
                         + 2.0 * (cosine * alongStart + sine * alongTurned))
                root = np.sqrt(np.maximum(inner, 0.0))
                depth = bodyB.radius[f] - root
                measure = scale * radius * abs(sweep) * 0.5 * (hi - lo)
                cubed = float(np.sum(weights * depth ** 3))
                energy += measure * cubed
                # Pointwise: dE/d(depth_i), then dE/d(inner_i) through depth = rho - sqrt(inner).
                toDepth = measure * weights * 3.0 * depth ** 2
                toInner = -0.5 * toDepth / np.maximum(root, 1e-300)
                turnedGap = np.array([-gap[1], gap[0]])
                cosineSum = float(np.sum(toInner * cosine))
                sineSum = float(np.sum(toInner * sine))
                totalInner = float(np.sum(toInner))
                toGap = 2.0 * (totalInner * gap + cosineSum * startVector + sineSum * turned)
                toStart = 2.0 * (cosineSum * gap - sineSum * turnedGap)
                gradB.radius[f] += float(np.sum(toDepth))
                gradB.center[f] += -toGap
                gradA.head[previous] += toStart
                gradA.center[p] += toGap - toStart
                gradA.radius[p] += (2.0 * radius * totalInner
                                    + scale * abs(sweep) * 0.5 * (hi - lo) * cubed)
                gradA.sweep[p] += (float(np.sum(toInner * parameter * 2.0
                                                * (cosine * alongTurned - sine * alongStart)))
                                   + scale * radius * orientation * 0.5 * (hi - lo) * cubed)
    return float(energy), gradA, gradB


# UNVERIFIED(Cam)
def bodySensitivity(loop, rho, step = 1e-30):
    """``d(body arrays)/d(loop, rho)`` for one body, as a ``(degrees of freedom, 8n)`` matrix.

    STILL A COMPLEX STEP, and deliberately. The corner map is LOCAL and CHEAP -- ``cornerFrame`` on
    ``(v_prev, v, v_next, rho)`` -- so this costs ``3n`` evaluations of a vectorized closed form ONCE
    PER BODY per force evaluation, not once per pair. The thing that had to stop scaling was the
    per-pair integral work, and that is what ``pairEnergyBodyGradient`` fixed; differentiating the
    corner frame by hand would buy a constant on a term that is already negligible, and it would be one
    more place for the value and the derivative to drift apart.

    Column order matches ``BodyGradient.flat``: centre, radius, sweep, tail, head."""
    loop = np.asarray(loop, dtype = float).reshape(-1, 2)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    count = len(loop)
    rows = []
    for which in range(3 * count):
        complexLoop, complexRho = loop.astype(complex), rho.astype(complex)
        if which < 2 * count:
            complexLoop[which // 2, which % 2] += 1j * step
        else:
            complexRho[which - 2 * count] += 1j * step
        body = bodyFromBackbone(complexLoop, complexRho)
        rows.append(np.concatenate([np.imag(body.center).ravel(), np.imag(body.radius),
                                    np.imag(body.sweep), np.imag(body.tail).ravel(),
                                    np.imag(body.head).ravel()]) / step)
    return np.stack(rows)


# UNVERIFIED(Cam)
def applyBodySensitivity(sensitivity, gradient, count):
    """``(dE/dloop, dE/drho)`` from ``dE/d(body arrays)`` and a ``bodySensitivity`` matrix."""
    flat = sensitivity @ gradient.flat()
    return flat[:2 * count].reshape(count, 2), flat[2 * count:]


# UNVERIFIED(Cam)
def pairGradientAnalytic(loopA, rhoA, loopB, rhoB, stiffness = 1.0, quadratureOrder = 24,
                         sensitivityA = None, sensitivityB = None):
    """``pairGradient``'s signature and return, with the integrals differentiated ANALYTICALLY.

    Pass ``sensitivityA`` / ``sensitivityB`` when the caller already holds them -- they depend on one
    body alone, so a packing builds each once and reuses it across every pair that body appears in."""
    loopA = np.asarray(loopA, dtype = float).reshape(-1, 2)
    loopB = np.asarray(loopB, dtype = float).reshape(-1, 2)
    rhoA = np.asarray(rhoA, dtype = float).reshape(-1)
    rhoB = np.asarray(rhoB, dtype = float).reshape(-1)
    bodyA, bodyB = bodyFromBackbone(loopA, rhoA), bodyFromBackbone(loopB, rhoB)
    piece, lowValue, highValue, kind, feature, _ = substretches(bodyA, bodyB)
    if not len(piece):
        return (0.0, np.zeros_like(loopA), np.zeros_like(rhoA),
                np.zeros_like(loopB), np.zeros_like(rhoB))

    energy, gradA, gradB = pairEnergyBodyGradient(
        bodyA, bodyB, (piece, lowValue, highValue, kind, feature), stiffness, quadratureOrder)
    if sensitivityA is None:
        sensitivityA = bodySensitivity(loopA, rhoA)
    if sensitivityB is None:
        sensitivityB = bodySensitivity(loopB, rhoB)
    loopGradA, rhoGradA = applyBodySensitivity(sensitivityA, gradA, len(loopA))
    loopGradB, rhoGradB = applyBodySensitivity(sensitivityB, gradB, len(loopB))
    return energy, loopGradA, rhoGradA, loopGradB, rhoGradB


# UNVERIFIED(Cam)
def _greenIntegral(body, piece, lo, hi):
    """``int (x dy - y dx)`` along a piece of ``body`` over ``[lo, hi]``. Twice the swept area.

    Segment: with ``x = tail + t v`` the ``t`` terms cancel identically and it is
    ``(tail_x v_y - tail_y v_x)(hi - lo)``.

    Arc: with ``x = c + w(psi)`` and ``w = Rot(psi) v0``, ``dx = J w dpsi`` and the integrand collapses
    to ``(c . w + r^2) dpsi`` because ``|w| = r``. Parametrized by ROTATION, so no ``arctan2``."""
    n = body.count
    if piece >= n:
        k = piece - n
        tail = body.tail[k]
        vector = body.head[k] - body.tail[k]
        return (tail[0] * vector[1] - tail[1] * vector[0]) * (hi - lo)
    center = body.center[piece]
    radius, sweep = body.radius[piece], body.sweep[piece]
    startVector = body.head[(piece - 1) % n] - center
    turned = np.stack([-startVector[1], startVector[0]])
    psiLo, psiHi = sweep * lo, sweep * hi
    return (radius * radius * (psiHi - psiLo)
            + (center @ startVector) * (np.sin(psiHi) - np.sin(psiLo))
            - (center @ turned) * (np.cos(psiHi) - np.cos(psiLo)))


# UNVERIFIED(Cam)
def overlapArea(bodyA, bodyB):
    """EXACT area of ``A and B`` for two segment-and-arc bodies. No quadrature, no clipping.

    THE BOUNDARY OF THE INTERSECTION IS ALREADY IN HAND. For two counter-clockwise regions it is
    exactly the pieces of ``dA`` lying inside ``B`` plus the pieces of ``dB`` lying inside ``A``, each
    kept in its original direction -- which is what ``spans`` returns. Green's theorem then adds those
    contributions with no need to sort them into loops or to intersect anything: the pieces already
    close up, and ``int (x dy - y dx)`` is additive over them however they are ordered.

    That is why this needs no polygon clipper. A general arc-aware clipper would have to build the
    intersection loops explicitly and get every degenerate junction right; here the same partition that
    the depth law integrates over does the whole job."""
    total = 0.0
    for body, other in ((bodyA, bodyB), (bodyB, bodyA)):
        piece, low, high = spans(body, other)
        for p, lo, hi in zip(piece, low, high):
            total = total + _greenIntegral(body, p, lo, hi)
    return float(0.5 * np.real(total))


# UNVERIFIED(Cam)
def frozenOverlapArea(loopA, rhoA, loopB, rhoB, partitionA, partitionB):
    """``overlapArea`` with both span sets held fixed -- analytic, for the gradient. See
    ``frozenPairEnergy`` on why freezing the combinatorics is exact rather than approximate."""
    A = bodyFromBackbone(loopA, rhoA)
    B = bodyFromBackbone(loopB, rhoB)
    total = 0.0 * (loopA[0, 0] + loopB[0, 0] + rhoA[0] + rhoB[0])
    for body, (piece, low, high) in ((A, partitionA), (B, partitionB)):
        for p, lo, hi in zip(piece, low, high):
            total = total + _greenIntegral(body, p, lo, hi)
    return 0.5 * total


# UNVERIFIED(Cam)
def _pieceFrame(body, piece, parameter):
    """``(point, outwardNormal, speed)`` along a piece at ``parameter``, all complex-step safe.

    ``speed`` is ``|dx/dparameter|``, so ``ds = speed d(parameter)``."""
    n = body.count
    if piece >= n:
        k = piece - n
        vector = body.head[k] - body.tail[k]
        speed = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        point = body.tail[k] + parameter[:, None] * vector
        normal = np.stack([vector[1], -vector[0]]) / speed
        return point, np.broadcast_to(normal, point.shape), speed
    center, radius, sweep = body.center[piece], body.radius[piece], body.sweep[piece]
    startVector = body.head[(piece - 1) % n] - center
    turned = np.stack([-startVector[1], startVector[0]])
    psi = sweep * parameter
    offset = (np.cos(psi)[:, None] * startVector[None, :]
              + np.sin(psi)[:, None] * turned[None, :])
    point = center[None, :] + offset
    # The corner circle is cut in from INSIDE, so its centre is interior and the outward normal points
    # back toward the arc from the centre.
    normal = offset / radius
    return point, normal, radius * sweep * np.sign(np.real(sweep))


# UNVERIFIED(Cam)
def areaGradient(loopA, rhoA, loopB, rhoB, quadratureOrder = 12, step = 1e-30):
    """``(area, dA/dloopA, dA/drhoA, dA/dloopB, dA/drhoB)`` for the exact overlap area.

    NOT the frozen-partition derivative. That works for the ENERGY because its integrand vanishes at a
    crossing, so the Leibniz boundary terms die; Green's integrand does NOT vanish there, and freezing
    the partition is measured 40-60% wrong. The shape derivative of a set intersection has no boundary
    terms at all:

        dA/dp = int_{dA cap B} (v_A . n_A) ds + int_{dB cap A} (v_B . n_B) ds

    with ``v = dx/dp`` the velocity of the boundary point at FIXED parameter and ``n`` the outward
    normal. The moving crossings enter only through the domains, and a point is measure zero. So the
    spans may be frozen here after all -- but for a different reason, and via a different formula."""
    loopA, loopB = np.asarray(loopA, float), np.asarray(loopB, float)
    rhoA, rhoB = np.asarray(rhoA, float), np.asarray(rhoB, float)
    A, B = bodyFromBackbone(loopA, rhoA), bodyFromBackbone(loopB, rhoB)
    partitions = (spans(A, B), spans(B, A))
    area = overlapArea(A, B)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadratureOrder))

    def normalFlux(parts, whichBody):
        """int (v . n) ds over that body's spans, with v taken by complex step."""
        body = bodyFromBackbone(parts[0], parts[1]) if whichBody == 0 else \
            bodyFromBackbone(parts[2], parts[3])
        piece, low, high = partitions[whichBody]
        total = 0.0
        for p, lo, hi in zip(piece, low, high):
            if hi - lo <= 0.0:
                continue
            t = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
            point, normal, speed = _pieceFrame(body, p, t)
            velocity = np.imag(point) / step
            outward = np.real(normal)
            total += float(np.sum(weights * np.sum(velocity * outward, axis = 1))
                           * 0.5 * (hi - lo) * float(np.real(speed)))
        return total

    gradients = []
    for which, template in ((0, loopA), (1, rhoA), (2, loopB), (3, rhoB)):
        out = np.zeros_like(template)
        for index in np.ndindex(template.shape):
            parts = [loopA.astype(complex), rhoA.astype(complex),
                     loopB.astype(complex), rhoB.astype(complex)]
            parts[which] = parts[which].copy()
            parts[which][index] += 1j * step
            out[index] = normalFlux(parts, 0 if which < 2 else 1)
        gradients.append(out)
    return (area, *gradients)


# UNVERIFIED(Cam)
def overlapAreaBodyGradient(bodyA, bodyB, spansA, spansB):
    """``(area, gradA, gradB)`` -- the shape derivative of ``overlapArea`` in the BODY arrays,
    analytically and with NO QUADRATURE at all.

    ``areaGradient`` evaluates ``int (v . n) ds`` by Gauss-Legendre once per degree of freedom. Written
    out, both pieces integrate in closed form and three of the five body arrays drop out entirely:

      * ``x`` does not contain ``radius`` or ``sweep`` -- the point on an arc is
        ``center + cos psi sv + sin psi (J sv)`` with ``sv = head[k-1] - center`` -- so their
        velocities are zero and only ``center``, ``tail`` and ``head`` can move the area;
      * along an arc the TANGENTIAL velocity has no normal component: ``offset . (-sin psi sv +
        cos psi J sv) = 0`` identically, which is why ``d/dsweep`` vanishes rather than merely being
        small;
      * ``(cos psi I + sin psi J)^T offset = Rot(-psi) offset = sv``, so the whole ``sv`` term collapses
        to ``|sweep| (hi - lo) (dsv . sv)`` with no trigonometry left in it.

    What remains is ``int offset dt``, which is elementary. So this is EXACTER than the quadrature it
    replaces as well as cheaper: the arc integrand is trigonometric, and order-12 Gauss-Legendre was
    very accurate on it but never exact.

    THE SHAPE DERIVATIVE, NOT THE FROZEN-PARTITION ONE -- see ``areaGradient`` for why the energy's
    argument does not transfer."""
    gradA, gradB = BodyGradient(bodyA.count), BodyGradient(bodyB.count)
    area = 0.0
    for body, gradient, (piece, low, high) in ((bodyA, gradA, spansA), (bodyB, gradB, spansB)):
        n = body.count
        for p, lo, hi in zip(piece, low, high):
            area += 0.5 * float(np.real(_greenIntegral(body, p, lo, hi)))
            if hi - lo <= 0.0:
                continue
            if p >= n:
                k = p - n
                vector = body.head[k] - body.tail[k]
                scaledNormal = np.array([vector[1], -vector[0]])
                second = 0.5 * (hi * hi - lo * lo)
                gradient.tail[k] += scaledNormal * ((hi - lo) - second)
                gradient.head[k] += scaledNormal * second
            else:
                radius, sweep = float(body.radius[p]), float(body.sweep[p])
                if radius <= 0.0 or sweep == 0.0:
                    continue
                center = body.center[p]
                startVector = body.head[(p - 1) % n] - center
                turned = np.array([-startVector[1], startVector[0]])
                psiLo, psiHi = sweep * lo, sweep * hi
                moment = ((np.sin(psiHi) - np.sin(psiLo)) / sweep * startVector
                          + (np.cos(psiLo) - np.cos(psiHi)) / sweep * turned)
                magnitude = abs(sweep)
                toStart = magnitude * (hi - lo) * startVector
                gradient.center[p] += magnitude * moment - toStart
                gradient.head[(p - 1) % n] += toStart
    return area, gradA, gradB


# UNVERIFIED(Cam)
def areaGradientAnalytic(loopA, rhoA, loopB, rhoB, sensitivityA = None, sensitivityB = None):
    """``areaGradient``'s signature and return, differentiated analytically."""
    loopA = np.asarray(loopA, dtype = float).reshape(-1, 2)
    loopB = np.asarray(loopB, dtype = float).reshape(-1, 2)
    rhoA = np.asarray(rhoA, dtype = float).reshape(-1)
    rhoB = np.asarray(rhoB, dtype = float).reshape(-1)
    bodyA, bodyB = bodyFromBackbone(loopA, rhoA), bodyFromBackbone(loopB, rhoB)
    area, gradA, gradB = overlapAreaBodyGradient(bodyA, bodyB, spans(bodyA, bodyB), spans(bodyB, bodyA))
    if sensitivityA is None:
        sensitivityA = bodySensitivity(loopA, rhoA)
    if sensitivityB is None:
        sensitivityB = bodySensitivity(loopB, rhoB)
    loopGradA, rhoGradA = applyBodySensitivity(sensitivityA, gradA, len(loopA))
    loopGradB, rhoGradB = applyBodySensitivity(sensitivityB, gradB, len(loopB))
    return area, loopGradA, rhoGradA, loopGradB, rhoGradB


# UNVERIFIED(Cam)
def bodyReach(body):
    """``(centroid, radius)`` bounding every point of the body, ARCS INCLUDED.

    The farthest point of an arc is not one of its endpoints, so a hull over the stored points would
    under-estimate and the cull would drop pairs that really do touch. ``|center - c| + radius`` bounds
    the whole arc."""
    points = np.concatenate([body.tail, body.head])
    centroid = points.mean(axis = 0)
    reach = np.hypot(*(points - centroid).T).max()
    if body.count:
        arcs = np.hypot(*(body.center - centroid).T) + body.radius
        reach = max(reach, float(arcs.max()))
    return centroid, float(reach)


# UNVERIFIED(Cam)
def candidatePairs(bodies, exterior = None):
    """Pairs whose reaches overlap, plus every pair involving ``exterior``.

    AN EXTERIOR BODY IS NEVER CULLED, and this is correctness rather than tuning: a wall's region is
    its COMPLEMENT, which is unbounded, so a body far from the wall's centroid is not far from the
    obstacle -- it is deep inside it. ``polyContactSystem.circumradii`` records what culling that pair
    costs: the confinement force goes to exactly zero and a body that drifts out can never be pushed
    back."""
    frames = [bodyReach(b) for b in bodies]
    pairs = []
    for a in range(len(bodies)):
        for b in range(a + 1, len(bodies)):
            if exterior is not None and (a == exterior or b == exterior):
                pairs.append((a, b))
                continue
            gap = np.hypot(*(frames[a][0] - frames[b][0]))
            if gap <= frames[a][1] + frames[b][1]:
                pairs.append((a, b))
    return pairs


# UNVERIFIED(Cam)
def packingEnergyForce(packing, rho, stiffness = 1.0, wallStiffness = 1.0, quadratureOrder = 24):
    """``(energy, force, rhoForce)`` for a whole packing on the EXACT-ARC depth law.

    The adapter between this module and the rest of the project, and deliberately the same shape as
    ``polyContactSystem.packingEnergyForce``: force is MINUS the gradient, shape ``(V, 2)``, plus the
    per-vertex ``-dE/drho`` the chorded path returns through ``Model.getRhoForces``.

    THE CONTAINER IS THE EXTERIOR REGION, exactly as in ``polyContactSystem``. A wall handed over as
    drawn would be an obstacle containing every body, and the law would push them all out of it. The
    confining region is its COMPLEMENT, and membership here is read from the winding just as it is
    there -- ``nearestFeature`` uses the counter-clockwise outward-normal convention, so a CLOCKWISE
    loop has its inside and outside exchanged, which is precisely the wall.

    THE GRADIENT IS ANALYTIC and assembled in the BODY arrays, not in the backbone: every pair adds to
    ``dE/d(body)`` and each body is pulled back to ``(loop, rho)`` exactly ONCE at the end. That is the
    only ordering in which the corner map's own derivative costs ``O(bodies)`` rather than
    ``O(pairs)``, and it is the ordering a kernel wants -- scatter into body arrays, then one pass to
    convert."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    container = getattr(packing, "containerIndex", None)
    count = len(starts) - 1

    loops = [vertices[starts[p]:starts[p + 1]] for p in range(count)]
    radii = [rho[starts[p]:starts[p + 1]] for p in range(count)]
    energy = 0.0
    force = np.zeros_like(vertices)
    rhoForce = np.zeros_like(rho)

    bodies = [bodyFromBackbone(loops[p], radii[p]) for p in range(count)]
    gradients = [BodyGradient(bodies[p].count) for p in range(count)]
    for a, b in candidatePairs(bodies, exterior = container):
            pairStiffness = stiffness
            if container is not None and (a == container or b == container):
                pairStiffness *= wallStiffness
            # Symmetrized: half of each ordered direction, as contactEnergy does.
            for first, second in ((a, b), (b, a)):
                piece, low, high, kind, feature, _ = substretches(bodies[first], bodies[second])
                if not len(piece):
                    continue
                value, gradFirst, gradSecond = pairEnergyBodyGradient(
                    bodies[first], bodies[second], (piece, low, high, kind, feature),
                    0.5 * pairStiffness, quadratureOrder)
                energy += value
                gradients[first].add(gradFirst)
                gradients[second].add(gradSecond)

    for p in range(count):
        gradLoop, gradRho = applyBodySensitivity(
            bodySensitivity(loops[p], radii[p]), gradients[p], len(loops[p]))
        force[starts[p]:starts[p + 1]] -= gradLoop
        rhoForce[starts[p]:starts[p + 1]] -= gradRho
    return float(energy), force, rhoForce


# UNVERIFIED(Cam)
def packingBodies(packing, rho):
    """``(bodies, container)`` for a whole packing, with the wall reversed if it arrives
    counter-clockwise -- the same convention ``packingAreaEnergyForce`` uses."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    container = getattr(packing, "containerIndex", None)
    count = len(starts) - 1
    loops = [vertices[starts[p]:starts[p + 1]] for p in range(count)]
    radii = [rho[starts[p]:starts[p + 1]] for p in range(count)]
    if container is not None:
        following = np.roll(loops[container], -1, axis = 0)
        signed = 0.5 * np.sum(loops[container][:, 0] * following[:, 1]
                              - following[:, 0] * loops[container][:, 1])
        if signed > 0.0:
            loops[container] = loops[container][::-1].copy()
            radii[container] = radii[container][::-1].copy()
    return [bodyFromBackbone(loops[p], radii[p]) for p in range(count)], container


# UNVERIFIED(Cam)
def packingMeasurements(packing, rho, stiffness = 1.0, wallStiffness = 1.0, quadratureOrder = 24):
    """``(pairEnergy, wallEnergy, pairArea, wallArea)`` on the EXACT arcs.

    The measurement counterpart of ``packingEnergyForce``, and it exists because the getters must see
    the same shape the law does. Under chorded geometry ``Model.measuredGeometry`` hands every getter
    the chorded polygon, which is right there and wrong here: the force would be exact while
    ``getExcessEnergy`` -- the number the load controller steers on -- read a shape that differs from it
    by up to 5% at the round end of a schedule.

    SPLIT BY WHETHER THE CONTAINER IS INVOLVED, because every jamming criterion in this project treats
    body-body contact and containment as ALTERNATIVES rather than as one number: a confined packing
    under stress relieves it either by bearing on its neighbors or by extruding through the wall, and a
    criterion written on the total accepts the second as jammed."""
    bodies, container = packingBodies(packing, rho)
    pairEnergyTotal, wallEnergyTotal = 0.0, 0.0
    pairAreaTotal, wallAreaTotal = 0.0, 0.0
    for a, b in candidatePairs(bodies, exterior = container):
        isWall = container is not None and (a == container or b == container)
        energy = contactEnergy(bodies[a], bodies[b],
                               stiffness * (wallStiffness if isWall else 1.0), quadratureOrder)
        area = overlapArea(bodies[a], bodies[b])
        if isWall:
            wallEnergyTotal += energy
            wallAreaTotal += area
        else:
            pairEnergyTotal += energy
            pairAreaTotal += area
    return pairEnergyTotal, wallEnergyTotal, pairAreaTotal, wallAreaTotal


# UNVERIFIED(Cam)
def wallPenetration(packing, rho, perArc = 64):
    """Deepest excursion of any body outside the container, as a DISTANCE.

    Same definition and same algorithm as ``Model.getWallPenetration`` -- the largest distance from the
    wall's boundary to a point of a body lying outside it -- but sampled from the EXACT arcs rather
    than from the chorded polygon, so the deepest point of an arc is not missed between two chords."""
    bodies, container = packingBodies(packing, rho)
    if container is None:
        return 0.0
    wall = bodies[container]
    worst = 0.0
    for index, body in enumerate(bodies):
        if index == container:
            continue
        points = body.sample(perArc)
        inside = signedDistance(points, wall)
        # The wall is the EXTERIOR region, so a point INSIDE it (positive depth) is outside the box.
        worst = max(worst, float(np.max(np.maximum(inside, 0.0))))
    return worst


# UNVERIFIED(Cam)
# Corners per body the CUDA kernel is compiled for -- ROUNDED_MAXN in cuda/roundedContact.cuh. It is a
# COMPILE-TIME STRIDE, so a wider body would read past the end of its block rather than fail; callers
# check against this and fall back to numpy instead.
CUDA_MAX_CORNERS = 16


# UNVERIFIED(Cam)
def packingEnergyForceCuda(packing, rho, stiffness = 1.0, wallStiffness = 1.0):
    """``packingEnergyForce`` with the integrals on the GPU. Same contract, same return.

    THE DEVICE RETURNS ``dE/d(body arrays)`` AND THE CORNER MAP IS APPLIED HERE. That split is the
    point of the whole design: the integral work is O(pairs) and belongs on the device, the corner
    map's own derivative is O(bodies) and runs once per force evaluation, so moving it across would
    buy a constant on a negligible term.

    EXPECT ~1e-07 ON FORCES, NOT 1e-14. The sub-stretch breakpoints are only placed to sqrt(machine
    epsilon), because every switch involving an arc is solved by squaring away a square root, and where
    a segment meets its OWN tangent arc the two distances coincide rather than crossing so nothing can
    sharpen them. The energy is first-order insensitive to that and agrees to 1e-14; the frozen
    gradient is first-order sensitive and does not. See the note above ``_switchesAlongSegment``."""
    import cudaOverlap
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    count = len(starts) - 1
    bodies, container = packingBodies(packing, rho)

    # packingBodies may have reversed the wall; rebuild the flat arrays it actually used so the device
    # sees the same winding, and remember to un-reverse that block on the way out.
    loops, radii, reversedWall = [], [], False
    for p in range(count):
        loop = vertices[starts[p]:starts[p + 1]]
        radius = rho[starts[p]:starts[p + 1]]
        if p == container:
            following = np.roll(loop, -1, axis = 0)
            signed = 0.5 * np.sum(loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1])
            if signed > 0.0:
                loop, radius, reversedWall = loop[::-1].copy(), radius[::-1].copy(), True
        loops.append(loop)
        radii.append(radius)

    energy, flat = cudaOverlap.roundedContactCuda(
        np.concatenate(loops), starts.astype(np.int32), np.concatenate(radii),
        containerIndex = container, stiffness = stiffness, wallStiffness = wallStiffness)

    force = np.zeros_like(vertices)
    rhoForce = np.zeros_like(rho)
    for p in range(count):
        n = len(loops[p])
        gradient = BodyGradient(n)
        stride = flat.shape[1] // 8
        row = flat[p]
        gradient.center[:] = row[0:2 * n].reshape(n, 2)
        gradient.radius[:] = row[2 * stride:2 * stride + n]
        gradient.sweep[:] = row[3 * stride:3 * stride + n]
        gradient.tail[:] = row[4 * stride:4 * stride + 2 * n].reshape(n, 2)
        gradient.head[:] = row[6 * stride:6 * stride + 2 * n].reshape(n, 2)
        gradLoop, gradRho = applyBodySensitivity(
            bodySensitivity(loops[p], radii[p]), gradient, n)
        if p == container and reversedWall:
            gradLoop, gradRho = gradLoop[::-1], gradRho[::-1]
        force[starts[p]:starts[p + 1]] -= gradLoop
        rhoForce[starts[p]:starts[p + 1]] -= gradRho
    return float(energy), force, rhoForce


# UNVERIFIED(Cam)
def _packingLoops(packing, rho):
    """``(loops, radii, container, reversedWall)`` with the wall flipped if it arrived counter-clockwise.

    The winding flip lives here rather than in the kernels: the wall is the EXTERIOR region, and which
    side "inside" means is decided by orientation. Getting it backwards is SILENT -- the energy stays
    smooth and its gradient stays self-consistent, so only an independent area check would catch it."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    container = getattr(packing, "containerIndex", None)
    loops, radii, reversedWall = [], [], False
    for p in range(len(starts) - 1):
        loop = vertices[starts[p]:starts[p + 1]]
        radius = rho[starts[p]:starts[p + 1]]
        if p == container:
            following = np.roll(loop, -1, axis = 0)
            signed = 0.5 * np.sum(loop[:, 0] * following[:, 1] - following[:, 0] * loop[:, 1])
            if signed > 0.0:
                loop, radius, reversedWall = loop[::-1].copy(), radius[::-1].copy(), True
        loops.append(loop)
        radii.append(radius)
    return loops, radii, container, reversedWall


# UNVERIFIED(Cam)
def _scatterBodyGradients(flat, loops, radii, starts, container, reversedWall, shape):
    """Pull each body's flat ``dE/d(body)`` block back to ``(loop, rho)`` and accumulate the forces."""
    force = np.zeros(shape)
    rhoForce = np.zeros(starts[-1])
    stride = flat.shape[1] // 8
    for p in range(len(loops)):
        n = len(loops[p])
        gradient = BodyGradient(n)
        row = flat[p]
        gradient.center[:] = row[0:2 * n].reshape(n, 2)
        gradient.radius[:] = row[2 * stride:2 * stride + n]
        gradient.sweep[:] = row[3 * stride:3 * stride + n]
        gradient.tail[:] = row[4 * stride:4 * stride + 2 * n].reshape(n, 2)
        gradient.head[:] = row[6 * stride:6 * stride + 2 * n].reshape(n, 2)
        gradLoop, gradRho = applyBodySensitivity(
            bodySensitivity(loops[p], radii[p]), gradient, n)
        if p == container and reversedWall:
            gradLoop, gradRho = gradLoop[::-1], gradRho[::-1]
        force[starts[p]:starts[p + 1]] -= gradLoop
        rhoForce[starts[p]:starts[p + 1]] -= gradRho
    return force, rhoForce


# UNVERIFIED(Cam)
def packingAreaEnergyForceCuda(packing, rho, kOverlap = 1.0, kContainer = 1.0):
    """``packingAreaEnergyForce`` with the geometry on the GPU. Same contract, same return.

    TWO PASSES, because the functional is a function OF the area: ``U = 2k (a/norm)^2`` needs each
    pair's whole overlap before it can weight that pair's shape derivative. The device measures the
    areas, this computes the weights -- which is where the container's normalizer lives, so that
    convention stays in one place -- and the device scatters."""
    import cudaOverlap
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    targetArea = np.asarray(packing.targetArea, dtype = float)
    loops, radii, container, reversedWall = _packingLoops(packing, rho)
    count = len(loops)
    flatLoops = np.concatenate(loops)
    flatRadii = np.concatenate(radii)
    starts32 = starts.astype(np.int32)

    areas = cudaOverlap.roundedAreaCuda(flatLoops, starts32, flatRadii, containerIndex = container)
    energy = 0.0
    weights = np.zeros_like(areas)
    for a in range(count):
        for b in range(a + 1, count):
            area = areas[a, b]
            if area == 0.0:
                continue
            if container is not None and (a == container or b == container):
                # The wall's OWN target area is deliberately not in the normalizer -- it is the size of
                # the whole box, and using it would make the wall tens of times softer than a contact.
                norm, strength = 2.0 * targetArea[b if a == container else a], kContainer
            else:
                norm, strength = targetArea[a] + targetArea[b], kOverlap
            energy += 2.0 * strength * (area / norm) ** 2
            weights[a, b] = 4.0 * strength * area / (norm * norm)

    flat = cudaOverlap.roundedAreaGradientCuda(flatLoops, starts32, flatRadii, weights,
                                               containerIndex = container)
    force, rhoForce = _scatterBodyGradients(flat, loops, radii, starts, container, reversedWall,
                                            vertices.shape)
    return float(energy), force, rhoForce


# UNVERIFIED(Cam)
def packingAreaEnergyForce(packing, rho, kOverlap = 1.0, kContainer = 1.0):
    """``(energy, force, rhoForce)`` for a whole packing on the EXACT-ARC AREA law.

    The same NORMALIZED-SQUARED functional ``energies.sharpOverlapEnergyForce`` and
    ``sharpContainerEnergyForce`` use, so the two are the same contact law measured on two different
    shapes rather than two different laws:

        U = 2 k sum_{A<B} (a_AB / (targetArea_A + targetArea_B))^2
          + 2 kContainer sum_S (a_S / (2 targetArea_S))^2

    with ``a_S`` the area of ``S`` lying OUTSIDE the wall. The wall's own target area is deliberately
    NOT in its normalizer -- it is the size of the whole box, and using it would make the wall tens of
    times softer than a body-body contact.

    WINDING decides which side the overlap reports, and getting it backwards is SILENT: the energy
    stays smooth and its gradient stays self-consistent, so only an independent area check catches it.
    The wall is reversed here when it arrives counter-clockwise, and the gradient is un-reversed on the
    way out, exactly as ``polyContactSystem.packingEnergyForce`` does."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    rho = np.asarray(rho, dtype = float).reshape(-1)
    targetArea = np.asarray(packing.targetArea, dtype = float)
    container = getattr(packing, "containerIndex", None)
    count = len(starts) - 1

    loops = [vertices[starts[p]:starts[p + 1]] for p in range(count)]
    radii = [rho[starts[p]:starts[p + 1]] for p in range(count)]
    reversedWall = False
    if container is not None:
        following = np.roll(loops[container], -1, axis = 0)
        signed = 0.5 * np.sum(loops[container][:, 0] * following[:, 1]
                              - following[:, 0] * loops[container][:, 1])
        if signed > 0.0:
            loops[container] = loops[container][::-1].copy()
            radii[container] = radii[container][::-1].copy()
            reversedWall = True

    energy = 0.0
    force = np.zeros_like(vertices)
    rhoForce = np.zeros_like(rho)

    bodies = [bodyFromBackbone(loops[p], radii[p]) for p in range(count)]
    gradients = [BodyGradient(bodies[p].count) for p in range(count)]
    for a, b in candidatePairs(bodies, exterior = container):
            isWall = container is not None and (a == container or b == container)
            if isWall:
                shape = b if a == container else a
                norm = 2.0 * targetArea[shape]
                strength = kContainer
            else:
                norm = targetArea[a] + targetArea[b]
                strength = kOverlap
            area, gradFirst, gradSecond = overlapAreaBodyGradient(
                bodies[a], bodies[b], spans(bodies[a], bodies[b]), spans(bodies[b], bodies[a]))
            if area == 0.0:
                continue
            energy += 2.0 * strength * (area / norm) ** 2
            weight = 4.0 * strength * area / (norm * norm)
            gradients[a].add(gradFirst, weight)
            gradients[b].add(gradSecond, weight)

    for p in range(count):
        gradLoop, gradRho = applyBodySensitivity(
            bodySensitivity(loops[p], radii[p]), gradients[p], len(loops[p]))
        if p == container and reversedWall:
            gradLoop, gradRho = gradLoop[::-1], gradRho[::-1]
        block = slice(starts[p], starts[p + 1])
        force[block] -= gradLoop
        rhoForce[block] -= gradRho
    return float(energy), force, rhoForce
