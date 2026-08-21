"""Polygon-polygon contact for simple polygons, convex or NOT -- vectorized port of the reference.

    E = 1/2 sum over ordered pairs (P, Q) of  int_{dP cap Q} (k/3) d_Q(x)^3 dl(x)

with ``d_Q`` the EXACT distance from x to ``dQ``. Everything -- closed forms, gradients, scaling,
regularity -- follows from that one expression. Ported from ``notes/files(3).zip``
``reference/polycontact_ref.py``; ``tests/polyContactCheck.py`` is the contract and the reference is
ground truth. The port must not change WHAT is computed, only how fast.

HOW THIS DIFFERS FROM softDepth, WHICH IT SUPERSEDES. softDepth measures depth by a log-sum-exp softmin
of half-plane distances, which is exact only for CONVEX loops and carries a regularizer ``epsilon``.
This law uses the exact distance to the boundary and needs no regularization length anywhere, because
integrating along ``dl`` buys back the derivative that a pointwise depth potential would lose at a
feature switch. Two of the things built for softDepth are explicitly REJECTED by the handoff, with
numbers:

  - vertex-sampled quadrature: blind to shallow face-on-face contact, and it INVERTS the face/vertex
    contrast (true ratio 2.00, reported 0.001) -- the bug found independently here on 2026-07-31;
  - convex decomposition: low by a factor of NINETEEN on a convex control case, because the distance to
    a piece's boundary is not the distance to the body's boundary. That is what ``convexDifference.py``
    does, and it is why this law handles nonconvexity with no decomposition at all.

CONVENTIONS ARE FIXED AND MUST NOT BE RE-DERIVED. Four sign errors during the reference's development
all came from re-deriving the outward normal.

    CCW vertices, signed area > 0
    edge j:            g_j = V[j+1] - V[j],   tau_j = g_j / |g_j|
    OUTWARD normal:    n_j = (tau_j.y, -tau_j.x)
    signed line dist:  ell_j(x) = n_j . (V[j] - x)       > 0 inside
    perpendicular foot on line j:   q = x + ell_j n_j    NOT x - ell_j n_j

THE VALIDITY LIMIT IS THE ONE HARD CONSTRAINT. ``d_B`` has a ridge -- the medial axis -- at roughly the
inradius from the boundary. Crossing it is not an accuracy loss but a SIGN REVERSAL: past the ridge the
leading edge's depth decreases and the bodies are pulled through. Assert
``max over overlap components of d_max / r_in << 1`` every step, and for a limbed shape ``r_in`` is the
LIMB half-width, not the particle size. ``validityRatio`` computes it.
"""

# UNVERIFIED(Cam)

import numpy as np


# UNVERIFIED(Cam)
def edgeFrame(loop):
    """``(edges, lengths, tangents, normals)`` for a CCW loop; ``normals`` are OUTWARD.

    The reference's ``edges()``. Kept as a single function so the normal convention lives in exactly
    one place -- see the module docstring on why that matters."""
    loop = np.asarray(loop, dtype = float)
    following = np.roll(loop, -1, axis = 0)
    vectors = following - loop
    lengths = np.linalg.norm(vectors, axis = 1)
    tangents = vectors / lengths[:, None]
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis = 1)
    return vectors, lengths, tangents, normals


def signedArea(loop):
    loop = np.asarray(loop, dtype = float)
    following = np.roll(loop, -1, axis = 0)
    return 0.5 * float(np.sum(loop[:, 0] * following[:, 1] - loop[:, 1] * following[:, 0]))


def makeCounterClockwise(loop):
    loop = np.asarray(loop, dtype = float)
    return loop if signedArea(loop) > 0.0 else loop[::-1].copy()


# UNVERIFIED(Cam)
def isReflex(loop):
    """Per-vertex reflex flag, vectorized.

    MUST BE RECOMPUTED EVERY STEP. ``membership`` depends on it, and a deforming body can flip a vertex
    convex<->reflex. Caching these behind a neighbor list silently inverts the inside/outside test for
    exterior points near a flipped vertex: a sign error in ``d`` with no crossing and nothing raised.
    It is one cross product per vertex, so there is nothing to save."""
    loop = np.asarray(loop, dtype = float)
    previous = np.roll(loop, 1, axis = 0)
    following = np.roll(loop, -1, axis = 0)
    back = loop - previous
    forward = following - loop
    return (back[:, 0] * forward[:, 1] - back[:, 1] * forward[:, 0]) < 0.0


# UNVERIFIED(Cam)
def insideParity(points, loop):
    """Ray-cast parity. Exact for any simple polygon, O(M), reference-only -- ``nearestFeature``
    delivers membership as a by-product, so production never needs this."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    loop = np.asarray(loop, dtype = float)
    following = np.roll(loop, -1, axis = 0)
    y = points[:, 1][:, None]
    x = points[:, 0][:, None]
    y0, y1 = loop[:, 1][None, :], following[:, 1][None, :]
    x0, x1 = loop[:, 0][None, :], following[:, 0][None, :]
    straddles = (y0 > y) != (y1 > y)
    with np.errstate(all = "ignore"):
        crossingX = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return np.sum(straddles & (x < crossingX), axis = 1) % 2 == 1


# UNVERIFIED(Cam)
def nearestFeature(points, loop, reflex = None, frame = None):
    """``(kind, index, distance, inside)`` for every point, against every feature of ``loop``.

    ``kind`` is 0 for an EDGE and 1 for a VERTEX. Vectorized over all points and all 2M candidates at
    once -- the reference loops, but the min-reduce is branch-free and this is the shape the GPU wants.

    MEMBERSHIP COMES FREE FROM THE SAME QUERY, no ray cast:

        nearest is edge j    -> inside iff n_j . (V[j] - x) > 0
        nearest is vertex j  -> inside iff V[j] is REFLEX

    The reflex clause is the whole content of the rule; a convex-only test passes it trivially.

    CANDIDATES ARE INTERLEAVED (E_0, V_0, E_1, V_1, ...) to match the reference's iteration order
    exactly. Ties are measure-zero for the distance, but the INSIDE flag can differ between a tied edge
    and a tied vertex, so the tie-break order is part of what is being ported."""
    points = np.atleast_2d(np.asarray(points, dtype = float))
    loop = np.asarray(loop, dtype = float)
    if reflex is None:
        reflex = isReflex(loop)
    _, lengths, tangents, normals = edgeFrame(loop) if frame is None else frame

    offset = points[:, None, :] - loop[None, :, :]
    foot = np.einsum("pjc,jc->pj", offset, tangents)
    signedLine = -np.einsum("pjc,jc->pj", offset, normals)
    withinEdge = (foot >= -1e-15) & (foot <= lengths[None, :] + 1e-15)
    edgeDistance = np.where(withinEdge, np.abs(signedLine), np.inf)
    vertexDistance = np.linalg.norm(offset, axis = 2)

    count = len(loop)
    distance = np.empty((len(points), 2 * count))
    distance[:, 0::2] = edgeDistance
    distance[:, 1::2] = vertexDistance
    winner = np.argmin(distance, axis = 1)

    isVertex = (winner % 2) == 1
    index = winner // 2
    rows = np.arange(len(points))
    inside = np.where(isVertex, reflex[index], signedLine[rows, index] > 0.0)
    return isVertex.astype(int), index, distance[rows, winner], inside


def realizingPoint(loop, kind, index, point):
    """The actual nearest POINT of a feature. Whether two tied features SHARE this point is what
    separates a benign C1 feature switch from the medial axis -- counting near-tied features does not
    distinguish them. Measured: benign switch, separation 0; medial axis, 0.32 apart (the full limb
    width) with the slope of ``d`` turning +1 -> -1."""
    loop = np.asarray(loop, dtype = float)
    point = np.asarray(point, dtype = float)
    if kind == 1:
        return loop[index].copy()
    _, _, _, normals = edgeFrame(loop)
    return point + (normals[index] @ (loop[index] - point)) * normals[index]


def signedDistance(points, loop, reflex = None):
    _, _, distance, inside = nearestFeature(points, loop, reflex)
    return np.where(inside, distance, -distance)


# UNVERIFIED(Cam)
def spans(loopA, loopB, frameA = None, frameB = None):
    """Maximal stretches of ``dA`` lying inside B, as ``(edge, t0, t1)`` arrays.

    Exact for any simple polygon: no convexity assumption, no decomposition, no root finding. Every
    edge of A is cut at its crossings with ``dB``, and each resulting sub-stretch is kept if its
    midpoint is inside B.

    Vectorized over the (edge of A) x (edge of B) crossing test; the per-edge sort stays in Python
    because the number of crossings is data-dependent and tiny."""
    loopA = np.asarray(loopA, dtype = float)
    loopB = np.asarray(loopB, dtype = float)
    frameA = edgeFrame(loopA) if frameA is None else frameA
    frameB = edgeFrame(loopB) if frameB is None else frameB
    vectorsA, vectorsB = frameA[0], frameB[0]

    # t along edge i of A, s along edge j of B, for every (i, j) at once.
    determinant = (vectorsA[:, None, 0] * vectorsB[None, :, 1]
                   - vectorsA[:, None, 1] * vectorsB[None, :, 0])
    offset = loopB[None, :, :] - loopA[:, None, :]
    with np.errstate(divide = "ignore", invalid = "ignore"):
        alongA = (offset[:, :, 0] * vectorsB[None, :, 1]
                  - offset[:, :, 1] * vectorsB[None, :, 0]) / determinant
        alongB = (offset[:, :, 0] * vectorsA[:, None, 1]
                  - offset[:, :, 1] * vectorsA[:, None, 0]) / determinant
    genuine = ((np.abs(determinant) >= 1e-14)
               & (alongA > 1e-12) & (alongA < 1 - 1e-12)
               & (alongB >= -1e-12) & (alongB <= 1 + 1e-12))

    edges, lower, upper, midpoints = [], [], [], []
    for i in range(len(loopA)):
        cuts = np.unique(np.concatenate([[0.0, 1.0], alongA[i][genuine[i]]]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            edges.append(i)
            lower.append(a)
            upper.append(b)
            midpoints.append(loopA[i] + 0.5 * (a + b) * vectorsA[i])
    if not edges:
        return np.zeros(0, dtype = int), np.zeros(0), np.zeros(0)

    _, _, _, inside = nearestFeature(np.asarray(midpoints), loopB, frame = frameB)
    keep = np.asarray(inside)
    return np.asarray(edges)[keep], np.asarray(lower)[keep], np.asarray(upper)[keep]


# UNVERIFIED(Cam)
def overlapArea(loopA, loopB):
    """Exact overlap area from the same span set, by Green's theorem: ``1/2 sum p x q`` over span
    segments. Free once ``spans`` is computed, and the handoff's recommended repair energy when a
    configuration has to be rescued -- ``1/2 k (area)^2`` drove a maximal-depth state to exactly zero
    overlap in ~120 iterations where the depth potential stalled, because the depth potential can lower
    itself by making overlaps shallower and WIDER."""
    total = 0.0
    for first, second in ((loopA, loopB), (loopB, loopA)):
        first = np.asarray(first, dtype = float)
        vectors, _, _, _ = edgeFrame(first)
        edges, lower, upper = spans(first, second)
        if not len(edges):
            continue
        start = first[edges] + lower[:, None] * vectors[edges]
        end = first[edges] + upper[:, None] * vectors[edges]
        total += float(np.sum(start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]))
    return 0.5 * total


# UNVERIFIED(Cam)
def maximumDepth(loopA, loopB, samples = 200):
    """Deepest penetration over the pair -- THE validity monitor.

    Assert ``maximumDepth / inradius << 1``. Past the medial axis the repulsion REVERSES: measured,
    ``dE/ddelta`` is positive up to ``delta = r_in`` and negative beyond, so the bodies are pulled
    through. The signature is persistent rather than transient -- crossed limbs report ``d/r_in ~ 1``
    for as long as they stay crossed -- so checking the converged state suffices for packing
    generation. For AQS it does NOT: there the trajectory is the observable and a pass-through injects
    a spurious irreversible event, i.e. contaminates the signal being measured."""
    best = 0.0
    fractions = np.linspace(0.0, 1.0, samples)
    for first, second in ((loopA, loopB), (loopB, loopA)):
        first = np.asarray(first, dtype = float)
        vectors, _, _, _ = edgeFrame(first)
        edges, lower, upper = spans(first, second)
        if not len(edges):
            continue
        parameter = lower[:, None] + fractions[None, :] * (upper - lower)[:, None]
        points = (first[edges][:, None, :]
                  + parameter[:, :, None] * vectors[edges][:, None, :]).reshape(-1, 2)
        _, _, distance, _ = nearestFeature(points, second)
        best = max(best, float(distance.max()))
    return best


# UNVERIFIED(Cam)
def inradius(loop, resolution = 400, probes = 4000):
    """Largest inscribed-circle radius, by grid probe. The denominator of the validity ratio.

    FOR A LIMBED SHAPE THIS IS THE LIMB HALF-WIDTH, not the particle size -- which is the whole reason
    the ratio is expressed against it."""
    loop = np.asarray(loop, dtype = float)
    low, high = loop.min(axis = 0), loop.max(axis = 0)
    grid = np.stack(np.meshgrid(np.linspace(low[0], high[0], resolution),
                                np.linspace(low[1], high[1], resolution)), axis = -1).reshape(-1, 2)
    interior = grid[insideParity(grid, loop)]
    if not len(interior):
        return 0.0
    interior = interior[::max(1, len(interior) // probes)]
    _, _, distance, _ = nearestFeature(interior, loop)
    return float(distance.max())


def validityRatio(loopA, loopB, samples = 200):
    """``maximumDepth / min(inradius)`` -- assert this is well below 1 every step."""
    smallest = min(inradius(loopA), inradius(loopB))
    return maximumDepth(loopA, loopB, samples) / max(smallest, 1e-300)


# UNVERIFIED(Cam)
def _quadraticCandidates(start, vector, loop, frame = None):
    """Squared distance from ``start + t vector`` to every feature is a QUADRATIC in t.

    Returns ``(a, b, c, windowLow, windowHigh, isVertex, index)``, all length 2M. Edge candidates carry
    a validity window from the affine foot parameter -- outside it the perpendicular foot leaves the
    segment and the vertex candidate takes over."""
    loop = np.asarray(loop, dtype = float)
    _, lengths, tangents, normals = edgeFrame(loop) if frame is None else frame
    count = len(loop)
    offset = loop - start

    # Edge candidates: d^2 = (al - m t)^2.
    alpha = np.einsum("jc,jc->j", normals, offset)
    slope = normals @ vector
    footStart = -np.einsum("jc,jc->j", tangents, offset)
    footRate = tangents @ vector
    with np.errstate(divide = "ignore", invalid = "ignore"):
        rootLow = (0.0 - footStart) / footRate
        rootHigh = (lengths - footStart) / footRate
    degenerate = np.abs(footRate) < 1e-14
    alwaysInside = degenerate & (footStart >= 0.0) & (footStart <= lengths)
    windowLow = np.where(degenerate, np.where(alwaysInside, -np.inf, 1.0),
                         np.minimum(rootLow, rootHigh))
    windowHigh = np.where(degenerate, np.where(alwaysInside, np.inf, 0.0),
                          np.maximum(rootLow, rootHigh))

    # Vertex candidates: d^2 = |start - V_j + t vector|^2.
    toVertex = start - loop
    squaredLength = float(vector @ vector)

    return (np.concatenate([slope * slope, np.full(count, squaredLength)]),
            np.concatenate([-2.0 * alpha * slope, 2.0 * (toVertex @ vector)]),
            np.concatenate([alpha * alpha, np.einsum("jc,jc->j", toVertex, toVertex)]),
            np.concatenate([windowLow, np.full(count, -np.inf)]),
            np.concatenate([windowHigh, np.full(count, np.inf)]),
            np.concatenate([np.zeros(count, int), np.ones(count, int)]),
            np.concatenate([np.arange(count), np.arange(count)]))


# UNVERIFIED(Cam)
def march(start, vector, loop, lower, upper, maximumSteps = 64, frame = None):
    """Single-chord ``marchBatch``. Returns ``(breakpoints, [], [])``.

    A THIN WRAPPER ON PURPOSE. The earlier standalone implementation -- and the reference's ``march``,
    which has the same structure -- SHRANK the candidate array after prefiltering and then searched for
    crossings only among the survivors. A pruned candidate can still become the winner later in the
    interval, so its crossing was never found and the switch was missed: measured, 5 genuine feature
    switches missed across 107 spans of a 9-body, 12-gon packing, shifting the total energy by 0.7%.

    ``marchBatch`` searches crossings against ALL candidates and only uses the prefilter to pick the
    winner. That can add a spurious breakpoint, which is harmless -- subdividing a stretch whose nearest
    feature is constant changes nothing -- but it cannot miss one.

    The reference's own suite cannot catch this: ``E_pair_closed`` partitions by bisection rather than
    by ``march``, and its march test compares only two hand-picked chords of one shape.

    The winner lists are empty because they were never trustworthy anyway (identifying the winner at
    ``t + 1e-13`` does not separate two candidates that cross shallowly). Callers re-identify features
    at sub-stretch midpoints."""
    frame = edgeFrame(loop) if frame is None else frame
    breakpoints = marchBatch(np.atleast_2d(start), np.atleast_2d(vector), loop,
                             np.atleast_1d(lower), np.atleast_1d(upper), frame, maximumSteps)[0]
    trimmed = [breakpoints[0]]
    for value in breakpoints[1:]:
        if value > trimmed[-1] + 1e-14:
            trimmed.append(value)
    if trimmed[-1] < upper - 1e-14:
        trimmed.append(upper)
    return np.asarray(trimmed), [], []


def _vertexAntiderivative(w, r):
    """``Phi_r(w) = int (w^2 + r^2)^(3/2) dw``, the vertex-nearest branch's exact integral."""
    s = np.hypot(w, r)
    return w * (2.0 * w * w + 5.0 * r * r) * s / 8.0 + 3.0 * r ** 4 * np.arcsinh(w / r) / 8.0


# UNVERIFIED(Cam)
def _substretches(loopA, loopB, frameA = None, frameB = None):
    """Every nearest-feature-constant sub-stretch of ``dA`` inside B, flattened.

    Returns ``(edgeOfA, lower, upper, kind, featureOfB, spanId)``. One array set for the whole pair, so
    the energy and the gradient can then work on all sub-stretches at once instead of nesting two Python
    loops. ``spanId`` is kept because the gradient's MEASURE term is per SPAN, not per sub-stretch."""
    loopA = np.asarray(loopA, dtype = float)
    # The two edge frames are constant for the whole pair, so they are built ONCE here and threaded
    # through spans, march and nearestFeature. They were being rebuilt ~1000 times per system energy
    # evaluation, which profiled at 18% of it.
    frameA = edgeFrame(loopA) if frameA is None else frameA
    frameB = edgeFrame(loopB) if frameB is None else frameB
    vectorsA = frameA[0]
    edges, spanLow, spanHigh = spans(loopA, loopB, frameA, frameB)
    if not len(edges):
        empty = np.zeros(0, dtype = int)
        return empty, np.zeros(0), np.zeros(0), empty, empty, empty

    cuts = marchBatch(loopA[edges], vectorsA[edges], loopB, spanLow, spanHigh, frameB)
    lowSide, highSide = cuts[:, :-1], cuts[:, 1:]
    wide = (highSide - lowSide) >= 1e-14
    spanOf, pieceOf = np.nonzero(wide)
    outEdge = edges[spanOf]
    outLow, outHigh = lowSide[spanOf, pieceOf], highSide[spanOf, pieceOf]
    outSpan = spanOf

    # THE WINNING FEATURE IS RE-IDENTIFIED AT THE MIDPOINT, NOT TAKEN FROM march.
    # march's breakpoints are exact, but its winner list is not reliable: it identifies the winner by
    # evaluating the candidate quadratics at t + 1e-13, and where two candidates cross shallowly that
    # offset does not separate them. Measured on crossed bars perturbed by 1e-5, march reported the
    # winners as (E3, E3) where the truth is (E3, E1) -- the breakpoint itself was correct to 1e-9 --
    # which inflated the energy EIGHTFOLD and showed up only as a finite-difference failure. The
    # reference has the same limitation and works around it the same way; its own march test recomputes
    # the features from midpoints rather than trusting them.
    midpoint = loopA[outEdge] + (0.5 * (outLow + outHigh))[:, None] * vectorsA[outEdge]
    kind, feature, _, _ = nearestFeature(midpoint, loopB, frame = frameB)
    return (outEdge, outLow, outHigh, np.asarray(kind, dtype = int),
            np.asarray(feature, dtype = int), np.asarray(outSpan, dtype = int))


# UNVERIFIED(Cam)
def pairEnergy(loopA, loopB, stiffness = 1.0):
    """``int_{dA cap B} (k/3) d_B^3 dl`` in CLOSED FORM. No quadrature anywhere.

    THE EDGE BRANCH IS POLYNOMIAL, DELIBERATELY. ``int d^3 dt`` has a compact antiderivative that
    divides by ``m = n_j . e_i``, and that vanishes EXACTLY for face-parallel contact -- the dominant
    configuration -- and cancels catastrophically near it. Expanding in the moments ``M_q`` instead
    gives a polynomial: no division, no branch, no tolerance.

    The vertex branch has no such issue and uses the exact antiderivative directly."""
    loopA = np.asarray(loopA, dtype = float)
    loopB = np.asarray(loopB, dtype = float)
    frameA, frameB = edgeFrame(loopA), edgeFrame(loopB)
    edges, low, high, kind, feature, _ = _substretches(loopA, loopB, frameA, frameB)
    if not len(edges):
        return 0.0
    vectorsA, lengthsA, tangentsA, _ = frameA
    normalsB = frameB[3]

    length = lengthsA[edges]
    moment0 = high - low
    moment1 = (high ** 2 - low ** 2) / 2.0
    moment2 = (high ** 3 - low ** 3) / 3.0
    moment3 = (high ** 4 - low ** 4) / 4.0
    integral = np.zeros(len(edges))

    onEdge = kind == 0
    if onEdge.any():
        j = feature[onEdge]
        alpha = np.einsum("jc,jc->j", normalsB[j], loopB[j] - loopA[edges[onEdge]])
        slope = np.einsum("jc,jc->j", normalsB[j], vectorsA[edges[onEdge]])
        integral[onEdge] = (alpha ** 3 * moment0[onEdge]
                            - 3.0 * alpha ** 2 * slope * moment1[onEdge]
                            + 3.0 * alpha * slope ** 2 * moment2[onEdge]
                            - slope ** 3 * moment3[onEdge])

    onVertex = kind == 1
    if onVertex.any():
        j = feature[onVertex]
        edge = edges[onVertex]
        span = length[onVertex]
        tangent = tangentsA[edge]
        toVertex = loopB[j] - loopA[edge]
        along = np.einsum("jc,jc->j", tangent, toVertex)
        foot = along / span
        perpendicular = -toVertex + along[:, None] * tangent
        radius = np.maximum(np.linalg.norm(perpendicular, axis = 1), 1e-14)
        wLow = span * (low[onVertex] - foot)
        wHigh = span * (high[onVertex] - foot)
        integral[onVertex] = (_vertexAntiderivative(wHigh, radius)
                              - _vertexAntiderivative(wLow, radius)) / span
    return float(np.sum(length * (stiffness / 3.0) * integral))


def contactEnergy(loopA, loopB, stiffness = 1.0):
    """Symmetrized contact energy for the ordered pair."""
    return 0.5 * (pairEnergy(loopA, loopB, stiffness) + pairEnergy(loopB, loopA, stiffness))


# UNVERIFIED(Cam)
def pairGradient(loopA, loopB, stiffness = 1.0):
    """``(energy, dE/dA, dE/dB)`` for ``int_{dA cap B} (k/3) d_B^3 dl``, all closed form.

    Leibniz gives three groups:

      MEASURE    ``dL_i/dv`` times the span's own integral -- per SPAN, not per sub-stretch.
      DOMAIN     IDENTICALLY ZERO here: ``phi(0) = 0`` at a crossing, and ``dt/dv = 0`` at a vertex.
      INTEGRAND  needs only ``P0 = int phi' dt`` and ``P1 = int phi' t dt``.

    Sub-stretch breakpoints contribute nothing either: ``d_B`` is continuous across a feature switch, so
    the two adjacent boundary terms cancel exactly. That is why the bisection locating a switch never
    has to be differentiated, and why no regularization length appears anywhere in this formulation --
    integrating along ``dl`` buys back the derivative that a pointwise depth potential loses.

    FOUR TRAPS, all of which pass the obvious test:

      T1  the integrand group carries the arclength factor ``L_i``. Omitting it is EXACT whenever the
          contacting edge has unit length, so a suite built on unit squares hides a 48% error.
      T2  the edge branch must use the POLYNOMIAL moment form, never the ``1/m`` antiderivative: ``m``
          vanishes exactly for face-parallel contact, the dominant configuration.
      T3  in the vertex branch the perpendicular offset enters as the FULL VECTOR ``rho``, undivided.
          Using the unit vector gives ~1e-3 relative errors that conservation does not catch.
      T4  whatever is evaluated for the energy is what must be differentiated. Pairing a fixed-node
          quadrature with this Leibniz formula gave ~1e-2 relative gradient errors.

    And T5: conservation is a nearly worthless test here. Net force and torque vanish STRUCTURALLY,
    independent of whether the terms are right -- they passed on every buggy intermediate, including one
    with a 48% error. Only finite differencing localizes anything."""
    loopA = np.asarray(loopA, dtype = float)
    loopB = np.asarray(loopB, dtype = float)
    gradientA = np.zeros_like(loopA)
    gradientB = np.zeros_like(loopB)
    frameA, frameB = edgeFrame(loopA), edgeFrame(loopB)
    edges, low, high, kind, feature, spanId = _substretches(loopA, loopB, frameA, frameB)
    if not len(edges):
        return 0.0, gradientA, gradientB

    countA, countB = len(loopA), len(loopB)
    vectorsA, lengthsA, tangentsA, _ = frameA
    _, lengthsB, tangentsB, normalsB = frameB

    length = lengthsA[edges]
    nextA = (edges + 1) % countA
    moment0 = high - low
    moment1 = (high ** 2 - low ** 2) / 2.0
    moment2 = (high ** 3 - low ** 3) / 3.0
    moment3 = (high ** 4 - low ** 4) / 4.0
    perStretch = np.zeros(len(edges))

    onEdge = kind == 0
    if onEdge.any():
        j = feature[onEdge]
        edge = edges[onEdge]
        span = length[onEdge]
        alpha = np.einsum("jc,jc->j", normalsB[j], loopB[j] - loopA[edge])
        slope = np.einsum("jc,jc->j", normalsB[j], vectorsA[edge])
        perStretch[onEdge] = stiffness * (alpha ** 3 * moment0[onEdge]
                                          - 3.0 * alpha ** 2 * slope * moment1[onEdge]
                                          + 3.0 * alpha * slope ** 2 * moment2[onEdge]
                                          - slope ** 3 * moment3[onEdge]) / 3.0
        first = stiffness * (alpha ** 2 * moment0[onEdge] - 2.0 * alpha * slope * moment1[onEdge]
                             + slope ** 2 * moment2[onEdge])
        second = stiffness * (alpha ** 2 * moment1[onEdge] - 2.0 * alpha * slope * moment2[onEdge]
                              + slope ** 2 * moment3[onEdge])
        toward = -normalsB[j]
        np.add.at(gradientA, edge, (span * (first - second))[:, None] * toward)
        np.add.at(gradientA, nextA[onEdge], (span * second)[:, None] * toward)
        footStart = np.einsum("jc,jc->j", tangentsB[j], loopA[edge] - loopB[j]) / lengthsB[j]
        footRate = np.einsum("jc,jc->j", tangentsB[j], vectorsA[edge]) / lengthsB[j]
        shared = footStart * first + footRate * second
        np.add.at(gradientB, j, (span * (first - shared))[:, None] * normalsB[j])
        np.add.at(gradientB, (j + 1) % countB, (span * shared)[:, None] * normalsB[j])

    onVertex = kind == 1
    if onVertex.any():
        j = feature[onVertex]
        edge = edges[onVertex]
        span = length[onVertex]
        tangent = tangentsA[edge]
        toVertex = loopB[j] - loopA[edge]
        along = np.einsum("jc,jc->j", tangent, toVertex)
        foot = along / span
        # T3: the FULL perpendicular vector, not its direction.
        perpendicular = -toVertex + along[:, None] * tangent
        radius = np.maximum(np.linalg.norm(perpendicular, axis = 1), 1e-14)
        wLow = span * (low[onVertex] - foot)
        wHigh = span * (high[onVertex] - foot)
        sLow, sHigh = np.hypot(wLow, radius), np.hypot(wHigh, radius)

        deltaJ0 = ((wHigh * sHigh + radius ** 2 * np.arcsinh(wHigh / radius))
                   - (wLow * sLow + radius ** 2 * np.arcsinh(wLow / radius))) / 2.0
        deltaJ1 = (sHigh ** 3 - sLow ** 3) / 3.0
        deltaJ2 = ((wHigh * sHigh ** 3 / 4.0 - radius ** 2 * wHigh * sHigh / 8.0
                    - radius ** 4 * np.arcsinh(wHigh / radius) / 8.0)
                   - (wLow * sLow ** 3 / 4.0 - radius ** 2 * wLow * sLow / 8.0
                      - radius ** 4 * np.arcsinh(wLow / radius) / 8.0))
        # The k/3 belongs here exactly as it does in the edge branch above. Without it a vertex-nearest
        # sub-stretch contributes three times its energy to both the returned energy and the MEASURE
        # group, and NOTHING in a convex-only suite notices: vertex-nearest stretches arise only from
        # REFLEX vertices of loopB, so the branch is dead code until a nonconvex obstacle appears.
        perStretch[onVertex] = stiffness * (_vertexAntiderivative(wHigh, radius)
                                            - _vertexAntiderivative(wLow, radius)) / (3.0 * span)

        firstVector = stiffness * (tangent * deltaJ1[:, None]
                                   + perpendicular * deltaJ0[:, None]) / span[:, None]
        secondVector = stiffness * (foot[:, None] * (tangent * deltaJ1[:, None]
                                                     + perpendicular * deltaJ0[:, None])
                                    + (tangent * deltaJ2[:, None]
                                       + perpendicular * deltaJ1[:, None]) / span[:, None]
                                    ) / span[:, None]
        np.add.at(gradientA, edge, span[:, None] * (firstVector - secondVector))
        np.add.at(gradientA, nextA[onVertex], span[:, None] * secondVector)
        np.add.at(gradientB, j, -span[:, None] * firstVector)

    # MEASURE term, per span: dL_i/dv times that span's whole integral.
    spanCount = int(spanId.max()) + 1
    spanIntegral = np.zeros(spanCount)
    np.add.at(spanIntegral, spanId, perStretch)
    spanEdge = np.zeros(spanCount, dtype = int)
    spanEdge[spanId] = edges
    spanTangent = tangentsA[spanEdge]
    np.add.at(gradientA, (spanEdge + 1) % countA, spanIntegral[:, None] * spanTangent)
    np.add.at(gradientA, spanEdge, -spanIntegral[:, None] * spanTangent)

    energy = float(np.sum(lengthsA[spanEdge] * spanIntegral))
    return energy, gradientA, gradientB


def contactGradient(loopA, loopB, stiffness = 1.0):
    """Symmetrized energy and gradients for the ordered pair."""
    energyOne, gradientOneA, gradientOneB = pairGradient(loopA, loopB, stiffness)
    energyTwo, gradientTwoB, gradientTwoA = pairGradient(loopB, loopA, stiffness)
    return (0.5 * (energyOne + energyTwo),
            0.5 * (gradientOneA + gradientTwoA),
            0.5 * (gradientOneB + gradientTwoB))


# UNVERIFIED(Cam)
def _batchQuadraticCandidates(starts, vectors, loop, frame):
    """``_quadraticCandidates`` for MANY chords at once: every array is (S, 2M).

    Same quantities, same order (edge candidates then vertex candidates), just carrying a span axis."""
    loop = np.asarray(loop, dtype = float)
    _, lengths, tangents, normals = frame
    count = len(loop)

    offset = loop[None, :, :] - starts[:, None, :]
    alpha = np.einsum("sjc,jc->sj", offset, normals)
    slope = vectors @ normals.T
    footStart = -np.einsum("sjc,jc->sj", offset, tangents)
    footRate = vectors @ tangents.T
    with np.errstate(divide = "ignore", invalid = "ignore"):
        rootLow = (0.0 - footStart) / footRate
        rootHigh = (lengths[None, :] - footStart) / footRate
    degenerate = np.abs(footRate) < 1e-14
    alwaysInside = degenerate & (footStart >= 0.0) & (footStart <= lengths[None, :])
    edgeLow = np.where(degenerate, np.where(alwaysInside, -np.inf, 1.0),
                       np.minimum(rootLow, rootHigh))
    edgeHigh = np.where(degenerate, np.where(alwaysInside, np.inf, 0.0),
                        np.maximum(rootLow, rootHigh))

    toVertex = -offset
    squared = np.einsum("sc,sc->s", vectors, vectors)[:, None]
    infLow = np.full((len(starts), count), -np.inf)
    infHigh = np.full((len(starts), count), np.inf)
    return (np.concatenate([slope * slope, np.broadcast_to(squared, (len(starts), count))], axis = 1),
            np.concatenate([-2.0 * alpha * slope, 2.0 * np.einsum("sjc,sc->sj", toVertex, vectors)],
                           axis = 1),
            np.concatenate([alpha * alpha, np.einsum("sjc,sjc->sj", toVertex, toVertex)], axis = 1),
            np.concatenate([edgeLow, infLow], axis = 1),
            np.concatenate([edgeHigh, infHigh], axis = 1))


# UNVERIFIED(Cam)
def marchBatch(starts, vectors, loop, lowers, uppers, frame, maximumSteps = 64):
    """``march`` over MANY chords simultaneously; returns a (S, maximumSteps+1) breakpoint array.

    Rows are padded with each span's own ``upper``, so a zero-width trailing sub-stretch is discarded
    downstream and no span needs its own loop trip count.

    THE WALK IS SEQUENTIAL IN t BUT THE SPANS ARE INDEPENDENT, so they step in lockstep behind an
    active mask instead of being marched one at a time in Python. Profiled at 296 separate ``march``
    calls per system energy evaluation and ~54% of it, all interpreter overhead on small arrays; the
    arithmetic per step is unchanged. This is also the shape the GPU wants -- one thread per span.

    The winner list is deliberately NOT returned. It is unreliable near a shallow crossing (identifying
    at ``t + 1e-13`` does not separate two candidates that cross there), and the caller re-identifies
    features at sub-stretch midpoints. See ``_substretches``."""
    starts = np.asarray(starts, dtype = float)
    vectors = np.asarray(vectors, dtype = float)
    lowers = np.asarray(lowers, dtype = float)
    uppers = np.asarray(uppers, dtype = float)
    spanCount = len(starts)
    a, b, c, windowLow, windowHigh = _batchQuadraticCandidates(starts, vectors, loop, frame)

    # NO PREFILTER. The reference prunes candidates whose minimum over the interval exceeds the best
    # achievable maximum, and the first version here copied that. It is WRONG: the pruning bound is
    # computed on each candidate's window CLIPPED to the interval, which understates the true maximum
    # of a candidate that is invalid over part of it, so the bound comes out too tight and genuine
    # winners get pruned. The winner is then wrong, the crossings are computed for the wrong curve, and
    # the switch is missed -- measured, 23 genuine switches missed over 120 random chords, and 5 over
    # 107 spans of a real packing, shifting that packing's energy by 0.7%.
    #
    # It also bought nothing here. In the batched form the arrays stay full-size and the prefilter is
    # only a mask, so dropping it costs no arithmetic. The window test in `inWindow` below is the only
    # restriction needed, and it is exact.

    breakpoints = np.repeat(uppers[:, None], maximumSteps + 1, axis = 1)
    breakpoints[:, 0] = lowers
    at = lowers.copy()
    rows = np.arange(spanCount)
    active = np.ones(spanCount, dtype = bool)

    for step in range(maximumSteps):
        probe = (at + 1e-13)[:, None]
        inWindow = (windowLow - 1e-15 <= probe) & (probe <= windowHigh + 1e-15)
        value = np.where(inWindow, a * probe * probe + b * probe + c, np.inf)
        winner = np.argmin(value, axis = 1)

        deltaA = a[rows, winner][:, None] - a
        deltaB = b[rows, winner][:, None] - b
        deltaC = c[rows, winner][:, None] - c
        linear = np.abs(deltaA) < 1e-14
        with np.errstate(divide = "ignore", invalid = "ignore"):
            straight = np.where(linear & (np.abs(deltaB) > 1e-14), -deltaC / deltaB, np.inf)
            discriminant = deltaB * deltaB - 4.0 * deltaA * deltaC
            root = np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
            plus = np.where(~linear, (-deltaB + root) / (2.0 * deltaA), np.inf)
            minus = np.where(~linear, (-deltaB - root) / (2.0 * deltaA), np.inf)
        candidate = np.concatenate([straight, plus, minus,
                                    windowLow, windowHigh], axis = 1)
        candidate = np.where(np.isfinite(candidate), candidate, np.inf)
        ahead = (candidate > (at + 1e-12)[:, None]) & (candidate < uppers[:, None])
        nextAt = np.where(ahead, candidate, np.inf).min(axis = 1)
        nextAt = np.where(np.isfinite(nextAt), nextAt, uppers)

        at = np.where(active, np.minimum(nextAt, uppers), at)
        breakpoints[:, step + 1] = np.where(active, at, uppers)
        active &= nextAt < uppers - 1e-15
        if not active.any():
            break
    return breakpoints
