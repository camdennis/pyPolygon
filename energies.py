"""Overlap and self-avoidance energies for the sharp / mollified polygon model.

Three interaction energies, all driven off a Packing:
  - the SHARP overlap area + gradient (polygonPairIntersections ... updateOverlapGradient), the
    exact geometric overlap and the sigma -> 0 reference;
  - the PLUMMER mollified overlap (softened-log kernel): the fully-analytic, closed-form
    (no-quadrature) tier plummerOverlapExact -- C-infinity in the vertices -- plus its analytic
    Hessian plummerOverlapHessian;
  - the intra-polygon edge-edge SELF-REPULSION barrier that keeps each loop simple (so the overlap
    reduction stays valid).
(Merged from polygonOverlap.py + plummerOverlap.py + selfRepulsion.py.)
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from packing import minImageShift
_IMAGES = [(dx, dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)]
_NO_IMAGES = [(0.0, 0.0)]
# Default for updateIntersections' ``candidates``: "use whatever the Model attached, else all-to-all".
# A sentinel rather than None, so that passing None EXPLICITLY still means all-to-all -- that is how the
# reference path is requested, and the neighbor tests compare against it.
_ATTACHED = object()


def imagesFor(box):
    """Periodic image offsets to sum a pair interaction over: the 3x3 neighborhood for a periodic
    box, and the single (0, 0) self-image in FREE SPACE.

    Must be consulted rather than assuming ``_IMAGES``. ``minImageShift`` already returns zero when
    ``box is None``, but the image loop that follows is a separate thing: summing the 3x3 offsets in
    free space invents copies of every polygon one unit away, and two shapes exactly one unit apart
    then land on top of each other. That produced a spurious E = 0.41 between two squares that do not
    touch."""
    return _IMAGES if box is not None else _NO_IMAGES

# ---------------------------------------------------------------------------
# SHARP overlap: area + gradient (from polygonOverlap.py)
# ---------------------------------------------------------------------------

def polygonPairIntersections(rA, rB):
    """Edge-edge intersections
    sA = cross(w, d2) / cross(d1, d2),   sB = cross(w, d1) / cross(d1, d2),   w = rB[b] - rA[a],
    a genuine intersection needs 0 < sA < 1 and 0 < sB < 1 -- the two edge indices and the fraction along each;"""
    rA = np.asarray(rA, dtype = float)
    rB = np.asarray(rB, dtype = float)
    nA, nB = len(rA), len(rB)
    intersections = []
    for a in range(nA):
        p0 = rA[a]
        # difference for A
        dA = rA[(a + 1) % nA] - p0
        for b in range(nB):
            q0 = rB[b]
            dB = rB[(b + 1) % nB] - q0
            denom = dA[0] * dB[1] - dA[1] * dB[0]
            if abs(denom) < 1e-14:
                continue
            w = q0 - p0
            sA = (w[0] * dB[1] - w[1] * dB[0]) / denom
            sB = (w[0] * dA[1] - w[1] * dA[0]) / denom
            if 0.0 < sA < 1.0 and 0.0 < sB < 1.0:
                intersections.append((a, b, sA, sB))
    return intersections

def pointInPolygon(point, loop):
    """True if point, v, is inside the closed polygon loop (n, 2), by even-odd ray casting."""
    x, y = point[0], point[1]
    loop = np.asarray(loop, dtype = float)
    n = len(loop)
    inside = False
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xCross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xCross:
                inside = not inside
    return inside

def updateIntersections(packing, candidates = _ATTACHED):
    """Every edge-edge intersection in the packing, stored in ``packing.intersections`` as an (M, 6)
    float array with columns ``(rho_i, rho_j, edge_j, s_j, edge_i, s_i)``.

    ``candidates`` is an ``(edgeI, edgeJ)`` pair of global edge-index arrays from
    ``neighbors.NeighborList`` -- the edges whose bounding balls overlap. Given one, only those pairs
    are tested and the whole thing runs in a single vectorized pass.

    THREE WAYS TO CALL IT, and the distinction matters for the tests. Left at its default, it uses
    ``packing.candidatePairs`` if a Model has attached one and falls back to all-to-all if not, so the
    call sites here need no change. Passed a list explicitly, it uses that. Passed ``None``
    explicitly, it runs ALL-TO-ALL -- which is how the reference is requested, and why the default is
    a sentinel rather than None.

    THE ALL-TO-ALL PATH IS KEPT DELIBERATELY, not left behind. It is the reference the candidate path
    is tested against, and the comparison is exact set equality rather than a tolerance: a dropped
    candidate pair does not crash, it silently under-reports overlap, which makes an invalid packing
    look valid and corrupts the density a sweep reports. See ``tests/neighborCheck.py``."""
    if candidates is _ATTACHED:
        candidates = getattr(packing, "candidatePairs", None)
    if candidates is not None:
        return _intersectionsFromCandidates(packing, candidates)
    r = packing.positions.reshape(-1, 2)
    starts = packing.startIndices
    box = packing.box
    packing.intersections = []
    for polyA in range(packing.numPolygons):
        rA = r[starts[polyA] : starts[polyA + 1]]
        for polyB in range(polyA + 1, packing.numPolygons):
            shift = minImageShift(r[starts[polyB]] - r[starts[polyA]], box)
            rB = r[starts[polyB] : starts[polyB + 1]] + shift
            for (a, b, sA, sB) in polygonPairIntersections(rA, rB):
                eA = rA[(a + 1) % len(rA)] - rA[a]
                eB = rB[(b + 1) % len(rB)] - rB[b]
                edgeA = int(starts[polyA]) + a
                edgeB = int(starts[polyB]) + b
                if eB[0] * eA[1] - eB[1] * eA[0] > 0.0:
                    packing.intersections.append((polyA, polyB, edgeB, sB, edgeA, sA))
                else:
                    packing.intersections.append((polyB, polyA, edgeA, sA, edgeB, sB))
    if packing.intersections:
        packing.intersections = np.array(packing.intersections, dtype = float)
    else:
        packing.intersections = np.zeros((0, 6))


# UNVERIFIED(Cam)
def _intersectionsFromCandidates(packing, candidates):
    """The same intersection set as the all-to-all scan, tested only on candidate edge pairs and
    computed in ONE vectorized pass rather than a triple Python loop.

    Two speedups compound here and they are worth separating. The candidate list removes edge pairs
    that cannot cross; vectorizing removes the per-pair interpreter overhead from the ones that remain.
    Even with every pair a candidate, this is far faster than the loop it replaces.

    The line-line solve is the same one ``polygonPairIntersections`` does, with the same acceptance
    ``|denom| > 1e-14`` and strict ``0 < s < 1``, so a flush or parallel pair is treated identically.
    Orientation decides which polygon is the LEAVE polygon, exactly as in the reference: the pair is
    emitted with A leaving when ``cross(e_B, e_A) > 0`` and with B leaving otherwise."""
    edgeI, edgeJ = candidates
    if edgeI.size == 0:
        packing.intersections = np.zeros((0, 6))
        return
    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    nextIndex = np.asarray(packing.next, dtype = int)
    shapeId = np.asarray(packing.shapeId, dtype = int)
    box = packing.box

    shapeA, shapeB = shapeId[edgeI], shapeId[edgeJ]
    # One shift per PAIR, taken from the polygons' first vertices -- the single-image assumption the
    # rest of the pipeline is built on, and what the all-to-all loop does per polygon pair.
    shift = minImageShift(r[starts[shapeB]] - r[starts[shapeA]], box)

    p0 = r[edgeI]
    d1 = r[nextIndex[edgeI]] - p0
    q0 = r[edgeJ] + shift
    d2 = r[nextIndex[edgeJ]] - r[edgeJ]

    denom = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    good = np.abs(denom) >= 1e-14
    safe = np.where(good, denom, 1.0)
    w = q0 - p0
    sA = (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / safe
    sB = (w[:, 0] * d1[:, 1] - w[:, 1] * d1[:, 0]) / safe
    hit = good & (sA > 0.0) & (sA < 1.0) & (sB > 0.0) & (sB < 1.0)
    if not hit.any():
        packing.intersections = np.zeros((0, 6))
        return

    iHit, jHit = edgeI[hit], edgeJ[hit]
    aHit, bHit = shapeA[hit], shapeB[hit]
    sAHit, sBHit = sA[hit], sB[hit]
    eA, eB = d1[hit], d2[hit]
    aLeaves = (eB[:, 0] * eA[:, 1] - eB[:, 1] * eA[:, 0]) > 0.0

    rows = np.empty((iHit.size, 6), dtype = float)
    rows[:, 0] = np.where(aLeaves, aHit, bHit)
    rows[:, 1] = np.where(aLeaves, bHit, aHit)
    rows[:, 2] = np.where(aLeaves, jHit, iHit)
    rows[:, 3] = np.where(aLeaves, sBHit, sAHit)
    rows[:, 4] = np.where(aLeaves, iHit, jHit)
    rows[:, 5] = np.where(aLeaves, sAHit, sBHit)
    packing.intersections = rows

def updateFollowers(packing):
    """For every intersection, the array index of its FOLLOWER -- the next intersection along the
    CCW overlap boundary (notes sec 5) -- stored in packing.followerIndices (M,). At
    intersection alpha we leave along edge i on polygon rho_i (continuous coord i + s_i); the
    follower is the intersection beta of the SAME polygon pair with the polygons swapped
    (rho_i[beta] = rho_j[alpha], rho_j[beta] = rho_i[alpha]) whose arrival edge on rho_i (its j
    field, l + s_l) is the next one forward from i. Requires ``updateIntersections`` first.

    SORTED, NOT AN O(M^2) SCAN. The intersections are ordered by the pair key ``(rho_i, rho_j)``, so
    the partners of a given alpha -- which are exactly the entries with the pair SWAPPED -- form one
    contiguous block, found by binary search instead of by scanning every other intersection. This is
    the scheme the CUDA ``followersKernel`` already runs, ported so the two tiers share a complexity
    rather than only an answer: the device was O(M log M) while this was O(M^2).

    Only the block search is shared work; the minimal-forward-distance choice inside a block is still a
    scan, but a block holds only the crossings of one ordered polygon pair, which is a handful."""
    X = packing.intersections
    starts = np.asarray(packing.startIndices, dtype = int)
    M = len(X)
    if M == 0:
        packing.followerIndices = np.zeros(0, dtype = int)
        return

    rhoI = X[:, 0].astype(int)
    rhoJ = X[:, 1].astype(int)
    numPoly = int(packing.numPolygons)
    # One integer per ordered pair, so a sort groups every (rho_i, rho_j) block together.
    key = rhoI * numPoly + rhoJ
    order = np.argsort(key, kind = "stable")
    sortedKey = key[order]

    nEdges = starts[rhoI + 1] - starts[rhoI]
    cLeave = X[:, 4] + X[:, 5] - starts[rhoI]
    cArrive = X[:, 2] + X[:, 3] - starts[rhoJ]

    followerIndices = np.full(M, -1, dtype = int)
    swapped = rhoJ * numPoly + rhoI
    lo = np.searchsorted(sortedKey, swapped, side = "left")
    hi = np.searchsorted(sortedKey, swapped, side = "right")
    for alpha in range(M):
        best, bestDist = -1, np.inf
        for slot in range(lo[alpha], hi[alpha]):
            beta = order[slot]
            # Beta arrives on alpha's LEAVE polygon, so its arrival coordinate is measured there --
            # which is beta's own rho_j, the swapped pair being what selected this block.
            dist = (cArrive[beta] - cLeave[alpha]) % nEdges[alpha]
            if 0.0 < dist < bestDist:
                best, bestDist = beta, dist
        followerIndices[alpha] = best
    packing.followerIndices = followerIndices

def _h(P, Q, R):
    """Chord-triangle area contribution h(P, Q) = 1/2 (P - R) x (Q - R) (notes eq 3.2)."""
    return 0.5 * ((P[0] - R[0]) * (Q[1] - R[1]) - (P[1] - R[1]) * (Q[0] - R[0]))

def updateOverlapArea(packing):
    """Total overlap area U = U_ex + U_int (notes eq 3.5) from packing.intersections and
    packing.followerIndices, stored in packing.overlapArea (and the split in packing.uEx/uInt). For
    each intersection alpha (leaving along edge i on polygon rho_i) and its follower beta (arriving
    on rho_i along edge l), the run from q_i to q_l contributes: a single chord h(q_i, q_l) if the
    two share an edge, else two end-cap chords h(q_i, v_{z(i)}) + h(v_l, q_l) (into U_ex) plus every
    full interior edge h(v_m, v_{z(m)}) between them (into U_int). h uses R = the lowest-indexed
    vertex of the lower-indexed polygon of the pair; the higher-indexed polygon is min-image shifted
    into the lower's image (matching updateIntersections). Requires updateIntersections +
    updateFollowers."""
    X = packing.intersections
    fol = packing.followerIndices
    r = packing.positions.reshape(-1, 2)
    starts = packing.startIndices
    nxt = packing.next
    box = packing.box
    shapeId = packing.shapeId
    # PER PAIR as well as in total. The total alone is enough only for an energy LINEAR in area; the
    # model's contact law is the normalized square U = 2 sum (a_AB / norm_AB)^2, which needs each
    # pair's own area because the normalizer is per pair. Every intersection already carries its
    # polygon pair in columns 0 and 1, so this costs a dictionary update.
    perPair = {}
    # U_ex: parallel over intersections -- the end caps of each run (or its single chord).
    AEx = 0.0
    for alpha in range(len(X)):
        beta = int(fol[alpha])
        rhoI, rhoJ = int(X[alpha, 0]), int(X[alpha, 1])
        R = r[int(starts[min(rhoI, rhoJ)])]
        shift = minImageShift(r[int(starts[rhoI])] - R, box)
        edgeI = int(X[alpha, 4]); sI = X[alpha, 5]
        edgeL = int(X[beta, 2]);  sL = X[beta, 3]
        qI = r[edgeI] + shift + sI * (r[nxt[edgeI]] - r[edgeI])
        qL = r[edgeL] + shift + sL * (r[nxt[edgeL]] - r[edgeL])
        key = (min(rhoI, rhoJ), max(rhoI, rhoJ))
        if edgeI == edgeL and sI <= sL:
            contribution = _h(qI, qL, R)
        else:
            contribution = _h(qI, r[nxt[edgeI]] + shift, R) + _h(r[edgeL] + shift, qL, R)
        AEx += contribution
        perPair[key] = perPair.get(key, 0.0) + contribution
    # U_int: parallel over edges -- edge m contributes once for each overlap it is fully interior
    # to (delta_pm = 1), i.e. each run on m's polygon whose leave->arrive span strictly contains m.
    AInt = 0.0
    for m in range(packing.numVertices):
        rho = int(shapeId[m])
        nEdges = int(starts[rho + 1]) - int(starts[rho])
        for alpha in range(len(X)):
            if int(X[alpha, 0]) != rho:
                continue
            beta = int(fol[alpha])
            edgeI = int(X[alpha, 4]); sI = X[alpha, 5]
            edgeL = int(X[beta, 2]);  sL = X[beta, 3]
            if edgeI == edgeL:
                span = 0 if sI <= sL else nEdges
            else:
                span = (edgeL - edgeI) % nEdges
            if 0 < (m - edgeI) % nEdges < span:
                rhoJ = int(X[alpha, 1])
                R = r[int(starts[min(rho, rhoJ)])]
                shift = minImageShift(r[int(starts[rho])] - R, box)
                contribution = _h(r[m] + shift, r[nxt[m]] + shift, R)
                AInt += contribution
                key = (min(rho, rhoJ), max(rho, rhoJ))
                perPair[key] = perPair.get(key, 0.0) + contribution
    packing.overlapArea = AEx + AInt
    packing.pairOverlapArea = perPair

def updateOverlapGradient(packing, pairWeight = None):
    """Gradient of the overlap area, dA_cap / dv_k (notes eq 4.3), stored in packing.overlapForce
    (numVertices, 2); the physical force on a vertex is its negative. Each overlap-boundary segment
    on edge m from lambda_0 to lambda_f (sbar = (l0+lf)/2, ds = lf-l0) deposits ds*(1-sbar)*(eps e_m)
    on vertex m and ds*sbar*(eps e_m) on vertex z(m), with (eps e_m) = (e_m^y, -e_m^x). The segments
    are exactly the end caps (A_ex) and full interior edges (A_int) of updateOverlapArea, so no
    reference point or min-image shift is needed -- eq 4.3 depends only on edge vectors and the s
    parameters. Requires updateIntersections + updateFollowers."""
    X = packing.intersections
    fol = packing.followerIndices
    r = packing.positions.reshape(-1, 2)
    starts = packing.startIndices
    nxt = packing.next
    shapeId = packing.shapeId
    grad = np.zeros((packing.numVertices, 2))

    def weightOf(rhoI, rhoJ):
        """Chain-rule factor dU/da for this pair, or 1 for the bare area gradient."""
        if pairWeight is None:
            return 1.0
        return float(pairWeight.get((min(rhoI, rhoJ), max(rhoI, rhoJ)), 0.0))

    def deposit(edge, l0, lf, weight):
        e = r[nxt[edge]] - r[edge]
        epsE = np.array([e[1], -e[0]])
        sbar = 0.5 * (lf + l0)
        ds = lf - l0
        grad[edge] += weight * (1.0 - sbar) * ds * epsE
        grad[nxt[edge]] += weight * sbar * ds * epsE

    # end-cap segments -- parallel over intersections (mirror of A_ex)
    for alpha in range(len(X)):
        beta = int(fol[alpha])
        edgeI = int(X[alpha, 4]); sI = X[alpha, 5]
        edgeL = int(X[beta, 2]);  sL = X[beta, 3]
        weight = weightOf(int(X[alpha, 0]), int(X[alpha, 1]))
        if edgeI == edgeL and sI <= sL:
            deposit(edgeI, sI, sL, weight)
        else:
            deposit(edgeI, sI, 1.0, weight)
            deposit(edgeL, 0.0, sL, weight)

    # full interior edges -- parallel over edges (mirror of A_int)
    for m in range(packing.numVertices):
        rho = int(shapeId[m])
        nEdges = int(starts[rho + 1]) - int(starts[rho])
        for alpha in range(len(X)):
            if int(X[alpha, 0]) != rho:
                continue
            beta = int(fol[alpha])
            edgeI = int(X[alpha, 4]); sI = X[alpha, 5]
            edgeL = int(X[beta, 2]);  sL = X[beta, 3]
            if edgeI == edgeL:
                span = 0 if sI <= sL else nEdges
            else:
                span = (edgeL - edgeI) % nEdges
            if 0 < (m - edgeI) % nEdges < span:
                deposit(m, 0.0, 1.0, weightOf(rho, int(X[alpha, 1])))
    packing.overlapGradient = grad


# UNVERIFIED(Cam)
def overlapAreaEnergyForce(packing, kOverlap = 1.0):
    """Total overlap AREA and its gradient -- the raw geometric measure, LINEAR in area.

    Not the model's contact law; kept because the bare area is the right quantity for asking whether a
    packing is valid at all (it is identically zero below jamming) and for the container complement.
    For the contact law use ``sharpOverlapEnergyForce``."""
    updateIntersections(packing)
    updateFollowers(packing)
    updateOverlapArea(packing)
    updateOverlapGradient(packing)
    return kOverlap * packing.overlapArea, -kOverlap * packing.overlapGradient


# UNVERIFIED(Cam)
def sharpOverlapEnergyForce(packing, kOverlap = 1.0):
    """Whole-packing SHARP (unmollified) overlap energy and vertex FORCE -dU/dv, in the model's
    NORMALIZED-SQUARED contact law -- the same functional the mollified tier uses:

        U = 2 kOverlap sum_{A<B} (a_AB / norm_AB)^2,     norm_AB = targetArea[A] + targetArea[B]

    so dU/dv = 4 kOverlap sum (a_AB / norm_AB^2) da_AB/dv.

    This used to return ``kOverlap * (total overlap area)`` -- linear in area, a DIFFERENT functional
    from the mollified tier. The two then could not be compared: an exponent measured on the sharp
    energy was the exponent of the overlap AREA, while the model's own energy scales as area squared,
    so every sharp-vs-mollified scaling comparison was off by a factor of two in the exponent. The
    sigma -> 0 reference has to be the sigma -> 0 limit of the SAME contact law, not of a different one.

    The per-pair normalizer is why ``updateOverlapArea`` now accumulates per pair: a total alone
    cannot be normalized pair-by-pair. Container pairs are excluded -- the wall is penalised by the
    area OUTSIDE it, with its own normalizer (see ``sharpContainerEnergyForce``).

    Returns ``(energy, force)`` with force shape (numVertices, 2)."""
    updateIntersections(packing)
    updateFollowers(packing)
    updateOverlapArea(packing)
    targetArea = np.asarray(packing.targetArea, dtype = float)
    container = getattr(packing, "containerIndex", None)
    energy = 0.0
    weights = {}
    for (A, B), a in packing.pairOverlapArea.items():
        if container is not None and (A == container or B == container):
            continue
        norm = targetArea[A] + targetArea[B]
        energy += 2.0 * kOverlap * (a / norm) ** 2
        weights[(A, B)] = 4.0 * kOverlap * a / (norm * norm)
    updateOverlapGradient(packing, pairWeight = weights)
    return energy, -packing.overlapGradient


# ---------------------------------------------------------------------------
# PLUMMER mollified overlap (from plummerOverlap.py)
# ---------------------------------------------------------------------------


def _edges(loop):
    """(v0, edgeVector, length, outward-unit-normal) for each edge of a closed CCW loop (n, 2)."""
    v0 = np.asarray(loop, dtype = float)
    e = np.roll(v0, -1, axis = 0) - v0
    length = np.hypot(e[:, 0], e[:, 1])
    tau = e / length[:, None]
    n = np.stack([tau[:, 1], -tau[:, 0]], axis = 1)
    return v0, e, length, n

def plummerMeasure(Y, loopB, sigma):
    """Psi_B(y) = int_B K_sigma(y - z) dz for a batch of points ``Y`` (P, 2), in closed form.

    Reducing int_B K dz = oint_{dB} F(z - y) . n_B ds and noting (z - y) . n_B = w . n_B is CONSTANT
    along an edge (w = v0 - y, since the edge tangent is perpendicular to its own normal), each edge
    leaves int_0^1 dt / (a t^2 + b t + c) = an arctangent. So Psi_B is a sum of arctangents over B's
    edges -- the sigma-softened winding number (-> 1 inside B, 0 outside as sigma -> 0). Returns (P,).
    """
    v0, e, L, n = _edges(loopB)
    w = v0[None, :, :] - Y[:, None, :]                       # (P, E, 2)  = v0 - y
    a = (L * L)[None, :]                                      # (1, E)     = |e|^2
    b = 2.0 * np.einsum("pec,ec->pe", w, e)                   # (P, E)     = 2 w . e
    c = (w * w).sum(-1) + sigma ** 2                          # (P, E)     = |w|^2 + sigma^2
    sq = np.sqrt(4.0 * a * c - b * b)                         # (P, E)     > 0 (= 2 sqrt(|e x w|^2 + |e|^2 sigma^2))
    j = (2.0 / sq) * (np.arctan((2.0 * a + b) / sq) - np.arctan(b / sq))
    wDotN = np.einsum("pec,ec->pe", w, n)                     # (P, E)
    return (wDotN * L[None, :] * j).sum(1) / (2.0 * np.pi)

def _switch(d, rOn, rOff):
    """C2 smoothstep of the centroid separation d: returns (S, dS/dd), with S=1 for d<=rOn and 0 for
    d>=rOff. Makes far polygon pairs vanish smoothly so they can be pruned EXACTLY (tex sec 7.2)."""
    t = (d - rOn) / (rOff - rOn)
    if t <= 0.0:
        return 1.0, 0.0
    if t >= 1.0:
        return 0.0, 0.0
    S = 1.0 - (6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3)
    dSdd = -(30 * t ** 4 - 60 * t ** 3 + 30 * t ** 2) / (rOff - rOn)
    return S, dSdd

# UNVERIFIED(Cam)
def coveringRadii(packing):
    """Per-polygon covering radius: the ACTUAL max vertex distance from the polygon's centroid.

    This is the radius the overlap switch is built on (rOn/rOff = rad_A + rad_B + g*sigma), so it must
    bound the polygon's extent -- two polygons whose centroids are farther apart than rad_A + rad_B
    cannot geometrically touch, and the g*sigma margin covers the mollification.

    It replaces the earlier ``targetPerimeter / 4`` for two reasons. It is TIGHTER: for a regular
    decagon the true circumradius is P/6.18, so P/4 overstated it by ~1.55x, inflating the cutoff and
    leaving ~42% of pairs active at N=32 when far fewer can actually interact. And it is RIGOROUS:
    P/4 bounds the extent only for the ACTUAL perimeter, while ``targetPerimeter`` is a target the
    soft springs let the real perimeter exceed -- so the old bound could in principle be violated by a
    stretched polygon, silently truncating a real interaction. Measured from the current positions, it
    cannot be."""
    return polygonCentroidsRadii(packing)[1]


# UNVERIFIED(Cam)
def polygonCentroidsRadii(packing):
    """``(centroids, coveringRadii)`` per polygon, vectorized -- both are needed together by every
    overlap call, and this runs on EVERY force evaluation, so a Python loop over polygons here shows
    up directly in the minimizer's step time. The uniform-vertex-count case (what Model builds) folds
    the CSR block layout into a (P, n, 2) view and reduces with plain numpy; ragged packings fall back
    to a per-polygon loop."""
    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    numPolygons = packing.numPolygons
    counts = np.diff(starts)
    if counts.size and np.all(counts == counts[0]):
        block = r.reshape(numPolygons, int(counts[0]), 2)
        cent = block.mean(axis = 1)
        d = block - cent[:, None, :]
        rad = np.sqrt(np.einsum("pkc,pkc->pk", d, d)).max(axis = 1)
        return cent, rad
    cent = np.empty((numPolygons, 2))
    rad = np.empty(numPolygons)
    for p in range(numPolygons):
        loop = r[int(starts[p]):int(starts[p + 1])]
        cent[p] = loop.mean(0)
        d = loop - cent[p]
        rad[p] = np.sqrt(np.einsum("ij,ij->i", d, d)).max()
    return cent, rad


def _assembleOverlap(packing, sigma, pairEnergy, pairGradient, gOn, gOff, numActive = None):
    """Overlap ENERGY U = 2 sum_{A<B} (A_cap^{AB} / (A_A^t + A_B^t))^2 and its vertex gradient (the
    normalized, squared form of plan eq 9.1). A_cap^{AB} is the softened overlap area between A and the
    near periodic images of B (summed over images), each image weighted by the smooth switch
    S(|c_A - c_B|) so far pairs are EXACTLY zero (tex sec 7.2). The normalizer is the constant TARGET
    areas, so dU/dv = 4 sum (A_cap/norm^2) dA_cap/dv -- squaring makes the contact force vanish as the
    overlap closes, so the packing jams near the target phi instead of shrinking. Rebuilt-each-call
    neighbor list on the centroid separation (covering radius = ``coveringRadii``, tex sec 10);
    ``pairEnergy`` / ``pairGradient`` are the closed-form exact-tier edge-pair evaluators."""
    r = packing.positions.reshape(-1, 2)
    starts = packing.startIndices
    box = packing.box
    # numActive restricts the pair loop to the first ``numActive`` polygons, which is how a CONTAINER
    # (always last) is kept out of the ordinary overlap -- it confines rather than repels and is
    # handled by containerEnergyForce.
    N = packing.numPolygons if numActive is None else int(numActive)
    sl = [slice(int(starts[p]), int(starts[p + 1])) for p in range(N)]
    cent = np.array([r[s].mean(0) for s in sl])
    rad = coveringRadii(packing)[:N]
    Atgt = np.asarray(packing.targetArea, dtype = float)
    nv = np.array([s.stop - s.start for s in sl])
    grad = np.zeros((packing.numVertices, 2))
    energy = 0.0
    for A in range(N):
        loopA = r[sl[A]]
        for B in range(A + 1, N):
            baseShift = minImageShift(cent[B] - cent[A], box)
            rOn = rad[A] + rad[B] + gOn * sigma
            rOff = rad[A] + rad[B] + gOff * sigma
            a = 0.0
            gA = np.zeros((nv[A], 2)); gB = np.zeros((nv[B], 2))
            for dx, dy in imagesFor(box):
                shift = baseShift + np.array([dx, dy])
                delta = cent[B] + shift - cent[A]
                d = float(np.hypot(*delta))
                if d >= rOff:
                    continue
                loopB = r[sl[B]] + shift
                S, dSdd = _switch(d, rOn, rOff)
                eP = pairEnergy(loopA, loopB, sigma)
                a += S * eP
                gA += S * pairGradient(loopA, loopB, sigma)
                gB += S * pairGradient(loopB, loopA, sigma)
                if dSdd != 0.0:
                    dhat = delta / d
                    gA -= (eP * dSdd / nv[A]) * dhat
                    gB += (eP * dSdd / nv[B]) * dhat
            if a != 0.0:
                norm = Atgt[A] + Atgt[B]
                energy += 2.0 * (a / norm) ** 2
                w = 4.0 * a / (norm * norm)
                grad[sl[A]] += w * gA
                grad[sl[B]] += w * gB
    return energy, grad


# ===========================================================================
# Fully analytic tier (Cam's derivation: notes/plummerOverlap.tex secs 6, verified
# to 25 digits in notes/verify_plummer_analytic.py). The outer integral over dA
# closes in the elementary master primitives below plus the single real Clausen
# core _tCoreReal; no quadrature remains. This is THE overlap energy of the model.
# ===========================================================================


# --- the six elementary master primitives (tex secs 6.3): q(s)=a s^2+b s+c with D=4ac-b^2>0;
#     Q2(xi)=(1+al^2)xi^2+2 al be xi+be^2+sg^2; Theta=arctan[(al xi+be)/sqrt(xi^2+sg^2)] ---
def _lam0(s, a, b, c):
    """int ln(a s^2 + b s + c) ds."""
    q = a * s * s + b * s + c; sD = np.sqrt(4 * a * c - b * b)
    return (s + b / (2 * a)) * np.log(q) - 2 * s + (sD / a) * np.arctan((2 * a * s + b) / sD)


def _lam1(s, a, b, c):
    """int s ln(a s^2 + b s + c) ds."""
    q = a * s * s + b * s + c; sD = np.sqrt(4 * a * c - b * b)
    return ((s * s) / 2 - (b * b - 2 * a * c) / (4 * a * a)) * np.log(q) - s * s / 2 + b * s / (2 * a) \
        - (b * sD / (2 * a * a)) * np.arctan((2 * a * s + b) / sD)


def _q2(xi, al, be, sg):
    return (1 + al ** 2) * xi * xi + 2 * al * be * xi + be * be + sg * sg


def _xi0(xi, al, be, sg):
    """int dxi / Q2."""
    A = 1 + al ** 2; delta = np.sqrt(be ** 2 + A * sg ** 2)
    return np.arctan((A * xi + al * be) / delta) / delta


def _rPrim(xi, al, be, sg):
    """int xi (al sg^2 - be xi) / Q2 dxi."""
    A = 1 + al ** 2; x0 = _xi0(xi, al, be, sg)
    x1 = np.log(_q2(xi, al, be, sg)) / (2 * A) - (al * be / A) * x0
    return -be * xi / A + (al * (sg ** 2 * A + 2 * be ** 2) / A) * x1 + (be * (be ** 2 + sg ** 2) / A) * x0


def _theta(xi, al, be, sg):
    return np.arctan((al * xi + be) / np.sqrt(xi * xi + sg * sg))


def _m1(xi, al, be, sg):
    """int xi/sqrt(xi^2+sg^2) Theta dxi (elementary)."""
    A = 1 + al ** 2; delta = np.sqrt(be ** 2 + A * sg ** 2)
    return np.sqrt(xi * xi + sg * sg) * _theta(xi, al, be, sg) + (be / (2 * A)) * np.log(_q2(xi, al, be, sg)) \
        - (al * delta / A) * np.arctan((A * xi + al * be) / delta)


# Cl2 series about 0, coefficients |B_2k| / (2k (2k+1)!) (definitions.tex Clausen block,
# derived in notes/arcsinhClausen.nb); 20 terms give ~1e-15 on the reduced range [0, pi].
_CL2COEFFS = np.array([
    1.388888888888888889e-2, 6.944444444444444444e-5, 7.873519778281683044e-7,
    1.148221634332745444e-8, 1.897886998897099907e-10, 3.387301370953521272e-12,
    6.372636443183180397e-14, 1.246205991295067230e-15, 2.510544460899954551e-17,
    5.178258806090623507e-19, 1.088735736830084884e-20, 2.325744114302087224e-22,
    5.035195213147389561e-24, 1.102649929438121533e-25, 2.438658550900734474e-27,
    5.440142678856252316e-29, 1.222834013121735212e-30, 2.767263468967950584e-32,
    6.300090591832013949e-34, 1.442086838841847521e-35])


def _cl2(x):
    """Clausen function Cl2(x) = -int_0^x ln|2 sin(t/2)| dt, real and vectorized. Reduce the
    argument to [0, pi] by 2pi-periodicity and oddness, then sum t - t ln t + sum c_k t^(2k+1)."""
    t = np.mod(x, 2 * np.pi)
    t = np.where(t > np.pi, t - 2 * np.pi, t)
    s = np.where(t < 0.0, -1.0, 1.0)
    t = np.abs(t)
    tsafe = np.where(t == 0.0, 1.0, t)
    t2 = t * t
    p = np.zeros_like(t * 1.0)
    for c in _CL2COEFFS[::-1]:
        p = p * t2 + c
    val = t - t * np.log(tsafe) + p * t * t2
    return s * np.where(t == 0.0, 0.0, val)

def _tCoreReal(xi, al, be, sg):
    """The single transcendental T = -2 J_arcsinh, evaluated with no complex arithmetic via the real
    Clausen form (definitions.tex Clausen block; mollifiedDerivation.tex sec 5). With m = al,
    nu = be/sg, w = sqrt(1+m^2+nu^2), and y = (xi+sqrt(xi^2+sg^2))/sg, the two upper-half poles give
    Im G(eta_+/-) as Bloch-Wigner D -> Clausen; T = Im G(y). Verified against direct quadrature of
    J_arcsinh to ~1e-16 (notes/verify_realroute.py, notes/arcsinhClausen.nb)."""
    m = al; nu = be / sg
    w = np.sqrt(1.0 + m * m + nu * nu)
    psi = np.arctan2(1.0, m); L = np.log(w - nu) - 0.5 * np.log(1.0 + m * m)
    y = (xi + np.sqrt(xi * xi + sg * sg)) / sg
    phiP = np.arctan2(y, w - nu - m * y); phiM = np.arctan2(y, w + nu + m * y)
    DP = 0.5 * (_cl2(2 * psi + 2 * phiP) - _cl2(2 * psi) - _cl2(2 * phiP))
    DM = 0.5 * (_cl2(2 * psi) - _cl2(2 * phiM) - _cl2(2 * psi - 2 * phiM))
    return (DP + L * phiP) - (DM - L * phiM)


def _vPlus(xi, sg):  return (xi * np.sqrt(xi * xi + sg * sg) + sg ** 2 * np.arcsinh(xi / sg)) / 2
def _vMinus(xi, sg): return (xi * np.sqrt(xi * xi + sg * sg) - sg ** 2 * np.arcsinh(xi / sg)) / 2

def _masterM(xi, al, be, sg, v, kappa):
    """Master arctan-integral M[V] = int V'(xi) Theta dxi = V Theta - rho/2 + kappa (sg^2/2) T
    (definitions.tex sec 8, the one family for energy panel and gradient). The energy panel I_tan
    (v = _vPlus, kappa = -1) and the gradient W1 core M1' (v = _vMinus, kappa = +1) share the single
    transcendental T = _tCoreReal = -2 J_arcsinh; rho = _rPrim is elementary. Here al = slope mu,
    be = intercept sigma*nu. The W0 core M1 is the elementary V = sqrt case (see _m1)."""
    return v(xi, sg) * _theta(xi, al, be, sg) - _rPrim(xi, al, be, sg) / 2 \
        + kappa * (sg ** 2 / 2) * _tCoreReal(xi, al, be, sg)

def _m2(xi, al, be, sg):  return _masterM(xi, al, be, sg, _vPlus, -1)
def _m1Prime(xi, al, be, sg): return _masterM(xi, al, be, sg, _vMinus, +1)

def _pairGrid(loopA, loopB):
    """Edge-pair frame (tex sec 6.2) for ALL edge pairs of two loops at once. Returns the A/B edge
    data and the pair arrays P0, P1, X0, X1, LA, LB, each shaped (nA, nB)."""
    a0, ea, LA, nA = _edges(loopA); b0, eb, LB, nB = _edges(loopB)
    bh = eb / LB[:, None]
    w0 = a0[:, None, :] - b0[None, :, :]
    dot2 = lambda u, v: u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
    cross2 = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    eag = ea[:, None, :]; bhg = bh[None, :, :]
    P0 = dot2(w0, bhg); P1 = dot2(eag, bhg); X0 = cross2(w0, bhg); X1 = cross2(eag, bhg)
    LAg = LA[:, None] + 0.0 * P0; LBg = LB[None, :] + 0.0 * P0
    return a0, ea, LA, nA, b0, eb, LB, nB, P0, P1, X0, X1, LAg, LBg


# Near-parallel bridge (tex sec 6.7): 2x24 Gauss on [0, 1], per half of the peak-split interval.
# Triggered per pair when |X1| <= _NEARPAR * |e_A|, where the 1/X1 closed form loses precision.
_BNODES, _BWTS = leggauss(24)
_BNODES = 0.5 * (_BNODES + 1.0)
_BWTS = 0.5 * _BWTS
_NEARPAR = 1e-2


# UNVERIFIED(Cam)
def _ceBridge(P0, P1, X0, X1, U0, sg):
    """Near-parallel-stable value of the arctan panel term. Writing xi = X0 + u X1, the closed form
    ceGen = (M2(X0+X1) - M2(X0))/X1 is exactly the regular integral (definitions.tex sec 6.7)

        int_0^1 sqrt(xi^2 + sg^2) * arctan((U0 + P1 u)/sqrt(xi^2 + sg^2)) du,

    with no 1/X1 and no blow-up of al = P1/X1. Split at the arctan node u* = -U0/P1 (the zero of the
    argument) so a fixed Gauss rule resolves the width-~sg peak; 2x24 nodes give ~1e-15 down to sg~1e-3.
    Vectorized over the (nA, nB) pair grid."""
    ustar = np.clip(-U0 / np.where(np.abs(P1) < 1e-300, 1.0, P1), 0.0, 1.0)
    tot = np.zeros_like(P0)
    for a, b in ((np.zeros_like(ustar), ustar), (ustar, np.ones_like(ustar))):
        span = b - a
        u = a[..., None] + span[..., None] * _BNODES
        r = X0[..., None] + u * X1[..., None]
        R = np.sqrt(r * r + sg * sg)
        f = R * np.arctan((U0[..., None] + P1[..., None] * u) / R)
        tot = tot + span * (f @ _BWTS)
    return tot


# UNVERIFIED(Cam)
def _wBridge(P0, P1, X0, X1, U0, sg):
    """Near-parallel-stable forms of dM1/X1 and (dM1p - X0 dM1)/X1^2 (the W0, W1 moment cores of
    _wClosedVec). With xi = X0 + u X1 these are the regular u-integrals (definitions.tex sec 6.7)

        wb0 = int_0^1 g(u) du,   wb1 = int_0^1 u g(u) du,
        g(u) = (xi / R) arctan((U0 + P1 u)/R),   R = sqrt(xi^2 + sg^2),

    with no 1/X1 blow-up. Split at the arctan node u* = -U0/P1. Returns (wb0, wb1) over the pair grid."""
    ustar = np.clip(-U0 / np.where(np.abs(P1) < 1e-300, 1.0, P1), 0.0, 1.0)
    wb0 = np.zeros_like(P0); wb1 = np.zeros_like(P0)
    for a, b in ((np.zeros_like(ustar), ustar), (ustar, np.ones_like(ustar))):
        span = b - a
        u = a[..., None] + span[..., None] * _BNODES
        r = X0[..., None] + u * X1[..., None]
        R = np.sqrt(r * r + sg * sg)
        g = (r / R) * np.arctan((U0[..., None] + P1[..., None] * u) / R)
        wb0 = wb0 + span * (g @ _BWTS)
        wb1 = wb1 + span * ((u * g) @ _BWTS)
    return wb0, wb1

def _iClosedVec(P0, P1, X0, X1, LA, LB, sg, tol = 1e-12):
    """Closed-form pair integral I = int int ln Q ds dt (tex eq 33) for arrays of edge pairs; the
    non-parallel branch (tex eq 34-36), the near-parallel bridge (tex sec 6.7), and the parallel
    branch are selected per pair by np.where."""
    tot = np.zeros_like(P0)
    par = np.abs(X1) <= tol * LA
    near = (np.abs(X1) <= _NEARPAR * LA) & ~par
    X1s = np.where(par, 1.0, X1)
    h = np.sqrt(X0 * X0 + sg ** 2)
    fa = lambda u: u * np.arctan(u / h) - (h / 2) * np.log(u * u + h * h)
    for e, eps in ((0, 1), (1, -1)):
        U0 = P0 - e * LB
        aq = LA * LA; bq = 2 * (U0 * P1 + X0 * X1); cq = U0 * U0 + X0 * X0 + sg ** 2
        Ae = U0 * (_lam0(1.0, aq, bq, cq) - _lam0(0.0, aq, bq, cq)) \
             + P1 * (_lam1(1.0, aq, bq, cq) - _lam1(0.0, aq, bq, cq))
        Be = U0 + P1 / 2
        al = P1 / X1s; be = U0 - al * X0
        ceGen = (_m2(X0 + X1, al, be, sg) - _m2(X0, al, be, sg)) / X1s
        P1s = np.where(np.abs(P1) < 1e-300, 1.0, P1)
        cePar = (h / P1s) * (fa(U0 + P1) - fa(U0))
        ceB = _ceBridge(P0, P1, X0, X1, U0, sg)
        ce = np.where(par, cePar, np.where(near, ceB, ceGen))
        tot += eps * (Ae - 2 * Be + 2 * ce)
    return tot / LB

def _wClosedVec(P0, P1, X0, X1, LA, LB, sg, tol = 1e-12):
    """Closed-form gradient moments W0, W1 (tex eq 39) for arrays of edge pairs, parallel branch
    selected per pair by np.where."""
    W0 = np.zeros_like(P0); W1 = np.zeros_like(P0)
    par = np.abs(X1) <= tol * LA
    near = (np.abs(X1) <= _NEARPAR * LA) & ~par
    X1s = np.where(par, 1.0, X1)
    h = np.sqrt(X0 * X0 + sg ** 2)
    fa = lambda u: u * np.arctan(u / h) - (h / 2) * np.log(u * u + h * h)
    fua = lambda u: ((u * u + h * h) / 2) * np.arctan(u / h) - h * u / 2
    for e, eps in ((0, 1), (1, -1)):
        U0 = P0 - e * LB
        al = P1 / X1s; be = U0 - al * X0
        dM1 = _m1(X0 + X1, al, be, sg) - _m1(X0, al, be, sg)
        dM1p = _m1Prime(X0 + X1, al, be, sg) - _m1Prime(X0, al, be, sg)
        w0g = -eps / (2 * np.pi) * dM1 / X1s
        w1g = -eps / (2 * np.pi) * (dM1p - X0 * dM1) / (X1s * X1s)
        wb0, wb1 = _wBridge(P0, P1, X0, X1, U0, sg)
        w0n = -eps / (2 * np.pi) * wb0
        w1n = -eps / (2 * np.pi) * wb1
        P1s = np.where(np.abs(P1) < 1e-300, 1.0, P1)
        d0 = (fa(U0 + P1) - fa(U0)) / P1s
        d1 = ((fua(U0 + P1) - fua(U0)) / P1s - U0 * (fa(U0 + P1) - fa(U0)) / P1s) / P1s
        w0p = -eps * (X0 / (2 * np.pi * h)) * d0
        w1p = -eps * (X0 / (2 * np.pi * h)) * d1
        W0 += np.where(par, w0p, np.where(near, w0n, w0g))
        W1 += np.where(par, w1p, np.where(near, w1n, w1g))
    return W0, W1

def plummerPairEnergyExact(loopA, loopB, sigma):
    """Pair overlap energy, fully analytic and vectorized over all edge pairs:
    -(1/4pi) sum (nA.nB) LA LB I."""
    a0, ea, LA, nA, b0, eb, LB, nB, P0, P1, X0, X1, LAg, LBg = _pairGrid(loopA, loopB)
    I = _iClosedVec(P0, P1, X0, X1, LAg, LBg, sigma)
    return -((nA @ nB.T) * LAg * LBg * I).sum() / (4 * np.pi)

def plummerPairGradientExact(loopA, loopB, sigma):
    """dA_cap/d(A's vertices), fully analytic and vectorized: the hat-weighted moments W0, W1 of
    Psi_B summed over B's edges, deposited (W0-W1) on each A-edge's start vertex and W1 on its end."""
    a0, ea, LA, nA, b0, eb, LB, nB, P0, P1, X0, X1, LAg, LBg = _pairGrid(loopA, loopB)
    W0, W1 = _wClosedVec(P0, P1, X0, X1, LAg, LBg, sigma)
    sumW0 = W0.sum(1); sumW1 = W1.sum(1)
    nAv = len(a0); grad = np.zeros((nAv, 2))
    grad += (LA * (sumW0 - sumW1))[:, None] * nA
    np.add.at(grad, (np.arange(nAv) + 1) % nAv, (LA * sumW1)[:, None] * nA)
    return grad


# ===========================================================================
# Analytic Hessian tier (notes/mollifiedDerivation.tex sec 9). The one new object is
#     grad_x Psi_B(x) = -oint_{dB} K_sigma(|x-y|) n_B(y) dl_y ,
# the mollifier smeared along dB -- elementary per edge (int dr/q^2 = rational + arctan), so the
# second derivative introduces NO new transcendentals. The outer edge integral is quadratured (the
# integrand is smooth); the Hessian only sets Newton's convergence path, while the EXACT force sets
# where it converges, so outer quadrature here costs nothing in final precision.
# ===========================================================================

# UNVERIFIED(Cam)
def _plummerQuadInts(Y, loopB, sigma):
    """Per (point, B-edge): I0 = int_0^1 dr/q^2 and I1 = int_0^1 r dr/q^2, with
    q(r) = a r^2 + b r + c the same softened quadratic as plummerMeasure. Both elementary:
        I0 = [(2ar+b)/(D^2 q)]_0^1 + (4a/D^3)[arctan((2ar+b)/D)]_0^1,  D^2 = 4ac - b^2,
        I1 = (1/2a)(1/q(0) - 1/q(1)) - (b/2a) I0     (from int (2ar+b)/q^2 = -1/q).
    Returns (I0, I1, L, n) with I0, I1 shaped (P, E)."""
    v0, e, L, n = _edges(loopB)
    w = v0[None, :, :] - Y[:, None, :]
    a = (L * L)[None, :]
    b = 2.0 * np.einsum("pec,ec->pe", w, e)
    c = (w * w).sum(-1) + sigma ** 2
    D2 = 4.0 * a * c - b * b
    D = np.sqrt(D2)
    q0 = c
    q1 = a + b + c
    I0 = (2.0 * a + b) / (D2 * q1) - b / (D2 * q0) \
         + (4.0 * a / D ** 3) * (np.arctan((2.0 * a + b) / D) - np.arctan(b / D))
    I1 = (1.0 / (2.0 * a)) * (1.0 / q0 - 1.0 / q1) - (b / (2.0 * a)) * I0
    return I0, I1, L, n


# UNVERIFIED(Cam)
def _gradPlummerMeasure(Y, loopB, sigma):
    """grad_x Psi_B(x) = -oint_{dB} K_sigma n_B dl for a batch of points Y (P, 2). Returns (P, 2)."""
    I0, _, L, n = _plummerQuadInts(Y, loopB, sigma)
    return -(sigma ** 2 / np.pi) * np.einsum("pe,e,ec->pc", I0, L, n)


# UNVERIFIED(Cam)
def _dPlummerMeasureDvB(Y, loopB, sigma):
    """dPsi_B(x)/d(B's vertices) for a batch of points Y (P, 2). Moving the boundary gives
    dPsi_B = oint K_sigma (dy . n_B) dl, so edge k deposits int(1-r)K dr on v_k and int r K dr on
    v_{z(k)}, both times |e_k| n_k. Returns (P, nB, 2)."""
    I0, I1, L, n = _plummerQuadInts(Y, loopB, sigma)
    nB = len(L)
    Ln = L[:, None] * n
    startW = (sigma ** 2 / np.pi) * (I0 - I1)
    endW = (sigma ** 2 / np.pi) * I1
    out = startW[:, :, None] * Ln[None, :, :]
    tail = endW[:, :, None] * Ln[None, :, :]
    idx = (np.arange(nB) + 1) % nB
    res = np.zeros_like(out)
    res += out
    np.add.at(res, (slice(None), idx), tail)
    return res


_ROT = np.array([[0.0, 1.0], [-1.0, 0.0]])      # L * n_hat = _ROT @ e

# Outer-edge quadrature for the Hessian (the ONE quadrature left in the Plummer tier: the energy and
# gradient are fully closed-form, but the Hessian's outer A-edge integral of grad Psi_B / dPsi_B/dvB
# is not). 96 points because those integrands are sharper than Psi_B itself; an under-resolved Hessian
# costs Newton its quadratic tail (it still converges to the same force floor, just in more steps).
_HNODES, _HWTS = leggauss(96)
_HNODES = 0.5 * (_HNODES + 1.0)
_HWTS = 0.5 * _HWTS


# UNVERIFIED(Cam)
def _pairHessianOneSided(loopA, loopB, sigma):
    """Differentiate dA_cap/d(A's vertices) with respect to A's and B's vertices.

    With the hat moments Wa_k = int (1-s) Psi_B ds and Wb_k = int s Psi_B ds along A-edge k, the
    gradient is grad_m = Wa_m (R e_m) + Wb_{z'(m)} (R e_{z'(m)}), R = _ROT. Differentiating:
    the geometric part uses d(R e_k)/dv_n = R (delta_{z(k)n} - delta_{kn}); the moment part uses the
    second moments S20/S11/S02 of grad Psi_B (for A's vertices) and Ta/Tb of dPsi_B/dv_B (for B's).
    Returns (HAA, HAB) shaped (nA, 2, nA, 2) and (nA, 2, nB, 2)."""
    v0A, eA, LA, _ = _edges(loopA)
    nA = len(v0A); nB = len(loopB)
    s = _HNODES; wq = _HWTS; G = len(s)
    Y = v0A[:, None, :] + s[None, :, None] * eA[:, None, :]
    Yf = Y.reshape(-1, 2)
    psi = plummerMeasure(Yf, loopB, sigma).reshape(nA, G)
    gps = _gradPlummerMeasure(Yf, loopB, sigma).reshape(nA, G, 2)
    dpb = _dPlummerMeasureDvB(Yf, loopB, sigma).reshape(nA, G, nB, 2)
    om = wq[None, :]
    Wa = (om * (1.0 - s)[None, :] * psi).sum(1)
    Wb = (om * s[None, :] * psi).sum(1)
    S20 = (om[:, :, None] * ((1.0 - s) ** 2)[None, :, None] * gps).sum(1)
    S11 = (om[:, :, None] * (s * (1.0 - s))[None, :, None] * gps).sum(1)
    S02 = (om[:, :, None] * (s ** 2)[None, :, None] * gps).sum(1)
    Ta = (om[:, :, None, None] * (1.0 - s)[None, :, None, None] * dpb).sum(1)
    Tb = (om[:, :, None, None] * s[None, :, None, None] * dpb).sum(1)
    Re = np.stack([eA[:, 1], -eA[:, 0]], axis = 1)
    HAA = np.zeros((nA, 2, nA, 2))
    HAB = np.zeros((nA, 2, nB, 2))
    for m in range(nA):
        zm = (m + 1) % nA
        zp = (m - 1) % nA
        HAA[m, :, m, :] += np.outer(Re[m], S20[m])
        HAA[m, :, zm, :] += np.outer(Re[m], S11[m])
        HAA[m, :, zp, :] += np.outer(Re[zp], S11[zp])
        HAA[m, :, m, :] += np.outer(Re[zp], S02[zp])
        HAB[m] += Re[m][:, None, None] * Ta[m][None, :, :]
        HAB[m] += Re[zp][:, None, None] * Tb[zp][None, :, :]
        HAA[m, :, zm, :] += Wa[m] * _ROT
        HAA[m, :, m, :] -= Wa[m] * _ROT
        HAA[m, :, m, :] += Wb[zp] * _ROT
        HAA[m, :, zp, :] -= Wb[zp] * _ROT
    return HAA, HAB


# UNVERIFIED(Cam)
def plummerPairHessian(loopA, loopB, sigma):
    """Analytic Hessian of A_cap^sigma with respect to ALL vertices of the pair, ordered
    [A's vertices, B's vertices]. Returns a dense (2(nA+nB), 2(nA+nB)) symmetric matrix."""
    nA = len(loopA); nB = len(loopB)
    HAA, HAB = _pairHessianOneSided(loopA, loopB, sigma)
    HBB, HBA = _pairHessianOneSided(loopB, loopA, sigma)
    n = nA + nB
    H = np.zeros((n, 2, n, 2))
    H[:nA, :, :nA, :] = HAA
    H[:nA, :, nA:, :] = HAB
    H[nA:, :, nA:, :] = HBB
    H[nA:, :, :nA, :] = HBA
    H = H.reshape(2 * n, 2 * n)
    return 0.5 * (H + H.T)


# UNVERIFIED(Cam)
def _switch2(d, rOn, rOff):
    """Second derivative of the C2 smoothstep _switch with respect to d."""
    t = (d - rOn) / (rOff - rOn)
    if t <= 0.0 or t >= 1.0:
        return 0.0
    return -(120 * t ** 3 - 180 * t ** 2 + 60 * t) / (rOff - rOn) ** 2


# UNVERIFIED(Cam)
def plummerOverlapHessian(packing, sigma, gOn = 2.0, gOff = 3.0, pairGradient = None):
    """Analytic Hessian of the overlap energy U = 2 sum_{A<B} (a/norm)^2, a = sum_images S(d) A_cap.

    Chain rule: dU = (4a/norm^2) da and d2U = (4/norm^2)(da (x) da + a d2a), with
        da   = S dA_cap + A_cap S' dd,
        d2a  = S d2A_cap + S'(dd (x) dA_cap + dA_cap (x) dd) + A_cap(S'' dd (x) dd + S' d2d),
    where d is the centroid separation, dd/dv_n = eps_n dhat/nv_n (eps = -1 on A, +1 on B) and
    d2d/dv_n dv_m = eps_n eps_m (I - dhat dhat^T)/(d nv_n nv_m). The A_cap Hessian is
    plummerPairHessian. Returns a dense (2 Nv, 2 Nv) symmetric matrix."""
    if pairGradient is None:
        pairGradient = plummerPairGradientExact
    r = packing.positions.reshape(-1, 2)
    starts = packing.startIndices
    box = packing.box
    N = packing.numPolygons
    sl = [slice(int(starts[p]), int(starts[p + 1])) for p in range(N)]
    cent = np.array([r[s].mean(0) for s in sl])
    rad = coveringRadii(packing)
    Atgt = np.asarray(packing.targetArea, dtype = float)
    nv = np.array([s.stop - s.start for s in sl])
    Nv = packing.numVertices
    H = np.zeros((2 * Nv, 2 * Nv))
    eye2 = np.eye(2)
    for A in range(N):
        loopA = r[sl[A]]
        for B in range(A + 1, N):
            nA, nB = int(nv[A]), int(nv[B])
            loc = 2 * (nA + nB)
            idx = np.concatenate([2 * np.arange(sl[A].start, sl[A].stop)[:, None] + [0, 1],
                                  2 * np.arange(sl[B].start, sl[B].stop)[:, None] + [0, 1]]).ravel()
            baseShift = minImageShift(cent[B] - cent[A], box)
            rOn = rad[A] + rad[B] + gOn * sigma
            rOff = rad[A] + rad[B] + gOff * sigma
            a = 0.0
            da = np.zeros(loc)
            d2a = np.zeros((loc, loc))
            for dx, dy in imagesFor(box):
                shift = baseShift + np.array([dx, dy])
                delta = cent[B] + shift - cent[A]
                d = float(np.hypot(*delta))
                if d >= rOff:
                    continue
                loopB = r[sl[B]] + shift
                S, dSdd = _switch(d, rOn, rOff)
                eP = plummerPairEnergyExact(loopA, loopB, sigma)
                g = np.concatenate([pairGradient(loopA, loopB, sigma).ravel(),
                                    pairGradient(loopB, loopA, sigma).ravel()])
                Hp = plummerPairHessian(loopA, loopB, sigma)
                a += S * eP
                da += S * g
                d2a += S * Hp
                if dSdd != 0.0:
                    dhat = delta / d
                    dd = np.zeros(loc)
                    dd[:2 * nA] = np.tile(-dhat / nA, nA)
                    dd[2 * nA:] = np.tile(dhat / nB, nB)
                    da += eP * dSdd * dd
                    d2a += dSdd * (np.outer(dd, g) + np.outer(g, dd))
                    M = (eye2 - np.outer(dhat, dhat)) / d
                    sgn = np.concatenate([np.full(nA, -1.0 / nA), np.full(nB, 1.0 / nB)])
                    d2d = np.kron(np.outer(sgn, sgn), M)
                    d2a += eP * (_switch2(d, rOn, rOff) * np.outer(dd, dd) + dSdd * d2d)
            if a != 0.0:
                norm = Atgt[A] + Atgt[B]
                blk = (4.0 / (norm * norm)) * (np.outer(da, da) + a * d2a)
                H[np.ix_(idx, idx)] += blk
    return 0.5 * (H + H.T)


# UNVERIFIED(Cam)
def plummerOverlapExact(packing, sigma, gOn = 2.0, gOff = 3.0, numActive = None):
    """Whole-packing overlap energy + vertex gradient (THE overlap energy of the model): the near
    polygon pairs of a centroid neighbor list, each weighted by a smooth switch (rOn/rOff = covering
    radii + gOn/gOff * sigma) so far pairs are EXACTLY zero, evaluated with the closed-form
    (no-quadrature) edge-pair evaluators plummerPairEnergyExact / plummerPairGradientExact. Returns
    ``(energy, grad)`` with ``grad`` = dA/dv; the overlap FORCE is its negative."""
    return _assembleOverlap(packing, sigma, plummerPairEnergyExact, plummerPairGradientExact,
                           gOn, gOff, numActive = numActive)


# ---------------------------------------------------------------------------
# Intra-polygon self-repulsion (from selfRepulsion.py)
# ---------------------------------------------------------------------------

_NODES, _WTS = leggauss(12)
_SX = 0.5 * (_NODES + 1.0)               # quadrature nodes on [0, 1]
_SW = 0.5 * _WTS                          # and weights


# UNVERIFIED(Cam)
def _selfRepEdgePairs(packing):
    """Global vertex indices of every non-adjacent intra-polygon edge pair: arrays (iA0, iA1, iB0, iB1)
    of the four endpoint vertices. Cached on the packing (topology-only; rebuilt if it changes)."""
    starts = packing.startIndices
    key = (int(packing.numVertices), tuple(int(s) for s in starts))
    if getattr(packing, "_selfRepKey", None) == key:
        return packing._selfRepIdx
    iA0 = []; iA1 = []; iB0 = []; iB1 = []
    for p in range(packing.numPolygons):
        i0 = int(starts[p]); n = int(starts[p + 1]) - i0
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                iA0.append(i0 + i); iA1.append(i0 + (i + 1) % n)
                iB0.append(i0 + j); iB1.append(i0 + (j + 1) % n)
    idx = tuple(np.asarray(a, dtype = int) for a in (iA0, iA1, iB0, iB1))
    packing._selfRepKey = key; packing._selfRepIdx = idx
    return idx


def selfRepulsionEnergyForce(packing, kSelf, delta):
    """Total self-repulsion energy and force. Returns ``(energy, force)`` with ``force`` the flat
    (2N,) array ``-dU/dv``. ``kSelf`` sets the barrier height, ``delta`` its range. Vectorized over all
    non-adjacent intra-polygon edge pairs at once (the topology index is cached by _selfRepEdgePairs)."""
    r = packing.positions.reshape(-1, 2)
    iA0, iA1, iB0, iB1 = _selfRepEdgePairs(packing)
    force = np.zeros_like(r)
    if iA0.size == 0:
        return 0.0, force.reshape(-1)
    a0 = r[iA0]; ea = r[iA1] - a0                          # (K, 2)
    b0 = r[iB0]; eb = r[iB1] - b0
    y = a0[:, None, :] + _SX[None, :, None] * ea[:, None, :]     # (K, G, 2)
    z = b0[:, None, :] + _SX[None, :, None] * eb[:, None, :]
    d = y[:, :, None, :] - z[:, None, :, :]                       # (K, G, G, 2)
    phi = np.exp(-(d * d).sum(-1) / (2.0 * delta ** 2))          # (K, G, G)
    w2 = _SW[:, None] * _SW[None, :]                             # (G, G)
    energy = 0.5 * kSelf * float((w2[None] * phi).sum())
    dUdd = (0.5 * kSelf * w2[None] * phi)[..., None] * (-d / delta ** 2)   # (K, G, G, 2)
    gA0 = (dUdd * (1.0 - _SX)[None, :, None, None]).sum((1, 2))  # (K, 2)
    gA1 = (dUdd * _SX[None, :, None, None]).sum((1, 2))
    gB0 = (-dUdd * (1.0 - _SX)[None, None, :, None]).sum((1, 2))
    gB1 = (-dUdd * _SX[None, None, :, None]).sum((1, 2))
    np.add.at(force, iA0, -gA0); np.add.at(force, iA1, -gA1)
    np.add.at(force, iB0, -gB0); np.add.at(force, iB1, -gB1)
    return energy, force.reshape(-1)


# ---------------------------------------------------------------------------
# Fixed (container) boundary: a wall polygon that CONFINES the packing
# ---------------------------------------------------------------------------

# UNVERIFIED(Cam)
def containerOrientationSign(packing, containerIndex):
    """``+1`` when the container loop is CLOCKWISE, ``-1`` when counter-clockwise.

    The wall penalty needs "the area of S lying OUTSIDE the container", and the overlap A_cap comes
    out with a sign set by the container's winding: a CW container gives A_cap = -area(S and C), a
    CCW one gives +area(S and C). Returning the sign lets the caller write one formula,
    ``a = area(S) + sign * A_cap``, that is correct either way -- so a container drawn in either
    direction behaves identically and nobody has to remember which way round to list the corners."""
    r = packing.positions.reshape(-1, 2)
    a = int(packing.startIndices[containerIndex])
    b = int(packing.startIndices[containerIndex + 1])
    loop = r[a : b]
    signed = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1)
                          - np.roll(loop[:, 0], -1) * loop[:, 1])
    return 1.0 if signed < 0.0 else -1.0


# UNVERIFIED(Cam)
def sharpContainerEnergyForce(packing, kContainer = 1.0):
    """EXACT (unmollified) confinement energy + force from the container wall.

    Same quantity the mollified container penalises -- the area of each shape lying OUTSIDE the wall,

        a_S = area(S) - overlapArea(S, C)

    -- but with the exact overlap rather than the sigma-softened one, and in the model's
    NORMALIZED-SQUARED form, matching both the mollified container and the sharp inter-particle law.
    Within a model type the functional has to be consistent, or the wall and the inter-particle contact
    would obey different contact laws:

        U = 2 kContainer sum_S (a_S / norm_S)^2,     norm_S = 2 * targetArea[S]

    (the wall's own target area is not used in the normalizer -- it is the size of the whole box, so
    the usual sum would make the wall tens of times softer than a shape-shape contact and the packing
    would sink into it).

    The obstacle used to be that the sharp tier has no single-pair routine, its intersection/follower
    pipeline running over the whole packing at once. It now accumulates PER PAIR, so the container's
    contribution is simply read off. It relies on ``updateIntersections`` being ALL-TO-ALL with no
    distance pruning, which is what it is -- a neighbor-list version would have to guarantee container
    pairs are never culled, and the wall is exactly the pair a covering-radius test would drop (its
    radius is the box half-diagonal).

    WINDING decides which side the pair overlap already reports. A container wound CLOCKWISE has its
    interior reversed, so the pipeline's "overlap" for (S, C) is the part of S OUTSIDE the wall --
    exactly the penalised quantity, no complement needed. Wound counter-clockwise it reports the part
    inside, and the complement is taken. Getting this backwards is SILENT: the energy stays smooth and
    its gradient stays self-consistent (finite differences of the wrong energy agreed with the wrong
    force to 4e-10), so only an INDEPENDENT area check catches it -- here, convex clipping.

    Returns ``(energy, force)`` with force flat (2N,), matching ``containerEnergyForce``."""
    containerIndex = packing.containerIndex
    if containerIndex is None:
        return 0.0, np.zeros_like(packing.positions)
    numShapes = int(containerIndex)
    starts = np.asarray(packing.startIndices, dtype = int)
    cA = int(starts[numShapes])

    updateIntersections(packing)
    updateFollowers(packing)
    updateOverlapArea(packing)

    r = packing.positions.reshape(-1, 2)
    shapeOf = packing.shapeId[: cA]
    rNext = r[packing.next[: cA]]
    rPrev = r[packing.prev[: cA]]
    cross = r[: cA, 0] * rNext[:, 1] - rNext[:, 0] * r[: cA, 1]
    shapeArea = 0.5 * np.bincount(shapeOf, weights = cross, minlength = numShapes)
    gradientArea = 0.5 * np.stack([rNext[:, 1] - rPrev[:, 1], rPrev[:, 0] - rNext[:, 0]], axis = 1)

    sign = containerOrientationSign(packing, containerIndex)
    capArea = np.zeros(numShapes)
    for shape in range(numShapes):
        key = (min(shape, containerIndex), max(shape, containerIndex))
        capArea[shape] = packing.pairOverlapArea.get(key, 0.0)
    outside = capArea if sign > 0.0 else shapeArea - capArea

    norm = 2.0 * np.asarray(packing.targetArea, dtype = float)[: numShapes]
    energy = float(np.sum(2.0 * kContainer * (outside / norm) ** 2))
    weight = 4.0 * kContainer * outside / (norm * norm)          # dU/da_S

    # d a_S/dv is +d(pair area)/dv for a clockwise wall, and d(area_S)/dv - d(pair area)/dv otherwise.
    pairWeight = {}
    for shape in range(numShapes):
        key = (min(shape, containerIndex), max(shape, containerIndex))
        pairWeight[key] = weight[shape] * (1.0 if sign > 0.0 else -1.0)
    updateOverlapGradient(packing, pairWeight = pairWeight)
    gradient = np.array(packing.overlapGradient, dtype = float)
    if sign <= 0.0:
        gradient[: cA] += weight[shapeOf][:, None] * gradientArea
    return energy, (-gradient).reshape(-1)


# UNVERIFIED(Cam)
def containerEnergyForce(packing, sigma, kContainer = 1.0, mollified = True):
    """Confinement energy + force from the CONTAINER polygon (``packing.containerIndex``).

    For each ordinary polygon S the penalised quantity is the area of S lying OUTSIDE the wall,

        a_S = area(S) + sign * A_cap(S, C)

    which is 0 when S is fully contained, rises to area(S) once it is fully outside, and is smooth in
    between. ``area(S)`` is the plain shoelace area -- exact even for the mollified model, since the
    mollifier integrates to 1 -- and ``A_cap`` is the same overlap the ordinary pairs use. The energy
    keeps the model's normalized-squared form, ``U = 2 sum (a_S / norm)^2``.

    Two deliberate differences from an ordinary pair:

    NO SWITCH. Ordinary pairs are cut off by the covering-radius smoothstep, but the container's
    covering radius is the box half-diagonal (0.707), so a shape near a corner sits right at the
    cutoff and the wall would switch OFF exactly where it is needed most. Every shape is evaluated
    against the wall.

    NORM. ``norm = 2 * targetArea[S]`` rather than ``targetArea[S] + targetArea[C]``. The container is
    the size of the whole box, so the usual sum would make the wall tens of times softer than a
    shape-shape contact and the packing would sink into it; this keeps wall stiffness comparable to
    inter-particle stiffness.

    Vectorized over EVERY shape edge at once rather than looping shapes. Each pair call handles only
    nS * nC panels (40 for a 10-gon against a square), so a per-shape loop is pure numpy call
    overhead -- it measured 374 ms against 14 ms for the entire rest of the force evaluation. Batching
    every shape edge against the container's edges turns ~3N calls into three.

    Returns ``(energy, force)`` with force a flat (2N,) array. Requires ``packing.containerIndex``."""
    containerIndex = packing.containerIndex
    if containerIndex is None:
        return 0.0, np.zeros_like(packing.positions)
    if not mollified:
        return sharpContainerEnergyForce(packing, kContainer)

    r = packing.positions.reshape(-1, 2)
    starts = packing.startIndices
    sign = containerOrientationSign(packing, containerIndex)
    cA, cB = int(starts[containerIndex]), int(starts[containerIndex + 1])
    loopC = r[cA : cB]
    numShapes = containerIndex

    # --- every shape edge, as one flat batch -------------------------------------------------
    edgeStart = np.arange(cA)                       # vertices before the container own the edges
    edgeEnd = packing.next[:cA]
    shapeOf = packing.shapeId[:cA]
    a0 = r[edgeStart]
    ea = r[edgeEnd] - a0
    LA = np.hypot(ea[:, 0], ea[:, 1])
    tauA = ea / np.where(LA > 1e-300, LA, 1.0)[:, None]
    normalA = np.stack([tauA[:, 1], -tauA[:, 0]], axis = 1)

    b0, eb, LB, normalB = _edges(loopC)
    bh = eb / LB[:, None]
    w0 = a0[:, None, :] - b0[None, :, :]
    dot2 = lambda u, v: u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
    cross2 = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    eag = ea[:, None, :]; bhg = bh[None, :, :]
    P0 = dot2(w0, bhg); P1 = dot2(eag, bhg)
    X0 = cross2(w0, bhg); X1 = cross2(eag, bhg)
    LAg = LA[:, None] + 0.0 * P0; LBg = LB[None, :] + 0.0 * P0

    # A_cap per shape: -(1/4pi) sum (nA.nB) LA LB I, summed over that shape's edges.
    I = _iClosedVec(P0, P1, X0, X1, LAg, LBg, sigma)
    perEdge = -((normalA @ normalB.T) * LAg * LBg * I).sum(1) / (4.0 * np.pi)
    cap = np.bincount(shapeOf, weights = perEdge, minlength = numShapes)

    # Shoelace area and its gradient for every shape, vectorized the same way.
    rNext = r[packing.next[:cA]]
    rPrev = r[packing.prev[:cA]]
    cross = r[:cA, 0] * rNext[:, 1] - rNext[:, 0] * r[:cA, 1]
    area = 0.5 * np.bincount(shapeOf, weights = cross, minlength = numShapes)
    gradArea = 0.5 * np.stack([rNext[:, 1] - rPrev[:, 1], rPrev[:, 0] - rNext[:, 0]], axis = 1)

    aOut = area + sign * cap
    norm = 2.0 * np.asarray(packing.targetArea, dtype = float)[:numShapes]
    energy = float(np.sum(2.0 * kContainer * (aOut / norm) ** 2))
    weight = 4.0 * kContainer * aOut / (norm * norm)          # dU/da per shape
    wEdge = weight[shapeOf]

    force = np.zeros_like(r)
    # dA_cap/d(shape vertices): hat moments of Psi_C along each shape edge.
    W0, W1 = _wClosedVec(P0, P1, X0, X1, LAg, LBg, sigma)
    sumW0 = W0.sum(1); sumW1 = W1.sum(1)
    depositStart = (LA * (sumW0 - sumW1))[:, None] * normalA
    depositEnd = (LA * sumW1)[:, None] * normalA
    np.add.at(force, edgeStart, -wEdge[:, None] * (gradArea + sign * depositStart))
    np.add.at(force, edgeEnd, -wEdge[:, None] * (sign * depositEnd))

    # dA_cap/d(container vertices): the reverse frame, container edges as the outer loop. SKIPPED
    # when the whole wall is pinned, which is the normal case -- the result would be multiplied by
    # zero in applyPins immediately afterwards, and it costs as much as the shape-side gradient (a
    # third of the whole term). Computed only if some wall vertex is actually free to move.
    pinned = getattr(packing, "pinned", None)
    if pinned is not None and bool(pinned[cA : cB].all()):
        return energy, force.reshape(-1)

    w0r = b0[:, None, :] - a0[None, :, :]
    ahat = ea / np.where(LA > 1e-300, LA, 1.0)[:, None]
    ebg = eb[:, None, :]; ahg = ahat[None, :, :]
    Q0 = dot2(w0r, ahg); Q1 = dot2(ebg, ahg)
    Y0 = cross2(w0r, ahg); Y1 = cross2(ebg, ahg)
    LBr = LB[:, None] + 0.0 * Q0; LAr = LA[None, :] + 0.0 * Q0
    V0, V1 = _wClosedVec(Q0, Q1, Y0, Y1, LBr, LAr, sigma)
    # Each shape carries its own chain-rule weight, so the sum over shape edges is weighted.
    sumV0 = (V0 * wEdge[None, :]).sum(1); sumV1 = (V1 * wEdge[None, :]).sum(1)
    nC = len(b0)
    contribStart = (LB * (sumV0 - sumV1))[:, None] * normalB
    contribEnd = (LB * sumV1)[:, None] * normalB
    np.add.at(force, cA + np.arange(nC), -sign * contribStart)
    np.add.at(force, cA + (np.arange(nC) + 1) % nC, -sign * contribEnd)
    return energy, force.reshape(-1)


def _shoelaceAreaGradient(loop):
    """Signed shoelace area of one loop and its gradient dA/dv, shape (n, 2).

    dA/dv_k = (1/2)(y_{k+1} - y_{k-1}, x_{k-1} - x_{k+1}) -- the same expression softBody uses for the
    area spring, restated here for a single loop."""
    nxt = np.roll(loop, -1, axis = 0)
    prv = np.roll(loop, 1, axis = 0)
    area = 0.5 * np.sum(loop[:, 0] * nxt[:, 1] - nxt[:, 0] * loop[:, 1])
    grad = 0.5 * np.stack([nxt[:, 1] - prv[:, 1], prv[:, 0] - nxt[:, 0]], axis = 1)
    return area, grad
