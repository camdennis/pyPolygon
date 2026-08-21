"""Alternating short/long edge PAIRS with right-angled corners -- squares in one ramp.

A 4m-gon whose edges run in consecutive PAIRS, alternating short-pair, long-pair, is a square once
two things have happened: the short pairs have shrunk to nothing, and each long pair has bent to a
right angle. Both are one monotone walk, and at the end the shape is a square carrying collinear
extras, so the vertex count comes off exactly and without a second ramp.

    edges (mod 4):   parity, parity+1   SHORT      parity+2, parity+3   LONG
    the corner:      the vertex BETWEEN a long pair's two edges, i.e. index parity+3 (mod 4)

ONE FREE NUMBER PER POLYGON for the lengths. With 4 short pairs of edge s and 4 long pairs of edge l,

    4(2s) + 4(2l) = P    =>    l = 2P/n - s,

so holding the perimeter makes the long edge a function of the short one and the target set is
(A, P, s). Its dimensionless form ``u = s / (2P/n)`` is 1/2 when every edge is equal and 0 at the end.

THE CORNER IS THE FLATNESS FAMILY, REPOINTED. Cam's condition is that a pair's two ends sit
``sqrt(2)`` times its edge length apart. For a pair the entering and leaving edges are equal, so the
existing scale-free flatness ``d / (a + b) = cos(theta/2)`` -- with theta the TURNING angle -- is
exactly ``d / (2 l)``, and

    straight      theta = 0    d/(a+b) = 1
    right angle   theta = 90   d/(a+b) = 1/sqrt(2) = 0.70710678   <- Cam's sqrt(2) l

So ``setConstraints(flatten = True)`` on a mask of the long-pair corners, with ``setFlatTargets`` at
``RIGHT_ANGLE``, is the whole corner mechanism -- an FD-verified Jacobian that is already in the code,
and dimensionless, so it holds while the polygon SIZES are still free. ``RIGHT_ANGLE`` is an interior
point of ``d/(a + b)``, not the bound at 1 where a flatten-to-straight ramp runs out of gradient, so
this target can actually be MET rather than merely approached.

THE CORNER IS PINNED, NOT RAMPED, and that is a measured result rather than a preference. Four
schedules over 30 rounds, u walking 0.5 -> 0.003, worst max|C| over the whole ramp:

    corner pinned at 90        springs on   5.2e-13    u reached 0.00300 on both polygons
    corner pinned at 90        springs off  1.4e-01    lost at t = 0.53
    open to 0.93, then close   springs on   7.1e+01    lost at t = 0.50
    0.85 -> 90 linear          springs on   9.2e+01    lost at t = 0.70
    straight -> 90             springs on   2.4e+01    lost at t = 0.53

Anything that moves the corner away from 90 and back loses the retraction, because the seed already
IS a square: with the corners flat the four sharp vertices have to migrate to the short-pair middles
and then migrate back, which is a rearrangement, not a ramp. Pinning them costs no compliance that
matters, because u supplies it -- see the slack law below. The SPRING relax between rounds is not
optional: the same pinned schedule without it loses the retraction halfway.

COMPLIANCE COMES FROM u, AND IT WITHDRAWS ITSELF. Measured maximum area of such a 16-gon at kappa = 4
against its target 4 p^2 (p = s + l), with the corners at 90 degrees:

    u        0.5      0.3      0.2      0.1      0.05
    slack   +2.9%   +0.66%   +0.20%   +0.025%   +0.003%

about ``u^3 / 4``. The polygon keeps 8 shape degrees of freedom throughout, but the area row pinches
them as u falls, so driving u to zero withdraws the shape freedom CONTINUOUSLY and lands on the
square. Nothing ever goes infeasible: the square itself satisfies the target set at every u, with the
short pairs lying straight along its sides.

PUT THE RIGHT ANGLE ON THE LONG PAIRS ONLY. On every pair it is infeasible at every u > 0 -- the
shortfall is exactly ``4 (2 - sqrt(2)) s l``, so the best such 16-gon encloses
``(2 + sqrt(2)) p^2 = 3.414 p^2`` at u = 1/2 against a target of ``4 p^2``, 15% short.
"""

import numpy as np

from build import cyclicArea, regularShapeIndex

# d/(a + b) at a 90 degree turn, since d = 2 l cos(theta/2) for equal edges. The corner ramp's end.
RIGHT_ANGLE = float(1.0 / np.sqrt(2.0))

# UNVERIFIED(Cam)
def polygonSlices(packing, skipContainer = True):
    """``[(polygon, start, stop)]`` over the polygons a stage acts on, container excluded.

    Read off ``startIndices`` rather than from a vertex count, because the counts are NOT uniform --
    the container is a 4-gon while the shapes are 16-gons, and a stage that collapses one polygon
    before another would make them ragged among themselves too."""
    starts = np.asarray(packing.startIndices, dtype = int)
    container = getattr(packing, "containerIndex", None)
    out = []
    for polygon in range(packing.numPolygons):
        if skipContainer and container is not None and polygon == int(container):
            continue
        out.append((polygon, int(starts[polygon]), int(starts[polygon + 1])))
    return out

# UNVERIFIED(Cam)
def requirePairs(polygon, count):
    """Refuse a vertex count the pair structure does not divide."""
    if count % 4:
        raise ValueError(f"polygon {polygon} has {count} vertices; alternating short/long PAIRS need "
                         f"a multiple of 4 (4m edges = m pairs, alternating).")

# UNVERIFIED(Cam)
def chooseParity(packing, skipContainer = True):
    """The parity whose LONG pairs straddle the vertices that are already sharpest.

    Which of the four phases is called "short" is not cosmetic, and getting it wrong is expensive.
    A square doubled twice carries its real 90 degree corners at vertices 0, 4, 8, 12 -- index 0
    mod 4 -- so a mask that constrains index 3 instead asks every corner to move three slots round
    the polygon: the existing corners must flatten while four flat vertices sharpen. Measured, the
    retraction never recovered -- max|C| 1.5e+04 and u stuck at 0.11 against a target of 0.003.

    Picking the phase by sharpness puts the constraint where the geometry already is, so the ramp
    only has to bend corners it is already sitting on. Returns the ``parity`` (index mod 4 of the
    first SHORT edge) such that ``cornerMask`` lands on that phase, i.e. ``phase + 1``."""
    r = packing.positions.reshape(-1, 2)
    score = np.zeros(4)
    for polygon, a, b in polygonSlices(packing, skipContainer):
        requirePairs(polygon, b - a)
        loop = r[a : b]
        behind = loop - np.roll(loop, 1, axis = 0)
        ahead = np.roll(loop, -1, axis = 0) - loop
        turn = np.abs(np.arctan2(np.cross(behind, ahead), np.einsum("ij,ij->i", behind, ahead)))
        for phase in range(4):
            score[phase] += float(turn[phase::4].sum())
    return int((int(np.argmax(score)) + 1) % 4)

# UNVERIFIED(Cam)
def parityOf(packing, parity = None):
    """The parity to use: the caller's, else the stored one, else 0.

    Stored on the PACKING so the edge mask, the corner mask and the ramp cannot disagree -- the same
    reason ``diagonalMask`` lives there."""
    if parity is not None:
        return int(parity)
    return int(getattr(packing, "pairParity", 0) or 0)

# UNVERIFIED(Cam)
def ensureParity(packing, parity = None, skipContainer = True):
    """The parity, CHOSEN and stored on first use -- whichever caller gets there first.

    Order-independence matters here because the two halves of the protocol are set up in separate
    calls. Measured when they disagreed: ``setAlternatingEdges`` wrote its targets at parity 0 while
    ``selectPairCorners`` then chose 1, so each constrained corner had one long edge and one short
    one. ``d/(a + b)`` is only ``cos(theta/2)`` when the two are EQUAL, so the rows were satisfied
    exactly -- residual 8e-16 -- at corners of 95.68 degrees rather than 90."""
    stored = getattr(packing, "pairParity", None)
    if parity is None:
        parity = chooseParity(packing, skipContainer) if stored is None else int(stored)
    packing.pairParity = int(parity)
    return int(parity)

# UNVERIFIED(Cam)
def shortMask(packing, parity = None, skipContainer = True):
    """Per-EDGE bool, True on the SHORT pairs -- two of every four.

    One entry per vertex, matching ``targetEdgeLength`` and ``Model.getEdgeLengths``: entry k is the
    edge LEAVING vertex k. ``parity`` is the index (mod 4) of the first short edge."""
    parity = parityOf(packing, parity)
    mask = np.zeros(packing.numVertices, dtype = bool)
    for polygon, a, b in polygonSlices(packing, skipContainer):
        requirePairs(polygon, b - a)
        for offset in (0, 1):
            mask[a + (parity + offset) % 4 : b : 4] = True
    return mask

# UNVERIFIED(Cam)
def cornerMask(packing, parity = None, skipContainer = True):
    """Per-VERTEX bool, True at the middle of each LONG pair -- the future square's corners.

    These are the vertices the flatness family holds: their two edges are the long pair's, so
    ``d/(a + b)`` there is ``d/(2 l)`` and aiming it at ``RIGHT_ANGLE`` is Cam's ``d = sqrt(2) l``.
    One vertex in four, and the ONLY ones constrained -- holding every vertex would bend the short
    pairs too, which is infeasible at any u > 0 (see the module docstring)."""
    parity = parityOf(packing, parity)
    mask = np.zeros(packing.numVertices, dtype = bool)
    for polygon, a, b in polygonSlices(packing, skipContainer):
        requirePairs(polygon, b - a)
        mask[a + (parity + 3) % 4 : b : 4] = True
    return mask

# UNVERIFIED(Cam)
def shortRatios(packing, parity = None, skipContainer = True):
    """Live ``u = s / (2P/n)`` per polygon, from the GEOMETRY.

    Measured as the share of the perimeter the short pairs carry, ``sum(short)/P``, which equals
    ``s/(s + l)`` when the pairs are uniform and degrades gracefully when they are not -- the
    constraint rows hold each edge individually, so they never drift far apart, but the geometry is
    what is being reported and it should not be read through an assumption of uniformity."""
    r = packing.positions.reshape(-1, 2)
    edges = r[packing.next] - r
    lengths = np.sqrt(np.einsum("ij,ij->i", edges, edges))
    mask = shortMask(packing, parity, skipContainer)
    out = []
    for polygon, a, b in polygonSlices(packing, skipContainer):
        perimeter = float(lengths[a : b].sum())
        out.append(float(lengths[a : b][mask[a : b]].sum()) / max(perimeter, 1e-300))
    return np.array(out)

# UNVERIFIED(Cam)
def pairTargets(packing, u, parity = None, skipContainer = True):
    """Per-EDGE target lengths realizing ratio ``u`` (scalar, or one per non-container polygon).

    Short edges get ``u * 2P/n`` and long ones ``(1 - u) * 2P/n``, so the perimeter target is
    reproduced EXACTLY by construction -- each adjacent short/long pairing sums to 2P/n and there are
    n/2 of them. The area target is not touched, so the target shape index does not move either: this
    family redistributes a polygon's perimeter and changes nothing else about what it is asked to be.

    Container edges are returned unchanged."""
    parity = ensureParity(packing, parity, skipContainer)
    l0 = np.array(packing.targetEdgeLength, dtype = float)
    slices = polygonSlices(packing, skipContainer)
    values = np.asarray(u, dtype = float).ravel()
    if values.size not in (1, len(slices)):
        raise ValueError(f"u has {values.size} entries; expected 1 or {len(slices)} (one per "
                         f"non-container polygon).")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        bad = int(np.argmax((values <= 0.0) | (values >= 1.0)))
        raise ValueError(
            f"u = {values[bad]:.6g} is outside (0, 1). u is the share of a short/long PAIRING taken "
            f"by the short edge, so u >= 1 asks for a negative long edge and u <= 0 asks for a "
            f"zero-length one, whose fractional residual l^2/l0^2 - 1 divides by zero. Ramp toward 0 "
            f"and stop short of it.")
    mask = shortMask(packing, parity, skipContainer)
    for index, (polygon, a, b) in enumerate(slices):
        count = b - a
        requirePairs(polygon, count)
        pairing = 2.0 * float(packing.targetPerimeter[polygon]) / count
        value = float(values[0] if values.size == 1 else values[index])
        block = np.where(mask[a : b], value * pairing, (1.0 - value) * pairing)
        l0[a : b] = block
    return l0

# UNVERIFIED(Cam)
def regularTargets(packing, skipContainer = True):
    """Per-EDGE target lengths making every polygon REGULAR at its own target area.

    The endgame of a cascade, and the one place the perimeter target should be derived rather than
    carried. Each collapse loses a little area, and syncing the AREA to the geometry afterwards banks
    that loss as a permanent excess in the shape index. At n = 4 that is fatal in a visible way:
    kappa = 4.0071 instead of 4 makes the rhombus solution ``sin(theta) = (4/kappa)^2`` and the
    corners come out at 85.2 degrees, not 90.

    Deriving the perimeter from the area instead sets kappa to the regular floor exactly, which at
    n = 4 means the edge target is ``sqrt(A0)`` and the square is the ONLY shape satisfying the set."""
    l0 = np.array(packing.targetEdgeLength, dtype = float)
    for polygon, a, b in polygonSlices(packing, skipContainer):
        count = b - a
        area = abs(float(packing.targetArea[polygon]))
        l0[a : b] = regularShapeIndex(count) * np.sqrt(area) / count
    return l0

# UNVERIFIED(Cam)
def targetShapeMargin(packing, skipContainer = True):
    """``(index, floor)`` per polygon: the target shape index and the smallest its EDGE TARGETS admit.

    The feasibility number this protocol needs, and the one ``ShapeConstraints.infeasibleReason`` does
    not compute -- that check bounds the area by the REGULAR n-gon's, ``P^2/(4 n tan(pi/n))``, which
    ignores how the perimeter is distributed. Unequal edges enclose strictly less.

    ``floor`` is ``minShapeIndex`` of the edge targets (via ``cyclicArea``), so it accounts for the
    short/long split but NOT for the corner constraints, which tighten it further; see the module
    docstring for the measured slack once the corners are at 90 degrees. ``index <= floor`` means no
    polygon with those edge lengths encloses the target area at all."""
    l0 = np.asarray(packing.targetEdgeLength, dtype = float)
    index, floor = [], []
    for polygon, a, b in polygonSlices(packing, skipContainer):
        lengths = l0[a : b]
        area = abs(float(packing.targetArea[polygon]))
        index.append(float(lengths.sum()) / np.sqrt(max(area, 1e-300)))
        floor.append(float(lengths.sum()) / np.sqrt(max(cyclicArea(lengths), 1e-300)))
    return np.array(index), np.array(floor)

# UNVERIFIED(Cam)
def normalQuantiles(count):
    """The ``count`` standard-normal quantiles at plotting positions ``(i + 1/2) / count``.

    Bisected on ``erf`` rather than pulled from scipy, which the project does not depend on. Eighty
    halvings of [-8, 8] land well under double precision, and the endpoints are ~1e-15 quantiles, so
    nothing is clipped for any count a packing will ever have."""
    from math import erf
    probability = (np.arange(int(count)) + 0.5) / int(count)
    out = np.empty(int(count))
    for i, target in enumerate(probability):
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / np.sqrt(2.0))) < target:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out

# UNVERIFIED(Cam)
def logNormalQuantiles(count, mean, cv):
    """Sorted quantiles of the log-normal with this MEAN and coefficient of variation.

    Parameterized by the mean and the CV rather than by (mu, sigma) because those are the two things
    a schedule walks. ``cv = 0`` is the degenerate limit and returns ``count`` copies of the mean,
    which is what the end of a ramp wants -- no special case needed at the call site."""
    sigma = np.sqrt(np.log(1.0 + float(cv) ** 2))
    mu = np.log(float(mean)) - 0.5 * sigma ** 2
    return np.exp(mu + sigma * normalQuantiles(count))

# UNVERIFIED(Cam)
def rankOrderTargets(current, sample):
    """Give each entry of ``current`` the same-RANK entry of ``sample``: monotone transport.

    This IS the map ``P^-1 . F`` -- F the current distribution's CDF, P the scheduled one's -- with F
    taken empirically: ranking a value among its peers is evaluating its own empirical CDF, and
    reading that rank out of the sorted schedule is inverting P's. Written in ranks because that form
    is exact for a finite packing (no histogram, no bandwidth) and because it CANNOT reorder the
    polygons -- the polygon with the widest short edges keeps that place, so nothing swaps under a
    schedule step and the geometry only ever has to move a little."""
    current = np.asarray(current, dtype = float)
    sample = np.asarray(sample, dtype = float)
    if current.size != sample.size:
        raise ValueError(f"rank-order transport needs matching sizes, got {current.size} current "
                         f"values and {sample.size} scheduled ones.")
    return np.sort(sample)[np.argsort(np.argsort(current))]

# UNVERIFIED(Cam)
def ratioSchedule(t, startMean = 0.5, startCv = 0.4, endMean = 0.02, endCv = 0.0):
    """``(mean, cv)`` of the scheduled u distribution P(t) at ramp parameter ``t`` in [0, 1].

    The mean is walked GEOMETRICALLY and the CV linearly. Geometric is the right ramp for a quantity
    whose destination is zero: it takes equal RELATIVE steps, so the last rounds -- where the
    constraint rows are stiffest, their norm growing like 1/u, and where the feasible set is thinnest,
    its slack falling like u^3 -- move the targets least.

    ``endMean`` is not 0 and must not be: a zero edge target divides by zero in the fractional
    residual, and the feasible set closes onto the square as u -> 0, so the constraints lose
    transversality on the way in. Stop short and let the COLLAPSE finish the job."""
    t = float(np.clip(t, 0.0, 1.0))
    mean = float(startMean) * (float(endMean) / float(startMean)) ** t
    return mean, (1.0 - t) * float(startCv) + t * float(endCv)
