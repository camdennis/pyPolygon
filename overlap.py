"""Phase 6a/6b -- overlap area between rounded polygons (Cam's "Area overlap" formulation).

Area(A and B) = 1/2 eps_{ab} oint X^a dX^b over the overlap boundary d(A cap B), accumulated
per boundary feature relative to a reference point R = the first backbone vertex of polygon A
(the lower-indexed polygon, A < B). Each feature piece from point P to Q contributes

  edge:  h_e(P, Q) = 1/2 eps_{ab} (P - R)^a (Q - R)^b
  arc:   h_a(P, Q) = h_e(P, Q) + 1/2 rho^2 (theta - sin theta),
                     theta = atan2( eps(P - z, Q - z), (P - z).(Q - z) )

i.e. the chord-from-R plus the circular segment (theta the signed central angle of the sub-arc).
The total is R-invariant (the R terms telescope around the closed loop); R keeps the arithmetic
local and matches the gradient bookkeeping to come.

The overlap boundary is the kept runs of dA inside B and dB inside A, selected by the
``entering`` flag, each from an intersection to its outersection (next intersection in sigma order). The
walk (overlapAreas) sums them run by run; the parallel split (overlapAreasParallel) is the same
total reorganized as U_out (partial features at the intersections, per run) + U_in (whole interior
features, per feature). [Cam's handwritten notes, "Area overlap" pages 16-21.]
"""

import bisect

import numpy as np

from box import minImageShift
from intersections import arcSweepSign


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def edgeArea(P, Q, R):
    """h_e: 1/2 eps(P - R, Q - R), the chord triangle relative to R."""
    # notes roundedPolygons.tex eq (4.1): h_e (chord triangle from R)
    return 0.5 * _cross(P - R, Q - R)

def arcArea(P, Q, R, z, rho):
    """h_a: chord-from-R plus circular segment 1/2 rho^2 (theta - sin theta), theta the signed
    central angle of the sub-arc P->Q about z."""
    dP = P - z
    dQ = Q - z
    theta = np.arctan2(_cross(dP, dQ), dP @ dQ)
    # notes roundedPolygons.tex eq (4.1)+(4.2): arc piece h_e + h_a, h_a = 1/2 rho^2 (theta - sin theta) the segment
    return 0.5 * _cross(P - R, Q - R) + 0.5 * rho * rho * (theta - np.sin(theta))

def featurePoint(features, vertex, onArc, frac, shift):
    """The point at fraction ``frac`` along ``vertex``'s feature (edge or arc), center/edge
    offset by ``shift``. frac 0 / 1 give the feature's start / end kiss points."""
    if not onArc:
        return features["edgeP0"][vertex] + shift + frac * features["edgeU"][vertex]
    z = features["z"][vertex] + shift
    sign = arcSweepSign(features["phiMinus"][vertex], features["phiPlus"][vertex],
                        features["psi"][vertex])
    phi = features["phiMinus"][vertex] + sign * frac * features["psi"][vertex]
    return z + features["rho"][vertex] * np.array([np.cos(phi), np.sin(phi)])

def featureArea(features, vertex, onArc, P, Q, R, shift):
    """h(P, Q): h_e for an edge, h_a for an arc (the arc's center is z[vertex] + shift)."""
    # notes roundedPolygons.tex eq (4.3): feature selector h = h_e (edge) / h_e + h_a (arc)
    if not onArc:
        return edgeArea(P, Q, R)
    return arcArea(P, Q, R, features["z"][vertex] + shift, features["rho"][vertex])

def _feature(packing, polygon, f):
    """(vertex, onArc) for global feature index f (mod 2n) of ``polygon``."""
    start = int(packing.startIndices[polygon])
    fl = f % (2 * (int(packing.startIndices[polygon + 1]) - start))
    return start + fl // 2, (fl % 2 == 0)

def runArea(packing, features, polygon, sigStart, ptStart, sigEnd, ptEnd, shift, R):
    """Area integral over an inside-run of d(polygon) from (sigStart, ptStart) to
    (sigEnd, ptEnd), traversing increasing sigma (cyclically). Sums h over the partial start
    feature, the whole interior features, and the partial end feature."""
    twoN = 2 * (int(packing.startIndices[polygon + 1]) - int(packing.startIndices[polygon]))
    if sigEnd < sigStart:
        sigEnd += twoN
    fStart = int(np.floor(sigStart))
    fEnd = int(np.floor(sigEnd))
    total = 0.0
    P = ptStart
    # notes roundedPolygons.tex eq (4.4): sum h over the run (partial start, interior, partial end)
    for f in range(fStart, fEnd + 1):
        vertex, onArc = _feature(packing, polygon, f)
        Q = ptEnd if f == fEnd else featurePoint(features, vertex, onArc, 1.0, shift)
        total += featureArea(features, vertex, onArc, P, Q, R, shift)
        P = Q
    return total

def _insideRuns(packing, pairIntersections):
    """Inside-runs of one overlap: dA runs inside B (from ``entering`` intersections, along sigmaA)
    and dB runs inside A (from non-``entering`` intersections, along sigmaB). Each run is
    (polygon, shift, sigStart, ptStart, sigEnd, ptEnd)."""
    r = packing.positions.reshape(-1, 2)
    polyA = int(packing.shapeId[pairIntersections[0].i])
    polyB = int(packing.shapeId[pairIntersections[0].j])
    shiftB = minImageShift(r[pairIntersections[0].j] - r[pairIntersections[0].i], packing.box)
    m = len(pairIntersections)
    runs = []
    bySigA = sorted(pairIntersections, key = lambda c: c.sigmaA)
    for k in range(m):
        if bySigA[k].entering:
            nx = bySigA[(k + 1) % m]
            runs.append((polyA, np.zeros(2), bySigA[k].sigmaA, bySigA[k].point, nx.sigmaA, nx.point))
    bySigB = sorted(pairIntersections, key = lambda c: c.sigmaB)
    for k in range(m):
        if not bySigB[k].entering:
            nx = bySigB[(k + 1) % m]
            runs.append((polyB, shiftB, bySigB[k].sigmaB, bySigB[k].point, nx.sigmaB, nx.point))
    return runs

def _groupByPair(packing, intersections):
    groups = {}
    for c in intersections:
        groups.setdefault((int(packing.shapeId[c.i]), int(packing.shapeId[c.j])), []).append(c)
    return groups

def overlapAreas(packing, features, intersections):
    """Overlap area per polygon pair via the boundary walk -- sum the area integral over the
    inside-runs. Returns {(polyA, polyB): area}."""
    r = packing.positions.reshape(-1, 2)
    areas = {}
    for (polyA, polyB), g in _groupByPair(packing, intersections).items():
        R = r[int(packing.startIndices[polyA])]
        areas[(polyA, polyB)] = sum(
            runArea(packing, features, poly, sS, pS, sE, pE, shift, R)
            for (poly, shift, sS, pS, sE, pE) in _insideRuns(packing, g))
    return areas

def _insideAreaBinary(packing, features, polygon, pairIntersections, side, shift, R):
    """Area of one polygon's boundary lying inside its partner, computed feature by feature
    (the parallel shape). Each feature reads its inside/outside state at its start from the
    predecessor intersection -- a binary search in the intersections already sorted by sigma -- then
    sums its inside sub-intervals. ``side`` selects the coordinate / orientation: 'A' uses sigmaA
    with the stored ``entering`` flag (dA enters B); 'B' uses sigmaB with its negation (dB
    enters A)."""
    entries = sorted(
        ((c.sigmaA if side == "A" else c.sigmaB,
          c.entering if side == "A" else (not c.entering),
          c.point) for c in pairIntersections),
        key = lambda e: e[0])
    sigmas = [e[0] for e in entries]
    numIntersections = len(entries)
    start = int(packing.startIndices[polygon])
    twoN = 2 * (int(packing.startIndices[polygon + 1]) - start)
    area = 0.0
    for f in range(twoN):
        vertex, onArc = _feature(packing, polygon, f)
        lo = bisect.bisect_left(sigmas, f)
        hi = bisect.bisect_left(sigmas, f + 1)
        inside = entries[(lo - 1) % numIntersections][1]
        P = featurePoint(features, vertex, onArc, 0.0, shift)
        for idx in range(lo, hi):
            if inside:
                area += featureArea(features, vertex, onArc, P, entries[idx][2], R, shift)
            P = entries[idx][2]
            inside = not inside
        if inside:
            Q = featurePoint(features, vertex, onArc, 1.0, shift)
            area += featureArea(features, vertex, onArc, P, Q, R, shift)
    return area

def overlapAreasParallel(packing, features, intersections):
    """Overlap area per polygon pair, computed feature by feature -- the parallel-friendly shape
    (one independent pass per feature, interior and partial sub-intervals together). Each feature
    finds its inside/outside state by binary-searching the sorted intersections rather than walking
    runs, so there is no interior list. Cross-checked against overlapAreas. Returns
    {(polyA, polyB): area}."""
    r = packing.positions.reshape(-1, 2)
    areas = {}
    for (polyA, polyB), g in _groupByPair(packing, intersections).items():
        R = r[int(packing.startIndices[polyA])]
        shiftB = minImageShift(r[g[0].j] - r[g[0].i], packing.box)
        areas[(polyA, polyB)] = (
            _insideAreaBinary(packing, features, polyA, g, "A", np.zeros(2), R)
            + _insideAreaBinary(packing, features, polyB, g, "B", shiftB, R))
    return areas
