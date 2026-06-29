"""Phase 5 -- boundary intersections (ee/ea/ae/aa) between rounded polygons.

From notes/intersections.tex. Each rounded boundary is a CCW alternation of corner arcs
C_k (center z_k, radius rho_k, from a^-_k to a^+_k, bulging toward V_k) and straight edges
E_k (from a^+_k to a^-_{next[k]}). Two polygons intersect at four feature combinations:

  ee  edge x edge   -- segment / segment
  ea  edge x arc    -- segment / circle, point on the arc
  ae  arc x edge    -- circle / segment, point on the arc
  aa  arc x arc     -- circle / circle, point on both arcs

Each candidate is filtered by the segment range [0,1] and the arc-validity test
(R - z).mHat >= rho cos(psi/2), with mHat = (V - z)/|V - z|. The search is driven by the
Phase-4 neighbor list: only the candidate vertex pairs are tested (no O(nA*nB) sweep), and
each partner vertex is shifted to its single minimum image (we never check more than one).
The outersection ordering / Phase-6 overlap integration build on top of these intersections.
"""

import numpy as np
from dataclasses import dataclass

from box import minImageShift
from geometry import rhoPerVertex

def boundaryFeatures(packing, cg, rho):
    """Per-vertex boundary features as a dict of arrays:
    edgeP0 = a^+_k, edgeU = a^-_{next} - a^+_k (the straight edge E_k); z, rho, the
    arc-validity direction mHat_k = (v_k - z_k)/|.|, and cosHalf_k = cos(psi_k/2). For the
    boundary coordinate sigma it also carries the arc sweep psi and the end angles
    phiMinus/phiPlus = atan2(a^-/a^+ - z), the bounds of the a^- -> a^+ sweep."""
    r = packing.positions.reshape(-1, 2)
    rhoVert = rhoPerVertex(packing, rho)
    # notes roundedPolygons.tex eq (3.1): edge P_k(s)=a^+_k+s u_k and arc Q_k(phi) parametrization
    mVec = r - cg.z
    mHat = mVec / np.sqrt(np.einsum("ij,ij->i", mVec, mVec))[:, None]
    aMinusRel = cg.aMinus - cg.z
    aPlusRel = cg.aPlus - cg.z
    return {
        "edgeP0": cg.aPlus,
        "edgeU": cg.aMinus[packing.next] - cg.aPlus,
        "z": cg.z,
        "rho": rhoVert,
        "mHat": mHat,
        "cosHalf": np.cos(0.5 * cg.psi),
        "psi": cg.psi,
        "phiMinus": np.arctan2(aMinusRel[:, 1], aMinusRel[:, 0]),
        "phiPlus": np.arctan2(aPlusRel[:, 1], aPlusRel[:, 0]),
    }

def segSegParams(p0, u, q0, v):
    """(s, t) solving p0 + s u = q0 + t v, or None if (near) parallel."""
    # notes roundedPolygons.tex eq (3.3): edge-edge intersection fractions s_k, s_ell
    denom = u[0] * v[1] - u[1] * v[0]
    if abs(denom) < 1e-14:
        return None
    w = q0 - p0
    s = (w[0] * v[1] - w[1] * v[0]) / denom
    t = (w[0] * u[1] - w[1] * u[0]) / denom
    return s, t

def lineCircleParams(p0, u, z, rho):
    """The s-roots of |p0 + s u - z| = rho (up to two), or [] if the line misses."""
    # notes roundedPolygons.tex eq (3.4): edge-arc roots S_pm
    w = p0 - z
    A = u @ u
    B = 2.0 * (w @ u)
    C = w @ w - rho * rho
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return []
    sq = np.sqrt(disc)
    return [(-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)]

def circleCirclePoints(z1, rho1, z2, rho2):
    """The (up to two) intersection points of circles (z1,rho1) and (z2,rho2), or []."""
    # notes roundedPolygons.tex eq (3.5): arc-arc points X_pm (radical line)
    delta = z2 - z1
    d = np.sqrt(delta @ delta)
    if d < 1e-14 or d > rho1 + rho2 or d < abs(rho1 - rho2):
        return []
    a = (d * d + rho1 * rho1 - rho2 * rho2) / (2.0 * d)
    h = np.sqrt(max(rho1 * rho1 - a * a, 0.0))
    nHat = delta / d
    nPerp = np.array([nHat[1], -nHat[0]])
    foot = z1 + a * nHat
    return [foot + h * nPerp, foot - h * nPerp]

def onArc(point, z, rho, mHat, cosHalf):
    """Arc-validity: (point - z) . mHat >= rho * cos(psi/2)."""
    # notes roundedPolygons.tex eq (1.3): arc-validity test (AVT)
    return float((point - z) @ mHat) >= rho * cosHalf

def intersectionsAtPair(features, i, j, shift):
    """Boundary intersections between vertex i's two features (arc C_i, edge E_i) and vertex j's
    (arc C_j, edge E_j), with j's features translated by ``shift`` (j's minimum image
    relative to i). Tests the four combinations ee/ea/ae/aa; returns a list of
    (point (2,), type, i, j).
    """
    eP0, eU = features["edgeP0"], features["edgeU"]
    z, rho, mHat, cosHalf = features["z"], features["rho"], features["mHat"], features["cosHalf"]
    ep0i, eui = eP0[i], eU[i]
    zi, ri, mi, ci = z[i], rho[i], mHat[i], cosHalf[i]
    ep0j, euj = eP0[j] + shift, eU[j]
    zj, rj, mj, cj = z[j] + shift, rho[j], mHat[j], cosHalf[j]

    out = []
    res = segSegParams(ep0i, eui, ep0j, euj)
    if res is not None and 0.0 <= res[0] <= 1.0 and 0.0 <= res[1] <= 1.0:
        out.append((ep0i + res[0] * eui, "ee", i, j))

    for s in lineCircleParams(ep0i, eui, zj, rj):
        if 0.0 <= s <= 1.0:
            R = ep0i + s * eui
            if onArc(R, zj, rj, mj, cj):
                out.append((R, "ea", i, j))

    for s in lineCircleParams(ep0j, euj, zi, ri):
        if 0.0 <= s <= 1.0:
            R = ep0j + s * euj
            if onArc(R, zi, ri, mi, ci):
                out.append((R, "ae", i, j))

    for R in circleCirclePoints(zi, ri, zj, rj):
        if onArc(R, zi, ri, mi, ci) and onArc(R, zj, rj, mj, cj):
            out.append((R, "aa", i, j))
    return out

@dataclass
class Intersection:
    """One boundary intersection of polygon A with polygon B. ``point`` is in A's image frame;
    ``kind`` is ee/ea/ae/aa; ``i``/``j`` are the A-/B-vertices owning the two features.
    ``sigmaA``/``sigmaB`` are the global CCW boundary coordinates on A and B -- featureIndex +
    fraction, with features numbered C_k = 2k, E_k = 2k+1 within each polygon -- so sorting
    intersections of a pair by sigmaA gives the order they are met walking A's boundary CCW.
    ``entering`` is True when dA crosses from outside B to inside B here (tangent_A x tangent_B
    < 0); dB then enters A here iff ``entering`` is False. Used by the Phase-6a overlap walk."""
    point: np.ndarray
    kind: str
    i: int
    j: int
    sigmaA: float
    sigmaB: float
    entering: bool

def arcFraction(phiR, phiMinus, phiPlus, psi):
    """Fraction in [0,1] from a^- to a^+ along the corner arc for a circle point at angle
    phiR about the center. Traversal sense is whichever makes the a^- -> a^+ sweep equal psi
    (not 2*pi - psi), so it is correct for convex and reflex corners alike."""
    twoPi = 2.0 * np.pi
    ccwSweep = (phiPlus - phiMinus) % twoPi
    cwSweep = (phiMinus - phiPlus) % twoPi
    if abs(ccwSweep - psi) <= abs(cwSweep - psi):
        # a^- -> a^+ is CCW about the center
        # Convex
        return ((phiR - phiMinus) % twoPi) / psi
    # a^- -> a^+ is CW about the center
    # Concave
    return ((phiMinus - phiR) % twoPi) / psi

def boundaryCoordinate(packing, features, point, onArc, vertex, shift):
    """Global CCW boundary coordinate sigma = featureIndex + lambda for ``point`` lying on
    ``vertex``'s arc C (onArc True) or outgoing edge E (onArc False), with this polygon's
    features offset by ``shift`` (0 for A, the pair shift for B). Features are numbered
    C_k = 2k, E_k = 2k+1 within the polygon (local index = vertex - polygon start)."""
    # Shift is for PBCs
    # notes roundedPolygons.tex eq (3.2): boundary coordinate sigma = feature index + lambda
    local = vertex - packing.startIndices[packing.shapeId[vertex]]
    if onArc:
        z = features["z"][vertex] + shift
        phiR = np.arctan2(point[1] - z[1], point[0] - z[0])
        lam = arcFraction(phiR, features["phiMinus"][vertex], features["phiPlus"][vertex],
                          features["psi"][vertex])
        return 2.0 * local + lam
    a0 = features["edgeP0"][vertex] + shift
    u = features["edgeU"][vertex]
    lam = float((point - a0) @ u) / float(u @ u)
    return 2.0 * local + 1.0 + lam

def arcSweepSign(phiMinus, phiPlus, psi):
    """+1 if the corner arc a^- -> a^+ is traversed CCW about the center (phi increasing),
    -1 if CW. Matches arcFraction's branch selection (+1 convex, -1 reflex)."""
    twoPi = 2.0 * np.pi
    ccwSweep = (phiPlus - phiMinus) % twoPi
    cwSweep = (phiMinus - phiPlus) % twoPi
    return 1.0 if abs(ccwSweep - psi) <= abs(cwSweep - psi) else -1.0

def featureTangent(features, vertex, onArc, point, shift):
    """Unit tangent of ``vertex``'s feature at ``point``, in the CCW boundary direction. Edge:
    the edge direction edgeU. Arc: +-90deg from the radius (point - center), sign from the
    arc's sweep sense. ``shift`` offsets the center only (directions are shift-invariant)."""
    if not onArc:
        u = features["edgeU"][vertex]
        return u / np.sqrt(u @ u)
    rel = point - (features["z"][vertex] + shift)
    sign = arcSweepSign(features["phiMinus"][vertex], features["phiPlus"][vertex],
                        features["psi"][vertex])
    tang = sign * np.array([-rel[1], rel[0]])
    return tang / np.sqrt(tang @ tang)

def findIntersections(packing, features, neighbors):
    """All boundary intersections between DIFFERENT polygons, driven by the neighbor list.

    Walks only the cross-polygon candidate pairs (``neighbors.sameShape`` False) and tests
    the four feature combinations at each, shifting the partner vertex j to its single
    minimum image relative to i. Returns a list of Intersection records, each tagged with its
    boundary coordinates sigmaA/sigmaB.

    Same-polygon pairs are skipped here: self-repulsion is handled separately, scanned
    directly per polygon (see the NOTE in neighbors.py). Once findNeighbors is reduced to
    inter-polygon pairs (Phase 6), the ``sameShape`` filter here becomes a no-op -- this
    inter-polygon-only behavior is the intended final form.
    """
    r = packing.positions.reshape(-1, 2)
    out = []
    for i, j in neighbors.pairs[~neighbors.sameShape]:
        shift = minImageShift(r[j] - r[i], packing.box)
        for point, kind, _, _ in intersectionsAtPair(features, i, j, shift):
            onArcA = kind in ("ae", "aa")
            onArcB = kind in ("ea", "aa")
            sigmaA = boundaryCoordinate(packing, features, point, onArcA, i, np.zeros(2))
            sigmaB = boundaryCoordinate(packing, features, point, onArcB, j, shift)
            tA = featureTangent(features, i, onArcA, point, np.zeros(2))
            tB = featureTangent(features, j, onArcB, point, shift)
            # dA enters B when tA x tB < 0
            entering = (tA[0] * tB[1] - tA[1] * tB[0]) < 0.0
            out.append(Intersection(point, kind, i, j, sigmaA, sigmaB, entering))
    return out

def outersections(packing, intersections):
    """Group intersections by polygon pair and order each group CCW along dA for outersection
    pairing. The pair key is (polyA, polyB) = (shapeId[i], shapeId[j]); within a group the
    intersections are sorted by sigmaA (the CCW boundary coordinate on A), so the outersection of
    the m-th intersection -- the next one met walking dA -- is the (m+1 mod M)-th. Returns a dict
    {(polyA, polyB): [intersections sorted by sigmaA]}.
    """
    shapeId = packing.shapeId
    groups = {}
    for c in intersections:
        groups.setdefault((int(shapeId[c.i]), int(shapeId[c.j])), []).append(c)
    return {key: sorted(g, key = lambda c: c.sigmaA) for key, g in groups.items()}
