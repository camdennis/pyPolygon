"""Phase 2 -- roundify: corner-circle geometry of a backbone polygon.

Each corner is rounded by a circle of radius rho tangent to both incident edges.
Per vertex this computes:
  z       -- circle center (inside the polygon at convex corners, outside at reflex)
  aMinus  -- kiss point on the incoming edge (toward prev[k])
  aPlus   -- kiss point on the outgoing edge (toward next[k])
  t       -- kiss offset from the vertex along each edge, t = rho * cot(theta/2)
  psi     -- arc the corner sweeps, psi = pi - theta  (arc length rho*psi)
  convex  -- True at convex (left-turn) corners, False at reflex

Edge unit vectors (see notes/roundedDefinitions.pdf): eOutHat = e_i (outgoing edge,
v_i -> v_next) and eInHat = e_{z'(i)} (incoming edge, v_prev -> v_i). theta is the wedge
angle arccos(-eInHat . eOutHat) in [0, pi]; the center sits along the bisector
bHat proportional to (eOutHat - eInHat), which points into that wedge -- automatically
the interior for convex corners and the exterior for reflex ones, so no sign flip needed.

rho must be feasible: on every edge the two corner kiss offsets must fit,
t_k + t_next <= edge length. Per Cam, infeasible configs are REJECTED (raise), so rho
must be small enough for the sharp star corners. Geometry is raw (free space), matching
the eqSoftBody build in softBody.py.
"""

import numpy as np
from dataclasses import dataclass

from softBody import backboneEdgeLengths

@dataclass
class CornerGeometry:
    """Per-vertex rounded-corner geometry. z/aMinus/aPlus are (N,2); psi/t/convex (N,)."""
    z: np.ndarray
    aMinus: np.ndarray
    aPlus: np.ndarray
    psi: np.ndarray
    t: np.ndarray
    convex: np.ndarray

def rhoPerVertex(packing, rho):
    """Broadcast a scalar or per-polygon rho to a per-vertex (N,) array."""
    rho = np.asarray(rho, dtype = float)
    if rho.ndim == 0:
        return np.full(packing.numVertices, float(rho))
    return rho[packing.shapeId]                       # per-polygon -> per-vertex

def checkFeasibility(packing, t):
    """Raise if any edge cannot fit its two corner kiss offsets (t_k + t_next > L)."""
    edge = backboneEdgeLengths(packing)
    overrun = t + t[packing.next] - edge
    if np.any(overrun > 1e-12):
        worst = int(np.argmax(overrun))
        raise ValueError(
            f"rho infeasible at edge {worst}: t_k + t_next = "
            f"{t[worst] + t[packing.next[worst]]:.4f} > edge length {edge[worst]:.4f} "
            f"(overrun {overrun[worst]:.3e}); reduce rho."
        )

def cornerGeometry(packing, rho):
    """Rounded-corner geometry for every vertex; ``rho`` is a scalar or per-polygon array.

    Implements the boxes in notes/roundedDefinitions.pdf, with edge unit vectors
        eOutHat = e_i        outgoing edge  (v_i -> v_next)
        eInHat  = e_{z'(i)}  incoming edge  (v_prev -> v_i)
    Returns a CornerGeometry. Raises ValueError if any corner is infeasible.
    """
    r = packing.positions.reshape(-1, 2)
    prv = packing.prev
    nxt = packing.next

    eIn = r - r[prv]
    eOut = r[nxt] - r
    eInHat = eIn / np.sqrt(np.einsum("ij,ij->i", eIn, eIn))[:, None]
    eOutHat = eOut / np.sqrt(np.einsum("ij,ij->i", eOut, eOut))[:, None]

    theta = np.arccos(np.clip(-np.einsum("ij,ij->i", eInHat, eOutHat), -1.0, 1.0))  # arccos(-e_z' . e_i)
    halfTheta = 0.5 * theta
    rhoVert = rhoPerVertex(packing, rho)
    t = rhoVert / np.tan(halfTheta)

    aMinus = r - t[:, None] * eInHat
    aPlus = r + t[:, None] * eOutHat

    bisector = eOutHat - eInHat
    bHat = bisector / np.sqrt(np.einsum("ij,ij->i", bisector, bisector))[:, None]
    z = r + (rhoVert / np.sin(halfTheta))[:, None] * bHat

    psi = np.pi - theta

    convex = (eIn[:, 0] * eOut[:, 1] - eIn[:, 1] * eOut[:, 0]) > 0.0

    checkFeasibility(packing, t)
    return CornerGeometry(z = z, aMinus = aMinus, aPlus = aPlus, psi = psi, t = t,
                          convex = convex)

def roundedPerimeter(packing, cg, rho):
    """Per-polygon rounded perimeter: straight runs between kiss points plus corner arcs.

    Edge k->next contributes its straight run (edge length minus the two kiss offsets)
    and vertex k contributes its arc rho_k*psi_k:
        P_p = sum_{k in p} (L_k - t_k - t_next) + sum_{k in p} rho_k * psi_k.
    """
    edge = backboneEdgeLengths(packing)
    rhoVert = rhoPerVertex(packing, rho)
    perVertex = edge - cg.t - cg.t[packing.next] + rhoVert * cg.psi
    return np.bincount(packing.shapeId, weights = perVertex,
                       minlength = packing.numPolygons)

def roundedArea(packing, cg, rho):
    """Per-polygon rounded area (notes/roundedDefinitions.pdf): the polygon through the
    kiss points plus each corner's circular segment.

    The kiss-point polygon (every corner cut by its chord a^- -> a^+) contributes, per
    corner k, the shoelace of its two edges -- the chord (a^-_k -> a^+_k) and the straight
    run (a^+_k -> a^-_next). Then each corner's circular segment

        Da = (1/2) rho^2 (psi - sin psi)

    is ADDED at convex corners (the arc bulges out past the chord) and SUBTRACTED at reflex
    corners: A_p = A_kissPolygon + sum_k s_k Da_k, with s = +1 convex / -1 reflex.
    """
    am = cg.aMinus
    ap = cg.aPlus
    amNext = cg.aMinus[packing.next]
    chord = am[:, 0] * ap[:, 1] - am[:, 1] * ap[:, 0]           # cross(a^-_k, a^+_k)
    run = ap[:, 0] * amNext[:, 1] - ap[:, 1] * amNext[:, 0]     # cross(a^+_k, a^-_next)
    kissPolygon = np.bincount(packing.shapeId, weights = 0.5 * (chord + run),
                              minlength = packing.numPolygons)

    rhoVert = rhoPerVertex(packing, rho)
    segment = 0.5 * rhoVert ** 2 * (cg.psi - np.sin(cg.psi))    # Da, the circular segment
    sign = np.where(cg.convex, 1.0, -1.0)
    return kissPolygon + np.bincount(packing.shapeId, weights = sign * segment,
                                     minlength = packing.numPolygons)

def cornerAngleGradients(packing):
    """Per corner k, the gradient of the wedge angle theta_k w.r.t. its three vertices.

    Returns (gPrev, gCenter, gNext), each (N,2) = d theta_k / d r_{prev[k], k, next[k]},
    for theta_k = arccos(uHat . wHat) with uHat toward prev[k] and wHat toward next[k].
    """
    r = packing.positions.reshape(-1, 2)
    u = r[packing.prev] - r
    w = r[packing.next] - r
    a = np.sqrt(np.einsum("ij,ij->i", u, u))
    b = np.sqrt(np.einsum("ij,ij->i", w, w))
    uHat = u / a[:, None]
    wHat = w / b[:, None]
    cos = np.einsum("ij,ij->i", uHat, wHat)
    sin = np.sqrt(np.clip(1.0 - cos ** 2, 1e-300, None))
    pu = (wHat - cos[:, None] * uHat) / a[:, None]
    pw = (uHat - cos[:, None] * wHat) / b[:, None]
    gPrev = -pu / sin[:, None]
    gNext = -pw / sin[:, None]
    gCenter = (pu + pw) / sin[:, None]
    return gPrev, gCenter, gNext

def roundedPerimeterGradient(packing, cg, rho):
    """Gradient of each polygon's rounded perimeter: row m is d P_{shapeId[m]} / d r_m,
    shape (N,2). P = sum L - 2 sum t + sum rho * psi; the L part is the backbone edge unit
    vectors, the t / psi part is dP/dtheta_k = rho / sin^2(theta / 2) - rho scattered via theta.
    """
    r = packing.positions.reshape(-1, 2)
    idx = np.arange(packing.numVertices)
    rhoVert = rhoPerVertex(packing, rho)
    theta = np.pi - cg.psi

    grad = np.zeros((packing.numVertices, 2))
    e = r[packing.next] - r
    ehat = e / np.sqrt(np.einsum("ij,ij->i", e, e))[:, None]
    np.add.at(grad, packing.next, ehat)
    np.add.at(grad, idx, -ehat)

    gPrev, gCenter, gNext = cornerAngleGradients(packing)
    weight = (rhoVert / np.sin(0.5 * theta) ** 2 - rhoVert)[:, None]    # dP/dtheta_k
    np.add.at(grad, packing.prev, weight * gPrev)
    np.add.at(grad, idx, weight * gCenter)
    np.add.at(grad, packing.next, weight * gNext)
    return grad

def roundedAreaGradient(packing, cg, rho):
    """Gradient of each polygon's rounded area, matching the notes' decomposition
    A = A_kissPolygon + sum_k s_k Da_k, with Da = (1/2) rho^2 (psi - sin psi).

    Row m is d A_{shapeId[m]} / d r_m, shape (N,2). The kiss-polygon part is the shoelace
    gradient w.r.t. the kiss points a^- = r + t*uHat, a^+ = r + t*wHat (uHat toward prev,
    wHat toward next), pushed through their Jacobians; the segment part scatters
    d psi_k = -d theta_k. dt/dtheta = -rho/(2 sin^2(theta/2)).
    """
    r = packing.positions.reshape(-1, 2)
    prv, nxt = packing.prev, packing.next
    idx = np.arange(packing.numVertices)
    rhoVert = rhoPerVertex(packing, rho)

    u = r[prv] - r
    w = r[nxt] - r
    a = np.sqrt(np.einsum("ij,ij->i", u, u))
    b = np.sqrt(np.einsum("ij,ij->i", w, w))
    uHat = u / a[:, None]
    wHat = w / b[:, None]
    theta = np.pi - cg.psi
    dtDtheta = -rhoVert / (2.0 * np.sin(0.5 * theta) ** 2)

    am, ap = cg.aMinus, cg.aPlus
    rot = lambda v: np.stack([v[:, 1], -v[:, 0]], axis = 1)         # (vy, -vx)
    gAminus = 0.5 * rot(ap - ap[prv])                              # d A_kiss / d a^-_k
    gAplus = 0.5 * rot(am[nxt] - am)                               # d A_kiss / d a^+_k

    dotMinus = np.einsum("ij,ij->i", gAminus, uHat)
    dotPlus = np.einsum("ij,ij->i", gAplus, wHat)
    perpMinus = gAminus - dotMinus[:, None] * uHat                # component perp to uHat
    perpPlus = gAplus - dotPlus[:, None] * wHat
    toA = (cg.t / a)[:, None] * perpMinus
    toB = (cg.t / b)[:, None] * perpPlus

    sign = np.where(cg.convex, 1.0, -1.0)
    wSeg = -sign * 0.5 * rhoVert ** 2 * (1.0 - np.cos(cg.psi))     # segment's theta weight
    weight = (dtDtheta * (dotMinus + dotPlus) + wSeg)[:, None]     # total scatter on d theta

    gPrev, gCenter, gNext = cornerAngleGradients(packing)
    grad = np.zeros((packing.numVertices, 2))
    np.add.at(grad, prv, weight * gPrev + toA)
    np.add.at(grad, idx, weight * gCenter + gAminus + gAplus - toA - toB)
    np.add.at(grad, nxt, weight * gNext + toB)
    return grad
