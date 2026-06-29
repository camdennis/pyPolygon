"""Phase 6c -- gradient of the rounded-polygon overlap area (the overlap force).

The overlap energy uses only the overlap area for now (the other springs are zero), so the
force is ``-d A_cap / d v``. We use the shape-derivative form of that gradient: by
``dA = oint dX x dX`` over the (closed) overlap boundary, each kept feature piece from P to Q
contributes through the MATERIAL velocity of its own feature -- the corner kiss points / arc
centers move with their polygon's vertices -- and the intersection points need no separate
gradient (their motion cancels around the loop). Per piece, with d = Q - P:

  edge  E_m (from a^+_m to a^-_{next}):  contribution = 1/2 ( dX_P + dX_Q ) x d,
        dX(s) the material velocity of the edge point at parameter s (= (1-s) da^+_m + s da^-_next);
  arc   C_m (center z_m):                contribution = d z_m x d   (the circle translates rigidly).

The per-vertex gradient routes these through the corner Jacobians d a^+/d v, d a^-/d v, d z/d v
(notes/overlapForce.pdf, sec. 3), FD-validated against d(overlapAreas)/dv. [overlapForce.tex]
"""

import numpy as np

from geometry import cornerGeometry, rhoPerVertex, cornerAngleGradients
from overlap import _groupByPair, _insideRuns, _feature, featurePoint


def _outer(a, b):
    return np.einsum("na,nb->nab", a, b)

def _proj(eHat):
    # notes roundedPolygons.tex eq (6.1): projection Pi_u = I - u_hat u_hat^T
    return np.eye(2)[None, :, :] - _outer(eHat, eHat)

class CornerJacobians:
    """Per-vertex 2x2 blocks of d a^+_k, d a^-_k, d z_k with respect to v_prev, v_k, v_next.
    Field X has X.aPlusPrev[k] = d a^+_k / d v_{prev[k]} (rows alpha, cols gamma), etc."""

    def __init__(self, packing, cg, rho):
        r = packing.positions.reshape(-1, 2)
        prv, nxt = packing.prev, packing.next
        rhoVert = rhoPerVertex(packing, rho)

        # notes eq (1.1) edge units; their derivative Pi/|e| is eq (6.2), used in the a^pm blocks
        eOut = r[nxt] - r
        lOut = np.sqrt(np.einsum("ij,ij->i", eOut, eOut))
        eOutH = eOut / lOut[:, None]
        eIn = r - r[prv]
        lIn = np.sqrt(np.einsum("ij,ij->i", eIn, eIn))
        eInH = eIn / lIn[:, None]

        theta = np.pi - cg.psi
        half = 0.5 * theta
        sinH, cosH = np.sin(half), np.cos(half)
        # notes eq (6.7): dt/dtheta
        dtdth = -rhoVert / (2.0 * sinH ** 2)
        c = rhoVert / sinH
        # notes eq (6.8): dc/dtheta
        dcdth = -rhoVert * cosH / (2.0 * sinH ** 2)

        gPrev, gCtr, gNext = cornerAngleGradients(packing)
        t = cg.t

        pOut = _proj(eOutH)
        pIn = _proj(eInH)
        b = eOutH - eInH
        nb = np.sqrt(np.einsum("ij,ij->i", b, b))
        bH = b / nb[:, None]
        pB = _proj(bH)

        tOut = (t / lOut)[:, None, None]
        tIn = (t / lIn)[:, None, None]

        # notes eq (6.4): d a^+_k / d v (the Pi terms are d e_hat, eq (6.2))
        self.aPlusPrev = _outer(eOutH, dtdth[:, None] * gPrev)
        self.aPlusSelf = np.eye(2)[None, :, :] + _outer(eOutH, dtdth[:, None] * gCtr) - tOut * pOut
        self.aPlusNext = _outer(eOutH, dtdth[:, None] * gNext) + tOut * pOut

        # notes eq (6.5): d a^-_k / d v
        self.aMinusPrev = -_outer(eInH, dtdth[:, None] * gPrev) + tIn * pIn
        self.aMinusSelf = np.eye(2)[None, :, :] - _outer(eInH, dtdth[:, None] * gCtr) - tIn * pIn
        self.aMinusNext = -_outer(eInH, dtdth[:, None] * gNext)

        # notes eq (6.3): d b_hat_k / d v
        dbPrev = (1.0 / (nb * lIn))[:, None, None] * np.einsum("nab,nbc->nac", pB, pIn)
        dbNext = (1.0 / (nb * lOut))[:, None, None] * np.einsum("nab,nbc->nac", pB, pOut)
        dbSelf = -(dbPrev + dbNext)
        cc = c[:, None, None]
        # notes eq (6.6): d z_k / d v
        self.zPrev = _outer(bH, dcdth[:, None] * gPrev) + cc * dbPrev
        self.zSelf = np.eye(2)[None, :, :] + _outer(bH, dcdth[:, None] * gCtr) + cc * dbSelf
        self.zNext = _outer(bH, dcdth[:, None] * gNext) + cc * dbNext

def _crossCols(jac, d):
    """For a 2x2 Jacobian ``jac`` (rows alpha, cols gamma) and chord ``d``, the (gamma,) vector
    whose entry is the 2D cross (column_gamma x d)."""
    # notes roundedPolygons.tex eq (7.5): column-gamma of the Jacobian crossed with chord d
    return jac[0] * d[1] - jac[1] * d[0]

def _accumEdge(grad, cj, packing, m, sbar, d):
    """Distribute an edge piece (corner m, mean parameter sbar, chord d) onto the four vertices its
    endpoints depend on (m's prev/self/next via a^+_m, and next's self/next via a^-_next)."""
    # notes eq (7.6) edge case: d f_m / d v = (1-sbar) d a^+_m + sbar d a^-_{z(m)}
    n2 = int(packing.next[m])
    s = sbar
    grad[int(packing.prev[m])] += _crossCols((1.0 - s) * cj.aPlusPrev[m], d)
    grad[m] += _crossCols((1.0 - s) * cj.aPlusSelf[m] + s * cj.aMinusPrev[n2], d)
    grad[n2] += _crossCols((1.0 - s) * cj.aPlusNext[m] + s * cj.aMinusSelf[n2], d)
    grad[int(packing.next[n2])] += _crossCols(s * cj.aMinusNext[n2], d)

def _accumArc(grad, cj, packing, m, d):
    """Distribute an arc piece (corner m, chord d) onto m's prev/self/next via d z_m."""
    # notes eq (7.6) arc case: d f_m / d v = d z_m / d v
    grad[int(packing.prev[m])] += _crossCols(cj.zPrev[m], d)
    grad[m] += _crossCols(cj.zSelf[m], d)
    grad[int(packing.next[m])] += _crossCols(cj.zNext[m], d)

def _runGradient(packing, features, polygon, sigStart, ptStart, sigEnd, ptEnd, shift, cj, grad):
    """Accumulate the area gradient over one inside-run, feature by feature (mirrors runArea)."""
    twoN = 2 * (int(packing.startIndices[polygon + 1]) - int(packing.startIndices[polygon]))
    if sigEnd < sigStart:
        sigEnd += twoN
    fStart = int(np.floor(sigStart))
    fEnd = int(np.floor(sigEnd))
    P = ptStart
    for f in range(fStart, fEnd + 1):
        vertex, onArc = _feature(packing, polygon, f)
        lamStart = (sigStart - fStart) if f == fStart else 0.0
        if f == fEnd:
            Q = ptEnd
            lamEnd = sigEnd - fEnd
        else:
            Q = featurePoint(features, vertex, onArc, 1.0, shift)
            lamEnd = 1.0
        d = Q - P
        if onArc:
            _accumArc(grad, cj, packing, vertex, d)
        else:
            _accumEdge(grad, cj, packing, vertex, 0.5 * (lamStart + lamEnd), d)
        P = Q

def overlapAreaGradient(packing, features, intersections, cg, rho):
    """Per-vertex gradient d(total overlap area)/d v as an (numVertices, 2) array, via the
    shape-derivative walk over the inside-runs of every overlapping pair."""
    cj = CornerJacobians(packing, cg, rho)
    grad = np.zeros((packing.numVertices, 2))
    # notes eq (7.5): sum over overlapping pairs and their kept feature pieces
    for (polyA, polyB), g in _groupByPair(packing, intersections).items():
        for (poly, shift, sS, pS, sE, pE) in _insideRuns(packing, g):
            _runGradient(packing, features, poly, sS, pS, sE, pE, shift, cj, grad)
    return grad

def overlapForces(packing, features, intersections, cg, rho):
    """Overlap force on every vertex, ``-d(total overlap area)/d v`` (the overlap energy is the
    overlap area with the other springs zero). Returns an (numVertices, 2) array."""
    return -overlapAreaGradient(packing, features, intersections, cg, rho)
