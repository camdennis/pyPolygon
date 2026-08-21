"""eqSoftBody energy/force for relaxing a backbone polygon (build step 1).

``eqSoftBody`` is the spring model that turns a random star seed (see
``Packing.fromSinglePolygon``) into an equilateral polygon at a chosen perimeter
and area:

    E = (1/2) kEdge * sum_k (l_k - l0)^2  +  (1/2) kArea * (A - A0)^2

with per-edge length springs (rest length l0 = targetEdgeLength = perimeter/n)
plus a single area spring (target A0 = targetArea). The force is -dE/dr.

GEOMETRY IS RAW (no periodic wrap): the seed spreads points across all of
[0,1)^2, so the single polygon spans the box; minimum-image-wrapping its own long
edges would flip them around the periodic boundary and tear the polygon apart.
eqSoftBody is intrinsic single-polygon shape relaxation, so periodicity does not
enter here -- it belongs to the inter-polygon collision phases (step 4 onward).

Positions are flat (2N,), interleaved [x0, y0, x1, y1, ...]; reshaped to (N, 2)
here as r, so vertex k is r[k].
"""

import numpy as np

def backboneEdgeLengths(packing):
    """Per-edge backbone length, raw (no periodic wrap).

    Returns an (N,) array whose entry k is |r[next[k]] - r[k]|, the length of the
    edge leaving vertex k toward its CCW neighbor.
    """
    r = packing.positions.reshape(-1, 2)
    e = r[packing.next] - r
    return np.sqrt(np.einsum("ij,ij->i", e, e))

def backboneArea(packing):
    """Signed shoelace area per polygon (CCW positive), raw coords.

    Returns a (P,) array; A_p = (1/2) sum_{k in p} (x_k * y_next - x_next * y_k).
    """
    r = packing.positions.reshape(-1, 2)
    rn = r[packing.next]
    cross = r[:, 0] * rn[:, 1] - rn[:, 0] * r[:, 1]
    return 0.5 * np.bincount(
        packing.shapeId, weights = cross, minlength = packing.numPolygons
    )

def eqSoftBodyEnergyForce(packing, kEdge, kArea, relative = False):
    """Energy and force of the eqSoftBody spring model (build step 1).

    Absolute form (``relative = False``):
        E = (1/2) kEdge sum_k (l_k - l0)^2 + (1/2) kArea sum_p (A_p - A0_p)^2
    Relative / dimensionless form (``relative = True``):
        E = (1/2) kEdge sum_k (l_k/l0 - 1)^2 + (1/2) kArea sum_p (A_p/A0 - 1)^2

    using the per-polygon ``targetEdgeLength`` (l0) and ``targetArea`` (A0) on the packing. The
    relative form is what the PACKING uses: its terms are O(1) and unit-free, so they balance the
    dimensionless overlap energy instead of being swamped by it (the absolute form is ~l0^2 / A0^2
    weaker, which lets polygons shrink to shed overlap). The absolute form is kept for the free-space
    backbone build, whose wild random-star seed has huge (l - l0) that the stiff relative springs
    would blow up -- the equilateral minimum is identical either way. Returns (energy, force) with
    force a flat (2N,) array = -dE/dr. Pure (does not mutate the packing)."""
    if packing.targetEdgeLength is None or packing.targetArea is None:
        raise ValueError(
            "eqSoftBody needs targetEdgeLength and targetArea set on the packing."
        )

    r = packing.positions.reshape(-1, 2)
    nxt = packing.next
    prv = packing.prev
    shapeId = packing.shapeId

    # Edge spring
    e = r[nxt] - r
    l = np.sqrt(np.einsum("ij,ij->i", e, e))
    safeL = np.where(l > 1e-15, l, 1.0)
    uhat = e / safeL[:, None]
    l0 = packing.targetEdgeLength            # per EDGE, no broadcast
    wEdge = 1.0 / l0 ** 2 if relative else np.ones_like(l0)
    stretch = wEdge * (l - l0)

    energyEdge = 0.5 * kEdge * np.sum(wEdge * (l - l0) ** 2)
    forceEdge = kEdge * (stretch[:, None] * uhat - stretch[prv][:, None] * uhat[prv])

    # Area spring
    area = backboneArea(packing)
    A0 = packing.targetArea
    wArea = 1.0 / A0 ** 2 if relative else np.ones_like(A0)
    areaResidual = wArea * (area - A0)
    energyArea = 0.5 * kArea * np.sum(wArea * (area - A0) ** 2)

    rNext = r[nxt]
    rPrev = r[prv]
    # dA/dr_m = (1/2)(y_next - y_prev, x_prev - x_next)
    gradA = 0.5 * np.stack(
        [rNext[:, 1] - rPrev[:, 1], rPrev[:, 0] - rNext[:, 0]], axis = 1
    )
    forceArea = -(kArea * areaResidual)[shapeId][:, None] * gradA

    force = (forceEdge + forceArea).reshape(-1)
    energy = energyEdge + energyArea
    return energy, force
