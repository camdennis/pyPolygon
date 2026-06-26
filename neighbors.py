"""Phase 4 -- neighbor pairs within a per-polygon ball D (brute force, no spatial hash).

The ball radius for vertex i is
    D_i = globalPercentage * (maxEdgeLength + edgeLength[i]) + 2*rho,
where the global maxEdgeLength term guarantees coverage of the largest possible partner
and the +2*rho margin covers the rounded-feature / self-repulsion reach. A vertex pair
(i, j) is a candidate when their minimum-image separation is below max(D_i, D_j).

Same-polygon ADJACENT pairs are dropped (they share an edge); the rest are kept and flagged
``sameShape``. Distances use the box's minimum image (raw difference for box=None). This is
O(N^2) and rebuilt on demand; the critical-displacement rebuild is step 8.

NOTE -- NOT THE FINAL FORM (design revised 2026-06-25). Self-repulsion is being moved OUT of
the neighbor list: a floppy polygon's self-intersection is prevented by repelling its own
non-adjacent vertices, which we will scan DIRECTLY per polygon (own vertices are few, the
repulsion is short-range -- no spatial neighbor tracking needed). So the same-polygon
(``sameShape`` True) pairs kept here are TRANSITIONAL; in Phase 6 ``findNeighbors`` will be
reduced to inter-polygon pairs only, and ``sameShape`` will go away. The +2*rho margin then
covers only the rounded-feature reach for inter-polygon crossings, not self-repulsion.
"""

import numpy as np
from dataclasses import dataclass

from box import wrap
from geometry import rhoPerVertex

@dataclass
class Neighbors:
    """Candidate vertex pairs. ``pairs`` is (M, 2) with i < j; ``sameShape`` is (M,) bool."""
    pairs: np.ndarray
    sameShape: np.ndarray

def ballRadius(packing, rho, globalPercentage = 1.0):
    """Per-vertex neighbor ball radius D_i = gp * (maxEdge + edge_i) + 2 * rho, shape (N,)."""
    edge = packing.targetEdgeLength[packing.shapeId]
    maxEdge = packing.targetEdgeLength.max()
    return globalPercentage * (maxEdge + edge) + 2.0 * rhoPerVertex(packing, rho)

def findNeighbors(packing, rho, globalPercentage = 1.0):
    """Brute-force candidate vertex pairs within the per-polygon ball D. Returns Neighbors.

    Minimum-image separations (via packing.box, raw if box is None); a pair is kept when
    dist < max(D_i, D_j), excluding same-polygon adjacent pairs.
    """
    r = packing.positions.reshape(-1, 2)
    diff = r[None, :, :] - r[:, None, :]
    if packing.box is not None:
        diff = wrap(diff, packing.box)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    D = ballRadius(packing, rho, globalPercentage)
    within = dist < np.maximum(D[:, None], D[None, :])
    i, j = np.where(np.triu(within, k = 1))

    sameShape = packing.shapeId[i] == packing.shapeId[j]
    adjacent = sameShape & ((packing.next[i] == j) | (packing.next[j] == i))
    keep = ~adjacent
    return Neighbors(pairs = np.stack([i[keep], j[keep]], axis = 1),
                     sameShape = sameShape[keep])
