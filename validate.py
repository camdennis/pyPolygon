"""Monte-Carlo validation of the overlap area.

``mcOverlapArea`` estimates the system's total pairwise overlap area -- the sum over polygon
pairs of |A_i intersect A_j| -- by sampling points in the unit box, counting how many polygons
contain each (under the periodic minimum image), and summing C(count, 2). It is the independent
cross-check for ``overlap.overlapAreas`` (compare to ``sum(overlapAreas(...).values())``).

Note: like the analytic walk, this counts overlaps via boundary geometry, so a polygon wholly
contained in another (no boundary intersections) is the one case where the two can disagree -- not
expected for similar-sized polygons at small rho.
"""

import numpy as np
from matplotlib.path import Path

from geometry import cornerGeometry
from box import wrap
from distributions import asRng
from visualize import roundedBoundary

def getMCOverlapArea(packing, rho, samples = 500000, rng = None, arcSamples = 200):
    """Monte-Carlo estimate of the system's total pairwise overlap area (the unit box has area
    1). Samples points uniformly in the box; for each, counts how many polygons contain it under
    the periodic minimum image and adds C(count, 2). ``rng`` (int seed or Generator) seeds the
    sampling. Returns the estimated total overlap area."""
    rng = asRng(rng)
    cg = cornerGeometry(packing, rho)
    r = packing.positions.reshape(-1, 2)
    paths = []
    centroids = []
    for p in range(packing.numPolygons):
        a = int(packing.startIndices[p])
        b = int(packing.startIndices[p + 1])
        paths.append(Path(roundedBoundary(packing, cg, rho, p, arcSamples)))
        centroids.append(r[a : b].mean(axis = 0))
    pts = rng.random((samples, 2))
    count = np.zeros(samples, dtype = int)
    for p in range(packing.numPolygons):
        if packing.box is not None:
            probe = centroids[p] + wrap(pts - centroids[p], packing.box)
        else:
            probe = pts
        count += paths[p].contains_points(probe)
    return float((count * (count - 1) // 2).mean())
