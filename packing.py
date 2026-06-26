"""The Packing container: flat-array state for a packing of N polygons.

Uses a flat CSR layout (CUDA-portable) so values cross-check directly:

  positions        (2N,)  vertex coords, interleaved [x0, y0, x1, y1, ...]
  shapeId          (N,)   polygon index each vertex belongs to
  startIndices     (P+1,) CSR start offset of each polygon's vertex block
  next, prev       (N,)   cyclic neighbor vertex index within a polygon
  targetEdgeLength (P,)   per-polygon target backbone edge length
  targetArea       (P,)   per-polygon target area
  targetPerimeter  (P,)   per-polygon target perimeter
  force, velocities(2N,)  integrator state

Scalars: rho (corner radius), box (Box), energyType (EnergyType). Per the plan,
K_adh / K_A / K_P are global and live with the energy code, not on the Packing.
"""

import numpy as np
from enums import EnergyType


def buildConnectivity(startIndices):
    """Derive (shapeId, next, prev) from CSR ``startIndices``.

    Polygon p owns vertices [startIndices[p], startIndices[p + 1]); its boundary
    is a cycle, so ``next`` advances one vertex (wrapping back to the polygon's
    first) and ``prev`` steps back one (wrapping to its last).

    Returns three int arrays of length N = startIndices[-1].
    """
    startIndices = np.asarray(startIndices, dtype = int)
    numVertices = int(startIndices[-1])
    shapeId = np.empty(numVertices, dtype = int)
    nxt = np.empty(numVertices, dtype = int)
    prv = np.empty(numVertices, dtype = int)
    for p in range(startIndices.size - 1):
        a = startIndices[p]
        b = startIndices[p + 1]
        m = b - a
        local = np.arange(m)
        shapeId[a : b] = p
        nxt[a : b] = a + (local + 1) % m
        prv[a : b] = a + (local - 1) % m
    return shapeId, nxt, prv


class Packing:
    """Flat-array state for a packing of N polygons (backbone geometry for now)."""

    def __init__(
        self,
        positions,
        startIndices,
        box = None,
        energyType = EnergyType.eqSoftBody,
        rho = None,
        targetEdgeLength = None,
        targetArea = None,
        targetPerimeter = None,
    ):
        self.positions = np.array(positions, dtype = float).reshape(-1)
        self.startIndices = np.asarray(startIndices, dtype = int)
        self.numVertices = self.positions.size // 2
        self.numPolygons = self.startIndices.size - 1
        self.shapeId, self.next, self.prev = buildConnectivity(self.startIndices)

        self.box = box  # None == free space (eqSoftBody); a Box for the periodic packing
        self.energyType = energyType
        self.rho = rho

        numPolygons = self.numPolygons
        self.targetEdgeLength = self.asPerPolygon(targetEdgeLength, numPolygons)
        self.targetArea = self.asPerPolygon(targetArea, numPolygons)
        self.targetPerimeter = self.asPerPolygon(targetPerimeter, numPolygons)

        self.force = np.zeros_like(self.positions)
        self.velocities = np.zeros_like(self.positions)
        self.energy = 0.0

    @staticmethod
    def asPerPolygon(value, numPolygons):
        """Broadcast a scalar / array / None target into a length-P float array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype = float)
        if arr.ndim == 0:
            return np.full(numPolygons, float(arr))
        return arr.reshape(numPolygons)

    @classmethod
    def fromSinglePolygon(
        cls,
        n,
        rng = None,
        center = (0.5, 0.5),
        box = None,
        energyType = EnergyType.eqSoftBody,
        rho = None,
        targetEdgeLength = None,
        targetArea = None,
        targetPerimeter = None,
    ):
        """Build a one-polygon Packing: n points in the unit box, ordered CCW.

        Scatters ``n`` points uniformly in [0, 1)^2, then orders them
        counter-clockwise by angle about ``center`` (default the box center
        (0.5, 0.5)) to form a (generally irregular, star-shaped) backbone. This
        is the starting configuration that the eqSoftBody minimizer relaxes
        toward equal edge lengths and the target area (build step 1).

        ``rng`` may be a numpy Generator or an int seed (for reproducibility);
        None draws a fresh default_rng().
        """
        if rng is None or isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)
        cx, cy = center
        pts = rng.random((n, 2))                         
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        pts = pts[np.argsort(angles)]
        return cls(
            pts.reshape(-1),
            [0, n],
            box = box,
            energyType = energyType,
            rho = rho,
            targetEdgeLength = targetEdgeLength,
            targetArea = targetArea,
            targetPerimeter = targetPerimeter,
        )
