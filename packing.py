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
from enums import EnergyType, PackingType


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
        # targetEdgeLength is PER EDGE (one per vertex, the edge leaving it), so every edge carries
        # its own target and a polygon need not be equilateral. targetPerimeter is DERIVED from it --
        # a single source of truth, so the two can never disagree.
        self.targetEdgeLength = self.asPerVertex(targetEdgeLength, self.numVertices)
        # Per-VERTEX skip-one diagonal targets, |r_{k+1} - r_{k-1}|, written by Model.setShapeTemplate.
        # None until a template is asked for: edges alone leave a polygon free to flex, and this is
        # what pins the turning angles.
        self.targetDiagonal = None
        self.targetArea = self.asPerPolygon(targetArea, numPolygons)
        self.targetPerimeter = None
        if self.targetEdgeLength is not None:
            self.syncTargetPerimeter()
        elif targetPerimeter is not None:
            self.targetPerimeter = self.asPerPolygon(targetPerimeter, numPolygons)

        self.force = np.zeros_like(self.positions)
        self.velocities = np.zeros_like(self.positions)
        self.energy = 0.0
        # Boolean (numVertices,) mask of PINNED vertices, or None when nothing is pinned. A pinned
        # vertex still exerts force on its neighbors and still enters its polygon's area/edge terms;
        # it simply never moves. Set it through Model.pinVertices, which validates the indices.
        self.pinned = None
        # Index of the CONTAINER polygon (a wall confining the packing), or None. Set by
        # Model.setBoundaryConditions("fixed"); the container interacts through a different energy
        # (see energies.containerEnergyForce) and is excluded from the ordinary overlap.
        self.containerIndex = None

    @staticmethod
    def asPerPolygon(value, numPolygons):
        """Broadcast a scalar / array / None target into a length-P float array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype = float)
        if arr.ndim == 0:
            return np.full(numPolygons, float(arr))
        return arr.reshape(numPolygons)

    @staticmethod
    def asPerVertex(value, numVertices):
        """Broadcast a scalar / array / None into a length-numVertices float array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype = float)
        if arr.ndim == 0:
            return np.full(numVertices, float(arr))
        return arr.reshape(numVertices)

    def syncTargetPerimeter(self):
        """Recompute targetPerimeter as the SUM of each polygon's edge targets.

        Call after any change to targetEdgeLength. The perimeter is not independently storable any
        more -- it is defined by the edge targets, which is what keeps a per-edge model consistent."""
        self.targetPerimeter = np.bincount(self.shapeId, weights = self.targetEdgeLength,
                                           minlength = self.numPolygons)
        return self.targetPerimeter

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


# ---------------------------------------------------------------------------
# Periodic box and minimum-image wrapping (merged from box.py)
# ---------------------------------------------------------------------------


class Box:
    """Periodic simulation cell.

    Parameters
    ----------
    packingType : PackingType
        ``square`` -> unit square [0, 1)^2 (default); ``latticeVector`` -> the
        cell is the parallelepiped spanned by the columns of ``h`` (step 9).
    h : array_like or None
        2x2 lattice matrix whose columns are the lattice vectors [a1 | a2]. Used
        only when ``packingType is latticeVector``. For ``square`` it is left as
        ``None`` (the cell is the unit square, i.e. an implicit identity ``h``).
    """

    def __init__(self, packingType = PackingType.square, h = None):
        self.type = packingType
        self.h = None if h is None else np.asarray(h, dtype = float).reshape(2, 2)


def wrap(dr, box):
    """Minimum-image displacement under the box's periodic boundary.

    Branches on ``box.type`` (prompt line 27).

    square: positions are in [0, 1)^2, so a raw displacement is mapped to its
    nearest periodic image in [-0.5, 0.5) componentwise via
    ``dr - floor(dr + 0.5)``.

    Parameters
    ----------
    dr : array_like
        Displacement(s); any shape (operates componentwise).
    box : Box

    Returns
    -------
    np.ndarray
        Minimum-image displacement, same shape as ``dr``.
    """
    dr = np.asarray(dr, dtype = float)
    if box.type is PackingType.square:
        return dr - np.floor(dr + 0.5)
    if box.type is PackingType.latticeVector:
        raise NotImplementedError(
            "latticeVector wrap is added in build step 9 (Phase 9)."
        )
    raise ValueError(f"unknown PackingType: {box.type!r}")

def minImageShift(displacement, box):
    """Lattice translation that carries ``displacement`` to its minimum image, i.e.
    ``wrap(displacement) - displacement``. This is the rigid offset to add to a far point
    so it lands in its nearest periodic image; with single-image interactions it brings one
    polygon next to another for the intersection / overlap tests. Dispatches on the box type via
    ``wrap``; zero in free space (``box is None``).
    """
    displacement = np.asarray(displacement, dtype = float)
    if box is None:
        return np.zeros_like(displacement)
    return wrap(displacement, box) - displacement

def wrapIntoCell(positions, box):
    """Map positions into the periodic cell. Called unconditionally by the
    minimizers, so wrapping is governed entirely by the box (no per-call flag):

      box is None   -- free space (eqSoftBody shape-build): returned unchanged.
      square        -- each coordinate into [0, 1) via p - floor(p).
      latticeVector -- into the parallelepiped via fractional coords (step 9; stub).

    The area-1 cell calibration lives with the periodic box (square is the unit
    square; latticeVector will enforce |det h| = 1 at step 9).

    This wraps each coordinate independently, so it is only correct for a cloud of
    independent points (e.g. the free-space shape-build, where it is a no-op). A
    connected polygon straddling the boundary would be torn; wrap whole polygons
    with ``wrapPolygonsIntoCell`` instead.
    """
    if box is None:
        return positions
    positions = np.asarray(positions, dtype = float)
    if box.type is PackingType.square:
        return positions - np.floor(positions)
    if box.type is PackingType.latticeVector:
        raise NotImplementedError(
            "latticeVector cell-wrap is added in build step 9 (Phase 9)."
        )
    raise ValueError(f"unknown PackingType: {box.type!r}")

def wrapPolygonsIntoCell(packing):
    """Translate each WHOLE polygon by an integer cell vector so its centroid lands in the cell,
    modifying ``packing.positions`` in place. Unlike ``wrapIntoCell`` this keeps every polygon
    intact (per-vertex wrapping would tear a polygon that straddles the boundary, and
    ``cornerGeometry`` reads raw edge differences with no minimum image). No-op in free space
    (box is None), so it is safe in the shape-build minimizer as well."""
    box = packing.box
    if box is None:
        return
    if box.type is not PackingType.square:
        raise NotImplementedError(
            "latticeVector polygon-wrap is added in build step 9 (Phase 9)."
        )
    r = packing.positions.reshape(-1, 2)
    counts = np.bincount(packing.shapeId, minlength = packing.numPolygons)
    centroid = np.zeros((packing.numPolygons, 2))
    np.add.at(centroid, packing.shapeId, r)
    centroid /= counts[:, None]
    shift = np.floor(centroid)
    # A polygon holding a PINNED vertex is never wrapped. The translation is an exact lattice vector
    # so the physics would be unchanged, but it would relocate a vertex the user asked to hold fixed,
    # and pinned coordinates are meant to be readable as given.
    if packing.pinned is not None:
        anchored = np.zeros(packing.numPolygons, dtype = bool)
        anchored[packing.shapeId[packing.pinned]] = True
        shift[anchored] = 0.0
    r -= shift[packing.shapeId]
