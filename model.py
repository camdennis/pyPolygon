"""High-level Model facade for a rounded-polygon packing.

Model wraps the flat reference modules (packingBuilder, geometry, neighbors,
intersections, overlap, visualize, validate) behind one object, following an update/get
scheme: an ``updateX`` method computes a quantity from the current packing and caches it
on the model, and the matching ``getX`` reads it back (mirroring how the CUDA kernels
will write and expose state). The demo-notebook interface:

    packing = Model(N = 32, n = 10, seed = 42)
    packing.generateEquilateralRPs(rho = 0.1 / 32, phi = 1.0, kappa = 4.0)
    packing.draw()
    packing.updateOverlapAreasParallel()
    packing.updateMCOverlapArea()
    print(sum(packing.getOverlapAreas().values()), packing.getMCOverlapArea())

It holds the underlying Packing in ``self.packing``, the corner radius in ``self.rho``
once generated, and a seeded numpy Generator in ``self.rng`` driving every random draw.
"""

from enums import PackingType
from box import Box
from packingBuilder import (buildEquilateralPacking, shapeBackbones,
                            setBidispersePerimeter, shapeIndices,
                            _warnLargePhi, _warnLargeRho)
from distributions import asRng
from geometry import cornerGeometry
from neighbors import findNeighbors
from intersections import boundaryFeatures, findIntersections
from overlap import overlapAreas, overlapAreasParallel
from overlapForce import overlapForces
import visualize
import validate

class Model:
    """A packing of N polygons, n vertices each, built and measured as rounded polygons.
    Quantities are computed by ``updateX`` (cached on the model) and read by ``getX``."""

    def __init__(self, N, n, seed = None):
        self.N = N
        self.n = n
        self.rng = asRng(seed)
        self.packing = None
        self.rho = None
        self._intersections = None
        self._overlapAreas = None
        self._mcOverlapArea = None
        self._forces = None
        self._energy = None

    def generateEquilateralRPs(self, rho, phi, kappa, maxSteps = 200000):
        """Build N monodisperse equilateral rounded polygons: shape index ``kappa``, area
        phi/N each, corner radius ``rho``. Seeds random stars from the model's RNG, relaxes
        them to equilateral in free space, then places them in the periodic square box.
        Warns if phi or rho is large enough to break the overlap machinery's single-image /
        feasibility assumptions. Returns self."""
        self.rho = rho
        self.packing = buildEquilateralPacking(self.N, self.n, kappa, areaKind = "mono",
                                               phi = phi, rho = rho, rng = self.rng)
        shapeBackbones(self.packing, maxSteps = maxSteps)
        self.packing.box = Box(PackingType.square)
        _warnLargePhi(self.packing, self.n)
        _warnLargeRho(self.packing, self.n)
        return self

    def setBidispersePerimeter(self, ratio = 1.4):
        """Rescale so the first half's perimeter is ``ratio`` times the second half's, at
        fixed packing fraction. Returns self."""
        setBidispersePerimeter(self.packing, ratio)
        return self

    def draw(self, ax = None, highlightIntersections = False, **kwargs):
        """Draw the rounded packing (filled shapes, darker outlines, dotted backbones; with
        ``highlightIntersections`` the boundary intersections colored by type). Returns the axes."""
        return visualize.draw(self.packing, self.rho, ax = ax,
                              highlightIntersections = highlightIntersections, **kwargs)

    def _featuresAndIntersections(self):
        """Boundary features and the inter-polygon intersection records for the current
        packing -- the shared input to the intersection and overlap updates."""
        cg = cornerGeometry(self.packing, self.rho)
        features = boundaryFeatures(self.packing, cg, self.rho)
        intersections = findIntersections(self.packing, features,
                                          findNeighbors(self.packing, self.rho))
        return features, intersections

    def updateIntersections(self):
        """Compute the inter-polygon boundary intersections and cache them. Returns self."""
        _, self._intersections = self._featuresAndIntersections()
        return self

    def getIntersections(self):
        """Return the cached boundary intersection records (None before an update)."""
        return self._intersections

    def updateOverlapAreas(self):
        """Compute the per-pair overlap areas with the serial boundary walk and cache them.
        Returns self."""
        features, intersections = self._featuresAndIntersections()
        self._overlapAreas = overlapAreas(self.packing, features, intersections)
        return self

    def updateOverlapAreasParallel(self):
        """Compute the per-pair overlap areas with the parallel (per-feature binary-search)
        method and cache them. Returns self."""
        features, intersections = self._featuresAndIntersections()
        self._overlapAreas = overlapAreasParallel(self.packing, features, intersections)
        return self

    def getOverlapAreas(self):
        """Return the per-pair overlap areas cached by the last overlap update, as
        {(polyA, polyB): area} (None before any update)."""
        return self._overlapAreas

    def updateOverlapForcesParallel(self):
        """Compute the overlap force on every vertex (-d total overlap area / d v) and cache it;
        read it with ``getForces``. The energy/force is the overlap term only -- the adhesion,
        area, and perimeter springs are zero for now. Returns self."""
        cg = cornerGeometry(self.packing, self.rho)
        features = boundaryFeatures(self.packing, cg, self.rho)
        intersections = findIntersections(self.packing, features,
                                          findNeighbors(self.packing, self.rho))
        self._forces = overlapForces(self.packing, features, intersections, cg, self.rho)
        return self

    def getForces(self):
        """Return the cached per-vertex force array, shape (numVertices, 2); None before an
        update."""
        return self._forces

    def updateEnergy(self):
        """Compute the total energy and cache it. Currently the overlap term only (the other
        springs are zero): the total pairwise overlap area. Returns self."""
        features, intersections = self._featuresAndIntersections()
        self._energy = float(sum(overlapAreas(self.packing, features, intersections).values()))
        return self

    def getEnergy(self):
        """Return the cached total energy (overlap term only for now); None before an update."""
        return self._energy

    def updateMCOverlapArea(self, samples = 500000):
        """Monte-Carlo estimate the total pairwise overlap area (using the model's RNG) and
        cache it -- the cross-check for the overlap-area updates. Returns self."""
        self._mcOverlapArea = validate.getMCOverlapArea(self.packing, self.rho,
                                                        samples = samples, rng = self.rng)
        return self

    def getMCOverlapArea(self):
        """Return the cached Monte-Carlo total overlap area (None before an update)."""
        return self._mcOverlapArea

    def shapeIndices(self):
        """Realized shape index (perimeter / sqrt(area)) of each polygon."""
        return shapeIndices(self.packing)
