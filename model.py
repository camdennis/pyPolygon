"""High-level Model facade for a sharp polygon packing with a Plummer-mollified overlap.

Model builds the packing (N equilateral backbones relaxed to their eqSoftBody targets, placed in
the periodic square box) and exposes an update/get + minimize interface driven off the mollified
overlap (energies.plummerOverlapExact) plus the eqSoftBody springs and the intra-polygon
self-repulsion. Typical use:

    m = Model(N = 32, n = 10, seed = 42)
    m.generateEquilateralPolygons(phi = 1.0, kappa = 4.0)
    m.setBiPerimeter()
    m.setSpringConstants(adhesion = 0, area = 1, perimeter = 1, edge = 1)
    m.setConstraints(area = True, perimeter = False, edge = True)   # or leave it to the springs
    m.setSofteningFraction(0.10)              # sigma = 0.10 * mean edge length
    m.updateForcesParallel(); m.draw(forces = m.getForces())
    m.minimizeFIRE(maxSteps = 1000, fThreshold = 1e-3)   # robust relaxation
    m.minimizeCG(fThreshold = 1e-12)                     # polish, no Hessian needed

The shape terms can be held either SOFTLY by the springs (``setSpringConstants``) or RIGIDLY by
constraints (``setConstraints``), term by term. Constraining a term ignores its spring: rigidity is
the stiff-spring limit, so paying both would double-count it. Rigid shapes take the stiff spring modes
out of the dynamics, which lifts FIRE's timestep ceiling and drops the condition number -- see
``constraints.py``.

The overlap contact law is selected by ``setModelType`` ("sharp" default, or "mollified"). For the
mollified model the width ``sigma`` softens the contact: ``setMollification`` sets it as an absolute
length, ``setSofteningFraction`` as a fraction of the mean edge length. ``rho`` is retained (stored,
saved) but the backbone is sharp-edged.
"""

import contextlib
import os
import warnings

import numpy as np

from enums import PackingType, EnergyType
from packing import Box, Packing, buildConnectivity
from build import (buildEquilateralPacking, shapeBackbones, setBiPerimeter, setMonoPerimeter,
                   setLogNormalTargetPerimeter, setLogNormalTargetArea,
                   setLogNormalTargetEdgeLength, setLogNormalPerimeter, setLogNormalArea,
                   setLogNormalScale, setSizePolydispersity, shapeIndices, _warnLargePhi,
                   asRng, regularShapeIndex)
from softBody import eqSoftBodyEnergyForce, backboneArea, backboneEdgeLengths
from energies import (plummerOverlapExact, sharpOverlapEnergyForce, selfRepulsionEnergyForce,
                      plummerMeasure, containerEnergyForce, overlapAreaEnergyForce,
                      containerOrientationSign, pointInPolygon)
from constraints import (ShapeConstraints, DistributionConstraints, CompositeConstraints)
from transient import TransientTargets, targetForces
import alternating
import minimize
import anneal
import records
import roundedGeometry
try:
    import cudaOverlap
except Exception:
    cudaOverlap = None

_TARGETS = ("targetEdgeLength", "targetArea", "targetPerimeter")
_CORE = {"positions", "startIndices", "boxType", "energyType", "rho"} | set(_TARGETS)
_MIN_STABLE_SOFTENING_FRACTION = 0.01
# Wall stiffness relative to the inter-particle contact. NOT 1.0: at equal stiffness an overjammed
# packing relieves stress by escaping through the wall rather than by overlapping its neighbors, since
# escape lowers the confinement for everyone while overlap relieves nothing globally. Measured on 11
# squares, one seed, five points -- phi = 0.6637 at k = 1, 0.6836 at 3, 0.6998 at 10, 0.6973 at 30,
# 0.6944 at 100. A broad peak at 10, worth +5.4% density, with the decline after it consistent with the
# stiffer mode shortening FIRE's timestep (its ceiling scales as 1/sqrt(k)).
_DEFAULT_CONTAINER_STIFFNESS = 10.0

# Said ONCE per session. The text carries no live numbers, but it fires from every old notebook and
# test that still says setModelType("sharp"), and a rename notice repeated a hundred times reads as
# breakage rather than as guidance.
_WARNED_SHARP_TIER = False

# Sentinel for Model.draw(colorBy = Model.shapeIndex): shade by P / sqrt(A), computed at draw time.
_SHAPE_INDEX = object()
# How much area a decimation may quietly cost before it is worth saying so. 2% is well above the
# 32 -> 16 rung of a smooth polygon (98.1% kept) and well below 8 -> 4 (70.7%).
_RESAMPLE_LOSS = 0.02
# Narrowest distribution a mean+variance moment pair can hold before its two rows go parallel. A
# moment family can only be NARROWED, so a width at this floor is a request for a point mass.
_MIN_MOMENT_WIDTH = 1e-4
# Shortest chord a free-space spring will pull along. Below it the direction is meaningless, so the
# term is dropped rather than divided by; see updateAlternatingDiagonals.
_MIN_SPRING_LENGTH = 1e-14


# UNVERIFIED(Cam)
def _distanceToLoop(point, loop):
    """Shortest distance from ``point`` to the closed polyline ``loop`` (n, 2), unsigned.

    Point-to-SEGMENT rather than point-to-line, so a point off the end of an edge measures to the
    nearest endpoint instead of to the edge's infinite extension -- which is what makes it correct near
    a container corner, the exact place the deepest excursions occur."""
    loop = np.asarray(loop, dtype = float)
    start = loop
    end = np.roll(loop, -1, axis = 0)
    edge = end - start
    lengthSquared = np.einsum("ij,ij->i", edge, edge)
    safe = np.where(lengthSquared > 0.0, lengthSquared, 1.0)
    t = np.clip(np.einsum("ij,ij->i", point - start, edge) / safe, 0.0, 1.0)
    closest = start + t[:, None] * edge
    return float(np.min(np.hypot(*(point - closest).T)))
_SELF_REP_WARN_ENERGY = 1e-6


# UNVERIFIED(Cam)
def _loopArea(loop):
    """Unsigned shoelace area of one closed loop of vertices, shape (n, 2)."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def savePacking(packing, path, **extra):
    """Write ``packing``'s full state to ``path`` (npz): positions, start indices, rho, box/energy
    type, and every non-None target. Extra keyword scalars/arrays ride along under their names."""
    box = packing.box
    data = {
        "positions": np.asarray(packing.positions, dtype = float),
        "startIndices": np.asarray(packing.startIndices, dtype = int),
        "boxType": "none" if box is None else box.type.name,
        "energyType": packing.energyType.name,
        "rho": np.asarray(np.nan if packing.rho is None else packing.rho, dtype = float),
    }
    for name in _TARGETS:
        value = getattr(packing, name, None)
        if value is not None:
            data[name] = np.asarray(value, dtype = float)
    for key, value in extra.items():
        data[key] = np.asarray(value)
    tmpPath = str(path) + ".tmp"
    with open(tmpPath, "wb") as handle:
        np.savez(handle, **data)
    os.replace(tmpPath, path)


def loadPacking(path):
    """Rebuild a Packing from a ``savePacking`` npz. Returns ``(packing, extras)``."""
    with np.load(path, allow_pickle = False) as handle:
        data = {key: handle[key] for key in handle.files}
    boxType = str(data["boxType"])
    box = None if boxType == "none" else Box(PackingType[boxType])
    rho = data["rho"]
    rho = None if np.isnan(rho).all() else (float(rho) if rho.ndim == 0 else rho)
    packing = Packing(
        positions = data["positions"],
        startIndices = data["startIndices"],
        box = box,
        energyType = EnergyType[str(data["energyType"])],
        rho = rho,
        targetEdgeLength = data.get("targetEdgeLength"),
        targetArea = data.get("targetArea"),
        targetPerimeter = data.get("targetPerimeter"),
    )
    extras = {key: data[key] for key in data if key not in _CORE}
    return packing, extras


class Model:
    """A packing of N polygons, n vertices each, with a Plummer-mollified overlap energy."""
    def __init__(self, N, n, seed = None):
        self.N = N
        self.n = n
        self.rng = asRng(seed)
        self.packing = None
        self.rho = None                   # per-vertex corner radius; see setRho / setGeometryType
        # TWO INDEPENDENT AXES. The TIER is the contact law ("area", "mollified", "softDepth",
        # "depth"); the GEOMETRY is what shape that law is handed ("sharp" or "round"). They compose:
        # a rounded backbone can be measured by the area law or by the depth law without either
        # knowing about the rounding, because it is handed an ordinary polygon either way.
        self.modelType = "area"           # the TIER; "sharp" is accepted as the old name for this
        self.geometryType = "sharp"       # the GEOMETRY; "round" rounds the corners by rho
        self.arcSegments = 6              # chords per rounded corner; see setGeometryType
        self.sigma = None                 # softening width (absolute); set via setMollification
        self.sigmaFraction = None
        self.kAdh = 0.0                   # adhesion (not yet used)
        self.kArea = 1.0
        self.kPerimeter = 1.0            # perimeter spring (not yet used; softBody is edge+area)
        self.kEdge = 0.0
        self.kSelf = 1.0                 # self-repulsion barrier height
        self.kContainer = _DEFAULT_CONTAINER_STIFFNESS   # see setContainerStiffness
        self.softEpsilon = None          # soft-depth softmin length; see setSoftDepth
        self.depthStiffness = 1.0        # exact-distance contact law; see setDepthContact
        self.depthWallStiffness = 1.0    # container stiffness RELATIVE to it; see setDepthContact
        self.softStiffness = 1.0
        self.adhesionWork = 0.0
        self.adhesionRange = None
        self.selfRepFraction = 0.05      # barrier range as a fraction of the edge length
        self.constraints = None          # ShapeConstraints once setConstraints() is called
        self.boundaryConditions = "periodic"   # see setBoundaryConditions
        self.dofType = "fixed"           # "fixed" or "transient"; see setDOFType
        self.moments = [1]               # conserved moments of the target distribution
        self.transient = None            # TransientTargets once both are set
        self.neighbors = None            # neighbors.NeighborList, built lazily; see setNeighborSkin
        self.useNeighborList = True      # False falls back to the all-to-all intersection scan
        self._forces = None
        self._energy = None

    # UNVERIFIED(Cam)
    def generatePolygons(self, phi, kappa, edgePolydispersity = 0.0, maxSteps = 200000):
        """Build N monodisperse backbone polygons (shape index ``kappa``, area phi/N each): seed random
        stars, relax to the targets in free space (eqSoftBody / FIRE), place in the periodic square box.
        Sharp model, so no corner radius. Returns self.

        ``edgePolydispersity`` is the coefficient of variation of each polygon's EDGE targets. At 0 the
        polygons are equilateral, which is what ``generateEquilateralPolygons`` asks for. Above 0 the
        edges are drawn log-normally within each polygon at the SAME size and the SAME shape index --
        the perimeter is held by normalizing the draw, and the area target never enters it. Only the
        shape changes.

        Worth doing when an anneal is going to withdraw the edge-length spread. Equilateral polygons have
        no within-polygon spread to withdraw: the pooled edge CV is then entirely inherited from the size
        distribution and sits exactly on the floor that fixed areas impose, so a ramp driving it to zero
        asks for a state the build already occupies. Seeding the spread here gives that ramp something to
        take back, and taking it back is precisely what makes the polygons regular.

        Too wide a draw is REFUSED rather than quietly approximated -- unequal edges enclose less area
        than equal ones, so they raise the floor on the shape index (see ``build.minShapeIndex``), and
        past the point where that floor reaches ``kappa`` the targets describe a polygon that does not
        exist. ``getEdgePolydispersity`` reports what was realized."""
        self.rho = 0.0
        self.packing = buildEquilateralPacking(self.N, self.n, kappa, areaKind = "mono",
                                               phi = phi, rho = 0.0, rng = self.rng,
                                               edgePolydispersity = edgePolydispersity)
        shapeBackbones(self.packing, maxSteps = maxSteps)
        self.packing.box = Box(PackingType.square)
        _warnLargePhi(self.packing, self.n)
        return self

    def generateEquilateralPolygons(self, phi, kappa, maxSteps = 200000):
        """Build N monodisperse EQUILATERAL backbone polygons (shape index ``kappa``, area phi/N each).

        The zero-spread case of ``generatePolygons``; see there for the general build and for why an
        anneal that withdraws the edge-length spread wants a nonzero one. Returns self."""
        return self.generatePolygons(phi, kappa, edgePolydispersity = 0.0, maxSteps = maxSteps)

    def setBiPerimeter(self, ratio = 1.4):
        """Make the polygons bidisperse: the first half's target perimeter is ``ratio`` times the
        second half's, at fixed packing fraction (updates the eqSoftBody targets). Needs an even
        number of polygons. Reverse it with ``setMonoPerimeter``. Returns self."""
        setBiPerimeter(self.packing, ratio)
        return self

    def _nonContainer(self):
        """Number of leading polygons that are not the container wall."""
        container = getattr(self.packing, "containerIndex", None)
        return self.packing.numPolygons if container is None else int(container)

    def syncTargetAreas(self):
        """Set every polygon's target AREA to the area it actually has. Returns self.

        The build relaxes each polygon toward its targets but stops at a finite tolerance, so the
        geometry lands a little off: measured max|A/A0 - 1| = 1.3e-06 straight out of
        ``generateEquilateralPolygons``. Syncing closes that gap in the other direction -- instead of
        moving vertices to meet the targets, it moves the targets to meet the vertices.

        Worth doing before constraining, because it makes the target set exactly REALIZABLE: the current
        configuration satisfies it by construction, so there is no question of asking for a shape that
        cannot exist. That is the clean way past the isoperimetric trap described in
        ``constraints.ShapeConstraints.infeasibleReason`` -- a synced target set has a shape index taken
        from real geometry, so it is on the feasible side of the bound automatically.

        The container is left alone: its target area is the signed area of the wall, set deliberately by
        ``setBoundaryConditions``."""
        stop = self._nonContainer()
        self.packing.targetArea[:stop] = backboneArea(self.packing)[:stop]
        self._forces = None
        self._energy = None
        return self

    def syncTargetPerimeters(self):
        """Set every EDGE target to the length that edge actually has. Returns self.

        The perimeter targets follow as the sum, so the realized perimeters match exactly too -- see
        ``syncTargetAreas`` for why this is worth doing before constraining.

        Note this syncs per EDGE, not per polygon, which is what makes an ``edge = True`` constraint set
        exactly consistent. The consequence is that the targets stop being exactly equilateral (by
        ~1e-07 on a fresh build, i.e. by however far the relax fell short). If you would rather keep
        perfectly equilateral targets and only match the total, that is a different operation -- say so
        and I will add it.

        NB not to be confused with ``Packing.syncTargetPerimeter`` (singular), which does something
        unrelated: it recomputes the DERIVED ``targetPerimeter`` from the existing edge targets and never
        looks at the geometry."""
        stop = self._nonContainer()
        upTo = int(self.packing.startIndices[stop])
        self.packing.targetEdgeLength[:upTo] = backboneEdgeLengths(self.packing)[:upTo]
        self.packing.syncTargetPerimeter()
        self._forces = None
        self._energy = None
        return self

    # Pass as ``draw(colorBy = Model.shapeIndex)`` to shade by P / sqrt(A).
    shapeIndex = _SHAPE_INDEX

    def getSizeStatistics(self):
        """``(mean, std)`` of the polygon target AREAS, excluding any container.

        Reported separately rather than only as their ratio because they say different things: the MEAN
        is set by the packing fraction and should hold constant through an anneal (the schedule narrows
        the distribution without resizing the packing), while the STD is the quantity being driven to
        zero. A single ratio hides which of the two moved."""
        container = getattr(self.packing, "containerIndex", None)
        stop = self.packing.numPolygons if container is None else int(container)
        areas = np.asarray(self.packing.targetArea, dtype = float)[:stop]
        return float(areas.mean()), float(np.std(areas))

    def getSizePolydispersity(self):
        """Std / mean of the polygon target AREAS -- the size spread, excluding any container.

        The quantity an anneal drives toward zero to reach the monodisperse problem; the edge-length
        spread merely inherits it (edge ~ sqrt(A)). See ``getSizeStatistics`` for the two moments on
        their own."""
        mean, std = self.getSizeStatistics()
        return float(std / mean) if mean > 0.0 else 0.0

    # UNVERIFIED(Cam)
    # UNVERIFIED(Cam)
    def getReachableWidth(self):
        """The narrowest edge-length CV the FIXED areas allow -- the floor a width ramp should aim at.

        Exactly the value ``setTargetPolydispersity`` clamps to, so a ramp aimed here lands ON it
        rather than a hair under. ``getEdgePolydispersity()['between']`` is very nearly the same number
        and is the right thing to READ (it is measured from the geometry), but it is not the same
        computation: this one is derived from the target AREAS via the regular n-gon's edge, and the
        two agree only to ~2e-10 relative. Aiming a ramp at the measured proxy therefore trips the
        clamp's ``floor * (1 - 1e-9)`` guard on the final round and prints two warnings about targets
        being unreachable -- which are true, by two parts in ten billion, and entirely misleading."""
        return anneal._reachableWidth(self)

    def getEdgePolydispersity(self):
        """Split the edge-target spread into its WITHIN- and BETWEEN-polygon parts, as a dict.

        ``pooled`` is the CV of every edge target in the packing -- the quantity ``edge = [1, 2]``
        constrains and ``setTargetPolydispersity`` re-aims. It is exactly the sum in quadrature of the
        other two (a variance decomposition, weighted by vertex count so ragged polygons are handled):

            ``between`` -- the spread of the polygons' MEAN edges, i.e. their sizes. Frozen by
            ``setConstraints(area = True)``, which is why the pooled CV has a floor
            (``anneal._reachableWidth``) that no ramp can push through.

            ``within`` -- the spread INSIDE each polygon, i.e. how far from equilateral they are. This
            is the part a stiffening ramp can actually withdraw, and an equilateral build starts with
            none of it. See ``generatePolygons(edgePolydispersity = ...)``.

        Reading only the pooled number is what hides an anneal that has nothing to do: it can sit pinned
        at its floor for the whole run while the printout shows a plausible nonzero width.

        Measured on the REALIZED edge lengths, not the targets, because that is what the moment
        machinery acts on -- ``edge = [1, 2]`` constrains the lengths the polygons actually have and
        never reads ``targetEdgeLength`` at all. Taking the targets instead makes the ramp look like a
        no-op: they sit where the build left them for the whole run while the geometry moves underneath.
        This matches ``getPolydispersity()['edge']``, which is the same pooled number."""
        packing = self.packing
        stop = self._nonContainer()
        starts = np.asarray(packing.startIndices, dtype = int)
        edges = backboneEdgeLengths(packing)[:starts[stop]]
        counts = np.diff(starts[:stop + 1]).astype(float)
        mean = float(edges.mean())
        if mean <= 0.0:
            return dict(pooled = 0.0, within = 0.0, between = 0.0)
        shapeId = np.asarray(packing.shapeId, dtype = int)[:starts[stop]]
        polygonMean = np.bincount(shapeId, weights = edges, minlength = stop) / counts
        between = np.sqrt(float(np.sum(counts * (polygonMean - mean) ** 2) / counts.sum()))
        within = np.sqrt(max(float(np.mean((edges - polygonMean[shapeId]) ** 2)), 0.0))
        return dict(pooled = float(np.std(edges) / mean),
                    within = float(within / mean), between = float(between / mean))

    def setSizePolydispersity(self, polydispersity):
        """Narrow the SIZE distribution to the given coefficient of variation, holding the packing
        fraction and every polygon's shape fixed. Returns self.

        The handle an anneal needs, and it acts on the AREAS. Sizes live in ``targetArea``; under
        ``setConstraints(area = True)`` those are frozen, so squeezing the edge-length moments cannot
        narrow the size distribution -- the edge lengths inherit their spread from the areas and are
        already as equal as the fixed areas allow. Pushing them further only distorts shapes. Each
        polygon is rescaled about its own centroid, carrying its edge targets, so the shape index is
        untouched: squares stay squares while their sizes converge."""
        setSizePolydispersity(self.packing, polydispersity)
        self._forces = None
        self._energy = None
        return self

    def setLogNormalScale(self, polydispersity = 0.1):
        """Draw polygon SIZES from a log-normal distribution, holding every polygon's SHAPE fixed.
        Returns self.

        The one to use for a packing of same-shaped objects at different sizes. Each polygon is scaled
        by its own factor, so its area target moves as the square and its edge targets linearly, which
        leaves the shape index ``P / sqrt(A)`` exactly where it was. The packing fraction is preserved.

        NOT the same as ``setLogNormalTargetPerimeter``, and the difference is a silent trap. That one
        moves the perimeter targets ALONE; combined with ``setConstraints(area = True)`` it does not ask
        for bigger squares but for DISTORTED ones, since ``p0 = P0 / sqrt(A0)`` then moves by the full
        polydispersity. Measured: a 0.25 draw demanded a shape index near 5.0 against 4.0 for a square,
        and the springs delivered 49-degree rhombi with exactly the right areas.

        ``polydispersity`` is the coefficient of variation of the AREA."""
        setLogNormalScale(self.packing, polydispersity, rng = self.rng)
        self._forces = None
        self._energy = None
        return self

    def setLogNormalTargetPerimeter(self, polydispersity = 0.1):
        """Draw the polygon TARGET perimeters from a log-normal distribution about their current mean.

        ``polydispersity`` is the coefficient of variation (std / mean). Targets only -- no vertex
        moves, and the area targets are left alone, so each polygon ends up with its own shape index.
        Since ``targetPerimeter`` is defined as the sum of a polygon's edge targets, this scales those
        edge targets together (the polygon stays equilateral); there is no separate perimeter target to
        set on its own. Uses the model's seeded rng, so it is reproducible. Returns self.

        CAUTION: combining this with ``setLogNormalTargetArea`` and then
        ``setConstraints(area = True, edge = True)`` is generically INFEASIBLE -- an equilateral build
        sits exactly on the isoperimetric bound, so perturbing area and perimeter independently asks
        about half the polygons to enclose more area than their edges allow. ``setConstraints`` refuses
        it with the offending polygon named."""
        setLogNormalTargetPerimeter(self.packing, polydispersity, rng = self.rng)
        return self

    def setLogNormalTargetArea(self, polydispersity = 0.1):
        """Draw the polygon TARGET areas from a log-normal distribution about their current mean.

        ``polydispersity`` is the coefficient of variation (std / mean) of the AREA -- the counterpart of
        ``setLogNormalTargetPerimeter``, and independent of it. Targets only. Uses the model's seeded
        rng. Returns self."""
        setLogNormalTargetArea(self.packing, polydispersity, rng = self.rng)
        return self

    def setLogNormalTargetEdgeLength(self, polydispersity = 0.1):
        """Draw every EDGE target independently from a log-normal distribution about its current mean.

        Unlike ``setLogNormalTargetPerimeter``, which scales a polygon's edges together, this makes the
        polygons non-equilateral. Targets only. Returns self."""
        setLogNormalTargetEdgeLength(self.packing, polydispersity, rng = self.rng)
        return self

    def setLogNormalPerimeter(self, polydispersity = 0.1):
        """Randomize the REALIZED perimeters log-normally, by scaling each polygon about its centroid.
        No target is touched. Returns self.

        The geometry counterpart of ``setLogNormalTargetPerimeter``: that one sets what the shapes aim
        for, this one changes what they are. Scaling is isotropic, so the realized AREAS move as the
        square of the factor -- which a hard area constraint will undo on its next retraction. For an
        area-PRESERVING spread (the one an anneal wants) use ``spreadShapes``."""
        setLogNormalPerimeter(self.packing, polydispersity, rng = self.rng)
        self._forces = None
        self._energy = None
        return self

    def setLogNormalArea(self, polydispersity = 0.1):
        """Randomize the REALIZED areas log-normally, by scaling each polygon about its centroid. No
        target is touched. Returns self.

        The geometry counterpart of ``setLogNormalTargetArea``; see ``setLogNormalPerimeter`` for the
        target-versus-geometry split and the area-preserving alternative."""
        setLogNormalArea(self.packing, polydispersity, rng = self.rng)
        self._forces = None
        self._energy = None
        return self

    def setMonoPerimeter(self):
        """Give every polygon the SAME target perimeter, at fixed packing fraction -- the counterpart
        of ``setBiPerimeter``, and the way back to a monodisperse packing after it. Places no parity
        requirement on N and is idempotent. Returns self."""
        setMonoPerimeter(self.packing)
        return self

    def setSpringConstants(self, adhesion = 0.0, area = 1.0, perimeter = 0.0, edge = 1.0):
        """Set the shape-holding / adhesion spring constants (overlap K = 1 sets the overall scale).
        The default is no adhesion with the AREA and EDGE springs at 1 (the eqSoftBody model);
        ``perimeter`` and ``adhesion`` are stored for the API but not yet in the energy. Returns self."""
        self.kAdh = adhesion
        self.kArea = area
        self.kPerimeter = perimeter
        self.kEdge = edge
        return self

    def addShape(self, vertices, targetArea = None):
        """Append a polygon with the given ``vertices`` (n, 2) to the packing. Returns its index.

        Its eqSoftBody targets are taken from the geometry as given (area, perimeter and mean edge
        length of the supplied loop) unless ``targetArea`` overrides the area, so the shape is already
        at its own targets and the springs do not immediately deform it. The vertex count need not
        match the other polygons.

        The usual reason to call this is to add a WALL: append a loop enclosing the packing, pin it,
        then ``setBoundaryConditions("fixed")``. Orientation does not matter -- the wall energy reads
        the winding and adapts (see ``energies.containerOrientationSign``)."""
        loop = np.asarray(vertices, dtype = float).reshape(-1, 2)
        if loop.shape[0] < 3:
            raise ValueError(f"a shape needs at least 3 vertices, got {loop.shape[0]}")
        packing = self.packing
        index = packing.numPolygons

        edges = np.roll(loop, -1, axis = 0) - loop
        perimeter = float(np.hypot(edges[:, 0], edges[:, 1]).sum())
        area = abs(0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1)
                                - np.roll(loop[:, 0], -1) * loop[:, 1]))

        packing.positions = np.concatenate([packing.positions, loop.reshape(-1)])
        packing.startIndices = np.concatenate(
            [packing.startIndices, [packing.startIndices[-1] + loop.shape[0]]])
        packing.numVertices = packing.positions.size // 2
        packing.numPolygons = packing.startIndices.size - 1
        packing.shapeId, packing.next, packing.prev = buildConnectivity(packing.startIndices)
        packing.force = np.zeros_like(packing.positions)
        packing.velocities = np.zeros_like(packing.positions)
        if packing.targetArea is not None:
            packing.targetArea = np.concatenate(
                [packing.targetArea, [area if targetArea is None else float(targetArea)]])
        if packing.targetEdgeLength is not None:
            # One target PER EDGE of the new shape, taken from its own geometry.
            lengths = np.hypot(edges[:, 0], edges[:, 1])
            packing.targetEdgeLength = np.concatenate([packing.targetEdgeLength, lengths])
            packing.syncTargetPerimeter()
        if packing.pinned is not None:
            packing.pinned = np.concatenate([packing.pinned, np.zeros(loop.shape[0], dtype = bool)])
        # The cached self-repulsion topology and any constraint blocks are keyed to the old shape
        # set, so both have to be rebuilt.
        packing._selfRepKey = None
        if self.constraints is not None:
            warnings.warn("\n*** shape added while constraints were active ***\n"
                          "    Re-run setConstraints() -- the existing constraint blocks do not "
                          "cover the new polygon.", stacklevel = 2)
        self.N = packing.numPolygons
        self._forces = None
        self._energy = None
        return index

    def doubleNumEdges(self, powerOfTwo = 1):
        """Refine every polygon by inserting the MIDPOINT of each edge, doubling its vertex count.

        ``powerOfTwo`` repeats the refinement, so the vertex count is multiplied by ``2 **
        powerOfTwo``: 1 doubles (the default), 2 quadruples, 3 gives eight times as many edges.

        The geometry is exactly unchanged -- a midpoint lies on the edge it splits, so area, perimeter
        and shape index are identical to the last bit, and so is the overlap energy. Only the
        RESOLUTION changes: each polygon can now bend where it previously could not. That makes this
        the natural way to approach a smooth boundary, relaxing cheaply at low n and refining, rather
        than starting at high n where the packing has many more soft modes to explore.

        Targets follow the geometry: ``targetEdgeLength`` halves, ``targetArea`` and
        ``targetPerimeter`` are untouched. A new midpoint is pinned only if BOTH its endpoints were
        (so a pinned wall stays fully pinned, while a partly-pinned polygon does not creep). Shape
        constraints are rebuilt automatically, since their block size depends on n.

        Note the self-repulsion range is a fraction of the edge length, so it halves too and stays
        silent -- were it still tied to an absolute sigma, doubling would push sigma/l0 up and light
        the barrier between valid non-adjacent edges. Returns self."""
        if int(powerOfTwo) < 1:
            raise ValueError(f"powerOfTwo must be at least 1 (it is an exponent: the vertex count is "
                             f"multiplied by 2 ** powerOfTwo), got {powerOfTwo}")
        for _ in range(int(powerOfTwo) - 1):
            self.doubleNumEdges(powerOfTwo = 1)
        packing = self.packing
        r = packing.positions.reshape(-1, 2)
        starts = np.asarray(packing.startIndices, dtype = int)
        pinned = packing.pinned

        newLoops = []
        newPinned = []
        for p in range(packing.numPolygons):
            a, b = int(starts[p]), int(starts[p + 1])
            loop = r[a : b]
            midpoints = 0.5 * (loop + np.roll(loop, -1, axis = 0))
            # Interleave: v0, mid(v0,v1), v1, mid(v1,v2), ... so the ring order is preserved.
            refined = np.empty((2 * loop.shape[0], 2))
            refined[0::2] = loop
            refined[1::2] = midpoints
            newLoops.append(refined)
            if pinned is not None:
                own = pinned[a : b]
                flags = np.empty(2 * own.size, dtype = bool)
                flags[0::2] = own
                flags[1::2] = own & np.roll(own, -1)
                newPinned.append(flags)

        counts = np.array([loop.shape[0] for loop in newLoops], dtype = int)
        packing.positions = np.concatenate([loop.reshape(-1) for loop in newLoops])
        packing.startIndices = np.concatenate([[0], np.cumsum(counts)])
        packing.numVertices = packing.positions.size // 2
        packing.shapeId, packing.next, packing.prev = buildConnectivity(packing.startIndices)
        packing.force = np.zeros_like(packing.positions)
        packing.velocities = np.zeros_like(packing.positions)
        packing.pinned = np.concatenate(newPinned) if pinned is not None else None
        if packing.targetEdgeLength is not None:
            # Each new midpoint splits its parent edge, so both halves inherit HALF that edge's own
            # target -- per-edge targets refine with the geometry instead of being averaged away.
            packing.targetEdgeLength = np.repeat(packing.targetEdgeLength, 2) / 2.0
            packing.syncTargetPerimeter()
        packing._selfRepKey = None

        self.n = None if self.n is None else 2 * self.n
        if self.constraints is not None:
            c = self.constraints
            self.setConstraints(area = c.area, perimeter = c.perimeter, edge = c.edge)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def _rebuildConstraints(self):
        """Re-apply the CURRENT constraint configuration after the vertex count changed.

        Any operation that adds or removes vertices invalidates the constraint objects -- their padded
        block shapes and index tables are built for a specific count -- so they have to be rebuilt from
        the live flags. Reading those flags back rather than assuming them is what keeps a cascade from
        silently dropping a family: an earlier version rebuilt only area/perimeter/edge and so lost
        ``equilateral`` and every moment family the caller had set, which turns a constrained relaxation
        into an unconstrained one without a word."""
        current = self.constraints
        if current is None:
            return self
        block = getattr(current, "block", current)
        distribution = getattr(current, "distribution", None)
        listed = {}
        if distribution is not None:
            for name in ("area", "edge", "diagonal"):
                if getattr(distribution, name, False):
                    listed[name] = list(distribution.familyMoments(name))
        return self.setConstraints(
            area = listed.get("area", bool(getattr(block, "area", False))),
            perimeter = bool(getattr(block, "perimeter", False)),
            edge = listed.get("edge", bool(getattr(block, "edge", False))),
            equilateral = getattr(block, "equilateral", None),
            flatten = bool(getattr(block, "flatten", False)),
            diagonal = listed.get("diagonal", bool(getattr(block, "diagonal", False))))

    # UNVERIFIED(Cam)
    def selectFlattening(self, stride = 2):
        """Mark every ``stride``-th vertex of each polygon for flattening, choosing the PHASE that is
        already flattest, and store the selection on the packing. Returns self.

        The set a cascade is about to remove. Picking the flattest phase is the path of least
        resistance -- those vertices are closest to carrying no geometry, so the ramp has least work to
        do and the packing least reason to rearrange. It is the same test ``halveNumEdges`` applies
        when it decides which alternating set to drop, so the two agree on the same vertices by
        construction rather than by luck.

        The mask lives on the packing as ``diagonalMask`` so it survives the constraint rebuilds
        ``setConstraints`` performs, and so the diagonal moment family can read it. Chosen ONCE per
        stage and held: re-choosing mid-ramp would make the target jump as the flattest set changed
        underneath it.

        The container is never selected -- it is a pinned wall with nothing to flatten."""
        packing = self.packing
        container = getattr(packing, "containerIndex", None)
        r = packing.positions.reshape(-1, 2)
        starts = np.asarray(packing.startIndices, dtype = int)
        mask = np.zeros(packing.numVertices, dtype = bool)
        for polygon in range(packing.numPolygons):
            if container is not None and polygon == int(container):
                continue
            a, b = int(starts[polygon]), int(starts[polygon + 1])
            loop = r[a : b]
            if loop.shape[0] % int(stride):
                raise ValueError(
                    f"polygon {polygon} has {loop.shape[0]} vertices, which stride = {stride} does "
                    f"not divide. The selection has to be evenly spaced or the removal afterwards "
                    f"cannot be.")
            ahead = np.roll(loop, -1, axis = 0) - loop
            behind = loop - np.roll(loop, 1, axis = 0)
            turn = np.abs(np.arctan2(np.cross(behind, ahead),
                                     np.einsum("ij,ij->i", behind, ahead)))
            offset = int(np.argmin([turn[o :: int(stride)].max() for o in range(int(stride))]))
            mask[a : b][offset :: int(stride)] = True
        packing.diagonalMask = mask
        return self._rebuildConstraints()

    # UNVERIFIED(Cam)
    def _maskAlternatingDiagonals(self):
        """Mark local vertices 1, 3, 5, ... of every polygon and store the mask. Returns it.

        ODD, not even, and the phase is the whole point. A flatness row is indexed by the vertex the
        diagonal is CENTRED on, so selecting vertex k constrains the chord ``|v_{k+1} - v_{k-1}|``.
        Selecting the odd ones therefore constrains exactly ``|v_2i - v_2i-2|`` -- the chords joining
        EVEN-indexed vertices, which is the set ``getAlternatingDiagonals`` measures and
        ``updateAlternatingDiagonals`` drives. Picking the even ones instead would give the
        complementary set, offset by a single vertex, and constraining one while driving the other
        flattens every vertex and collapses the polygon.

        The deterministic counterpart of ``selectFlattening``, which picks the flattest PHASE. Here the
        phase is fixed, so the selected set is knowable from the indices alone and does not move when
        the geometry does -- which is what makes a schedule written against it reproducible.

        The container is never selected; it is a pinned wall with nothing to flatten."""
        packing = self.packing
        container = getattr(packing, "containerIndex", None)
        starts = np.asarray(packing.startIndices, dtype = int)
        mask = np.zeros(packing.numVertices, dtype = bool)
        for polygon in range(packing.numPolygons):
            if container is not None and polygon == int(container):
                continue
            first, stop = int(starts[polygon]), int(starts[polygon + 1])
            # AN ODD COUNT CANNOT ALTERNATE AROUND A LOOP: the first and last selected vertices would
            # come out adjacent, so the alternating set would contain a neighbouring pair and the
            # removal afterwards could not be exact. Refused here rather than producing a mask that
            # silently means something else.
            if (stop - first) % 2:
                raise ValueError(
                    f"polygon {polygon} has {stop - first} vertices, which is ODD -- alternating "
                    f"selection does not close around the loop, so the first and last selected "
                    f"vertices would be adjacent.")
            mask[first : stop][1 :: 2] = True
        packing.diagonalMask = mask
        return mask

    # UNVERIFIED(Cam)
    def setFlatTargets(self, target):
        """Aim each SELECTED vertex's flatness ``d / (a + b)`` at its own value. Returns self.

        ``target`` is a scalar applied to every selected vertex, or one value per selected vertex in
        the order ``getFlatness`` returns them. Unselected vertices are left at 1.0, where their rows
        are inactive anyway.

        The ramp handle for flattening. Walk each vertex from where it started toward JUST SHORT of 1
        -- 1 is the triangle-inequality bound, so a row aimed exactly there has no gradient left. How
        short is set by what ``halveNumEdges`` will accept: it wants the flats turning under 5% of the
        sharpest corner, which is t > 0.999952 at the 32 -> 16 stage and t > 0.999229 at 8 -> 4."""
        packing = self.packing
        mask = getattr(packing, "diagonalMask", None)
        if mask is None:
            raise ValueError("no vertices are selected; call selectFlattening() first.")
        current = getattr(packing, "flatTarget", None)
        if current is None or np.size(current) != packing.numVertices:
            current = np.ones(packing.numVertices, dtype = float)
        current = np.array(current, dtype = float)
        values = np.asarray(target, dtype = float)
        current[mask] = values if values.size == int(mask.sum()) else float(values)
        packing.flatTarget = current
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getFlatness(self):
        """Per-SELECTED-vertex ``d / (a + b)``, the dimensionless flatness, as a flat array.

        1 is exactly flat. This is what the diagonal moment family holds and what a flattening ramp
        drives to 1; read it to see how close a stage is to being removable.

        ALWAYS FLATNESS, never the deficit. In DEVIATION mode the family's own ``quantity`` is
        ``1 - d/(a+b)`` -- the nonnegative budget a barrier ramp holds -- so it is converted back here.
        Getting this wrong is silent: both values live in [0, 1], so a caller reading the deficit as
        flatness sees a plausible number that means the opposite of what it says."""
        distribution = self._distributionConstraints()
        if distribution is not None and distribution.diagonal:
            t = distribution.quantity(self.packing, "diagonal")
            return 1.0 - t if distribution.deviation else t
        block = getattr(self.constraints, "block", self.constraints)
        if block is not None and getattr(block, "flatten", False):
            return block.flatness(self.packing)[block.diagonalSelected]
        raise ValueError(
            "no flatness constraint is active. Use setConstraints(flatten = True), which holds each "
            "selected vertex's d/(a+b) against its own target and is what a flattening ramp wants; "
            "setConstraints(diagonal = [...]) holds the same quantity as a DISTRIBUTION, which shares "
            "the work but cannot place individual vertices -- see setFlatTargets.")

    # UNVERIFIED(Cam)
    def setFlatnessTarget(self, mean, width = None, project = True):
        """Aim the DIAGONAL MOMENT family at this mean flatness, and optionally this spread.

        The ramp handle for global flattening, and the counterpart of ``setShapeBudget``: the moment
        family holds ``sum_i t_i^k`` over the SELECTED vertices' ``d/(a + b)``, so a schedule that
        wants a mean has to convert it. With ``m`` selected vertices, mean ``u`` and standard
        deviation ``w``:

            k = 1  ->  m u              k = 2  ->  m (u^2 + w^2)

        and a general ``k`` is taken about the mean to the same order. Requires
        ``setConstraints(..., diagonal = [...])``.

        WIDTH IS NOT OPTIONAL WHILE ANNEALING. Two rows holding a mean and a variance go numerically
        parallel as the width they hold goes to zero, so asking for ``width = 0`` requests exactly the
        degeneracy that stops the family finishing -- and the whole point of the moment form is that
        polygons SHARE the work, which they cannot do if the distribution is a point. Leave a width,
        walk the mean, and hand off to ``setConstraints(flatten = True)`` for the last approach;
        ``constraintConditioning`` says when.

        ``width = None`` keeps whatever spread the geometry currently has -- but that is NOT safe at a
        symmetric seed, where every polygon is identical and the spread is zero. Capturing zero asks
        the variance row for a point mass, and a moment constraint can only ever be NARROWED (the
        variance is at a minimum on a monodisperse configuration, so the retraction has no direction to
        widen along). Measured on the loaded cascade: the row asked for width 0.000000 while the load
        produced 0.070126, and the moment retraction stuck at residual 8.2e-02 after 358 passes. So a
        captured width below ``_MIN_MOMENT_WIDTH`` is refused rather than written."""
        distribution = self._distributionConstraints()
        if distribution is None or not getattr(distribution, "diagonal", False):
            raise ValueError(
                "no diagonal moment family is active; call setConstraints(..., diagonal = [1, 2]) "
                "first. For per-vertex targets -- the endgame of a ramp -- use setFlatTargets with "
                "setConstraints(flatten = True) instead.")
        current = np.asarray(distribution.quantity(self.packing, "diagonal"), dtype = float)
        count = int(current.size)
        if count == 0:
            raise ValueError("no vertices are selected; call selectFlattening() first.")
        mean = float(mean)
        captured = width is None
        width = float(np.std(current)) if captured else float(width)
        if width < _MIN_MOMENT_WIDTH:
            raise ValueError(
                f"the flatness width is {width:.3e}, at or below the degenerate floor "
                f"{_MIN_MOMENT_WIDTH:.0e}"
                + (" -- captured from a geometry whose polygons are all identical" if captured else "")
                + ".\n    Two rows holding a mean and a variance go numerically parallel as the width "
                f"goes to zero, and a moment family can only ever be NARROWED, so asking for a point "
                f"mass makes the retraction fight whatever spread the packing then develops (measured: "
                f"asked 0.000000 against a realized 0.070126, residual 8.2e-02 after 358 passes).\n"
                f"    Pass an explicit width and narrow it on a schedule. If the geometry really is "
                f"symmetric there is nothing to share yet -- load the packing first, or use "
                f"setConstraints(flatten = True) for per-vertex targets.")
        moments = np.asarray(distribution.familyMoments("diagonal"), dtype = float)
        # E[t^k] about the mean to second order: u^k + C(k,2) u^(k-2) w^2. Exact for k = 1 and 2,
        # which are the exponents a flattening ramp actually uses.
        reference = []
        for k in moments:
            value = mean ** k
            if k >= 2:
                value += 0.5 * k * (k - 1.0) * mean ** (k - 2.0) * width ** 2
            reference.append(count * value)
        distribution.setReference("diagonal", reference)
        if project:
            self.constraints.projectPositions(self.packing)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def setAlternatingEdges(self, u, parity = None):
        """Edges in consecutive PAIRS, alternating short-pair / long-pair at ratio ``u``. Returns self.

        Two of every four edges are short and two are long, so a short/long pairing sums to ``2P/n``:
        the perimeter target is reproduced exactly however u moves, and the area target is untouched.
        A scalar applies to every polygon, or pass one value each (container excluded). ``u = 0.5`` is
        equilateral, and the stage drives it toward 0 -- see ``alternating`` for why not TO 0.

        Pair it with ``selectPairCorners`` + ``setConstraints(flatten = True)``, which bends each LONG
        pair to a right angle: shorts gone and corners square IS a square, so one ramp gets there.
        Collapse with ``halveNumEdges(criterion = 'short')``, then again with ``'flat'``."""
        self.packing.targetEdgeLength = alternating.pairTargets(self.packing, u, parity = parity)
        self.packing.syncTargetPerimeter()
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def placeOnGrid(self, margin = 0.02):
        """Translate each polygon so its centroid sits on a lattice filling the unit cell. Returns self.

        THE DEPTH CONTACT LAW IS NON-MONOTONIC PAST HALF OVERLAP, so where a run STARTS decides
        whether it can ever separate. Penetration depth is the shortest translation that separates two
        shapes, and once one has passed more than halfway through its neighbour the shorter way out is
        the far side -- so the energy peaks near half overlap and FALLS toward full interpenetration.
        Measured on two 16-gons at kappa 4, walking one across the other:

            offset/side   0.00      0.25      0.50      0.75
            pair overlap  0.0833    0.0706    0.0467    0.0000
            pair energy   4.07e-06  2.71e-05  1.23e-04  3.84e-05

        A fully stacked pair is therefore a genuine force-balanced MINIMUM, not a violation. The build
        places polygons at random centres, so at any interesting density they start on the wrong side
        of that barrier and the relaxation pulls them further in. Measured at phi 0.45 on 5 polygons:
        from random centres the pair overlap went 0.134 -> 0.259 under relaxation and
        ``holdExcessEnergy`` decompressed to phi 0.216 without ever shedding its excess; from this grid
        it went 0.0045 -> 0.000006 and the controller compressed to phi 0.698 at the requested excess.

        No relaxation, no rotation -- a rigid translation each, so shapes and targets are untouched.
        The container is left alone.

        A SMALL residual overlap is fine and is not treated as failure. What matters is only that no
        pair starts more than halfway through, and the lattice cannot guarantee that by geometry alone
        because the polygons are arbitrarily ORIENTED -- a square of side 0.3 spans 0.424 across its
        diagonal, so it will not fit a 0.333 cell in every orientation while still being far too small
        to reach half overlap. Measured at phi 0.45 on 5 such polygons: the grid left an overlap of
        0.0045 and the relaxation took it to 0.000006. So the span is only WARNED about, and the
        caller should check the realized ``getPairOverlapArea`` -- the honest test."""
        packing = self.packing
        stop = self._nonContainer()
        r = packing.positions.reshape(-1, 2)
        starts = np.asarray(packing.startIndices, dtype = int)
        columns = int(np.ceil(np.sqrt(stop)))
        rows = int(np.ceil(stop / columns))
        widest = max(float(np.ptp(r[starts[p]:starts[p + 1]], axis = 0).max()) for p in range(stop))
        cell = min(1.0 / columns, 1.0 / rows)
        if widest > (1.0 + float(margin)) * cell and not getattr(self, "_warnedGridFit", False):
            # ONCE PER MODEL: the text carries live numbers, so it would never de-duplicate on its own.
            self._warnedGridFit = True
            warnings.warn(
                f"\n*** the polygons overhang a {columns} x {rows} lattice ***\n"
                f"    the widest spans {widest:.4f} against a cell of {cell:.4f}, so neighbouring "
                f"cells will overlap a little. That is usually harmless -- only a pair more than "
                f"HALFWAY through each other is trapped -- but check getPairOverlapArea() before "
                f"minimizing, and start at a lower packing fraction if it is a large fraction of a "
                f"polygon's area.", stacklevel = 2)
        for polygon in range(stop):
            block = r[starts[polygon]:starts[polygon + 1]]
            target = np.array([(polygon % columns + 0.5) / columns,
                               (polygon // columns + 0.5) / rows])
            block += target - block.mean(axis = 0)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def placeRandomly(self, margin = 0.0, rng = None):
        """Throw every polygon at a uniformly random point in the cell. Returns self.

        The counterpart of ``placeOnGrid``, and the right seed once the untangling is done on the
        overlap-AREA tier. That tier has no barrier -- the area is maximal at full stacking, so its
        gradient always pushes a pair apart -- whereas the depth law peaks near HALF overlap and falls
        to nothing beyond it, which is what made a lattice necessary when the depth tier saw the seed
        first. Untangle on area, and a random throw is safe however much it overlaps.

        It is also the better seed for a SEARCH. A lattice is one arrangement, and for most N it does
        not even divide evenly: five polygons on the 3 x 2 lattice ``placeOnGrid`` builds leaves a cell
        empty, and the packing jams around that vacancy at phi 0.450 against a record of 0.682. A throw
        gives a different basin per seed, which is what restarts need.

        ``margin`` insets the centroids from the wall, as a fraction of the cell; the container is a
        contact energy rather than a constraint, so a polygon thrown across it is pushed back in rather
        than being an error. Rigid translations only -- shapes and targets are untouched."""
        packing = self.packing
        stop = self._nonContainer()
        rng = self.rng if rng is None else asRng(rng)
        r = packing.positions.reshape(-1, 2)
        starts = np.asarray(packing.startIndices, dtype = int)
        low, high = float(margin), 1.0 - float(margin)
        if high <= low:
            raise ValueError(f"margin = {margin} leaves no room to place anything.")
        for polygon in range(stop):
            block = r[starts[polygon]:starts[polygon + 1]]
            block += rng.uniform(low, high, 2) - block.mean(axis = 0)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def selectPairCorners(self, parity = None):
        """Mark the middle vertex of every LONG pair for the flatness family. Returns self.

        The counterpart of ``selectFlattening`` for the pair protocol. The corner is the vertex
        between a long pair's two edges, index ``parity + 3`` (mod 4), and it is the one that has to
        become a right angle -- one vertex in four. ``parity = None`` CHOOSES it, by putting the
        corners on the phase that is already sharpest (``alternating.chooseParity``); the choice is
        stored on the packing so the edge mask and the ramp inherit it and cannot disagree. Passing
        the wrong phase asks every corner to move three slots round the polygon and the retraction
        does not recover -- measured max|C| 1.5e+04.

        With the mask set, ``setConstraints(flatten = True)`` holds ``d/(a + b)`` there, which for a
        pair's two equal edges is ``d/(2 l) = cos(theta/2)``. Walk it with ``setFlatTargets`` from just
        under 1 (straight) to ``alternating.RIGHT_ANGLE`` = 0.70710678, which is Cam's
        ``d = sqrt(2) l``. Selecting EVERY vertex instead would bend the short pairs too, and that
        target set is infeasible at every u > 0 -- see ``alternating``."""
        parity = alternating.ensureParity(self.packing, parity)
        self.packing.diagonalMask = alternating.cornerMask(self.packing, parity = parity)
        return self._rebuildConstraints()

    # UNVERIFIED(Cam)
    def setCornerTargets(self, target):
        """Aim every selected corner's ``d/(a + b)`` at ``target``. Returns self.

        A thin alias for ``setFlatTargets`` that exists to say what the number means here: 1 is
        straight, ``alternating.RIGHT_ANGLE`` is 90 degrees, and the ramp runs between them."""
        return self.setFlatTargets(target)

    # UNVERIFIED(Cam)
    def getCornerRatios(self):
        """Live ``d/(a + b)`` at each selected corner -- 1 straight, 0.70710678 a right angle."""
        return self.getFlatness()

    # UNVERIFIED(Cam)
    def setRegularTargets(self):
        """Make every polygon's edge targets REGULAR at its own target AREA. Returns self.

        The end of a cascade: the perimeter target is DERIVED from the area rather than carried, so
        the shape index lands exactly on the regular floor. At n = 4 that floor is 4 and the square
        is the only shape left -- see ``alternating.regularTargets`` for why carrying the perimeter
        instead leaves 85 degree corners."""
        self.packing.targetEdgeLength = alternating.regularTargets(self.packing)
        self.packing.syncTargetPerimeter()
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getAlternatingRatios(self, parity = None):
        """Live ``u`` per polygon from the GEOMETRY -- the share of the perimeter the short pairs hold.

        What a ramp watches: it starts near the drawn distribution, tracks the targets through each
        round, and reaching ~0 is what makes the collapse exact."""
        return alternating.shortRatios(self.packing, parity = parity)

    # UNVERIFIED(Cam)
    def getAlternatingMask(self, parity = None):
        """Per-EDGE bool, True on the short pairs -- for ``draw(edgeMask = ...)`` and for slicing."""
        return alternating.shortMask(self.packing, parity = parity)

    # UNVERIFIED(Cam)
    def getTargetShapeMargin(self):
        """``(index, floor)`` per polygon: target shape index vs the smallest its EDGE TARGETS admit.

        The feasibility guard for an alternating ramp, and NOT the one ``setConstraints`` applies --
        that one bounds the area by the regular n-gon's, which is blind to how the perimeter is
        distributed. See ``alternating.targetShapeMargin``: ``index <= floor`` means the constraint
        set is empty."""
        return alternating.targetShapeMargin(self.packing)

    # UNVERIFIED(Cam)
    def relaxShapes(self, maxSteps = 20000, fThreshold = 1e-12, kEdge = 1.0, kArea = 1.0,
                    **fireKwargs):
        """Relax every polygon onto its OWN edge and area targets with springs alone, no overlap.

        Each shape moves independently in free space, exactly as the build does -- the packing energy,
        the container and the constraints are all absent. That is what makes it the cheap way to
        follow a target that has just moved: the geometry is walked to the new shape before the
        expensive constrained relaxation ever sees it, so SHAKE starts near the manifold instead of
        being asked to jump onto it.

        The container and any pinned vertex are FROZEN (their force is zeroed), so the wall is not
        deformed by springs it was never meant to have. Returns (energy, steps, converged).

        ABSOLUTE springs, as the free-space build uses, NOT the packing's dimensionless ones. The
        relative edge term weighs each edge by 1/l0^2, so its curvature is 1/l0^2 and the stable
        timestep is O(l0) -- which goes to zero exactly when an alternating ramp is driving a target
        length there. Measured at u = 0.2 on 16-gons of area 0.1: the default FIRE step blew up to an
        infinite energy in 97 steps. The absolute form's curvature is the spring constant itself, so
        one timestep serves the whole ramp. The usual objection does not apply here -- each shape
        relaxes ALONE in free space, so the weighting sets the rate, not which minimum is found."""
        packing = self.packing
        frozen = np.zeros(packing.numVertices, dtype = bool)
        container = getattr(packing, "containerIndex", None)
        if container is not None:
            starts = np.asarray(packing.startIndices, dtype = int)
            frozen[int(starts[container]) : int(starts[container + 1])] = True
        if packing.pinned is not None:
            frozen |= np.asarray(packing.pinned, dtype = bool)

        def forceEnergy(p):
            energy, force = eqSoftBodyEnergyForce(p, kEdge, kArea)
            force = force.reshape(-1, 2)
            force[frozen] = 0.0
            return energy, force.reshape(-1)

        fireKwargs.setdefault("progress", False)
        energy, steps, converged = minimize.minimizeFIRE(
            packing, forceEnergy, maxSteps = maxSteps, fThreshold = fThreshold, **fireKwargs)
        self._forces = None
        self._energy = None
        return energy, steps, converged

    # UNVERIFIED(Cam)
    def updateAlternatingDiagonals(self, targets, kEdge = 1.0, kDiagonal = 1.0, kArea = 1.0,
                                   maxSteps = 20000, fThreshold = 1e-12, **fireKwargs):
        """Springs on every EDGE, every OTHER DIAGONAL and every AREA, pulled to their targets and
        nothing else. Returns ``(energy, steps, converged)``.

        No overlap, no container, no self-repulsion, no constraints -- the same free-space relaxation
        ``relaxShapes`` performs, with the skip-one diagonals added. Each shape moves alone, so this is
        the cheap way to follow a diagonal target that has just moved: walk the geometry to the new
        shape first, then let the expensive constrained relaxation start near the manifold instead of
        being asked to jump onto it.

        IT LEAVES THE CONSTRAINT MANIFOLD, deliberately. Nothing here knows about ``setConstraints``,
        so ``max|C|`` will be nonzero afterwards -- project or relax before reading anything from the
        packing.

            U = kEdge/2 sum_edges (l - l0)^2  +  kDiagonal/2 sum_selected (d - d0)^2
                                              +  kArea/2 sum_polygons (A - A0)^2

        ALL THREE TOGETHER ARE FEASIBLE AT kappa = 4, which is not obvious and is what makes this
        usable. Flattening an n-gon onto its alternating diagonals turns it into an n/2-gon of side 2l
        at the SAME perimeter, and the most area such a polygon can enclose is the regular one's. At
        kappa 4 the area being held is well under that ceiling -- 64 l^2 against 80.44 at n = 32 -> 16,
        so 20% of slack -- and the margin closes only at n = 4, where the square is the unique answer.
        Without the area spring the shape simply drifts: measured pushing to 0.98 x 2l, the area moved
        +1.92% and kappa wandered 3.602 -> 3.568.

        THE AREA TERM IS ABSOLUTE, like the other two, so its residual carries L^2 where theirs carry
        L. On a packing of small polygons that makes it much the weakest of the three at equal
        stiffness -- at ``A ~ 0.06`` an area error of 1% is 6e-4 against an edge error of 1% at 6e-4 on
        a term whose own scale is ten times larger. Raise ``kArea`` if the areas lag; the check reports
        what a given ratio holds.

        ``targets`` is indexed the way the notebook builds it: entry ``[p, i]`` is the target for
        ``|v_{2i} - v_{2i-2}|``, the diagonal joining EVEN-indexed vertices and spanning the odd vertex
        between them, with ``i = 0`` wrapping to the last one. That is ``positions[:, ::2]`` against
        ``np.roll(..., 1)``. Pass ``(bodies, n/2)`` when every polygon has the same count, or a flat
        array in polygon order when they do not.

        THE SAME SET AS ``setConstraints(alternatingDiagonal = ...)`` and ``getAlternatingDiagonals``,
        by construction and by test. The three are easy to get one vertex out of phase with each other
        -- a flatness row is indexed by the vertex a diagonal is CENTRED on, while these targets are
        indexed by the chord's own position -- and the failure is silent: constraining one phase while
        driving the other flattens every vertex and collapses the polygon.

        ABSOLUTE springs, like the free-space build and for the reason recorded in ``relaxShapes``: the
        relative form weighs each term by 1/l0^2, so its stable timestep is O(l0) and vanishes exactly
        when a ramp drives a target length toward zero."""
        packing = self.packing
        starts = np.asarray(packing.startIndices, dtype = int)
        container = getattr(packing, "containerIndex", None)
        bodies = [p for p in range(packing.numPolygons)
                  if container is None or p != int(container)]
        counts = [int(starts[p + 1] - starts[p]) for p in bodies]
        for polygon, count in zip(bodies, counts):
            if count % 2:
                raise ValueError(
                    f"polygon {polygon} has {count} vertices, which is ODD -- alternating diagonals do "
                    f"not close around the loop, so the target array has no consistent length.")
        wanted = sum(count // 2 for count in counts)
        values = np.asarray(targets, dtype = float).reshape(-1)
        if values.size != wanted:
            raise ValueError(
                f"targets has {values.size} entries; this packing needs {wanted} -- one per alternating "
                f"diagonal, summed over the {len(bodies)} non-container polygons with vertex counts "
                f"{counts}. The container is never included.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("every diagonal target must be positive and finite.")

        # Index pairs built from startIndices rather than from a uniform n, because nothing guarantees
        # one -- halveNumEdges alone can leave one polygon at 16 and another at 8.
        diagonalFrom, diagonalTo, edgeFrom, edgeTo, edgeBack = [], [], [], [], []
        for polygon, count in zip(bodies, counts):
            base = int(starts[polygon])
            even = np.arange(0, count, 2)
            diagonalTo.append(base + even)
            diagonalFrom.append(base + (even - 2) % count)
            local = np.arange(count)
            edgeFrom.append(base + local)
            edgeTo.append(base + (local + 1) % count)
            edgeBack.append(base + (local - 1) % count)
        diagonalFrom = np.concatenate(diagonalFrom)
        diagonalTo = np.concatenate(diagonalTo)
        edgeFrom = np.concatenate(edgeFrom)
        edgeTo = np.concatenate(edgeTo)
        edgeBack = np.concatenate(edgeBack)
        edgeTargets = np.asarray(packing.targetEdgeLength, dtype = float)[edgeFrom]
        # Which BODY polygon each of those vertices belongs to, renumbered 0..len(bodies)-1 so the
        # container's row is absent rather than merely skipped -- its signed area is negative by
        # winding and would otherwise be chased toward a positive target.
        owner = np.concatenate([np.full(count, index, dtype = int)
                                for index, count in enumerate(counts)])
        areaTargets = np.asarray(packing.targetArea, dtype = float)[np.asarray(bodies, dtype = int)]

        # THE TRIANGLE INEQUALITY CAPS EVERY DIAGONAL AT a + b, and a ramp aimed at "2 x the edge
        # length" is aimed exactly AT that cap. A target past it cannot be met by any geometry, so the
        # two spring sets simply fight and settle at a compromise -- which reads as a converged
        # relaxation that quietly missed. Measured against the EDGE TARGETS rather than the live edges,
        # since those are where the edge springs are pulling to.
        reach = np.asarray(packing.targetEdgeLength, dtype = float)
        adjacent = reach[diagonalFrom] + reach[(diagonalFrom + 1) % packing.numVertices]
        beyond = values > adjacent
        if np.any(beyond) and not getattr(self, "_warnedDiagonalReach", False):
            self._warnedDiagonalReach = True
            worst = float(np.max(values[beyond] / adjacent[beyond]))
            warnings.warn(
                f"\n*** {int(beyond.sum())} diagonal targets exceed a + b and cannot be reached ***\n"
                f"    the worst asks for {worst:.4f} times the sum of its own two edges, and the "
                f"triangle inequality caps a diagonal at exactly that sum.\n"
                f"    The edge and diagonal springs will settle at a compromise and the relaxation "
                f"will report itself converged, having missed. Note that a target AT the cap is "
                f"degenerate rather than merely tight: the vertex is then collinear, so the pull that "
                f"straightens it has vanished. Aim just short.", stacklevel = 2)

        frozen = np.zeros(packing.numVertices, dtype = bool)
        if container is not None:
            frozen[int(starts[int(container)]) : int(starts[int(container) + 1])] = True
        if packing.pinned is not None:
            frozen |= np.asarray(packing.pinned, dtype = bool)

        def springs(coordinates, first, second, rest, stiffness, force):
            """Hookean pull along each chord, accumulated into ``force``. Returns the energy."""
            span = coordinates[second] - coordinates[first]
            length = np.hypot(span[:, 0], span[:, 1])
            # A chord of exactly zero length has no direction; its pull is dropped rather than made
            # into a division by zero. It cannot happen at an edge that a target is holding open.
            safe = np.where(length > _MIN_SPRING_LENGTH, length, 1.0)
            stretch = np.where(length > _MIN_SPRING_LENGTH, length - rest, 0.0)
            pull = (stiffness * stretch / safe)[:, None] * span
            np.add.at(force, first, pull)
            np.add.at(force, second, -pull)
            return 0.5 * stiffness * float(np.sum(stretch ** 2))

        def areaSpring(coordinates, force):
            """``kArea/2 sum_p (A_p - A0_p)^2`` on the BODY polygons. Returns the energy.

            The shoelace area and its gradient ``dA/dr_k = (y_next - y_prev, x_prev - x_next) / 2``,
            the same pair ``eqSoftBodyEnergyForce`` uses, gathered per body vertex."""
            here = coordinates[edgeFrom]
            ahead = coordinates[edgeTo]
            areas = np.zeros(len(counts), dtype = float)
            np.add.at(areas, owner, here[:, 0] * ahead[:, 1] - ahead[:, 0] * here[:, 1])
            areas *= 0.5
            # RELATIVE, unlike the two length terms, and the asymmetry is deliberate. An area residual
            # carries L^2 where a length residual carries L, so at equal stiffness the absolute form is
            # weaker by the polygon's own scale -- measured on A ~ 0.06, kArea = 1 left the area 1.45%
            # adrift where kArea = 100 held it to 0.04%. A default that silently does nothing is worse
            # than no default, and dividing by A0 makes the residual dimensionless so kArea = 1 means
            # the same thing at any polygon size. The length terms keep the absolute form for the
            # reason relaxShapes records: their relative weight 1/l0^2 makes the stable timestep O(l0),
            # which vanishes when a ramp drives a target length toward zero. An area target never does.
            residual = areas / areaTargets - 1.0
            behind = coordinates[edgeBack]
            gradient = 0.5 * np.stack([ahead[:, 1] - behind[:, 1],
                                       behind[:, 0] - ahead[:, 0]], axis = 1)
            np.add.at(force, edgeFrom,
                      -(kArea * residual / areaTargets)[owner][:, None] * gradient)
            return 0.5 * kArea * float(np.sum(residual ** 2))

        def forceEnergy(current):
            coordinates = current.positions.reshape(-1, 2)
            force = np.zeros((current.numVertices, 2), dtype = float)
            energy = springs(coordinates, edgeFrom, edgeTo, edgeTargets, kEdge, force)
            energy += springs(coordinates, diagonalFrom, diagonalTo, values, kDiagonal, force)
            energy += areaSpring(coordinates, force)
            force[frozen] = 0.0
            return energy, force.reshape(-1)

        fireKwargs.setdefault("progress", False)
        energy, steps, converged = minimize.minimizeFIRE(
            packing, forceEnergy, maxSteps = maxSteps, fThreshold = fThreshold, **fireKwargs)
        self._forces = None
        self._energy = None
        return energy, steps, converged

    # UNVERIFIED(Cam)
    def getAlternatingDiagonals(self):
        """The lengths ``updateAlternatingDiagonals`` takes targets for, in the same order. Shape ``(bodies, n/2)``
        when every polygon has the same count, otherwise flat.

        The measurement half of the pair, so a ramp can read where it is before saying where to go
        without rebuilding the indexing by hand."""
        packing = self.packing
        starts = np.asarray(packing.startIndices, dtype = int)
        container = getattr(packing, "containerIndex", None)
        bodies = [p for p in range(packing.numPolygons)
                  if container is None or p != int(container)]
        coordinates = packing.positions.reshape(-1, 2)
        lengths = []
        for polygon in bodies:
            base, stop = int(starts[polygon]), int(starts[polygon + 1])
            count = stop - base
            even = np.arange(0, count, 2)
            span = coordinates[base + even] - coordinates[base + (even - 2) % count]
            lengths.append(np.hypot(span[:, 0], span[:, 1]))
        counts = {len(row) for row in lengths}
        return np.array(lengths) if len(counts) == 1 else np.concatenate(lengths)

    # UNVERIFIED(Cam)
    def resampleEdges(self, count, skipContainer = True):
        """Re-place every polygon's vertices at ``count`` points spaced equally by ARC LENGTH, and make
        the edge targets equilateral at the perimeter they already had. Returns self.

        The coarsener for the vertex-count cascade, and the counterpart of ``halveNumEdges`` for the
        case where the flats are NOT already collinear. That one is exact but presupposes a completed
        morph stage -- it refuses unless the alternating vertices carry no geometry -- so it cannot
        start a cascade in which nothing has imposed corners yet. This one always works and is
        approximate: chords cut across whatever curvature lies between the samples, so the realized
        area and perimeter both drop slightly and the following relaxation puts them back.

        WHY THIS IS A STIFFENING RAMP ALL BY ITSELF. Under ``setConstraints(area = True, edge = True)``
        with equal edge targets, a polygon carries ``n`` edge rows and one area row against ``2n - 3``
        shape degrees of freedom, leaving exactly ``n - 4`` free. So the vertex count IS the
        compliance: 28 at n = 32, 12 at 16, 4 at 8, and ZERO at n = 4 -- where an equilateral
        quadrilateral is a rhombus with ``kappa = 4 / sqrt(sin theta)``, so ``kappa = 4`` forces
        ``theta = 90`` degrees and the square is the only shape left. Nothing has to impose corners or
        pin a diagonal; the square is what the constraint set MEANS once the count comes down.

        The targets are rewritten rather than carried: every edge becomes ``targetPerimeter / count``,
        which preserves each polygon's perimeter target exactly while making it equilateral, and the
        area target is untouched. So the target shape index is unchanged and stays reachable -- the
        regular floor rises with falling n (3.5506 at 32, 3.6407 at 8, exactly 4.0000 at 4) and only
        MEETS kappa = 4 at the end of the cascade.

        Sampling starts at each polygon's vertex 0 and is otherwise blind to where the corners are. If
        the shape already has corners, resampling will generally not land a vertex on one; that costs
        accuracy at low counts, and is the reason to step 32 -> 16 -> 8 -> 4 rather than jump.

        ``targetDiagonal`` is dropped, as in ``halveNumEdges`` -- it describes the old spacing."""
        count = int(count)
        if count < 3:
            raise ValueError(f"a polygon needs at least 3 vertices, got count = {count}")
        packing = self.packing
        container = getattr(packing, "containerIndex", None)
        r = packing.positions.reshape(-1, 2)
        starts = np.asarray(packing.startIndices, dtype = int)
        pinned = packing.pinned

        newLoops, newPinned, newEdge, retained = [], [], [], []
        for polygon in range(packing.numPolygons):
            a, b = int(starts[polygon]), int(starts[polygon + 1])
            loop = r[a : b]
            isContainer = container is not None and polygon == int(container)
            if isContainer and skipContainer:
                newLoops.append(loop)
                if pinned is not None:
                    newPinned.append(pinned[a : b])
                if packing.targetEdgeLength is not None:
                    newEdge.append(packing.targetEdgeLength[a : b])
                continue
            edges = np.roll(loop, -1, axis = 0) - loop
            lengths = np.linalg.norm(edges, axis = 1)
            walk = np.concatenate([[0.0], np.cumsum(lengths)])
            perimeter = float(walk[-1])
            if perimeter <= 0.0:
                raise ValueError(f"polygon {polygon} has zero perimeter and cannot be resampled")
            if count < loop.shape[0]:
                # KEEP THE CORNERS, do not place points blindly. Equal arc-length spacing ignores where
                # the geometry actually lives, and on a smooth polygon it throws away area at a rate
                # that grows as the count falls: measured on regular polygons, 32 -> 16 keeps 98.1% but
                # 8 -> 4 keeps only 70.7% and 5 -> 4 keeps 65.5%. The area TARGET does not move, so the
                # constraint projection then re-inflates the polygon by up to 19% in linear size -- and
                # the container is a contact energy rather than a constraint, so that inflation pushes
                # vertices straight through the wall. Measured in a packing: 1 vertex outside became 17
                # the instant the first rung landed.
                #
                # EVENLY SPACED, at the PHASE that keeps the most area. Ranking vertices by curvature
                # and taking the sharpest sounds better and is worse: on a packed polygon -- flat
                # where it presses a neighbour, curved elsewhere -- it keeps every curved vertex and
                # skips whole flats, so the loss concentrates into a few long chords. Measured on a
                # four-faced 32-gon, curvature ranking spanned 6 original edges with one chord and
                # kept 95.71%, against 97.95% for even spacing: chord area-loss grows about as the
                # CUBE of the span, so spreading it beats aiming it.
                #
                # The phase search is what makes the cornered case exact, and it is the rule
                # ``halveNumEdges`` already used: on a square carrying collinear extras, one phase
                # lands on the corners and keeps 100%.
                order = np.arange(count) * (loop.shape[0] / count)
                best, keep = -1.0, None
                for offset in range(loop.shape[0]):
                    candidate = np.unique(np.floor(order + offset).astype(int) % loop.shape[0])
                    if candidate.size != count:
                        continue
                    kept = _loopArea(loop[candidate])
                    if kept > best:
                        best, keep = kept, candidate
                if keep is None:
                    raise ValueError(
                        f"polygon {polygon}: cannot choose {count} evenly spaced vertices from "
                        f"{loop.shape[0]}")
                newLoops.append(loop[keep])
            else:
                wanted = np.arange(count) * (perimeter / count)
                # searchsorted on the cumulative walk finds which ORIGINAL edge each new vertex lands
                # on; the remainder is how far along it. Used only when REFINING, where nothing is lost.
                segment = np.clip(np.searchsorted(walk, wanted, side = "right") - 1,
                                  0, lengths.size - 1)
                along = (wanted - walk[segment]) / np.maximum(lengths[segment], 1e-300)
                newLoops.append(loop[segment] + along[:, None] * edges[segment])
            retained.append(_loopArea(newLoops[-1]) / max(_loopArea(loop), 1e-300))
            if pinned is not None:
                # A resampled vertex is a new point, so a pin cannot follow it. Refuse rather than
                # silently drop the constraint the caller asked for.
                if pinned[a : b].any():
                    raise ValueError(
                        f"polygon {polygon} has pinned vertices, which resampling would move. Pin the "
                        f"container instead (skipContainer = True holds it fixed), or release the "
                        f"pins before resampling.")
                newPinned.append(np.zeros(count, dtype = bool))
            if packing.targetEdgeLength is not None:
                # EQUILATERAL at the perimeter it already had: the perimeter target is preserved and
                # the area target untouched, so the target shape index does not move -- EXCEPT where
                # that would be infeasible at the new count.
                #
                # THE END OF THE CASCADE IS A KNIFE EDGE and this is what keeps it on the right side.
                # The isoperimetric floor RISES as n falls (3.5506 at 32, 4.0000 at 4), so a target
                # shape index of 4 has slack everywhere except n = 4, where it sits EXACTLY on the
                # bound. A relaxation that left kappa at 3.999999 -- six digits correct -- is then
                # infeasible by 5e-7, and setConstraints refuses the set outright. Measured: the
                # cascade died at exactly that, "target area is 1.0000x the maximum".
                #
                # So the perimeter target is raised to the shortest that CAN enclose the area target
                # at this count. Above n = 4 it never binds; at n = 4 it lands the target exactly on
                # the square, which is where the protocol was aiming anyway.
                target = float(packing.targetPerimeter[polygon])
                floor = regularShapeIndex(count) * np.sqrt(
                    abs(float(packing.targetArea[polygon])))
                newEdge.append(np.full(count, max(target, floor) / count))

        counts = np.array([loop.shape[0] for loop in newLoops], dtype = int)
        packing.positions = np.concatenate([loop.reshape(-1) for loop in newLoops])
        packing.startIndices = np.concatenate([[0], np.cumsum(counts)])
        packing.numVertices = packing.positions.size // 2
        packing.shapeId, packing.next, packing.prev = buildConnectivity(packing.startIndices)
        packing.force = np.zeros_like(packing.positions)
        packing.velocities = np.zeros_like(packing.positions)
        packing.pinned = np.concatenate(newPinned) if pinned is not None else None
        if packing.targetEdgeLength is not None:
            packing.targetEdgeLength = np.concatenate(newEdge)
            packing.syncTargetPerimeter()
        packing.targetDiagonal = None
        packing._selfRepKey = None

        self.n = int(counts[0])
        self._rebuildConstraints()
        # HOW MUCH SHAPE THIS RUNG COST, which is the thing to watch. Decimation is exact only when
        # the dropped vertices carried no geometry; anything less means the area target now exceeds
        # what the remaining vertices enclose, and the projection makes up the difference by inflating
        # the polygon -- outward, through a container that is a contact energy and cannot stop it.
        worst = min(retained) if retained else 1.0
        if worst < 1.0 - _RESAMPLE_LOSS:
            warnings.warn(
                f"\n*** resampling to {count} vertices cost real shape ***\n"
                f"    the worst polygon kept only {100 * worst:.1f}% of its area, so its area target "
                f"now asks for {1.0 / max(worst, 1e-300):.3f}x what its vertices enclose and the "
                f"constraint projection will INFLATE it by {(1.0 / max(worst, 1e-300)) ** 0.5:.3f}x "
                f"in linear size.\n"
                f"    That push is not stopped by the container, which is a contact energy rather "
                f"than a constraint, so expect vertices outside the box. The shape was not ready to "
                f"lose vertices: relax further at the previous count until the vertices being dropped "
                f"lie flat, or step down more gently.", stacklevel = 2)
        self.lastResampleRetention = worst
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def halveNumEdges(self, separation = 0.05, skipContainer = True, criterion = "flat"):
        """Drop every OTHER vertex of each polygon, halving the vertex count. Returns self.

        The inverse of ``doubleNumEdges``, and the step that makes a symmetry cascade affordable. After
        a morph stage has driven a 2m-gon onto an m-cornered template, the m vertices between the
        corners are COLLINEAR -- they carry no geometry and full cost -- so removing them is exact and
        leaves a regular m-gon. Verified to machine precision: 32 -> 16 -> 8 -> 4 reproduces the regular
        turning angle (22.5000, 45.0000, 90.0000 degrees) and shape index (3.5680, 3.6407, 4.0000) with
        an edge spread of ~1e-16.

        REMOVAL IS ONLY EXACT AT THE END OF A STAGE, so it is checked rather than assumed. The test is
        SCALE-FREE: the flattest alternating set must turn less than ``separation`` times the sharpest
        corner. An absolute tolerance cannot work here -- a relaxed configuration never sits exactly on
        its template, so any threshold tight enough to be meaningful at one stage rejects a perfectly
        good packing at another. The two populations are far apart by construction (at the 16-gon stage
        the corners turn 22.5 degrees against flats that should turn ~0), so a ratio separates them
        cleanly at any scale. Measured mid-stage the flats still turn 8.4 degrees at morph 0.25 and 2.8
        at 0.75 against a 90 degree corner, so a premature call is still refused.

        This is what lets a halving cascade carry no state but the vertex count: each stage is the same
        operation on a regular polygon, so nothing has to know or infer which step it is on.

        ``criterion = "short"`` collapses the vanishing short PAIRS an ``setAlternatingEdges`` ramp
        produces: a pair of short edges brings three vertices together, so two of every four go and
        the count still halves. The default test cannot be used there and would refuse every such
        packing -- when an edge shrinks to nothing its endpoints coincide but do not straighten, and
        the vanishing edge keeps whatever direction it had. Measured on 16-gons at u = 2e-3, the
        FLATTEST alternating set turned 173.9 degrees. The short test is the matching scale-free one:
        the longest edge being dropped must be under ``separation`` times the shortest surviving.

        Run it, then run the default: with the long pairs bent to right angles the collapsed shape is
        a square carrying collinear midpoints, which is exactly what ``criterion = "flat"`` removes
        exactly. Two decimations, one ramp."""
        if criterion not in ("flat", "short"):
            raise ValueError(f"criterion must be 'flat' (drop collinear vertices) or 'short' (drop "
                             f"the far end of a vanishing edge), got {criterion!r}.")
        packing = self.packing
        container = getattr(packing, "containerIndex", None)
        r = packing.positions.reshape(-1, 2)
        starts = np.asarray(packing.startIndices, dtype = int)
        pinned = packing.pinned

        # THE SELECTION FOLLOWS THE VERTICES. ``diagonalMask`` is indexed by vertex, so leaving it at
        # the old length does not raise -- the ragged gather just reads whatever now sits at those
        # indices, and the flatness family silently holds the wrong vertices.
        selection = getattr(packing, "diagonalMask", None)
        newSelection = []

        newLoops, newPinned, newEdge, newDiagonal = [], [], [], []
        for polygon in range(packing.numPolygons):
            a, b = int(starts[polygon]), int(starts[polygon + 1])
            loop = r[a : b]
            isContainer = container is not None and polygon == int(container)
            if (isContainer and skipContainer) or loop.shape[0] < 8 or loop.shape[0] % 2:
                keep = np.arange(loop.shape[0])
            elif criterion == "short":
                if loop.shape[0] % 4:
                    raise ValueError(
                        f"polygon {polygon} has {loop.shape[0]} vertices; the short-PAIR collapse "
                        f"needs a multiple of 4 (edges run in pairs, alternating short and long).")
                lengths = np.linalg.norm(np.roll(loop, -1, axis = 0) - loop, axis = 1)
                # WHICH PHASE IS SHORT IS MEASURED, not assumed: the ramp's parity is a caller's
                # choice and a packing reloaded from disk does not carry it.
                total = [float(lengths[offset::4].sum() + lengths[(offset + 1) % 4::4].sum())
                         for offset in range(4)]
                parity = int(np.argmin(total))
                vanishing = np.concatenate([lengths[parity::4], lengths[(parity + 1) % 4::4]])
                surviving = np.concatenate([lengths[(parity + 2) % 4::4], lengths[(parity + 3) % 4::4]])
                longest, shortest = float(vanishing.max()), float(surviving.min())
                if longest > float(separation) * max(shortest, 1e-30):
                    raise ValueError(
                        f"polygon {polygon} has no collapsible pairs: the longest of the vanishing "
                        f"set is {longest:.6g} against {shortest:.6g} for the shortest survivor, a "
                        f"ratio of {longest / max(shortest, 1e-30):.4f} against the "
                        f"{separation:.3f} allowed. Run the u ramp further before collapsing.")
                # A short PAIR collapses THREE vertices into one, so two of every four go: the pair's
                # two interior vertices. What survives is the vertex the short pair leaves from and
                # the long pair's corner, which is the square's.
                local = np.arange(loop.shape[0])
                keep = local[((local - parity) % 4 == 0) | ((local - parity) % 4 == 3)]
            else:
                following = np.roll(loop, -1, axis = 0)
                preceding = np.roll(loop, 1, axis = 0)
                ahead = following - loop
                behind = loop - preceding
                turn = np.abs(np.arctan2(np.cross(behind, ahead),
                                         np.einsum("ij,ij->i", behind, ahead)))
                # The odd slots are the ones a morph stage flattens; both parities are measured so a
                # packing that happens to be phased the other way is still handled.
                worst = [float(turn[offset::2].max()) for offset in (0, 1)]
                drop = int(np.argmin(worst))
                sharpest = float(turn.max())
                if worst[drop] > float(separation) * max(sharpest, 1e-30):
                    raise ValueError(
                        f"polygon {polygon} has no removable vertices: the flattest alternating set "
                        f"turns {np.degrees(worst[drop]):.4f} degrees against a sharpest corner of "
                        f"{np.degrees(sharpest):.4f}, a ratio of "
                        f"{worst[drop] / max(sharpest, 1e-30):.3f} against the {separation:.3f} "
                        f"allowed. Halving is exact only once a morph stage has completed "
                        f"(morph = 1), where those vertices are collinear.")
                keep = np.arange(loop.shape[0])[1 - drop::2]
            newLoops.append(loop[keep])
            if selection is not None:
                newSelection.append(np.asarray(selection, dtype = bool)[a : b][keep])
            if pinned is not None:
                newPinned.append(pinned[a : b][keep])
            if packing.targetEdgeLength is not None:
                # A kept vertex now owns its own edge plus the flattened one that followed it, and the
                # two were collinear, so the target is their SUM rather than either alone.
                own = packing.targetEdgeLength[a : b]
                merged = own[keep]
                if len(keep) < loop.shape[0]:
                    # A kept vertex now owns every edge up to the NEXT kept one, so its target is
                    # their sum. Written as the general cyclic run rather than ``own[k] + own[k+1]``
                    # because a short-pair collapse drops two vertices in a row, not one: the runs are
                    # three edges and then one. Reduces to the pair sum for a stride-2 drop.
                    bounds = np.append(keep, keep[0] + loop.shape[0])
                    merged = np.array([
                        float(own[np.arange(bounds[j], bounds[j + 1]) % loop.shape[0]].sum())
                        for j in range(len(keep))])
                    # THE END OF THE CASCADE IS A KNIFE EDGE, and this is the same guard
                    # ``resampleEdges`` carries. The isoperimetric floor RISES as n falls -- 3.5506 at
                    # 32, exactly 4.0000 at 4 -- so a shape index of 4 has slack everywhere except the
                    # last rung, where it sits ON the bound. ``syncTargetAreas`` leaves the realized
                    # area a hair above the target, which puts the index a hair BELOW 4, and
                    # setConstraints then refuses the merged set outright: measured "target area is
                    # 1.0000x the maximum, shape index 4.000000 against a floor of 4.000000". Raising
                    # the perimeter to the shortest that can enclose the area target fixes it, and
                    # above n = 4 it never binds.
                    floor = regularShapeIndex(merged.size) * np.sqrt(
                        abs(float(packing.targetArea[polygon])))
                    total = float(merged.sum())
                    if total < floor:
                        merged = merged * (floor / total)
                newEdge.append(merged)
            if packing.targetDiagonal is not None:
                newDiagonal.append(packing.targetDiagonal[a : b][keep])

        counts = np.array([loop.shape[0] for loop in newLoops], dtype = int)
        packing.positions = np.concatenate([loop.reshape(-1) for loop in newLoops])
        packing.startIndices = np.concatenate([[0], np.cumsum(counts)])
        packing.numVertices = packing.positions.size // 2
        packing.shapeId, packing.next, packing.prev = buildConnectivity(packing.startIndices)
        packing.force = np.zeros_like(packing.positions)
        packing.velocities = np.zeros_like(packing.positions)
        packing.pinned = np.concatenate(newPinned) if pinned is not None else None
        if selection is not None:
            packing.diagonalMask = np.concatenate(newSelection)
        # THE FLAT TARGETS ARE DROPPED, not carried, and the difference is not cosmetic. An unset
        # target makes ``flatTargets`` CAPTURE the live flatness, so switching the family on is inert;
        # a carried one COMMANDS, and the vertices a new stage selects are nowhere near the values the
        # last stage left. Measured when they were carried through: the 32 -> 16 -> 8 -> 4 chain in
        # flattenCascadeCheck was refused at the last rung. Re-aim with setFlatTargets after selecting.
        packing.flatTarget = None
        if packing.targetEdgeLength is not None:
            packing.targetEdgeLength = np.concatenate(newEdge)
            packing.syncTargetPerimeter()
        # The diagonals describe the OLD vertex spacing and are meaningless at the new one. Dropped
        # rather than rescaled, so the next setShapeTemplate writes them fresh; leaving stale values is
        # how a template comes to describe two different polygons at once.
        packing.targetDiagonal = None
        packing._selfRepKey = None

        self.n = None if self.n is None else int(counts[0])
        self._rebuildConstraints()
        self._forces = None
        self._energy = None
        return self

    def getTargetAreas(self):
        """Per-POLYGON target areas, shape (numPolygons,)."""
        return self.packing.targetArea

    def getTargetEdgeLengths(self):
        """Per-EDGE target lengths, shape (numVertices,) -- one per vertex, for the edge leaving it.

        Every edge carries its own target, so a polygon need not be equilateral. Group them by
        polygon with ``packing.shapeId`` if you need per-polygon numbers."""
        return self.packing.targetEdgeLength

    def getTargetPerimeters(self):
        """Per-POLYGON target perimeters, shape (numPolygons,).

        DERIVED: the sum of each polygon's edge targets, kept in step by
        ``Packing.syncTargetPerimeter``. Change the edge targets to change it."""
        return self.packing.targetPerimeter

    # UNVERIFIED(Cam)
    def getVertices(self, polygon = None):
        """The ACTUAL coordinates. ``(numVertices, 2)`` for the whole packing, or one polygon's own
        ``(n_i, 2)`` loop when ``polygon`` is given.

        ``packing.positions`` is stored FLAT, shape ``(2 numVertices,)``, because that is the layout
        the minimizers and the constraint Jacobians want -- a gradient is a vector, not a list of
        points. Reading geometry out of it therefore costs a reshape and a ``startIndices`` slice at
        every call site, which is why ``positions.reshape(-1, 2)`` appears a hundred times across the
        package. This is that expression, once.

        A VIEW, not a copy: writing to the result moves the packing. That is deliberate and matches
        how ``positions`` is already used throughout -- the minimizers and ``projectPositions`` all
        mutate it in place -- but it means a caller who wants a snapshot has to ``.copy()`` it.
        Invalidate nothing yourself: after writing, the cached forces and energy are stale, so go
        through the methods that clear them if you can.

        THE COUNT IS PER POLYGON. ``startIndices`` is the only thing that knows where each loop
        begins; there is no ``(numPolygons, n, 2)`` form, because nothing guarantees a uniform vertex
        count -- ``halveNumEdges`` alone can leave one polygon at 16 and another at 8. Ask for a
        polygon by index, or iterate ``range(getNumPolygons())``.

        The CONTAINER is a polygon like any other and is included, at ``packing.containerIndex``."""
        r = self.packing.positions.reshape(-1, 2)
        if polygon is None:
            return r
        starts = np.asarray(self.packing.startIndices, dtype = int)
        index = int(polygon)
        if not 0 <= index < self.packing.numPolygons:
            raise IndexError(f"polygon {index} is out of range; the packing has "
                             f"{self.packing.numPolygons}.")
        return r[starts[index] : starts[index + 1]]

    def getAreas(self):
        """Per-POLYGON ACTUAL areas, shape (numPolygons,) -- the counterpart of ``getTargetAreas``.

        SIGNED shoelace areas, so a container wall wound the other way reads negative (that is what
        ``setBoundaryConditions("fixed")`` relies on) and a folded polygon reads as the difference of
        its lobes. Compare against ``getTargetAreas`` to see how far the shapes sit from their targets;
        under ``setConstraints(area = True)`` the ratio should be 1 to ~1e-16."""
        return backboneArea(self.packing)

    def getEdgeLengths(self):
        """Per-EDGE ACTUAL lengths, shape (numVertices,) -- one per vertex, for the edge leaving it.
        The counterpart of ``getTargetEdgeLengths``; group by ``packing.shapeId`` for per-polygon
        numbers."""
        return backboneEdgeLengths(self.packing)

    def getPerimeters(self):
        """Per-POLYGON ACTUAL perimeters, shape (numPolygons,) -- the counterpart of
        ``getTargetPerimeters``, summed from the actual edge lengths."""
        return np.bincount(self.packing.shapeId, weights = self.getEdgeLengths(),
                           minlength = self.packing.numPolygons)

    def getShapeIndices(self):
        """Per-POLYGON ACTUAL shape index ``P / sqrt(A)``, shape (numPolygons,).

        The scale-free measure of how far a polygon is from regular: its floor is
        ``sqrt(4 n tan(pi/n))`` -- 4 for a square, ~3.72 for a hexagon -- reached only by the regular
        polygon. Worth watching during an anneal, since it says how far intermediate shapes have
        wandered from the shape being searched for. Meaningless for a container wall (negative area)."""
        areas = self.getAreas()
        return self.getPerimeters() / np.sqrt(np.abs(areas))

    # UNVERIFIED(Cam)
    def getShapeDistortions(self):
        """Per-POLYGON ``P / sqrt(A) / sqrt(4 n tan(pi/n)) - 1``, shape (numPolygons,), ``nan`` at a
        container wall.

        The shape index made comparable ACROSS vertex counts: zero exactly at the regular n-gon,
        positive for anything else, and reading as a relative distortion (0.05 = 5% more perimeter than
        the regular polygon of the same area). Since the floor depends on n, raw shape indices from a
        packing of mixed n cannot be compared or reduced to one number; these can.

        This is the quantity to watch alongside ``getSizePolydispersity``, and it answers a DIFFERENT
        question. Size polydispersity is the spread of the AREAS -- whether the polygons are the same
        SIZE as each other. This is whether each one is the right SHAPE. Under
        ``setConstraints(area = True, edge = [1, 2])`` the two are decoupled by construction: the areas
        are pinned per polygon while the edges are held only in their global moments, so every polygon
        can distort into a kite or a trapezoid with the size distribution perfectly monodisperse."""
        counts = np.diff(np.asarray(self.packing.startIndices, dtype = int)).astype(float)
        regular = np.sqrt(4.0 * counts * np.tan(np.pi / counts))
        distortions = self.getShapeIndices() / regular - 1.0
        container = getattr(self.packing, "containerIndex", None)
        if container is not None:
            distortions[int(container)] = np.nan
        return distortions

    # UNVERIFIED(Cam)
    def getShapeBudget(self):
        """The REALIZED total distortion ``sum_i d_i`` of the current geometry -- what a shape anneal
        walks to zero. Needs no constraints; it is a measurement."""
        return float(np.nansum(self.getShapeDistortions()))

    # UNVERIFIED(Cam)
    def setShapeBudget(self, budget, project = True):
        """Aim the shape constraint at a new total distortion and SHAKE the geometry onto it. Returns
        self.

        Requires ``setConstraints(..., shape = True)``: this moves a CONSTRAINT, so the retraction is
        what actually reshapes the polygons, and every other constraint is respected while it happens.
        Lowering it in small steps with a relaxation between is the shape anneal -- the continuous
        replacement for the one-shot rigid projection, which surrendered everything the anneal had
        gained (measured: overlap 2.59e-03 -> 9.21e-04 across the anneal, back to 2.91e-03 the instant
        the shapes were projected onto squares).

        The budget cannot be driven all the way to zero. ``d_i`` is minimized at the regular polygon,
        so the constraint row's gradient vanishes exactly where it is headed and the retraction loses
        its first-order direction -- ``constraintConditioning`` falls with it. Ramp down to a small
        budget and hand off to ``setConstraints(area = True, edge = True)`` for the last step."""
        if self.constraints is None or not getattr(self.constraints, "shape", False):
            raise ValueError("no shape constraint is active; call setConstraints(..., shape = True) "
                             "first.")
        self.constraints.setShapeBudget(float(budget))
        if project:
            self.constraints.projectPositions(self.packing)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getShapeDeficit(self):
        """Total ISOPERIMETRIC DEFICIT ``sum_i (P_i - sqrt(4 n_i tan(pi/n_i) A_i))``, measured from the
        geometry. Zero only when every polygon is regular.

        The deviation-form counterpart of ``getShapeBudget``, and the quantity the paired
        ``k = +1, -1`` moments hold. It is a LENGTH, not the dimensionless distortion, because that is
        what the moments act on."""
        counts = np.diff(np.asarray(self.packing.startIndices, dtype = int))
        g = np.sqrt(4.0 * counts * np.tan(np.pi / counts))
        deficit = self.getPerimeters() - g * np.sqrt(np.abs(self.getAreas()))
        container = getattr(self.packing, "containerIndex", None)
        stop = self.packing.numPolygons if container is None else int(container)
        return float(np.sum(deficit[:stop]))

    # UNVERIFIED(Cam)
    def setShapeDeficit(self, total, project = True):
        """Aim the paired shape-deviation moments at a new TOTAL deficit and SHAKE onto them.
        Returns self.

        Requires ``setConstraints(..., shape = [...], deviation = True)``. EVERY moment row moves
        together, SELF-SIMILARLY: the deviations are asked to scale by ``lambda = total / sum delta``,
        so each target follows exactly

            Phi_k -> lambda^k Phi_k

        for whatever exponents are active -- ``[1, -1]``, ``[1, 2, -1, 4]``, anything.

        THIS IS WHAT KEEPS THE TARGET SET FEASIBLE. Moments of a nonnegative sequence are not
        independent (Cauchy-Schwarz gives ``Phi_2 Phi_0 >= Phi_1^2``, AM-HM gives ``Phi_1 Phi_-1 >=
        P^2``, and so on up the Stieltjes conditions), so moving them independently eventually asks for
        a combination no nonnegative sequence can realize and the retraction grinds against an empty
        feasible set. A self-similar scaling can never do that: the scaled sequence realizes it by
        construction. It also preserves the distribution's SHAPE while shrinking its scale, which is
        what an anneal of the deviations should mean -- pinning the inverse sum while the mean fell
        would force the spread to widen, which is backwards.

        Unlike ``setShapeBudget`` this can be driven a long way down: the ``-1`` row's gradient carries
        ``-delta^-2`` and stiffens as the deviations shrink, where the direct budget's vanished
        (measured, row norms 2.6e+02 -> 6.4e+04 against 24.5 -> 3.96 over the same range). Measured on
        6 squares, ramping the deficit down by 1000x: every target hit exactly, no deviation reaching
        zero (smallest 2.5e-09), the worst distortion falling 2.5e-02 -> 1.8e-05, hard areas exact to
        7e-16 throughout."""
        if self.constraints is None or not getattr(self.constraints, "shape", False):
            raise ValueError("no shape constraint is active; call "
                             "setConstraints(..., shape = [1, -1], deviation = True) first.")
        distribution = getattr(self.constraints, "distribution", self.constraints)
        if not getattr(distribution, "deviation", False):
            raise ValueError("setShapeDeficit needs the DEVIATION form; the direct shape budget is "
                             "driven with setShapeBudget instead.")
        deviations = distribution.deviations(self.packing, "shape")
        current = float(np.sum(deviations))
        total = float(total)
        if current <= 0.0:
            raise ValueError("the deviations have already collapsed to zero; there is nothing left to "
                             "scale. Seed a spread (spreadShapes) before ramping.")
        factor = total / current
        moments = distribution.familyMoments("shape")
        # Scaled from the GEOMETRY's own moments, never from the stored reference. Scaling the
        # reference compounds drift: whenever a retraction lands slightly off, the next call multiplies
        # that error rather than correcting it, and the +1 target silently stops being ``total`` at all.
        # Measured, a six-exponent ramp of five steps ended with its deficit 6.2x from what was asked
        # while reporting a 1.6e-12 residual -- entirely truthful, because the geometry really was on
        # the drifted reference it had been given. Anchoring here makes the +1 target exactly ``total``
        # by construction, so a miss is a retraction failure and shows up as residual.
        distribution.setReference("shape", [float(np.sum(deviations ** k)) * factor ** k
                                            for k in moments])
        if project:
            self.constraints.projectPositions(self.packing)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getMaxShapeDistortion(self):
        """The worst per-polygon shape distortion, container excluded -- one number saying how far the
        packing is from being made of the regular polygons it is supposed to be made of."""
        distortions = self.getShapeDistortions()
        return float(np.nanmax(distortions)) if distortions.size else 0.0

    # UNVERIFIED(Cam)
    def bisectJamming(self, low = None, rounds = 20, tolerance = 1e-6, maxUnbalancedForce = 1e-8,
                      maxSteps = 20000, minimizer = "lbfgs", progressBar = False, **kwargs):
        """Binary search for the jamming density on the exact overlap verdict. Returns a
        ``SweepResult``.

        The fast alternative to ``energySweep``'s density ladder: it brackets by halving the gap
        downward, then bisects, so it costs the logarithm of the ladder's step count. It TELEPORTS in
        density, though, and this landscape is glassy -- so prefer the ladder while the density steps
        are still carrying an anneal, and use this once the shapes are fixed and the only question is
        how far the arrangement will compress."""
        return anneal.bisectJamming(
            self, low = low, rounds = rounds, tolerance = tolerance,
            maxUnbalancedForce = maxUnbalancedForce, maxSteps = maxSteps, minimizer = minimizer,
            progressBar = progressBar, **kwargs)

    # UNVERIFIED(Cam)
    def compressToJamming(self, pressure = 1e-3, finalPressure = 1e-9, pressureRounds = 8,
                          maxUnbalancedForce = 1e-8, maxSteps = 20000, minimizer = "lbfgs",
                          rigidify = True, progressBar = False, **boxKwargs):
        """Compress by giving the BOX a scale degree of freedom, under a pressure ramped to zero.

        The alternative to ``energySweep``: the polygons are never resized, the wall is, so the
        configuration rearranges while the box closes instead of being teleported to each new phi and
        re-relaxed. Density is an OUTPUT, not a control parameter -- there is no phi ladder and no
        bisection. Needs a container. Returns a ``SweepResult``.

        Use it INSTEAD of ``energySweep`` when the question is "how small a box holds these shapes",
        and alongside it when a sweep looks basin-trapped: this removes the per-step kick, though it
        does not by itself escape a branch -- the shape and size anneals are what do that."""
        return anneal.compressToJamming(
            self, pressure = pressure, finalPressure = finalPressure,
            pressureRounds = pressureRounds, maxUnbalancedForce = maxUnbalancedForce,
            maxSteps = maxSteps, minimizer = minimizer, rigidify = rigidify,
            progressBar = progressBar, **boxKwargs)

    def energySweep(self, finalPolydispersity = None, finalEnergy = 0.0, finalSigma = None,
                    annealRounds = 10, phiStep = 0.004, refineRounds = 10, minPhi = 0.2,
                    maxUnbalancedForce = 1e-8, maxSteps = 20000, progressBar = False,
                    innerProgressBar = False, verbose = False, drawEvery = None,
                    finishRigid = True, annealShape = True, finalDistortion = 1e-6,
                    shapeRounds = 12, sharpDecompress = None, maxSigmaRatio = 2.0,
                    wallTolerance = None, compressStep = None, compressRounds = 6,
                    minimizer = "cg", maxShapeRatio = 2.0, excessEnergy = None,
                    excessTolerance = 0.05, maxDensityStep = 1.01):
        """Anneal the shape distribution and the contact sharpness, then DECOMPRESS until the packing
        is valid; returns a ``SweepResult`` and leaves the model in the packed configuration.

        ``excessEnergy`` replaces the fixed density of the anneal phase with a fixed dimensionless
        CONTACT ENERGY (``getExcessEnergy``), re-established after every round. Prefer it whenever the
        jamming density is not already known: it is what makes "start above jamming" a requirement the
        protocol satisfies rather than one the caller has to guess, and it tracks the jamming density
        as the anneal moves it. See ``holdExcessEnergy``.

        Start ABOVE the jamming density -- where the shapes cannot avoid overlapping -- and this lowers
        the density until they can pack. The density it reaches is the answer: for 5 unit squares the
        optimum is 5/2.7071^2 = 0.68227, so starting at 5/2.7^2 = 0.68587 asks how close to optimal the
        protocol gets. ``result.phi`` is that density, ``result.history`` every step.

        THE VERDICT IS TWO TESTS, because only one of the two quantities is exact. ``finalEnergy``
        (default 0.0) is the tolerance on POLYGON-POLYGON overlap area, which really is a sign change:
        measured across a sweep through jamming it reads identically 0.000000e+00 at every valid
        density. ``wallTolerance`` (default 1e-4 of the mean edge) is the containment tolerance, and it
        is a penetration DEPTH.

        Containment cannot be tested against zero. What survives a long relaxation is a corner just
        clipping the wall, whose area goes as delta^2 and whose restoring force goes as delta^3
        (measured slopes 1.978 and 2.966). The minimizer stops when that force sinks into the ~3e-12
        force noise, so each factor of ten in delta would cost a factor of a thousand in precision. The
        old 1e-12 tolerance on the TOTAL overlap therefore rejected every jammed state and decompressed
        until nothing touched: it returned 0.665692 with max|F| = 0.0 exactly, where the packing is in
        fact valid to at least 0.673692.

        A depth beats an area because the two differ by a square -- 3.56e-10 of outside area is
        1.88e-05 of depth, 7.5e-05 of an edge. The mollified energy cannot serve as either: it does not
        vanish on a valid packing and rises as one is compressed.

        ``finalSigma`` defaults to the dynamics floor (0.01 of the mean edge). Below that the ~1/sigma
        contact force makes FIRE and CG diverge, so a smaller request is clamped with a warning; the
        verdict does not need it, having no sigma in it at all.

        Needs a moment mechanism for the polydispersity ramp -- ``setDOFType("transient")`` or
        ``setConstraints(edge = [1, 2])``. Without one the width is left alone and only sigma and the
        density are annealed. See ``anneal.py`` for the three phases.

        ``progressBar`` shows ONE bar for the whole sweep, labelled with the phase, the current density
        and the residual overlap -- a sweep runs dozens of minimizations, so handing the flag to each of
        them would print dozens of bars that each vanish at once. Pass ``innerProgressBar = True`` as
        well if you do want the individual relaxations' bars, and ``verbose = True`` for a printed line
        per step instead.

        ``drawEvery = n`` draws the packing every n steps (and always at the end), titled with the
        phase, density and residual overlap. Worth using on a long run: the failure modes here are
        GEOMETRIC -- a polygon folding, shapes drifting away from square, the arrangement coming
        apart -- and none of those show up in a scalar residual.

        ``finishRigid`` (on by default) freezes the annealed shapes into RIGID REGULAR polygons before
        any density verdict is taken. Without it the answer is not a packing of squares: constraining
        the area alone constrains SIZE, not shape, and a fixed-area quadrilateral is any quadrilateral,
        so the shapes drift under the anneal and the reported density describes whatever they became.
        The handoff sets each edge target to the regular value for that polygon's area and relaxes onto
        it, which with the area fixed forces an actual square for n = 4.

        ``annealShape`` (on by default) makes that handoff CONTINUOUS. The shapes are held by their
        total distortion ``sum_i d_i`` (``setShapeBudget``), and the budget is walked down alongside the
        density over ``shapeRounds`` decompression steps, ending rigid at ``finalDistortion``. The
        one-shot projection it replaces gave back everything the anneal had gained -- measured, overlap
        went 2.59e-03 -> 9.21e-04 across the anneal and straight back to 2.91e-03 when the shapes were
        projected. Set it False to get that behavior back for comparison.

        ``sharpDecompress`` (on by default) turns the mollification OFF before decompressing, so the
        density is decided by the same exact energy that judges it. The Plummer contact does not vanish
        on a valid packing (8.2e-04 measured where the true overlap was zero), so leaving it on keeps
        pushing separated polygons apart and settles at a looser density than the shapes admit.
        ``maxSigmaRatio`` caps how fast sigma may fall per anneal round, lengthening the anneal rather
        than letting the contact stiffness outrun the relaxation.

        ``minimizer`` is what relaxes between schedule steps: ``"cg"`` (default), ``"lbfgs"``, or
        ``"fire"`` for the historical FIRE-then-CG-polish pair. On the contact tiers L-BFGS is the
        fastest of the three by a wide margin -- 13.6x over CG on the depth tier at N = 32, reaching a
        lower energy -- because it accepts the unit step and so spends ~1.2 force evaluations per step
        against CG's ~20. The transient and sharp tiers still use FIRE regardless; the reasons are in
        ``anneal._relax`` and are not about speed."""
        return anneal.energySweep(
            self, finalPolydispersity = finalPolydispersity, finalEnergy = finalEnergy,
            finalSigma = finalSigma, annealRounds = annealRounds, phiStep = phiStep,
            refineRounds = refineRounds, minPhi = minPhi,
            maxUnbalancedForce = maxUnbalancedForce, maxSteps = maxSteps,
            progressBar = progressBar, innerProgressBar = innerProgressBar, verbose = verbose,
            drawEvery = drawEvery, finishRigid = finishRigid, annealShape = annealShape,
            finalDistortion = finalDistortion, shapeRounds = shapeRounds,
            sharpDecompress = sharpDecompress, maxSigmaRatio = maxSigmaRatio,
            wallTolerance = wallTolerance, compressStep = compressStep,
            compressRounds = compressRounds, minimizer = minimizer,
            maxShapeRatio = maxShapeRatio, excessEnergy = excessEnergy,
            excessTolerance = excessTolerance, maxDensityStep = maxDensityStep)

    def getOverlapArea(self):
        """Total EXACT violation area: polygon-polygon overlap plus any area lying outside a container.

        Zero if and only if the configuration is a valid packing, and it does not depend on ``sigma`` at
        all -- so "has this packed?" is a SIGN CHANGE rather than a threshold anyone has to choose.

        This is the number to test, not the relaxation energy. The mollified energy does not vanish on
        a valid packing: measured 8.2e-04 for 8 squares whose true overlap is exactly zero, and it RISES
        as a perfectly valid packing is compressed, because its kernel has a tail between merely
        touching faces. It reports proximity; this reports overlap.

        Verified against two independent constructions -- Sutherland-Hodgman convex clipping and Monte
        Carlo coverage -- agreeing on zero and nonzero across six configurations."""
        return self.getPairOverlapArea() + self.getContainerOverlapArea()

    # UNVERIFIED(Cam)
    def getPairOverlapArea(self):
        """POLYGON-POLYGON overlap area only, with any container excluded.

        Worth having on its own because it is the part of the verdict that is EXACT. Measured across a
        density sweep through jamming it reads identically ``0.000000e+00`` at every valid density --
        the sign change is perfect here, with no floor to calibrate against. The residual that does
        survive a long relaxation lives entirely in the container term, and it is a CORNER contact
        whose force vanishes as delta^3 (measured slopes: area 1.978, energy 3.955, force 2.966 against
        the predicted 2, 4, 3). Mixing the two into one number hides an exact test behind an inexact
        one."""
        if self.getGeometryType() == "round":
            if self._exactArcs():
                return float(self._exactMeasurements()[2])
            with self.measuredGeometry():
                return self.getPairOverlapArea()
        self._refreshNeighbors()
        overlapAreaEnergyForce(self.packing, kOverlap = 1.0)
        pairs = self.packing.pairOverlapArea
        container = getattr(self.packing, "containerIndex", None)
        total = 0.0
        for (A, B), overlap in pairs.items():
            if container is None or (A != container and B != container):
                total += overlap
        return float(total)

    # UNVERIFIED(Cam)
    def getContainerOverlapArea(self):
        """Total area lying OUTSIDE the container, zero when there is none."""
        if self.getGeometryType() == "round":
            if self._exactArcs():
                return float(self._exactMeasurements()[3])
            with self.measuredGeometry():
                return self.getContainerOverlapArea()
        self._refreshNeighbors()
        overlapAreaEnergyForce(self.packing, kOverlap = 1.0)
        pairs = self.packing.pairOverlapArea
        container = getattr(self.packing, "containerIndex", None)
        total = 0.0
        if container is not None:
            # A CLOCKWISE wall has its interior reversed, so the pair overlap already reports the part
            # of the shape OUTSIDE it; counter-clockwise it reports the part inside and the complement
            # is taken. See energies.sharpContainerEnergyForce.
            sign = containerOrientationSign(self.packing, container)
            areas = self.getAreas()
            for shape in range(int(container)):
                cap = pairs.get((min(shape, container), max(shape, container)), 0.0)
                total += cap if sign > 0.0 else areas[shape] - cap
        return float(total)

    # UNVERIFIED(Cam)
    def getWallPenetration(self):
        """How far the packing sticks out of the container, as a DISTANCE. Zero when nothing does.

        The honest unit for a containment tolerance, and the reason to prefer it over the outside AREA
        is that the two are related by a square: the measured resting state carried 3.56e-10 of area,
        which sounds negligible, and 1.88e-05 of depth, which is 7.5e-05 of an edge. An area tolerance
        of 1e-09 grants 3.1e-05 of depth -- four orders larger than it looks.

        It also sidesteps a precision fight. That residual is not roundoff: the contact is a CORNER, so
        the restoring force vanishes as delta^3 (measured slope 2.966), and the minimizer stops when
        that force sinks into the ~3e-12 force noise rather than when the geometry is clean. Resolving
        it by relaxing harder needs the noise floor down by the CUBE of the improvement wanted. Measured
        geometry has no such problem.

        Computed as the largest distance from the container's boundary to any polygon vertex lying
        outside it. That is the true maximum: the outside region's extreme point must be one of its
        corners, and those are polygon vertices, wall vertices or crossings -- the last two sit ON the
        boundary at distance zero, so only polygon vertices can be deepest."""
        if self.getGeometryType() == "round":
            if self._exactArcs():
                import roundedContact
                return float(roundedContact.wallPenetration(self.packing, self.rho))
            with self.measuredGeometry():
                return self.getWallPenetration()
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            return 0.0
        r = self.packing.positions.reshape(-1, 2)
        a, b = int(self.packing.startIndices[container]), int(self.packing.startIndices[container + 1])
        wall = r[a:b]
        stop = int(self.packing.startIndices[container])
        worst = 0.0
        for vertex in r[:stop]:
            if pointInPolygon(vertex, wall):
                continue
            worst = max(worst, _distanceToLoop(vertex, wall))
        return float(worst)

    def getSharpEnergy(self):
        """Exact (unmollified) contact energy of the current configuration, container included.

        The same normalized-squared functional the mollified tier uses, so this is genuinely its
        sigma -> 0 limit (verified: the mollified energy converges to it as 0.938, 0.989, 0.9986,
        0.9998 for sigma = 3e-2 ... 1e-3). Leaves the model's cached force/energy and model type
        untouched, so it can be read mid-relaxation."""
        self._refreshNeighbors()
        energy, _ = sharpOverlapEnergyForce(self.packing, kOverlap = 1.0)
        container = getattr(self.packing, "containerIndex", None)
        if container is not None:
            wall, _ = containerEnergyForce(self.packing, self.sigma, mollified = False)
            energy += wall
        return float(energy)

    # UNVERIFIED(Cam)
    def getContactEnergy(self):
        """The CONTACT term of the energy ALONE -- overlap or penetration plus the wall, with the shape
        springs and the self-repulsion left out. Computed on whichever tier is active.

        This is the energy a jamming criterion has to be written against, and ``getEnergy`` cannot
        serve. Only a CONSTRAINED term's spring is dropped, so under
        ``setConstraints(area = True, perimeter = True)`` the edge springs are still live and the total
        carries a shape penalty that says nothing about whether anything is touching -- and, worse, does
        not vanish below jamming, which is exactly the sign change a density controller needs.

        Leaves the cached force/energy and the model type untouched, so it can be read mid-relaxation."""
        if self.getGeometryType() == "round":
            if self._exactArcs():
                if self.modelType == "area":
                    import roundedContact
                    energy, _, _ = roundedContact.packingAreaEnergyForce(
                        self.packing, self.rho, kOverlap = 1.0, kContainer = self.kContainer)
                    return float(energy)
                pair, wall, _, _ = self._exactMeasurements()
                return float(pair + wall)
            with self.measuredGeometry():
                return self.getContactEnergy()
        self._refreshNeighbors()
        container = getattr(self.packing, "containerIndex", None)
        if self.modelType == "depth":
            import polyContactSystem
            energy, _ = polyContactSystem.packingEnergyForce(
                self.packing, self.depthStiffness, wallStiffness = self.depthWallStiffness)
            return float(energy)
        if self.modelType == "softDepth":
            if self.softEpsilon is None:
                raise ValueError("soft depth unconfigured -- call setSoftDepth(...) first")
            import softDepth
            energy, _ = softDepth.packingEnergyForce(
                self.packing, self.softEpsilon, self.softStiffness, self.adhesionWork,
                self.adhesionRange, self.kContainer, getattr(self, "quadratureOrder", 16))
            return float(energy)
        if self.modelType == "mollified":
            self._requireSigma()
            energy, _ = plummerOverlapExact(self.packing, self.sigma, numActive = container)
        else:
            energy, _ = sharpOverlapEnergyForce(self.packing, kOverlap = 1.0)
        if container is not None:
            wall, _ = containerEnergyForce(self.packing, self.sigma, kContainer = self.kContainer,
                                           mollified = (self.modelType == "mollified"))
            energy = energy + wall
        return float(energy)

    # UNVERIFIED(Cam)
    def getPairContactEnergy(self):
        """POLYGON-POLYGON contact energy only, with the container's confinement term removed.

        The same split as ``getOverlapArea`` / ``getPairOverlapArea``, and for a sharper reason: these
        two terms are not merely different, they are ALTERNATIVES the packing chooses between. A
        confined packing under stress can relieve it by bearing on its neighbors or by extruding
        through the wall, and whichever is softer wins. Measured on the depth tier, where the wall
        carries the same stiffness as a body contact, escape wins outright -- a state held at a total
        excess of 1.04e-06 had 100.00% of that energy in wall penetration, 1.83e-19 in pair contact,
        and a pair overlap area of EXACTLY 0.000e+00. Nothing was touching anything: 17 vertices were
        simply outside the box. A jamming criterion written on the TOTAL accepts that as jammed.

        On the depth tier the wall term is subtracted using ``confinementEnergyGradient``, the slow
        per-body reference -- it agrees with the batched path to 1e-19, and its ~77 ms is charged once
        per controller round rather than once per force evaluation."""
        if self.getGeometryType() == "round":
            if self._exactArcs():
                if self.modelType != "depth":
                    # The area tier's functional is normalized-SQUARED per pair, so its container term
                    # does not subtract out of a total the way the depth tier's does. Said rather than
                    # silently answered with the chorded shape, which is the failure this branch exists
                    # to prevent.
                    raise NotImplementedError(
                        "getPairContactEnergy on exact arcs is wired to the 'depth' tier only; the "
                        "area tier's normalized-squared form has no total to subtract the wall from. "
                        "Use getPairOverlapArea(), which is exact on both.")
                return float(self._exactMeasurements()[0])
            with self.measuredGeometry():
                return self.getPairContactEnergy()
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            return self.getContactEnergy()
        if self.modelType == "depth":
            import polyContactSystem
            # At the EFFECTIVE wall stiffness, since getContactEnergy now includes the surplus. The law
            # is linear in the stiffness, so one call at k*wallStiffness is the whole wall term.
            wall, _ = polyContactSystem.confinementEnergyGradient(
                self.packing, self.depthStiffness * self.depthWallStiffness)
            return self.getContactEnergy() - float(wall)
        if self.modelType == "softDepth":
            # The wall is folded inside softDepth's own kernel and cannot be pulled back out here. That
            # tier is superseded by ``depth``; this is stated rather than silently approximated.
            raise NotImplementedError(
                "getPairContactEnergy has no split for the softDepth tier -- its container term is "
                "computed inside softDepth.packingEnergyForce. Use setModelType('depth').")
        self._refreshNeighbors()
        if self.modelType == "mollified":
            self._requireSigma()
            energy, _ = plummerOverlapExact(self.packing, self.sigma, numActive = container)
        else:
            energy, _ = sharpOverlapEnergyForce(self.packing, kOverlap = 1.0)
        return float(energy)

    # UNVERIFIED(Cam)
    def getExcessEnergy(self):
        """``getPairContactEnergy`` in units of ONE polygon indented by a whole edge length -- a
        dimensionless distance ABOVE jamming.

        POLYGON-POLYGON ONLY, deliberately; ``getPairContactEnergy`` carries the measurement that
        forced it. Jamming is the bodies bearing on each other, and containment is a separate question
        that ``energySweep`` already judges separately as a penetration DEPTH against ``wallTolerance``.
        Rolling the wall into this number lets a packing satisfy it by leaking out of its container
        while nothing inside touches at all.

        Exactly 0.0 at and below jamming (the depth and sharp tiers return zero when nothing touches),
        rising steeply above it, and with the N, the polygon size and the contact stiffness divided out
        -- so the same number means the same amount of overlap in any run. See
        ``holdExcessEnergy``, which holds it fixed in place of the density."""
        return self.getPairContactEnergy() / anneal.energyScale(self)

    # UNVERIFIED(Cam)
    def holdExcessEnergy(self, excess, tolerance = 0.05, maxDensityStep = 1.01, maxRounds = 80,
                         maxUnbalancedForce = 1e-8, maxSteps = 20000, minimizer = "lbfgs",
                         verbose = False, **kwargs):
        """Move the density until the RELAXED contact energy sits a fixed excess above jamming.
        Returns ``(excess, phi)`` as achieved.

        The two-sided alternative to naming a density: it COMPRESSES a loose packing and DECOMPRESSES
        an overjammed one, so a run no longer has to know its own jamming density in advance."""
        return anneal.holdExcessEnergy(
            self, excess, tolerance = tolerance, maxDensityStep = maxDensityStep,
            maxRounds = maxRounds, maxUnbalancedForce = maxUnbalancedForce, maxSteps = maxSteps,
            minimizer = minimizer, verbose = verbose, **kwargs)

    def getPackingFraction(self):
        """Total polygon area divided by the available area -- the packing fraction phi, measured from
        the ACTUAL geometry rather than from the targets.

        With a container the available area is the wall's own (its signed area is negative, so the
        magnitude is taken); otherwise it is the periodic cell, which is calibrated to area 1. Overlap
        is not subtracted, so above jamming this is the nominal phi the protocol asked for, which is the
        control parameter a sweep wants."""
        areas = self.getAreas()
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            return float(areas.sum())
        return float(areas[:int(container)].sum() / abs(areas[int(container)]))

    # UNVERIFIED(Cam)
    def checkDensityFeasible(self, warn = True):
        """Compare the phi the AREA TARGETS ask for against the published ceiling for that many
        squares. Returns ``(asked, ceiling)``, with ``ceiling`` None when the count is not listed.

        Asking for more than the ceiling is not a hard search, it is impossible -- and from inside a run
        it does not look impossible. The constraint retraction simply stops converging, which reads as a
        budget problem and invites a bigger iteration count that cannot help; downstream the geometry
        eventually diverges and the traceback lands on a NaN in the energy tier, three layers from the
        cause. A cascade burned an hour that way with targets summing to phi 0.971 against a ceiling of
        0.500, so this exists to say so in one line at the point the targets are SET.

        Measured from the targets, not the geometry: the geometry is what is being driven toward them,
        so it is the targets that are either reachable or not."""
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            return None, None
        index = int(container)
        targets = np.asarray(self.packing.targetArea, dtype = float)
        asked = float(targets[:index].sum()) / float(abs(targets[index]))
        ceiling = records.maximumDensity(index)
        if warn and ceiling is not None and asked > ceiling:
            warnings.warn(
                f"\n*** AREA TARGETS ASK FOR MORE THAN CAN EXIST ***\n"
                f"    {index} polygons ask for phi {asked:.6f}; the "
                f"{'PROVED optimum' if records.isProved(index) else 'best known packing'} for that "
                f"many squares allows {ceiling:.6f}.\n"
                f"    No arrangement satisfies these targets, so the constraint retraction cannot "
                f"converge -- a larger maxIter or maxSteps will not help, and the eventual failure "
                f"will surface far from here as a non-finite force or Jacobian.\n"
                f"    Lower the density before minimizing.")
        return asked, ceiling

    # UNVERIFIED(Cam)
    def getBoxArea(self):
        """Area enclosed by the container wall. Raises without one."""
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            raise ValueError("there is no container; getBoxArea needs setBoundaryConditions('fixed') "
                             "and an addShape wall.")
        return float(abs(self.getAreas()[int(container)]))

    # UNVERIFIED(Cam)
    def scaleBox(self, factor):
        """Scale the CONTAINER about its own centroid by ``factor``, leaving every polygon alone.
        Returns self.

        The mirror image of ``setPackingFraction``, which resizes the polygons and never touches the
        box. Both change phi; which one moves is a modelling choice, not a detail. Resizing the box is
        what "smallest square containing N squares" actually asks, and it keeps the polygons' own
        geometry -- and therefore their hard area constraints -- untouched, so no SHAKE repair is
        needed afterwards."""
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            raise ValueError("there is no container to scale.")
        factor = float(factor)
        if factor <= 0.0:
            raise ValueError(f"box scale must be positive, got {factor}")
        r = self.packing.positions.reshape(-1, 2)
        a = int(self.packing.startIndices[int(container)])
        b = int(self.packing.startIndices[int(container) + 1])
        centroid = r[a : b].mean(axis = 0)
        r[a : b] = centroid + factor * (r[a : b] - centroid)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def boxScaleGradient(self, pressure = 0.0):
        """``(enthalpy, dH/dlambda)`` for a uniform scaling of the container about its centroid.

        The box carries ONE degree of freedom here, its scale, and the packing is driven by the
        enthalpy

            H = E_contact + p A_box,

        so shrinking the box is opposed by contact and driven by ``p``. At ``dH/dlambda = 0`` the wall
        pressure balances the packing's resistance; ramping ``p`` down to zero walks that balance to
        the jamming point. With ``lambda`` measured about the CURRENT geometry (so lambda = 1 here),

            dH/dlambda = sum_i g_i . (v_i - c)  +  2 p A_box,

        the first term being a virial over the wall vertices alone. That sum is exactly what the pinned
        container has been throwing away: the reaction the wall carries.

        THE CONTACT LAW HAS NO VALIDITY LIMIT AT A CONVEX WALL, which is what makes pressure control
        safe here. Body-body contact is capped by dMax/rIn << 1 -- past the medial axis the repulsion
        reverses -- but the exterior of a convex region has no medial axis, so the box may be pressed
        arbitrarily hard without leaving the formulation."""
        container = getattr(self.packing, "containerIndex", None)
        if container is None:
            raise ValueError("there is no container; boxScaleGradient needs one.")
        energy, force = self._forceEnergy(self.packing)
        gradient = -np.asarray(force, dtype = float).reshape(-1, 2)
        a = int(self.packing.startIndices[int(container)])
        b = int(self.packing.startIndices[int(container) + 1])
        r = self.packing.positions.reshape(-1, 2)
        centroid = r[a : b].mean(axis = 0)
        virial = float(np.sum(gradient[a : b] * (r[a : b] - centroid)))
        area = self.getBoxArea()
        return float(energy) + float(pressure) * area, virial + 2.0 * float(pressure) * area

    def setPackingFraction(self, phi):
        """Scale every polygon about its own centroid so the packing fraction becomes ``phi``. Returns
        self.

        Geometry AND targets move together by the same factor, so a hard area constraint stays exactly
        satisfied and no SHAKE step is needed to repair it -- which is what makes this usable as the
        compression step of a sweep. The container is never scaled: phi changes by resizing the
        polygons, not the box, so the box stays the fixed reference the wall energy is written against.

        Note the polygons keep their relative sizes and positions of their centroids, so this walks
        along ONE branch rather than rebuilding a fresh random configuration at each phi -- the
        difference between following a jammed state up in density and sampling a new basin every time."""
        phi = float(phi)
        if phi <= 0.0:
            raise ValueError(f"packing fraction must be positive, got {phi}")
        factor = np.sqrt(phi / self.getPackingFraction())
        stop = self._nonContainer()
        upTo = int(self.packing.startIndices[stop])
        r = self.packing.positions.reshape(-1, 2)
        for polygon in range(stop):
            a = int(self.packing.startIndices[polygon])
            b = int(self.packing.startIndices[polygon + 1])
            centroid = r[a : b].mean(axis = 0)
            r[a : b] = centroid + factor * (r[a : b] - centroid)
        self.packing.targetArea[:stop] *= factor ** 2
        self.packing.targetEdgeLength[:upTo] *= factor
        # The diagonal targets are lengths and scale with the edges. Omitting them leaves a template
        # asking for the new size's edges and the old size's turning angles, which no polygon satisfies:
        # the retraction then fights an infeasible set, and under a density controller that repeats
        # every step it inverts polygons outright (measured: phi ran away to -62 and the shape index to
        # 14.4 over twelve steps). This is the same coupling ``build._rescaleToAreas`` maintains.
        if self.packing.targetDiagonal is not None:
            self.packing.targetDiagonal[:upTo] *= factor
        self.packing.syncTargetPerimeter()
        # The MOMENT constraints have to come too, for exactly the same reason as the diagonals. They
        # hold absolute sums (sum l, sum l^2 under edge = [1, 2]), so a pure size change moves them
        # while leaving the distribution's shape untouched -- and the retraction then fights the
        # density move rather than merely riding it. Measured before this line existed: a x1.10 step
        # left residual 4.06e-01 with the retraction not converging, and pulled phi back 0.117%.
        distribution = self._distributionConstraints()
        if distribution is not None:
            distribution.rescale(factor)
        self._forces = None
        self._energy = None
        return self

    def getNumVertices(self):
        """Total number of vertices in the packing, across all polygons."""
        return int(self.packing.numVertices)

    def getNumPolygons(self):
        """Number of polygons in the packing, including any container."""
        return int(self.packing.numPolygons)

    def setBoundaryConditions(self, mode = "periodic"):
        """Select how the packing is bounded. Returns self.

        ``"periodic"`` (the default) -- the unit square with wrap-around: polygons interact with the
        3x3 neighborhood of periodic images and a shape leaving one side re-enters the opposite one.

        ``"free"`` -- no boundary at all. There are no images and no wrapping, so coordinates may run
        anywhere in the plane and a packing under no other confinement will simply expand.

        ``"fixed"`` -- a rigid WALL confining the packing. The LAST polygon in the packing is taken
        to be the wall, so add it with ``addShape(...)`` (and normally pin it) first. Selecting
        ``"fixed"`` ALWAYS switches to free space: a wall is meaningless under periodicity, because
        its own periodic images tile the plane, leaving no outside for it to keep shapes out of and
        making the confinement identically zero.

        The wall is held out of the ordinary pairwise overlap and handled by
        ``energies.containerEnergyForce``, which penalises the area of each shape lying OUTSIDE it.
        Its winding does not matter -- the sign is read from the geometry.
        """
        if mode == "periodic":
            self.packing.box = Box(PackingType.square)
            self.packing.containerIndex = None
        elif mode == "free":
            self.packing.box = None
            self.packing.containerIndex = None
        elif mode == "fixed":
            if self.packing.numPolygons < 2:
                raise ValueError(
                    "'fixed' needs a wall polygon: add one with addShape(...) first. The LAST "
                    "polygon in the packing is taken to be the wall.")
            # Free space is a PREREQUISITE, not a side effect, so it is set unconditionally.
            self.packing.box = None
            index = self.packing.numPolygons - 1
            self.packing.containerIndex = index
            # The wall's shape targets must match its ACTUAL geometry, signed. A clockwise wall has a
            # NEGATIVE shoelace area, so an unsigned target leaves the area spring with a residual of
            # -2 A and a constant energy of 2 (for a unit wall) at every packing fraction -- which
            # silently floors the total energy and makes "does this configuration pack?" unanswerable.
            r = self.packing.positions.reshape(-1, 2)
            a, b = int(self.packing.startIndices[index]), int(self.packing.startIndices[index + 1])
            loop = r[a:b]
            self.packing.targetArea[index] = 0.5 * np.sum(
                loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
        else:
            raise ValueError(
                f"unknown boundary condition {mode!r}; use 'periodic', 'free' or 'fixed'")
        self.boundaryConditions = mode
        self._forces = None
        self._energy = None
        return self

    def getBoundaryConditions(self):
        """The current boundary mode: ``"periodic"``, ``"free"`` or ``"fixed"``."""
        return getattr(self, "boundaryConditions",
                       "periodic" if self.packing.box is not None else "free")

    def setDOFType(self, dofType = "fixed"):
        """Choose whether the per-polygon TARGETS are fixed inputs or degrees of freedom.

        ``"fixed"`` (the default) -- targetArea / targetPerimeter are what you set them to.

        ``"transient"`` -- they are minimized jointly with the vertex positions, a DOUBLE
        OPTIMIZATION (Arzash, Tah, Liu & Manning, Phys. Rev. Research 7, 013157). Free targets simply
        chase the realised shapes, which would collapse the shape energy, so a set of MOMENTS of the
        target distribution is held fixed -- see ``setMoments``, which you must call as well.

        The reference notes that convergence is far more robust if the POSITIONS are relaxed first
        and the target DOF switched on afterwards, so call this after an initial minimize.
        Returns self."""
        if dofType not in ("fixed", "transient"):
            raise ValueError(f"unknown DOF type {dofType!r}; use 'fixed' or 'transient'")
        self.dofType = dofType
        self.transient = self._makeTransient() if dofType == "transient" else None
        return self

    def _makeTransient(self):
        """Build the transient target state, freeing only the families that actually have a DRIVE.

        A target family is a degree of freedom only if something pushes on it, and ``targetForces`` is
        built entirely from the spring residuals: the area term carries a factor of ``kArea`` and the
        edge term a factor of ``kEdge``. So a family is inert in either of two ways --

          RIGID     a hard constraint drives its residual to zero by construction (measured 3.2e-15
                    for area under ``setConstraints(area = True)``);
          NO SPRING its stiffness is zero, e.g. ``setSpringConstants(area = 0, edge = 1)``, so the
                    residual is multiplied by nothing.

        Either way the targets cannot move, yet their moments would still be re-projected and restored
        on every single step -- pure wasted work, and badly conditioned work at that. With
        ``setMoments([1, 2, -1, 4])`` on 5 polygons the area family is 4 constraints on 5 values,
        leaving one degree of freedom, and the restore overflowed outright.

        NB this reads the spring constants and constraints as they stand NOW, so call ``setDOFType``
        AFTER ``setSpringConstants`` and ``setConstraints``. Rebuilding it later would silently
        re-baseline the conserved moments to whatever the targets had drifted to."""
        constraints = self.constraints
        rigid = lambda term: constraints is not None and bool(getattr(constraints, term, False))
        areaFree = not rigid("area") and float(self.kArea) != 0.0
        edgeFree = not rigid("edge") and float(self.kEdge) != 0.0
        if not (areaFree or edgeFree):
            reason = []
            if rigid("area") or rigid("edge"):
                reason.append("held rigid by setConstraints")
            if float(self.kArea) == 0.0 or float(self.kEdge) == 0.0:
                reason.append("has a zero spring constant")
            raise ValueError(
                f"setDOFType('transient') has nothing to free: every target family is inert "
                f"({' and '.join(reason)}), so no force acts on the targets and they cannot move. "
                f"Release a constraint or give the corresponding spring a nonzero constant first.")
        return TransientTargets(self.packing, self.moments, area = areaFree, perimeter = edgeFree)

    def setMoments(self, moments = (1,)):
        """Moments of the target distribution to hold fixed while the targets relax.

        Each entry ``k`` pins ``sum_i t_i^k`` for every free target family. ``k = 1`` holds the mean,
        ``k = 2`` the spread, and a NEGATIVE ``k`` blows up as any target approaches zero, which is
        what stops the distribution degenerating -- the reference uses k in {-1,-2,-3,1,2,3}, so
        ``[1, 2, -1]`` is a typical choice. Only meaningful with ``setDOFType("transient")``.
        Returns self."""
        self.moments = [int(k) for k in np.atleast_1d(moments)]
        if self.dofType == "transient":
            self.transient = self._makeTransient()
        return self

    def setTargetPolydispersity(self, polydispersity):
        """Re-aim the conserved moments at the given coefficient of variation, keeping the mean.

        The annealing handle for a search: relax at a broad distribution, tighten toward monodisperse,
        re-relaxing at each step. Drives whichever moment mechanism is active -- the moment
        constraints from ``setConstraints(edge = [1, 2])``, or the transient target DOF from
        ``setDOFType("transient")``. Returns self.

        A REQUEST BELOW THE REACHABLE FLOOR IS CLAMPED, because it is not merely optimistic -- it is
        infeasible, and the retraction that chases it wrecks the packing rather than failing quietly.
        With the areas held rigid the edge lengths inherit their spread (a regular n-gon of area A has
        edge ``sqrt(4 A tan(pi/n) / n)``), so the edge CV cannot fall below the CV of ``sqrt(A0)``,
        roughly half the area CV. Measured when a notebook asked for exactly 0 against a floor of
        ~0.125: the moment retraction ran 796 passes without converging, reported "the targets are
        UNREACHABLE" at residual 8.5e-02, and left a polygon 2.57 EDGE LENGTHS outside its container --
        after which every density and energy downstream was meaningless. ``energySweep`` has clamped
        this for its own ramp since it was written; the setter did not, so any hand-written ramp walked
        straight into it."""
        floor = anneal._reachableWidth(self)
        if float(polydispersity) < floor * (1.0 - 1e-9):
            # ONCE PER MODEL. The text carries the requested value, so it would never de-duplicate on
            # its own -- and the caller that trips this is typically a ramp, which would print it once
            # per round with a different number each time and bury the one line that matters.
            if not getattr(self, "_warnedPolydispersityFloor", False):
                self._warnedPolydispersityFloor = True
                warnings.warn(
                    f"\n*** target polydispersity {float(polydispersity):.4g} is below the reachable "
                    f"floor {floor:.4g} ***\n"
                    f"    With the areas fixed, the edge lengths cannot be more equal than sqrt(A0) "
                    f"is. Asking for less does not tighten the distribution -- it forces the polygons "
                    f"to stop being regular, and the retraction chasing it can drive one clean out of "
                    f"the container. Clamping to the floor, here and for the rest of this model.\n"
                    f"    To end monodisperse, make the AREAS monodisperse (setMonoPerimeter or "
                    f"setSizePolydispersity) rather than squeezing the edge moments.", stacklevel = 2)
            polydispersity = floor
        distribution = getattr(self.constraints, "distribution", None)
        if distribution is None and isinstance(self.constraints, DistributionConstraints):
            distribution = self.constraints
        if distribution is None and self.transient is None:
            raise ValueError("setTargetPolydispersity needs a moment mechanism -- either "
                             "setConstraints with a list of exponents (e.g. edge = [1, 2]) or "
                             "setDOFType('transient').")
        if distribution is not None:
            distribution.retarget(self.packing, float(polydispersity))
            self.constraints.projectPositions(self.packing)
        if self.transient is not None:
            self.transient.retarget(self.packing, float(polydispersity))
            self.transient.restore(self.packing)
        self._forces = None
        self._energy = None
        return self

    def momentDrift(self):
        """Largest relative drift of any conserved moment (0 when targets are fixed)."""
        return 0.0 if self.transient is None else self.transient.momentDrift(self.packing)

    def setConstraints(self, area = True, perimeter = False, edge = True, shape = False,
                       deviation = False, distortion = False, diagonal = False,
                       equilateral = None, flatten = False, alternatingDiagonal = False):
        """Hold the named shape terms RIGID with hard constraints instead of the stiff springs, and
        relax on the resulting constraint manifold. Term names match ``setSpringConstants``; pass all
        three False to go back to a purely spring-held model.

        A CONSTRAINED TERM'S SPRING IS IGNORED: rigidity is the stiff-spring limit, so its penalty is
        dropped from the energy and any force it would contribute is projected out anyway. The spring
        constants are kept, so flipping a term back off restores the soft model at the same ``k`` --
        the soft-vs-rigid comparison is a one-flag change.

        The default (area + edge) is the hard form of the default active springs. ``perimeter = True``
        with ``edge = False`` is the loose alternative: the shape is free to mold against its
        neighbors with only its size and shape index pinned.

        Positions are projected onto the manifold immediately (SHAKE), so the configuration is valid
        and the forces drawn next are real. Under constraints the reported force is the TANGENTIAL
        force -- the true residual on the manifold; the normal part is carried by the constraints and
        is not an unbalanced force. Dropping the stiff spring modes from the dynamics is what lets
        FIRE take a much larger timestep on a far better conditioned problem. Returns self.

        A term may instead be given a LIST OF EXPONENTS, which holds only the global moments
        ``sum_i t_i^k`` of that quantity across the packing and leaves the individual shapes free to
        trade with one another:

            setConstraints(area = True, edge = [1, 2])

        pins every polygon's area exactly while holding only the mean and the variance of the edge
        lengths -- so a square may reshape into a same-area rectangle, but nothing can shrink. That is
        the annealing handle: relax with the edge distribution wide, then tighten it toward
        monodisperse with ``setTargetPolydispersity``.

        A moment constraint cannot impose monodispersity outright -- the mean and variance rows lose
        transversality in direct proportion to the width they are holding, so anneal close and then
        switch that term to ``True`` for the final relaxation; ``constraintConditioning`` says when.
        Only ``area`` and ``edge`` accept moments; ``perimeter`` is per-object only, being a sum of
        edges already.

        ``shape = True`` adds the SHAPE BUDGET: one row holding ``sum_i d_i``, the total relative
        distortion away from the regular n-gon (see ``getShapeDistortions``). It takes only True/False
        because it exists in one flavor -- a per-object shape-index constraint is what
        ``area = True, perimeter = True`` already is, and the budget's whole purpose is to let polygons
        TRADE distortion, one staying bent while another straightens. Walk it down with
        ``setShapeBudget``: the terms are nonnegative, so driving the sum to zero drives every polygon
        regular, which makes it the continuous replacement for a rigid handoff.

            setConstraints(area = True, edge = [1, 2], shape = True)

        holds every area exactly, the edge distribution's mean and variance, and the total distortion.

        ``deviation = True`` changes what the MOMENT families measure: the distance from the ideal
        rather than the quantity itself (``shape`` becomes the isoperimetric deficit
        ``P - sqrt(4 n tan(pi/n) A)``, ``area`` becomes the shrink-only ``A0 - A``, ``edge`` becomes
        ``|l - l0|``). These are nonnegative by construction, which is what makes a NEGATIVE exponent
        usable:

            setConstraints(area = True, shape = [1, -1], deviation = True)

        holds the mean deviation AND its inverse sum. The ``k = -1`` row is a barrier -- its gradient
        carries ``-delta^-2`` and so GROWS without bound as the deviation shrinks, exactly where the
        direct budget's gradient vanishes. Measured on 6 squares, shrinking the deficit 4.2e-02 ->
        8.5e-04: the ``k = +1`` row's norm fell 24.5 -> 3.96 (the degeneracy) while the ``k = -1``
        row's rose 2.6e+02 -> 6.4e+04. Note ``area`` deviations must be SEEDED before use with a
        negative exponent -- they start at exactly zero, which is the singular point.

        ``alternatingDiagonal = [1, 2]`` is ``diagonal = [1, 2]`` that CHOOSES ITS OWN VERTICES: the
        chords ``|v_2i - v_2i-2|``, exactly the set ``getAlternatingDiagonals`` measures and
        ``updateAlternatingDiagonals`` drives. The plain ``diagonal`` moments cover whatever
        ``packing.diagonalMask`` holds, and with no mask that is EVERY vertex, which drives the corners
        flat too and collapses the polygon. Since the mask is invisible in the row count -- a moment
        family is two rows whether it covers half the vertices or all of them -- the quiet failure was
        easy to reach; this spelling makes the alternation part of the call.

        THE PHASE IS THE SUBTLE PART. A flatness row is indexed by the vertex the diagonal is CENTRED
        on, so this marks the ODD local indices in order to constrain the chords joining the EVEN ones.
        Marking the even indices instead gives the complementary set, off by a single vertex, and
        constraining that while driving the other flattens everything.

        IT DIFFERS FROM ``selectFlattening`` IN WHICH PHASE IT PICKS. That routine chooses the phase
        that is already flattest, which is the path of least resistance and, more to the point, the
        same test ``halveNumEdges`` applies when deciding which set to drop -- so the two agree on the
        same vertices by construction. A fixed phase is predictable and reproducible instead, but it is
        NOT automatically the set the removal will want. Check with ``getFlatness`` before halving, or
        call ``selectFlattening`` and use ``diagonal`` if agreement with the removal matters more than
        knowing the indices.
        """
        if alternatingDiagonal is not False and alternatingDiagonal is not None:
            if diagonal is not False and diagonal is not None:
                raise ValueError(
                    "setConstraints(alternatingDiagonal = ...) conflicts with diagonal = ...: both "
                    "drive the same family, one choosing the alternating vertices for you and one "
                    "reading packing.diagonalMask. Pass exactly one.")
            if alternatingDiagonal is True:
                raise ValueError(
                    "setConstraints(alternatingDiagonal = True) is not a thing: True selects the "
                    "PER-OBJECT diagonal rows, which hold every diagonal LENGTH against "
                    "packing.targetDiagonal and do not read the alternating mask at all -- so it "
                    "would not be every other one. Pass exponents, e.g. alternatingDiagonal = [1, 2], "
                    "for the distribution; or flatten = True for one row per selected vertex.")
            self._maskAlternatingDiagonals()
            diagonal = alternatingDiagonal
        moments = {"area": area, "edge": edge, "diagonal": diagonal}
        hard = {name: value is True for name, value in moments.items()}
        listed = {name: value for name, value in moments.items()
                  if value is not None and value is not True and value is not False}
        for name, value in listed.items():
            if np.ndim(value) == 0:
                raise ValueError(
                    f"setConstraints({name} = {value!r}) is neither a flag nor a list of moment "
                    f"exponents. Pass True to pin every polygon's {name} individually, or a list "
                    f"like [1, 2] to hold only the distribution's mean and variance.")
        if perimeter is not True and perimeter not in (False, None):
            raise ValueError("setConstraints(perimeter = ...) takes only True/False: the perimeter is "
                             "already the sum of a polygon's edge targets, so constraining its "
                             "distribution is what edge = [1, 2] does.")
        if distortion is not False and distortion is not None and shape:
            raise ValueError(
                "setConstraints takes shape OR distortion, not both: they are the same family read "
                "two ways. 'distortion' is the DIMENSIONLESS d_i = P/(g sqrt(A)) - 1; 'shape' with "
                "deviation = True is the isoperimetric DEFICIT P - g sqrt(A), a length.")
        if (distortion is not False and distortion is not None) and edge is True:
            # PER-OBJECT EDGES LEAVE THE DISTORTION NO FREEDOM, so pinning both is contradictory rather
            # than merely redundant. For a quadrilateral it is exactly contradictory: equal edges at
            # fixed area is a rhombus with A = l^2 sin(theta), so both together force sin(theta) = 1 --
            # a square, d = 0 -- while the distortion rows hold d at whatever it currently is.
            # Measured on 11 squares seeded by spreadShapes: the retraction diverged inside
            # setConstraints itself, conditioning 2.0e-26, and the distortion it was meant to hold at
            # 0.26 came out at 3.5e+06. ``edge`` DEFAULTS to True, so this is reached by writing
            # nothing at all, which is why it raises instead of warning.
            raise ValueError(
                "setConstraints(distortion = [...]) conflicts with edge = True (the DEFAULT). Holding "
                "every edge at its own target already fixes each polygon's shape -- with the area "
                "fixed too, an equal-edged quadrilateral is a square -- so the distortion rows have "
                "nothing left to hold and the retraction diverges. Pass edge = False to let the shape "
                "move, or edge = [1, 2] to hold only the edge DISTRIBUTION.\n"
                "    Note spreadShapes moves the geometry and NOT the edge targets, so after seeding "
                "a shape spread the per-object edge targets are far from the geometry by design.")
        fullShapeMoments = False
        if distortion is not False and distortion is not None:
            # THE DIMENSIONLESS FORM IS THE RIGHT ONE WHEN THE AREAS ARE POLYDISPERSE. The deviation
            # family's deficit is a LENGTH, so a moment of it weights a big polygon more than a small
            # one at equal relative distortion, and a sum over a size-spread packing then measures size
            # as much as shape. d_i divides that out.
            shape = distortion
            fullShapeMoments = distortion is not True
        shapeExponents = None
        if shape is not True and shape not in (False, None):
            if not (deviation or fullShapeMoments):
                raise ValueError(
                    "setConstraints(shape = [...]) needs deviation = True. Without it the shape budget "
                    "is a single row by construction (sum_i d_i, d_i >= 0) with no exponents to "
                    "choose; the deviation form takes moments of the isoperimetric deficit, where a "
                    "list -- and in particular a negative exponent -- is the whole point. To take "
                    "moments of the DIMENSIONLESS distortion instead, pass distortion = [...].")
            shapeExponents = list(np.atleast_1d(shape))

        if diagonal is True and self.packing.targetDiagonal is None:
            raise ValueError(
                "setConstraints(diagonal = True) needs diagonal targets. Call setShapeTemplate() "
                "first -- it writes the skip-one diagonals of the template shape, which is what turns "
                "edge lengths into an actual SHAPE rather than a flexible linkage.")
        # EQUILATERAL AT A FIXED SHAPE INDEX, SIZE FREE. ``edge = True`` pins each edge to a stored
        # number and so fixes the size with it; this pins every edge to ``kappa sqrt(A) / n`` computed
        # from the polygon's OWN live area, so the shape is held while the size is left free for the
        # packing to trade. The two cannot both be on -- they hold the same quantity against different
        # targets, and together they simply re-pin the size.
        if equilateral is not None and hard["edge"]:
            raise ValueError(
                "setConstraints(equilateral = ...) conflicts with edge = True (the DEFAULT). Both "
                "hold the edge lengths: edge = True at their stored targets, which fixes each "
                "polygon's SIZE, and equilateral at kappa sqrt(A)/n from the polygon's own area, "
                "which does not. Pass edge = False to keep the size free, which is the point of "
                "equilateral, or drop equilateral to pin the sizes.")
        block = None
        if (hard["area"] or perimeter or hard["edge"] or hard["diagonal"]
                or equilateral is not None or flatten):
            block = ShapeConstraints(self.packing, area = hard["area"], perimeter = bool(perimeter),
                                     edge = hard["edge"], diagonal = hard["diagonal"],
                                     equilateral = equilateral, flatten = bool(flatten))
            reason = block.redundancyReason()
            if reason is not None:
                warnings.warn(f"\n*** redundant constraint set ***\n    {reason}; the dependent row "
                              f"is dropped automatically (harmless, but it is doing no work).",
                              stacklevel = 2)
            reason = block.infeasibleReason(self.packing)
            if reason is not None:
                raise ValueError(f"\n*** infeasible constraint set ***\n    {reason}")

        distribution = None
        if listed or shape:
            everyList = dict(listed)
            if shapeExponents is not None:
                everyList["shape"] = shapeExponents
            exponents = next(iter(everyList.values())) if everyList else [1]
            if len({tuple(np.atleast_1d(v).tolist()) for v in everyList.values()}) > 1:
                raise ValueError("moment exponents must match across families; pass the same list to "
                                 "each, or constrain one family's distribution at a time.")
            distribution = DistributionConstraints(self.packing, exponents,
                                                   area = "area" in listed, edge = "edge" in listed,
                                                   diagonal = "diagonal" in listed,
                                                   shape = bool(shape), deviation = bool(deviation),
                                                   fullShapeMoments = fullShapeMoments)
        if block is None and distribution is None:
            self.constraints = None
        elif distribution is None:
            self.constraints = block
        else:
            self.constraints = CompositeConstraints(block, distribution)
        if self.constraints is not None:
            self.constraints.projectPositions(self.packing)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def setShapeTemplate(self, morph = 1.0, sides = 4, keepEdges = False):
        """Aim every polygon at a TEMPLATE shape interpolated between its regular n-gon and a
        ``sides``-cornered polygon, and write the edge + diagonal targets that pin it. Returns self.

        ``keepEdges`` leaves the EDGE targets alone and writes only the diagonals, derived from those
        edges and the template's turning angles -- for a packing whose edge lengths are a degree of
        freedom (``generatePolygons(edgePolydispersity = ...)``) rather than something the template is
        entitled to overwrite. See the note on the diagonal below; and note that prescribing both the
        edges and the turning angles generically leaves a polygon that does not CLOSE, which is checked
        for and refused rather than handed to the springs.

        ``morph = 0`` is the regular n-gon (a disc, for large n); ``morph = 1`` is the cornered shape
        -- a square at ``sides = 4``. This is the handle for a shape CONTINUATION: jam a packing of
        discs, then walk ``morph`` to 1 and let the packing rearrange as the shapes grow corners.

        ``morph`` MAY BE PER POLYGON, an array of one value each, and that is how a spread in shape
        index is obtained. A single scalar makes every polygon the same shape at every step, so kappa
        is monodisperse by construction however the areas are drawn. Handing each polygon its own morph
        makes the kappa spread an ANNEALING FREEDOM in its own right: open it in the middle of the walk
        and close it by driving every entry to 1, exactly as a size spread is opened and closed. It is
        a spread in SHAPE at fixed area, orthogonal to what setLogNormalScale does to SIZE at fixed
        shape.

        THE INTERPOLATION IS ON TURNING ANGLES, NOT VERTICES, and that choice buys two things. The
        polygon stays EQUILATERAL at every ``morph`` -- turning by equal steps with equal edges -- so
        the per-edge targets remain one number per polygon and ``edge = True`` is valid the whole way.
        And a square drawn with n vertices is itself equilateral, so both endpoints are consistent
        rather than only the start.

        CLOSURE IS NOT AUTOMATIC and is worth stating. A polygon closes only if its edge vectors sum to
        zero, which interpolated turning angles do not guarantee in general. It holds here because the
        target is ``sides``-fold symmetric and n is a multiple of ``sides``: the vertices then fall into
        ``sides`` groups related by rotation through ``2 pi / sides``, whose sum vanishes identically.
        A template without that symmetry would need the vertices interpolated instead, at the cost of
        equilaterality. This raises rather than drifting open. UNEQUAL EDGES BREAK THAT ARGUMENT -- the
        rotation groups no longer cancel -- so ``keepEdges`` re-checks closure against the edge targets
        it was handed rather than against the template's own.

        THE DIAGONAL TARGET IS DERIVED FROM ITS ADJACENT EDGES, never copied out of the template. At
        vertex k the skip-one distance obeys the law of cosines on the two edges meeting there,
        ``d^2 = a^2 + b^2 - 2 a b cos(theta)``, so a diagonal length lifted from the template asserts
        the template's EDGE lengths along with its angle. Equal edges hide this -- the two agree
        exactly -- but with drawn edges they contradict each other, and the constraint set becomes
        infeasible in a way that reads as a convergence failure rather than as a bad target.

        It stays a LENGTH rather than becoming an angle, which is the other way to write the same
        condition. Measured: the gradient of a live-angle constraint VANISHES at a straight vertex
        (4e-10 against 4.4 for the length form at 180 degrees), and the square template makes 28 of a
        32-gon's vertices straight -- so an angle constraint would go rank-deficient across most of the
        polygon and leave exactly the floppy linkage the diagonals exist to prevent."""
        packing = self.packing
        stop = self._nonContainer()
        starts = np.asarray(packing.startIndices, dtype = int)
        morph = np.broadcast_to(np.asarray(morph, dtype = float).ravel(), (stop,)) \
            if np.ndim(morph) else np.full(stop, float(morph))
        if morph.shape != (stop,):
            raise ValueError(f"morph must be a scalar or one value per non-container polygon "
                             f"({stop}), got shape {np.shape(morph)}")
        if packing.targetDiagonal is None:
            packing.targetDiagonal = np.zeros(packing.numVertices, dtype = float)
        for polygon in range(stop):
            a, b = int(starts[polygon]), int(starts[polygon + 1])
            count = b - a
            if count % int(sides):
                raise ValueError(
                    f"polygon {polygon} has {count} vertices, which is not a multiple of "
                    f"sides = {sides}. The corners have to land ON vertices for the template to be "
                    f"representable, and the closure argument needs the {sides}-fold symmetry.")
            turnRegular = np.full(count, 2.0 * np.pi / count)
            turnTarget = np.zeros(count)
            turnTarget[::count // int(sides)] = 2.0 * np.pi / int(sides)
            here = float(morph[polygon])
            turn = (1.0 - here) * turnRegular + here * turnTarget
            heading = np.concatenate([[0.0], np.cumsum(turn)[:-1]])
            step = np.stack([np.cos(heading), np.sin(heading)], axis = 1)
            template = np.concatenate([[[0.0, 0.0]], np.cumsum(step, axis = 0)[:-1]])
            closure = float(np.linalg.norm(step.sum(axis = 0)))
            if closure > 1e-9 * count:
                raise ValueError(f"template for polygon {polygon} does not close "
                                 f"(|sum of edges| = {closure:.3e}); see the symmetry note.")
            area = 0.5 * abs(float(np.sum(template[:, 0] * np.roll(template[:, 1], -1)
                                          - np.roll(template[:, 0], -1) * template[:, 1])))
            scale = float(np.sqrt(abs(packing.targetArea[polygon]) / max(area, 1e-300)))
            template = template * scale
            if not keepEdges:
                packing.targetEdgeLength[a : b] = np.linalg.norm(
                    np.roll(template, -1, axis = 0) - template, axis = 1)
            else:
                # The turning angles alone no longer determine a closed polygon once the edges differ,
                # so it is checked here instead of being left to the springs to discover.
                lengths = np.asarray(packing.targetEdgeLength[a : b], dtype = float)
                gap = float(np.linalg.norm((lengths[:, None] * step).sum(axis = 0)))
                if gap > 1e-9 * float(lengths.sum()):
                    raise ValueError(
                        f"polygon {polygon}'s edge targets do not close under this template: the edge "
                        f"vectors miss by {gap:.3e}, which is {gap / float(lengths.sum()):.2%} of the "
                        f"perimeter. Prescribing the turning angles AND unequal edge lengths is two "
                        f"conditions too many -- a {sides}-cornered template with n vertices forces "
                        f"opposite runs of edges to have equal total length. Narrow the edge spread "
                        f"first (setTargetPolydispersity toward the reachable floor), or drop "
                        f"keepEdges and let the template equalize them.")
            # THE DIAGONAL FOLLOWS THE EDGES IT SPANS, by the law of cosines on the turning angle at
            # the vertex it is centred on -- entry k spans |r_{k+1} - r_{k-1}|, so its edges are the
            # one entering k (leaving k-1) and the one leaving k. Copying it off the template instead
            # would assert the template's edge lengths along with its angles.
            # ``turn`` is indexed by the edge it PRECEDES, so the turn at vertex k is turn[k-1]; the
            # interior angle is pi minus it, which flips the sign of the cosine term. Verified against
            # the template's own diagonals, which it reproduces exactly.
            edge = np.asarray(packing.targetEdgeLength[a : b], dtype = float)
            entering = np.roll(edge, 1)
            packing.targetDiagonal[a : b] = np.sqrt(
                entering ** 2 + edge ** 2 + 2.0 * entering * edge * np.cos(np.roll(turn, 1)))
        packing.syncTargetPerimeter()
        self._forces = None
        self._energy = None
        return self

    def spreadShapes(self, polydispersity, rng = None):
        """Stretch each polygon at CONSTANT AREA to seed a spread in edge lengths. Returns self.

        The opening move of an anneal under ``setConstraints(area = True, edge = [1, 2])``. A moment
        constraint can only ever be NARROWED -- the variance is at a minimum on a monodisperse
        configuration, so its gradient vanishes there and the retraction has no direction to widen
        along. The width therefore has to be put into the geometry directly, and then ramped down.

        Each polygon gets ``R(theta) diag(s, 1/s) R(-theta)`` about its own centroid, with a random axis
        and ``s = exp(g)``, ``g ~ N(0, polydispersity)``. The map has unit determinant, so every area is
        preserved EXACTLY and the hard area constraints stay satisfied -- only the shapes change, a
        square becoming a rotated rectangle. Call this BEFORE ``setConstraints``, so the moments are
        captured at the seeded width.

        Note this seeds the spread ACROSS a polygon's own edges as much as between polygons, which is
        what the global edge moments measure.
        """
        c = float(polydispersity)
        if c < 0.0:
            raise ValueError("polydispersity must be non-negative.")
        rng = self.rng if rng is None else np.random.default_rng(rng)
        packing = self.packing
        r = packing.positions.reshape(-1, 2)
        container = getattr(packing, "containerIndex", None)
        stop = packing.numPolygons if container is None else int(container)
        for polygon in range(stop):
            a, b = packing.startIndices[polygon], packing.startIndices[polygon + 1]
            theta = rng.uniform(0.0, np.pi)
            s = float(np.exp(c * rng.standard_normal()))
            cosine, sine = np.cos(theta), np.sin(theta)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            stretch = rotation @ np.diag([s, 1.0 / s]) @ rotation.T
            centroid = r[a : b].mean(axis = 0)
            r[a : b] = centroid + (r[a : b] - centroid) @ stretch.T
        self._forces = None
        self._energy = None
        return self

    def _distributionConstraints(self):
        """The active DistributionConstraints, however they were wrapped, or None."""
        if isinstance(self.constraints, DistributionConstraints):
            return self.constraints
        distribution = getattr(self.constraints, "distribution", None)
        return distribution if isinstance(distribution, DistributionConstraints) else None

    def constraintRank(self):
        """Live rank of the moment rows, or None when no distribution constraint is active.

        A blunt diagnostic -- see ``constraintConditioning``, which is the one that warns in time."""
        distribution = self._distributionConstraints()
        return None if distribution is None else distribution.rank(self.packing)

    def constraintConditioning(self):
        """How transverse the moment constraints still are: smallest/largest singular value, or None.

        Decays in direct proportion to the width of the distribution being held, so it is the signal
        for when an anneal should stop and hand off to per-object constraints. Its reciprocal is
        roughly the factor by which the projection amplifies force noise."""
        distribution = self._distributionConstraints()
        return None if distribution is None else distribution.conditioning(self.packing)

    def getPolydispersity(self):
        """Realized std/mean of each moment-constrained family, as a dict."""
        distribution = self._distributionConstraints()
        return {} if distribution is None else distribution.polydispersity(self.packing)

    def pinVertices(self, indices):
        """Hold the listed vertices FIXED: forces never move them. Pass ``None`` (or an empty list) to
        release every pin. ``indices`` may be a list/array of global vertex indices or a boolean mask
        of length ``numVertices``. Returns self.

        A pinned vertex is not removed from the physics -- it still pushes on its neighbors and still
        enters its polygon's area / edge terms. It simply does not move, and the reaction force it
        carries is excluded from ``getMaxUnbalancedForce``, since a pin's reaction is not an
        unbalanced force any more than a constraint's normal component is.

        Pinning composes with ``setConstraints``: the shape constraints are restricted to the free
        vertices, so SHAKE satisfies them by moving only what is allowed to move. Pin enough of a
        polygon and its shape constraints can become unsatisfiable (a triangle with all three
        vertices pinned has a fixed area, whatever the target says) -- ``constraintResidual`` will
        show the leftover, so check it after pinning heavily."""
        packing = self.packing
        if indices is None:
            packing.pinned = None
        else:
            mask = np.zeros(packing.numVertices, dtype = bool)
            arr = np.asarray(indices)
            if arr.dtype == bool:
                if arr.size != packing.numVertices:
                    raise ValueError(f"boolean pin mask must have length {packing.numVertices}, "
                                     f"got {arr.size}")
                mask[:] = arr
            elif arr.size:
                arr = arr.astype(int).reshape(-1)
                if arr.min() < 0 or arr.max() >= packing.numVertices:
                    raise ValueError(f"pin index out of range for {packing.numVertices} vertices")
                mask[arr] = True
            packing.pinned = mask if mask.any() else None
        self._forces = None
        self._energy = None
        return self

    def pinPolygons(self, polygons):
        """Pin every vertex of the listed polygons (indices or a boolean mask over polygons) -- the
        common case of holding whole shapes as fixed walls or obstacles. Returns self."""
        arr = np.asarray(polygons)
        if arr.dtype != bool:
            selected = np.zeros(self.packing.numPolygons, dtype = bool)
            selected[arr.astype(int).reshape(-1)] = True
        else:
            selected = arr
        return self.pinVertices(selected[self.packing.shapeId])

    def getPinnedVertices(self):
        """Global indices of the currently pinned vertices (empty array when none are pinned)."""
        pinned = self.packing.pinned
        return np.flatnonzero(pinned) if pinned is not None else np.zeros(0, dtype = int)

    def getShapeModel(self):
        """How each shape term is currently held: ``'rigid'`` (constrained), or the spring constant
        when it is soft. ``{'area': 'rigid', 'perimeter': 0.0, 'edge': 'rigid'}`` for the default."""
        c = self.constraints
        springs = {"area": self.kArea, "perimeter": self.kPerimeter, "edge": self.kEdge}
        return {name: ("rigid" if c is not None and getattr(c, name) else springs[name])
                for name in springs}

    def constraintResidual(self):
        """Largest fractional shape-constraint violation max|C| at the current positions (0 when
        unconstrained). The drift diagnostic: it should sit at the SHAKE tolerance (~1e-14) throughout
        a constrained run."""
        if self.constraints is None:
            return 0.0
        return self.constraints.maxResidual(self.packing)

    def setModelType(self, modelType):
        """Select the contact TIER -- the law that measures contact. Returns self.

        ``"area"``       exact unmollified overlap AREA (the default)
        ``"mollified"``  the C-infinity Plummer overlap area; needs ``setMollification``
        ``"softDepth"``  the smooth penetration DEPTH of notes/softDepth-1.pdf; needs ``setSoftDepth``
        ``"depth"``      the exact-distance contact law; needs ``setDepthContact``

        The first two measure contact by overlap area between loops. ``softDepth`` measures a
        penetration depth of a point into a loop instead and builds a Hertzian law on it, so its force
        is C2 at first contact rather than jumping, and it is real-analytic everywhere -- no medial-axis
        kink, no Voronoi-wall Hessian jump. See ``softDepth.py``.

        ``"sharp"`` WAS THE NAME OF THE AREA TIER and still selects it, once, with a warning. The word
        was needed for the other axis: a tier says how contact is measured, a GEOMETRY says what shape
        is handed to it (``setGeometryType``), and "sharp" is the natural name for the unrounded shape.
        Keeping it meaning both would make ``setModelType('sharp')`` and ``setGeometryType('sharp')``
        two unrelated statements that read identically."""
        if modelType == "sharp":
            global _WARNED_SHARP_TIER
            if not _WARNED_SHARP_TIER:
                _WARNED_SHARP_TIER = True
                warnings.warn(
                    "\n*** the 'sharp' TIER is now called 'area' ***\n"
                    "    setModelType('sharp') still selects it. 'sharp' now names a GEOMETRY "
                    "(setGeometryType), so the old spelling reads as a statement about shape rather "
                    "than about the contact law. Say setModelType('area').", stacklevel = 2)
            modelType = "area"
        if modelType not in ("area", "mollified", "softDepth", "depth"):
            raise ValueError(f"unknown tier {modelType!r}; use 'area', 'mollified', 'softDepth' or "
                             f"'depth'")
        self.modelType = modelType
        # Switching the contact law changes the energy and the force, so the cache cannot survive it.
        # Without this, getForces / getMaxUnbalancedForce right after a switch report the OLD tier's
        # numbers -- measured, a packing read max|F| = 9.957e-05 (its mollified value) immediately after
        # going sharp, where the sharp force was three decades larger.
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def setGeometryType(self, geometryType, rho = None, arcSegments = None, exact = None):
        """Select the GEOMETRY handed to the contact tier: ``"sharp"`` or ``"round"``. Returns self.

        ``"sharp"``  the backbone itself, straight edges and corners (the default)
        ``"round"``  each corner cut by a circle of radius ``rho_i`` pushed into it from inside,
                     the corner replaced by the arc -- see ``roundedGeometry`` and
                     ``notes/roundedDefinitions.tex``

        THIS IS ORTHOGONAL TO THE TIER. The rounded shape is an ordinary polygon by the time any law
        sees it, so ``area``, ``depth`` and their CUDA kernels all work on it unchanged. That is the
        point of chording the arcs rather than integrating along them.

        THE BACKBONE REMAINS THE STATE. Constraints, springs and the reported edge lengths and areas
        are the BACKBONE's, so ``setConstraints(area = True, edge = True)`` keeps meaning exactly what
        it meant. ``rho`` only ever removes material -- ``getRoundedAreas`` reports what is left.

        ``exact`` chooses HOW the round geometry is evaluated. False (the default) chords each corner
        into ``arcSegments`` pieces and hands an ordinary polygon to the tier, which keeps the CUDA
        kernels. True uses ``roundedContact``, where the corner IS an arc and every distance, crossing
        and integral is taken against it -- exact, but numpy-only and far slower, and currently only on
        the ``depth`` tier.

        THE TWO AGREE, AND THE CHORDED ONE IS THE APPROXIMATION. Measured on one pair, the chorded
        energy approaches the exact one as ``1/arcSegments^2`` -- relative error 1.0e-02, 2.5e-03,
        6.3e-04 at 6, 12 and 24 segments. Past about 100 vertices per body the chorded path stops
        converging and starts to DEGRADE (measured 10.6% and 17.9% wrong at 196 and 388 vertices,
        against a brute-force line integral of its own geometry), so raising ``arcSegments`` is not a
        route to accuracy -- ``exact = True`` is.

        ``rho`` defaults to zero, which is the sharp shape written the long way. A strictly positive
        floor is enforced when rounding is active, because at ``rho = 0`` a corner's whole arc collapses
        onto the vertex and a zero-length edge has no tangent for the contact law to divide by. Take
        the limit by switching back to ``"sharp"``, not by driving ``rho`` to zero underneath the law."""
        if geometryType not in ("sharp", "round"):
            raise ValueError(f"unknown geometry {geometryType!r}; use 'sharp' or 'round'")
        self.geometryType = geometryType
        if exact is not None:
            self.exactArcs = bool(exact)
        if arcSegments is not None:
            self.arcSegments = int(arcSegments)
        if rho is not None:
            self.setRho(rho)
        elif geometryType == "round" and self.rho is None:
            raise ValueError("geometry 'round' needs radii: pass rho= or call setRho() first")
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getGeometryType(self):
        """The current geometry, ``"sharp"`` or ``"round"``."""
        return getattr(self, "geometryType", "sharp")

    # UNVERIFIED(Cam)
    def _exactMeasurements(self):
        """``roundedContact.packingMeasurements`` at the current state -- the exact-arc counterpart of
        ``measuredGeometry``.

        WHY THE GETTERS NEED THEIR OWN PATH. ``measuredGeometry`` hands them the CHORDED polygon, which
        is exactly right when the law is chorded too and quietly wrong when it is not: the force would
        be exact while ``getExcessEnergy`` -- the number a load controller steers on -- read a shape
        differing from it by up to 5% at the round end of a schedule. Measured at N = 26, chorded
        against exact: 5.4e-02 relative at q = 0.85, 8.7e-04 at q = 0.20, 3.9e-05 at q = 0.02."""
        import roundedContact
        depth = self.modelType == "depth"
        return roundedContact.packingMeasurements(
            self.packing, self.rho,
            self.depthStiffness if depth else 1.0,
            self.depthWallStiffness if depth else 1.0)

    # UNVERIFIED(Cam)
    def _exactArcs(self):
        """True when the tier is being handed true arcs rather than a chorded polygon."""
        return self.getGeometryType() == "round" and getattr(self, "exactArcs", False)

    # UNVERIFIED(Cam)
    @contextlib.contextmanager
    def measuredGeometry(self):
        """Point the model at the shape the TIER ACTUALLY SEES, for the duration of a measurement.

        Without this every getter would report the backbone. That is not a small discrepancy under
        rounding: the backbone squares of a valid rounded packing routinely OVERLAP, because the
        corners that would have collided were cut away. ``getOverlapArea`` on the backbone would then
        report a violation for a configuration that is perfectly disjoint, and any jamming criterion
        written on it would be reading a shape that is not being simulated.

        A no-op under sharp geometry, so wrapping a getter costs nothing there."""
        if self.getGeometryType() != "round":
            yield
            return
        saved = (self.packing, self.geometryType)
        self.packing, self.geometryType = self.roundedPacking(), "sharp"
        try:
            yield
        finally:
            self.packing, self.geometryType = saved

    # UNVERIFIED(Cam)
    def setRho(self, rho):
        """Set the per-vertex corner radius. Scalar, per-polygon or per-vertex. Returns self.

        FEASIBILITY IS PER EDGE, NOT PER CORNER: each edge must fit the two kiss offsets it carries,
        ``t_k + t_next <= |e_k|``. Two corners share an edge, so a radius that is fine on its own can
        be infeasible beside a large neighbor -- which is why this validates the whole set rather than
        clamping each value. ``getMaxRho()`` reports the largest EQUAL radius per vertex."""
        packing = self.packing
        rho = np.asarray(rho, dtype = float)
        if rho.ndim == 0:
            rho = np.full(packing.numVertices, float(rho))
        elif rho.size == packing.numPolygons:
            rho = rho[packing.shapeId]
        rho = rho.reshape(packing.numVertices).copy()
        # The container is never rounded: its region is its exterior, and a wall with rounded corners
        # is a different container rather than the same one drawn more smoothly.
        container = getattr(packing, "containerIndex", None)
        if container is not None:
            rho[int(packing.startIndices[int(container)]):] = 0.0
        if np.any(rho < 0.0):
            raise ValueError("rho must be non-negative")
        slack = roundedGeometry.edgeSlack(packing.positions, packing.prev, packing.next, rho)
        active = slice(0, self._activeVertexCount())
        # Relative, because the bound is MET exactly at the disk limit: two neighbors both at the cap
        # give t_k + t_next = |e_k| to the last bit, and a strict test then rejects the very state the
        # cap is defined by over a 3e-17 rounding error.
        tolerance = 1e-12 * float(np.max(np.hypot(*(packing.positions.reshape(-1, 2)
                                                   [packing.next] - packing.positions
                                                   .reshape(-1, 2)).T)))
        if np.any(slack[active] < -tolerance):
            worst = int(np.argmin(slack[active]))
            raise ValueError(
                f"rho is infeasible at edge {worst}: the two kiss offsets overrun it by "
                f"{-slack[worst]:.3e}. Feasibility is a per-EDGE condition (t_k + t_next <= |e_k|), "
                f"so reduce this radius or its neighbor; getMaxRho() gives the equal-radius bound.")
        self.rho = rho
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getRho(self):
        """The per-vertex corner radius, or None if never set."""
        return None if self.rho is None else np.asarray(self.rho, dtype = float)

    # UNVERIFIED(Cam)
    def getMaxRho(self, safety = 1.0):
        """Largest EQUAL radius every corner could take, per vertex. Half the edge at a square, where
        the four arcs meet and the shape becomes a disk."""
        return roundedGeometry.maxRho(self.packing.positions, self.packing.prev,
                                      self.packing.next, safety = safety)

    def _activeVertexCount(self):
        """Vertices belonging to real polygons, i.e. excluding any container."""
        container = getattr(self.packing, "containerIndex", None)
        return (self.packing.numVertices if container is None
                else int(self.packing.startIndices[int(container)]))

    # UNVERIFIED(Cam)
    def roundedPacking(self):
        """The packing the tier actually sees under geometry ``"round"``: arcs in place of corners.

        The container is copied through UNROUNDED, so the wall stays a wall and the rounding never has
        to survive a dedup at ``rho = 0`` on a polygon that has no radii anyway.

        THE OBJECT IS REUSED ACROSS CALLS, only its coordinates rewritten. This is built on EVERY force
        evaluation, and ``Packing.__init__`` runs ``buildConnectivity``, which is a Python loop over
        polygons -- so constructing it afresh cost more than the contact law it feeds. Measured on 11
        rounded squares, a 20000-step FIRE run spent 507 s rebuilding what changes only when the vertex
        COUNT changes, which it does not during a relaxation. The candidate list is dropped each time
        because the coordinates did move."""
        packing = self.packing
        stop = self._activeVertexCount()
        sign = roundedGeometry.convexSign(packing.positions, packing.prev, packing.next)
        arcs = roundedGeometry.roundedPositions(packing.positions, packing.prev, packing.next,
                                                self.rho, sign, self.arcSegments)
        arcs = arcs.reshape(-1, self.arcSegments + 1, 2)[:stop].reshape(-1, 2)
        vertices = packing.positions.reshape(-1, 2)
        pieces = [arcs]
        counts = list(np.diff(packing.startIndices[:self._containerSlot() + 1])
                      * (self.arcSegments + 1))
        container = getattr(packing, "containerIndex", None)
        if container is not None:
            pieces.append(vertices[stop:])
            counts.append(len(vertices) - stop)
        positions = np.concatenate(pieces).reshape(-1)
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(int)

        # THE TARGET AREAS COME ACROSS FROM THE BACKBONE. They are what the AREA tier normalizes its
        # overlap by -- U = 2k sum (a_AB / (targetArea_A + targetArea_B))^2 -- so leaving them at the
        # constructor's 1.0 silently rescales that whole tier. Measured at N = 11, phi = 0.72: a
        # normalizer of 1.0 against a true target of 0.0655 made the reported contact energy 234 times
        # too small, uniformly across the pair and container terms. It is a per-POLYGON quantity, so it
        # transfers unchanged even though the vertex count does not; the per-vertex edge targets cannot
        # and stay at 1.0, which is harmless because no tier normalizes by them.
        targetArea = np.asarray(packing.targetArea, dtype = float)[:len(starts) - 1].copy()

        cached = getattr(self, "_roundedCache", None)
        if cached is not None and cached.positions.size == positions.size \
                and np.array_equal(cached.startIndices, starts):
            cached.positions[:] = positions
            cached.targetArea[:] = targetArea
            cached.candidatePairs = None
            cached._forces = None
            return cached
        rounded = Packing(positions = positions, startIndices = starts,
                          box = packing.box, targetArea = targetArea, targetEdgeLength = 1.0)
        rounded.containerIndex = container
        self._roundedCache = rounded
        return rounded

    def _containerSlot(self):
        container = getattr(self.packing, "containerIndex", None)
        return self.packing.numPolygons if container is None else int(container)

    # UNVERIFIED(Cam)
    def getRoundedAreas(self):
        """Per-polygon area AFTER the corner cuts: the backbone area minus each corner's kite-minus-
        wedge, ``rho t - rho^2 psi / 2``. Analytic, so it does not inherit the chording error."""
        packing = self.packing
        loss = roundedGeometry.cornerAreaLoss(packing.positions, packing.prev, packing.next,
                                              self.rho)
        areas = self.getAreas().astype(float).copy()
        for polygon in range(self._containerSlot()):
            block = slice(int(packing.startIndices[polygon]), int(packing.startIndices[polygon + 1]))
            areas[polygon] -= float(loss[block].sum())
        return areas

    # UNVERIFIED(Cam)
    def getRhoForces(self):
        """``-dE/drho`` per vertex from the last ``calcForceEnergy``, or None under sharp geometry.

        POSITIVE means the energy falls as that corner is cut deeper. Under overlap it generally is
        positive: a larger radius removes more material, so rounding is the cheapest way for a loaded
        packing to shed contact. That is exactly the freedom a schedule then has to withdraw."""
        return getattr(self, "_rhoForces", None)

    # UNVERIFIED(Cam)
    def setSoftDepth(self, fraction = None, epsilon = None, stiffness = 1.0, adhesionWork = 0.0,
                     adhesionRange = None, quadratureOrder = 16):
        """Configure the soft-depth contact law and switch to it. Returns self.

        ``fraction`` sets ``epsilon`` as a FRACTION of the mean edge length, mirroring
        ``setSofteningFraction``, and is the spelling to prefer: an absolute ``epsilon`` silently
        becomes nonsense when the edges are short. At N=32, n=32 the mean edge is 0.0218, so an
        innocent-looking ``epsilon = 1e-2`` is 46% of an edge and gives a corner rounding radius 9.3x
        the polygon itself. ``fraction = 1e-2`` is the default and gives 2.18e-04.

        DO NOT reach for ``setSofteningFraction`` to get this -- that method switches the model to
        ``"mollified"`` as its first action, so calling it after ``setModelType("softDepth")`` silently
        puts you back on the mollified tier. Measured on that path, FIRE stalls at 3.96e-06 and CG then
        burns all 1000 steps over 185 s without moving ``max|F|`` a single bit; the same packing on
        soft depth reaches 1.9e-10 in 67 FIRE steps and CG confirms it in 7.

        ``epsilon`` is the softmin length of ``notes/softDepth-1.pdf`` eq (5), and it is a SHAPE
        parameter rather than a numerical regulator: it sets the corner rounding radius (15), the
        medial-axis smearing width, and the Hessian magnitude all at once. Defaults to 1% of the mean
        edge length.

        ``adhesionWork`` (W) and ``adhesionRange`` (lambda) add the adhesive term of (19). W is exactly
        the work required to separate a bond, independently of k and lambda (21). Leaving W at zero
        gives the purely repulsive Hertzian law (16).

        ``quadratureOrder`` is the Gauss-Legendre order per panel of the boundary integral
        ``E = int_{dA} phi(h_eps^B) dl``. THE ORDER IS COUPLED TO ``epsilon``: the integrand varies on
        the scale of ``epsilon``, so a sharper shape needs a higher order. Measured on two squares in
        face-to-face contact, relative error in the energy is 3.9e-06 at order 16 for
        ``epsilon/edge = 1e-2``, but only 7.7e-05 at order 16 for ``epsilon/edge = 1e-3``, reaching
        7.9e-08 there at order 32. Corner contacts are far easier -- 7.0e-10 at order 16. Use
        ``tests/softDepthCheck.py`` check 10 to pick an order for a given epsilon rather than guessing.

        THE PARAMETER HIERARCHY MATTERS (sec 17). The note requires ``epsilon << lambda`` -- otherwise
        adhesion samples the corner rounding and bond strength becomes an artifact of the regularizer
        -- and ``lambda << min edge``, or the contact chord is smeared across several faces and the
        face/vertex contact distinction is lost. ``adhesionRange`` therefore defaults to 10*epsilon,
        and a violation of either inequality warns."""
        if self.packing is None:
            raise ValueError("build a packing before configuring the soft depth")
        meanEdge = float(np.mean(self.packing.targetEdgeLength))
        if fraction is not None and epsilon is not None:
            raise ValueError("give setSoftDepth either fraction or epsilon, not both")
        if epsilon is not None:
            self.softEpsilon = float(epsilon)
        else:
            self.softEpsilon = float(1e-2 if fraction is None else fraction) * meanEdge
        self.softFraction = self.softEpsilon / meanEdge
        self.softStiffness = float(stiffness)
        self.adhesionWork = float(adhesionWork)
        self.adhesionRange = float(adhesionRange) if adhesionRange is not None \
            else 10.0 * self.softEpsilon
        self.quadratureOrder = int(quadratureOrder)
        if self.adhesionWork != 0.0:
            if self.softEpsilon >= self.adhesionRange:
                warnings.warn(
                    f"\n*** epsilon {self.softEpsilon:.3g} is not << adhesionRange "
                    f"{self.adhesionRange:.3g} ***\n    Adhesion then samples the CORNER ROUNDING "
                    f"rather than the shape, so the bond strength becomes an artifact of the "
                    f"regularizer (notes/softDepth-1.pdf sec 17).", stacklevel = 2)
            shortest = float(np.min(self.packing.targetEdgeLength))
            if self.adhesionRange >= shortest:
                warnings.warn(
                    f"\n*** adhesionRange {self.adhesionRange:.3g} is not << the shortest edge "
                    f"{shortest:.3g} ***\n    The contact chord is smeared across several faces and "
                    f"the face/vertex contact distinction -- the source of the debond/rebond "
                    f"bistability -- is lost.", stacklevel = 2)
        self.modelType = "softDepth"
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def setDepthContact(self, stiffness = 1.0, wallStiffness = 1.0):
        """Select the EXACT-DISTANCE contact law of ``notes/polygonContact`` and set its stiffness.

            E = 1/2 sum over ordered pairs  int_{dP cap Q} (k/3) d_Q(x)^3 dl(x)

        ``d_Q`` is the exact distance to the boundary, so unlike ``setSoftDepth`` there is NO
        regularization length to choose and no convexity requirement: nonconvex bodies are handled with
        no decomposition. Closed-form energy and gradient, no quadrature.

        THE ONE HARD CONSTRAINT is ``dMax / rIn << 1``. Past the medial-axis ridge the repulsion
        REVERSES SIGN and bodies are pulled through, so this is a correctness limit rather than an
        accuracy one, and for a limbed shape ``rIn`` is the LIMB half-width. Check it with
        ``polyContactSystem.systemValidity``; the initialization protocol in that module enforces it.

        THE CONTAINER is handled by winding: the wall is passed CLOCKWISE so the confining region is
        its exterior, and it then rides the same batched pair loop as every body. (The older note here
        saying no container term exists was left behind by that work.)

        ``wallStiffness`` MULTIPLIES the contact stiffness for the container term alone, and it exists
        because at 1.0 the wall is the softest way out of a stressed packing. Body contact and wall
        penetration are alternatives, not independent terms: a confined packing relieves stress
        through whichever is cheaper, and escaping lowers the confinement for everybody while
        overlapping a neighbor relieves nothing globally. Measured here at ``wallStiffness = 1``, a
        state the density controller reported as jammed had 100.00% of its contact energy in wall
        penetration, 1.83e-19 in body contact, and a pair overlap area of EXACTLY zero -- nothing was
        touching anything, 17 vertices were just outside the box. The mollified tier met the same
        failure and answered it with ``_DEFAULT_CONTAINER_STIFFNESS = 10.0``, worth +5.4% in density.

        IT IS FREE. The multiplier rides the same batched pair loop -- and the same CUDA kernel -- as
        everything else, applied to the work items that touch the exterior body, which is exact because
        energy and gradient are both linear in the stiffness. An earlier spelling added the surplus
        through ``confinementEnergyGradient`` instead and cost 76.9 ms of a 92.3 ms force evaluation at
        N = 11, since that is the slow per-body reference rather than the kernel.

        There is no validity limit at a convex wall however hard it is pressed: the exterior of a
        convex region has no medial axis, so the ``dMax / rIn`` cap that governs body contact does not
        apply there."""
        self.modelType = "depth"
        self.depthStiffness = float(stiffness)
        self.depthWallStiffness = float(wallStiffness)
        if self.depthWallStiffness <= 0.0:
            raise ValueError(f"wallStiffness must be positive, got {wallStiffness}")
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def getEquilibriumIndentation(self):
        """The preferred overlap ``h*`` of the current adhesive law, eq (24). Zero without adhesion."""
        from softDepth import equilibriumIndentation
        return equilibriumIndentation(self.softStiffness, self.adhesionWork, self.adhesionRange)

    def setMollification(self, sigma):
        """Set the mollification (softening) width ``sigma`` directly, as an ABSOLUTE length, and
        switch to the mollified model. Also sizes the self-repulsion range (delta = sigma). Requires a
        built packing. Returns self."""
        self.modelType = "mollified"
        self.sigma = float(sigma)
        self.sigmaFraction = self.sigma / float(np.mean(self.packing.targetEdgeLength))
        return self

    def setSofteningFraction(self, fraction):
        """Set the mollification width as a FRACTION of the mean edge length: ``sigma = fraction *
        mean(targetEdgeLength)``. Switches to the mollified model. Also sizes the self-repulsion range
        (delta = sigma). Requires a built packing. Returns self."""
        self.modelType = "mollified"
        self.sigmaFraction = fraction
        self.sigma = float(fraction) * float(np.mean(self.packing.targetEdgeLength))
        return self

    @property
    def delta(self):
        """Self-repulsion range: a fraction of the mean target edge length (see
        ``setSelfRepulsionRange``).

        Deliberately NOT tied to sigma. The barrier guards against a polygon folding through itself,
        so its scale belongs to the polygon's own geometry, not to the contact mollification. When it
        was sigma, an absolute sigma against an edge length l0 = P/n that shrinks with n drove
        sigma/l0 up to 0.23 by n = 16 -- and at that ratio the Gaussian still has weight at distance
        l0, so the barrier fired between edges i and i+2 in a perfectly valid, unfolded polygon. The
        self-repulsion energy grew 13 orders of magnitude between n = 6 and n = 16 purely from that.
        A fraction of l0 scales correctly with n and stays silent unless edges genuinely approach."""
        return self.selfRepFraction * float(np.mean(self.packing.targetEdgeLength))

    def setSelfRepulsionRange(self, fraction = 0.05):
        """Set the self-repulsion barrier range as a FRACTION of the mean target edge length.

        The barrier only has to notice edges that are about to cross, so this wants to sit well below
        the natural spacing between non-adjacent edges (about one edge length). The default 0.05 puts
        the Gaussian 20x below that spacing -- exp(-1/(2*0.05^2)) is zero to any precision at distance
        l0 -- while still rising sharply once two edges close to within a few percent of an edge.
        Raise it only if a genuine fold is slipping past. Returns self."""
        if not 0.0 < fraction < 0.5:
            raise ValueError(f"self-repulsion fraction {fraction} must be in (0, 0.5); it is a "
                             f"fraction of the edge length and must stay well below 1")
        self.selfRepFraction = float(fraction)
        self._forces = None
        self._energy = None
        return self

    # UNVERIFIED(Cam)
    def setNeighborSkin(self, skin = None, enabled = True):
        """Size the neighbor list's Verlet skin, or turn the list off entirely. Returns self.

        ``skin`` defaults to a quarter of the mean edge length. Bigger means fewer rebuilds and more
        candidate pairs; smaller the reverse. ``enabled = False`` reverts to the all-to-all
        intersection scan, which is the reference the list is tested against and is worth having as an
        escape hatch, not as a routine choice: measured at n = 16, N = 128, the scan takes 9.4 SECONDS
        against 0.62 ms with the list.

        Read the realized trade with ``neighborStatistics()``; its ``reuse`` field is the mean number
        of force evaluations per rebuild, which is the number that says whether the skin is sized
        well."""
        self.useNeighborList = bool(enabled)
        self.neighbors = None
        if self.packing is not None:
            self.packing.candidatePairs = None
        self._skin = None if skin is None else float(skin)
        return self

    # UNVERIFIED(Cam)
    def neighborStatistics(self):
        """``{'skin', 'builds', 'uses', 'reuse', 'pairs'}`` for the live neighbor list, or ``None``."""
        return None if self.neighbors is None else self.neighbors.statistics()

    # UNVERIFIED(Cam)
    def _refreshNeighbors(self):
        """Attach an up-to-date candidate list to the packing before an intersection scan.

        SELF-INVALIDATING BY CONSTRUCTION, which is the point. The list compares the current positions
        against the ones it was built at and rebuilds when any vertex has moved more than half the
        skin, so there is no register of geometry-moving methods to keep in sync -- ``setPackingFraction``,
        ``spreadShapes``, ``setShapeDeficit``, a SHAKE retraction and a FIRE step are all just motion,
        and a changed vertex count is caught by the shape mismatch. A manual invalidation list would be
        one forgotten call away from silently under-reporting overlap."""
        # ONLY WHEN SOMETHING WILL CONSUME IT. The list feeds ``updateIntersections``, which runs for
        # the sharp overlap, either container term, and ``getOverlapArea``. The mollified tier in a
        # PERIODIC box touches none of those, so refreshing there rebuilds a list nothing reads --
        # measured at 6.1% of a mollified FIRE run (133 rebuilds in 300 steps) before this gate.
        if (not self.useNeighborList or self.packing is None
                or (self.modelType != "area"
                    and getattr(self.packing, "containerIndex", None) is None)):
            if self.packing is not None:
                self.packing.candidatePairs = None
            return
        if self.neighbors is None:
            from neighbors import NeighborList
            self.neighbors = NeighborList(self.packing, skin = getattr(self, "_skin", None))
        self.packing.candidatePairs = self.neighbors.candidates(self.packing)

    # UNVERIFIED(Cam)
    def setContainerStiffness(self, kContainer):
        """Set the wall's contact stiffness RELATIVE to the inter-particle one. Returns self.

        The default is 10.0, NOT 1.0 -- pass 1.0 explicitly for the old equal-stiffness behavior.

        The ratio decides how an overjammed packing relieves stress, and at 1.0 it picks the wrong way.
        Both contacts obey the same normalized-squared law, but escaping through the wall lowers the
        effective confinement for EVERY polygon while overlapping a neighbor relieves nothing globally,
        so leaking out is the cheaper route. Measured on 11 squares decompressing from above: the
        polygon-polygon overlap reached exactly zero at phi = 0.7015, and the remaining 0.04 of density
        was spent waiting for the packing to climb back inside the box, which it was sticking out of by
        5e-03 -- on a unit box, not a marginal contact. The reported density was 0.6635 where the
        polygons had stopped overlapping at 0.7015.

        Raising this makes escape more expensive than overlap. Measured on 11 squares, same build and
        seed, five points on one curve: phi = 0.6637 at k = 1, 0.6836 at 3, 0.6998 at 10, 0.6973 at 30,
        0.6944 at 100. A monotone rise to a broad peak at 10 (+5.4% density) and a gentle decline after,
        consistent with the stiffer mode shortening FIRE's timestep -- its ceiling scales as 1/sqrt(k).

        Both tiers carry the stiffness, numpy and CUDA alike, agreeing to 5e-15 relative in energy and
        force at k = 1, 10 and 100.

        IT ROUTES BY TIER, because the wall multiplier is stored in two places. ``sharp``, ``mollified``
        and ``softDepth`` read ``kContainer``; the exact-distance ``depth`` law reads
        ``depthWallStiffness`` instead, which until now was reachable only through ``setDepthContact``
        -- and that also resets the contact stiffness and re-asserts the tier, so it cannot be used to
        adjust the wall alone. Writing ``kContainer`` while the depth tier is live did NOTHING, silently:
        measured on 11 polygons, ``setContainerStiffness(100)`` moved the depth energy by 1.08e-19 and
        the force by 8.13e-20 -- rounding -- while ``depthWallStiffness`` stayed at 1.0. That is the
        worst possible failure for this particular knob, since 1.0 is the value whose measured
        consequence is a packing that reports itself jammed with nothing touching and 17 vertices
        outside the box.

        THE VALUE IS PER TIER AND DOES NOT FOLLOW A SWITCH. Set it AFTER choosing the contact law. The
        defaults differ for measured reasons -- 10.0 for the area tiers, where the density peaks, and
        1.0 for depth -- so carrying one tier's choice across to the other would be a guess, not a
        conversion."""
        if kContainer <= 0.0:
            raise ValueError(f"container stiffness must be positive, got {kContainer}")
        if self.modelType == "depth":
            self.depthWallStiffness = float(kContainer)
        else:
            self.kContainer = float(kContainer)
        self._forces = None
        self._energy = None
        return self

    def _requireSigma(self):
        if self.sigma is None:
            raise ValueError("softening width unset -- call setSofteningFraction(...) first")

    def _warnIfSofteningTooSharp(self):
        """Loudly warn before dynamics when the softening fraction is far below the edge scale. The
        mollified contact force scales like 1/sigma, so a razor-thin sigma (e.g. setSofteningFraction
        (1e-8)) makes any FIRE/Newton step overshoot the contact and diverge to NaN. Tiny sigma is
        still fine for STATIC energy/force/overlap-area inspection, so this fires only here, at the
        minimizer entry points."""
        frac = self.sigmaFraction
        # A hair of slack, because the anneal deliberately RAMPS sigma down TO the floor: the mean edge
        # length shifts by roundoff as the packing relaxes, so a sigma set exactly at the floor reads
        # back as 0.00999805 of it and would fire this warning on every step of a correct schedule.
        if frac is not None and frac < _MIN_STABLE_SOFTENING_FRACTION * (1.0 - 1e-3):
            warnings.warn(
                f"\n*** softening fraction {frac:g} is far too small for dynamics ***\n"
                f"    sigma = {self.sigma:.2e} is only {frac:g} of the mean edge; the mollified "
                f"contact force ~ 1/sigma will make FIRE/Newton diverge to NaN.\n"
                f"    Use setSofteningFraction(~0.05-0.10) for stable minimization. Very small sigma "
                f"is fine only for static energy/force inspection (updateForcesParallel etc.).",
                stacklevel = 3)

    def _warnIfSelfRepActive(self):
        """Warn (after a minimize) when the self-repulsion barrier is meaningfully active -- its energy
        above ``_SELF_REP_WARN_ENERGY``. Self-repulsion is meant to be a SAFETY NET that stays ~0; if
        it fires, a polygon is near-folding and the relaxed state is partly held up by the barrier
        rather than by the overlap / shape springs. See ``energyBreakdown``."""
        if self.modelType != "mollified":
            return                        # self-repulsion is a mollified-model term only
        eR, _ = selfRepulsionEnergyForce(self.packing, self.kSelf, self.delta)
        if eR > _SELF_REP_WARN_ENERGY:
            eO, _ = plummerOverlapExact(self.packing, self.sigma)
            eS, _ = eqSoftBodyEnergyForce(self.packing, self.kEdge, self.kArea, relative = True)
            warnings.warn(
                f"\n*** self-repulsion fired (energy {eR:.2e}, {eR / (eO + eS + eR) * 100:.1f}% of "
                f"total) ***\n    it should stay ~0 as a safety net -- a polygon is near-folding. "
                f"Stiffen the springs / check the shape (energyBreakdown() shows the split).",
                stacklevel = 3)

    def _forceEnergy(self, packing):
        """(energy, flat force) for the selected model type, plus the relative eqSoftBody springs.
        ``mollified``: fully-analytic (closed-form) Plummer overlap + self-repulsion -- the overlap
        gradient is self-consistent with its energy to the machine floor, so Newton's FD Hessian is
        accurate. ``sharp``: exact unmollified overlap area, no softening needed.

        A CONSTRAINED term's spring is omitted (per term, so an unconstrained one still acts): the
        shape is held there by projection, and adding its penalty too would double-count it and
        reintroduce the very stiffness the constraints exist to remove."""
        # ROUNDED GEOMETRY IS A CHANGE OF SHAPE, NOT A CHANGE OF LAW. The tier below is handed the
        # arc-chorded polygon and never learns that it was rounded; the gradient it returns is pulled
        # back onto (backbone, rho) afterwards. Done here, at the top, so every tier inherits it.
        if self.getGeometryType() == "round" and packing is self.packing:
            if getattr(self, "exactArcs", False):
                return self._exactForceEnergy(packing)
            return self._roundForceEnergy(packing)

        # Refresh the candidate list first: on the sharp tier BOTH the container term and the overlap
        # term run an intersection scan, and this is the call that dominates a sweep.
        self._refreshNeighbors()

        # The soft-depth tier is a boundary-area law, not an area law, so it replaces the overlap AND
        # the container terms outright rather than sitting alongside them -- there is no intersection
        # scan and no overlap area anywhere in it. The shape springs still apply.
        # The EXACT-DISTANCE contact law of notes/polygonContact: E = 1/2 sum int_{dP cap Q} (k/3)
        # d_Q^3 dl, closed form throughout, no regularization length, nonconvex with no decomposition.
        # It supersedes softDepth; see notes/penetrationDepthReview.md and TODO.md.
        if self.modelType == "depth":
            import polyContactSystem
            kEdgeDepth, kAreaDepth = self.kEdge, self.kArea
            if self.constraints is not None:
                kEdgeDepth = 0.0 if self.constraints.edgeHeld else kEdgeDepth
                kAreaDepth = 0.0 if self.constraints.area else kAreaDepth
            if kEdgeDepth == 0.0 and kAreaDepth == 0.0:
                eS, fS = 0.0, np.zeros_like(packing.positions)
            else:
                eS, fS = eqSoftBodyEnergyForce(packing, kEdgeDepth, kAreaDepth, relative = True)
            # The wall's stiffness rides the SAME batched pair loop as everything else, applied as a
            # per-pair multiplier on the work items that touch the exterior body. It is exact -- energy
            # and gradient are both linear in k -- and it is free, which the earlier spelling was not:
            # adding the surplus through confinementEnergyGradient cost 76.9 ms of a 92.3 ms evaluation
            # at N = 11, because that is the slow per-body reference rather than the kernel.
            eD, fD = polyContactSystem.packingEnergyForce(
                packing, self.depthStiffness, wallStiffness = self.depthWallStiffness)
            return eD + eS, fD.reshape(-1) + fS

        if self.modelType == "softDepth":
            if self.softEpsilon is None:
                raise ValueError("soft depth unconfigured -- call setSoftDepth(...) first")
            import softDepth
            kEdgeSoft, kAreaSoft = self.kEdge, self.kArea
            if self.constraints is not None:
                kEdgeSoft = 0.0 if self.constraints.edgeHeld else kEdgeSoft
                kAreaSoft = 0.0 if self.constraints.area else kAreaSoft
            if kEdgeSoft == 0.0 and kAreaSoft == 0.0:
                eS, fS = 0.0, np.zeros_like(packing.positions)
            else:
                eS, fS = eqSoftBodyEnergyForce(packing, kEdgeSoft, kAreaSoft, relative = True)
            eD, fD = softDepth.packingEnergyForce(
                packing, self.softEpsilon, self.softStiffness,
                self.adhesionWork, self.adhesionRange, self.kContainer,
                getattr(self, "quadratureOrder", 16))
            return eD + eS, fD + fS

        kEdge, kArea = self.kEdge, self.kArea
        if self.constraints is not None:
            kEdge = 0.0 if self.constraints.edgeHeld else kEdge
            kArea = 0.0 if self.constraints.area else kArea
        onGpu = cudaOverlap is not None and cudaOverlap.isAvailable()

        # SKIP the shape term entirely when both stiffnesses are zero -- whether the user set them so
        # or a constraint zeroed them above. The result is identically zero, but computing it is not
        # free: the CPU path builds every edge length and polygon area, and the GPU path uploads the
        # whole packing to return a buffer of zeros. Under setConstraints(area = True, edge = True)
        # that is wasted on EVERY force evaluation of the run.
        if kEdge == 0.0 and kArea == 0.0:
            eS, fS = 0.0, np.zeros_like(packing.positions)
        elif onGpu:
            eS, fS = cudaOverlap.springsCuda(packing, kEdge, kArea)
        else:
            eS, fS = eqSoftBodyEnergyForce(packing, kEdge, kArea, relative = True)

        # A container is not an ordinary polygon: it confines rather than repels, so it is held out of
        # the pairwise overlap and handled by containerEnergyForce. It is always the LAST polygon, so
        # excluding it is a matter of shortening the pair loop.
        container = packing.containerIndex
        # ONLY THE MOLLIFIED RETURN CONSUMES eW. The area tier falls through to its own
        # containerEnergyForce below with identical arguments, so computing it here as well threw a
        # whole numpy container pass away on every force evaluation -- measured at 27.5 ms/eval for 11
        # rounded squares against 14.6 once this gate was added, i.e. very nearly half the tier.
        if container is None or self.modelType != "mollified":
            eW, fW = 0.0, 0.0
        else:
            # The CUDA path omits the WALL's own gradient, so it is only valid when the wall is
            # fully pinned (the normal case) -- otherwise fall back to numpy, which computes it.
            wallPinned = (packing.pinned is not None
                          and bool(packing.pinned[packing.startIndices[container]:].all()))
            if onGpu and wallPinned:
                eW, fW = cudaOverlap.containerEnergyForceCuda(packing, self.sigma,
                                                              kContainer = self.kContainer)
            else:
                eW, fW = containerEnergyForce(packing, self.sigma, kContainer = self.kContainer,
                                              mollified = True)

        if self.modelType == "mollified":
            self._requireSigma()
            if onGpu:
                eO, gO = cudaOverlap.plummerOverlapCuda(packing, self.sigma, packing.targetArea,
                                                        packing.targetPerimeter,
                                                        numActive = container)
                eR, fR = cudaOverlap.selfRepulsionCuda(packing, self.kSelf, self.delta)
            else:
                eO, gO = plummerOverlapExact(packing, self.sigma, numActive = container)
                eR, fR = selfRepulsionEnergyForce(packing, self.kSelf, self.delta)
            return eO + eS + eR + eW, -gO.reshape(-1) + fS + fR + fW
        if onGpu:
            eO, fO = cudaOverlap.sharpOverlapCuda(packing, kOverlap = 1.0)
        else:
            eO, fO = sharpOverlapEnergyForce(packing, kOverlap = 1.0)
        eC, fC = containerEnergyForce(packing, self.sigma, kContainer = self.kContainer,
                                      mollified = False) \
            if container is not None else (0.0, np.zeros_like(packing.positions))
        return eO + eS + eC, fO.reshape(-1) + fS + fC

    def _exactForceEnergy(self, packing):
        """``(energy, flat force)`` on the EXACT-ARC law, plus ``_rhoForces``. See ``roundedContact``.

        Both the ``depth`` and ``area`` tiers are wired; the other two are not, and say so rather than
        quietly falling back to chords."""
        if self.modelType not in ("depth", "area"):
            raise NotImplementedError(
                f"exact arcs are wired to the 'depth' and 'area' tiers, not {self.modelType!r}. "
                f"Use setGeometryType('round', exact = False) for the chorded path, which every tier "
                f"accepts because it hands them an ordinary polygon.")
        import roundedContact
        widest = int(np.diff(np.asarray(packing.startIndices, dtype = int)).max())
        useCuda = widest <= roundedContact.CUDA_MAX_CORNERS
        if useCuda:
            try:
                import cudaOverlap
                useCuda = cudaOverlap.isAvailable()
            except ImportError:
                useCuda = False

        if self.modelType == "area":
            driver = (roundedContact.packingAreaEnergyForceCuda if useCuda
                      else roundedContact.packingAreaEnergyForce)
            energy, force, rhoForce = driver(
                packing, self.rho, kOverlap = 1.0, kContainer = self.kContainer)
        else:
            # THE GPU IS TRIED FIRST AND THE FALLBACK IS EXACT, not approximate: the same law, the
            # same partition, computed in numpy. Measured with a container, against numpy: depth 5.5x
            # at N = 11 and 10.3x at N = 26, area 4.5x and 6.7x, agreeing to 1e-12 and 1e-14. The
            # kernel's corner count is a COMPILE-TIME stride, so exceeding it would read past the end
            # of a body rather than fail -- which is why the width is checked on this side.
            if useCuda:
                energy, force, rhoForce = roundedContact.packingEnergyForceCuda(
                    packing, self.rho, self.depthStiffness,
                    wallStiffness = self.depthWallStiffness)
            else:
                energy, force, rhoForce = roundedContact.packingEnergyForce(
                    packing, self.rho, self.depthStiffness,
                    wallStiffness = self.depthWallStiffness)

        kEdge, kArea = self.kEdge, self.kArea
        if self.constraints is not None:
            kEdge = 0.0 if self.constraints.edgeHeld else kEdge
            kArea = 0.0 if self.constraints.area else kArea
        if kEdge == 0.0 and kArea == 0.0:
            springEnergy, springForce = 0.0, np.zeros_like(packing.positions)
        else:
            springEnergy, springForce = eqSoftBodyEnergyForce(packing, kEdge, kArea, relative = True)

        self._rhoForces = rhoForce
        return energy + springEnergy, force.reshape(-1) + springForce

    def _roundForceEnergy(self, packing):
        """``(energy, flat backbone force)`` under geometry ``"round"``, plus ``_rhoForces``.

        The tier runs on the ROUNDED packing, with the model temporarily pointed at it so the neighbor
        list, the container index and the CUDA path all describe the shape actually being measured.
        The shape springs are switched off for that call and applied to the BACKBONE afterwards --
        they constrain the backbone square, not the arcs, and the rounded loop carries dummy targets.

        The pullback is ``J^T`` from ``roundedGeometry``, which complex-steps the same function that
        built the loop. The container block rides through unrounded, so its gradient maps one-to-one."""
        segments = self.arcSegments
        stop = self._activeVertexCount()
        rounded = self.roundedPacking()

        # The geometry is switched to "sharp" for the inner call, not merely the packing swapped: the
        # rounded loop IS the sharp shape as far as the tier is concerned, and leaving the flag set
        # would send this method straight back into itself.
        saved = (self.packing, self.kEdge, self.kArea, self.geometryType)
        self.packing, self.kEdge, self.kArea, self.geometryType = rounded, 0.0, 0.0, "sharp"
        try:
            energy, force = self._forceEnergy(rounded)
        finally:
            self.packing, self.kEdge, self.kArea, self.geometryType = saved

        gradient = -np.asarray(force, dtype = float).reshape(-1, 2)
        arcCount = stop * (segments + 1)
        # The Jacobian is defined over EVERY vertex, so the container's rows are zero-padded rather
        # than sliced out -- prev/next wrap within their own polygon and must stay addressable.
        padded = np.zeros((packing.numVertices, segments + 1, 2))
        padded[:stop] = gradient[:arcCount].reshape(stop, segments + 1, 2)
        sign = roundedGeometry.convexSign(packing.positions, packing.prev, packing.next)
        backbone, rhoGradient = roundedGeometry.roundedJacobianApply(
            packing.positions, packing.prev, packing.next, self.rho, sign, segments, padded)
        backbone[stop:] += gradient[arcCount:]

        kEdge, kArea = self.kEdge, self.kArea
        if self.constraints is not None:
            kEdge = 0.0 if self.constraints.edgeHeld else kEdge
            kArea = 0.0 if self.constraints.area else kArea
        if kEdge == 0.0 and kArea == 0.0:
            springEnergy, springForce = 0.0, np.zeros_like(packing.positions)
        else:
            springEnergy, springForce = eqSoftBodyEnergyForce(packing, kEdge, kArea, relative = True)

        self._rhoForces = -rhoGradient
        return energy + springEnergy, -backbone.reshape(-1) + springForce

    def initForceEnergy(self):
        """Prepare the force/energy calculation for the current model type (validates the config and
        clears the cached force/energy). Call once before ``calcForceEnergy``. Returns self."""
        if self.modelType == "mollified":
            self._requireSigma()
        self._forces = None
        self._energy = None
        return self

    def calcForceEnergy(self):
        """Compute the total force + energy at the current positions and cache them; read with
        ``getForces`` / ``getEnergy``. Under ``setConstraints`` the cached force is the TANGENTIAL
        force: the constraint-normal part is carried by the constraints, so it is not an unbalanced
        force and would only mislead an arrow plot or a residual check. Returns self."""
        e, f = self._forceEnergy(self.packing)
        if self.constraints is not None:
            f = self.constraints.projectVector(self.packing, f)
        f = minimize.applyPins(self.packing, np.array(f, dtype = float))
        self._forces = f.reshape(-1, 2)
        self._energy = e
        self.packing.force[:] = f
        self.packing.energy = e
        return self

    def updateForcesParallel(self):
        """Alias of ``calcForceEnergy`` (compute + cache the total force/energy). Returns self."""
        return self.calcForceEnergy()

    def updateEnergyParallel(self):
        """Compute the total energy and cache it; read with ``getEnergy``. Returns self."""
        self._energy, _ = self._forceEnergy(self.packing)
        return self

    def getForces(self):
        """Cached per-vertex force array, shape (numVertices, 2); None before a calc."""
        return self._forces

    def getEnergy(self):
        """Cached total energy; None before a calc."""
        return self._energy

    def getMaxUnbalancedForce(self):
        """Largest per-vertex net force magnitude max_k |F_k| from the cached forces (call
        ``calcForceEnergy`` or a minimizer first)."""
        if self._forces is None:
            raise ValueError("no forces cached -- call calcForceEnergy() first")
        return float(np.max(np.hypot(self._forces[:, 0], self._forces[:, 1])))

    def energyBreakdown(self):
        """Current energy split into ``{'overlap', 'spring', 'selfRep'}`` (recomputed at the current
        positions). Use it to check that self-repulsion stays a small SAFETY NET rather than a
        load-bearing term -- ``selfRep`` should be orders of magnitude below the others at a good
        relaxed packing."""
        self._requireSigma()
        pk = self.packing
        eO, _ = plummerOverlapExact(pk, self.sigma)
        eS, _ = eqSoftBodyEnergyForce(pk, self.kEdge, self.kArea, relative = True)
        eR, _ = selfRepulsionEnergyForce(pk, self.kSelf, self.delta)
        return {"overlap": eO, "spring": eS, "selfRep": eR}

    @staticmethod
    def _convergenceThreshold(maxUnbalancedForce, fThreshold, default):
        """Resolve the stopping tolerance from either spelling, for EVERY minimizer.

        ``maxUnbalancedForce`` is the preferred name -- it says what the number IS, and matches
        ``getMaxUnbalancedForce()``. ``fThreshold`` is kept as an alias so existing notebooks keep
        working. Every minimizer on the ladder takes both, so a tolerance that works for FIRE can be
        handed to CG or Newton unchanged."""
        threshold = maxUnbalancedForce if maxUnbalancedForce is not None else fThreshold
        # A target under the force noise floor cannot be met by any minimizer or any step budget --
        # repeated evaluations of one configuration disagree at that level. Said HERE, before the
        # first step, because it is the cheapest of all the ways a run burns its whole budget and the
        # only one that no adaptive stopping rule can rescue.
        return minimize.checkReachable(float(default if threshold is None else threshold),
                                       "the requested maxUnbalancedForce")

    # UNVERIFIED(Cam)
    def getStopReason(self):
        """Why the last minimizer stopped, or None if it converged or was never stalled.

        ``'noise'`` -- the residual reached the force noise floor; this IS converged and the tolerance
        was below what the arithmetic can resolve. ``'flat'`` -- the residual stopped moving while far
        above that floor, which is a floor of the ENERGY rather than of the minimizer (a C1 tier has a
        kink no descent passes). ``'slow'`` -- still converging, but too slowly to reach the tolerance
        within any sane budget.

        The three want different responses, which is the whole point of separating them: accept the
        answer, change the tier or the tolerance, or spend more steps."""
        return getattr(self.packing, "stopReason", None)

    def minimizeFIRE(self, maxSteps = 100000, fThreshold = None, dt = 0.0075, dtMax = 0.03,
                     progressBar = False, maxUnbalancedForce = None, **fireKwargs):
        """Relax to equilibrium with FIRE on the current model's overlap energy (robust, the
        workhorse). Returns the number of steps taken; caches the final force/energy (read the energy
        with ``getEnergy`` and the residual with ``getMaxUnbalancedForce``). ``progressBar`` shows a
        live bar. The default step is deliberately gentle: the relative springs are stiff (~1/l0^2)
        and the mollified contact is stiff (~1/sigma^2), so FIRE's usual dtMax = 0.1 diverges (NaN) --
        dtMax = 0.03 is stable here. Pass dt / dtMax to override.

        Under ``useShapeConstraints`` the stiff spring modes are gone from the dynamics (only the
        physical contact stiffness is left), so the gentle default is no longer needed -- raise dtMax
        back toward FIRE's usual 0.1 or beyond, and the residual reported by
        ``getMaxUnbalancedForce`` is the TANGENTIAL force, the true residual on the manifold."""
        threshold = self._convergenceThreshold(maxUnbalancedForce, fThreshold, 1e-8)
        if self.modelType == "mollified":
            self._warnIfSofteningTooSharp()
        energy, steps, _ = minimize.minimizeFIRE(
            self.packing, self._forceEnergy, maxSteps = maxSteps, fThreshold = float(threshold),
            dt = dt, dtMax = dtMax, constraints = self.constraints, progress = progressBar,
            transient = self._transientStep, **fireKwargs)
        self._forces = self.packing.force.reshape(-1, 2).copy()
        self._energy = energy
        self._warnIfSelfRepActive()
        return steps

    def minimizeMovie(self, path, minimizer = "fire", maxSteps = 2000, fThreshold = None,
                      frameEvery = 10, fps = 20, figsize = (6, 6), dpi = 120, forces = False,
                      indicatorColorMap = None, indicatorResolution = 160, progressBar = True,
                      maxUnbalancedForce = None, **minimizerKwargs):
        """Relax while recording a VIDEO of the packing, written to ``path`` (.mp4 or .gif).

        Runs the chosen ``minimizer`` ("fire", "cg", "lbfgs" or "gd") with a frame captured every
        ``frameEvery`` steps, annotated with the step number, energy and max unbalanced force.
        ``forces = True`` overlays the force arrows; ``indicatorColorMap`` shades the mollified
        membership field instead of drawing the polygons (see ``draw``). Returns
        ``(energy, steps, converged)``.

        The frame hook is the minimizers' existing ``callback`` / ``callbackEvery`` mechanism, and the
        run is DELEGATED to this class's own ``minimizeFIRE`` / ``minimizeCG``, so a recorded run
        follows a bit-identical trajectory to the same run without a movie (``tests/movieCheck.py``
        asserts this). Delegating matters: calling ``minimize.minimizeFIRE`` directly would pick up
        that module's defaults instead of the Model's gentler ones -- dt = 0.01 rather than 0.0075 --
        and quietly record a different relaxation than the one the user gets otherwise. Rendering
        dominates the wall time, so keep ``frameEvery`` well above 1.

        Writer: ``.mp4`` needs ffmpeg, ``.gif`` uses pillow. If ffmpeg is missing for an .mp4 this
        falls back to a .gif beside it rather than losing the run."""
        import matplotlib.pyplot as plt
        from matplotlib import animation

        fThreshold = self._convergenceThreshold(maxUnbalancedForce, fThreshold, 1e-9)
        path = str(path)
        extension = os.path.splitext(path)[1].lower()
        if extension == ".mp4" and not animation.writers.is_available("ffmpeg"):
            path = os.path.splitext(path)[0] + ".gif"
            extension = ".gif"
            warnings.warn("\n*** ffmpeg is unavailable; writing a .gif instead ***\n"
                          f"    Movie will be saved to {path}.", stacklevel = 2)
        writer = (animation.PillowWriter(fps = fps) if extension == ".gif"
                  else animation.FFMpegWriter(fps = fps, bitrate = 2400))

        figure, ax = plt.subplots(figsize = figsize)
        state = {"frames": 0}

        def capture(step, energy, force):
            ax.clear()
            self._forces = force.reshape(-1, 2)
            self.draw(ax = ax, forces = self._forces if forces else None,
                      indicatorColorMap = indicatorColorMap,
                      indicatorResolution = indicatorResolution)
            magnitude = float(np.max(np.hypot(self._forces[:, 0], self._forces[:, 1])))
            ax.set_title(f"step {step}    E = {energy:.6g}    max|F| = {magnitude:.2e}",
                         fontsize = 10)
            writer.grab_frame()
            state["frames"] += 1

        if minimizer not in ("fire", "cg", "lbfgs", "gd"):
            raise ValueError(f"unknown minimizer {minimizer!r}; use 'fire', 'cg', 'lbfgs' or 'gd'")

        def run():
            """Delegate to this class's own minimizer so the recorded trajectory matches an
            unrecorded one exactly. Returns the common (energy, steps, converged) triple."""
            hooks = dict(callback = capture, callbackEvery = frameEvery)
            if minimizer == "fire":
                steps = self.minimizeFIRE(maxSteps = maxSteps, fThreshold = fThreshold,
                                          progressBar = progressBar, **hooks, **minimizerKwargs)
                return self._energy, steps, self.getMaxUnbalancedForce() < float(fThreshold)
            if minimizer == "cg":
                return self.minimizeCG(maxSteps = maxSteps, fThreshold = fThreshold,
                                       progressBar = progressBar, **hooks, **minimizerKwargs)
            if minimizer == "lbfgs":
                return self.minimizeLBFGS(maxSteps = maxSteps, fThreshold = fThreshold,
                                          progressBar = progressBar, **hooks, **minimizerKwargs)
            return minimize.minimizeGD(self.packing, self._forceEnergy, maxSteps = maxSteps,
                                       fThreshold = float(fThreshold), progress = progressBar,
                                       **hooks, **minimizerKwargs)

        with writer.saving(figure, path, dpi):
            self.calcForceEnergy()
            capture(0, self._energy, self._forces.reshape(-1))
            energy, steps, converged = run()
            capture(steps, energy, self.packing.force)
        plt.close(figure)

        self._forces = self.packing.force.reshape(-1, 2).copy()
        self._energy = energy
        print(f"wrote {path} ({state['frames']} frames, {state['frames'] / fps:.1f} s at {fps} fps)")
        return energy, steps, converged

    def _transientStep(self, packing, dt):
        """One explicit step of the TARGET degrees of freedom, or a no-op when they are fixed.

        Projects the target force into the moment-constraint tangent space, steps, then restores the
        moments exactly (the projection makes drift O(dt^2), the restore removes even that)."""
        if self.transient is None:
            return
        forces = targetForces(packing, self.kEdge, self.kArea)
        for name in self.transient.families():
            current = self.transient.free(packing, name)
            force = np.asarray(forces[name], dtype = float)[:current.size]
            # NON-DIMENSIONALIZE, as the reference does with <K_A><A0>^2. Without it the target force
            # is ~8000x the position force (dE/dA0 goes as 1/A0^2, and A0 ~ 0.06 while positions are
            # O(1)), so a shared FIRE timestep moves a target by ~15% PER STEP and the run diverges:
            # measured targets running away by 3e5 relative, going negative, moments drifting 8.7e9.
            scale = float(np.mean(current)) ** 2
            step = self.transient.constraints.project(current, force)
            trial = current + dt * scale * step
            # A target must stay positive: a negative one makes the shape terms and the negative
            # moments meaningless, and the moment restore then cannot converge.
            floor = 1e-6 * float(np.mean(current))
            self.transient.setFree(packing, name, np.maximum(trial, floor))
        self.transient.restore(packing)

    def minimizeCG(self, maxSteps = 2000, fThreshold = None, maxUnbalancedForce = None,
                   progressBar = False, **cgKwargs):
        """Polish with nonlinear conjugate gradient (Polak-Ribiere+ / strong-Wolfe line search) -- the
        cheap middle rung between FIRE and Newton. Each step is only a few force evals (NO Hessian),
        so it reaches far tighter than FIRE at a tiny fraction of Newton's cost. Runs on the analytic
        overlap force (~seconds/eval at N=32). Best launched from a FIRE-relaxed configuration. Returns
        (energy, steps, converged); caches the final force/energy.

        Under ``useShapeConstraints`` this runs as Riemannian CG on the constraint manifold (tangent
        direction, SHAKE-retracted line search) -- the polish to reach for instead of Newton, since it
        needs no Hessian and so avoids both the FD cost and the singular null space.

        Takes ``maxUnbalancedForce`` (preferred) or ``fThreshold`` (alias) and ``progressBar``, spelled
        exactly as ``minimizeFIRE`` spells them, so a call can move between the two unchanged.

        DOES NOT MOVE TRANSIENT TARGETS. Only FIRE takes the ``transient`` hook, so under
        ``setDOFType("transient")`` this relaxes the POSITIONS at frozen targets -- a single
        optimization, not the double one. That is not just missing plumbing: the strong-Wolfe line
        search assumes a fixed energy landscape, and retargeting mid-search would invalidate it. Use
        FIRE alone when the targets are meant to co-evolve."""
        threshold = self._convergenceThreshold(maxUnbalancedForce, fThreshold, 1e-9)
        if self.transient is not None:
            warnings.warn(
                "\n*** CG does not move transient targets ***\n"
                "    setDOFType('transient') is active, but only FIRE carries the target step, so this "
                "polish relaxes positions at FROZEN targets. Use minimizeFIRE for a joint relaxation.",
                stacklevel = 2)
        if self.modelType == "mollified":
            self._requireSigma()
            self._warnIfSofteningTooSharp()
        energy, steps, converged = minimize.minimizeCG(
            self.packing, self._forceEnergy, maxSteps = maxSteps, fThreshold = threshold,
            constraints = self.constraints, progress = progressBar, **cgKwargs)
        self._forces = self.packing.force.reshape(-1, 2).copy()
        self._energy = energy
        self._warnIfSelfRepActive()
        return energy, steps, converged

    # UNVERIFIED(Cam)
    def minimizeFireLBFGS(self, maxSteps = 2000, fThreshold = None, maxUnbalancedForce = None,
                          fireSteps = None, fireTolerance = None, coarseness = 1e3,
                          progressBar = False, **lbfgsKwargs):
        """FIRE to get roughly down, then L-BFGS to polish. Returns ``(energy, steps, converged)``.

        The two minimizers fail in opposite places, which is the whole reason to pair them. FIRE is
        damped dynamics: it does not care how rough the landscape is or how bad the starting point is,
        but it converges LINEARLY and stalls at a floor. L-BFGS is superlinear near a minimum, where its
        curvature memory is meaningful -- but far from one that memory describes a landscape the
        iterate has already left, and the strong-Wolfe line search spends its evaluations discovering
        that. Measured on the depth tier, from a 1e-4 perturbation of a minimum L-BFGS reached 9.8e-11
        in 41 iterations where FIRE stalled at 5.5e-6 after 150; the ordering reverses when the start is
        far away.

        ``coarseness`` sets the handoff: FIRE runs to ``coarseness`` times the final tolerance, so the
        default hands over three decades out -- comfortably inside L-BFGS's superlinear regime, and
        early enough that FIRE is not grinding through its linear tail. ``fireTolerance`` overrides that
        with an absolute number, and ``fireSteps`` caps the FIRE leg's step count (default: half of
        ``maxSteps``).

        If FIRE reaches the FINAL tolerance on its own the polish is skipped rather than run to
        discover there is nothing to do.

        Returns ``(energy, steps, converged)`` like ``minimizeLBFGS`` and ``minimizeCG``, NOT the bare
        step count ``minimizeFIRE`` returns -- the family is inconsistent about this and the tuple is
        the more useful of the two shapes, so the leg that finishes decides it."""
        threshold = self._convergenceThreshold(maxUnbalancedForce, fThreshold, 1e-9)
        coarse = float(fireTolerance) if fireTolerance is not None \
            else threshold * float(coarseness)
        # A handoff LOOSER than the target is the point; one tighter would mean FIRE doing the polish,
        # which is what this exists to avoid.
        coarse = max(coarse, threshold)
        fireBudget = int(fireSteps) if fireSteps is not None else max(int(maxSteps) // 2, 1)

        # ``patience`` belongs to BOTH legs, so it is forwarded rather than swallowed by **lbfgsKwargs.
        # The FIRE leg is the one that stalls -- it converges linearly and this pairing exists because
        # of that -- so leaving it without a stall guard would guard the wrong half.
        fireKwargs = {key: lbfgsKwargs[key] for key in ("patience", "stallFactor")
                      if key in lbfgsKwargs}
        steps = int(self.minimizeFIRE(maxUnbalancedForce = coarse, maxSteps = fireBudget,
                                      progressBar = progressBar, **fireKwargs))
        if self.getMaxUnbalancedForce() <= threshold:
            return self.getEnergy(), steps, True
        energy, polishSteps, converged = self.minimizeLBFGS(
            maxUnbalancedForce = threshold, maxSteps = int(maxSteps),
            progressBar = progressBar, **lbfgsKwargs)
        return energy, steps + polishSteps, converged

    def minimizeLBFGS(self, maxSteps = 2000, fThreshold = None, maxUnbalancedForce = None,
                      progressBar = False, **lbfgsKwargs):
        """Polish with limited-memory BFGS -- the same rung as ``minimizeCG`` and, on the contact tiers,
        the faster one. Both cost force evaluations only (NO Hessian), but L-BFGS carries curvature in
        its last ``memory = 10`` (s, y) pairs, so the unit step is usually accepted outright: about one
        force evaluation per step against CG's several on the same objective. Reach for it first when a
        FIRE run has stalled. Returns (energy, steps, converged); caches the final force/energy.

        Under ``useShapeConstraints`` this runs as Riemannian L-BFGS on the constraint manifold, exactly
        as CG does -- tangent direction, SHAKE-retracted line search.

        Takes ``maxUnbalancedForce`` (preferred) or ``fThreshold`` (alias) and ``progressBar``, spelled
        exactly as ``minimizeFIRE`` and ``minimizeCG`` spell them, so a call can move between the three
        unchanged. Extra keywords (``memory``, ``c1``, ``c2``, ...) pass through to the minimizer.

        DOES NOT MOVE TRANSIENT TARGETS, for the same reason CG does not: the strong-Wolfe line search
        assumes a fixed landscape, and retargeting mid-search would invalidate it. Use FIRE alone when
        the targets are meant to co-evolve."""
        threshold = self._convergenceThreshold(maxUnbalancedForce, fThreshold, 1e-9)
        if self.transient is not None:
            warnings.warn(
                "\n*** L-BFGS does not move transient targets ***\n"
                "    setDOFType('transient') is active, but only FIRE carries the target step, so this "
                "polish relaxes positions at FROZEN targets. Use minimizeFIRE for a joint relaxation.",
                stacklevel = 2)
        if self.modelType == "mollified":
            self._requireSigma()
            self._warnIfSofteningTooSharp()
        energy, steps, converged = minimize.minimizeLBFGS(
            self.packing, self._forceEnergy, maxSteps = maxSteps, fThreshold = threshold,
            constraints = self.constraints, progress = progressBar, **lbfgsKwargs)
        self._forces = self.packing.force.reshape(-1, 2).copy()
        self._energy = energy
        self._warnIfSelfRepActive()
        return energy, steps, converged

    def minimizeNewton(self, fThreshold = None, maxSteps = 40, maxUnbalancedForce = None,
                       progressBar = False, **newtonKwargs):
        """Polish toward the numerical floor with Newton's method. Runs on the analytic overlap force,
        which is self-consistent with its energy to the machine floor -- exactly what Newton needs;
        the near-parallel bridge (``energies.py`` ``_ceBridge``/``_wBridge``) removes the 1/X1^2
        cancellation, and Newton reaches ~9e-13 at N=6 in a few steps. Pass ``hessian=`` a callable
        ``hessian(packing)`` (e.g. built on ``energies.plummerOverlapHessian``) to use the analytic
        Hessian; otherwise a FD Hessian (2*(2N) force evals per step) is used, practical at small N.
        Best launched from a FIRE-relaxed configuration. Takes ``maxUnbalancedForce`` (preferred) or
        ``fThreshold`` (alias), either as a number or a numeric string, plus ``progressBar`` -- spelled
        as ``minimizeFIRE`` spells them. Returns (energy, steps, converged)."""
        threshold = self._convergenceThreshold(maxUnbalancedForce, fThreshold, 1e-12)
        if self.modelType == "mollified":
            self._warnIfSofteningTooSharp()
        energy, steps, converged = minimize.minimizeNewton(
            self.packing, self._forceEnergy, maxSteps = maxSteps, fThreshold = threshold,
            constraints = self.constraints, progress = progressBar, **newtonKwargs)
        self._forces = self.packing.force.reshape(-1, 2).copy()
        self._energy = energy
        self._warnIfSelfRepActive()
        return energy, steps, converged

    def draw(self, ax = None, forces = None, figsize = (6, 6), facecolor = "#9ec7ff",
             edgecolor = "#264d8c", indicatorColorMap = None, indicatorResolution = 160,
             colorBy = None, colorMap = "Blues", colorLabel = None, colorLimits = None,
             edgeMask = None, edgeColors = ("#1f77b4", "#d62728"), arcColor = "#eb6834",
             showBackbone = True, drawSegments = 48):
        """Draw the polygons, framed according to the BOUNDARY CONDITIONS.

        ``periodic``: filled and tiled across the unit cell [0,1]^2 -- each polygon is drawn at its
        own position and at the 8 neighboring periodic images, clipped to the cell, so a shape
        straddling a boundary reappears on the opposite side.

        ``free``: no images and no wrapping, so each polygon is drawn exactly once and the view is
        widened to whatever the packing actually occupies (with a 5% margin). Tiling here would show
        copies that do not exist, and for the indicator field it would be actively misleading -- the
        images of box-sized shapes tile the plane, which makes Psi read the same everywhere and hides
        the vacuum outside the packing.

        With ``forces`` (an (numVertices, 2) array, e.g. from ``getForces``) overlay per-vertex force
        arrows (tiled to match).

        With ``indicatorColorMap`` (a matplotlib colormap name), shade the background by the mollified
        softened-membership field Psi(x) = sum_polygons (K_sigma * 1_polygon)(x) -> 1 deep inside a
        shape, 0 far outside, blurred over sigma -- so you can see how aggressive the mollification
        width is. Requires the mollified model (``setMollification``); the polygon patches are omitted
        so only the field (and any forces) show.

        With ``colorBy``, shade each polygon by a per-polygon value and add a colorbar. Pass
        ``Model.shapeIndex`` for the shape ratio ``P / sqrt(A)`` -- 4 for a square, ~3.72 for a regular
        hexagon, larger the more distorted -- or any array of length numPolygons. The default ramp is
        SEQUENTIAL and single-hue (light to dark): the quantity is a magnitude with an order and no
        meaningful midpoint, so a multi-hue or rainbow map would imply categories that are not there.
        ``colorLimits`` pins the range, which is what makes a sequence of frames comparable -- with
        autoscaling every frame re-normalizes and the colors stop meaning the same thing.

        With ``edgeMask`` (a per-EDGE bool, e.g. ``getAlternatingMask()``) each edge is stroked in the
        first of ``edgeColors`` where the mask is True and the second where it is False. That is a
        CATEGORICAL split -- two named populations, not a magnitude -- so it takes two fixed hues and
        no colorbar.

        UNDER ROUND GEOMETRY the ROUNDED loop is drawn, not the backbone, because that is the shape
        every law sees -- the backbone squares of a valid rounded packing routinely overlap, since the
        corners that would have collided were cut away. The two kinds of boundary are separated:
        ``arcColor`` strokes the corner arcs and ``edgecolor`` the straight runs between kiss points,
        which is a CATEGORICAL split (a piece of boundary is one or the other) and so takes two fixed
        hues. ``showBackbone`` overlays the sharp backbone dashed, so what ``rho`` removed is visible
        as the gap between the dashed corner and the arc. A user-supplied ``edgeMask`` cannot be
        honored there -- it is one entry per BACKBONE edge and the drawn edges are arc chords -- so it
        warns and is ignored.

        Returns the axes."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon, PathPatch as MplPathPatch
        from matplotlib.path import Path as MplPath
        from matplotlib.collections import PatchCollection

        # DRAW WHAT IS BEING SIMULATED. Under round geometry the backbone is the state but the ROUNDED
        # loop is the shape every law sees, and the two look materially different: the backbone squares
        # of a valid rounded packing routinely overlap, because the corners that would have collided
        # were cut away. Drawing the backbone would show a picture that contradicts getOverlapArea.
        if self.getGeometryType() == "round":
            # Which drawn edges are ARC and which are the straight runs between kiss points. Corner k
            # owns arc points k*(S+1) .. k*(S+1)+S, so every edge inside that block is a chord of the
            # arc and the one LEAVING its last point is the straight run to the next corner's a^-.
            # THE PICTURE SAMPLES THE TRUE ARC, not the chords the tier is handed: `drawSegments`
            # is independent of `arcSegments`, so what is on screen is the shape the model MEANS
            # rather than its discretization. Say drawSegments = arcSegments to see what is actually
            # being simulated -- the difference is 1.2% of the corner radius at arcSegments = 5.
            saved = self.arcSegments
            self.arcSegments = max(int(drawSegments), 1)
            span = self.arcSegments + 1
            rounded = self.roundedPacking()
            isArc = np.zeros(rounded.numVertices, dtype = bool)
            arcCount = self._activeVertexCount() * span
            isArc[:arcCount] = True
            isArc[span - 1:arcCount:span] = False
            with self.measuredGeometry():
                ax = self.draw(ax = ax, forces = None, figsize = figsize, facecolor = facecolor,
                               edgecolor = edgecolor, indicatorColorMap = indicatorColorMap,
                               indicatorResolution = indicatorResolution, colorBy = colorBy,
                               colorMap = colorMap, colorLabel = colorLabel,
                               colorLimits = colorLimits, edgeMask = isArc,
                               edgeColors = (arcColor, edgecolor), showBackbone = False)
            self.arcSegments = saved
            self._roundedCache = None
            # The backbone is the STATE, so it is worth seeing even though no law is applied to it --
            # dashed and recessive, because it is the thing rho is measured from rather than the thing
            # being packed. Its corners stick out past the arcs by exactly what was cut away.
            if showBackbone:
                r = self.packing.positions.reshape(-1, 2)
                starts = self.packing.startIndices
                for polygon in range(self._containerSlot()):
                    loop = r[int(starts[polygon]):int(starts[polygon + 1])]
                    closed = np.vstack([loop, loop[:1]])
                    ax.plot(closed[:, 0], closed[:, 1], linestyle = "--", linewidth = 0.9,
                            color = "#7f8c8d", alpha = 0.9, zorder = 5)
            # Forces and edge masks are indexed by BACKBONE vertex and have no counterpart on the
            # arc-chorded loop, so they are overlaid here rather than handed to the call above.
            if forces is not None:
                r = self.packing.positions.reshape(-1, 2)
                f = np.asarray(forces).reshape(-1, 2)
                offsets = ([(dx, dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)]
                           if self.packing.box is not None else [(0.0, 0.0)])
                base = np.concatenate([r + shift for shift in offsets])
                tiled = np.tile(f, (len(offsets), 1))
                ax.quiver(base[:, 0], base[:, 1], tiled[:, 0], tiled[:, 1], color = "#c0392b",
                          angles = "xy", scale_units = "xy")
            if edgeMask is not None:
                warnings.warn(
                    "\n*** edgeMask is ignored under round geometry ***\n"
                    "    The mask is one entry per BACKBONE edge, and the drawn loop is the rounded "
                    "one, whose edges are arc chords with no correspondence to it. Draw the mask "
                    "under setGeometryType('sharp'), or color whole polygons with colorBy.",
                    stacklevel = 2)
            return ax

        if ax is None:
            _, ax = plt.subplots(figsize = figsize)
        r = self.packing.positions.reshape(-1, 2)
        starts = self.packing.startIndices
        # A CONTAINER is drawn as its OUTSIDE, not its inside: filling it would blanket the whole
        # figure, and its membership field is the complement of what confines the shapes. It is split
        # out here and rendered as a wall outline / an exterior field below.
        container = self.packing.containerIndex
        shapeCount = self.packing.numPolygons if container is None else container
        loops = [r[int(starts[p]):int(starts[p + 1])] for p in range(shapeCount)]
        containerLoop = (None if container is None
                         else r[int(starts[container]):int(starts[container + 1])])

        # Boundary-aware framing. PERIODIC: tile the 3x3 images and clip to the unit cell, so a shape
        # straddling a boundary reappears opposite. FREE: there are no images to draw and nothing
        # wraps, so tiling would show copies that do not exist -- draw each polygon once and widen the
        # view to whatever the packing actually occupies (in free space it may drift anywhere).
        periodic = self.packing.box is not None
        if periodic:
            images = [(dx, dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)]
            xLo, xHi = 0.0, 1.0
            yLo, yHi = 0.0, 1.0
        else:
            images = [(0.0, 0.0)]
            # Union the packing's extent with the FRAME OF REFERENCE, so a drifted packing reads as
            # drifted rather than as re-centered. That frame is the CONTAINER when there is one and the
            # unit cell otherwise: with the box carrying a scale degree of freedom (``scaleBox``,
            # ``compressToJamming``) it is no longer the unit square, and framing to [0,1] would let a
            # compressed wall walk out of view -- or, worse, sit inside the frame looking unchanged
            # while the packing around it shrank.
            if containerLoop is not None:
                reference = np.vstack([containerLoop.min(axis = 0), containerLoop.max(axis = 0)])
            else:
                reference = np.array([[0.0, 0.0], [1.0, 1.0]])
            lo = np.minimum(r.min(axis = 0), reference[0])
            hi = np.maximum(r.max(axis = 0), reference[1])
            span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-9))
            pad = 0.05 * span
            centerX = 0.5 * (lo[0] + hi[0])
            centerY = 0.5 * (lo[1] + hi[1])
            half = 0.5 * span + pad
            xLo, xHi = centerX - half, centerX + half
            yLo, yHi = centerY - half, centerY + half

        if indicatorColorMap is not None:
            self._requireSigma()
            gX = np.linspace(xLo, xHi, indicatorResolution)
            gY = np.linspace(yLo, yHi, indicatorResolution)
            gx, gy = np.meshgrid(gX, gY)
            # The field is the slowest thing in an interactive session by a wide margin -- at the
            # default resolution the numpy path is ~10.5 s per frame against 33 ms for a plain draw
            # (N * 9 * resolution^2 * n arctangents) -- so it goes to the GPU when one is present.
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            if cudaOverlap is not None and cudaOverlap.isAvailable():
                field = cudaOverlap.plummerMeasureGridCuda(
                    self.packing, gx.ravel(), gy.ravel(), self.sigma,
                    polygons = None if container is None else (0, shapeCount))
            else:
                field = np.zeros(pts.shape[0])
                for loop in loops:
                    for dx, dy in images:
                        field += plummerMeasure(pts, loop + np.array([dx, dy]), self.sigma)
            if containerLoop is not None:
                # The wall contributes the indicator of its EXTERIOR, 1 + sign * Psi_C: zero inside,
                # one outside. That is exactly the region the confinement energy penalises, so the
                # picture shows what the wall forbids rather than a slab covering the whole cell.
                from energies import containerOrientationSign
                sign = containerOrientationSign(self.packing, container)
                if cudaOverlap is not None and cudaOverlap.isAvailable():
                    psiC = cudaOverlap.plummerMeasureGridCuda(
                        self.packing, gx.ravel(), gy.ravel(), self.sigma,
                        polygons = (container, container + 1))
                else:
                    psiC = plummerMeasure(pts, containerLoop, self.sigma)
                field = field + (1.0 + sign * psiC)
            im = ax.imshow(field.reshape(indicatorResolution, indicatorResolution),
                           extent = (xLo, xHi, yLo, yHi), origin = "lower",
                           cmap = indicatorColorMap, vmin = 0.0)
            ax.figure.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04, label = r"$\Psi_\sigma$")
        elif colorBy is not None:
            # PER-POLYGON MAGNITUDE, so a SEQUENTIAL single-hue ramp light -> dark, never a rainbow:
            # the quantity has an order and no meaningful midpoint, and hue-cycling would imply
            # categories that are not there. The colorbar is the legend -- no per-polygon numbers.
            values = (self.getShapeIndices()[:shapeCount] if colorBy is _SHAPE_INDEX
                      else np.asarray(colorBy, dtype = float).ravel()[:shapeCount])
            low, high = (float(np.min(values)), float(np.max(values))) if colorLimits is None \
                else (float(colorLimits[0]), float(colorLimits[1]))
            if high - low < 1e-12:
                high = low + 1e-12                      # a flat field still needs a finite range
            patches = [MplPolygon(loop + shift, closed = True) for shift in images for loop in loops]
            collection = PatchCollection(patches, cmap = colorMap, edgecolor = edgecolor,
                                         alpha = 0.85, linewidths = 1.0)
            collection.set_array(np.tile(values, len(images)))
            collection.set_clim(low, high)
            ax.add_collection(collection)
            ax.figure.colorbar(collection, ax = ax, fraction = 0.046, pad = 0.04,
                               label = colorLabel)
        else:
            patches = [MplPolygon(loop + shift, closed = True) for shift in images for loop in loops]
            ax.add_collection(PatchCollection(patches, facecolor = facecolor, edgecolor = edgecolor,
                                              alpha = 0.55, linewidths = 1.0))
        if containerLoop is not None and indicatorColorMap is None:
            # Draw the wall as SOLID MATERIAL: the view rectangle with the container punched out of
            # it, filled in the same style as a polygon. That is the honest picture -- the container
            # is a void the shapes live in, and everything beyond it is forbidden. Filling the
            # container itself would blanket the figure and invert the meaning.
            outer = np.array([[xLo, yLo], [xHi, yLo], [xHi, yHi], [xLo, yHi]])
            inner = np.asarray(containerLoop, dtype = float)
            # The hole needs the opposite winding from the ring (nonzero fill rule). ``outer`` is
            # built CCW, so the container is reversed when it is also CCW.
            signed = 0.5 * np.sum(inner[:, 0] * np.roll(inner[:, 1], -1)
                                  - np.roll(inner[:, 0], -1) * inner[:, 1])
            if signed > 0.0:
                inner = inner[::-1]
            vertices = np.vstack([outer, outer[:1], inner, inner[:1]])
            codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(outer) - 1) + [MplPath.CLOSEPOLY]
                     + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner) - 1) + [MplPath.CLOSEPOLY])
            ax.add_patch(MplPathPatch(MplPath(vertices, codes), facecolor = facecolor,
                                      edgecolor = edgecolor, alpha = 0.55, linewidth = 1.0,
                                      zorder = 1))
        elif not periodic and containerLoop is None:
            # Outline the unit cell so the free-space view keeps a reference scale. Without it an
            # auto-fitted view is unreadable: a packing that has drifted or expanded looks identical
            # to one that has not, because the axes rescale with it.
            ax.plot([0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0],
                    linestyle = "--", linewidth = 1.0, color = "#7f8c8d", alpha = 0.8, zorder = 5)

        if edgeMask is not None:
            from matplotlib.collections import LineCollection
            mask = np.asarray(edgeMask, dtype = bool)
            segments, colors = [], []
            for polygon in range(shapeCount):
                a, b = int(starts[polygon]), int(starts[polygon + 1])
                loop = r[a : b]
                following = np.roll(loop, -1, axis = 0)
                for dx, dy in images:
                    offset = np.array([dx, dy])
                    for k in range(loop.shape[0]):
                        segments.append([loop[k] + offset, following[k] + offset])
                        colors.append(edgeColors[0] if mask[a + k] else edgeColors[1])
            ax.add_collection(LineCollection(segments, colors = colors, linewidths = 1.6,
                                             zorder = 4))

        if forces is not None:
            f = np.asarray(forces).reshape(-1, 2)
            base = np.concatenate([r + shift for shift in images])
            ftile = np.tile(f, (len(images), 1))
            ax.quiver(base[:, 0], base[:, 1], ftile[:, 0], ftile[:, 1], color = "#c0392b",
                      angles = "xy", scale_units = "xy")
        ax.set_xlim(xLo, xHi); ax.set_ylim(yLo, yHi); ax.set_aspect("equal")
        ax.set_axis_off()
        return ax

    def shapeIndices(self):
        """Realized shape index (perimeter / sqrt(area)) of each polygon."""
        return shapeIndices(self.packing)

    def save(self, path):
        """Save the packing (geometry + targets) to ``path`` (npz); reload with ``Model.load``."""
        savePacking(self.packing, path)
        return self

    @classmethod
    def load(cls, path):
        """Rebuild a Model from a ``save`` npz (geometry restored; RNG reseeded fresh).

        Built through ``__init__`` rather than ``__new__`` + a hand-copied attribute list, because that
        list DRIFTED: it had fallen five attributes behind (``transient``, ``dofType``,
        ``boundaryConditions``, ``moments``, ``selfRepFraction``), and a loaded model died inside FIRE
        with ``'Model' object has no attribute 'transient'``. Going through the constructor means every
        default is set in exactly one place and cannot fall behind again.

        What is NOT restored: the CONTAINER and the PINS, which live on the packing but are not in the
        npz (see ``savePacking``), and the constraints / mollification, which are choices rather than
        state. A saved packing with a wall therefore reloads as an ordinary free packing whose phi
        includes the wall's own negative signed area -- measured -0.334 for a packing saved at 0.6657.
        Re-apply ``pinVertices`` and ``setBoundaryConditions('fixed')`` after loading one."""
        packing, _ = loadPacking(path)
        counts = np.diff(packing.startIndices)
        n = int(counts[0]) if np.all(counts == counts[0]) else None
        model = cls(packing.numPolygons, n)
        model.packing = packing
        model.rho = packing.rho
        return model
