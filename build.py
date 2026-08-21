"""Build a packing of N equilateral backbone polygons, plus the area distribution.

Given a target shape index s = perimeter / sqrt(area), an area distribution, and a packing
fraction phi, this samples per-polygon areas, sets each polygon's eqSoftBody targets, seeds a
random star, and relaxes it to equilateral in FREE SPACE (eqSoftBody is per-polygon and intrinsic).
Placing the shaped polygons in the periodic cell and resolving overlaps is the packing phase.
(Merged from packingBuilder.py + distributions.py.)
"""

import warnings

import numpy as np

from enums import EnergyType, PackingType
from packing import Packing, Box
from softBody import eqSoftBodyEnergyForce, backboneEdgeLengths, backboneArea
from minimize import minimizeFIRE


# ---------------------------------------------------------------------------
# Area distribution + rng (merged from distributions.py)
# ---------------------------------------------------------------------------

def asRng(rng):
    """Coerce None / int seed / Generator into a numpy Generator."""
    if rng is None or isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    return rng

def sampleAreas(N, kind = "logNormal", phi = 1.0, cellArea = 1.0, rng = None,
                sigma = 0.2, sizeRatio = 1.4, fractionLarge = 0.5):
    """Return N polygon areas (np.ndarray) that sum to ``phi * cellArea``.

    logNormal:  raw_i ~ Lognormal(mean=0, sigma), then rescaled to the target sum.
    biDisperse: ``fractionLarge`` of the polygons get area ``sizeRatio**2``, the rest
                area 1, shuffled, then rescaled to the target sum.
    """
    rng = asRng(rng)
    if kind == "logNormal":
        raw = rng.lognormal(mean = 0.0, sigma = sigma, size = N)
    elif kind == "biDisperse":
        nLarge = int(round(fractionLarge * N))
        raw = np.ones(N)
        raw[:nLarge] = sizeRatio ** 2
    elif kind == "mono":
        raw = np.ones(N)
    else:
        raise ValueError(f"unknown area distribution: {kind!r}")
    return raw / raw.sum() * (phi * cellArea)


# ---------------------------------------------------------------------------
# Equilateral backbone build (from packingBuilder.py)
# ---------------------------------------------------------------------------

def regularShapeIndex(n):
    """Shape index (perimeter/sqrt(area)) of the regular n-gon -- the minimum a convex
    or non-convex n-gon can reach (nothing is more compact than regular)."""
    return 2.0 * np.sqrt(n * np.tan(np.pi / n))

# UNVERIFIED(Cam)
def cyclicArea(edgeLengths):
    """The LARGEST area a closed polygon with these edge lengths can enclose.

    That polygon is the cyclic one -- all vertices on a common circle -- which is the classical
    answer to the isoperimetric problem at fixed side lengths. With equal sides it is the regular
    n-gon, so this generalizes the quantity behind ``regularShapeIndex`` to unequal edges.

    Solved for the circumradius R. Edge ``a`` subtends a half-angle ``theta = asin(a / 2R)`` at the
    center, and the central angles must close: ``sum 2 theta_i = 2 pi`` when the center lies INSIDE.
    When one edge is long enough that they cannot (its own half-angle would have to exceed pi/2), the
    center falls outside and that edge subtends the reflex arc instead, which flips its sign in both
    the closure condition and the area. Both branches are monotone in R, so a bisection is enough.

    Raises when the edges cannot close at all -- the longest at least as long as the rest together."""
    a = np.asarray(edgeLengths, dtype = float)
    if a.size < 3:
        raise ValueError(f"a polygon needs at least 3 edges, got {a.size}")
    if np.any(a <= 0.0):
        raise ValueError("edge lengths must be positive")
    biggest = int(np.argmax(a))
    rest = float(a.sum() - a[biggest])
    if a[biggest] >= rest:
        raise ValueError(
            f"no closed polygon has these edge lengths: the longest is {a[biggest]:.6g} and the "
            f"other {a.size - 1} together are only {rest:.6g}")
    halfAngle = lambda R: np.arcsin(np.clip(a / (2.0 * R), -1.0, 1.0))
    lo = 0.5 * float(a[biggest])
    inside = float(2.0 * halfAngle(lo).sum()) >= 2.0 * np.pi
    sign = np.ones(a.size)
    if not inside:
        sign[:] = -1.0
        sign[biggest] = 1.0
    closure = 0.0 if not inside else 2.0 * np.pi
    residual = lambda R: float(2.0 * (sign * halfAngle(R)).sum()) - closure
    hi = lo
    while residual(hi) > 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    R = 0.5 * (lo + hi)
    weight = np.ones(a.size)
    if not inside:
        weight[biggest] = -1.0
    return float(0.5 * R * R * (weight * np.sin(2.0 * halfAngle(R))).sum())

# UNVERIFIED(Cam)
def minShapeIndex(edgeLengths):
    """The smallest shape index ``perimeter / sqrt(area)`` these edge lengths admit.

    The floor a target set has to clear, and the generalization of ``regularShapeIndex`` away from
    equal edges: area is bounded above by ``cyclicArea`` at fixed perimeter, so the shape index is
    bounded BELOW. Unequal edges raise that floor -- a spread of edge lengths costs enclosed area --
    which is what limits how far the edges can be spread at a fixed shape index."""
    a = np.asarray(edgeLengths, dtype = float)
    return float(a.sum() / np.sqrt(cyclicArea(a)))

def equilateralTargets(areas, n, shapeIndex):
    """Per-polygon (targetEdgeLength, targetArea, targetPerimeter) realizing shape index
    s = perimeter/sqrt(area): targetPerimeter = s*sqrt(area), targetEdgeLength = P/n.
    Requires s >= regularShapeIndex(n)."""
    areas = np.asarray(areas, dtype = float)
    sMin = regularShapeIndex(n)
    if shapeIndex < sMin - 1e-12:
        raise ValueError(
            f"shapeIndex {shapeIndex} below the regular minimum {sMin:.4f} for n={n}"
        )
    targetPerimeter = shapeIndex * np.sqrt(areas)
    # PER EDGE: every edge of a polygon starts at the same target, but they are independent from here
    # on -- nothing downstream assumes they stay equal.
    targetEdgeLength = np.repeat(targetPerimeter / n, n)
    return targetEdgeLength, areas.copy(), targetPerimeter

# UNVERIFIED(Cam)
def polydisperseTargets(areas, n, shapeIndex, polydispersity, rng = None):
    """``equilateralTargets`` with each polygon's n EDGE targets drawn log-normally rather than equal.

    Every polygon keeps EXACTLY ``shapeIndex``: the draw is normalized to unit mean WITHIN each polygon,
    so its perimeter is untouched, and the area target never enters the draw at all. Only the shape
    changes -- the polygons stop being equilateral while staying the same size and the same
    ``P / sqrt(A)``.

    The point is to make the edge-length spread a REAL degree of freedom. An equilateral build has no
    within-polygon spread at all, so the pooled edge CV is entirely inherited from the size distribution
    and sits exactly on the floor that fixed areas impose (``anneal._reachableWidth``); an anneal that
    drives it toward zero is asking for something it already has. Seeded here, the spread has a
    component that a ramp can actually withdraw, and withdrawing it is what makes the polygons regular.

    ``polydispersity`` is the coefficient of variation of the draw. The realized within-polygon CV comes
    out slightly lower, by about ``sqrt(1 - 1/n)``, because normalizing to a fixed perimeter removes one
    degree of freedom -- 1.6% low at n = 32, 13% at n = 4. Measure it rather than assuming it.

    Refuses a spread the shape index cannot support. Unequal edges enclose less area than equal ones of
    the same total (``cyclicArea``), so they raise the floor ``minShapeIndex``; past the point where that
    floor reaches ``shapeIndex`` the targets describe a polygon that does not exist, and the springs
    would answer with the nearest thing that does rather than failing."""
    areas = np.asarray(areas, dtype = float)
    targetEdgeLength, targetArea, targetPerimeter = equilateralTargets(areas, n, shapeIndex)
    if polydispersity <= 0.0:
        return targetEdgeLength, targetArea, targetPerimeter
    draw = _logNormalDraw(areas.size * n, polydispersity, rng).reshape(-1, n)
    draw /= draw.mean(axis = 1, keepdims = True)
    edges = targetEdgeLength.reshape(-1, n) * draw
    floor = np.array([minShapeIndex(e) for e in edges])
    if np.any(floor > shapeIndex):
        worst = int(np.argmax(floor))
        raise ValueError(
            f"edge polydispersity {polydispersity} is too wide for shape index {shapeIndex} at "
            f"n = {n}: {int((floor > shapeIndex).sum())} of {areas.size} polygons drew edges whose "
            f"most compact closed form is only {floor[worst]:.4f} (polygon {worst}). Unequal edges "
            f"enclose less area than equal ones, so they raise the floor -- lower the polydispersity, "
            f"raise the shape index, or reseed.")
    return edges.reshape(-1), targetArea, targetPerimeter

def starSeed(n, circumradius, center, rng):
    """One random star polygon: n CCW points scaled to ~circumradius, about ``center``."""
    pts = rng.random((n, 2)) - 0.5
    pts = pts[np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))]
    pts = pts / np.max(np.sqrt((pts ** 2).sum(axis = 1))) * circumradius
    return pts + np.asarray(center, dtype = float)

# UNVERIFIED(Cam)  -- edgePolydispersity + the seed-radius indexing fix
def buildEquilateralPacking(N, n, shapeIndex, areaKind = "logNormal", phi = 1.0,
                            rng = None, rho = None, edgePolydispersity = 0.0, **distKwargs):
    """Free-space Packing of N polygons (n vertices each) with eqSoftBody targets set for
    ``shapeIndex`` and the area distribution (areas sum to phi). Seeds are random stars
    sized to each polygon's target circumradius and placed at random centers in [0,1)^2;
    relax with ``shapeBackbones`` to make them equilateral. box is None (shaping context).
    ``rho`` (corner radius) is stored on the packing when given.

    ``edgePolydispersity`` > 0 draws each polygon's edge targets log-normally instead of equal, at the
    same size and the same shape index -- see ``polydisperseTargets``. The seeds and the relaxation are
    unchanged; only the targets they are relaxing toward differ.
    """
    rng = asRng(rng)
    areas = sampleAreas(N, areaKind, phi = phi, rng = rng, **distKwargs)
    targetEdgeLength, targetArea, targetPerimeter = polydisperseTargets(
        areas, n, shapeIndex, edgePolydispersity, rng)
    blocks = []
    startIndices = [0]
    for i in range(N):
        # targetPerimeter is PER POLYGON; targetEdgeLength is per EDGE, so indexing the latter by i
        # sizes every seed from the first polygon's first few edges instead of from its own. Harmless
        # while the areas are monodisperse and every edge is equal, wrong the moment either varies.
        circumradius = targetPerimeter[i] / n / (2.0 * np.sin(np.pi / n))
        blocks.append(starSeed(n, circumradius, rng.random(2), rng))
        startIndices.append(startIndices[-1] + n)
    positions = np.vstack(blocks).reshape(-1)
    return Packing(positions, startIndices, box = None,
                   energyType = EnergyType.eqSoftBody, rho = rho,
                   targetEdgeLength = targetEdgeLength, targetArea = targetArea,
                   targetPerimeter = targetPerimeter)

def shapeBackbones(packing, kEdge = 1.0, kArea = 1.0, **fireKwargs):
    """Relax every polygon to its eqSoftBody equilateral target (free space). Quiet (no progress bar)
    by default -- this is a one-time preprocessing step, not a user-facing minimize."""
    fe = lambda p: eqSoftBodyEnergyForce(p, kEdge, kArea)
    fireKwargs.setdefault("progress", False)
    return minimizeFIRE(packing, fe, **fireKwargs)

def shapeIndices(packing):
    """Realized shape index (perimeter / sqrt(area)) of each polygon (an (P,) array)."""
    edges = backboneEdgeLengths(packing)
    perimeter = np.bincount(packing.shapeId, weights = edges,
                            minlength = packing.numPolygons)
    area = backboneArea(packing)
    return perimeter / np.sqrt(np.abs(area))

def shrinkTargets(targetEdgeLength, targetPerimeter, fromPhi, toPhi):
    """Scale linear targets (edge length, perimeter) for a packing-fraction change
    phi -> toPhi at fixed shape index. Lengths scale as sqrt(toPhi/fromPhi); areas scale
    as toPhi/fromPhi. The first half of Cam's "shrink to minimize, then reset" protocol;
    the collision-minimization between shrink and reset arrives with build steps 6-7.
    """
    s = np.sqrt(toPhi / fromPhi)
    return targetEdgeLength * s, targetPerimeter * s

def generateEquilateralRPs(N, n, phi, kappa, rng = None, maxSteps = 200000):
    """Generate N monodisperse equilateral rounded polygons in a periodic unit box: n vertices
    each, shape index ``kappa``, area phi/N each, and rho = 0.1/N. Each polygon is seeded as a
    random star and relaxed to equilateral by eqSoftBody/FIRE (free space), then placed in the
    square periodic box with rho stored on the packing. Warns when phi is large enough that a
    polygon spans more than half the box, where the overlap machinery's single-image periodic
    assumption breaks down. ``rng`` (int seed or Generator) seeds the randomness. Returns the
    packing.
    """
    rho = 0.1 / N
    packing = buildEquilateralPacking(N, n, kappa, areaKind = "mono", phi = phi, rho = rho, rng = rng)
    shapeBackbones(packing, maxSteps = maxSteps)
    packing.box = Box(PackingType.square)
    _warnLargePhi(packing, n)
    _warnLargeRho(packing, n)
    return packing

# UNVERIFIED(Cam)  -- now measured from the geometry
def _warnLargePhi(packing, n):
    """Warn if the polygons are large enough relative to the unit box that the single-image
    periodic minimum-image assumption used by the overlap machinery may fail -- taken as a
    polygon's rounded diameter exceeding half the box.

    Measured from the GEOMETRY, which exists by the time this runs. Estimating the circumradius from the
    longest edge target instead reads the spread as size: at ``edgePolydispersity = 0.3`` it claimed a
    32-gon spanned 0.91 of the box when the polygon had not grown at all, only one of its edges."""
    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    circumradius = max(
        float(np.sqrt(((r[a : b] - r[a : b].mean(axis = 0)) ** 2).sum(axis = 1)).max())
        for a, b in zip(starts[:-1], starts[1:]))
    reach = 2.0 * (circumradius + packing.rho)
    if reach > 0.5:
        warnings.warn(
            f"phi is large: a polygon spans ~{reach:.2f} of the unit box (> 0.5), so the "
            f"single-image periodic assumption in the overlap machinery may be violated; "
            f"reduce phi.",
            stacklevel = 3)

def _warnLargeRho(packing, n):
    """Warn if rho is large enough that the corner roundings nearly fill the edges -- the kiss
    offsets t = rho/tan(pi/n) leave little straight run, so the rounded geometry degenerates and
    approaches the infeasible limit (the two offsets on an edge meeting)."""
    t = packing.rho / np.tan(np.pi / n)
    edge = float(packing.targetEdgeLength.min())
    straightFraction = (edge - 2.0 * t) / edge
    if straightFraction < 0.2:
        warnings.warn(
            f"rho is large: the corner offsets leave only {100 * straightFraction:.0f}% of the "
            f"shortest edge as straight run; the rounded geometry is near-degenerate. reduce rho.",
            stacklevel = 3)

def _rescaleToAreas(packing, newArea):
    """Scale every polygon about its own centroid so its target area becomes ``newArea[p]``, updating
    the area / edge-length / perimeter targets in place. Scaling about the centroid preserves the
    equilateral shape and the shape index, so only the SIZE changes. Returns the packing."""
    scale = np.sqrt(newArea / packing.targetArea)
    r = packing.positions.reshape(-1, 2)
    for p in range(packing.numPolygons):
        a = int(packing.startIndices[p])
        b = int(packing.startIndices[p + 1])
        centroid = r[a : b].mean(axis = 0)
        r[a : b] = centroid + scale[p] * (r[a : b] - centroid)
    packing.targetArea = newArea
    packing.targetEdgeLength = packing.targetEdgeLength * scale[packing.shapeId]
    # THE DIAGONAL TARGETS ARE LENGTHS TOO and must ride the same scale. Missing them leaves a shape
    # template describing edges at the new size and turning angles at the old one, which is not a
    # shape at all: measured, resizing after setShapeTemplate drove the constraint residual to 9.4 and
    # the shape index to 5.14 where the template asked for 4.0. Every size operation reaches this
    # function, so it is the one place the two families can be kept in step.
    if getattr(packing, "targetDiagonal", None) is not None:
        packing.targetDiagonal = packing.targetDiagonal * scale[packing.shapeId]
    packing.syncTargetPerimeter()
    return packing

def setBiPerimeter(packing, ratio = 1.4):
    """Rescale each polygon so the first half's perimeter is ``ratio`` times the second half's,
    holding the packing fraction (total area) fixed. With the shared shape index, P ~ sqrt(A),
    so the first half's area becomes ratio^2 times the second half's; areas renormalize to the
    original total. Each polygon is scaled about its centroid (preserving its equilateral
    shape) and its targets updated in place. Returns the packing.
    """
    N = packing.numPolygons
    if N % 2 != 0:
        raise ValueError("setBiPerimeter needs an even number of polygons")
    phi = float(packing.targetArea.sum())
    small = 2.0 * phi / (N * (1.0 + ratio * ratio))
    newArea = np.empty(N)
    newArea[: N // 2] = ratio * ratio * small
    newArea[N // 2 :] = small
    return _rescaleToAreas(packing, newArea)

def _logNormalTargets(packing, attribute, polydispersity, rng):
    """Randomize ``packing.<attribute>`` about its CURRENT MEAN, log-normally, in place.

    ``polydispersity`` is the coefficient of variation (std / mean). The mean is preserved, so only
    the SPREAD changes and the overall scale of the packing is untouched. Assigns TARGETS only: no
    vertex moves and no other target is altered."""
    if polydispersity < 0.0:
        raise ValueError(f"polydispersity is a coefficient of variation and cannot be negative, "
                         f"got {polydispersity}")
    rng = asRng(rng)
    current = np.asarray(getattr(packing, attribute), dtype = float)
    shape = np.sqrt(np.log(1.0 + polydispersity ** 2))
    # mean = -shape^2/2 makes the draw have unit expectation, so the mean is preserved exactly.
    draw = rng.lognormal(mean = -0.5 * shape ** 2, sigma = shape, size = current.size)
    setattr(packing, attribute, float(current.mean()) * draw)
    return packing

def _logNormalDraw(size, polydispersity, rng):
    """``size`` unit-mean log-normal factors with the requested coefficient of variation."""
    if polydispersity < 0.0:
        raise ValueError(f"polydispersity is a coefficient of variation and cannot be negative, "
                         f"got {polydispersity}")
    shape = np.sqrt(np.log(1.0 + polydispersity ** 2))
    return asRng(rng).lognormal(mean = -0.5 * shape ** 2, sigma = shape, size = size)


def setLogNormalTargetPerimeter(packing, polydispersity = 0.1, rng = None):
    """Randomize the target PERIMETERS log-normally about their current mean (CV = polydispersity).

    Independent of ``setLogNormalTargetArea``: applying both gives each polygon its own shape index
    p0 = P0 / sqrt(A0), which is the heterogeneous-target setting of the vertex-model literature rather
    than a uniform rescaling. Targets only -- no vertex moves.

    NB ``targetPerimeter`` is DERIVED, not stored: it is defined as the sum of a polygon's
    ``targetEdgeLength`` entries (``Packing.syncTargetPerimeter``), so there is no way to set it alone
    -- anything written straight to it is overwritten by the next sync. Setting a perimeter target
    therefore means scaling that polygon's EDGE targets together, which is what happens here. The
    polygon stays equilateral and its area target is untouched. Returns the packing."""
    before = packing.targetPerimeter.copy()
    _logNormalTargets(packing, "targetPerimeter", polydispersity, rng)
    packing.targetEdgeLength = packing.targetEdgeLength * (packing.targetPerimeter / before)[packing.shapeId]
    packing.syncTargetPerimeter()
    return packing

def setLogNormalTargetEdgeLength(packing, polydispersity = 0.1, rng = None):
    """Randomize every EDGE target independently, log-normally about the current mean.

    Unlike ``setLogNormalTargetPerimeter``, which scales each polygon's edges together, this gives each
    edge its own target -- so polygons stop being equilateral. The perimeter follows as the sum.
    Returns the packing."""
    _logNormalTargets(packing, "targetEdgeLength", polydispersity, rng)
    packing.syncTargetPerimeter()
    return packing

def setLogNormalTargetArea(packing, polydispersity = 0.1, rng = None):
    """Randomize the target AREAS log-normally about their current mean (CV = polydispersity).

    Independent of ``setLogNormalTargetPerimeter`` -- see that function. Targets only. Returns the
    packing."""
    return _logNormalTargets(packing, "targetArea", polydispersity, rng)


def setSizePolydispersity(packing, polydispersity):
    """Narrow (or widen) the SIZE distribution to the requested coefficient of variation, holding the
    total area -- and therefore the packing fraction -- and every polygon's SHAPE fixed.

    The annealing handle for polydispersity, and it has to act on the AREAS. The sizes of the polygons
    live in ``targetArea``; under ``setConstraints(area = True)`` those are frozen, so driving the
    edge-length moments toward zero cannot narrow the size distribution -- the edge lengths inherit
    their spread from the areas (edge ~ sqrt(A)) and are already as equal as those fixed areas permit.
    Squeezing them further only distorts shapes. Moving the areas is what actually anneals size.

    Deviations from the mean are scaled by ``polydispersity / current``, so the mean is preserved
    exactly (hence the packing fraction) and the CV lands on the request. Each polygon is then rescaled
    about its own centroid through ``_rescaleToAreas``, which carries the edge targets with it, so the
    shape index is untouched -- squares stay squares while their sizes converge."""
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    areas = np.asarray(packing.targetArea, dtype = float)
    free = areas[:stop]
    mean = float(free.mean())
    current = float(np.std(free) / mean) if mean > 0.0 else 0.0
    if current <= 0.0 or mean <= 0.0:
        return packing
    newArea = areas.copy()
    newArea[:stop] = mean + (free - mean) * (float(polydispersity) / current)
    if np.any(newArea[:stop] <= 0.0):
        raise ValueError(
            f"widening the size distribution to CV {polydispersity} would drive a target area "
            f"non-positive; narrow it instead, or reseed the spread with setLogNormalScale.")
    return _rescaleToAreas(packing, newArea)


def setLogNormalScale(packing, polydispersity = 0.1, rng = None):
    """Randomize polygon SIZE log-normally, holding every polygon's SHAPE fixed. Returns the packing.

    The one to use for a packing of same-shaped objects at different sizes. Each polygon is scaled by
    its own factor s, so its area target moves as s^2 and its edge targets as s -- which leaves the
    shape index p = P / sqrt(A) exactly where it was. The total area is renormalized, so the packing
    fraction does not move either.

    THE DIFFERENCE FROM ``setLogNormalTargetPerimeter`` MATTERS, and it is a silent trap. That function
    moves the perimeter targets ALONE. Combined with a hard area constraint it does not ask for bigger
    squares, it asks for DISTORTED ones: p0 = P0 / sqrt(A0) moves by the full polydispersity, so a 25%
    perimeter draw demands a shape index near 5.0 against 4.0 for a square, and the springs faithfully
    deliver a 49-degree rhombus with the right area. Independent area and perimeter targets are the
    heterogeneous-shape setting of the vertex-model literature -- correct there, wrong for packing
    equal-shaped objects.

    ``polydispersity`` is the coefficient of variation of the AREA (the linear scale varies as its
    square root)."""
    if polydispersity < 0.0:
        raise ValueError(f"polydispersity is a coefficient of variation and cannot be negative, "
                         f"got {polydispersity}")
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    draw = _logNormalDraw(stop, polydispersity, rng)
    newArea = np.asarray(packing.targetArea, dtype = float).copy()
    scaled = newArea[:stop] * draw
    newArea[:stop] = scaled * (newArea[:stop].sum() / scaled.sum())    # hold the packing fraction
    return _rescaleToAreas(packing, newArea)


def _scaleAboutCentroids(packing, scale):
    """Scale each non-container polygon about its own centroid by ``scale[p]``, in place.

    Geometry only: no target is touched. The container is skipped -- it is a pinned wall, and resizing
    it would move vertices the user asked to hold fixed."""
    r = packing.positions.reshape(-1, 2)
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    for p in range(stop):
        a = int(packing.startIndices[p])
        b = int(packing.startIndices[p + 1])
        centroid = r[a : b].mean(axis = 0)
        r[a : b] = centroid + scale[p] * (r[a : b] - centroid)
    return packing


def setLogNormalPerimeter(packing, polydispersity = 0.1, rng = None):
    """Randomize the REALIZED perimeters log-normally about their current mean, by scaling each polygon
    about its centroid. No target is touched.

    The geometry counterpart of ``setLogNormalTargetPerimeter``: that one says what the shapes should
    aim for, this one changes what they actually are. Use it to seed a spread that a moment constraint
    can then hold, since a moment constraint can only ever be NARROWED (see
    ``constraints.DistributionConstraints.retarget``).

    Scaling is isotropic, so the realized AREAS move too -- as the square of the perimeter factor. That
    makes this incompatible with a hard area constraint, which will simply undo it on the next
    retraction; use ``Model.spreadShapes`` for an area-PRESERVING spread. Returns the packing."""
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    lengths = backboneEdgeLengths(packing)
    current = np.bincount(packing.shapeId, weights = lengths,
                          minlength = packing.numPolygons)[:stop]
    draw = _logNormalDraw(stop, polydispersity, rng)
    return _scaleAboutCentroids(packing, float(current.mean()) * draw / current)


def setLogNormalArea(packing, polydispersity = 0.1, rng = None):
    """Randomize the REALIZED areas log-normally about their current mean, by scaling each polygon about
    its centroid. No target is touched.

    The geometry counterpart of ``setLogNormalTargetArea``; see ``setLogNormalPerimeter`` for the
    target-versus-geometry split and the area-preserving alternative. Area scales as the SQUARE of a
    centroid scaling, so the factor applied is the square root of the requested area ratio, and the
    realized perimeters move with it. Returns the packing."""
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    current = backboneArea(packing)[:stop]
    draw = _logNormalDraw(stop, polydispersity, rng)
    return _scaleAboutCentroids(packing, np.sqrt(float(current.mean()) * draw / current))

def setMonoPerimeter(packing):
    """Rescale every polygon to a COMMON perimeter, holding the packing fraction (total area) fixed --
    the counterpart of ``setBiPerimeter`` and the way back to a monodisperse packing after it.

    With the shared shape index an equal perimeter means an equal area, so each polygon's target
    becomes ``totalArea / N``. Unlike ``setBiPerimeter`` this places no parity requirement on N, and
    it is idempotent: applying it twice changes nothing. Returns the packing.
    """
    N = packing.numPolygons
    phi = float(packing.targetArea.sum())
    return _rescaleToAreas(packing, np.full(N, phi / N))