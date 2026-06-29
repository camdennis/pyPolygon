"""Build a packing of N equilateral backbone polygons.

Given a target shape index ``s = perimeter / sqrt(area)``, an area distribution, and a
packing fraction ``phi``, this sets each polygon's eqSoftBody targets and seeds a random
star to relax. All polygons share the side count ``n`` and shape index ``s``; their
sizes follow the area distribution (areas sum to phi).

Shaping happens in FREE SPACE (box = None): eqSoftBody is per-polygon and intrinsic, so
each polygon relaxes to its equilateral target independently of the others. Placing the
shaped polygons in the periodic cell and resolving overlaps at the target phi -- the
"shrink -> minimize collisions -> reset targets/phi" protocol Cam described -- needs the
collision model and is DEFERRED to the packing phases (build steps 6-7). The hook
``shrinkTargets`` below prepares the shrink half of that protocol.
"""

import warnings

import numpy as np

from enums import EnergyType, PackingType
from box import Box
from packing import Packing
from softBody import eqSoftBodyEnergyForce, backboneEdgeLengths, backboneArea
from minimize import minimizeFIRE
from distributions import sampleAreas, asRng

def regularShapeIndex(n):
    """Shape index (perimeter/sqrt(area)) of the regular n-gon -- the minimum a convex
    or non-convex n-gon can reach (nothing is more compact than regular)."""
    return 2.0 * np.sqrt(n * np.tan(np.pi / n))

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
    targetEdgeLength = targetPerimeter / n
    return targetEdgeLength, areas.copy(), targetPerimeter

def starSeed(n, circumradius, center, rng):
    """One random star polygon: n CCW points scaled to ~circumradius, about ``center``."""
    pts = rng.random((n, 2)) - 0.5
    pts = pts[np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))]
    pts = pts / np.max(np.sqrt((pts ** 2).sum(axis = 1))) * circumradius
    return pts + np.asarray(center, dtype = float)

def buildEquilateralPacking(N, n, shapeIndex, areaKind = "logNormal", phi = 1.0,
                            rng = None, rho = None, **distKwargs):
    """Free-space Packing of N polygons (n vertices each) with eqSoftBody targets set for
    ``shapeIndex`` and the area distribution (areas sum to phi). Seeds are random stars
    sized to each polygon's target circumradius and placed at random centers in [0,1)^2;
    relax with ``shapeBackbones`` to make them equilateral. box is None (shaping context).
    ``rho`` (corner radius) is stored on the packing when given.
    """
    rng = asRng(rng)
    areas = sampleAreas(N, areaKind, phi = phi, rng = rng, **distKwargs)
    targetEdgeLength, targetArea, targetPerimeter = equilateralTargets(areas, n, shapeIndex)
    blocks = []
    startIndices = [0]
    for i in range(N):
        circumradius = targetEdgeLength[i] / (2.0 * np.sin(np.pi / n))
        blocks.append(starSeed(n, circumradius, rng.random(2), rng))
        startIndices.append(startIndices[-1] + n)
    positions = np.vstack(blocks).reshape(-1)
    return Packing(positions, startIndices, box = None,
                   energyType = EnergyType.eqSoftBody, rho = rho,
                   targetEdgeLength = targetEdgeLength, targetArea = targetArea,
                   targetPerimeter = targetPerimeter)

def shapeBackbones(packing, kEdge = 1.0, kArea = 1.0, **fireKwargs):
    """Relax every polygon to its eqSoftBody equilateral target (free space)."""
    fe = lambda p: eqSoftBodyEnergyForce(p, kEdge, kArea)
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

def _warnLargePhi(packing, n):
    """Warn if the polygons are large enough relative to the unit box that the single-image
    periodic minimum-image assumption used by the overlap machinery may fail -- taken as a
    polygon's rounded diameter exceeding half the box."""
    circumradius = float(packing.targetEdgeLength.max()) / (2.0 * np.sin(np.pi / n))
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

def setBidispersePerimeter(packing, ratio = 1.4):
    """Rescale each polygon so the first half's perimeter is ``ratio`` times the second half's,
    holding the packing fraction (total area) fixed. With the shared shape index, P ~ sqrt(A),
    so the first half's area becomes ratio^2 times the second half's; areas renormalize to the
    original total. Each polygon is scaled about its centroid (preserving its equilateral
    shape) and its targets updated in place. Returns the packing.
    """
    N = packing.numPolygons
    if N % 2 != 0:
        raise ValueError("setBidispersePerimeter needs an even number of polygons")
    phi = float(packing.targetArea.sum())
    small = 2.0 * phi / (N * (1.0 + ratio * ratio))
    newArea = np.empty(N)
    newArea[: N // 2] = ratio * ratio * small
    newArea[N // 2 :] = small
    scale = np.sqrt(newArea / packing.targetArea)
    r = packing.positions.reshape(-1, 2)
    for p in range(N):
        a = int(packing.startIndices[p])
        b = int(packing.startIndices[p + 1])
        centroid = r[a : b].mean(axis = 0)
        r[a : b] = centroid + scale[p] * (r[a : b] - centroid)
    packing.targetArea = newArea
    packing.targetEdgeLength = packing.targetEdgeLength * scale
    packing.targetPerimeter = packing.targetPerimeter * scale
    return packing