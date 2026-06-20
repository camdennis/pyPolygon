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

import numpy as np

from enums import EnergyType
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
                            rng = None, **distKwargs):
    """Free-space Packing of N polygons (n vertices each) with eqSoftBody targets set for
    ``shapeIndex`` and the area distribution (areas sum to phi). Seeds are random stars
    sized to each polygon's target circumradius and placed at random centers in [0,1)^2;
    relax with ``shapeBackbones`` to make them equilateral. box is None (shaping context).
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
                   energyType = EnergyType.eqSoftBody,
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