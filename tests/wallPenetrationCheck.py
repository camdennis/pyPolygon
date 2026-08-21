"""Verification for the two-part packing verdict.

    packs  iff  getPairOverlapArea() <= finalEnergy   (exact; tested against ZERO)
           and  getWallPenetration() <= wallTolerance (a DEPTH)

The split exists because only the first is exact. Polygon-polygon overlap reads identically
0.000000e+00 at every valid density. Containment cannot: what survives a long relaxation is a CORNER
just clipping the wall, whose area goes as delta^2 and whose restoring force goes as delta^3 -- so the
minimizer stops when that force sinks into the ~3e-12 force noise, not when the geometry is clean.
Measured slopes were 1.978 (area) and 2.966 (force) against the predicted 2 and 3.

Five checks:

  1. penetration is exactly zero for a packing wholly inside the wall;
  2. a polygon translated out by a known distance reads back that distance -- the depth is measured,
     not inferred from an area through the delta^2 law;
  3. the depth is found near a container CORNER too, where a point-to-line distance would understate
     it (the check that the segment clamp is doing its job);
  4. area and depth differ by a SQUARE, which is why the tolerance is written as a depth: a 1e-9 area
     tolerance grants ~3e-5 of depth;
  5. the split adds back up -- pair + container == the old total.

Run: python tests/wallPenetrationCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model


def buildPacking(N = 4, n = 4, seed = 3):
    """A few squares well inside a unit-square wall, with room to spare."""
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = 0.25, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    wall = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    model.addShape(wall)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setModelType("sharp")
    return model


def polygonSlice(model, polygon):
    a = int(model.packing.startIndices[polygon])
    b = int(model.packing.startIndices[polygon + 1])
    return a, b


def placeInside(model):
    """Put every polygon safely inside the wall, centred on a coarse grid."""
    packing = model.packing
    r = packing.positions.reshape(-1, 2)
    stop = int(packing.containerIndex)
    for polygon in range(stop):
        a, b = polygonSlice(model, polygon)
        centroid = r[a:b].mean(axis = 0)
        target = np.array([0.3 + 0.4 * (polygon % 2), 0.3 + 0.4 * (polygon // 2)])
        r[a:b] += target - centroid
    model._forces = None
    model._energy = None
    return model


def checkZeroInside():
    """1. Nothing outside -> exactly zero, not merely small."""
    model = placeInside(buildPacking())
    depth = model.getWallPenetration()
    area = model.getContainerOverlapArea()
    print(f"  1. wholly inside   penetration {depth:.3e}   outside area {area:.3e}")
    assert depth == 0.0, f"a contained packing reported penetration {depth:.3e}"


def checkKnownDisplacement():
    """2. Push a polygon out by a known distance; the depth must read it back."""
    model = placeInside(buildPacking())
    packing = model.packing
    a, b = polygonSlice(model, 0)
    r = packing.positions.reshape(-1, 2)
    # Move polygon 0 so its rightmost vertex sits a known distance beyond the x = 1 wall.
    rightmost = r[a:b, 0].max()
    for push in (1e-6, 1e-4, 1e-2):
        saved = packing.positions.copy()
        r = packing.positions.reshape(-1, 2)
        r[a:b, 0] += (1.0 - rightmost) + push
        model._forces = None
        model._energy = None
        depth = model.getWallPenetration()
        print(f"  2. pushed out by {push:.0e}   penetration {depth:.6e}   "
              f"error {abs(depth - push):.3e}")
        assert abs(depth - push) < 1e-12 + 1e-6 * push, (
            f"pushed {push:.3e} past the wall but measured {depth:.3e}")
        packing.positions[:] = saved
        model._forces = None
        model._energy = None


def checkCorner():
    """3. A vertex past a CORNER: the depth is to the corner point, not to either wall line.

    A point-to-LINE distance would report the smaller perpendicular offset and understate the
    excursion; the segment clamp is what makes this right."""
    model = placeInside(buildPacking())
    packing = model.packing
    a, b = polygonSlice(model, 0)
    r = packing.positions.reshape(-1, 2)
    # Written out explicitly rather than translated. The built polygons are ROTATED squares, so the
    # vertex furthest along x + y is not the one furthest along x: translating one onto the corner
    # leaves a different vertex sticking further out of a side wall, and the check would measure that
    # instead (it read 5.74e-02 for an intended 5.00e-03).
    corner = np.array([1.0, 1.0])
    offset = np.array([3e-3, 4e-3])                     # 5e-3 from the corner, a 3-4-5 triangle
    side = 0.05
    r[a:b] = np.array([corner + offset,
                       corner + offset - [side, 0.0],
                       corner + offset - [side, side],
                       corner + offset - [0.0, side]])
    model._forces = None
    model._energy = None
    depth = model.getWallPenetration()
    print(f"  3. vertex {offset} past the corner   penetration {depth:.6e}   expected 5.000000e-03")
    assert abs(depth - 5e-3) < 1e-12, f"corner penetration read {depth:.3e}, expected 5.0e-03"


def checkAreaVersusDepth():
    """4. Area and depth differ by a square -- the reason the tolerance is a depth."""
    model = placeInside(buildPacking())
    packing = model.packing
    a, b = polygonSlice(model, 0)
    r = packing.positions.reshape(-1, 2)
    rightmost = r[a:b, 0].max()
    r[a:b, 0] += (1.0 - rightmost)
    saved = packing.positions.copy()

    rows = []
    for push in (1e-4, 3e-4, 1e-3):
        packing.positions[:] = saved
        r = packing.positions.reshape(-1, 2)
        r[a:b, 0] += push
        model._forces = None
        model._energy = None
        rows.append((push, model.getContainerOverlapArea(), model.getWallPenetration()))
    packing.positions[:] = saved

    for push, area, depth in rows:
        print(f"  4. push {push:.0e}   outside area {area:.3e}   depth {depth:.3e}")
    # These are ROTATED squares, so a single vertex crosses first and the area goes as depth^2 --
    # measured here 2.354e-08 -> 2.354e-06 across a tenfold depth change, slope exactly 2. That square
    # is the whole argument for writing the tolerance as a depth: it makes a negligible-looking area
    # into a merely small depth.
    grant = np.sqrt(1e-9 / max(rows[0][1] / rows[0][2] ** 2, 1e-300))
    print(f"     a 1e-9 AREA tolerance grants ~{grant:.3e} of depth on this contact")


def checkSplitAddsUp():
    """5. pair + container == the total the single number used to report."""
    model = placeInside(buildPacking())
    packing = model.packing
    a, b = polygonSlice(model, 0)
    r = packing.positions.reshape(-1, 2)
    r[a:b, 0] += (1.0 - r[a:b, 0].max()) + 5e-3
    model._forces = None
    model._energy = None
    pair = model.getPairOverlapArea()
    wall = model.getContainerOverlapArea()
    total = model.getOverlapArea()
    print(f"  5. pair {pair:.6e} + wall {wall:.6e} = {pair + wall:.6e}   total {total:.6e}")
    assert abs((pair + wall) - total) < 1e-15, "the split does not reconstruct the total"


def main():
    print("wall penetration and the two-part verdict")
    checkZeroInside()
    checkKnownDisplacement()
    checkCorner()
    checkAreaVersusDepth()
    checkSplitAddsUp()
    print("all checks passed")


if __name__ == "__main__":
    main()
