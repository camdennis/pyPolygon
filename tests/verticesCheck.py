"""Does ``Model.getVertices`` hand back the coordinates the rest of the library is actually using?

``packing.positions`` is flat, so reading geometry meant ``positions.reshape(-1, 2)`` plus a
``startIndices`` slice at every call site -- a hundred of them across the package. ``getVertices``
is that expression once, and the only way it earns its place is by agreeing with the measurements
the library takes independently: shoelace areas from these points must be ``getAreas``, and segment
lengths between them must be ``getEdgeLengths``. Matching ``positions`` alone would prove only that
a reshape reshapes.

The packing here is RAGGED on purpose -- 16-gons inside a 4-vertex container wall -- because nothing
guarantees a uniform vertex count and a per-polygon accessor that assumes one is wrong the first
time ``halveNumEdges`` touches a single polygon.

Run: python tests/verticesCheck.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model

warnings.filterwarnings("ignore")

passed, failed = 0, 0

def check(name, condition, detail = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}   {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")

N, n = 4, 16

model = Model(N = N, n = 4, seed = 42)
model.generateEquilateralPolygons(phi = 0.30, kappa = 4.0)
model.syncTargetAreas()
model.doubleNumEdges(int(np.log2(n // 4)))
model.placeOnGrid()
model.addShape(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
model.setBoundaryConditions("fixed")

packing = model.packing
counts = np.diff(np.asarray(packing.startIndices, dtype = int))

print(f"\n{N} polygons at n = {n} plus a 4-vertex container: counts {counts}")

print("\n1. shape and layout")
r = model.getVertices()
check("(numVertices, 2)", r.shape == (packing.numVertices, 2), f"shape {r.shape}")
check("the packing is genuinely RAGGED, so a uniform form would not exist",
      len(set(counts.tolist())) > 1, f"vertex counts {sorted(set(counts.tolist()))}")
check("same numbers as the flat positions",
      np.array_equal(r.reshape(-1), packing.positions), "elementwise")

print("\n2. it is a VIEW -- writing moves the packing")
before = packing.positions.copy()
r[0, 0] += 0.125
moved = packing.positions[0] - before[0]
check("a write reaches packing.positions", abs(moved - 0.125) < 1e-15, f"moved {moved:.6f}")
packing.positions[:] = before
check("and restoring the flat array restores the view",
      np.array_equal(model.getVertices().reshape(-1), before), "elementwise")

print("\n3. per-polygon loops follow startIndices, ragged or not")
starts = np.asarray(packing.startIndices, dtype = int)
worst = 0.0
for polygon in range(packing.numPolygons):
    loop = model.getVertices(polygon)
    if loop.shape != (int(counts[polygon]), 2):
        worst = np.inf
        break
    worst = max(worst, float(np.abs(loop - r[starts[polygon] : starts[polygon + 1]]).max()))
check("every polygon's loop matches its own slice", worst == 0.0,
      f"max difference {worst:.1e} over {packing.numPolygons} polygons of differing n")
check("the container is included, at containerIndex",
      model.getVertices(int(packing.containerIndex)).shape == (4, 2),
      f"container loop {model.getVertices(int(packing.containerIndex)).shape}")

print("\n4. AGAINST THE LIBRARY'S OWN MEASUREMENTS -- an independent construction, not a reshape")
areas = np.array([0.5 * float(np.cross(model.getVertices(p),
                                       np.roll(model.getVertices(p), -1, axis = 0)).sum())
                  for p in range(packing.numPolygons)])
error = float(np.abs(areas - model.getAreas()).max())
check("shoelace areas from these points == getAreas", error < 1e-14,
      f"max |difference| {error:.2e}   (container reads {areas[int(packing.containerIndex)]:.4f}, "
      f"negative by winding)")
lengths = np.concatenate([np.hypot(*(np.roll(model.getVertices(p), -1, axis = 0)
                                     - model.getVertices(p)).T)
                          for p in range(packing.numPolygons)])
error = float(np.abs(lengths - model.getEdgeLengths()).max())
check("segment lengths from these points == getEdgeLengths", error < 1e-14,
      f"max |difference| {error:.2e} over {lengths.size} edges")

print("\n5. an out-of-range index refuses rather than wrapping")
for bad in (packing.numPolygons, -1):
    try:
        model.getVertices(bad)
        check(f"getVertices({bad}) raises", False, "returned a loop instead")
    except IndexError as error:
        check(f"getVertices({bad}) raises", True, str(error)[:70])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
