"""Does ``setConstraints(alternatingDiagonal = [1, 2])`` hold the SAME diagonals the other two act on?

The plain ``diagonal = [...]`` moments cover whatever ``packing.diagonalMask`` holds, and with no mask
that is EVERY vertex -- which drives the corners flat too and collapses the polygon. The failure is
quiet: a moment family is two rows whether it covers 80 vertices or 160, so nothing in the constraint
set reports which. This spelling makes the alternation part of the call.

The phase is the part that bites. A flatness row is indexed by the vertex a diagonal is CENTRED on,
while ``getAlternatingDiagonals`` and ``updateAlternatingDiagonals`` index by the chord -- so the mask
has to mark the ODD vertices to constrain the chords joining the EVEN ones. One vertex out of phase
selects the complementary set, and a ramp that drives one while holding the other flattens EVERY
vertex. Check 0 exists for that and nothing else.

The packing is RAGGED on purpose (32-gons inside a 4-vertex wall), because the mask is built per
polygon from ``startIndices`` and a uniform-n shortcut would pass a rectangular test and fail here.

Run: python tests/alternatingDiagonalCheck.py
"""

# UNVERIFIED(Cam)

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

def build(N = 5, n = 32, phi = 0.30, seed = 42, walled = True):
    model = Model(N = N, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.syncTargetAreas()
    model.syncTargetPerimeters()
    model.doubleNumEdges(int(np.log2(n // 4)))
    model.placeOnGrid()
    if walled:
        model.addShape(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
        model.pinVertices(np.arange(model.getNumVertices())[-4:])
        model.setBoundaryConditions("fixed")
    return model

N, n = 5, 32

print("\n0. THE THREE PIECES ACT ON THE SAME DIAGONALS")
# The one that is silently wrong if it is wrong. A flatness row is indexed by the vertex a diagonal is
# CENTRED on; getAlternatingDiagonals indexes by the chord. Being one vertex out of phase constrains
# the complementary set, so a ramp that drives one and holds the other flattens EVERY vertex.
model = build()
model.setConstraints(area = True, edge = True, alternatingDiagonal = [1, 2])
constrained = model.getFlatness()
measured = np.asarray(model.getAlternatingDiagonals()).reshape(-1)
starts = np.asarray(model.packing.startIndices, dtype = int)
coordinates = model.getVertices()
# Rebuild d/(a+b) for the chords getAlternatingDiagonals reports, independently of either path.
independent = []
for polygon in range(N):
    base, stop = int(starts[polygon]), int(starts[polygon + 1])
    count = stop - base
    for even in range(0, count, 2):
        far = (even - 2) % count
        mid = (even - 1) % count
        chord = np.hypot(*(coordinates[base + even] - coordinates[base + far]))
        a = np.hypot(*(coordinates[base + mid] - coordinates[base + far]))
        b = np.hypot(*(coordinates[base + even] - coordinates[base + mid]))
        independent.append(chord / (a + b))
independent = np.array(independent)
check("the CONSTRAINED quantity is d/(a+b) of the MEASURED chords",
      constrained.size == independent.size
      and float(np.abs(np.sort(constrained) - np.sort(independent)).max()) < 1e-14,
      f"{constrained.size} values, max |difference| "
      f"{float(np.abs(np.sort(constrained) - np.sort(independent)).max()):.2e}")
check("and updateAlternatingDiagonals drives that same count",
      measured.size == constrained.size,
      f"getAlternatingDiagonals gives {measured.size}, the constraint covers {constrained.size}")

print("\n1. it selects alternating vertices, in the phase that matches those chords")
mask = np.asarray(model.packing.diagonalMask, dtype = bool)
container = int(model.packing.containerIndex)
first = mask[starts[0]:starts[1]]
check("ODD local indices -- the CENTRES of the even-joining chords", not bool(first[0]),
      f"first 8 of polygon 0: {first[:8].astype(int)}")
check("and it alternates", np.array_equal(first, np.arange(first.size) % 2 == 1),
      f"{int(first.sum())} of {first.size} selected")
check("every non-container polygon, ragged or not",
      all(np.array_equal(mask[starts[p]:starts[p + 1]],
                         np.arange(starts[p + 1] - starts[p]) % 2 == 1)
          for p in range(model.getNumPolygons()) if p != container),
      f"vertex counts {np.diff(starts)}")
check("the container is NOT selected", not mask[starts[container]:starts[container + 1]].any(),
      f"wall has {starts[container + 1] - starts[container]} vertices, {int(mask[starts[container]:starts[container + 1]].sum())} selected")

print("\n2. the family it feeds is the DISTRIBUTION, over exactly that set")
distribution = model._distributionConstraints()
values = np.asarray(distribution.quantity(model.packing, "diagonal"))
check("two rows for the whole packing",
      np.asarray(distribution.jacobian(model.packing)).shape[0] == 2,
      f"jacobian {np.asarray(distribution.jacobian(model.packing)).shape}")
check("covering N * n/2 values, not N * n", values.size == N * (n // 2),
      f"{values.size} values against {N * n} vertices in the bodies")
check("and they are flatness d/(a+b)", values.min() > 0.0 and values.max() <= 1.0 + 1e-12,
      f"range {values.min():.6f} .. {values.max():.6f}")

print("\n3. WITHOUT it, the same call covers everything -- the failure this spelling prevents")
loose = build()
loose.setConstraints(area = True, edge = True, diagonal = [1, 2])
every = loose._distributionConstraints()
check("diagonal = [1, 2] with no mask covers EVERY vertex",
      int(every.diagonalSelected.sum()) == N * n,
      f"{int(every.diagonalSelected.sum())} of {N * n} body vertices")
check("and reports the SAME two rows, so the row count cannot tell them apart",
      np.asarray(every.jacobian(loose.packing)).shape[0] == 2, "2 rows either way")

print("\n4. it agrees with getFlatness, which is what a ramp reads")
flat = model.getFlatness()
check("getFlatness returns exactly the selected set", flat.size == values.size,
      f"{flat.size} values, max |difference| {float(np.abs(flat - values).max()):.1e}")

print("\n5. the moment Jacobian against central differences")
x = model.packing.positions
base = x.copy()
analytic = np.asarray(distribution.jacobian(model.packing))
numeric = np.zeros_like(analytic)
step = 1e-6
for g in range(x.size):
    x[g] = base[g] + step
    plus = distribution.residual(model.packing).copy()
    x[g] = base[g] - step
    minus = distribution.residual(model.packing).copy()
    x[g] = base[g]
    numeric[:, g] = (plus - minus) / (2.0 * step)
error = np.abs(analytic - numeric).max() / max(1.0, np.abs(numeric).max())
check("analytic == finite difference", error < 1e-7, f"max relative error {error:.2e}")

print("\n6. it refuses what it cannot mean")
try:
    build().setConstraints(area = True, edge = True, alternatingDiagonal = [1, 2], diagonal = [1, 2])
    check("both spellings at once raises", False, "accepted both")
except ValueError as error:
    check("both spellings at once raises", True, str(error)[:70])
try:
    build().setConstraints(area = True, edge = True, alternatingDiagonal = True)
    check("alternatingDiagonal = True raises", False, "accepted it")
except ValueError as error:
    check("alternatingDiagonal = True raises", True, str(error)[:70])

odd = Model(N = 3, n = 5, seed = 7)
odd.generateEquilateralPolygons(phi = 0.20, kappa = 4.0)
odd.syncTargetAreas()
try:
    odd.setConstraints(area = True, edge = True, alternatingDiagonal = [1, 2])
    check("an ODD vertex count raises", False, "accepted a count that cannot alternate")
except ValueError as error:
    check("an ODD vertex count raises", True, str(error)[:70])

print("\n7. the selection SURVIVES a constraint rebuild, since it lives on the packing")
model.setConstraints(area = True, edge = True)
model.setConstraints(area = True, edge = True, diagonal = [1, 2])
rebuilt = model._distributionConstraints()
check("still every other one after switching families and back",
      int(rebuilt.diagonalSelected.sum()) == N * (n // 2),
      f"{int(rebuilt.diagonalSelected.sum())} of {N * n}")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
