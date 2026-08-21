"""Checks for the morph cascade: template + edge + diagonal, ramped, then decimated twice.

The protocol behind ``tests/morphCascade.ipynb``. Every number quoted in that notebook's preamble is
produced here, including the two counter-measurements (jitter, and leaving ``diagonal`` on across a
decimation) that say what NOT to do.

Run: python tests/morphCascadeCheck.py
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

def turns(model, polygon):
    starts = np.asarray(model.packing.startIndices, dtype = int)
    loop = model.packing.positions.reshape(-1, 2)[starts[polygon]:starts[polygon + 1]]
    behind = loop - np.roll(loop, 1, axis = 0)
    ahead = np.roll(loop, -1, axis = 0) - loop
    return np.degrees(np.arctan2(np.cross(behind, ahead),
                                 np.einsum("ij,ij->i", behind, ahead)))

def folded(model, N):
    """Worst total |turning angle|; exactly 360 for a simple polygon."""
    return max(float(np.abs(turns(model, p)).sum()) for p in range(N))

def corners(model, N):
    return max(float(np.max(np.abs(np.abs(turns(model, p)) - 90.0))) for p in range(N))

def seed(N = 5, n = 16, phi = 0.45, jitter = 0.0, rngSeed = 42):
    """Squares -> doubled -> lattice. That state IS ``morph = 1``, so the template applies unmoved."""
    model = Model(N = N, n = 4, seed = rngSeed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.syncTargetAreas()
    model.doubleNumEdges(int(np.log2(n // 4)))
    model.placeOnGrid()
    if jitter:
        edge = float(np.mean(model.getEdgeLengths()))
        model.packing.positions += model.rng.normal(0.0, jitter * edge,
                                                    model.packing.positions.shape)
    model.setShapeTemplate(morph = 1.0, sides = 4)
    model.setConstraints(area = True, edge = True, diagonal = True)
    return model

def ramp(model, openMorph = 0.35, rounds = 24, endMorph = 1.0):
    """Walk morph 1 -> openMorph -> endMorph, projecting each step. Returns the worst max|C|."""
    worst = 0.0
    for step in range(2 * rounds + 1):
        t = step / rounds
        morph = (1.0 - (1.0 - openMorph) * t) if t <= 1.0 \
            else (openMorph + (endMorph - openMorph) * (t - 1.0))
        model.setShapeTemplate(morph = morph, sides = 4)
        model.constraints.projectPositions(model.packing)
        worst = max(worst, model.constraints.maxResidual(model.packing))
    return worst

N = 5

# ---------------------------------------------------------------- the set is consistent
print("\n1. the over-determined set is CONSISTENT, not contradictory")
model = seed()
J = np.asarray(model.constraints.jacobian(model.packing))
singular = np.linalg.svd(J, compute_uv = False)
rank = (singular > 1e-10 * singular.max()).sum(axis = 1)
check("33 rows carry rank 2n - 3 = 29", J.shape[1] == 33 and np.all(rank == 29),
      f"shape {J.shape}, rank {rank}")
check("so the shape is RIGID -- zero shape DOF, which is why it cannot fold",
      int(rank[0]) == 2 * 16 - 3, "rank == 2n - 3")

print("\n2. the seed already IS morph = 1, so the template applies with no jump")
check("max|C| on arrival is at the SHAKE floor", model.constraints.maxResidual(model.packing) < 1e-10,
      f"max|C| {model.constraints.maxResidual(model.packing):.3e}")
check("and the polygons are simple", abs(folded(model, N) - 360.0) < 0.01,
      f"|turn| sum {folded(model, N):.4f} deg")
check("at kappa 4", np.max(np.abs(model.getShapeIndices()[:N] - 4.0)) < 1e-6,
      f"kappa {np.array2string(model.getShapeIndices()[:N], precision = 6)}")

# ---------------------------------------------------------------- the ramp
print("\n3. the ramp 1 -> 0.35 -> 1 holds the manifold, and step size is the only tunable")
for rounds in (12, 24, 48):
    worst = ramp(seed(), rounds = rounds)
    print(f"     rounds {rounds:3d} (each way): worst max|C| {worst:.3e}")
model = seed()
worst = ramp(model, rounds = 24)
check("24 sub-steps each way keeps max|C| under 1e-4", worst < 1e-4, f"worst {worst:.3e}")
check("nothing folded across the ramp", abs(folded(model, N) - 360.0) < 0.05,
      f"|turn| sum {folded(model, N):.4f} deg")

print("\n4. COUNTER-MEASUREMENT: jitter wrecks it, because the set is rigid up to REFLECTION")
jittered = seed(jitter = 1e-3)
worstJitter = ramp(jittered, rounds = 24)
print(f"     no jitter   worst max|C| {worst:.2e}   |turn| {folded(model, N):.3f}")
print(f"     jitter 1e-3 worst max|C| {worstJitter:.2e}   |turn| {folded(jittered, N):.3f}")
check("perturbing off the manifold invites a branch flip",
      worstJitter > 1e-3 and folded(jittered, N) > 400.0,
      "so this protocol takes NO jitter, unlike the alternating-pairs one")

# ---------------------------------------------------------------- the decimations
print("\n5. `diagonal` must be turned OFF before a decimation, and the code says so itself")
# halveNumEdges DROPS targetDiagonal on purpose -- the diagonals describe the old vertex spacing --
# and then rebuilds the live constraint set, which still asks for the family whose targets it just
# discarded. So the decimation refuses outright rather than silently constraining stale numbers.
model = seed()
ramp(model, rounds = 24)
raised = None
try:
    model.halveNumEdges()
except ValueError as error:
    raised = str(error)
check("halving with the diagonal family live is refused, not silently mishandled",
      raised is not None and "needs diagonal targets" in raised,
      "" if raised is None else raised.split(".")[0])
check("and the fix is in the message", raised is not None and "setShapeTemplate" in raised,
      "turn it off, halve, re-template if the next rung needs it")

print("\n6. with `diagonal` OFF first, the whole chain 16 -> 8 -> 4 is EXACT")
model = seed()
worst = ramp(model, rounds = 24)
before = model.getAreas()[:N].copy()
perimeter = np.array(model.packing.targetPerimeter[:N])
model.setConstraints(area = True, edge = True)
for rung in range(2):
    model.halveNumEdges()
    print(f"     -> n = {int(np.diff(model.packing.startIndices)[0])}   "
          f"dArea {np.max(np.abs(model.getAreas()[:N] / before - 1)):.2e}   "
          f"max|C| {model.constraints.maxResidual(model.packing):.2e}   "
          f"|turn| {folded(model, N):.4f}")
model.setRegularTargets()
model.setConstraints(area = True, edge = True)

check("down to quadrilaterals", np.all(np.diff(model.packing.startIndices)[:N] == 4),
      f"counts {np.diff(model.packing.startIndices)[:N]}")
lost = float(np.max(np.abs(model.getAreas()[:N] / before - 1)))
check("the decimations cost NO area", lost < 1e-12, f"worst |dA/A| {lost:.2e}")
drift = float(np.max(np.abs(np.array(model.packing.targetPerimeter[:N]) / perimeter - 1)))
check("and no perimeter", drift < 1e-12, f"worst |dP/P| {drift:.2e}")
check("every corner is a right angle", corners(model, N) < 1e-3,
      f"worst |turn - 90| {corners(model, N):.8f} deg")
check("kappa is 4", np.max(np.abs(model.getShapeIndices()[:N] - 4.0)) < 1e-6,
      f"kappa {np.array2string(model.getShapeIndices()[:N], precision = 6)}")
check("and they are still simple", abs(folded(model, N) - 360.0) < 1e-6,
      f"|turn| sum {folded(model, N):.6f} deg")

print("\n7. placeOnGrid keeps the seed off the wrong side of the half-overlap barrier")
onGrid = seed()
loose = Model(N = N, n = 4, seed = 42)
loose.generateEquilateralPolygons(phi = 0.45, kappa = 4.0)
loose.syncTargetAreas()
loose.doubleNumEdges(2)
print(f"     random centres: pair overlap {loose.getPairOverlapArea():.6f}")
print(f"     placeOnGrid   : pair overlap {onGrid.getPairOverlapArea():.6f}")
check("the grid start overlaps far less",
      onGrid.getPairOverlapArea() < 0.2 * loose.getPairOverlapArea(),
      "the contact law cannot push a pair back out once it is more than halfway through")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
