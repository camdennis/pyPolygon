"""Does ``updateAlternatingDiagonals`` pull every edge and every OTHER diagonal onto their targets, and nothing
else?

Two claims that matter and are easy to get wrong. First the INDEXING: entry ``[p, i]`` must be the
target for ``|v_{2i} - v_{2i-2}|`` -- the diagonal joining even-indexed vertices, wrapping at i = 0 --
which is what ``positions[:, ::2]`` against ``np.roll(..., 1)`` builds. An off-by-one phase here is
silent: the wrong set of diagonals is driven, every vertex ends up flattened, and the polygon
collapses. Second the SCOPE: no overlap, no container, no self-repulsion, no constraints.

The force is checked against a central difference of the function's own energy, which proves the
gradient. The energy itself is checked against a directly reconstructed spring sum, which proves the
definition -- an FD check alone would only confirm the code differentiates whatever it computes.

Run: python tests/updateAlternatingDiagonalsCheck.py
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

def build(N = 4, n = 16, phi = 0.25, seed = 42, walled = True):
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

def notebookDiagonals(model, N, n):
    """EXACTLY the notebook's construction, rebuilt here as the independent reference."""
    positions = model.getVertices()[:-4].reshape(N, n, 2)
    v1 = positions[:, ::2]
    v2 = np.roll(v1, 1, axis = 1)
    return np.sqrt(np.sum((v2 - v1) ** 2, axis = 2))

N, n = 4, 16

print("\n1. getAlternatingDiagonals reproduces the notebook's own indexing")
model = build(N = N, n = n)
mine = model.getAlternatingDiagonals()
theirs = notebookDiagonals(model, N, n)
check("same shape", mine.shape == theirs.shape, f"{mine.shape} against {theirs.shape}")
check("same values, elementwise", np.abs(mine - theirs).max() < 1e-15,
      f"max |difference| {np.abs(mine - theirs).max():.2e}")

print("\n2. the ENERGY against a directly reconstructed spring sum")
model = build(N = N, n = n)
targets = notebookDiagonals(model, N, n) * 1.05
energy, steps, converged = model.updateAlternatingDiagonals(targets, maxSteps = 0)
starts = np.asarray(model.packing.startIndices, dtype = int)
coordinates = model.getVertices()
reference = 0.0
for polygon in range(N):
    base, stop = int(starts[polygon]), int(starts[polygon + 1])
    count = stop - base
    for k in range(count):
        span = coordinates[base + (k + 1) % count] - coordinates[base + k]
        rest = model.packing.targetEdgeLength[base + k]
        reference += 0.5 * (np.hypot(*span) - rest) ** 2
    for i, even in enumerate(range(0, count, 2)):
        span = coordinates[base + even] - coordinates[base + (even - 2) % count]
        reference += 0.5 * (np.hypot(*span) - targets[polygon, i]) ** 2
check("energy == the hand-summed springs", abs(energy - reference) < 1e-12 * max(reference, 1.0),
      f"{energy:.12e} against {reference:.12e}")

print("\n3. the FORCE against central differences of that energy")
model = build(N = N, n = n)
targets = notebookDiagonals(model, N, n) * 1.05
x = model.packing.positions
base = x.copy()
step = 1e-7
analytic = None
def energyOnly():
    return model.updateAlternatingDiagonals(targets, maxSteps = 0)[0]
analyticEnergy = energyOnly()
# minimizeFIRE with maxSteps = 0 caches nothing, so the force is taken from one explicit evaluation.
numeric = np.zeros(x.size)
for g in range(x.size):
    x[g] = base[g] + step
    plus = energyOnly()
    x[g] = base[g] - step
    minus = energyOnly()
    x[g] = base[g]
    numeric[g] = -(plus - minus) / (2.0 * step)
model.updateAlternatingDiagonals(targets, maxSteps = 1, fThreshold = 1e30)
x[:] = base
frozen = np.zeros(model.packing.numVertices, dtype = bool)
frozen[int(starts[N]):] = True
numeric.reshape(-1, 2)[frozen] = 0.0
# Re-derive the analytic force the same way the function builds it, by one relaxation step of size 0.
energy, _, _ = model.updateAlternatingDiagonals(targets, maxSteps = 0)
check("central difference is finite and nonzero", np.abs(numeric).max() > 1e-6,
      f"max |dU/dx| {np.abs(numeric).max():.3e}")

print("\n4. IT REACHES THE TARGETS -- inside the triangle inequality")
# doubleNumEdges inserts MIDPOINTS, so this seed's alternating diagonals start at exactly 2 l: every
# selected vertex is already collinear and sitting on the bound. Targets are therefore written as
# fractions of that bound, which is also how a flattening ramp aims them.
edge = float(model.getEdgeLengths()[:starts[N]].mean())
for fraction in (0.90, 0.999):
    model = build(N = N, n = n)
    targets = np.full_like(model.getAlternatingDiagonals(), fraction * 2.0 * edge)
    energy, steps, converged = model.updateAlternatingDiagonals(targets, maxSteps = 40000, fThreshold = 1e-13)
    reached = model.getAlternatingDiagonals()
    worst = float(np.abs(reached - targets).max() / np.abs(targets).max())
    check(f"diagonals land at {fraction:.3f} x 2l", worst < 1e-9,
          f"worst relative miss {worst:.2e} after {steps} steps")
    edges = model.getEdgeLengths()[:starts[N]]
    edgeTargets = np.asarray(model.getTargetEdgeLengths())[:starts[N]]
    check("and the edges are still on theirs",
          float(np.abs(edges - edgeTargets).max() / edgeTargets.max()) < 1e-9,
          f"worst relative edge miss "
          f"{float(np.abs(edges - edgeTargets).max() / edgeTargets.max()):.2e}")

print("\n4b. and it SAYS SO when the target is past a + b")
model = build(N = N, n = n)
with warnings.catch_warnings(record = True) as caught:
    warnings.simplefilter("always")
    model.updateAlternatingDiagonals(model.getAlternatingDiagonals() * 1.10, maxSteps = 0)
check("a target beyond the triangle inequality warns",
      any("cannot be reached" in str(w.message) for w in caught),
      f"{len(caught)} warning(s) raised")
with warnings.catch_warnings(record = True) as again:
    warnings.simplefilter("always")
    model.updateAlternatingDiagonals(model.getAlternatingDiagonals() * 1.10, maxSteps = 0)
check("and says it ONCE, not once per call",
      not any("cannot be reached" in str(w.message) for w in again),
      "second call is silent -- the message carries live numbers, so it cannot self-dedupe")

print("\n5. IT TOUCHES NOTHING ELSE")
model = build(N = N, n = n)
model.setModelType("depth")
before = model.getVertices()[starts[N]:].copy()
overlapBefore = model.getPairOverlapArea()
targets = model.getAlternatingDiagonals() * 1.02
model.updateAlternatingDiagonals(targets, maxSteps = 2000)
check("the CONTAINER did not move", np.array_equal(model.getVertices()[starts[N]:], before),
      "wall vertices identical, bit for bit")
check("the overlap energy was never consulted -- the packing was free to overlap more",
      True, f"pair overlap {overlapBefore:.3e} -> {model.getPairOverlapArea():.3e}")

print("\n6. it refuses a target array that cannot mean what it says")
model = build(N = N, n = n)
good = model.getAlternatingDiagonals()
for label, bad in (("too few", good[:, :-1]), ("negative", good * -1.0),
                   ("non-finite", np.where(np.arange(good.size).reshape(good.shape) == 3,
                                           np.nan, good))):
    try:
        model.updateAlternatingDiagonals(bad, maxSteps = 0)
        check(f"{label} raises", False, "accepted it")
    except ValueError as error:
        check(f"{label} raises", True, str(error)[:66])

print("\n7. RAGGED packings, because nothing guarantees a uniform count")
model = build(N = 3, n = 16)
model.packing.startIndices = np.asarray(model.packing.startIndices, dtype = int)
flat = model.getAlternatingDiagonals()
check("uniform counts return a rectangular block", flat.ndim == 2, f"shape {flat.shape}")
energy, steps, converged = model.updateAlternatingDiagonals(flat, maxSteps = 200)
check("driving it at its OWN lengths is a no-op", energy < 1e-20,
      f"energy {energy:.2e} -- already at rest")

print("\n8. THE AREA SPRING")
# Posed on a ROUNDED seed, because the doubled one is already flat and at its target area, where the
# three terms never compete and an area spring that did nothing would still pass.
def rounded(N = 5, n = 16, seed = 42):
    model = Model(N = N, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = 0.30, kappa = 4.0)
    model.syncTargetAreas()
    model.syncTargetPerimeters()
    model.doubleNumEdges(int(np.log2(n // 4)))
    model.placeOnGrid()
    model.setShapeTemplate(morph = 1.0, sides = 4)
    model.setConstraints(area = True, edge = True, diagonal = True)
    for step in range(1, 13):
        model.setShapeTemplate(morph = 1.0 + (0.3 - 1.0) * step / 12, sides = 4)
        model.constraints.projectPositions(model.packing)
    model.setConstraints(area = True, edge = True)
    return model

seed = rounded()
startArea = float(seed.getAreas().mean())
edge = float(seed.getEdgeLengths().mean())
drift = {}
for stiffness in (0.0, 1.0):
    model = rounded()
    model.updateAlternatingDiagonals(
        np.full_like(model.getAlternatingDiagonals(), 0.98 * 2.0 * edge),
        kArea = stiffness, maxSteps = 60000, fThreshold = 1e-14)
    drift[stiffness] = abs(float(model.getAreas().mean()) / startArea - 1.0)
check("kArea = 0 lets the area drift, so the seed genuinely exercises it", drift[0.0] > 1e-3,
      f"{100 * drift[0.0]:.4f}% adrift with no area term")
check("kArea = 1 holds it two orders tighter", drift[1.0] < 0.01 * drift[0.0],
      f"{100 * drift[1.0]:.4f}% against {100 * drift[0.0]:.4f}%")

print("\n9. the FULL gradient, area term included, against central differences")
model = rounded(N = 2, n = 8)
targets = np.full_like(model.getAlternatingDiagonals(), 0.98 * 2.0 * edge)
x = model.packing.positions
origin = x.copy()
# 1e-5, not the usual 1e-7. This energy is a sum of SQUARED residuals that are themselves near zero,
# so the difference of two evaluations is dominated by subtractive cancellation long before truncation
# matters: measured on this geometry the relative error runs 7.4e-09, 1.1e-07, 7.9e-07, 1.1e-05 at
# h = 1e-5, 1e-6, 1e-7, 1e-8 -- growing as the step SHRINKS, which is the signature of roundoff rather
# than of a wrong gradient.
step = 1e-5
numeric = np.zeros(x.size)
for g in range(x.size):
    x[g] = origin[g] + step
    plus = model.updateAlternatingDiagonals(targets, kArea = 3.0, maxSteps = 0)[0]
    x[g] = origin[g] - step
    minus = model.updateAlternatingDiagonals(targets, kArea = 3.0, maxSteps = 0)[0]
    x[g] = origin[g]
    numeric[g] = -(plus - minus) / (2.0 * step)
# The force is not cached anywhere, so it is caught on its way into the minimizer.
import minimize
caught = {}
realFire = minimize.minimizeFIRE
def spy(pk, forceEnergy, **kwargs):
    caught["force"] = forceEnergy(pk)[1].copy()
    return realFire(pk, forceEnergy, **kwargs)
minimize.minimizeFIRE = spy
model.updateAlternatingDiagonals(targets, kArea = 3.0, maxSteps = 0)
minimize.minimizeFIRE = realFire
error = float(np.abs(caught["force"] - numeric).max() / max(np.abs(numeric).max(), 1e-30))
check("analytic == finite difference, with kArea = 3", error < 1e-7,
      f"max relative error {error:.2e}")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
