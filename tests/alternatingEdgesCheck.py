"""Checks for the alternating short/long PAIR family with right-angled corners (``alternating.py``).

Run: python tests/alternatingEdgesCheck.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alternating
from model import Model
from constraints import ShapeConstraints

passed, failed = 0, 0

def check(name, condition, detail = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}   {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")

def build(N = 4, n = 16, phi = 0.4, seed = 7, jitter = 0.0):
    """Squares, then doubled up to n. Building at n directly returns FOLDED polygons -- see check 10.

    ``jitter`` breaks the collinearity the doubling leaves behind. A doubled square sits at
    ``d/(a + b) = 1`` EXACTLY, which is the triangle-inequality bound and a maximum of the quantity,
    so the corner rows have no gradient there and the first projection diverges -- measured
    ``|A/A0 - 1| = 6.3e+04``. Gaussian noise, not a systematic shrink, so nothing is biased."""
    model = Model(N = N, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.syncTargetAreas()
    model.doubleNumEdges(int(np.log2(n // 4)))
    if jitter:
        edge = float(np.mean(model.getEdgeLengths()))
        model.packing.positions += model.rng.normal(0.0, jitter * edge,
                                                    model.packing.positions.shape)
    return model

def turns(model, polygon):
    starts = np.asarray(model.packing.startIndices, dtype = int)
    loop = model.packing.positions.reshape(-1, 2)[starts[polygon]:starts[polygon + 1]]
    behind = loop - np.roll(loop, 1, axis = 0)
    ahead = np.roll(loop, -1, axis = 0) - loop
    return np.degrees(np.arctan2(np.cross(behind, ahead), np.einsum("ij,ij->i", behind, ahead)))

# ---------------------------------------------------------------- the target algebra
print("\n1. edges run in PAIRS, two of every four short, and the pairing sums to 2P/n")
model = build()
perimeter0 = np.array(model.packing.targetPerimeter, dtype = float)
starts = np.asarray(model.packing.startIndices, dtype = int)
model.setAlternatingEdges(0.5)
parity = int(model.packing.pairParity)
pattern = model.getAlternatingMask()[starts[0]:starts[1]].astype(int)
check("the mask is two-on two-off, phased by the chosen parity",
      np.array_equal(pattern, np.roll(np.tile([1, 1, 0, 0], 4), parity)),
      f"parity {parity}: {pattern}")
check("half the edges are short", int(pattern.sum()) == 8, "8 of 16")

for u in (0.05, 0.5, 0.83):
    model.setAlternatingEdges(u)
    drift = np.max(np.abs(np.array(model.packing.targetPerimeter) / perimeter0 - 1.0))
    check(f"perimeter held at u = {u}", drift < 1e-14, f"max relative drift {drift:.2e}")
    l0 = model.getTargetEdgeLengths()
    mask = model.getAlternatingMask()
    ratio = float(l0[mask].sum() / l0.sum())
    check(f"short pairs carry u of the perimeter at u = {u}", abs(ratio - u) < 1e-14,
          f"got {ratio:.15f}")

print("\n2. u outside (0, 1) is refused, and so is a count the pairs do not divide")
for bad in (0.0, 1.0, 1.2, -0.1):
    try:
        model.setAlternatingEdges(bad)
        check(f"u = {bad} refused", False, "no error raised")
    except ValueError as error:
        check(f"u = {bad} refused", "outside (0, 1)" in str(error), "")
try:
    alternating.requirePairs(0, 6)
    check("a count the pairs do not divide is refused", False, "no error raised")
except ValueError as error:
    check("a count the pairs do not divide is refused", "multiple of 4" in str(error), "n = 6")

# ---------------------------------------------------------------- the corner is the flatness family
print("\n3. the corner mask picks the middle of each LONG pair, one vertex in four")
model = build()
model.selectPairCorners()
parity = int(model.packing.pairParity)
corners = model.packing.diagonalMask[starts[0]:starts[1]]
short = model.getAlternatingMask()[starts[0]:starts[1]]
check("one vertex in four is a corner", int(corners.sum()) == 4 and
      np.array_equal(np.nonzero(corners)[0] % 4, np.full(4, (parity + 3) % 4)),
      f"parity {parity}, corners at {np.nonzero(corners)[0]}")
entering = np.roll(short, 1)
check("each corner sits between two LONG edges",
      bool(np.all(~short[corners]) and np.all(~entering[corners])),
      "neither the edge leaving nor the one entering a corner is short")
check("a doubled square puts the corners where its 90 degree turns already are",
      float(np.max(np.abs(np.abs(turns(model, 0))[corners] - 90.0))) < 0.5,
      f"corner turns {np.array2string(np.abs(turns(model, 0))[corners], precision = 4)} "
      f"(the build relaxes to a tolerance, so 90 +- 0.1)")

print("\n3b. the parity is chosen ONCE, whichever call gets there first")
either = build(N = 2)
either.setAlternatingEdges(0.35)
either.selectPairCorners()
other = build(N = 2)
other.selectPairCorners()
other.setAlternatingEdges(0.35)
check("both call orders agree", int(either.packing.pairParity) == int(other.packing.pairParity),
      f"parity {int(either.packing.pairParity)} either way")
short = either.getAlternatingMask()[starts[0]:starts[1]]
picked = either.packing.diagonalMask[starts[0]:starts[1]]
check("so every corner still has two equal (long) edges",
      bool(np.all(~short[picked]) and np.all(~np.roll(short, 1)[picked])), "")

print("\n4. d/(a + b) = cos(theta/2): RIGHT_ANGLE is exactly Cam's d = sqrt(2) l")
check("RIGHT_ANGLE = 1/sqrt(2)", abs(alternating.RIGHT_ANGLE - 0.7071067811865475) < 1e-15,
      f"{alternating.RIGHT_ANGLE:.16f}")
model = build(N = 3, jitter = 1e-3)
model.setAlternatingEdges(0.35)
model.relaxShapes(maxSteps = 40000, fThreshold = 1e-13)
model.selectPairCorners()
model.setConstraints(area = True, edge = True, flatten = True)
model.setCornerTargets(alternating.RIGHT_ANGLE)
model.constraints.projectPositions(model.packing)
got = model.getCornerRatios()
check("the corners reach the target", np.max(np.abs(got - alternating.RIGHT_ANGLE)) < 1e-10,
      f"worst |d/(a+b) - 1/sqrt2| = {np.max(np.abs(got - alternating.RIGHT_ANGLE)):.2e}")
angle = np.abs(turns(model, 0))[np.asarray(model.packing.diagonalMask[:16])]
check("which IS a 90 degree turn", np.max(np.abs(angle - 90.0)) < 1e-6,
      f"corner turns {np.array2string(angle, precision = 6)}")

# ---------------------------------------------------------------- feasibility, measured
def ramp(cornerOf, rounds = 30, springs = True, endU = 3e-3, N = 2, phi = 0.2):
    """Walk u down with the corner target given by ``cornerOf(t)``; return (worst max|C|, model)."""
    model = build(N = N, n = 16, phi = phi, jitter = 1e-3)
    model.selectPairCorners()
    model.setConstraints(area = True, edge = True, flatten = True)
    worst = 0.0
    for step in range(rounds + 1):
        t = step / rounds
        model.setAlternatingEdges(alternating.ratioSchedule(t, endMean = endU, startCv = 0.0)[0])
        model.setCornerTargets(cornerOf(t))
        if springs:
            model.relaxShapes(maxSteps = 30000, fThreshold = 1e-13)
        model.constraints.projectPositions(model.packing)
        worst = max(worst, model.constraints.maxResidual(model.packing))
    return worst, model

print("\n5. the corner is PINNED at 90, not ramped -- and the spring step is not optional")
print("   worst max|C| over a 30-round ramp, u 0.5 -> 0.003:")
results = {}
for name, cornerOf, springs in (
        ("corner pinned at 90, springs on ", lambda t: alternating.RIGHT_ANGLE, True),
        ("corner pinned at 90, springs OFF", lambda t: alternating.RIGHT_ANGLE, False),
        ("open to 0.93 then close        ", lambda t: alternating.RIGHT_ANGLE
                                                     + (0.93 - alternating.RIGHT_ANGLE)
                                                     * np.sin(np.pi * t), True),
        ("straight -> 90                 ", lambda t: 0.999 + (alternating.RIGHT_ANGLE - 0.999) * t,
         True)):
    worst, model = ramp(cornerOf, springs = springs)
    results[name] = worst
    print(f"     {name}   max|C| {worst:8.2e}   final u "
          f"{np.array2string(model.getAlternatingRatios(), precision = 5)}")
check("pinning the corner holds the manifold the whole way",
      results["corner pinned at 90, springs on "] < 1e-10,
      f"max|C| {results['corner pinned at 90, springs on ']:.2e}")
check("dropping the spring step loses it",
      results["corner pinned at 90, springs OFF"] > 1e-6,
      f"max|C| {results['corner pinned at 90, springs OFF']:.2e}")
check("moving the corner away and back loses it",
      results["open to 0.93 then close        "] > 1e-6
      and results["straight -> 90                 "] > 1e-6,
      "the seed already IS a square, so a corner ramp is a rearrangement")

worst, model = ramp(lambda t: alternating.RIGHT_ANGLE)
angle = np.abs(turns(model, 0))
corners = angle[np.asarray(model.packing.diagonalMask[:16])]
check("the corners are right angles at the end of the ramp",
      np.max(np.abs(corners - 90.0)) < 1e-3,
      f"corner turns {np.array2string(corners, precision = 5)}")

print("\n6. rank-order transport is monotone and never reorders")
current = np.array([0.31, 0.08, 0.55, 0.21, 0.44])
sample = alternating.logNormalQuantiles(5, 0.2, 0.3)
target = alternating.rankOrderTargets(current, sample)
check("ranks are preserved", np.array_equal(np.argsort(current), np.argsort(target)),
      f"{np.argsort(current)} vs {np.argsort(target)}")
check("the target multiset IS the schedule", np.allclose(np.sort(target), np.sort(sample)), "")
for mean, cv in ((0.5, 0.4), (0.05, 0.15)):
    draw = alternating.logNormalQuantiles(4000, mean, cv)
    check(f"logNormalQuantiles mean {mean}, cv {cv}",
          abs(draw.mean() / mean - 1) < 2e-3 and abs(draw.std() / draw.mean() / cv - 1) < 5e-3,
          f"got mean {draw.mean():.6f}, cv {draw.std() / draw.mean():.6f}")
check("cv = 0 is the degenerate limit",
      np.allclose(alternating.logNormalQuantiles(11, 0.02, 0.0), 0.02, atol = 1e-15), "")

# ---------------------------------------------------------------- the collapse
print("\n7. the short-pair collapse: 16 -> 8, then the flat test takes 8 -> 4")
model = build(N = 3, n = 16, phi = 0.3, jitter = 1e-3)
model.selectPairCorners()
model.setConstraints(area = True, edge = True, flatten = True)
try:
    model.halveNumEdges(criterion = "short")
    check("refuses while u = 0.5", False, "no error raised")
except ValueError as error:
    check("refuses while u = 0.5", "no collapsible pairs" in str(error), "")

rounds = 30
model.setCornerTargets(alternating.RIGHT_ANGLE)
for step in range(rounds + 1):
    model.setAlternatingEdges(
        alternating.ratioSchedule(step / rounds, endMean = 2e-3, startCv = 0.0)[0])
    model.relaxShapes(maxSteps = 30000, fThreshold = 1e-13)
    model.constraints.projectPositions(model.packing)
areaBefore = model.getAreas()[:3].copy()
perimeterBefore = np.array(model.packing.targetPerimeter[:3])
model.halveNumEdges(criterion = "short")
counts = np.diff(model.packing.startIndices)
check("16 -> 8", np.all(counts[:3] == 8), f"counts {counts[:3]}")
after = np.array(model.packing.targetPerimeter[:3])
check("the perimeter targets survived the merge",
      np.max(np.abs(after / perimeterBefore - 1)) < 1e-14,
      f"max relative drift {np.max(np.abs(after / perimeterBefore - 1)):.2e}")
lost = 1.0 - model.getAreas()[:3] / areaBefore
check("the collapse kept the area", np.max(np.abs(lost)) < 1e-3,
      f"worst area lost {100 * np.max(np.abs(lost)):.4f}% at u = 2e-3")
alternate = np.abs(turns(model, 0))
check("the collapsed octagon alternates 90 and 0 degrees -- a square with midpoints",
      np.max(np.abs(np.sort(alternate)[:4])) < 0.5 and np.max(np.abs(np.sort(alternate)[4:] - 90)) < 0.5,
      f"turns {np.array2string(alternate, precision = 3)}")

model.halveNumEdges(criterion = "flat")
model.setRegularTargets()
counts = np.diff(model.packing.startIndices)
check("8 -> 4 on the existing flat test", np.all(counts[:3] == 4), f"counts {counts[:3]}")
model.setConstraints(area = True, edge = True)
model.relaxShapes(maxSteps = 60000, fThreshold = 1e-13)
worst = max(float(np.max(np.abs(np.abs(turns(model, p)) - 90.0))) for p in range(3))
check("every corner is a right angle", worst < 0.5, f"worst |turn - 90| = {worst:.4f} degrees")
kappa = model.getShapeIndices()[:3]
check("kappa is 4", np.max(np.abs(kappa - 4.0)) < 2e-3,
      f"worst {kappa[np.argmax(np.abs(kappa - 4))]:.6f}")

print("\n8. the general merge did not change the stride-2 path")
model = build(N = 2, n = 16, phi = 0.3)
before = np.array(model.packing.targetEdgeLength[:16])
model.halveNumEdges(criterion = "flat")
merged = np.array(model.packing.targetEdgeLength[:8])
following = np.roll(before, -1)
check("stride-2 merge is still the adjacent pair sum",
      np.allclose(merged, (before + following)[0::2]) or np.allclose(merged, (before + following)[1::2]),
      f"{np.array2string(merged, precision = 6)}")

print("\n9. the built-in feasibility check is blind to the short/long split")
model = build(N = 2, n = 16, phi = 0.2, jitter = 1e-3)
for u in (0.5, 0.05, 0.005):
    model.setAlternatingEdges(u)
    index, floor = model.getTargetShapeMargin()
    builtin = ShapeConstraints(model.packing, area = True, edge = True).infeasibleReason(model.packing)
    print(f"    u = {u:<7} kappa {index[0]:.6f}   true edge floor {floor[0]:.6f}   "
          f"margin {index[0] - floor[0]:+.2e}   built-in: {'quiet' if builtin is None else 'refuses'}")
model.setAlternatingEdges(0.5)
index, floor = model.getTargetShapeMargin()
check("equilateral has real slack", index[0] - floor[0] > 0.3, f"margin {index[0] - floor[0]:.4f}")
model.setAlternatingEdges(5e-3)
index, floor = model.getTargetShapeMargin()
# The EDGE-only floor barely moves here (3.568 -> 3.639, the regular 16-gon walking to the regular
# octagon) because it is blind to the corner constraints. What actually closes as u -> 0 is the
# margin WITH the corners at 90 degrees, which is the u^3/4 law in the module docstring -- and no
# check in this file, nor in setConstraints, computes it from the packing.
check("the edge-only floor does NOT see the corner constraint", index[0] - floor[0] > 0.3,
      f"margin {index[0] - floor[0]:.4f} at u = 5e-3, against u^3/4 = {5e-3 ** 3 / 4:.2e} in truth")

print("\n10. the build returns FOLDED polygons above n = 4, which is why build() squares and doubles")
for n in (8, 16, 32):
    direct = Model(N = 3, n = n, seed = 7)
    direct.generateEquilateralPolygons(phi = 0.2, kappa = 4.0)
    worst = max(float(np.abs(turns(direct, p)).sum()) for p in range(3))
    print(f"    generateEquilateralPolygons(n = {n:2d}, kappa = 4): worst |turn| sum {worst:7.1f} deg")
doubled = build(N = 3, n = 16, phi = 0.2)
worst = max(float(np.abs(turns(doubled, p)).sum()) for p in range(3))
check("squares doubled to 16-gons stay simple", abs(worst - 360.0) < 1e-6,
      f"|turn| sum {worst:.6f} deg (360 = simple)")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
