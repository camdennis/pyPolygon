"""Does ``setConstraints(alternatingDiagonal = [...], deviation = True)`` hold a NONNEGATIVE flatness
BUDGET, and does it beat the mean-only moment on the tail?

The mean/variance moment ``alternatingDiagonal = [1, 2]`` goes numerically parallel as its width goes
to zero -- unavoidably, since holding a mean AND a variance while driving the variance to a point mass
makes the two rows describe the same thing. Dropping to ``[1]`` (mean only) removes the parallelism but
not the blind spot: the mean has no purchase on the tail, and it advances by flattening vertices that
are already nearly flat while the worst one lags or regresses.

This is the third form: hold ``sum_i (1 - t_i)^p`` for the selected vertices, where ``t_i = d_i/(a_i+b_i)``
is flatness and ``1 - t_i >= 0`` is the DEFICIT, nonnegative by the triangle inequality -- the same kind
of theorem-backed one-sided quantity ``setShapeBudget`` already uses for the isoperimetric deficit. One
row, so it cannot go parallel with anything; and larger ``p`` weights the gradient toward the worst
vertices (``d(sum delta^p)/dr`` carries ``p delta^(p-1)``), which is the knob the mean-only form lacks.

Two things had to change to make this exist. ``deviations(packing, "diagonal")`` used to REFUSE --
"a skip-one distance has no one-sided ideal" -- which is true of a diagonal LENGTH against a stored
target and false of flatness against 1, a genuine bound. And ``getFlatness()`` read the family's raw
``quantity`` under all circumstances, which is the deficit under deviation mode -- silently returning
1 - (what its own docstring promises) whenever this family is active. Both are covered below.

    python tests/diagonalDeviationCheck.py
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

def build(N = 4, n = 16, phi = 0.30, seed = 42):
    """Rounded via the template, so the seed has genuine flatness spread to work with -- the doubled
    build alone is exactly collinear at every selected vertex, leaving nothing to flatten."""
    model = Model(N = N, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.syncTargetAreas()
    model.syncTargetPerimeters()
    model.doubleNumEdges(int(np.log2(n // 4)))
    model.placeOnGrid()
    model.setShapeTemplate(morph = 1.0, sides = 4)
    model.setConstraints(area = True, edge = True, diagonal = True)
    for step in range(1, 13):
        model.setShapeTemplate(morph = 1.0 + (0.3 - 1.0) * step / 12, sides = 4)
        model.constraints.projectPositions(model.packing)
    return model

print("\n1. the quantity is the DEFICIT, nonnegative, zero at flat")
model = build()
model._maskAlternatingDiagonals()
model.setConstraints(area = True, edge = True, alternatingDiagonal = [1], deviation = True)
distribution = model._distributionConstraints()
deficit = distribution.quantity(model.packing, "diagonal")
flatness = model.getFlatness()
check("deficit == 1 - flatness", float(np.abs(deficit - (1.0 - flatness)).max()) < 1e-14,
      f"max |difference| {float(np.abs(deficit - (1.0 - flatness)).max()):.2e}")
check("nonnegative everywhere", bool(np.all(deficit >= -1e-12)),
      f"min deficit {float(deficit.min()):.2e}")
check("getFlatness still reports FLATNESS, not the deficit, under deviation mode",
      float(flatness.min()) > 0.5,
      f"range {float(flatness.min()):.4f} .. {float(flatness.max()):.4f} -- a caller reading the "
      f"deficit here would see values near 0 instead of near 1")

print("\n2. the Jacobian against central differences, several exponents including the barrier")
for exponents, label in (([1], "p = 1"), ([2], "p = 2"), ([3], "p = 3"), ([1, -1], "p = [1, -1]")):
    model = build()
    model._maskAlternatingDiagonals()
    model.setConstraints(area = True, edge = True, alternatingDiagonal = exponents, deviation = True)
    distribution = model._distributionConstraints()
    x = model.packing.positions
    base = x.copy()
    step = 1e-6
    analytic = np.asarray(distribution.jacobian(model.packing))
    numeric = np.zeros_like(analytic)
    for g in range(x.size):
        x[g] = base[g] + step
        plus = distribution.residual(model.packing).copy()
        x[g] = base[g] - step
        minus = distribution.residual(model.packing).copy()
        x[g] = base[g]
        numeric[:, g] = (plus - minus) / (2.0 * step)
    error = float(np.abs(analytic - numeric).max() / max(1.0, float(np.abs(numeric).max())))
    check(f"{label}: analytic == finite difference", error < 1e-6, f"max relative error {error:.2e}")

print("\n3. the barrier ROW stays finite near the boundary (no NaN, no laundering through the rank test)")
model = build()
model._maskAlternatingDiagonals()
model.setConstraints(area = True, edge = True, alternatingDiagonal = [-1], deviation = True)
distribution = model._distributionConstraints()
jacobian = np.asarray(distribution.jacobian(model.packing))
check("finite at the seed", bool(np.all(np.isfinite(jacobian))), f"shape {jacobian.shape}")
check("reads as rank 1, not spuriously full rank on a poisoned block",
      int(np.linalg.matrix_rank(jacobian)) == 1, f"rank {int(np.linalg.matrix_rank(jacobian))}")

print("\n4. driven to the SAME nominal endpoint, larger p leaves a BETTER worst vertex")
# Each form is aimed at the mean a uniform distribution at FINAL_MEAN would have -- so the comparison
# is fair: same total flatness asked, differing only in how the retraction is free to distribute it.
FINAL_MEAN = 0.98
results = {}
for exponent, label, deviation in ((None, "mean-only", False), (1, "deviation p=1", True),
                                   (2, "deviation p=2", True), (4, "deviation p=4", True)):
    model = build()
    model._maskAlternatingDiagonals()
    if deviation:
        model.setConstraints(area = True, edge = True, alternatingDiagonal = [exponent],
                             deviation = True)
    else:
        model.setConstraints(area = True, edge = True, alternatingDiagonal = [1])
    distribution = model._distributionConstraints()
    start = model.getFlatness().copy()
    count = start.size
    startMean = float(start.mean())
    for step in range(1, 9):
        blend = step / 8
        meanTarget = startMean + (FINAL_MEAN - startMean) * blend
        if deviation:
            distribution.setReference("diagonal", [count * (1.0 - meanTarget) ** exponent])
        else:
            distribution.setReference("diagonal", [meanTarget * count])
        model.constraints.projectPositions(model.packing)
    flat = model.getFlatness()
    results[label] = {"mean": float(flat.mean()), "worst": float(flat.min()),
                      "sd": float(np.std(flat)),
                      "residual": float(model.constraints.maxResidual(model.packing))}
    print(f"   {label:<14} mean {results[label]['mean']:.4f}  worst {results[label]['worst']:.4f}  "
          f"sd {results[label]['sd']:.4f}  |C| {results[label]['residual']:.1e}")

check("p = 1 deviation reproduces the mean-only form exactly (a consistency check on the algebra)",
      abs(results["deviation p=1"]["worst"] - results["mean-only"]["worst"]) < 1e-9,
      f"{results['deviation p=1']['worst']:.6f} vs {results['mean-only']['worst']:.6f}")
check("worst vertex improves monotonically with p",
      results["mean-only"]["worst"] < results["deviation p=2"]["worst"]
      < results["deviation p=4"]["worst"],
      f"{results['mean-only']['worst']:.4f} < {results['deviation p=2']['worst']:.4f} < "
      f"{results['deviation p=4']['worst']:.4f}")
check("spread narrows monotonically with p",
      results["mean-only"]["sd"] > results["deviation p=2"]["sd"] > results["deviation p=4"]["sd"],
      f"{results['mean-only']['sd']:.4f} > {results['deviation p=2']['sd']:.4f} > "
      f"{results['deviation p=4']['sd']:.4f}")
check("every form leaves the per-object constraints satisfied",
      all(r["residual"] < 1e-10 for r in results.values()),
      f"worst residual {max(r['residual'] for r in results.values()):.2e}")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
