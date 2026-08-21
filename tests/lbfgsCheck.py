"""Correctness checks for ``minimize.minimizeLBFGS``.

Speed is measured in ``tests/minimizerCompare.py``; this file only asks whether the answer is right.
The two-loop recursion is checked against an objective whose minimizer is known in closed form rather
than against CG, since two minimizers agreeing proves only that they share a bug.

    python tests/lbfgsCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import minimize
import pyPolygon as pp

_FAILURES = []


def check(name, condition, detail = ""):
    tag = "PASS" if condition else "FAIL"
    if not condition:
        _FAILURES.append(name)
    print(f"  [{tag}] {name:56s} {detail}")


class StubPacking:
    """The minimum surface ``minimize`` uses: positions, force, energy, and a free-space box."""

    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype = float).ravel().copy()
        self.force = np.zeros_like(self.positions)
        self.energy = 0.0
        self.box = None


def quadraticProblem(size = 40, conditioning = 1e4, seed = 3):
    """E = 1/2 (x - xStar)^T A (x - xStar) with A symmetric positive definite and known spectrum."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size = (size, size)))[0]
    spectrum = np.geomspace(1.0, conditioning, size)
    matrix = basis @ np.diag(spectrum) @ basis.T
    target = rng.normal(size = size)

    def forceEnergy(packing):
        offset = packing.positions - target
        return 0.5 * float(offset @ matrix @ offset), -(matrix @ offset)

    return forceEnergy, target, matrix


def checkQuadratic():
    """A quadratic BENIGN enough that the memory can span it: n = 40 with memory 40 and a mild
    condition number is the regime where L-BFGS must reach the analytic minimizer outright. A stiff
    quadratic is NOT a correctness test -- with memory 10 and kappa = 1e4 the method genuinely crawls,
    which is what the parity check below establishes rather than assumes."""
    print("\n1. Exact quadratic, known minimizer")
    forceEnergy, target, matrix = quadraticProblem(size = 40, conditioning = 1e2, seed = 3)
    packing = StubPacking(np.zeros_like(target))
    energy, steps, converged = minimize.minimizeLBFGS(
        packing, forceEnergy, maxSteps = 400, fThreshold = 1e-10, memory = 40, progress = False)
    error = float(np.abs(packing.positions - target).max())
    check("converges on a kappa = 1e2 quadratic", converged, f"{steps} steps")
    check("reaches the analytic minimizer", error < 1e-9, f"max|x - xStar| = {error:.3e}")
    check("energy is the analytic minimum", abs(energy) < 1e-16, f"E = {energy:.3e}")


def checkAgainstScipy():
    """Parity with an INDEPENDENT implementation on the hard problem. This is the check that would
    catch a wrong two-loop recursion: a broken direction degrades to gradient descent, which on a
    kappa = 1e4 quadratic is orders of magnitude behind, not a few percent."""
    print("\n2. Parity with scipy L-BFGS-B on a kappa = 1e4 quadratic")
    try:
        from scipy.optimize import minimize as scipyMinimize
    except ImportError:
        print("       scipy unavailable -- skipped")
        return
    forceEnergy, target, matrix = quadraticProblem(size = 40, conditioning = 1e4, seed = 3)
    steps = 400

    packing = StubPacking(np.zeros_like(target))
    mine, _, _ = minimize.minimizeLBFGS(packing, forceEnergy, maxSteps = steps, fThreshold = 0.0,
                                        memory = 10, progress = False)
    theirs = scipyMinimize(lambda x: (0.5 * (x - target) @ matrix @ (x - target),
                                      matrix @ (x - target)),
                           np.zeros_like(target), jac = True, method = "L-BFGS-B",
                           options = dict(maxiter = steps, maxcor = 10, ftol = 1e-18, gtol = 1e-14))
    check("energy within 2x of scipy at equal iterations", mine < 2.0 * theirs.fun,
          f"ours {mine:.3e} vs scipy {theirs.fun:.3e}")
    check("not degraded to gradient descent", mine < 1e-4,
          f"E = {mine:.3e} after {steps} steps")


def checkMonotone():
    print("\n3. Energy decreases at every accepted step")
    forceEnergy, target, matrix = quadraticProblem(size = 24, conditioning = 1e3, seed = 8)
    packing = StubPacking(np.ones(24))
    history = []
    minimize.minimizeLBFGS(packing, forceEnergy, maxSteps = 60, fThreshold = 1e-12,
                           callback = lambda step, e, f: history.append(e), callbackEvery = 1,
                           progress = False)
    increases = [i for i in range(1, len(history)) if history[i] > history[i - 1]]
    check("no accepted step raises the energy", not increases,
          f"{len(history)} recorded, {len(increases)} increases")


def checkEvaluationCount():
    """The unit step must be accepted outright once the memory is warm. This is the entire reason to
    prefer L-BFGS here: every force evaluation is a CUDA kernel launch on the contact tiers, so a line
    search that spends several per step gives back what the direction won."""
    print("\n4. Force evaluations per step")
    forceEnergy, target, matrix = quadraticProblem(size = 40, conditioning = 1e3, seed = 3)
    packing = StubPacking(np.zeros_like(target))
    calls = [0]

    def counted(p):
        calls[0] += 1
        return forceEnergy(p)

    steps = 100
    minimize.minimizeLBFGS(packing, counted, maxSteps = steps, fThreshold = 0.0, progress = False)
    perStep = calls[0] / steps
    check("under 1.5 evaluations per step", perStep < 1.5, f"{perStep:.2f} evals/step")


def relaxedModel(steps = 200):
    packing = pp.Model(N = 12, n = 8, seed = 42)
    packing.generateEquilateralPolygons(phi = 1.0, kappa = 4.0)
    packing.setBiPerimeter()
    packing.setBoundaryConditions("periodic")
    packing.setSpringConstants()
    packing.setConstraints()
    packing.initForceEnergy()
    packing.minimizeFIRE(maxSteps = steps, fThreshold = 1e-16)
    return packing


def checkConstraintsHeld():
    print("\n5. Shape constraints survive the run")
    packing = relaxedModel()
    before = np.abs(packing.constraints.residual(packing.packing)).max()
    packing.minimizeLBFGS(maxSteps = 40, fThreshold = 1e-16)
    after = np.abs(packing.constraints.residual(packing.packing)).max()
    check("constraint residual stays at SHAKE tolerance", after < 1e-12,
          f"{before:.3e} -> {after:.3e}")


def checkPinsHeld():
    print("\n6. Pinned vertices do not move")
    packing = relaxedModel()
    packing.pinVertices([0, 1, 5])
    positions = packing.packing.positions.reshape(-1, 2).copy()
    packing.minimizeLBFGS(maxSteps = 30, fThreshold = 1e-16)
    moved = packing.packing.positions.reshape(-1, 2)
    drift = float(np.abs(moved[[0, 1, 5]] - positions[[0, 1, 5]]).max())
    check("pinned vertices are unmoved", drift == 0.0, f"max drift = {drift:.3e}")


def main():
    print("=" * 78)
    print("L-BFGS minimizer checks")
    print("=" * 78)
    checkQuadratic()
    checkAgainstScipy()
    checkMonotone()
    checkEvaluationCount()
    checkConstraintsHeld()
    checkPinsHeld()
    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"FAILED: {', '.join(_FAILURES)}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
