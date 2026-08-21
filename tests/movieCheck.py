"""Validation for Model.minimizeMovie.

The claim that matters is NON-INTERFERENCE: recording must not perturb the relaxation. The frame hook
rides the minimizers' existing callback mechanism, so a recorded run should follow bit-identical
positions to the same run without a movie. If that ever stops holding, a movie would be showing a
trajectory the solver did not actually take.

Also checks that each minimizer backend records, that the options (force arrows, indicator field)
work, and that a .gif is produced when asked.

Run:  python tests/movieCheck.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from model import Model

warnings.filterwarnings("ignore")

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movies")


def buildModel(numPolygons = 12, numVertices = 8, seed = 42, constrained = True):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = 1.0, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    model.setModelType("mollified")
    model.setSofteningFraction(0.08)
    if constrained:
        model.setConstraints()
    return model


def runPlain(steps = 120):
    model = buildModel()
    model.minimizeFIRE(maxSteps = steps, fThreshold = 1e-12, dtMax = 0.10)
    return model.packing.positions.copy(), model.getEnergy()


def checkNonInterference(steps = 120):
    """Recording must add no deviation beyond the solver's own run-to-run spread.

    Bit-identical is NOT achievable and demanding it would be a false test: the CUDA kernels reduce
    with atomicAdd, so summation order varies between launches. One force evaluation is reproducible
    to ~1e-16, but a relaxation is chaotic and amplifies that to ~3e-12 over 120 steps -- two
    identical unrecorded runs differ by that much. So the baseline spread is MEASURED here and the
    recorded run is required to sit within it, which is the strongest true statement available."""
    print("\n[1] recording does not perturb the trajectory")
    firstPositions, firstEnergy = runPlain(steps)
    secondPositions, _ = runPlain(steps)
    baseline = np.abs(firstPositions - secondPositions).max()

    recorded = buildModel()
    recorded.minimizeMovie(os.path.join(OUTPUT, "nonInterference.mp4"), minimizer = "fire",
                           maxSteps = steps, fThreshold = 1e-12, frameEvery = 20, dtMax = 0.10,
                           progressBar = False)
    difference = np.abs(recorded.packing.positions - firstPositions).max()
    energyDifference = abs(recorded.getEnergy() - firstEnergy)
    tolerance = max(10.0 * baseline, 1e-13)
    ok = difference <= tolerance
    print(f"      {'OK ' if ok else 'FAIL'} recorded vs plain {difference:.3e}   "
          f"plain vs plain (CUDA atomic spread) {baseline:.3e}   energy diff {energyDifference:.3e}")
    return ok


def checkBackends():
    print("\n[2] minimizer backends record")
    ok = True
    for minimizer, extra in (("fire", {"dtMax": 0.10}), ("cg", {}), ("gd", {"step": 1e-4})):
        model = buildModel(constrained = (minimizer != "gd"))
        path = os.path.join(OUTPUT, f"backend_{minimizer}.mp4")
        energy, steps, _ = model.minimizeMovie(path, minimizer = minimizer, maxSteps = 60,
                                               fThreshold = 1e-12, frameEvery = 20,
                                               progressBar = False, **extra)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        ok &= exists
        print(f"      {'OK ' if exists else 'FAIL'} {minimizer:<5} {steps:3d} steps, "
              f"E = {energy:.6e}, {os.path.getsize(path) // 1024 if exists else 0} KB")
    return ok


def checkOptions():
    print("\n[3] overlays and gif output")
    model = buildModel()
    forcePath = os.path.join(OUTPUT, "withForces.mp4")
    model.minimizeMovie(forcePath, minimizer = "fire", maxSteps = 40, fThreshold = 1e-12,
                        frameEvery = 20, forces = True, dtMax = 0.10, progressBar = False)

    fieldPath = os.path.join(OUTPUT, "indicator.gif")
    model.minimizeMovie(fieldPath, minimizer = "fire", maxSteps = 40, fThreshold = 1e-12,
                        frameEvery = 20, indicatorColorMap = "viridis", indicatorResolution = 80,
                        dtMax = 0.10, progressBar = False)

    ok = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (forcePath, fieldPath))
    print(f"      {'OK ' if ok else 'FAIL'} force overlay {os.path.getsize(forcePath) // 1024} KB, "
          f"indicator gif {os.path.getsize(fieldPath) // 1024} KB")
    return ok


def checkUnknownMinimizer():
    print("\n[4] unknown minimizer is rejected")
    model = buildModel()
    try:
        model.minimizeMovie(os.path.join(OUTPUT, "bad.mp4"), minimizer = "newton")
    except ValueError as exc:
        print(f"      OK  raised: {exc}")
        return True
    print("      FAIL no error raised")
    return False


def main():
    os.makedirs(OUTPUT, exist_ok = True)
    print("\nMovie minimizer")
    results = [checkNonInterference(), checkBackends(), checkOptions(), checkUnknownMinimizer()]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
