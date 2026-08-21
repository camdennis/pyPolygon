"""Timing analysis for the constrained minimizers: what projection COSTS and what it BUYS.

Two separate questions, reported separately because they trade against each other:

  cost  -- the per-step overhead of the projection machinery (Jacobian + SVD + SHAKE) measured
           directly, and as a fraction of a force evaluation.
  buy   -- convergence per unit WALL TIME (not per step) for spring FIRE against constrained FIRE at
           several timestep ceilings, since the point of constraining is that dtMax can be raised.

Run:  python tests/constraintTiming.py [--steps 800] [--numPolygons 32] [--vertices 10]
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import ShapeConstraints
from model import Model

warnings.filterwarnings("ignore")


def buildModel(numPolygons, numVertices, phi, softening, seed = 42):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    model.setModelType("mollified")
    model.setSofteningFraction(softening)
    return model


def timeCall(fn, repeats):
    """Mean seconds per call over ``repeats`` calls, after one warm-up."""
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def reportCost(model, repeats = 20):
    """Per-call cost of each projection primitive against a force evaluation."""
    print("\n[cost] projection overhead per call")
    packing = model.packing
    constraints = ShapeConstraints(packing, area = True, edge = True)
    vector = np.ones_like(packing.positions)

    force = timeCall(lambda: model._forceEnergy(packing), repeats)
    jacobian = timeCall(lambda: constraints.jacobian(packing), repeats)
    project = timeCall(lambda: constraints.projectVector(packing, vector), repeats)
    shake = timeCall(lambda: constraints.projectPositions(packing), repeats)
    overhead = 2.0 * project + shake

    print(f"    force evaluation      {force * 1e3:9.3f} ms")
    print(f"    jacobian              {jacobian * 1e3:9.3f} ms")
    print(f"    projectVector         {project * 1e3:9.3f} ms")
    print(f"    projectPositions      {shake * 1e3:9.3f} ms   (converged, so ~1 iteration)")
    print(f"    ---")
    print(f"    per FIRE step         {overhead * 1e3:9.3f} ms   (2 projections + 1 SHAKE)")
    print(f"    as a fraction of one force evaluation: {overhead / force * 100:.1f}%")
    return force, overhead


def reportConvergence(numPolygons, numVertices, phi, softening, steps, fThreshold):
    """Convergence per unit wall time: springs at their stable ceiling vs constraints at several."""
    print(f"\n[buy] convergence in {steps} FIRE steps (fThreshold = {fThreshold:g})")
    print(f"    {'configuration':<28} {'steps':>6} {'wall s':>8} {'s/step':>8} "
          f"{'max|F|':>11} {'max|C|':>9}")

    model = buildModel(numPolygons, numVertices, phi, softening)
    start = time.perf_counter()
    taken = model.minimizeFIRE(maxSteps = steps, fThreshold = fThreshold, dtMax = 0.03)
    elapsed = time.perf_counter() - start
    drift = ShapeConstraints(model.packing, area = True, edge = True).maxResidual(model.packing)
    print(f"    {'springs, dtMax = 0.03':<28} {taken:6d} {elapsed:8.1f} {elapsed / max(taken, 1):8.4f} "
          f"{model.getMaxUnbalancedForce():11.3e} {drift:9.1e}")

    for dtMax in (0.03, 0.10, 0.30):
        model = buildModel(numPolygons, numVertices, phi, softening)
        model.setConstraints()
        start = time.perf_counter()
        taken = model.minimizeFIRE(maxSteps = steps, fThreshold = fThreshold, dtMax = dtMax)
        elapsed = time.perf_counter() - start
        print(f"    {'constrained, dtMax = ' + f'{dtMax:.2f}':<28} {taken:6d} {elapsed:8.1f} "
              f"{elapsed / max(taken, 1):8.4f} {model.getMaxUnbalancedForce():11.3e} "
              f"{model.constraintResidual():9.1e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 800)
    parser.add_argument("--numPolygons", type = int, default = 32)
    parser.add_argument("--vertices", type = int, default = 10)
    parser.add_argument("--phi", type = float, default = 1.0)
    parser.add_argument("--softening", type = float, default = 0.05)
    parser.add_argument("--fThreshold", type = float, default = 1e-10)
    args = parser.parse_args()

    print(f"N = {args.numPolygons}, n = {args.vertices}, phi = {args.phi}, "
          f"mollified sigma = {args.softening} * edge")
    model = buildModel(args.numPolygons, args.vertices, args.phi, args.softening)
    reportCost(model)
    reportConvergence(args.numPolygons, args.vertices, args.phi, args.softening,
                      args.steps, args.fThreshold)


if __name__ == "__main__":
    main()
