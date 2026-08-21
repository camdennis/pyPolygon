"""CG against L-BFGS on the depth tier, from one shared starting configuration.

Both are force-evaluation-bound, so the honest comparison is wall time and final ``max|F|`` for the same
step budget, plus the evaluations each spent per step -- which is the quantity the two differ in.
Reproduces ``tests/penetrationDepth.ipynb`` up to its ``setModelType("depth")``.

    python tests/minimizerCompare.py [steps]
"""

# UNVERIFIED(Cam)

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import pyPolygon as pp
from depthTierTiming import notebookState


def run(label, method, steps):
    packing = notebookState()
    start = time.perf_counter()
    energy, taken, converged = method(packing, steps)
    elapsed = time.perf_counter() - start
    print(f"  {label:10s} {elapsed:8.1f} s   {taken:5d} steps   "
          f"max|F| = {packing.getMaxUnbalancedForce():.4e}   E = {energy:.6e}")
    return elapsed, packing.getMaxUnbalancedForce()


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"depth tier, N = 32, n = 32, seed 42 -- {steps} steps each")
    run("CG", lambda p, s: p.minimizeCG(maxSteps = s, fThreshold = 1e-16), steps)
    run("L-BFGS", lambda p, s: p.minimizeLBFGS(maxSteps = s, fThreshold = 1e-16), steps)


if __name__ == "__main__":
    main()
