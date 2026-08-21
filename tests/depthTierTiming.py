"""Times the depth tier on exactly the configuration ``tests/penetrationDepth.ipynb`` runs.

The notebook is the benchmark of record for this tier, so kernel changes are judged here rather than on
a synthetic packing. Reproduces its cells 3-16: N = 32, n = 32, seed 42, bi-perimeter, periodic,
springs + constraints, FIRE to relax, then ``setModelType("depth")``.

    python tests/depthTierTiming.py [fireSteps] [repeats]
"""

# UNVERIFIED(Cam)

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))

import pyPolygon as pp


def notebookState(fireSteps = 1000):
    packing = pp.Model(N = 32, n = 32, seed = 42)
    packing.generateEquilateralPolygons(phi = 1.0, kappa = 4.0)
    packing.setBiPerimeter()
    packing.setBoundaryConditions("periodic")
    packing.setSpringConstants()
    packing.setConstraints()
    packing.initForceEnergy()
    packing.minimizeFIRE(maxSteps = fireSteps, fThreshold = 1e-16)
    packing.setModelType("depth")
    packing.initForceEnergy()
    return packing


def main():
    fireSteps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    start = time.perf_counter()
    packing = notebookState(fireSteps)
    print(f"setup ({fireSteps} FIRE steps in the spring tier): {time.perf_counter() - start:.1f} s")

    packing.calcForceEnergy()
    start = time.perf_counter()
    for _ in range(repeats):
        packing.calcForceEnergy()
    perCall = (time.perf_counter() - start) / repeats
    print(f"depth force+energy: {perCall * 1e3:9.2f} ms   E = {packing.getEnergy():.6e}"
          f"   max|F| = {packing.getMaxUnbalancedForce():.4e}")


if __name__ == "__main__":
    main()
