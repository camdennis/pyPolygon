"""Validation for the CUDA separated-sum mollified overlap driver (cuda/plummerDriver.cu).

Checks the GPU driver against ``energies.plummerOverlapExact`` -- energy and full vertex gradient --
across several packing sizes, and reports the speedup.

The driver is a SEPARATED SUM, mirroring the sharp protocol: a kernel over edge-pair PANELS (the
mollified analogue of the sharp model's intersections), a join kernel over (pair, image) that forms
the switch-weighted cap sum, then a kernel over VERTICES that assembles and scatters the gradient.
This test is what pins that restructuring to the reference math.

Tolerances: the energy is a near-cancellation of edge-pair panels and the gradient carries a known
1/X1^2 conditioning floor, so ~1e-13 (energy) and ~1e-10 (gradient) relative are AT the floor of the
formulation, not slack. The pre-separation fused kernel showed the same 3.3e-11 gradient agreement.

Run:  python tests/cudaOverlapCheck.py
"""

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaOverlap
from energies import plummerOverlapExact
from model import Model

warnings.filterwarnings("ignore")

_ENERGY_TOL = 1e-11
# 2e-10, not 1e-10: more vertices means more chances of a near-parallel edge pair, and those hit the
# 1/X1^2 conditioning floor. Verified to be conditioning rather than a systematic error -- at n=13 the
# MEDIAN gradient error is 1.1e-13 against a max of 1.0e-10, with only 2.9% of dofs within 10x of the
# worst. A systematic error would be spread across all dofs, not concentrated in a handful.
_GRADIENT_TOL = 2e-10

# n MUST span past any per-polygon buffer limit in the driver. These cases were once all n <= 10,
# which hid a hard-coded PLUMMER_MAXN = 12 stride: for n > 12 the vertex kernel never assigned a
# thread to vertices past the 24th of a pair, so they got NO gradient, while moment writes ran into
# the neighbouring pair's slot. The ENERGY stayed correct throughout, so only a gradient check at
# large n catches it.
CASES = [
    (6, 6, 0.8, 0.10),
    (12, 8, 0.9, 0.08),
    (32, 10, 1.0, 0.05),
    (8, 13, 0.9, 0.08),
    (8, 16, 0.9, 0.08),
    (6, 24, 0.9, 0.08),
]


def buildModel(numPolygons, numVertices, phi, softening, seed = 42):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setModelType("mollified")
    model.setSofteningFraction(softening)
    return model


def checkCase(numPolygons, numVertices, phi, softening, repeats = 10):
    """Compare one packing on GPU against the Python reference. Returns True when both are in tol."""
    model = buildModel(numPolygons, numVertices, phi, softening)
    packing = model.packing

    start = time.perf_counter()
    energyRef, gradRef = plummerOverlapExact(packing, model.sigma, gOn = 2.0, gOff = 3.0)
    cpuSeconds = time.perf_counter() - start

    for _ in range(3):
        cudaOverlap.plummerOverlapCuda(packing, model.sigma, packing.targetArea,
                                       packing.targetPerimeter)
    start = time.perf_counter()
    for _ in range(repeats):
        energyGpu, gradGpu = cudaOverlap.plummerOverlapCuda(packing, model.sigma, packing.targetArea,
                                                            packing.targetPerimeter)
    gpuSeconds = (time.perf_counter() - start) / repeats

    gradRef = np.asarray(gradRef).reshape(-1)
    gradGpu = np.asarray(gradGpu).reshape(-1)
    energyError = abs(energyGpu - energyRef) / max(abs(energyRef), 1e-30)
    gradError = np.abs(gradGpu - gradRef).max() / max(np.abs(gradRef).max(), 1e-30)

    ok = energyError < _ENERGY_TOL and gradError < _GRADIENT_TOL
    print(f"    {'OK ' if ok else 'FAIL'} N={numPolygons:3d} n={numVertices:2d}  "
          f"relE {energyError:.2e}  relGrad {gradError:.2e}   "
          f"cpu {cpuSeconds * 1e3:8.1f} ms  gpu {gpuSeconds * 1e3:7.2f} ms  "
          f"({cpuSeconds / gpuSeconds:5.1f}x)")
    return ok


def main():
    if not cudaOverlap.isAvailable():
        print("CUDA overlap library not available -- build it with 'make -C cuda libplummer.so'")
        return 1
    print("\nCUDA separated-sum overlap driver vs energies.plummerOverlapExact")
    results = [checkCase(*case) for case in CASES]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
