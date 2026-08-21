"""Validation for the PERIODIC sharp overlap on the GPU (cuda/sharpKernels.cu + sharpDriver.cu).

The risk this test exists for is COMPLETENESS of the intersection set, not arithmetic. The kernels
find crossings through a cell grid, and in a periodic box a pair straddling the boundary is only
found if the neighborhood search wraps and the polygons are compared in a common minimum image.
Binning by absolute coordinate -- what the pre-periodic kernels did -- silently misses exactly those
pairs, and the failure is invisible in any test whose polygons all sit in the interior.

So the checks are:

  1. Energy + full gradient vs energies.sharpOverlapEnergyForce over many random configurations,
     which is only exact if every crossing was found (a missed crossing changes the area).
  2. Configurations deliberately translated so polygons straddle the periodic boundary -- the case
     the old kernels got wrong -- swept across a range of offsets.
  3. Free-space (box = None) still agrees, so the periodic path did not break the bounded grid.

Run:  python tests/sharpCudaCheck.py
"""

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaOverlap
from energies import sharpOverlapEnergyForce
from model import Model

warnings.filterwarnings("ignore")

_TOL = 1e-12


def buildModel(numPolygons, numVertices, phi, seed):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    return model


def agree(packing, label, verbose = True):
    """Compare CUDA against the numpy reference at the packing's current positions."""
    energyRef, forceRef = sharpOverlapEnergyForce(packing, kOverlap = 1.0)
    energyGpu, forceGpu = cudaOverlap.sharpOverlapCuda(packing, kOverlap = 1.0)
    forceRef = np.asarray(forceRef).reshape(-1)
    energyError = abs(energyGpu - energyRef) / max(abs(energyRef), 1e-300)
    forceError = np.abs(forceGpu - forceRef).max() / max(np.abs(forceRef).max(), 1e-300)
    ok = energyError < _TOL and forceError < _TOL
    if verbose or not ok:
        print(f"      {'OK ' if ok else 'FAIL'} {label:<34} relE {energyError:.2e}  relF {forceError:.2e}"
              f"   (E = {energyRef:.6e})")
    return ok


def checkRandomConfigurations(trials = 8):
    """Random perturbations at several seeds: a missed crossing changes the area, so exact agreement
    over many configurations is the completeness evidence."""
    print("\n[1] random configurations (periodic box)")
    results = []
    for seed in range(trials):
        model = buildModel(24, 8, 1.0, seed = seed)
        packing = model.packing
        rng = np.random.default_rng(1000 + seed)
        scale = 0.08 * float(np.mean(packing.targetEdgeLength))
        packing.positions += scale * rng.standard_normal(packing.positions.size)
        results.append(agree(packing, f"seed {seed}", verbose = (seed < 3)))
    print(f"      ... {sum(results)}/{len(results)} configurations exact")
    return all(results)


def checkBoundaryStraddling(offsets = (0.0, 0.1, 0.25, 0.37, 0.5, 0.63, 0.8, 0.95)):
    """Rigidly translate the whole packing so different polygons straddle the periodic boundary.

    A rigid translation of every vertex is an exact symmetry of the periodic overlap, so the energy
    must be INVARIANT across offsets as well as matching the reference at each one. That invariance
    is the sharpest statement of periodic correctness available."""
    print("\n[2] packing translated across the periodic boundary")
    model = buildModel(24, 8, 1.0, seed = 3)
    packing = model.packing
    base = packing.positions.copy()
    results = []
    energies = []
    for offset in offsets:
        packing.positions[:] = base + offset
        energyGpu, _ = cudaOverlap.sharpOverlapCuda(packing, kOverlap = 1.0)
        energies.append(energyGpu)
        results.append(agree(packing, f"offset {offset:.2f}"))
    packing.positions[:] = base
    spread = max(energies) - min(energies)
    invariant = spread < 1e-12 * max(abs(e) for e in energies)
    print(f"      {'OK ' if invariant else 'FAIL'} translation invariance: energy spread {spread:.2e}")
    return all(results) and invariant


def checkFreeSpace():
    """box = None must still use the bounded grid and agree (the periodic work did not break it)."""
    print("\n[3] free space (box = None)")
    model = buildModel(12, 6, 0.5, seed = 11)
    packing = model.packing
    packing.box = None
    rng = np.random.default_rng(7)
    packing.positions += 0.05 * float(np.mean(packing.targetEdgeLength)) * \
        rng.standard_normal(packing.positions.size)
    return agree(packing, "free space")


def reportSpeed():
    print("\n[4] speed")
    for numPolygons, numVertices in [(16, 8), (32, 10), (64, 10)]:
        model = buildModel(numPolygons, numVertices, 1.0, seed = 42)
        packing = model.packing
        start = time.perf_counter()
        sharpOverlapEnergyForce(packing, kOverlap = 1.0)
        cpuSeconds = time.perf_counter() - start
        cudaOverlap.sharpOverlapCuda(packing)
        start = time.perf_counter()
        for _ in range(10):
            cudaOverlap.sharpOverlapCuda(packing)
        gpuSeconds = (time.perf_counter() - start) / 10
        print(f"      N={numPolygons:3d} n={numVertices:2d}   numpy {cpuSeconds * 1e3:8.1f} ms   "
              f"cuda {gpuSeconds * 1e3:6.2f} ms   ({cpuSeconds / gpuSeconds:6.1f}x)")


def main():
    if not cudaOverlap.isAvailable():
        print("CUDA library not available -- build it with 'make -C cuda libplummer.so'")
        return 1
    print("\nPeriodic sharp overlap: CUDA vs energies.sharpOverlapEnergyForce")
    results = [checkRandomConfigurations(), checkBoundaryStraddling(), checkFreeSpace()]
    reportSpeed()
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
