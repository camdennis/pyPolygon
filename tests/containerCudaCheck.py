"""Validation for the CUDA fixed-boundary (container) term, cuda/container.cu.

Checks it against energies.containerEnergyForce for both wall windings and several sizes, confirms
the wall's OWN gradient is correctly absent (the kernel omits it, so the host must only use this path
when the wall is pinned), and that a shape fully inside the wall carries zero confinement energy.

Tolerances match the rest of the mollified stack: ~1e-12 on energy, ~5e-10 on the gradient, which is
the 1/X1^2 near-parallel conditioning floor and not slack -- at N=32 the MEDIAN force error is 6e-16
against a 1e-10 max, with 0.3% of dofs near the worst.

Run:  python tests/containerCudaCheck.py
"""

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaOverlap
from energies import containerEnergyForce
from model import Model

warnings.filterwarnings("ignore")

# 1e-11: the energy is a sum over every shape edge of terms that largely cancel (a = area + cap, and
# cap ~ -area for a contained shape), so ~1e-12 relative is the double-precision accumulation floor,
# not slack. Verified as absolute error 3.3e-13 on an energy of 0.277.
_ENERGY_TOL = 1e-11
_FORCE_TOL = 5e-10
CCW_WALL = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
CW_WALL = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])


def buildModel(numPolygons, numVertices, wall = CCW_WALL, phi = 1.0):
    model = Model(N = numPolygons, n = numVertices, seed = 42)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setModelType("mollified")
    model.setMollification(sigma = 1e-2)
    model.addShape(wall)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    return model


def checkAgainstReference():
    print("\n[1] CUDA vs numpy container term")
    ok = True
    for label, numPolygons, numVertices, wall in (
            ("N=32 n=10 CCW wall", 32, 10, CCW_WALL),
            ("N=32 n=10 CW wall", 32, 10, CW_WALL),
            ("N=16 n=16 CCW wall", 16, 16, CCW_WALL),
            ("N=64 n=8  CW wall", 64, 8, CW_WALL)):
        model = buildModel(numPolygons, numVertices, wall)
        packing = model.packing
        energyRef, forceRef = containerEnergyForce(packing, model.sigma)
        energyGpu, forceGpu = cudaOverlap.containerEnergyForceCuda(packing, model.sigma)
        shapeDofs = 2 * int(packing.startIndices[-2])
        energyError = abs(energyGpu - energyRef) / max(abs(energyRef), 1e-300)
        forceError = (np.abs(forceGpu[:shapeDofs] - forceRef[:shapeDofs]).max()
                      / max(np.abs(forceRef[:shapeDofs]).max(), 1e-300))
        good = energyError < _ENERGY_TOL and forceError < _FORCE_TOL
        ok &= good
        print(f"      {'OK ' if good else 'FAIL'} {label:<20} relE {energyError:.2e}  "
              f"relF {forceError:.2e}")
    return ok


def checkWallGradientOmitted():
    """The kernel does not compute the wall's own gradient; the host must only use it when pinned."""
    print("\n[2] wall gradient is absent from the CUDA result (by design)")
    model = buildModel(16, 8)
    packing = model.packing
    _, forceGpu = cudaOverlap.containerEnergyForceCuda(packing, model.sigma)
    wallDofs = 2 * int(packing.startIndices[-2])
    wallForce = np.abs(forceGpu[wallDofs:]).max()
    ok = wallForce == 0.0
    print(f"      {'OK ' if ok else 'FAIL'} wall force from CUDA = {wallForce:.2e} "
          f"(zero; applyPins would discard it anyway)")
    return ok


def checkContainedIsFree():
    """A contained packing must cost ~nothing, and the residual must VANISH WITH SIGMA.

    Not exactly zero: the mollified wall is blurred over sigma, so a shape sitting inside still
    overlaps the smeared boundary a little. The meaningful invariant is the scaling -- the residual
    goes as sigma^4 (E ~ a^2 with blur a ~ sigma^2), which is what distinguishes "soft wall behaving
    as designed" from "wall leaking"."""
    print("\n[3] a contained packing costs ~nothing, and the residual vanishes with sigma")
    energies = []
    for sigma in (2e-2, 1e-2, 5e-3, 2e-3):
        model = buildModel(8, 8, phi = 0.25)
        model.setMollification(sigma = sigma)
        packing = model.packing
        r = packing.positions.reshape(-1, 2)
        shapes = int(packing.startIndices[-2])
        r[:shapes] = 0.5 + 0.55 * (r[:shapes] - 0.5)
        energies.append(cudaOverlap.containerEnergyForceCuda(packing, sigma)[0])
    ratios = [energies[i] / energies[i + 1] for i in range(len(energies) - 1)]
    # halving sigma should drop the energy by ~16x (sigma^4); allow a wide band.
    ok = all(e < 1e-5 for e in energies) and all(4.0 < r < 64.0 for r in ratios)
    print(f"      {'OK ' if ok else 'FAIL'} E = {[f'{e:.1e}' for e in energies]} "
          f"as sigma halves; ratios {[f'{r:.0f}x' for r in ratios]} (sigma^4 => ~16x)")
    return ok


def reportSpeed():
    print("\n[4] speed")
    for numPolygons, numVertices in ((16, 8), (32, 10), (64, 10)):
        model = buildModel(numPolygons, numVertices)
        packing = model.packing
        containerEnergyForce(packing, model.sigma)
        start = time.perf_counter()
        for _ in range(5):
            containerEnergyForce(packing, model.sigma)
        cpuSeconds = (time.perf_counter() - start) / 5
        for _ in range(5):
            cudaOverlap.containerEnergyForceCuda(packing, model.sigma)
        start = time.perf_counter()
        for _ in range(20):
            cudaOverlap.containerEnergyForceCuda(packing, model.sigma)
        gpuSeconds = (time.perf_counter() - start) / 20
        print(f"      N={numPolygons:3d} n={numVertices:2d}   numpy {cpuSeconds * 1e3:7.2f} ms   "
              f"cuda {gpuSeconds * 1e3:6.2f} ms   ({cpuSeconds / gpuSeconds:5.1f}x)")


def main():
    if not cudaOverlap.isAvailable():
        print("CUDA library not available -- build it with 'make -C cuda libplummer.so'")
        return 1
    print("\nCUDA container (fixed boundary) term")
    results = [checkAgainstReference(), checkWallGradientOmitted(), checkContainedIsFree()]
    reportSpeed()
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
