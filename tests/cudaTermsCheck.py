"""Validation + timing for every energy term that runs on the GPU.

Terms checked against their numpy reference:

  overlap (Plummer)  cuda/plummerDriver.cu   vs energies.plummerOverlapExact
  self-repulsion     cuda/selfRepulsion.cu   vs energies.selfRepulsionEnergyForce
  springs            cuda/springs.cu         vs softBody.eqSoftBodyEnergyForce(relative = True)

IMPORTANT -- why the configuration is perturbed first. A freshly built packing sits AT its eqSoftBody
targets, so the spring residuals (l - l0) and (A - A0) are ~1e-9 and the spring force is ~1e-8. Any
reordering of a floating-point sum then shows up as a RELATIVE force error near 1e-7 while the
ABSOLUTE error is a clean 1e-15 -- cancellation amplification, not a defect. Perturbing the vertices
puts the springs in their normal working range so the relative comparison is meaningful. The overlap
and self-repulsion terms carry no such cancellation and are compared at the built configuration too.

Run:  python tests/cudaTermsCheck.py
"""

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaOverlap
from energies import plummerOverlapExact, selfRepulsionEnergyForce
from model import Model
from softBody import eqSoftBodyEnergyForce

warnings.filterwarnings("ignore")

_TOL = 1e-11
_GRADIENT_TOL = 1e-10
CASES = [(8, 6, 0.8, 0.10), (16, 8, 0.9, 0.08), (32, 10, 1.0, 0.05), (64, 10, 1.0, 0.05)]


def buildModel(numPolygons, numVertices, phi, softening, perturb = 0.0, seed = 42):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    model.setModelType("mollified")
    model.setSofteningFraction(softening)
    if perturb > 0.0:
        rng = np.random.default_rng(seed)
        scale = perturb * float(np.mean(model.packing.targetEdgeLength))
        model.packing.positions += scale * rng.standard_normal(model.packing.positions.size)
    return model


def compare(name, reference, gpu, repeatsRef, repeatsGpu, tol, gradientTol):
    """Report relative energy/force agreement plus the speedup. Returns True when within tolerance."""
    energyRef, forceRef = reference()
    start = time.perf_counter()
    for _ in range(repeatsRef):
        reference()
    refSeconds = (time.perf_counter() - start) / repeatsRef

    energyGpu, forceGpu = gpu()
    for _ in range(3):
        gpu()
    start = time.perf_counter()
    for _ in range(repeatsGpu):
        gpu()
    gpuSeconds = (time.perf_counter() - start) / repeatsGpu

    forceRef = np.asarray(forceRef).reshape(-1)
    forceGpu = np.asarray(forceGpu).reshape(-1)
    energyError = abs(energyGpu - energyRef) / max(abs(energyRef), 1e-300)
    forceError = np.abs(forceGpu - forceRef).max() / max(np.abs(forceRef).max(), 1e-300)
    ok = energyError < tol and forceError < gradientTol
    print(f"      {'OK ' if ok else 'FAIL'} {name:<16} relE {energyError:.2e}  relF {forceError:.2e}"
          f"   numpy {refSeconds * 1e3:8.2f} ms  cuda {gpuSeconds * 1e3:7.2f} ms"
          f"  ({refSeconds / gpuSeconds:6.1f}x)")
    return ok


def checkCase(numPolygons, numVertices, phi, softening):
    model = buildModel(numPolygons, numVertices, phi, softening, perturb = 0.05)
    packing = model.packing
    print(f"\n    N={numPolygons} n={numVertices}  (vertices perturbed 5% of an edge)")
    results = [
        compare("overlap",
                lambda: plummerOverlapExact(packing, model.sigma, gOn = 2.0, gOff = 3.0),
                lambda: cudaOverlap.plummerOverlapCuda(packing, model.sigma, packing.targetArea,
                                                       packing.targetPerimeter),
                3, 10, _TOL, _GRADIENT_TOL),
        compare("self-repulsion",
                lambda: selfRepulsionEnergyForce(packing, model.kSelf, model.delta),
                lambda: cudaOverlap.selfRepulsionCuda(packing, model.kSelf, model.delta),
                10, 20, _TOL, _TOL),
        compare("springs",
                lambda: eqSoftBodyEnergyForce(packing, model.kEdge, model.kArea, relative = True),
                lambda: cudaOverlap.springsCuda(packing, model.kEdge, model.kArea),
                20, 20, _TOL, _TOL),
    ]
    return all(results)


def main():
    if not cudaOverlap.isAvailable():
        print("CUDA library not available -- build it with 'make -C cuda libplummer.so'")
        return 1
    print("\nCUDA energy terms vs numpy reference")
    results = [checkCase(*case) for case in CASES]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
