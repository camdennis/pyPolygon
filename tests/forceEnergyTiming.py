"""Where the time goes in a force/energy evaluation -- the cost model behind every minimizer step.

Breaks ``Model._forceEnergy`` into its terms and scans system size, for both model types:

  sharp      sharp overlap (intersections + followers) + eqSoftBody springs
  mollified  Plummer overlap (CUDA if built) + self-repulsion + eqSoftBody springs

The mollified overlap is split further into HOST PREP (centroids, contiguous copies, done in numpy
per call) and the DEVICE CALL (H2D, the three separated-sum kernels, D2H), because those scale
differently and only one of them is on the GPU.

Constraint projection is reported alongside: it is not part of the force evaluation, but a
constrained FIRE step pays it twice plus a SHAKE, so it belongs in the same budget.

Run:  python tests/forceEnergyTiming.py [--repeats 10]
"""

import argparse
import ctypes
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cudaOverlap
from constraints import ShapeConstraints
from energies import plummerOverlapExact, selfRepulsionEnergyForce, sharpOverlapEnergyForce
from model import Model
from softBody import eqSoftBodyEnergyForce

warnings.filterwarnings("ignore")

CASES = [(8, 6), (16, 8), (32, 10), (64, 10)]


def buildModel(numPolygons, numVertices, phi, softening, modelType, seed = 42):
    model = Model(N = numPolygons, n = numVertices, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setBiPerimeter()
    model.setSpringConstants()
    if modelType == "mollified":
        model.setModelType("mollified")
        model.setSofteningFraction(softening)
    return model


def timeCall(fn, repeats):
    """Mean seconds per call after a warm-up."""
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def timeOverlapSplit(packing, sigma, repeats):
    """(hostPrepSeconds, deviceSeconds) for the CUDA overlap, mirroring cudaOverlap.plummerOverlapCuda.

    Replicated rather than monkey-patched so the split stays readable and the numbers are honestly
    attributable to the lines they name."""
    r = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = np.int32)
    numPoly = int(starts.size - 1)
    numVert = int(r.shape[0])

    def prep():
        cent = np.array([r[starts[p]:starts[p + 1]].mean(0) for p in range(numPoly)])
        rad = np.asarray(packing.targetPerimeter, dtype = np.float64) / 4.0
        Atgt = np.asarray(packing.targetArea, dtype = np.float64)
        return (np.ascontiguousarray(r.ravel()), np.ascontiguousarray(starts),
                np.ascontiguousarray(cent.ravel()), rad, Atgt)

    hostSeconds = timeCall(prep, repeats)
    pos, startsC, centC, rad, Atgt = prep()
    energy = np.zeros(1); grad = np.zeros(2 * numVert)
    dp = ctypes.POINTER(ctypes.c_double)
    args = (pos.ctypes.data_as(dp), numVert,
            startsC.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), numPoly,
            centC.ctypes.data_as(dp), rad.ctypes.data_as(dp), Atgt.ctypes.data_as(dp),
            float(sigma), 2.0, 3.0,
            energy.ctypes.data_as(dp), grad.ctypes.data_as(dp))
    deviceSeconds = timeCall(lambda: cudaOverlap._lib.plummerOverlapCuda(*args), repeats)
    return hostSeconds, deviceSeconds


def row(label, seconds, total):
    share = seconds / total * 100.0 if total > 0 else 0.0
    print(f"      {label:<26} {seconds * 1e3:9.3f} ms   {share:5.1f}%")


def reportSharp(numPolygons, numVertices, phi, repeats):
    model = buildModel(numPolygons, numVertices, phi, None, "sharp")
    packing = model.packing
    total = timeCall(lambda: model._forceEnergy(packing), repeats)
    overlap = timeCall(lambda: sharpOverlapEnergyForce(packing, kOverlap = 1.0), repeats)
    springs = timeCall(lambda: eqSoftBodyEnergyForce(packing, model.kEdge, model.kArea,
                                                     relative = True), repeats)
    print(f"\n    sharp   N={numPolygons} n={numVertices}   total {total * 1e3:.3f} ms")
    row("sharp overlap", overlap, total)
    row("eqSoftBody springs", springs, total)
    row("(unattributed)", total - overlap - springs, total)


def reportMollified(numPolygons, numVertices, phi, softening, repeats):
    model = buildModel(numPolygons, numVertices, phi, softening, "mollified")
    packing = model.packing
    useCuda = cudaOverlap.isAvailable()

    # Time exactly the routines _forceEnergy calls, so the parts sum to the whole.
    total = timeCall(lambda: model._forceEnergy(packing), repeats)
    if useCuda:
        overlap = timeCall(lambda: cudaOverlap.plummerOverlapCuda(
            packing, model.sigma, packing.targetArea, packing.targetPerimeter), repeats)
        hostPrep, device = timeOverlapSplit(packing, model.sigma, repeats)
        selfRep = timeCall(lambda: cudaOverlap.selfRepulsionCuda(packing, model.kSelf, model.delta),
                           repeats)
        springs = timeCall(lambda: cudaOverlap.springsCuda(packing, model.kEdge, model.kArea),
                           repeats)
    else:
        overlap = timeCall(lambda: plummerOverlapExact(packing, model.sigma), repeats)
        hostPrep = device = None
        selfRep = timeCall(lambda: selfRepulsionEnergyForce(packing, model.kSelf, model.delta),
                           repeats)
        springs = timeCall(lambda: eqSoftBodyEnergyForce(packing, model.kEdge, model.kArea,
                                                         relative = True), repeats)

    constraints = ShapeConstraints(packing, area = True, edge = True)
    vector = np.ones_like(packing.positions)
    project = timeCall(lambda: constraints.projectVector(packing, vector), repeats)
    shake = timeCall(lambda: constraints.projectPositions(packing), repeats)

    tag = "CUDA" if useCuda else "numpy"
    print(f"\n    mollified   N={numPolygons} n={numVertices}   total {total * 1e3:.3f} ms   "
          f"(overlap on {tag})")
    row("Plummer overlap", overlap, total)
    if useCuda:
        row("  - host prep (numpy)", hostPrep, total)
        row("  - device call", device, total)
    row("self-repulsion", selfRep, total)
    row("eqSoftBody springs", springs, total)
    row("(unattributed)", total - overlap - selfRep - springs, total)
    stepCost = 2.0 * project + shake
    print(f"      {'constraint projection':<26} {stepCost * 1e3:9.3f} ms   "
          f"[+{stepCost / total * 100:.1f}% per FIRE step, not part of the force]")


def main():
    parser = argparse.ArgumentParser()
    # 10 repeats is visibly noisy at small N (the sub-item split can exceed its parent); 25 is stable.
    parser.add_argument("--repeats", type = int, default = 25)
    parser.add_argument("--phi", type = float, default = 1.0)
    parser.add_argument("--softening", type = float, default = 0.05)
    args = parser.parse_args()

    print(f"force/energy breakdown  (phi = {args.phi}, sigma = {args.softening} * edge, "
          f"{args.repeats} repeats)")
    for numPolygons, numVertices in CASES:
        reportSharp(numPolygons, numVertices, args.phi, args.repeats)
        reportMollified(numPolygons, numVertices, args.phi, args.softening, args.repeats)


if __name__ == "__main__":
    main()
