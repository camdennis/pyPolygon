"""What the neighbor list and the sorted followers actually buy, measured.

Three things compound and are reported separately, because they scale differently:

  candidates   edge pairs tested, against the all-to-all count
  broadphase   wall-clock to BUILD the candidate list, numpy vs CUDA
  intersect    wall-clock for updateIntersections, candidate path vs all-to-all
  followers    wall-clock for updateFollowers, sorted vs the O(M^2) scan

The broad phase is amortized over however many force evaluations the Verlet skin buys, so its column
is the per-REBUILD cost, not a per-step one -- divide by the realized reuse before comparing it to the
others.

The intersection speedup has two sources -- fewer pairs AND one vectorized pass instead of a triple
Python loop -- so the pair-count ratio and the time ratio will not match, and both are worth seeing.

Run: python tests/neighborTiming.py
"""

# UNVERIFIED(Cam)

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from neighbors import NeighborList, candidateEdgePairs
import energies


def buildPacking(n, N, seed = 3, phi = 0.55):
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = phi,
                                      kappa = float(np.sqrt(4.0 * n * np.tan(np.pi / n))))
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    return model


def timeIt(call, repeats = 3):
    best = np.inf
    for _ in range(repeats):
        began = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - began)
    return best


def referenceFollowers(packing):
    X = packing.intersections
    starts = np.asarray(packing.startIndices, dtype = int)
    M = len(X)
    out = np.full(M, -1, dtype = int)
    for alpha in range(M):
        rhoI, rhoJ = int(X[alpha, 0]), int(X[alpha, 1])
        nEdges = int(starts[rhoI + 1]) - int(starts[rhoI])
        cLeave = X[alpha, 4] + X[alpha, 5] - int(starts[rhoI])
        best, bestDist = -1, np.inf
        for beta in range(M):
            if int(X[beta, 0]) == rhoJ and int(X[beta, 1]) == rhoI:
                cArrive = X[beta, 2] + X[beta, 3] - int(starts[rhoI])
                dist = (cArrive - cLeave) % nEdges
                if 0.0 < dist < bestDist:
                    best, bestDist = beta, dist
        out[alpha] = best
    return out


def main():
    try:
        import cudaOverlap
        onGpu = cudaOverlap.isAvailable()
    except ImportError:
        cudaOverlap, onGpu = None, False
    print(f"{'n':>3} {'N':>4} {'allPairs':>10} {'cands':>8} {'cull':>7} "
          f"{'build np':>9} {'build gpu':>10} "
          f"{'intersect all':>14} {'cand':>10} {'speedup':>8} "
          f"{'M':>5} {'fol O(M^2)':>11} {'sorted':>9} {'speedup':>8}")
    for n in (4, 16):
        for N in (8, 32, 128):
            model = buildPacking(n, N)
            packing = model.packing
            allPairs = 0
            starts = np.asarray(packing.startIndices, dtype = int)
            for A in range(packing.numPolygons):
                for B in range(A + 1, packing.numPolygons):
                    allPairs += (starts[A + 1] - starts[A]) * (starts[B + 1] - starts[B])

            neighbors = NeighborList(packing)
            pairs = neighbors.rebuild(packing)
            cands = int(pairs[0].size)

            tBuildNp = timeIt(lambda: candidateEdgePairs(packing, skin = neighbors.skin))
            tBuildGpu = timeIt(lambda: cudaOverlap.neighborPairsCuda(
                packing, skin = neighbors.skin)) if onGpu else float("nan")

            tAll = timeIt(lambda: energies.updateIntersections(packing, candidates = None))
            tCand = timeIt(lambda: energies.updateIntersections(packing, candidates = pairs))

            energies.updateIntersections(packing, candidates = None)
            M = len(packing.intersections)
            tFolRef = timeIt(lambda: referenceFollowers(packing), repeats = 1)
            tFolNew = timeIt(lambda: energies.updateFollowers(packing), repeats = 1)

            print(f"{n:>3} {N:>4} {allPairs:>10} {cands:>8} "
                  f"{allPairs / max(cands, 1):>6.0f}x "
                  f"{tBuildNp * 1e3:>8.2f}ms {tBuildGpu * 1e3:>9.2f}ms "
                  f"{tAll * 1e3:>13.2f}ms {tCand * 1e3:>9.2f}ms "
                  f"{tAll / max(tCand, 1e-12):>7.0f}x "
                  f"{M:>5} {tFolRef * 1e3:>10.2f}ms {tFolNew * 1e3:>8.2f}ms "
                  f"{tFolRef / max(tFolNew, 1e-12):>7.1f}x", flush = True)


if __name__ == "__main__":
    main()
