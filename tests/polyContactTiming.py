"""Timing harness for the exact-distance contact kernel.

Reports device milliseconds per energy+gradient evaluation on a compressed packing, which is the
configuration the minimizer actually spends its time in -- a lattice at spacing > 2R has an empty work
list and times nothing. Run it before and after any kernel change; the numbers here are what decides
whether an optimization stays.

    python tests/polyContactTiming.py [bodyCount] [vertexCount] [repeats]
"""

# UNVERIFIED(Cam)

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polyContact as pc
import polyContactSystem as sysm
import cudaOverlap


def squeezedPacking(bodyCount = 32, vertexCount = 32, squeeze = 0.86, seed = 11):
    """A lattice pulled in by ``squeeze`` so that neighbors genuinely overlap."""
    angles = np.arange(vertexCount) * 2.0 * np.pi / vertexCount
    shape = np.stack([np.cos(angles), np.sin(angles)], axis = 1) * 0.5
    bodies = sysm.certifiedLattice(shape, bodyCount)
    rng = np.random.default_rng(seed)
    centroids = sysm.bodyCentroids(bodies)
    middle = centroids.mean(axis = 0)
    positions = bodies.positions.copy()
    for body in range(bodies.count):
        low, high = bodies.startIndices[body], bodies.startIndices[body + 1]
        target = middle + squeeze * (centroids[body] - middle)
        jitter = rng.normal(scale = 0.01, size = 2)
        positions[low:high] += target + jitter - centroids[body]
    moved = sysm.BodySet([positions[bodies.startIndices[b]:bodies.startIndices[b + 1]]
                          for b in range(bodies.count)], boxSize = bodies.boxSize * squeeze)
    return moved


def timeDevice(bodies, repeats = 20, stiffness = 1.0):
    cudaOverlap.polyContactCuda(bodies, stiffness)
    start = time.perf_counter()
    for _ in range(repeats):
        energy, gradient = cudaOverlap.polyContactCuda(bodies, stiffness)
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed * 1e3, energy, gradient


def timeHost(bodies, repeats = 3, stiffness = 1.0):
    start = time.perf_counter()
    for _ in range(repeats):
        energy, gradient = sysm.systemEnergyGradient(bodies, stiffness, useCuda = False)
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed * 1e3, energy, gradient


def main():
    bodyCount = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    vertexCount = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    bodies = squeezedPacking(bodyCount, vertexCount)
    pairs = sysm.candidatePairs(bodies)
    print(f"N = {bodyCount}, n = {vertexCount}, candidate pairs = {len(pairs)}")

    if not cudaOverlap.isAvailable():
        print("  device unavailable")
        return
    deviceMs, deviceEnergy, deviceGradient = timeDevice(bodies, repeats)
    print(f"  device  {deviceMs:9.2f} ms   E = {deviceEnergy:.12e}")

    if bodyCount <= 32 and vertexCount <= 32:
        hostMs, hostEnergy, hostGradient = timeHost(bodies)
        error = abs(deviceEnergy - hostEnergy) / max(abs(hostEnergy), 1e-30)
        gradientError = np.abs(deviceGradient - hostGradient).max()
        print(f"  host    {hostMs:9.2f} ms   E = {hostEnergy:.12e}")
        print(f"  speedup {hostMs / deviceMs:9.1f} x   relative dE = {error:.3e},"
              f" max dG = {gradientError:.3e}")


if __name__ == "__main__":
    main()
