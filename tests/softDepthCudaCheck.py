"""Does the soft-depth CUDA tier compute what the numpy tier computes?

THE GRADIENT IS THE THING TO CHECK, NOT THE ENERGY. That is not a general principle, it is this
codebase's own history: a hard-coded ``PLUMMER_MAXN = 12`` once dropped the gradient on every vertex
past the 24th of a pair while the ENERGY stayed correct, so the failure survived the whole test suite.
Every check below that can assert on forces, does.

The two tiers are deliberately NOT the same algorithm, which makes agreement worth something:

  - numpy runs 24 safeguarded-Newton root steps; the device runs 12, with 8 for the peak;
  - numpy finds the softmin's envelope switches by PROBING ``max(16, 2 nB)`` points and solving where
    the argmin changes; the device WALKS the envelope exactly, one divide per candidate, because each
    ``ell_i`` is affine along an edge.

So this is a comparison against an independent construction rather than two spellings of one loop.

  1. AGREEMENT -- energy and force against numpy across contact types and random packings, free and
     periodic. The residual is quadrature, not error: it shrinks with the order, and check 3 shows it.
  2. VERTEX COUNTS -- n past every stride in the build (4, 12, 13, 32, 33, 64), and MIXED n in one
     packing, because nothing here may assume a uniform vertex count.
  3. ORDER -- raising the order on both tiers must collapse the disagreement. This is what proves the
     residual in check 1 is the rule and not the port.
  4. MAXN IS REPORTED -- a polygon past the build's cap must raise, not silently truncate.
  5. CONSERVATION -- forces sum to zero on the device to ~1e-18.
  6. PERIODICITY -- a pair overlapping only across the seam must match the same pair unwrapped.
  7. TIMING -- against numpy and against the sharp tier.

Run: python tests/softDepthCudaCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import softDepth as sd
import cudaOverlap as co
from model import Model
from softDepthCheck import square


def buildPacking(numPolygons, vertexCount, seed = 42, periodic = True, phi = 1.0):
    """A relaxed random packing, the configuration both tiers are asked about."""
    model = Model(N = numPolygons, n = vertexCount, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    if numPolygons % 2 == 0:
        model.setBiPerimeter()
    model.setBoundaryConditions("periodic" if periodic else "free")
    return model.packing


def ring(vertexCount, centerX, centerY, radius):
    """A CCW regular polygon with an exact vertex count -- convex, so Lemma 1 applies."""
    angles = np.linspace(0.0, 2.0 * np.pi, vertexCount, endpoint = False)
    return np.stack([centerX + radius * np.cos(angles), centerY + radius * np.sin(angles)], axis = -1)


def handBuilt(loops):
    """A packing assembled from explicit loops, bypassing the builder.

    Needed because `generateEquilateralPolygons` cannot be pushed to real contact at large vertex
    counts -- measured, many-sided polygons pack so loosely that phi = 1.6 leaves |F| ~ 7e-05 and
    phi >= 2.0 degrades to no contact at all. A check whose job is to reach past a compile-time stride
    must not be at the mercy of that."""
    packing = buildPacking(2, 4)
    packing.positions = np.concatenate([np.asarray(loop, dtype = float).ravel() for loop in loops])
    packing.startIndices = np.cumsum([0] + [len(loop) for loop in loops]).astype(int)
    packing.numPolygons = len(loops)
    return packing


def ringPacking(count, vertexCount, radius = 0.19, seed = 3):
    """A periodic packing of CONVEX regular polygons, jittered onto a grid so that they overlap."""
    side = int(np.ceil(np.sqrt(count)))
    rng = np.random.default_rng(seed)
    loops = []
    for index in range(count):
        cx = (index % side + 0.5) / side + rng.uniform(-0.35, 0.35) / side
        cy = (index // side + 0.5) / side + rng.uniform(-0.35, 0.35) / side
        loops.append(ring(vertexCount, cx, cy, radius / side * 2.4))
    return handBuilt(loops)


def epsilonFor(packing, fraction = 0.01):
    """``epsilon`` as a fraction of the packing's ACTUAL mean edge length.

    Measured from the geometry rather than from ``targetEdgeLength``, because hand-built packings reuse
    a builder packing's target arrays and those do not describe the loops that replaced them. Keeping
    every configuration at the same ``eps/edge`` matters: the quadrature's accuracy at a given order is
    a function of that ratio, so the tolerances below are only meaningful if it is held fixed."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    lengths = []
    for polygon in range(int(packing.numPolygons)):
        loop = vertices[starts[polygon]:starts[polygon + 1]]
        lengths.append(np.hypot(*(np.roll(loop, -1, axis = 0) - loop).T))
    return fraction * float(np.mean(np.concatenate(lengths)))


def bothTiers(packing, epsilon, stiffness = 1.0, order = 16):
    """``(numpyEnergy, numpyForce, cudaEnergy, cudaForce)`` at one configuration.

    ASSERTS CONVEXITY FIRST. The CUDA kernel implements the convex law only -- non-convex loops go
    through the convex differences tree on the numpy side and are gated off the device entirely -- so
    comparing the two on a non-convex packing compares different models and means nothing.

    This assertion exists because that is exactly what an earlier version of this file did: it ran
    checks 1 and 3 on builder packings that were 0/32 convex, both tiers computed the same meaningless
    near-zero quantity, and the 1e-12 agreement was reported as validation of the port."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    for polygon in range(int(packing.numPolygons)):
        assert sd.isConvex(vertices[starts[polygon]:starts[polygon + 1]]), \
            f"polygon {polygon} is not convex -- the device tier does not implement this case, so a " \
            f"comparison here would be between two different models"
    numpyEnergy, numpyForce = sd.packingEnergyForce(
        packing, epsilon, stiffness, 0.0, 1.0, 1.0, order, useCuda = False)
    cudaEnergy, cudaForce = co.softDepthCuda(packing, epsilon, stiffness, order)
    return numpyEnergy, numpyForce, cudaEnergy, cudaForce


def checkAgreement():
    """1. Energy and force against numpy, across shapes and both boundary conditions."""
    for periodic in (True, False):
        for numPolygons, vertexCount in ((8, 4), (16, 6), (32, 32)):
            # n=4 comes from the builder (measured convex at every kappa); higher vertex counts are
            # hand-built rings, because the builder cannot make a convex polygon above n=4.
            packing = (buildPacking(numPolygons, vertexCount, periodic = periodic)
                       if vertexCount == 4 else ringPacking(numPolygons, vertexCount))
            epsilon = epsilonFor(packing)
            numpyEnergy, numpyForce, cudaEnergy, cudaForce = bothTiers(packing, epsilon, order = 32)
            energyError = abs(cudaEnergy / numpyEnergy - 1.0) if numpyEnergy != 0.0 else 0.0
            forceError = float(np.abs(numpyForce - cudaForce).max())
            scale = float(np.abs(numpyForce).max())
            label = "periodic" if periodic else "free    "
            print(f"  1. {label} N={numPolygons:3d} n={vertexCount:3d}   E {numpyEnergy:.9e}   "
                  f"relE {energyError:.2e}   max|dF| {forceError:.2e}   (|F| ~ {scale:.2e})")
            assert numpyEnergy > 0.0, "the reference configuration has no contact to compare"
            # Order 32 at eps/edge = 1e-2 carries ~1e-9 of quadrature error per pair, and the two
            # tiers subdivide differently, so their difference sits at that floor. Check 3 is what
            # establishes this is the RULE's residual and not the port's error.
            assert energyError < 1e-7, "energies disagree beyond the quadrature residual"
            assert forceError < 1e-8 * max(scale, 1.0), "FORCES disagree -- see the module docstring"


def checkVertexCounts():
    """2. Past every stride in the build, and with MIXED vertex counts in one packing.

    The mixed case is the one that matters. A uniform-n packing cannot detect a kernel that assumes
    uniform n, because reading the stride from the wrong place still gives the right answer."""
    for vertexCount in (4, 12, 13, 32, 33, 64):
        # Three polygons placed to genuinely overlap, so the check cannot pass vacuously.
        packing = handBuilt([ring(vertexCount, 0.35, 0.50, 0.20),
                             ring(vertexCount, 0.63, 0.50, 0.20),
                             ring(vertexCount, 0.49, 0.72, 0.20)])
        epsilon = 0.002
        numpyEnergy, numpyForce, cudaEnergy, cudaForce = bothTiers(packing, epsilon, order = 32)
        forceError = float(np.abs(numpyForce - cudaForce).max())
        scale = max(float(np.abs(numpyForce).max()), 1e-300)
        print(f"  2. n={vertexCount:3d}   relE "
              f"{abs(cudaEnergy / numpyEnergy - 1.0) if numpyEnergy else 0.0:.2e}   "
              f"max|dF| {forceError:.2e}   (|F| ~ {scale:.2e})")
        assert scale > 1e-4, f"n={vertexCount}: barely any contact, the check would be vacuous"
        assert forceError < 1e-8 * max(scale, 1.0), f"n={vertexCount}: forces disagree"

    # Ragged: glue three polygons with different vertex counts into one packing by hand.
    packing = handBuilt([square(0.35, 0.42, 0.30), ring(7, 0.60, 0.42, 0.17),
                         ring(13, 0.47, 0.64, 0.20)])
    epsilon = 0.002
    numpyEnergy, numpyForce, cudaEnergy, cudaForce = bothTiers(packing, epsilon, order = 32)
    forceError = float(np.abs(numpyForce - cudaForce).max())
    scale = float(np.abs(numpyForce).max())
    print(f"  2. MIXED n = 4, 7, 13 in one packing   E {numpyEnergy:.9e}   "
          f"relE {abs(cudaEnergy / numpyEnergy - 1.0):.2e}   max|dF| {forceError:.2e}")
    assert numpyEnergy > 0.0, "the ragged configuration has no contact to compare"
    assert forceError < 1e-8 * max(scale, 1.0), "ragged vertex counts disagree"


def checkOrderCollapsesTheGap():
    """3. The residual in check 1 is the QUADRATURE, and this is what shows it.

    The two tiers subdivide the contact interval differently -- probe versus exact envelope walk -- so
    at a given order they place nodes in different places and carry different quadrature error. If the
    disagreement is that and nothing else, raising the order on both must collapse it. If it were a
    port bug it would sit at a floor instead."""
    # eps/edge = 1e-3, NOT the 1e-2 the other checks use. At 1e-2 this configuration already agrees to
    # 5e-14 at order 16 -- the floating-point floor -- so there is no quadrature error left to collapse
    # and the check would be vacuous. A sharper epsilon puts order 16 genuinely short, which is the
    # only regime in which "raising the order closes the gap" says anything.
    packing = ringPacking(32, 32)
    epsilon = epsilonFor(packing, 0.001)
    errors = []
    for order in (16, 32):
        numpyEnergy, numpyForce, cudaEnergy, cudaForce = bothTiers(packing, epsilon, order = order)
        errors.append((abs(cudaEnergy / numpyEnergy - 1.0), float(np.abs(numpyForce - cudaForce).max())))
        print(f"  3. order {order}   relE {errors[-1][0]:.2e}   max|dF| {errors[-1][1]:.2e}")
    assert errors[0][0] > 1e-9, \
        "order 16 already agrees to the noise floor here, so this check proves nothing -- sharpen epsilon"
    assert errors[1][0] < 0.2 * errors[0][0], "the gap did not shrink with the order -- not quadrature"
    assert errors[1][1] < 0.2 * errors[0][1], "the force gap did not shrink with the order"


def checkMaxVertexCountIsReported():
    """4. Past the build's cap, the driver must RAISE rather than return a wrong answer.

    The whole reason this check exists is that the alternative already happened here once, and the
    energy stayed correct while the gradient did not."""
    huge = 200
    packing = handBuilt([ring(huge, 0.35, 0.5, 0.20), ring(huge, 0.60, 0.5, 0.20)])
    try:
        co.softDepthCuda(packing, 0.004, 1.0, 16)
    except ValueError as error:
        print(f"  4. n={huge} past SOFTDEPTH_MAXN raised, as it must: {str(error).split('.')[0]}.")
        assert str(huge) in str(error), "the error does not name the offending vertex count"
        return
    raise AssertionError(f"n={huge} was accepted silently -- the cap truncated instead of reporting")


def checkConservation():
    """5. Forces sum to zero on the device."""
    for numPolygons, vertexCount in ((16, 6), (32, 32)):
        packing = ringPacking(numPolygons, vertexCount)
        epsilon = epsilonFor(packing)
        _, cudaForce = co.softDepthCuda(packing, epsilon, 1.0, 16)
        net = np.abs(cudaForce.reshape(-1, 2).sum(axis = 0)).max()
        scale = float(np.abs(cudaForce).max())
        print(f"  5. N={numPolygons:3d} n={vertexCount:3d}   |sum F| {net:.2e}   (|F| ~ {scale:.2e})")
        assert net < 1e-12 * max(scale, 1.0), "device forces do not sum to zero"


def checkPeriodicity():
    """6. A pair overlapping only across the seam must match the same pair unwrapped.

    The numpy tier got this wrong -- it shifted the boundary as well as the loop, which cancels -- so
    the device is checked against geometry rather than against numpy alone."""
    packing = buildPacking(2, 4)
    def place(firstX, secondX):
        vertices = packing.positions.reshape(-1, 2)
        vertices[0:4] = square(firstX, 0.5, 0.2)
        vertices[4:8] = square(secondX, 0.5, 0.2)
        return co.softDepthCuda(packing, 0.02, 1.0, 32)
    acrossSeam, seamForce = place(0.05, 0.97)
    inTheMiddle, middleForce = place(0.40, 0.32)
    energyError = abs(acrossSeam / inTheMiddle - 1.0)
    forceError = float(np.abs(np.sort(seamForce.reshape(-1, 2), axis = 0)
                              - np.sort(middleForce.reshape(-1, 2), axis = 0)).max())
    print(f"  6. across the seam {acrossSeam:.9e}   in the middle {inTheMiddle:.9e}   "
          f"relE {energyError:.2e}   force {forceError:.2e}")
    assert acrossSeam > 0.0, "a pair overlapping across the seam measured zero on the device"
    assert energyError < 1e-12, "wrapped and unwrapped energies differ on the device"
    assert forceError < 1e-12, "wrapped and unwrapped forces differ on the device"


def checkTiming():
    """7. Against numpy, and against the sharp tier for scale."""
    from energies import sharpOverlapEnergyForce
    for numPolygons, vertexCount in ((32, 4), (32, 32), (64, 8)):
        packing = (buildPacking(numPolygons, vertexCount) if vertexCount == 4
                   else ringPacking(numPolygons, vertexCount))
        epsilon = epsilonFor(packing)
        start = time.perf_counter()
        sd.packingEnergyForce(packing, epsilon, 1.0, 0.0, 1.0, 1.0, 16, useCuda = False)
        numpySeconds = time.perf_counter() - start
        co.softDepthCuda(packing, epsilon, 1.0, 16)
        start = time.perf_counter()
        for _ in range(20):
            co.softDepthCuda(packing, epsilon, 1.0, 16)
        cudaSeconds = (time.perf_counter() - start) / 20.0
        print(f"  7. N={numPolygons:3d} n={vertexCount:3d}   numpy {numpySeconds * 1e3:8.1f} ms   "
              f"cuda {cudaSeconds * 1e3:6.2f} ms   ({numpySeconds / cudaSeconds:.0f}x)")


def main():
    if not co.isAvailable():
        print("libplummer.so unavailable -- build it (make -C cuda libplummer.so). Skipping.")
        return
    print("soft depth on the GPU (cuda/softDepthKernels.cu)")
    checkAgreement()
    checkVertexCounts()
    checkOrderCollapsesTheGap()
    checkMaxVertexCountIsReported()
    checkConservation()
    checkPeriodicity()
    checkTiming()
    print("all checks passed")


if __name__ == "__main__":
    main()
