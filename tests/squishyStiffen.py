"""Does packing SQUISHY shapes and then stiffening them beat packing rigid squares directly?

Three protocols on the same seed, all ending at rigid squares so the comparison is fair:

  rigid       squares from the start -- the baseline, measured at side 4.0057 for N = 11
  jump        squishy while packing, then rigidified in one step
  stiffen     squishy while packing, then the edge distribution narrowed to zero GRADUALLY

"Squishy" is ``setConstraints(area = True, perimeter = True)``: area and perimeter both pinned, so the
shape index kappa = P/sqrt(A) is locked at the square's value of 4, but with n = 32 that leaves 61 shape
degrees of freedom. The polygons keep a square's isoperimetric ratio while moulding around each other.

The bet is that compliant objects find an ARRANGEMENT that rigid ones cannot reach, and that the
arrangement survives being hardened. ``jump`` and ``stiffen`` differ only in how the hardening happens,
which is the interesting question: if ``jump`` springs back to the baseline but ``stiffen`` does not,
the compliance was doing the work and it has to be withdrawn slowly.

    python tests/squishyStiffen.py [N] [n]
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

# Best known container side for N unit squares, for reference only.
# UNVERIFIED -- these are quoted from memory and should be checked against a reference before any
# conclusion rests on them.
_BEST_KNOWN = {5: 2.7071, 6: 3.0, 11: 3.877}


def build(N, n, seed = 42, slack = 0.95):
    """N polygons of n vertices at kappa = 4, in a pinned unit box, relaxed onto the depth tier."""
    reference = _BEST_KNOWN.get(N, np.sqrt(N))
    model = pp.Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = N / (reference * slack) ** 2, kappa = 4.0)
    model.syncTargetPerimeters()
    model.syncTargetAreas()
    model.addShape(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    return model


def release(model, N):
    """Decompress to the jamming density and report the equivalent container side.

    The wall tolerance is loosened from its default: it was calibrated on the sharp tier, where a
    converged contact leaves 1.9e-05 of depth, and the depth law's ``k d^2`` wall sits at a much larger
    residual by construction. At the default a packing with EXACTLY zero pair overlap was still
    rejected at 57x the tolerance and decompressed until it pulled away from every wall."""
    meanEdge = float(np.mean(model.packing.targetEdgeLength))
    result = model.energySweep(minimizer = "lbfgs", annealShape = False, finishRigid = False,
                               wallTolerance = 5e-3 * meanEdge)
    return (np.sqrt(N / result.phi) if result.packed else float("nan")), result


def rigidProtocol(N, n):
    model = build(N, 4)
    model.setConstraints(area = True, edge = True)
    model.minimizeFIRE(maxUnbalancedForce = 1e-3, maxSteps = 6000)
    model.setModelType("depth")
    return release(model, N)[0], model


def squishy(N, n):
    """Pack with the shape free apart from kappa, and return the model still compliant."""
    model = build(N, n)
    model.setConstraints(area = True, perimeter = True, edge = False)
    model.minimizeFIRE(maxUnbalancedForce = 1e-3, maxSteps = 6000)
    model.setModelType("depth")
    model.minimizeLBFGS(maxUnbalancedForce = 1e-8, maxSteps = 4000)
    release(model, N)
    return model


def jumpProtocol(N, n):
    model = squishy(N, n)
    model.setShapeTemplate(morph = 1.0, sides = 4)
    model.setConstraints(area = True, edge = True, diagonal = True)
    model.minimizeLBFGS(maxUnbalancedForce = 1e-8, maxSteps = 4000)
    return release(model, N)[0], model


def stiffenProtocol(N, n, rounds = 10):
    """Withdraw the compliance gradually instead of all at once.

    The edge lengths are held only in their DISTRIBUTION while the packing settles, then that
    distribution's width is driven to zero -- which makes every polygon equilateral without ever
    pinning an individual edge. Only at the very end is the square template imposed, by which point
    the shapes are already equilateral at kappa = 4 and the step is small."""
    model = squishy(N, n)
    model.setConstraints(area = True, perimeter = True, edge = [1, 2])
    width = model.getPolydispersity().get("edge", 0.0)
    for step in range(rounds):
        # LINEAR to exactly zero: a geometric ramp aimed at zero spends nearly every step in decades
        # where no geometry changes.
        model.setTargetPolydispersity(width * (1.0 - (step + 1) / rounds))
        model.minimizeLBFGS(maxUnbalancedForce = 1e-8, maxSteps = 4000)
    model.setShapeTemplate(morph = 1.0, sides = 4)
    model.setConstraints(area = True, edge = True, diagonal = True)
    model.minimizeLBFGS(maxUnbalancedForce = 1e-8, maxSteps = 4000)
    return release(model, N)[0], model


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    best = _BEST_KNOWN.get(N)
    print(f"N = {N}, n = {n};  best known side "
          f"{best if best else '?'}  (UNVERIFIED -- check against a reference)")
    print(f"{'protocol':10s} {'side':>9s} {'vs best':>9s} {'kappa':>8s} {'overlap':>11s}  time")
    for name, protocol in (("rigid", rigidProtocol), ("jump", jumpProtocol),
                           ("stiffen", stiffenProtocol)):
        start = time.perf_counter()
        try:
            side, model = protocol(N, n)
        except Exception as error:
            print(f"{name:10s} FAILED: {type(error).__name__}: {str(error).splitlines()[0][:60]}")
            continue
        kappa = float(np.nanmean(model.getShapeIndices()[:N]))
        excess = f"{100 * (side / best - 1):+.1f}%" if best and np.isfinite(side) else "-"
        print(f"{name:10s} {side:9.4f} {excess:>9s} {kappa:8.4f} "
              f"{model.getPairOverlapArea():11.2e}  {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
