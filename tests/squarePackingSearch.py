"""Search for optimal EQUAL-square packings using transient target degrees of freedom.

The optimal packing of N equal squares in a unit square is a JAMMING THRESHOLD: a perfect packing has
no overlap and nothing outside the wall, so E = 0 exactly, and above the optimal phi no configuration
reaches zero. So the quantity to find is the largest phi with min E = 0, located by bisection -- not
a minimization at a single phi.

The hard-square landscape is glassy, and fixed-target relaxation simply falls into whatever basin it
started in. The transient targets are what gets out: with the sizes free (subject to conserved
moments) a jammed packing can relieve stress along directions that do not exist when the targets are
frozen. Annealing the width of the size distribution from broad to zero then follows a solution
branch instead of dropping into a random basin:

    polydisperse (easy, shallow, many near-degenerate solutions)  ->  monodisperse (the hard problem)

This is a HEURISTIC, not a global optimizer: it beats naive fixed-target relaxation but certifies
nothing. Run several seeds. For N = 5 the known optimum is side 1/(2 + 1/sqrt(2)), i.e.
phi = 5 * side^2 = 0.68215..., with four squares axis-aligned and one rotated 45 degrees -- which
makes it a good test of whether the anneal escapes the symmetric basin.

Run:  python tests/squarePackingSearch.py [--squares 5] [--seeds 4]
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model

warnings.filterwarnings("ignore")

WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])   # clockwise: a container
ANNEAL = (0.20, 0.15, 0.10, 0.06, 0.03, 0.015, 0.0)


def knownOptimum(numSquares):
    """Published optimal packing fraction, where I have it, else None."""
    side = {5: 1.0 / (2.0 + 2.0 ** -0.5), 2: 0.5, 4: 0.5, 9: 1.0 / 3.0}
    return None if numSquares not in side else numSquares * side[numSquares] ** 2


def relaxAt(numSquares, phi, seed, softening, anneal = True, maxSteps = 4000):
    """Relax one packing at the given phi. Returns (energy, model)."""
    model = Model(N = numSquares, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = 4.0)
    model.setMonoPerimeter()
    if anneal:
        model.setLogNormalTargetArea(polydispersity = ANNEAL[0])
    model.addShape(WALL)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setSpringConstants()
    model.setModelType("mollified")
    model.setSofteningFraction(softening)

    # Positions first -- the reference notes convergence is far more robust that way.
    model.minimizeFIRE(maxUnbalancedForce = 1e-4, maxSteps = maxSteps)
    if anneal:
        model.setMoments([1, 2, -1])
        model.setDOFType("transient")
        for cv in ANNEAL[1:]:
            model.setTargetPolydispersity(cv)
            model.minimizeFIRE(maxUnbalancedForce = 1e-6, maxSteps = maxSteps)
        model.setDOFType("fixed")
    # Sharpen the contact and polish: a rounded square packs differently from a sharp one.
    model.setSofteningFraction(softening / 4.0)
    model.minimizeFIRE(maxUnbalancedForce = 1e-8, maxSteps = maxSteps)
    return model.getEnergy(), model


def bestOverSeeds(numSquares, phi, seeds, softening, anneal):
    """Lowest energy found at this phi across seeds (the landscape is glassy, so sample it)."""
    return min(relaxAt(numSquares, phi, seed, softening, anneal)[0] for seed in range(seeds))


def bisect(numSquares, seeds, softening, anneal, low, high, rounds, tolerance):
    """Largest phi whose best energy still looks like a valid packing."""
    for _ in range(rounds):
        mid = 0.5 * (low + high)
        energy = bestOverSeeds(numSquares, mid, seeds, softening, anneal)
        packed = energy < tolerance
        print(f"      phi = {mid:.4f}   best E = {energy:.3e}   {'packs' if packed else 'jams'}")
        if packed:
            low = mid
        else:
            high = mid
    return low


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squares", type = int, default = 5)
    parser.add_argument("--seeds", type = int, default = 4)
    parser.add_argument("--rounds", type = int, default = 6)
    parser.add_argument("--softening", type = float, default = 0.06)
    parser.add_argument("--tolerance", type = float, default = 1e-6)
    args = parser.parse_args()

    optimum = knownOptimum(args.squares)
    print(f"\n{args.squares} equal squares in a unit square, {args.seeds} seeds per phi")
    if optimum is not None:
        print(f"known optimum: phi = {optimum:.5f}")

    for label, anneal in (("fixed targets (baseline)", False),
                          ("transient targets (annealed)", True)):
        print(f"\n  {label}")
        found = bisect(args.squares, args.seeds, args.softening, anneal,
                       low = 0.40, high = 0.85, rounds = args.rounds,
                       tolerance = args.tolerance)
        line = f"  -> best phi = {found:.5f}"
        if optimum is not None:
            line += f"   ({found / optimum * 100:.1f}% of optimum)"
        print(line)


if __name__ == "__main__":
    main()
