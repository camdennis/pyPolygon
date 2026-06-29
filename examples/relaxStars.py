"""Demo: relax several random star polygons to equilateral with eqSoftBody + FIRE.

Run from anywhere:  python examples/relaxStars.py
Prints per-case convergence and saves a before/after figure to
phase1_star_relaxation.png at the repo root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")                                   # headless-safe
import matplotlib.pyplot as plt

from packing import Packing
from softBody import eqSoftBodyEnergyForce, backboneEdgeLengths, backboneArea
from minimize import minimizeFIRE
from visualize import plotBackbone


def regularArea(n, edgeLength):
    """Area of the regular n-gon with the given edge length (the equilateral max)."""
    R = edgeLength / (2 * np.sin(np.pi / n))
    return 0.5 * n * R ** 2 * np.sin(2 * np.pi / n)

def relaxStar(n, seed, edgeLength, areaFraction = 0.7, kEdge = 1.0, kArea = 1.0):
    """Seed a random star (n, seed) and relax it to edge=edgeLength, area=fraction*max."""
    targetArea = areaFraction * regularArea(n, edgeLength)
    pk = Packing.fromSinglePolygon(n, rng = seed, targetEdgeLength = edgeLength,
                                   targetArea = targetArea)
    seedPositions = pk.positions.copy()
    fe = lambda p: eqSoftBodyEnergyForce(p, kEdge, kArea)
    energy, steps, converged = minimizeFIRE(pk, fe, maxSteps = 50000)
    return pk, seedPositions, targetArea, energy, steps, converged

def main():
    cases = [(5, 0, 0.30), (7, 1, 0.25), (9, 2, 0.20), (12, 3, 0.16)]
    fig, axes = plt.subplots(2, len(cases), figsize = (3.4 * len(cases), 7))
    for col, (n, seed, L) in enumerate(cases):
        pk, seedPos, targetArea, energy, steps, converged = relaxStar(n, seed, L)
        plotBackbone(Packing(seedPos, [0, n]), ax = axes[0, col])
        axes[0, col].set_title(f"n={n} random star (seed)")
        plotBackbone(pk, ax = axes[1, col])
        ed = backboneEdgeLengths(pk)
        axes[1, col].set_title(
            f"relaxed: edge {ed.mean():.3f} (std {ed.std():.0e})\n"
            f"area {backboneArea(pk)[0]:.3f} / target {targetArea:.3f}"
        )
        print(f"n={n:2d} seed={seed} conv={converged} steps={steps:5d} "
              f"E={energy:.2e} edgeStd={ed.std():.1e} "
              f"area={backboneArea(pk)[0]:.4f} (target {targetArea:.4f})")
    fig.tight_layout()
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "phase1StarRelaxation.png"))
    fig.savefig(out, dpi = 110)
    print("saved", out)


if __name__ == "__main__":
    main()
