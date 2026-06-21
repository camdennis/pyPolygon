"""Demo: build packings of N equilateral polygons for a target shape index, area
distribution, and packing fraction; relax them, and plot the polygons + the area
histograms for both distributions.

Run:  python examples/buildPacking.py   ->  phase1Packing.png at the repo root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from packingBuilder import buildEquilateralPacking, shapeBackbones, shapeIndices
from softBody import backboneArea
from distributions import sampleAreas
from visualize import plotBackbone

def main():
    cases = [("logNormal", {"sigma": 0.3}), ("biDisperse", {"sizeRatio": 1.4})]
    fig, axes = plt.subplots(2, 2, figsize = (11, 10))
    for col, (kind, kw) in enumerate(cases):
        pk = buildEquilateralPacking(30, 6, shapeIndex = 4.0, areaKind = kind,
                                     phi = 0.6, rng = 7, **kw)
        energy, steps, converged = shapeBackbones(pk, maxSteps = 300000)
        si = shapeIndices(pk)
        plotBackbone(pk, ax = axes[0, col], showVertices = False)
        axes[0, col].set_title(f"{kind}: N=30 n=6 s=4.0 phi=0.6\n"
                               f"conv={converged} shapeIdx={si.mean():.3f} (std {si.std():.0e})")
        print(f"{kind:11s}: conv={converged} steps={steps:5d} E={energy:.1e} "
              f"shapeIdx={si.mean():.4f} areaSum={backboneArea(pk).sum():.4f}")
        sample = sampleAreas(5000, kind, phi = 1.0, rng = 0, **kw)
        axes[1, col].hist(sample, bins = 45)
        axes[1, col].set_title(f"{kind} area distribution (5000 samples, sum=1)")
        axes[1, col].set_xlabel("area")
        axes[1, col].set_ylabel("count")
    fig.tight_layout()
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "phase1Packing.png"))
    fig.savefig(out, dpi = 110)
    print("saved", out)


if __name__ == "__main__":
    main()
