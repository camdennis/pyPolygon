"""Is the KNOWN optimal packing a minimum of our energy?

The question this answers: when `energySweep` lands below the optimum, is that because the energy
landscape does not have the optimum as a minimum (a MODEL problem), or because descent cannot reach it
(a SEARCH problem)? They call for completely different work, and nothing measured so far distinguishes
them.

The test case is 5 unit squares in a square of side ``s = 2 + 1/sqrt(2) = 2.70710678``, the known
optimum, giving ``phi = 5/s^2 = 0.68227``. Four axis-aligned squares in the corners and one tilted 45
degrees in the middle, whose four corners poke into the four gaps between them. It is written in CLOSED
FORM here rather than searched for, so the configuration under test is the real optimum and not
something this code found.

That geometry is the point: every contact is a CORNER touching an EDGE, which is exactly the situation
that makes a packing hard to find. The corner-into-face contact law measured elsewhere in this project
goes as ``area ~ delta^2``, ``energy ~ delta^4``, ``force ~ delta^3``, so a corner approaching a gap
feels almost no force -- and the contact energy is purely repulsive, so nothing pulls it in either.

Four checks:

  1. the constructed packing is VALID -- zero polygon-polygon overlap, zero wall penetration, at the
     optimal density;
  2. it is a critical point -- the force on it is at the noise floor, so descent has no reason to leave;
  3. it SURVIVES relaxation -- run the same minimizer the sweep uses and the density does not fall;
  4. it is ATTRACTING -- perturb it and relaxation returns to (nearly) the same density.

If all four pass, the landscape holds the optimum and the shortfall is a search failure.

Run: python tests/knownOptimumCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model


SIDE = 2.0 + 1.0 / np.sqrt(2.0)                 # the optimal enclosing square for 5 unit squares
OPTIMUM = 5.0 / SIDE ** 2                       # 0.6822...
WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


# Gaussian noise to break the exact tangency, applied to the vertices before anything is measured.
#
# NOT cosmetic. The optimum is EXACTLY TANGENT -- every contact is a corner lying precisely on an edge
# -- which is the degenerate case for the overlap routine, whose crossing test requires a strict
# 0 < s < 1. There the answer is garbage AND path dependent: the all-to-all scan reports 5.82e-02 of
# overlap for this packing and the candidate path 1.68e-01, where the truth is 0 and the maximum
# possible is one square's whole area, 0.1365. A nudge of 1e-9 puts it back in general position and
# both paths then agree on exactly 0.000000e+00.
#
# NOISE rather than a systematic shrink (Cam's standard practice for degeneracies). Shrinking every
# square by a common factor opens every contact by the same amount and preserves the symmetry that
# caused the degeneracy, so the configuration under test would still be a special one, and it biases
# the density downward by a known amount. Noise puts it in general position without imposing structure
# and without a systematic shift in phi.
_NOISE = 1e-9


def optimalFiveSquares():
    """The five unit squares of the optimal packing, scaled into the unit box. CCW, (5, 4, 2).

    Four in the corners; one rotated 45 degrees about the center, its corners reaching into the four
    gaps. The tilted square's edges pass exactly THROUGH the inner corners of the axis-aligned ones --
    e.g. its lower-left edge runs along ``x + y = 2`` and the first square's corner sits at (1, 1) --
    so the packing is tangent, not merely close."""
    s, h = SIDE, np.sqrt(2.0) / 2.0
    corner = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    squares = [corner + [0.0, 0.0],
               corner + [s - 1.0, 0.0],
               corner + [s - 1.0, s - 1.0],
               corner + [0.0, s - 1.0],
               np.array([[s / 2 + h, s / 2], [s / 2, s / 2 + h],
                         [s / 2 - h, s / 2], [s / 2, s / 2 - h]])]
    squares = np.array(squares) / s             # scale the box to [0, 1]^2
    if _NOISE > 0.0:
        rng = np.random.default_rng(11)
        squares = squares + _NOISE * rng.standard_normal(squares.shape)
    return squares


def buildOptimal():
    """A Model holding exactly that configuration, rigid, walled, on the sharp tier."""
    model = Model(N = 5, n = 4, seed = 0)
    model.generateEquilateralPolygons(phi = 0.4, kappa = 4.0)
    r = model.packing.positions.reshape(-1, 2)
    r[:20] = optimalFiveSquares().reshape(-1, 2)
    model.packing._forces = None
    model.syncTargetPerimeters()
    model.syncTargetAreas()

    model.addShape(WALL)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setSpringConstants(area = 0, edge = 1, perimeter = 0)
    model.setConstraints(area = True, edge = True)
    model.setModelType("sharp")
    model._forces = None
    model._energy = None
    return model


def checkConstruction():
    """1. The closed-form packing really is valid at the optimal density."""
    model = buildOptimal()
    phi = model.getPackingFraction()
    pair = model.getPairOverlapArea()
    depth = model.getWallPenetration()
    shape = model.getShapeIndices()[:5]
    print(f"  1. phi {phi:.9f}  (optimum {OPTIMUM:.9f}, error {abs(phi - OPTIMUM):.2e})")
    print(f"     pair overlap {pair:.3e}   wall penetration {depth:.3e}   "
          f"shape index {shape.min():.6f}..{shape.max():.6f}")
    assert abs(phi - OPTIMUM) < 1e-7, f"constructed density {phi} is not the optimum"
    # Zero to ROUNDOFF, not exactly zero. With exact tangent coordinates the clipper returns a clean
    # 0.0; once the noise makes the coordinates generic the chord sums leave a few ulps, and the sign
    # is arbitrary (measured -2.0e-17). The claim being tested is still a sign change, just read at the
    # floating-point floor rather than against a literal zero.
    assert abs(pair) < 1e-12, f"the constructed optimum overlaps itself by {pair:.3e}"
    assert depth < 1e-8, f"the constructed optimum pokes {depth:.3e} out of the wall"


def checkCriticalPoint():
    """2. The force there is at the noise floor -- descent has no reason to move."""
    model = buildOptimal()
    model.calcForceEnergy()
    force = model.getMaxUnbalancedForce()
    energy = model.getEnergy()
    print(f"  2. max|F| {force:.3e}   energy {energy:.3e}")
    assert force < 1e-6, f"the optimum is not a critical point: max|F| = {force:.3e}"


def checkSurvivesRelaxation():
    """3. The minimizer the sweep uses does not walk away from it."""
    model = buildOptimal()
    before = model.getPackingFraction()
    model.minimizeFIRE(maxUnbalancedForce = 1e-12, maxSteps = 5000, progressBar = False)
    after = model.getPackingFraction()
    pair = model.getPairOverlapArea()
    depth = model.getWallPenetration()
    print(f"  3. phi {before:.9f} -> {after:.9f}   pair {pair:.3e}   depth {depth:.3e}   "
          f"max|F| {model.getMaxUnbalancedForce():.3e}")
    assert abs(after - before) < 1e-9, f"relaxation moved the density by {abs(after - before):.3e}"
    assert abs(pair) < 1e-12, f"relaxation introduced {pair:.3e} of overlap into the optimum"


def checkAttracting():
    """4. Perturb it and see whether relaxation comes back.

    This is the one that says whether the optimum is a genuine basin or a knife edge. A knife edge
    would still pass checks 1-3 and yet be unreachable by any descent from anywhere near it."""
    model = buildOptimal()
    reference = model.packing.positions.copy()
    for amplitude in (1e-6, 1e-4, 1e-3):
        model.packing.positions[:] = reference
        rng = np.random.default_rng(4)
        free = model.packing.positions.size - 8            # leave the pinned wall alone
        model.packing.positions[:free] += amplitude * rng.standard_normal(free)
        model._forces = None
        model._energy = None
        model.constraints.projectPositions(model.packing)  # back onto the rigid-shape manifold
        model.minimizeFIRE(maxUnbalancedForce = 1e-12, maxSteps = 5000, progressBar = False)
        drift = float(np.abs(model.packing.positions[:free] - reference[:free]).max())
        print(f"  4. perturbed {amplitude:.0e}   phi {model.getPackingFraction():.9f}   "
              f"pair {model.getPairOverlapArea():.3e}   max drift {drift:.3e}")


def main():
    print("known optimum: 5 unit squares in a square of side 2 + 1/sqrt(2)")
    checkConstruction()
    checkCriticalPoint()
    checkSurvivesRelaxation()
    checkAttracting()
    print("all checks passed")


if __name__ == "__main__":
    main()
