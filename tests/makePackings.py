# UNVERIFIED(Cam)
"""Relax valid square packings and save them, as inputs for the refinement notebook.

Deliberately simple and NOT the corner-cut protocol: the point is to produce a valid packing whose
contact graph can be read, not to set a record. Area tier to untangle, depth tier to settle, then a
bisection down to EXACTLY zero overlap -- the only number in the whole pipeline that is a sign change
rather than a threshold.

n = 5 IS THE SELF-TEST. Its optimum is PROVED at 2 + sqrt(2)/2, so if the refinement recovers that
expression from a numerically relaxed packing, the machinery works end to end. n = 11 is the open case.
"""
import os
import sys
import warnings

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

import records
from model import Model

WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])


# UNVERIFIED(Cam)
def knownFive(noise = 1e-9, seed = 0):
    """The PROVED n = 5 optimum, s = 2 + sqrt(2)/2, jittered to look like a relaxed packing.

    THE SELF-TEST FOR THE WHOLE PIPELINE. Four unit squares sit axis-aligned in the corners, each
    against two walls; the fifth is turned 45 degrees in the middle, and each corner square's inner
    VERTEX lands exactly on a FACE of it -- the tilted square's inradius is 1/2, and the corner sits
    exactly 1/2 from its centre. So every contact is the one primitive the system knows.

    The jitter is 1e-9, which is deliberately the precision a real relaxation delivers: large enough
    that Newton has to do the work, small enough that the contacts are still identifiable at a 1e-6
    tolerance. Recovering `2 + sqrt(2)/2` from this is the evidence that the refinement is sound."""
    side = 2.0 + np.sqrt(2.0) / 2.0
    rng = np.random.default_rng(seed)
    unit = 0.5 * np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    loops = []
    for cx, cy in ((0.5, 0.5), (side - 0.5, 0.5), (side - 0.5, side - 0.5), (0.5, side - 0.5)):
        loops.append(unit[::-1] + np.array([cx, cy]))
    turn = np.pi / 4.0
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    loops.append((unit[::-1] @ rotation.T) + np.array([side / 2.0, side / 2.0]))
    loops.append(np.array([[0.0, 0.0], [0.0, side], [side, side], [side, 0.0]]))

    positions = np.concatenate(loops)
    positions[:-4] += noise * rng.standard_normal(positions[:-4].shape)

    class Saved:
        pass

    packing = Saved()
    packing.positions = positions
    packing.startIndices = np.arange(len(loops) + 1) * 4
    packing.containerIndex = len(loops) - 1
    return packing


# UNVERIFIED(Cam)
def relax(N, seed = 0, start = 0.55):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(N = N, n = 4, seed = seed)
        model.generateEquilateralPolygons(phi = start, kappa = 4.0)
        model.setMonoPerimeter()
        model.placeOnGrid()
    model.addShape(WALL)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setSpringConstants(area = 1.0, edge = 1.0, perimeter = 0.0)
    model.setConstraints(area = True, edge = True)

    model.setModelType("area")
    model.minimizeFIRE(maxUnbalancedForce = 1e-8, maxSteps = 4000)
    model.setModelType("depth")
    model.setDepthContact(stiffness = 1.0, wallStiffness = 10.0)

    # HOLD LOAD WHILE COMPRESSING, OR NOTHING JAMS. Shrinking a non-overlapping packing does not work:
    # the depth law has zero energy AND zero force there, so the minimizer never rearranges anything
    # and the bisection simply finds where the first pair collides. Measured that way, an n = 11 run
    # came out with ONE gap at 1e-08 and the next at 2e-04 -- a single contact and eleven rattlers.
    # holdExcessEnergy is two-sided, so it compresses the loose packing and keeps a small overlap
    # engaged while the squares rearrange; walking the excess down then approaches the jamming point
    # from the side where every contact is still live.
    for excess in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        model.holdExcessEnergy(excess, tolerance = 0.15, maxRounds = 120,
                               maxUnbalancedForce = 1e-10, maxSteps = 8000)
    model.minimizeLBFGS(maxUnbalancedForce = 1e-11, maxSteps = 8000)
    return model, model.getPackingFraction()




# UNVERIFIED(Cam)
def save(model, path):
    packing = model.packing
    np.savez(path, positions = packing.positions.reshape(-1, 2),
             startIndices = np.asarray(packing.startIndices, dtype = int),
             containerIndex = int(getattr(packing, "containerIndex", -1)))


# UNVERIFIED(Cam)
def load(path, containerIndex = None):
    """A minimal stand-in carrying the three attributes the contact code reads."""
    data = np.load(path)

    class Saved:
        pass

    packing = Saved()
    packing.positions = data["positions"]
    packing.startIndices = data["startIndices"]
    if containerIndex is not None:
        index = containerIndex
    else:
        index = int(data["containerIndex"])
    packing.containerIndex = None if index < 0 else index
    return packing


if __name__ == "__main__":
    os.makedirs(os.path.join(root, "data"), exist_ok = True)
    packing = knownFive()
    np.savez(os.path.join(root, "data", "packingKnown5.npz"),
             positions = packing.positions, startIndices = packing.startIndices,
             containerIndex = packing.containerIndex)
    print("wrote data/packingKnown5.npz -- the proved n = 5 optimum, jittered by 1e-9")

    for N, seed in ((5, 0), (11, 0)):
        model, valid = relax(N, seed = seed)
        side = 1.0 / np.sqrt(valid / N)
        path = os.path.join(root, "data", f"packing{N}.npz")
        save(model, path)
        print(f"{records.describe(N, side)}")
        print(f"  saved at the jamming point: excess {model.getExcessEnergy():.2e}, "
              f"overlap {model.getOverlapArea():.3e} -> {os.path.relpath(path, root)}")


# UNVERIFIED(Cam)
def jam(path, ladder = (1e-6, 1e-7, 1e-8, 1e-9), wallStiffness = 10.0, verbose = True):
    """Re-settle a saved packing at its JAMMING POINT, where its contacts are actually engaged.

    A CONTACT GRAPH NEEDS A JAMMED PACKING AND A DECOMPRESSED ONE IS NOT JAMMED. A protocol that ends
    by bisecting down to zero overlap stops at the largest VALID size, and there the depth law has zero
    energy AND zero force -- so the last relaxation has no reason to hold anything in contact and every
    gap opens by whatever the decompression left. Measured on a 26-square packing at +0.214%: gaps
    grading smoothly from 6e-05 to 1e-03, no plateau anywhere in the tolerance ladder, every square a
    rattler.

    UNIFORM COMPRESSION DOES NOT RESCUE IT, which is worth knowing before trying: measured on the same
    packing, growing the squares by 0.1% already drove some pairs into overlap while others were still
    1e-03 apart. The contacts do not engage simultaneously under scaling, so there is no size at which
    they all touch. Only a real relaxation under load rearranges them into simultaneous contact.

    THIS IS THE WEAKER TOOL NOW: ``squeeze.squeeze`` minimizes the box side directly, under linearized
    non-overlap, and does in one pass what walking the load down only approximates. Measured on that
    same 26-square packing, jamming and re-growing left it at +0.760% while the squeeze took it to the
    record exactly. Keep this for settling a packing under a known load; use the squeeze to close it.

    So this walks the excess energy down instead. Each rung compresses or decompresses to hold that
    load and re-relaxes, which lets the squares rearrange while everything stays engaged; by the last
    rung the overlaps are at the excess level and the contact set is sharp. Returns the model -- pass
    ``model.packing`` straight to ``contacts.audit`` or ``refine.refine``."""
    from model import Model, loadPacking

    packing, extras = loadPacking(path)
    model = Model.load(path)
    # The CONTAINER AND THE PINS ARE NOT IN THE NPZ and Model.load says so; without them the wall is an
    # ordinary polygon, phi picks up its negative signed area, and holdExcessEnergy chases a number that
    # means nothing.
    container = int(extras["containerIndex"]) if "containerIndex" in extras else None
    if container is None:
        raise ValueError("this packing has no containerIndex, so there is no box to jam it against")
    model.packing.containerIndex = container
    starts = np.asarray(model.packing.startIndices, dtype = int)
    model.pinVertices(np.arange(int(starts[container]), int(starts[container + 1])))
    model.setBoundaryConditions("fixed")

    # Sharp squares: the refinement solves for exact unit squares, so any rounding is irrelevant here.
    model.setGeometryType("sharp")
    model.syncTargetAreas()
    model.syncTargetPerimeters()
    model.setSpringConstants(area = 1.0, edge = 1.0, perimeter = 0.0)
    model.setConstraints(area = True, edge = True)
    model.setModelType("depth")
    model.setDepthContact(stiffness = 1.0, wallStiffness = wallStiffness)

    for level in ladder:
        got, phi = model.holdExcessEnergy(level, tolerance = 0.15, maxRounds = 120,
                                          maxUnbalancedForce = 1e-10, maxSteps = 8000)
        if verbose:
            print(f"  excess {got:.2e} (asked {level:.0e})   phi {phi:.6f}   "
                  f"overlap {model.getOverlapArea():.3e}")
    return model
