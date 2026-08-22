# UNVERIFIED(Cam)
"""Corner-cut squares end to end: build, close, refine, and write two files. Prints nothing.

Runs the protocol of ``tests/cornerCutSquares.ipynb`` for ``count`` squares at ``seed``, closes the
slack the final decompression leaves with ``squeeze``, reads the contact graph off the closed packing
and refines it to an exact side, then writes

    <prefix>.png    the packing, colored by how the contacts hold each square
    <prefix>.txt    4 * count vertex positions, box normalized to the UNIT square

with ``prefix`` defaulting to ``~/data/packings/n<count>_s<side>_seed<seed>``. OUTSIDE THE REPO ON
PURPOSE: ``*.png`` is gitignored here, so a run writing into the working tree drops half its output
somewhere git will not keep and the other half where it will.

THE COLORING IS WHAT THE PICTURE IS FOR.

  red     RATTLER -- fewer than three contacts, so the graph does not hold it and it comes out of the
          system before anything is solved.
  amber   an INFINITESIMAL FLEX, a floppy mode: this square's own constraint block is rank-deficient,
          so the contacts do not pin it at FIRST order even though the packing is rigid at second.
          These are not rattlers and deleting them wrecks the packing.
  blue    fully pinned.

The title carries the side against the best known, POSITIVE MEANING BETTER: a smaller container for
the same unit squares is the improvement, so ``+0.05%`` beat the record and ``-0.21%`` fell short.
"""
import argparse
import os
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import mpmath as mp
from matplotlib import pyplot as plt

import contacts as ct
import records
import refine as rf
import squeeze as sq
from model import Model

OUTPUT_DIR = os.path.join("~", "data", "packings")
WALL = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
CORNERS = ((0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5))

BACKBONE_PHI = 1.2
Q_START = 0.85
Q_SPREAD = 0.60
Q_END = 0.02
Q_CAP = 0.995
EXCESS_START = 1e-6
EXCESS_END = 3e-8
ROUNDNESS_EXPONENT = 2.0
ROUNDED_OVERJAM = 1.10


# UNVERIFIED(Cam)
def cornerCut(count, seed, scheduleSteps = 100, exactArcs = True, bisect = False):
    """The corner-cut protocol, verbatim from the notebook with the printing and drawing removed.

    THE FINAL DECOMPRESSION IS OFF BY DEFAULT, because ``squeeze`` runs after this either way and
    redoes its work better. The bisection searches for the size at which the first pair collides WITH
    THE ARRANGEMENT FROZEN, which is exactly what leaves a packing with slack spread unevenly across
    its contacts; the squeeze minimizes the box side over every center, angle and ``s`` at once. It
    costs about twenty-five 4000-step minimizations to produce a state the next stage discards.

    SO WITH ``bisect = False`` THIS RETURNS A MARGINALLY OVERLAPPING PACKING, not a valid one --
    validity becomes ``squeeze.relieve``'s job. Pass ``bisect = True`` if you want this function to
    stand alone."""
    target = records.maximumDensity(count)
    if target is None:
        raise ValueError(f"n = {count} is not in the records table, so the schedule has no target "
                         f"density to come down onto. See records.describe.")

    model = Model(N = count, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = BACKBONE_PHI, kappa = 4.0)
    model.setMonoPerimeter()
    model.placeOnGrid()
    model.addShape(WALL)
    model.pinVertices(np.arange(model.getNumVertices())[-4:])
    model.setBoundaryConditions("fixed")
    model.setSpringConstants(area = 1.0, edge = 1.0, perimeter = 0.0)
    model.setConstraints(area = True, edge = True)
    active = slice(0, 4 * count)

    def applyRoundness(q):
        model.setRho(np.clip(q, 1e-9, Q_CAP) * model.getMaxRho())

    def projectLogNormal(logQ, mean, spread):
        centred = logQ - logQ.mean()
        width = centred.std()
        return mean + (spread / width if width > 1e-12 else 1.0) * centred

    def measuredAreas():
        if model.getGeometryType() == "round":
            return model.getRoundedAreas()[:count]
        return model.getAreas()[:count]

    def scalePolygons(factors):
        r = model.packing.positions.reshape(-1, 2)
        starts = model.packing.startIndices
        for p in range(count):
            block = slice(int(starts[p]), int(starts[p + 1]))
            centroid = r[block].mean(axis = 0)
            r[block] = centroid + factors[p] * (r[block] - centroid)
        model.syncTargetAreas()
        model.syncTargetPerimeters()
        model.packing._forces = None
        model._forces = model._energy = None

    def holdMeasuredArea(q, area):
        scalePolygons(np.sqrt(area / measuredAreas()))
        applyRoundness(q)

    def holdExcess(q, want, area, rounds = 60, tolerance = 0.10, maxStep = 1.01):
        fallbackSlope, previous = 20.0, None
        for _ in range(rounds):
            model.minimizeLBFGS(maxUnbalancedForce = 1e-9, maxSteps = 1000)
            got = model.getExcessEnergy()
            if got > 0.0 and abs(got - want) <= tolerance * want:
                break
            if got <= 0.0:
                step, previous = maxStep, None
            else:
                slope = fallbackSlope
                if previous is not None and abs(np.log(area / previous[0])) > 1e-12:
                    measured = np.log(got / previous[1]) / np.log(area / previous[0])
                    slope = float(np.clip(measured, 2.0, 200.0))
                previous = (area, got)
                step = float(np.exp(np.clip(np.log(want / got) / slope,
                                            -np.log(maxStep), np.log(maxStep))))
            area *= step
            holdMeasuredArea(q, area)
        return model.getExcessEnergy(), area

    def isValid():
        return model.getOverlapArea() == 0.0 and model.getWallPenetration() == 0.0

    def settle(q, area):
        holdMeasuredArea(q, area)
        model.minimizeLBFGS(maxUnbalancedForce = 1e-11, maxSteps = 4000)

    rng = np.random.default_rng(seed)
    q = np.zeros(model.getNumVertices())
    q[active] = np.clip(np.exp(projectLogNormal(rng.standard_normal(4 * count),
                                                np.log(Q_START), Q_SPREAD)), 1e-9, Q_CAP)
    applyRoundness(q)
    model.setGeometryType("round", exact = exactArcs)
    area = ROUNDED_OVERJAM * target / count
    holdMeasuredArea(q, area)

    model.setModelType("area")
    model.minimizeFIRE(maxUnbalancedForce = 1e-6, maxSteps = 2500)

    model.setDepthContact(stiffness = 1.0, wallStiffness = 10.0)
    mean, spread = np.log(Q_START), Q_SPREAD
    for _ in range(8):
        model.minimizeLBFGS(maxUnbalancedForce = 1e-9, maxSteps = 600)
        model.calcForceEnergy()
        gradient = -(model.getRhoForces() * model.getMaxRho())[active]
        scale = max(np.abs(gradient * q[active]).max(), 1e-30)
        logQ = np.log(q[active]) - 0.5 * gradient * q[active] / scale
        q[active] = np.clip(np.exp(projectLogNormal(logQ, mean, spread)), 1e-9, Q_CAP)
        applyRoundness(q)
        holdMeasuredArea(q, area)

    for rung in range(scheduleSteps + 1):
        progress = (rung / scheduleSteps) ** ROUNDNESS_EXPONENT
        mean = np.log(Q_START) + np.log(Q_END / Q_START) * progress
        spread = Q_SPREAD * (1.0 - progress)
        q[active] = np.clip(np.exp(projectLogNormal(np.log(q[active]), mean, spread)),
                            1e-9, Q_CAP)
        applyRoundness(q)
        holdMeasuredArea(q, area)
        want = EXCESS_START * (EXCESS_END / EXCESS_START) ** progress
        _, area = holdExcess(q, want, area)

    model.setGeometryType("sharp")
    model.minimizeLBFGS(maxUnbalancedForce = 1e-10, maxSteps = 4000)
    for level in EXCESS_END * np.array([0.3, 0.1, 0.03, 0.01]):
        _, area = holdExcess(q, level, area, rounds = 40)

    if not bisect:
        return model

    low, high = area, area
    for _ in range(200):
        if isValid():
            break
        low *= 0.995
        settle(q, low)
    else:
        raise RuntimeError(f"no valid packing down to area {low:.6e}: overlap "
                           f"{model.getOverlapArea():.3e}, wall {model.getWallPenetration():.3e}")
    for _ in range(24):
        middle = 0.5 * (low + high)
        settle(q, middle)
        if isValid():
            low = middle
        else:
            high = middle
    settle(q, low)
    backoff = 0.9999
    for _ in range(40):
        if isValid():
            break
        low *= backoff
        backoff *= backoff
        settle(q, low)
    return model


# UNVERIFIED(Cam)
def asPacking(model, count):
    """The model's sharp state as the packing object ``contacts`` and ``squeeze`` read."""

    class Built:
        pass

    packing = Built()
    packing.positions = model.packing.positions.reshape(-1, 2)
    packing.startIndices = np.asarray(model.packing.startIndices, dtype = int)
    packing.containerIndex = count
    return packing


# UNVERIFIED(Cam)
def analyze(packing, count, digits):
    """``(state, rattlers, flexible, result)`` for a packing: squeeze it, audit it, refine it.

    ``result`` is None when the refinement refuses, which is a verdict about the contact graph rather
    than a crash -- a hypostatic graph has no isolated solution to converge to. The caller still gets
    a closed packing and its coloring, so the two files are written either way."""
    squeezed, _ = sq.closePacking(packing, verbose = False)
    state, _ = ct.unitState(squeezed)
    tolerance = ct.suggestTolerance(state, count)
    if tolerance is None:
        warnings.warn("no tolerance separates the contacts even after squeezing; falling back to "
                      "1e-9. The contact graph in the picture may not be the right one.")
        tolerance = 1e-9
    verdict = ct.audit(state, count, tolerance = tolerance)
    ranks = ct.localRanks(state, verdict["equations"], count)
    flexible = {int(i) for i in verdict["free"] if ranks[i] < 4}
    rattlers = {int(i) for i in verdict["rattlers"]}
    try:
        result = rf.refine(squeezed, tolerance = tolerance, digits = digits)
    except Exception as failure:
        # NAME THE TYPE AND THE LINE. mpmath raises ZeroDivisionError with NO MESSAGE from a bare mpf
        # divide, so reporting str(failure) alone printed "refinement refused ()" and said nothing at
        # all about what went wrong or where.
        frame = traceback.extract_tb(failure.__traceback__)[-1]
        warnings.warn(f"refinement refused -- {type(failure).__name__}"
                      f"{': ' + str(failure) if str(failure) else ''} "
                      f"at {os.path.basename(frame.filename)}:{frame.lineno}; "
                      f"reporting the squeezed packing instead, so the side is the squeeze's "
                      f"~1e-12 value rather than an exact one")
        result = None
    return state, rattlers, flexible, result


# UNVERIFIED(Cam)
def unitCorners(state, count, result, digits):
    """``(corners, side)`` with the box scaled to ``[0, 1]^2``. Corners are ``count`` lists of four.

    Refined coordinates are used for every square the contact system solved for; RATTLERS KEEP THEIR
    SQUEEZED ONES, because they were removed from the system and so were never solved for. That is the
    honest thing to write out: their positions are good to the squeeze, not to the refinement."""
    mp.mp.dps = digits
    if result is None:
        side = mp.mpf(float(state[4 * count]))
        rows = [[mp.mpf(float(state[4 * i + k])) for k in range(4)] for i in range(count)]
    else:
        order, refined, side = result["order"], result["state"], result["side"]
        rows = []
        for i in range(count):
            if i in order:
                slot = 4 * order[i]
                rows.append([refined[slot + k] for k in range(4)])
            else:
                rows.append([mp.mpf(float(state[4 * i + k])) for k in range(4)])
    corners = []
    for x, y, a, b in rows:
        corners.append([((x + a * u - b * v) / side, (y + b * u + a * v) / side)
                        for u, v in CORNERS])
    return corners, side


# UNVERIFIED(Cam)
def writePositions(corners, path, digits):
    """Four lines per square, ``x y``, in the order the squares are numbered. Nothing else."""
    with open(path, "w") as handle:
        for loop in corners:
            for x, y in loop:
                handle.write(f"{mp.nstr(x, digits)} {mp.nstr(y, digits)}\n")


# UNVERIFIED(Cam)
def drawPacking(corners, count, side, rattlers, flexible, path):
    """The packing in its unit box, colored by how the contacts hold each square."""
    best = records.bestKnownSide(count)
    if best is None:
        heading = f"n = {count}    s = {mp.nstr(side, 12)}    no published record"
    else:
        # POSITIVE IS BETTER: a smaller container for the same unit squares is the improvement.
        margin = 100.0 * (best - float(side)) / best
        # THE RECORDS TABLE CARRIES ABOUT EIGHT DECIMALS, so a margin below the display resolution is
        # rounding in the published value, not a result. Printed with its sign it reads "-0.0000%",
        # which looks like missing the record by a hair when the two actually agree.
        shown = "0.0000%" if abs(margin) < 5e-5 else f"{margin:+.4f}%"
        heading = f"n = {count}    {shown} vs best known    s = {mp.nstr(side, 12)}"

    figure, axis = plt.subplots(figsize = (6.0, 6.3))
    axis.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill = False, lw = 1.4, color = "0.35"))
    for i, loop in enumerate(corners):
        points = np.array([[float(x), float(y)] for x, y in loop])
        if i in rattlers:
            fill, edge = "#f2b0b0", "#c0392b"
        elif i in flexible:
            fill, edge = "#ffe0a3", "#d68910"
        else:
            fill, edge = "#9ec7ff", "#2a78d6"
        closed = np.vstack([points, points[:1]])
        axis.fill(closed[:, 0], closed[:, 1], color = fill)
        axis.plot(closed[:, 0], closed[:, 1], color = edge, lw = 1.1)
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(heading, fontsize = 11)
    figure.savefig(path, dpi = 220, bbox_inches = "tight")
    plt.close(figure)


# UNVERIFIED(Cam)
def outputStem(count, side, seed):
    """``n026_s5.6213203436_seed0`` -- the count, then THE SIDE, then the seed.

    THE SIDE IS THE RESULT AND THE SEED IS ONLY PROVENANCE, so the side comes first: a listing then
    sorts into what each arrangement is worth, and two seeds that found the same optimum sit next to
    each other instead of being scattered by a number that means nothing.

    The count is ZERO PADDED because a lexicographic listing otherwise reads n11, n17, n26, n268, n27,
    n5. The side is not, and does not need to be: for a fixed count every side is within a percent or
    so of the record, so they all carry the same number of integer digits and sort correctly as they
    stand."""
    return f"n{int(count):03d}_s{float(side):.10f}_seed{int(seed)}"


# UNVERIFIED(Cam)
def main():
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("count", type = int, help = "number of unit squares")
    parser.add_argument("seed", type = int, help = "random seed for the build")
    parser.add_argument("--prefix", default = None,
                        help = "full output path stem; overrides --outputDir")
    parser.add_argument("--outputDir", default = OUTPUT_DIR,
                        help = f"directory for the two files (default {OUTPUT_DIR})")
    parser.add_argument("--steps", type = int, default = 100, help = "roundness schedule rungs")
    parser.add_argument("--digits", type = int, default = 60, help = "refinement precision")
    parser.add_argument("--chorded", action = "store_true",
                        help = "chorded arcs instead of the exact-arc contact law")
    parser.add_argument("--bisect", action = "store_true",
                        help = "also run the protocol's decompression bisection, which squeeze "
                               "otherwise makes redundant")
    given = parser.parse_args()

    model = cornerCut(given.count, given.seed, scheduleSteps = given.steps,
                      exactArcs = not given.chorded, bisect = given.bisect)
    state, rattlers, flexible, result = analyze(asPacking(model, given.count), given.count,
                                                given.digits)
    corners, side = unitCorners(state, given.count, result, given.digits)
    # NAMED AFTER THE RUN FINISHES, because the side is in the name and is not known before then.
    prefix = given.prefix or os.path.join(given.outputDir,
                                          outputStem(given.count, side, given.seed))
    prefix = os.path.expanduser(prefix)
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok = True)
    writePositions(corners, f"{prefix}.txt", given.digits)
    drawPacking(corners, given.count, side, rattlers, flexible, f"{prefix}.png")


if __name__ == "__main__":
    main()
