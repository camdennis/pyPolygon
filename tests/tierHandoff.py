"""One benchmarked run that crosses TIERS, inside its own process.

``tests/tierHandoff.ipynb`` is the experiment; this is the part a subprocess needs, in the same split
as ``tests/minimizerBenchmark.py`` and for the same reason.

WHY THIS EXPERIMENT EXISTS. Benchmarking a minimizer on one tier turned out to be the wrong question,
because neither tier can do the whole job:

  * The DEPTH law is invalid at deep overlap. Past the medial-axis ridge its repulsion reverses sign
    and bodies are pulled through, so a random or heavily overlapped start does not converge slowly --
    it converges confidently to a stacked configuration that is a genuine force-balanced minimum. The
    monitor for this is ``maximumDepth / inradius``, and the law wants it far below 1.
  * The SHARP law is only C1: its gradient jumps whenever a vertex crosses an edge. Measured on the
    sharp tier, a strong-Wolfe line search spends 17 to 35 force evaluations per step in some
    configurations against 2.04 in others, with no warning in between -- the curvature condition is
    asking for a specific value of a quantity that is discontinuous.

So the two-tier pipeline is forced rather than convenient, and the interesting free parameter is not
which minimizer to use but WHEN TO CROSS. This measures that.

THE CROSSING CRITERION IS THE POINT. A step count is the naive choice and is not expected to be the
right one; the principled criterion is the depth law's own validity ratio, which says when the second
tier is applicable at all. Both are measured here so they can be compared rather than assumed.

    python tests/tierHandoff.py            # a short self-test
"""

# UNVERIFIED(Cam)

import json
import os
import sys
import time
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polyContact
from minimizerBenchmark import CHUNK, _countEvaluations, feasibility, runStage, turning
from model import Model

# Above this ratio the depth law is outside its validity limit and its force can point the wrong way.
# polyContact.maximumDepth states the requirement as dMax / rIn << 1; 0.25 is a working reading of
# "<<" and is reported alongside the raw ratio so a different reading costs nothing.
VALIDITY_LIMIT = 0.25


def buildSystem(N, n, mode, phi = 1.0, kappa = 4.0, seed = 42, place = "random"):
    """The same construction as the single-tier benchmark, starting on the SHARP tier.

    ``place`` is the seeding, and it is a variable here rather than a constant: ``random`` is the case
    the depth law cannot start from, and ``grid`` is the overlap-free arrangement that lets a
    depth-only schedule be run as a control."""
    if n < 4 or (n & (n - 1)) != 0:
        raise ValueError(f"n = {n} must be a power of two and at least 4; the build doubles from 4.")
    model = Model(N = N, n = 4, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = kappa)
    model.syncTargetAreas()
    model.syncTargetPerimeters()
    if n > 4:
        model.doubleNumEdges(int(np.log2(n // 4)))
    if place == "grid":
        model.placeOnGrid()
    elif place == "random":
        model.placeRandomly()
    elif place != "asBuilt":
        raise ValueError(f"place must be 'random', 'grid' or 'asBuilt', got {place!r}")
    model.setModelType("sharp")
    if mode == "constrained":
        model.setConstraints(area = True, edge = True)
    elif mode != "springs":
        raise ValueError(f"mode must be 'springs' or 'constrained', got {mode!r}")
    return model


def validity(model, samples = 60):
    """``max over pairs of (maximumDepth / inradius)`` -- how far outside the depth law the state is.

    Computed from the loops directly rather than through ``polyContactSystem.systemValidity``, which
    takes that module's own ``bodies`` object; there is no adapter from a packing. Coarser sampling
    than the 200 that function defaults to, because this is called between chunks rather than once.

    Returns 0.0 when nothing overlaps, which is the state the depth tier wants to be handed.

    IT IS CONFOUNDED BY FOLDING, and the confound runs the wrong way. ``inradius`` is the denominator,
    and a self-intersecting polygon has almost none, so a folded state reports a LARGE ratio whether or
    not any pair is deeply overlapped. Measured on a random seed at N = 5: the ratio went 0.871 at the
    start to 0.998 after the sharp tier had driven the overlap to zero -- the polygons had folded, and
    the ratio was reading that rather than any penetration. Read it beside ``feasibility()['simple']``,
    never alone."""
    starts = np.asarray(model.packing.startIndices, dtype = int)
    container = getattr(model.packing, "containerIndex", None)
    bodies = [p for p in range(model.getNumPolygons())
              if container is None or p != int(container)]
    loops = {p: model.getVertices(p) for p in bodies}
    radii = {p: polyContact.inradius(loops[p]) for p in bodies}
    worst = 0.0
    for index, first in enumerate(bodies):
        for second in bodies[index + 1:]:
            depth = polyContact.maximumDepth(loops[first], loops[second], samples)
            if depth == 0.0:
                continue
            worst = max(worst, depth / max(min(radii[first], radii[second]), 1e-300))
    return float(worst)


def applyTier(model, tier, stiffness = 1.0, wallStiffness = 1.0):
    """Switch the contact law. Returns the model."""
    if tier == "sharp":
        model.setModelType("sharp")
    elif tier == "depth":
        model.setDepthContact(stiffness = stiffness, wallStiffness = wallStiffness)
    else:
        raise ValueError(f"tier must be 'sharp' or 'depth', got {tier!r}")
    return model


def crossWhenValid(model, kind, maxSteps, limit, forceTarget, energyTarget, deadline, tally, trace,
                   **kwargs):
    """Run ``kind`` on the SHARP tier until the validity ratio falls below ``limit``.

    Returns ``(outcome, steps, ratio)``. The alternative to a step count: it stops when the depth law
    becomes applicable rather than when a counter runs out, so the crossing point adapts to the system
    instead of being tuned per configuration."""
    taken = 0
    ratio = validity(model)
    while taken < maxSteps and ratio > limit:
        if time.time() > deadline:
            return "timeout", taken, ratio
        outcome, steps = runStage(model, kind, min(CHUNK, maxSteps - taken), forceTarget,
                                  energyTarget, deadline, tally, trace, **kwargs)
        taken += steps
        ratio = validity(model)
        if outcome in ("diverged", "timeout"):
            return outcome, taken, ratio
        if outcome in ("force", "energy"):
            return outcome, taken, ratio
    return ("valid" if ratio <= limit else "steps"), taken, ratio


def runOne(spec):
    """One tier-crossing schedule. Returns a JSON-serializable record.

    ``spec["plan"]`` is a list of ``[tier, minimizer, steps]``. When ``spec["cross"]`` is set, the
    FIRST stage instead runs until the validity ratio drops below it and the step count becomes an
    upper bound rather than the criterion."""
    if spec.get("device") == "numpy":
        import cudaOverlap
        cudaOverlap.isAvailable = lambda: False
    model = buildSystem(N = int(spec["N"]), n = int(spec["n"]), mode = spec["mode"],
                        phi = float(spec.get("phi", 1.0)),
                        kappa = float(spec.get("kappa", 4.0)),
                        seed = int(spec.get("seed", 42)),
                        place = spec.get("place", "random"))
    forceTarget = float(spec.get("forceTarget", 1e-9))
    energyTarget = float(spec.get("energyTarget", 0.0))
    wallStiffness = float(spec.get("wallStiffness", 1.0))
    cross = spec.get("cross")

    tally = _countEvaluations(model)
    trace = []
    start = time.time()
    deadline = start + float(spec.get("budget", np.inf))
    stages, outcome = [], "steps"
    for index, (tier, kind, steps) in enumerate([tuple(s) for s in spec["plan"]]):
        applyTier(model, tier, wallStiffness = wallStiffness)
        before = validity(model)
        if index == 0 and cross is not None:
            outcome, taken, after = crossWhenValid(model, kind, int(steps), float(cross),
                                                   forceTarget, energyTarget, deadline, tally, trace)
        else:
            outcome, taken = runStage(model, kind, int(steps), forceTarget, energyTarget,
                                      deadline, tally, trace)
            after = validity(model)
        stages.append({"tier": tier, "minimizer": kind, "steps": taken, "outcome": outcome,
                       "validityBefore": before, "validityAfter": after})
        if outcome in ("diverged", "timeout"):
            break
        # REACHING THE TARGET ON AN EARLY TIER IS NOT THE ANSWER, it is the cue to cross. The sharp
        # tier measures overlap AREA, so its converged state says the bodies do not overlap; it says
        # nothing about the depth-law energy the run is actually being judged on. Stopping here would
        # report a schedule as converged having never run the tier that defines the objective.
        if outcome in ("force", "energy") and index == len(spec["plan"]) - 1:
            break

    record = {"plan": "+".join(f"{tier}/{kind}:{steps}" for tier, kind, steps in spec["plan"]),
              "stages": stages,
              "outcome": outcome,
              "seconds": time.time() - start,
              "steps": sum(stage["steps"] for stage in stages),
              "evaluations": int(tally["count"]),
              "maxForce": float(model.getMaxUnbalancedForce()),
              "energy": float(model.getEnergy()),
              "validity": validity(model),
              "reached": outcome in ("force", "energy"),
              "feasibility": feasibility(model)}
    if spec.get("keepTrace"):
        record["trace"] = trace
    record.update({key: spec[key] for key in ("N", "n", "mode", "place", "cross") if key in spec})
    return record


if __name__ == "__main__" and "--spec" in sys.argv:
    warnings.filterwarnings("ignore")
    print(json.dumps(runOne(json.loads(sys.argv[sys.argv.index("--spec") + 1]))))
    sys.exit(0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("self-test: N = 5, n = 8, springs, thrown in at random\n")
    model = buildSystem(N = 5, n = 8, mode = "springs")
    print(f"  at the random start: validity {validity(model):.3f}   "
          f"(the depth law wants << 1; {VALIDITY_LIMIT} is the working limit here)")
    record = runOne(dict(N = 5, n = 8, mode = "springs",
                         plan = [["sharp", "lbfgs", 400], ["depth", "lbfgs", 400]],
                         cross = VALIDITY_LIMIT, forceTarget = 1e-8))
    for stage in record["stages"]:
        print(f"  {stage['tier']:<6}/{stage['minimizer']:<6} {stage['steps']:>5} steps  "
              f"{stage['outcome']:<8} validity {stage['validityBefore']:.3f} -> "
              f"{stage['validityAfter']:.3f}")
    print(f"  {record['outcome']}  {record['seconds']:.2f} s  max|F| {record['maxForce']:.2e}  "
          f"validity {record['validity']:.3f}  simple {record['feasibility']['simple']}")
