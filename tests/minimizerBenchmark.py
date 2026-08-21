"""What ONE benchmarked run does, inside its own process. Everything else lives in the notebook.

``tests/minimizerStatistics.ipynb`` is where the benchmark actually is -- the schedules, the race, the
handoff search, the sweep and the plots are all cells there, so they can be read and changed without
touching a module. This file holds only the part that CANNOT live in a cell: a subprocess needs an
importable entry point, and every run here is its own subprocess because the sharp CUDA kernel faults
during line searches (``CUDA error 700``) and poisons the context when it does.

So: the notebook decides what to run; this decides what running one thing means.

THREE KILLERS AND NOTHING ELSE, which is the point of this harness:

    steps        a per-stage step budget
    energy       stop when the total energy reaches a target
    force        stop when max|F| reaches a target

No stall detector (``patience`` is left at None, which is already the default), no noise-floor
warning, no state dumps, no progress bars. A non-finite result is not suppressed -- it is CAUGHT and
recorded as the outcome ``diverged``, because "this schedule blew up" is a benchmark result and
silently continuing past it would not be.

THE WALL-CLOCK BUDGET IS A RACE, not a fourth killer. A schedule is killed when it exceeds the best
time any schedule has yet taken to reach the same target; a schedule that beats it becomes the new
budget. So the first schedule sets the bar and every later one is only allowed the time it takes to
lose.

    python tests/minimizerBenchmark.py            # a short self-test of the harness
"""

# UNVERIFIED(Cam)

import os
import sys
import time
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import minimize
from model import Model

# How many steps a stage runs between killer checks. The minimizers have no abort hook, so the budget
# is spent in chunks and the killers are read between them. Small enough to be responsive, large
# enough that the per-call overhead does not show up in the timing being measured.
CHUNK = 50


def buildSystem(N, n, mode, phi = 1.0, kappa = 4.0, seed = 42, bidisperse = True, ratio = 1.4):
    """``N`` polygons of ``n`` vertices at packing fraction ``phi``, built the way
    ``tests/penetrationDepth.ipynb`` builds them.

    ``mode`` selects the formulation:

        "springs"      unconstrained -- edge and area springs, the shapes are compliant
        "constrained"  setConstraints(area = True, edge = True), the defaults

    ``bidisperse`` applies ``setBiPerimeter(ratio)``, giving the first half of the polygons a target
    perimeter ``ratio`` times the second half's. This is the standard way to keep a packing from
    ordering into a crystal, and it needs an even ``N``.

    THE POLYGONS ARE SELF-INTERSECTING AT HIGH n AND THAT IS NORMAL HERE. Measured on the reference
    setup (N = 32, n = 32, phi = 1, kappa = 4): the sum of turning angles is 1234.4 degrees straight
    out of ``generateEquilateralPolygons``, against the 360 of a simple polygon. Minimization then
    REDUCES it -- 1234.4 to 810.2 after 1000 sharp FIRE steps, to 719.9 after depth L-BFGS, arriving
    at max|F| = 2.2e-11. So ``turning`` is reported as an observable rather than enforced as a gate;
    a benchmark that rejected folded configurations would reject the whole working regime.

    Periodic, with no container, so phi is the cell fraction."""
    model = Model(N = N, n = n, seed = seed)
    model.generateEquilateralPolygons(phi = phi, kappa = kappa)
    if bidisperse:
        if N % 2:
            raise ValueError(f"setBiPerimeter needs an even N, got {N}.")
        model.setBiPerimeter(ratio)
    model.setBoundaryConditions("periodic")
    model.setModelType("sharp")
    if mode == "constrained":
        model.setConstraints(area = True, edge = True)
    elif mode != "springs":
        raise ValueError(f"mode must be 'springs' or 'constrained', got {mode!r}")
    model.initForceEnergy()
    return model


def turning(model):
    """Worst sum of |turning angle| over the polygons. 360 is simple; more means FOLDED.

    Reported with every run because nothing in either formulation forbids self-intersection, and a
    folded polygon reports a healthy energy the whole time."""
    starts = np.asarray(model.packing.startIndices, dtype = int)
    container = getattr(model.packing, "containerIndex", None)
    worst = 0.0
    for polygon in range(model.getNumPolygons()):
        if container is not None and polygon == int(container):
            continue
        loop = model.getVertices(polygon)
        behind = loop - np.roll(loop, 1, axis = 0)
        ahead = np.roll(loop, -1, axis = 0) - loop
        angle = np.degrees(np.arctan2(np.cross(behind, ahead),
                                      np.einsum("ij,ij->i", behind, ahead)))
        worst = max(worst, float(np.abs(angle).sum()))
    return worst


def _countEvaluations(model):
    """Wrap the model's force routine so evaluations can be counted, and return a reset handle.

    The cost of a step is not one force evaluation for the line-search minimizers -- strong Wolfe
    calls the objective several times, and under constraints every trial point also pays a SHAKE
    retraction. Evaluations per step is the quantity CG and L-BFGS actually differ in, so it is
    counted rather than assumed."""
    original = model._forceEnergy
    tally = {"count": 0}

    def counted(packing):
        tally["count"] += 1
        return original(packing)

    model._forceEnergy = counted
    return tally


def _stageOnce(model, kind, steps, **kwargs):
    """One chunk of one minimizer, with every guardrail off. Returns nothing; state is on the model."""
    common = dict(maxSteps = steps, progressBar = False, patience = None)
    if kind == "fire":
        model.minimizeFIRE(fThreshold = 0.0, **common, **kwargs)
    elif kind == "lbfgs":
        model.minimizeLBFGS(fThreshold = 0.0, **common, **kwargs)
    elif kind == "cg":
        model.minimizeCG(fThreshold = 0.0, **common, **kwargs)
    else:
        raise ValueError(f"unknown minimizer {kind!r}; use 'fire', 'lbfgs' or 'cg'.")


def runStage(model, kind, steps, forceTarget, energyTarget, deadline, tally, trace,
             foldTolerance = None, **kwargs):
    """Spend up to ``steps`` on one minimizer, checking the killers between chunks.

    Returns ``(outcome, stepsTaken)`` where outcome is one of ``force``, ``energy``, ``steps``,
    ``timeout``, ``folded`` or ``diverged``. ``fThreshold`` is pinned to 0 inside the chunks so that
    the minimizer's own convergence test never fires; the conditions here are the only stopping rule.

    ``foldTolerance`` is OFF by default. Self-intersection is the normal state of this build at high
    n -- the reference setup starts at 1234.4 degrees of turning and minimization reduces it -- so
    rejecting on it would reject the working regime. Set it to a number of degrees only when
    benchmarking a configuration that is expected to stay simple, such as n = 4."""
    taken = 0
    while taken < steps:
        if time.time() > deadline:
            return "timeout", taken
        chunk = min(CHUNK, steps - taken)
        try:
            _stageOnce(model, kind, chunk, **kwargs)
        except FloatingPointError:
            # Recorded, not suppressed: a schedule that blows up has produced a result.
            return "diverged", taken
        taken += chunk
        force = model.getMaxUnbalancedForce()
        energy = model.getEnergy()
        trace.append((kind, taken, time.time(), float(energy), float(force),
                      int(tally["count"])))
        if not np.isfinite(force) or not np.isfinite(energy):
            return "diverged", taken
        if foldTolerance is not None and abs(turning(model) - 360.0) > float(foldTolerance):
            return "folded", taken
        if force <= forceTarget:
            return "force", taken
        if energy <= energyTarget:
            return "energy", taken
    return "steps", taken


def runPlan(model, plan, forceTarget = 1e-9, energyTarget = 0.0, budget = np.inf,
            foldTolerance = None, **kwargs):
    """Run a SCHEDULE of stages, e.g. ``[("fire", 2000), ("lbfgs", 10000)]``.

    ``budget`` is the wall-clock allowance in seconds -- the current best time, so a schedule that
    cannot beat the incumbent is killed rather than finished. Returns a record dict."""
    tally = _countEvaluations(model)
    trace = []
    start = time.time()
    deadline = start + float(budget)
    outcome, stages = "steps", []
    for kind, steps in plan:
        outcome, taken = runStage(model, kind, steps, forceTarget, energyTarget, deadline,
                                  tally, trace, foldTolerance = foldTolerance, **kwargs)
        stages.append({"minimizer": kind, "steps": taken, "outcome": outcome})
        if outcome != "steps":
            break
    elapsed = time.time() - start
    force = model.getMaxUnbalancedForce()
    energy = model.getEnergy()
    return {"plan": "+".join(f"{kind}:{steps}" for kind, steps in plan),
            "stages": stages,
            "outcome": outcome,
            "seconds": elapsed,
            "steps": sum(stage["steps"] for stage in stages),
            "evaluations": int(tally["count"]),
            "maxForce": float(force),
            "energy": float(energy),
            "turning": turning(model),
            "reached": outcome in ("force", "energy"),
            "rejected": outcome == "folded",
            "trace": trace}


def feasibility(model, tolerance = 1e-9):
    """What the run is allowed to claim afterwards, as a dict.

    A low energy is not by itself a good answer. These are the ways a run can report one without
    having produced a packing: a FOLDED polygon whose signed area is the difference of its lobes, a
    constrained run whose retraction quietly stopped converging, and residual overlap that a
    near-zero energy can still hide on a compliant system."""
    record = {"turning": turning(model),
              "simple": abs(turning(model) - 360.0) < 5.0,
              "pairOverlap": float(model.getPairOverlapArea()),
              "packingFraction": float(model.getPackingFraction())}
    if model.constraints is not None:
        residual = float(model.constraints.maxResidual(model.packing))
        record["maxResidual"] = residual
        record["constraintsHeld"] = residual < tolerance
    areas = np.asarray(model.getAreas(), dtype = float)
    targets = np.asarray(model.getTargetAreas(), dtype = float)
    record["worstAreaError"] = float(np.abs(areas / targets - 1.0).max())
    edges = np.asarray(model.getEdgeLengths(), dtype = float)
    edgeTargets = np.asarray(model.getTargetEdgeLengths(), dtype = float)
    record["worstEdgeError"] = float(np.abs(edges / edgeTargets - 1.0).max())
    return record


def runOne(spec):
    """Build and run one configuration from a plain dict. Returns a JSON-serializable record.

    Separated from ``runPlan`` so a whole run can be described by data and handed to a subprocess."""
    if spec.get("device") == "numpy":
        import cudaOverlap
        cudaOverlap.isAvailable = lambda: False
    model = buildSystem(N = int(spec["N"]), n = int(spec["n"]), mode = spec["mode"],
                        phi = float(spec.get("phi", 1.0)),
                        kappa = float(spec.get("kappa", 4.0)),
                        seed = int(spec.get("seed", 42)))
    record = runPlan(model, [tuple(stage) for stage in spec["plan"]],
                     forceTarget = float(spec.get("forceTarget", 1e-9)),
                     energyTarget = float(spec.get("energyTarget", 0.0)),
                     budget = float(spec.get("budget", np.inf)),
                     foldTolerance = spec.get("foldTolerance", None))
    if not spec.get("keepTrace", False):
        record.pop("trace", None)
    record["feasibility"] = feasibility(model)
    record.update({key: spec[key] for key in ("N", "n", "mode") if key in spec})
    return record


if __name__ == "__main__" and "--spec" in sys.argv:
    import json
    warnings.filterwarnings("ignore")
    print(json.dumps(runOne(json.loads(sys.argv[sys.argv.index("--spec") + 1]))))
    sys.exit(0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("harness self-test: N = 4, n = 8, both formulations, tiny budgets\n")
    for mode in ("springs", "constrained"):
        model = buildSystem(N = 4, n = 8, mode = mode)
        print(f"  {mode:<12} phi {model.getPackingFraction():.4f}  "
              f"|turn| {turning(model):.2f}  energy {model.calcForceEnergy().getEnergy():.6e}")
        record = runPlan(model, [("fire", 200), ("lbfgs", 200)], forceTarget = 1e-6)
        print(f"    {record['outcome']:<9} {record['seconds']:.2f} s  "
              f"max|F| {record['maxForce']:.3e}  energy {record['energy']:.6e}  "
              f"{record['evaluations']} evaluations")
        print(f"    feasibility {feasibility(model)}\n")
