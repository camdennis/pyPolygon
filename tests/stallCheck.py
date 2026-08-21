"""Checks on the stall detector: does it fire when the steps stop paying, and NEVER when they do?

``maxSteps`` is a blunt instrument -- a run that has reached a floor spends its whole budget proving
it, and a run that needed twice the budget is silently truncated with no way to tell the two apart.
The detector watches the residual instead: ``patience`` steps must divide it by ``stallFactor``.

  0  a healthy run is never interrupted, including a NON-MONOTONE one (FIRE overshoots by design)
  1  a hard floor is caught, and named 'flat' rather than blamed on the budget
  2  the noise floor is caught and named 'noise', because that case is CONVERGED, not failed
  3  slow-but-real convergence is named 'slow' AND its projected step count is arithmetically right
  4  patience trades steps for certainty in the direction you would expect
  5  the up-front reachability check fires on a sub-noise tolerance, before any work is done
  6  end to end on a real numpy FIRE run, with no fabricated trace anywhere

The traces in 0-4 are synthetic ON PURPOSE. A detector has to be judged against inputs whose right
answer is known independently, and a real minimizer's trace is exactly what is not known in advance --
check 6 then confirms the wiring against a real one.

    python tests/stallCheck.py
"""

# UNVERIFIED(Cam)

import os
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ROOT))
sys.path.insert(0, _ROOT)

import minimize
from minimize import _Stall, NOISE_FLOOR


def run(trace, patience = 200, factor = 2.0, threshold = 1e-9, maxSteps = None):
    """Feed a residual trace to the detector; return (stepStopped or None, the detector).

    ``maxSteps`` defaults to the trace length, which is what a real minimizer would pass -- the 'slow'
    verdict is BUDGET-AWARE, so a detector given no budget can never reach it."""
    stall = _Stall(patience, factor, threshold, len(trace) if maxSteps is None else maxSteps)
    for step, residual in enumerate(trace):
        if stall.update(step, residual):
            return step, stall
    return None, stall


def checkHealthy():
    """CHECK 0: a run that is converging must never be cut off -- including a non-monotone one.

    The non-monotone case is the one that matters. FIRE is damped dynamics: it overshoots, zeroes its
    velocity and climbs before falling again, so its residual is not a decreasing sequence. A detector
    reading the CURRENT value would fire on an ordinary overshoot and truncate a perfectly healthy
    run; reading the window MINIMUM does not."""
    ok = True
    steps = np.arange(3000)

    smooth = 10.0 ** (-steps / 200.0)
    stopped, _ = run(smooth)
    ok = ok and stopped is None
    print(f"  smooth, 1 decade / 200 steps      stopped at {stopped}   {'ok' if stopped is None else 'FAIL'}")

    # FIRE-like: a falling envelope with violent excursions ABOVE it.
    rng = np.random.default_rng(0)
    spiky = 10.0 ** (-steps / 200.0) * (1.0 + 40.0 * rng.random(steps.size) ** 6)
    stopped, _ = run(spiky)
    ok = ok and stopped is None
    print(f"  same, with 40x overshoots         stopped at {stopped}   {'ok' if stopped is None else 'FAIL'}")

    # Only just paying its way: exactly the factor per window, so it must survive.
    marginal = 10.0 ** (-steps * np.log10(2.5) / 200.0)
    stopped, _ = run(marginal)
    ok = ok and stopped is None
    print(f"  marginal, x2.5 per window         stopped at {stopped}   {'ok' if stopped is None else 'FAIL'}")
    print(f"  CHECK 0 healthy runs survive {'PASS' if ok else 'FAIL'}")
    return ok


def checkFlat():
    """CHECK 1: a residual pinned far above the noise floor is 'flat' -- a floor of the ENERGY.

    This is the sharp tier's C1 kink, where FIRE converges linearly to a corner it cannot pass. The
    library's answer until now was ``anneal._SHARP_TOLERANCE = 1e-4``, a hardcoded per-tier number;
    the rate test reaches the same conclusion without being told which tier it is on."""
    trace = np.full(3000, 7.44e-03)                # the measured sharp-tier plateau
    stopped, stall = run(trace)
    ok = stopped is not None and stall.reason == "flat"
    print(f"  flat at 7.44e-03: stopped at step {stopped}, reason {stall.reason!r}")
    print(f"    -> {stall.describe('FIRE')[:110]}...")
    print(f"  CHECK 1 hard floor named 'flat' {'PASS' if ok else 'FAIL'}")
    return ok


def checkNoise():
    """CHECK 2: at the noise floor the verdict is 'noise', which means CONVERGED.

    Distinguishing this from 'flat' is the difference between "your answer is good, your tolerance was
    impossible" and "your energy has a kink". Same trace shape, opposite advice."""
    rng = np.random.default_rng(1)
    trace = NOISE_FLOOR * (1.0 + 0.3 * rng.standard_normal(3000) ** 2)
    stopped, stall = run(trace, threshold = 1e-13)
    ok = stopped is not None and stall.reason == "noise"
    print(f"  jitter at {NOISE_FLOOR:.0e}: stopped at step {stopped}, reason {stall.reason!r}")
    print(f"    -> {stall.describe('L-BFGS')[:110]}...")
    print(f"  CHECK 2 noise floor named 'noise' {'PASS' if ok else 'FAIL'}")
    return ok


def checkSlow():
    """CHECK 3: 'slow' must come with a projected step count that is ARITHMETICALLY right.

    The projection is the actionable part -- "40,000 more steps" is a decision the caller can make,
    where "did not converge" is not. Checked against the trace's own known decay rate rather than
    against the formula that produced it."""
    ok = True
    for decadeSteps in (5000.0, 20000.0):
        steps = np.arange(4000)
        start = 1e-4
        trace = start * 10.0 ** (-steps / decadeSteps)
        stopped, stall = run(trace, threshold = 1e-9, maxSteps = 4000)
        # Truth: from where it stopped, at this trace's rate, how many steps to reach 1e-9?
        truth = np.log10(stall.best / 1e-9) * decadeSteps
        projected = np.log10(stall.best / stall.threshold) / stall.rate
        good = stall.reason == "slow" and abs(projected / truth - 1.0) < 1e-9
        ok = ok and good
        print(f"  1 decade / {decadeSteps:6.0f} steps: reason {stall.reason!r}   "
              f"projected {projected:11,.0f}   true {truth:11,.0f}   {'ok' if good else 'FAIL'}")
    print(f"  CHECK 3 'slow' projection is exact {'PASS' if ok else 'FAIL'}")
    return ok


def checkPatience():
    """CHECK 4: patience buys certainty at the price of steps, monotonically.

    A detector whose behaviour was not monotone in its own knob would be untunable. The trace decays
    for a while and then dies, so a longer patience must always wait longer before calling it."""
    steps = np.arange(6000)
    trace = np.where(steps < 1000, 10.0 ** (-steps / 400.0), 10.0 ** (-1000.0 / 400.0))
    previous = -1
    ok = True
    for patience in (50, 100, 200, 400, 800):
        stopped, stall = run(trace, patience = patience)
        good = stopped is not None and stopped > previous
        ok = ok and good
        print(f"  patience {patience:4d}: stopped at step {stopped:5d}, reason {stall.reason!r}   "
              f"{'ok' if good else 'FAIL'}")
        previous = stopped
    print(f"  CHECK 4 monotone in patience {'PASS' if ok else 'FAIL'}")
    return ok


def checkReachability():
    """CHECK 5: a sub-noise tolerance is refused UP FRONT, before a single step is spent.

    No stopping rule can rescue this one -- the residual is not converging slowly, it has already
    arrived and is jittering. The only useful moment to say so is before the run."""
    ok = True
    for threshold, expected in ((1e-13, True), (3e-12, True), (1e-11, True), (1e-9, False),
                                (1e-6, False)):
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            minimize.checkReachable(threshold)
        fired = len(caught) > 0
        good = fired == expected
        ok = ok and good
        print(f"  tolerance {threshold:.0e}: warned {fired}   expected {expected}   "
              f"{'ok' if good else 'FAIL'}")
    print(f"  CHECK 5 sub-noise tolerance refused up front {'PASS' if ok else 'FAIL'}")
    return ok


def checkRealRun():
    """CHECK 6: wired into a real minimizer, on a real trace, with no CUDA.

    The free-space backbone relax is pure numpy, so this exercises the plumbing -- signature, loop
    test, warning, ``stopReason`` -- without needing the card. Two runs from the same seed: one with
    a reachable tolerance that must converge untouched, one with an impossible one that must stall
    rather than spend its whole budget."""
    import build

    ok = True
    packing = build.buildEquilateralPacking(6, 8, 4.0, areaKind = "mono", phi = 0.3, rng = 1)
    forceEnergy = lambda p: __import__("softBody").eqSoftBodyEnergyForce(p, 1.0, 1.0)
    energy, steps, converged = minimize.minimizeFIRE(
        packing, forceEnergy, maxSteps = 40000, fThreshold = 1e-8, patience = 500, progress = False)
    good = converged and getattr(packing, "stopReason", None) is None
    ok = ok and good
    print(f"  reachable 1e-8: converged {converged} in {steps} steps, "
          f"reason {getattr(packing, 'stopReason', None)!r}   {'ok' if good else 'FAIL'}")

    packing = build.buildEquilateralPacking(6, 8, 4.0, areaKind = "mono", phi = 0.3, rng = 1)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        energy, steps, converged = minimize.minimizeFIRE(
            packing, forceEnergy, maxSteps = 40000, fThreshold = 1e-20, patience = 500,
            progress = False)
    reason = getattr(packing, "stopReason", None)
    good = (not converged) and reason is not None and steps < 40000 and len(caught) > 0
    ok = ok and good
    print(f"  impossible 1e-20: stopped at step {steps} of 40000, reason {reason!r}, "
          f"{len(caught)} warning(s)   {'ok' if good else 'FAIL'}")
    if caught:
        print(f"    -> {str(caught[-1].message).strip().splitlines()[0][:110]}")
    print(f"  CHECK 6 real run, real trace {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("stall detection", flush = True)
    results = []
    for name, check in (("healthy runs survive", checkHealthy),
                        ("hard floor", checkFlat),
                        ("noise floor", checkNoise),
                        ("slow convergence", checkSlow),
                        ("monotone in patience", checkPatience),
                        ("reachability up front", checkReachability),
                        ("real run", checkRealRun)):
        print(f"\n{name}:", flush = True)
        results.append(check())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
