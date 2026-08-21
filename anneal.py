"""Annealed decompression: find the densest packing a branch can reach.

THE QUESTION. Start a packing ABOVE its jamming density, where it cannot avoid overlapping, and lower
the density until it can pack. The density where that first happens is how good a packing was found --
for 5 unit squares the optimum is 5/2.7071^2 = 0.68227, so a run starting at 5/2.7^2 = 0.68587 is
asking "how close to optimal can this protocol get?".

WHY IT NEEDS AN ANNEAL. At a packing-relevant density rigid shapes have almost no free volume, so a
local minimizer cannot rearrange them -- it relaxes into whatever basin it started in. Two knobs make
the landscape temporarily easier and are then closed:

  POLYDISPERSITY  a spread in the shape targets lets polygons trade size, which is a cheap way past a
                  geometric obstruction; driving it to zero recovers the monodisperse problem.
  SIGMA           the mollification rounds every corner. A packing relaxed at large sigma is a packing
                  of ROUNDED shapes, which pack differently; sharpening it is what makes the answer a
                  statement about the actual polygons.

Both are reduced on a schedule while the configuration is re-relaxed, so the run tracks a solution
branch instead of dropping into an unrelated basin at each step.

THE VERDICT IS GEOMETRIC, NOT ENERGETIC. Validity is decided by the EXACT overlap area
(``Model.getOverlapArea``), which is identically zero below jamming and positive above -- a sign
change with no threshold to tune and no sigma dependence. The mollified relaxation energy cannot serve
this purpose: it does not vanish on a valid packing (measured 8.2e-04 where the true overlap is exactly
zero) and it RISES as a valid packing is compressed, because its kernel has a tail between merely
touching faces.

SIGMA HAS A FLOOR. The mollified contact force scales as 1/sigma, so below roughly 0.01 of the mean
edge any FIRE or CG step overshoots the contact and diverges. The schedule stops there; going further
is unnecessary because the verdict is read from the sharp overlap, which has no sigma in it at all.
"""

import warnings

import numpy as np


# Below this fraction of the mean edge the mollified dynamics diverge -- mirrors
# model._MIN_STABLE_SOFTENING_FRACTION, which is the warning the minimizers raise.
_MIN_STABLE_SOFTENING_FRACTION = 0.01


# UNVERIFIED(Cam)
class SweepResult:
    """Outcome of an ``energySweep``: the density reached, whether it packed, and the full history."""

    def __init__(self):
        self.phi = float("nan")
        self.packed = False
        self.overlap = float("nan")
        self.history = []
        self.bracket = (float("nan"), float("nan"))

    def record(self, phase, model, overlap):
        """Append one step's diagnostics."""
        areas = model.getAreas()
        targets = np.asarray(model.getTargetAreas(), dtype = float)
        stop = model.getNumPolygons() if getattr(model.packing, "containerIndex", None) is None \
            else int(model.packing.containerIndex)
        self.history.append(dict(
            phase = phase,
            phi = model.getPackingFraction(),
            # The SIZE spread (std/mean of the target areas) -- what the anneal drives to zero. This
            # field previously held ``model.sigmaFraction``, which is sigma/meanEdge, the MOLLIFICATION
            # fraction: a different quantity entirely, recorded under the wrong name.
            polydispersity = model.getSizePolydispersity(),
            # The SHAPE spread, which polydispersity cannot see: with the areas hard-constrained and
            # the edges held only in their moments, every polygon can distort while the size
            # distribution reads monodisperse.
            distortion = model.getMaxShapeDistortion(),
            # Recorded separately because only the first is exact; see the verdict note in energySweep.
            pairOverlap = model.getPairOverlapArea(),
            penetration = model.getWallPenetration(),
            sigma = model.sigma,
            softeningFraction = model.sigmaFraction,
            overlapArea = overlap,
            energy = model.getEnergy(),
            areaError = float(np.abs(areas[:stop] / targets[:stop] - 1.0).max()),
            residual = model.constraintResidual()))
        return self.history[-1]

    def __repr__(self):
        if not self.packed:
            return f"<SweepResult NOT PACKED, {len(self.history)} steps>"
        # The residual is shown, never hidden: a nonzero one means the reported density is slightly
        # ABOVE what actually packs, since a valid packing has exactly zero overlap.
        return (f"<SweepResult phi = {self.phi:.6f}, residual overlap {self.overlap:.2e}, "
                f"{len(self.history)} steps>")


def _geometric(start, stop, rounds):
    """A geometric ramp from ``start`` down to ``stop`` in ``rounds`` steps, endpoints included.

    Geometric rather than linear because both knobs act multiplicatively: halving sigma matters as much
    at 0.05 as at 0.005, and a linear ramp would spend nearly all its steps in the regime where the
    knob no longer changes anything."""
    start = max(float(start), 1e-300)
    stop = max(float(stop), 1e-300)
    if rounds < 1:
        return [stop]
    return list(start * (stop / start) ** (np.arange(1, rounds + 1) / rounds))


_TRANSIENT_TOLERANCE = 1e-7
# What the SHARP tier can actually deliver, and how long it takes to get there. The exact overlap area
# is only C1 -- its gradient turns a corner at every vertex-edge contact change -- so FIRE converges
# linearly and then stops. Measured on 11 squares at phi = 0.8035:
#
#   FIRE steps    200      500      1000     2000     4000
#   max|F|        3.6e-03  1.3e-03  2.1e-04  1.6e-04  1.8e-04
#   overlap       7.4e-03  6.3e-03  6.066e-03  6.066e-03  6.066e-03
#
# The answer is settled by ~1000 steps and the residual then wanders around 2e-04 forever. Asking for
# 1e-8 there costs 190 s per density step and changes nothing.
_SHARP_TOLERANCE = 1e-4
_SHARP_MAX_STEPS = 1500
# Containment tolerance as a fraction of the mean edge, used when the caller gives none. A packing is
# accepted with at most this much of any polygon sticking out of the wall. 1e-4 of an edge sits about
# five times above the residual a converged corner contact leaves behind (1.88e-05 of depth = 7.5e-05
# of an edge, measured), so it accepts a jammed packing without accepting a visibly escaped one.
_WALL_TOLERANCE_FRACTION = 1e-4
# Plausible range for a locally fitted jamming exponent. contact.tex measures the corner exponents at 3
# (face-on-face) and 4 (sharp corner), and the contact NUMBER grows alongside the depth, so a real fit
# lands a little above those. Outside this band the fit is reading noise, not scaling -- see
# predictJamming.
_JAMMING_EXPONENT_RANGE = (2.0, 8.0)


# UNVERIFIED(Cam)
def bisectJamming(model, low = None, rounds = 20, tolerance = 1e-6, maxUnbalancedForce = 1e-8,
                  maxSteps = 20000, minimizer = "lbfgs", finalEnergy = 0.0, wallTolerance = None,
                  minPhi = 0.2, predict = True, probeStep = 0.004, probes = 4, margin = 0.01,
                  progressBar = False, verbose = False, drawEvery = None):
    """Binary search for the jamming density on the GEOMETRIC verdict. Returns a ``SweepResult``.

    Replaces the fixed ladder of ``energySweep``'s phase B, which needs ``(phiStart - phiJ) / phiStep``
    relaxations just to bracket the answer; this needs ``log2`` of that. The bracket is found first by
    halving the gap downward from the current density until something packs, then narrowed by
    bisection until it is thinner than ``tolerance``.

    IT TELEPORTS, AND THAT IS A REAL COST HERE. A jammed landscape is glassy: relaxing at a density is
    path dependent, and ``_snapshot`` records a state accepted at zero overlap coming back at 5.3e-06
    when re-derived by setting phi back and relaxing again -- not a valid packing at all. A ladder makes
    small steps and tracks one branch; a bisection jumps, so it can land in a different basin than
    walking there would have found. Two things limit the damage: every trial warm-starts from the last
    ACCEPTED configuration rather than from wherever the previous trial ended, and the answer is
    returned as coordinates (``_snapshot``) rather than as a density to be recomputed.

    Use it when the shapes are already fixed and the question is purely "how far will this arrangement
    compress" -- after a cascade, or in place of a final ``energySweep``. Prefer the ladder when the
    density steps are still carrying an anneal, since the schedule is what makes those steps small."""
    if wallTolerance is None:
        wallTolerance = _WALL_TOLERANCE_FRACTION * _meanEdge(model)
    result = SweepResult()
    bar = _progressBarOrNone(progressBar, rounds)

    _relax(model, maxUnbalancedForce, maxSteps, False, minimizer)
    high = model.getPackingFraction()
    valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
    result.record("start", model, model.getOverlapArea())
    if valid:
        warnings.warn(
            f"\n*** already packed at the starting density ***\n"
            f"    phi = {high:.6f} carries no overlap, so there is nothing to bisect DOWN to. Compress "
            f"first, or start the search from a denser configuration.", stacklevel = 2)
        result.phi, result.packed, result.overlap = high, True, overlap
        return result

    best = None
    # PREDICT THE BRACKET RATHER THAN HALVING INTO IT. Halving toward minPhi puts the first probe
    # roughly midway to the floor -- measured, that landed at phi 0.52 from a 0.85 start and teleported
    # the packing into a loose basin, after which every later bisection refined within THAT basin and
    # returned 0.5819 where the ladder walked to 0.6734. A few small probing steps instead give
    # predictJamming enough to extrapolate (it reaches +/-0.004 from as far as 0.06 above jamming), so
    # the first big move lands just below the answer instead of far past it.
    samples = []
    if predict and low is None:
        trial = high
        for _ in range(int(probes)):
            samples.append((trial, model.getEnergy()))
            guess, alpha = predictJamming(samples)
            if guess is not None and guess > minPhi:
                aim = guess - float(margin)
                model.setPackingFraction(aim)
                _relax(model, maxUnbalancedForce, maxSteps, False, minimizer)
                valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
                result.record("predict", model, model.getOverlapArea())
                if verbose:
                    print(f"  predict  phiJ ~ {guess:.6f} (alpha {alpha:.2f}); probing {aim:.6f} -> "
                          f"{'packs' if valid else 'overlaps'}")
                if valid:
                    low, best = aim, _snapshot(model, aim, overlap)
                    break
                # The prediction was optimistic; that probe is now the new upper bound.
                high = aim
                samples = [(aim, model.getEnergy())]
                trial = aim
                continue
            trial -= float(probeStep)
            if trial <= minPhi:
                break
            model.setPackingFraction(trial)
            _relax(model, maxUnbalancedForce, maxSteps, False, minimizer)
            valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
            result.record("probe", model, model.getOverlapArea())
            if verbose:
                print(f"  probe    phi {trial:.6f}  overlap {overlap:.3e}  "
                      f"{'packs' if valid else 'overlaps'}")
            if valid:
                low, best = trial, _snapshot(model, trial, overlap)
                break
            high = trial

    # Fallback: halve the gap to minPhi until something packs.
    if low is None:
        trial = high
        while trial - minPhi > tolerance:
            trial = 0.5 * (trial + minPhi)
            model.setPackingFraction(trial)
            _relax(model, maxUnbalancedForce, maxSteps, False, minimizer)
            valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
            step = result.record("bracket", model, model.getOverlapArea())
            _maybeDraw(model, drawEvery, len(result.history), "bracket", trial, overlap)
            if verbose:
                print(f"  bracket  phi {trial:.6f}  overlap {overlap:.3e}  "
                      f"{'packs' if valid else 'overlaps'}")
            if valid:
                low = trial
                best = _snapshot(model, trial, overlap)
                break
            high = trial
        if low is None:
            warnings.warn(
                f"\n*** nothing packed down to minPhi = {minPhi} ***\n"
                f"    the bracket never closed, so there is no interval to bisect. Either minPhi is "
                f"too high or the shapes cannot pack at any density this search reached.",
                stacklevel = 2)
            return result
    else:
        best = _snapshot(model, low, model.getOverlapArea())

    for _ in range(int(rounds)):
        if high - low <= tolerance:
            break
        middle = 0.5 * (low + high)
        # Warm-start from the last ACCEPTED state, not from wherever the previous trial landed: a
        # rejected trial is a different basin, and carrying it forward propagates that choice.
        model.packing.positions[:] = best["positions"]
        model.packing.targetArea[:] = best["targetArea"]
        model.packing.targetEdgeLength[:] = best["targetEdgeLength"]
        model.setPackingFraction(middle)
        _relax(model, maxUnbalancedForce, maxSteps, False, minimizer)
        valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
        result.record("bisect", model, model.getOverlapArea())
        _maybeDraw(model, drawEvery, len(result.history), "bisect", middle, overlap)
        if verbose:
            print(f"  bisect   phi {middle:.6f}  overlap {overlap:.3e}  "
                  f"{'packs' if valid else 'overlaps'}")
        if valid:
            low = middle
            best = _snapshot(model, middle, overlap)
        else:
            high = middle
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    model.packing.positions[:] = best["positions"]
    model.packing.targetArea[:] = best["targetArea"]
    model.packing.targetEdgeLength[:] = best["targetEdgeLength"]
    model.packing.syncTargetPerimeter()
    model._forces = None
    model._energy = None
    result.bracket = (low, high)
    result.phi, result.packed, result.overlap = best["phi"], True, best["overlap"]
    result.record("final", model, model.getOverlapArea())
    return result


# UNVERIFIED(Cam)
def predictJamming(samples, floor = 1e-30):
    """Extrapolate the jamming density from ``(phi, energy)`` samples. Returns ``(phiJ, alpha)``, or
    ``(None, None)`` when the samples cannot support a fit.

    Assumes ``E = C (phi - phiJ)^alpha`` LOCALLY and solves the three unknowns from the last three
    usable points. Taking logs and differencing twice eliminates ``C`` and leaves one equation in
    ``phiJ``, which is bracketed and bisected -- no starting guess and no optimizer.

    THE EXPONENT IS FITTED, NOT ASSUMED, and that is the whole reason this works. Measured on 11
    squares over phi 0.76 -> 0.68, a GLOBAL fit returns alpha = 4.9 with a log-residual of 2.7 -- the
    scaling is simply not one power law across that range, because the contact NUMBER grows alongside
    the depth and the packing is far too small for the asymptotic argument. Local three-point fits over
    the same data return alpha = 3.6 to 4.2, which is where ``notes/polygonContact/contact.tex`` puts
    the corner exponents (3 for face-on-face, 4 for a sharp corner), and predict phiJ to within 0.005 --
    often 0.0002, and from as far as 0.055 above jamming. Fixing alpha at any single value would import
    the global fit's error.

    The caller must treat the answer as a SUGGESTION. It is an extrapolation from a power law that is
    only locally true, so it belongs in front of a step limiter and a validity test, never in place of
    them."""
    usable = [(float(phi), float(energy)) for phi, energy in samples if energy > floor]
    if len(usable) < 3:
        return None, None
    (p1, e1), (p2, e2), (p3, e3) = usable[-3:]
    if not (p1 > p2 > p3):
        return None, None

    def mismatch(phiJ):
        """Zero when the three points lie on one power law about ``phiJ``."""
        a, b, c = np.log(p1 - phiJ), np.log(p2 - phiJ), np.log(p3 - phiJ)
        left = (np.log(e1) - np.log(e2)) * (b - c)
        right = (np.log(e2) - np.log(e3)) * (a - b)
        return left - right

    low, high = p3 - 10.0 * (p1 - p3), p3 - 1e-12
    try:
        lowValue, highValue = mismatch(low), mismatch(high)
    except (ValueError, FloatingPointError):
        return None, None
    if not np.isfinite(lowValue) or not np.isfinite(highValue) or lowValue * highValue > 0.0:
        return None, None
    for _ in range(200):
        middle = 0.5 * (low + high)
        value = mismatch(middle)
        if not np.isfinite(value):
            return None, None
        if value * lowValue <= 0.0:
            high = middle
        else:
            low, lowValue = middle, value
    phiJ = 0.5 * (low + high)
    alpha = float((np.log(e1) - np.log(e3)) / (np.log(p1 - phiJ) - np.log(p3 - phiJ)))
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None, None
    # AN IMPLAUSIBLE EXPONENT MEANS THE SAMPLES ARE TOO CLOSE TO JAMMING TO CARRY A FIT. The energies
    # there are down in the force-balance noise, so three nearly-equal tiny numbers fit an arbitrarily
    # steep power law. Measured on the same data, fitting from phi = 0.6833 -- one step above the true
    # 0.6817 -- returned alpha = 13.5 and phiJ = 0.6711, an answer BELOW the density that still packed.
    # The contact law's own corner exponents are 3 and 4, so anything far outside that is not a
    # measurement of the law and is refused rather than acted on.
    if not (_JAMMING_EXPONENT_RANGE[0] <= alpha <= _JAMMING_EXPONENT_RANGE[1]):
        return None, None
    return float(phiJ), alpha


# UNVERIFIED(Cam)
def _packs(model, finalEnergy, wallTolerance):
    """``(packed, pairOverlap, wallPenetration)`` -- the two-part verdict.

    ``finalEnergy`` applies to the POLYGON-POLYGON overlap, which is exact and normally tested against
    zero. ``wallTolerance`` is a DEPTH applied to containment, which cannot be tested against zero: see
    the note in ``energySweep``."""
    pairOverlap = model.getPairOverlapArea()
    penetration = model.getWallPenetration()
    return (pairOverlap <= finalEnergy and penetration <= wallTolerance), pairOverlap, penetration


def _relax(model, tolerance, maxSteps, innerProgressBar = False, minimizer = "cg"):
    """Relax to equilibrium between schedule steps, warm-started from the previous configuration.

    FIRE then a CG polish, EXCEPT when transient target DOF are active. CG is worth the extra few force
    evaluations normally: the verdict is a sign test on the overlap area, and a residual left at FIRE's
    linear rate can leave a sliver of overlap that reads as "did not pack" and pushes the bracket to
    the wrong side.

    CG IS NOT USED WITH TRANSIENT TARGETS. ``minimize.minimizeCG`` has no ``transient`` hook -- only
    FIRE takes one -- so a CG polish would relax the positions at FROZEN targets, quietly turning the
    double optimization into a single one for the last leg of every step. It is not merely missing
    plumbing either: CG's strong-Wolfe line search assumes a fixed energy landscape, and moving the
    targets mid-search would violate that assumption rather than just complicate it. FIRE alone runs to
    ``_TRANSIENT_TOLERANCE`` instead, which it reaches without the polish.

    CG IS NOT USED ON THE SHARP TIER EITHER, and the tolerance is loosened to what that tier can reach.
    The exact overlap area is C1 but not C2 -- the contact set changes discontinuously as a vertex
    crosses an edge -- so the strong-Wolfe line search has no smooth curvature to exploit and FIRE
    converges linearly to a floor rather than to zero (see ``_SHARP_TOLERANCE`` for the measurement).
    Chasing the mollified tolerance there burns thousands of steps without moving the overlap.

    ``minimizer`` selects what does the work on the smooth tiers: ``"fire"`` keeps the historical
    FIRE-then-CG-polish pair, while ``"cg"`` and ``"lbfgs"`` drop the FIRE leg and run that minimizer
    alone. The transient and sharp cases above are unaffected -- their reasons for needing FIRE are
    stated there and are not about speed."""
    if model.transient is not None:
        model.minimizeFIRE(maxUnbalancedForce = max(tolerance, _TRANSIENT_TOLERANCE),
                           maxSteps = maxSteps, progressBar = innerProgressBar)
        return
    if model.modelType == "area":
        model.minimizeFIRE(maxUnbalancedForce = max(tolerance, _SHARP_TOLERANCE),
                           maxSteps = min(maxSteps, _SHARP_MAX_STEPS),
                           progressBar = innerProgressBar)
        return
    if minimizer == "fire":
        model.minimizeFIRE(maxUnbalancedForce = tolerance, maxSteps = maxSteps,
                           progressBar = innerProgressBar)
        model.minimizeCG(maxUnbalancedForce = tolerance, maxSteps = 400,
                         progressBar = innerProgressBar)
        return
    if minimizer == "fireLbfgs":
        # FIRE coarsely, then L-BFGS to the target -- the two fail in opposite places, so pairing them
        # beats either alone from a far start. See ``Model.minimizeFireLBFGS``.
        model.minimizeFireLBFGS(maxUnbalancedForce = tolerance, maxSteps = maxSteps,
                                progressBar = innerProgressBar)
        return
    if minimizer not in ("cg", "lbfgs"):
        raise ValueError(f"unknown minimizer {minimizer!r}; use 'cg', 'lbfgs', 'fire' or "
                         f"'fireLbfgs'")
    polish = model.minimizeLBFGS if minimizer == "lbfgs" else model.minimizeCG
    polish(maxUnbalancedForce = tolerance, maxSteps = maxSteps, progressBar = innerProgressBar)


# UNVERIFIED(Cam)
def _setRegularEdgeTargets(model):
    """Point every edge target at the REGULAR value for that polygon's target area.

    ``l0 = sqrt(4 A0 tan(pi/n) / n)``, so the constraint set describes the regular n-gon and SHAKE
    pulls the shapes onto it. Deliberately NOT a sync to current geometry, which would freeze whatever
    distortion is present -- that is the whole difference between rigidifying and giving up.

    Shared by ``_rigidify`` and ``compressToJamming(rigidify = True)`` so the two cannot drift apart."""
    packing = model.packing
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    counts = np.diff(np.asarray(packing.startIndices, dtype = int))[:stop].astype(float)
    targetArea = np.asarray(packing.targetArea, dtype = float)[:stop]
    regular = np.sqrt(4.0 * targetArea * np.tan(np.pi / counts) / counts)
    upTo = int(packing.startIndices[stop])
    packing.targetEdgeLength[:upTo] = np.repeat(regular, counts.astype(int))
    packing.syncTargetPerimeter()
    return model


def _rigidify(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer = "cg"):
    """Hand off from moment-held edges to RIGID regular polygons, and relax onto them.

    This is what makes the answer a packing of SQUARES rather than of quadrilaterals that happen to
    have the right area. Constraining the area alone does not constrain shape at all -- a fixed-area
    quadrilateral is any quadrilateral -- so a run that never does this ends in distorted shapes no
    matter how well it converged, and its density is a statement about those shapes.

    The handoff cannot simply sync the edge targets to the current geometry: that would FREEZE whatever
    distortion is present. It sets each edge target to the REGULAR value for that polygon's target
    area, ``l0 = sqrt(4 A0 tan(pi/n) / n)``, so the constraint set describes the regular n-gon and
    SHAKE pulls the shapes onto it. With the area also fixed this is exactly a square for n = 4: an
    equal-edged quadrilateral is a rhombus with ``A = l^2 sin(theta)``, so fixing both forces
    ``sin(theta) = 1``.

    The target set lands exactly ON the isoperimetric bound (a regular polygon is the equality case),
    which ``ShapeConstraints.infeasibleReason`` tolerates by design -- its slack exists for precisely
    this.

    Transient targets are switched off at the same time: with both families now rigid they have no
    drive, so leaving them on would only churn the moment restore every step."""
    _setRegularEdgeTargets(model)
    model.dofType = "fixed"
    model.transient = None
    model.setConstraints(area = True, edge = True)
    _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
    return model


def _snapshot(model, phi, overlap):
    """Copy the CONFIGURATION of a valid packing, not just its density.

    Relaxation here is path dependent -- a jammed landscape is glassy, so re-relaxing at the same
    density from a different starting state lands in a different basin. Re-deriving the answer by
    setting the density back to ``best`` and relaxing again therefore does NOT reproduce the packing
    that was found: measured, a state accepted at zero overlap came back at 5.3e-06 when recomputed
    that way, which is not a valid packing at all. The answer is a set of coordinates."""
    return dict(phi = phi, overlap = overlap,
                positions = model.packing.positions.copy(),
                targetArea = np.array(model.packing.targetArea, dtype = float),
                targetEdgeLength = np.array(model.packing.targetEdgeLength, dtype = float))


def _restore(model, snapshot):
    """Put a saved packing back, targets included -- transient targets moved while it was found."""
    model.packing.positions[:] = snapshot["positions"]
    model.packing.targetArea[:] = snapshot["targetArea"]
    model.packing.targetEdgeLength[:] = snapshot["targetEdgeLength"]
    model.packing.syncTargetPerimeter()
    model._forces = None
    model._energy = None
    return model


def _shapeLimits(model):
    """Colour range for the shape index, pinned ONCE for the whole sweep.

    Anchored at the regular value (4 for a square, ~3.72 for a hexagon), because that is the floor no
    polygon can go below and the reference every frame should be read against. Autoscaling per frame
    would re-normalize each picture, so a colour would stop meaning the same thing from one step to the
    next and a distortion that grew would look identical to one that shrank."""
    container = getattr(model.packing, "containerIndex", None)
    stop = model.packing.numPolygons if container is None else int(container)
    counts = np.diff(np.asarray(model.packing.startIndices, dtype = int))[:stop].astype(float)
    regular = float(np.min(np.sqrt(4.0 * counts * np.tan(np.pi / counts))))
    worst = float(np.max(model.getShapeIndices()[:stop]))
    return regular, max(worst, regular * 1.02)


def _maybeDraw(model, drawEvery, count, phase, phi, overlap, limits = None):
    """Draw the packing every ``drawEvery`` steps, coloured by shape index.

    A sweep runs for minutes with nothing to look at but a progress bar, and the failure modes that
    matter here are GEOMETRIC -- a polygon folding, shapes drifting away from regular, the arrangement
    coming apart -- none of which a scalar residual reveals. Shading by ``P / sqrt(A)`` puts exactly
    that on the picture: pale is regular, dark is distorted.

    The title carries the packing fraction and the mollification width, the two knobs the schedule is
    turning, so a frame is self-describing when scrolled back to.

    Figures are emitted in sequence rather than redrawn in place: clearing the cell output would take
    the progress bar with it, and the point is to compare successive states."""
    if not drawEvery or count % int(drawEvery) != 0:
        return
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return
    axes = model.draw(colorBy = model.shapeIndex, colorLimits = limits,
                      colorLabel = r"shape index  $P/\sqrt{A}$")
    # Scientific notation for the polydispersity: it is the quantity driven to zero, so in fixed point
    # it reads 0.00000 across the whole endgame, exactly where its value matters most.
    #
    # Polydispersity and distortion are BOTH shown because either alone is misleading. Polydispersity
    # is the spread of the AREAS -- are the polygons the same size -- and it hits 1e-16 the moment
    # setSizePolydispersity runs, while the shapes may be nothing like square. Distortion is the
    # complementary number the colour already encodes, and it is what the rigidify handoff drives to
    # zero.
    # Once the sharp tier is running there is no sigma in the energy at all, so printing the last value
    # it held would describe a term that is no longer being evaluated.
    softening = rf"$\sigma_{{\rm moll}}$ = {model.sigma:.3e}" if model.modelType == "mollified" \
        else rf"$\sigma_{{\rm moll}}$ = 0 ({model.modelType})"
    axes.set_title(f"[{phase}] step {count}\n"
                   rf"$\phi$ = {phi:.6f}    " + softening + "\n"
                   f"polydispersity = {model.getSizePolydispersity():.3e}    "
                   f"distortion = {model.getMaxShapeDistortion():.3e}    "
                   f"overlap = {overlap:.3e}")
    plt.show()


class _SweepBar:
    """One progress bar for the WHOLE sweep, not one per relaxation.

    A sweep runs dozens of minimizations, so forwarding ``progressBar`` to each of them would print
    dozens of bars that each vanish immediately -- noise rather than progress. This tracks the sweep's
    own steps and shows the phase, the current density and the residual overlap, which is what actually
    says how the run is going. Inner minimizer bars stay available separately via ``innerProgressBar``.

    The total is known in advance: the anneal rounds, the worst-case number of decompression steps, the
    refinement rounds, and the final re-relaxation."""

    def __init__(self, enabled, annealRounds, phi, phiStep, minPhi, refineRounds):
        self.bar = None
        if not enabled:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return
        descend = max(int(np.ceil((phi - minPhi) / max(phiStep, 1e-12))), 1)
        # The compression phase is bounded but its length is not known in advance -- it runs until the
        # packing first overlaps. Budgeting a quarter of the descent keeps the bar from finishing early
        # and sitting at 100% while work continues; tqdm handles an overrun without complaint.
        compress = max(descend // 4, 1) + refineRounds
        self.bar = tqdm(total = annealRounds + descend + refineRounds + compress + 1,
                        desc = "energySweep")

    def step(self, phase, phi, overlap, model = None):
        if self.bar is not None:
            softening = "" if model is None else (
                f" sigmaMoll {model.sigma:.2e}" if model.modelType == "mollified"
                else f" {model.modelType}")
            extra = "" if model is None else (
                softening +
                f" polydispersity {model.getSizePolydispersity():.3e}"
                f" distortion {model.getMaxShapeDistortion():.3e}")
            self.bar.set_description(
                f"energySweep [{phase}] phi {phi:.5f}{extra} overlap {overlap:.2e}")
            self.bar.update(1)

    def close(self):
        if self.bar is not None:
            self.bar.close()


# UNVERIFIED(Cam)
def _reachableWidth(model):
    """The narrowest edge-length distribution the FIXED areas allow, without distorting shapes.

    A regular n-gon of area A has edge ``sqrt(4 A tan(pi/n) / n)``, so once the areas are held rigid the
    edge lengths inherit their spread: the edge CV cannot go below the CV of ``sqrt(A0)``, which is
    about half the area CV. Asking for less is asking the polygons to have more equal edges than their
    own areas permit, and the only way to comply is to STOP BEING REGULAR.

    Measured on 11 squares with area CV 0.222: the reachable edge CV is 0.111, the packing already sat
    at 0.1114, and demanding 0.0439 drove the worst shape index from 4.07 to 4.28 while TRIPLING the
    overlap -- before any minimization had run."""
    packing = model.packing
    container = getattr(packing, "containerIndex", None)
    stop = packing.numPolygons if container is None else int(container)
    counts = np.diff(np.asarray(packing.startIndices, dtype = int))[:stop].astype(float)
    edges = np.sqrt(4.0 * np.asarray(packing.targetArea, dtype = float)[:stop]
                    * np.tan(np.pi / counts) / counts)
    mean = float(np.mean(edges))
    return float(np.std(edges) / mean) if mean > 0.0 else 0.0


# UNVERIFIED(Cam)
def _freeCount(model):
    """Number of ordinary polygons, container excluded."""
    container = getattr(model.packing, "containerIndex", None)
    return model.getNumPolygons() if container is None else int(container)


# UNVERIFIED(Cam)
def _meanEdge(model):
    """Mean target edge length over the ORDINARY polygons, with the container's own edges excluded.

    The wall is stored as one more polygon, so ``mean(packing.targetEdgeLength)`` averages the packing's
    edges together with the box's -- and the box is the size of the whole system. At N = 6, n = 8 that
    turns a true mean edge of 0.133 into 0.200; ``energyScale`` raises it to the fourth power, so the
    unit the excess is measured in would be 5x wrong and would move whenever the box was scaled, which
    is precisely what a control parameter must not do."""
    container = getattr(model.packing, "containerIndex", None)
    lengths = np.asarray(model.packing.targetEdgeLength, dtype = float)
    if container is not None:
        lengths = lengths[:int(model.packing.startIndices[int(container)])]
    return float(np.mean(lengths))


# UNVERIFIED(Cam)
def _enableShapeBudget(model):
    """Turn the shape budget on WITHOUT discarding whatever else is already constrained.

    The distortion ramp needs ``setConstraints(..., shape = True)``, but calling that directly would
    silently drop the caller's own constraint choices, since ``setConstraints`` rebuilds the whole set
    from its arguments. The current configuration is read back off the live constraint object and
    replayed with ``shape = True`` added."""
    current = model.constraints
    block = getattr(current, "block", current)
    distribution = getattr(current, "distribution", None)
    # THE SHAPE FAMILY BEING ON IS NOT ENOUGH -- it has to be the DEVIATION form. ``setShapeDeficit``
    # drives the barrier, and only the deviation set has one; the DIRECT distortion family
    # (``setConstraints(distortion = [...])``) is the same family read as a dimensionless d_i with no
    # barrier at all. Returning early on the direct form left it in place and then demanded a deviation
    # set from it one phase later, which is a crash rather than a wrong answer, but only by luck.
    alreadyBarrier = (getattr(current, "shape", False)
                      and bool(getattr(distribution, "deviation", False)))
    if current is not None and alreadyBarrier:
        return model
    if current is not None and getattr(current, "shape", False):
        warnings.warn(
            "\n*** direct distortion constraint replaced by the deviation barrier ***\n"
            "    setConstraints(distortion = [...]) holds the dimensionless d_i directly, but the "
            "shape anneal drives its target to zero and a DIRECT budget's gradient vanishes exactly "
            "there -- the ramp would stall short and hand off with a jump.\n"
            "    The deviation form (shape = [1, -1]) is substituted for the sweep: its k = -1 row "
            "carries -delta^-2 and stiffens as the deficit shrinks, so the ramp can actually arrive. "
            "Pass energySweep(annealShape = False) to keep your own constraint set instead.",
            stacklevel = 3)
    area = True
    edge = False
    perimeter = False
    if block is not None:
        area = bool(getattr(block, "area", False))
        edge = bool(getattr(block, "edge", False))
        perimeter = bool(getattr(block, "perimeter", False))
    if distribution is not None:
        moments = list(distribution.moments)
        if getattr(distribution, "area", False):
            # AN AREA MOMENT FAMILY CANNOT COME ALONG EITHER, and the reason is structural rather than
            # a policy choice: ``deviation`` is a flag on the whole SET, not per family, so turning the
            # shape rows into a barrier turns the area rows into one too. Area deviation is the
            # shrink-only ``A0 - A``, which is negative the moment any polygon has grown past its
            # target -- measured here at -8.6e-03 after an ordinary relaxation -- and a k = -1 row on a
            # negative base is singular. Areas are held PER OBJECT instead, which is what the default
            # path does anyway and what the sweep's density bookkeeping already assumes.
            area = True
            warnings.warn(
                "\n*** area moments replaced by per-object area constraints for the shape anneal ***\n"
                "    the distortion ramp needs the deviation form, and 'deviation' applies to the whole "
                "constraint set, so area moments would become a shrink-only A0 - A barrier -- singular "
                "as soon as any polygon exceeds its target area, which relaxation routinely does.\n"
                "    Areas are pinned individually instead, so polygons can no longer TRADE size during "
                "the sweep. Pass energySweep(annealShape = False) to keep your own set.",
                stacklevel = 3)
        if getattr(distribution, "edge", False):
            edge = moments
    # AN EDGE MOMENT FAMILY IS DROPPED HERE, AND IT MUST BE. Under ``deviation = True`` the edge rows
    # stop measuring lengths and start measuring |l_ik - l0_i|, each edge's distance from its polygon's
    # ideal, held away from zero by the k = -1 barrier -- so no edge can ever BE ideal. The shape ramp
    # drives the isoperimetric deficit to zero, which is exactly the state where every edge is ideal.
    # One family therefore forbids the point the other is aiming at, and the retraction grinds against
    # an empty feasible set: measured on 11 squares, residual stuck at 5.13e-01 over 126 passes while
    # the two families' rows went collinear (conditioning 1.18e-04). See ``deviations`` in
    # constraints.py: "for driving a packing to regular polygons prefer shape".
    if edge not in (True, False):
        warnings.warn(
            "\n*** edge moments dropped for the shape anneal ***\n"
            "    setConstraints(edge = [...]) and the distortion ramp cannot both hold: as DEVIATIONS "
            "the edge rows keep every edge away from its ideal length, which is the exact state the "
            "ramp drives toward. The shape family alone is kept.\n"
            "    Pass energySweep(annealShape = False) instead if the edge moments matter more than "
            "reaching regular polygons.", stacklevel = 3)
        edge = False
    # The DEVIATION form with a k = -1 barrier, not the direct budget. The budget's gradient vanishes
    # at the regular polygon it is aiming for, so its ramp has to stop early and hand off with a jump;
    # the barrier's gradient carries -delta^-2 and stiffens instead, so the ramp can actually arrive.
    # Moments must match across families, so any area moment list is replaced by [1, -1] here --
    # which is a real restriction and the reason the shape anneal owns the moment set once it is on.
    return model.setConstraints(area = area if area is True else [1, -1],
                                perimeter = perimeter,
                                edge = edge,
                                shape = [1, -1], deviation = True)


# UNVERIFIED(Cam)
def _slowRounds(start, target, rounds, maxRatio):
    """Number of geometric steps needed to go ``start -> target`` without any single step exceeding a
    factor of ``maxRatio``.

    A ramp is only an anneal if each step is small enough that the relaxation between steps can follow
    it. Compressing a factor of 30 into 4 rounds is a factor of 2.3 per step in sigma, which moves the
    contact stiffness (~1/sigma) faster than FIRE re-equilibrates -- the state that arrives at the next
    step is not the relaxed state of the previous one, so the branch is not being followed. This
    lengthens the ramp rather than warning about it."""
    if start <= 0.0 or target <= 0.0 or rounds < 1:
        return max(int(rounds), 1)
    span = abs(np.log(target / start))
    needed = int(np.ceil(span / np.log(float(maxRatio))))
    return max(int(rounds), needed, 1)


# UNVERIFIED(Cam)
def compressToJamming(model, pressure = 1e-3, finalPressure = 1e-9, pressureRounds = 8,
                      maxUnbalancedForce = 1e-8, maxSteps = 20000, minimizer = "lbfgs",
                      boxTolerance = 1e-10, maxBoxSteps = 40, maxScaleStep = 1.05,
                      rigidify = True, progressBar = False, innerProgressBar = False,
                      verbose = False, drawEvery = None):
    """Compress by giving the BOX a degree of freedom, under a pressure ramped to zero.

    The alternative protocol to ``energySweep``, and it differs in what moves. ``energySweep`` steps a
    density ladder by RESIZING THE POLYGONS, relaxes, and bisects; each step teleports the
    configuration to a new phi and lets it fall. That kick is not free -- ``_snapshot`` records a state
    accepted at zero overlap coming back at 5.3e-06 when re-derived by setting phi back and relaxing
    again, which is not a valid packing at all. Here the polygons are never touched: the box carries one
    scale degree of freedom and is driven inward by an applied pressure against the packing's own
    resistance, so the configuration rearranges WHILE the box closes rather than after each jump.

        H = E_contact + p A_box,        dH/dlambda = sum_i g_i . (v_i - c) + 2 p A_box

    At ``dH/dlambda = 0`` the wall pressure balances the packing; ramping ``p`` geometrically to zero
    walks that balance to the jamming point. There is no phi ladder and no bisection -- the density is
    an OUTPUT.

    WHY THIS IS SAFE UNDER THIS LAW AND WOULD NOT BE UNDER BODY-BODY CONTACT ALONE. Pressing harder is
    normally bounded by the validity limit dMax/rIn << 1, past which the repulsion reverses sign. The
    exterior of a CONVEX container has no medial axis at all -- beyond an edge the nearest feature is
    that edge, beyond a corner it is that corner, and the seam between them is a C^1 tie between
    INCIDENT features -- so the wall term has no such limit and the box may be pressed arbitrarily hard.
    A nonconvex container does have a medial axis and this argument lapses.

    IT IS NOT BASIN-HOPPING. Removing the kick tracks one branch more faithfully; it does not escape a
    branch. Anneal the shape and size knobs (``energySweep``'s phase A) for that.

    ``rigidify`` (ON by default) points the edge targets at the regular n-gon first, exactly as
    ``energySweep``'s handoff does, and constrains area and edges together. Without it this compresses
    whatever quadrilaterals the previous relaxation left, and the density it reports is a statement
    about THOSE shapes rather than about squares -- the same trap ``energySweep`` guards by refusing to
    accept a verdict before its rigid handoff. Turn it off only when the distorted shapes are the
    object of study.

    Returns a ``SweepResult`` whose ``phi`` is the density reached at the final pressure."""
    if getattr(model.packing, "containerIndex", None) is None:
        raise ValueError("compressToJamming needs a container: setBoundaryConditions('fixed') with an "
                         "addShape wall. Without one there is no box to give a degree of freedom to.")
    if rigidify:
        _setRegularEdgeTargets(model)
        model.setConstraints(area = True, edge = True)
    result = SweepResult()
    pressures = _geometric(pressure, finalPressure, int(pressureRounds))
    bar = _progressBarOrNone(progressBar, len(pressures))
    shapeLimits = _shapeLimits(model) if drawEvery else None

    for index, currentPressure in enumerate(pressures):
        # Solve dH/dlambda = 0 at this pressure. The residual is monotone in lambda over the range that
        # matters -- shrinking the box can only stiffen the contact -- so a sign change brackets the
        # root and bisection finishes it. Before a bracket exists the step is geometric and capped by
        # ``maxScaleStep``, because an uncapped secant step on a contact law that is flat until first
        # touch will happily jump the box through the packing.
        low = high = None
        for _ in range(int(maxBoxSteps)):
            _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
            enthalpy, slope = model.boxScaleGradient(currentPressure)
            scale = max(abs(enthalpy), model.getBoxArea() * currentPressure, 1e-300)
            if abs(slope) <= boxTolerance * scale:
                break
            if slope > 0.0:                       # box wants to shrink
                high = 1.0 if high is None else high
                step = 1.0 / maxScaleStep if low is None else None
            else:                                 # box wants to grow
                low = 1.0 if low is None else low
                step = maxScaleStep if high is None else None
            if step is None:
                step = np.sqrt(maxScaleStep) if slope < 0.0 else 1.0 / np.sqrt(maxScaleStep)
            model.scaleBox(step)
        overlap = model.getOverlapArea()
        step = result.record("compressBox", model, overlap)
        step["pressure"] = currentPressure
        if bar is not None:
            bar.update(1)
        _maybeDraw(model, drawEvery, len(result.history), "compressBox", step["phi"], overlap,
                   limits = shapeLimits)
        if verbose:
            print(f"  box      p {currentPressure:.3e}  phi {step['phi']:.6f}  "
                  f"boxArea {model.getBoxArea():.6f}  overlap {overlap:.3e}")
    if bar is not None:
        bar.close()

    valid, pairOverlap, penetration = _packs(model, 0.0, _WALL_TOLERANCE_FRACTION)
    result.phi = model.getPackingFraction()
    result.packed = bool(valid)
    result.overlap = model.getOverlapArea()
    result.record("final", model, result.overlap)
    return result


def _progressBarOrNone(progress, total):
    """The minimizers' own tqdm helper, reused so a box run looks like every other progress bar."""
    if not progress:
        return None
    from minimize import _progressBar
    return _progressBar(total, "box compression", progress)


# UNVERIFIED(Cam)
def energyScale(model):
    """The unit ``excessEnergy`` counts in: one polygon indented by a WHOLE edge length.

    A bare contact energy is not transferable, because the tiers do not even share units. The
    exact-distance law integrates ``(k/3) d^3`` along a contact, so its energy carries ``k L^4``; the
    sharp and mollified tiers measure an overlap AREA, ``L^2``. Dividing by the matching power of the
    mean edge, by the contact stiffness, and by the polygon count leaves a number that means the same
    amount of overlap at any N, any size and any stiffness -- which is what makes it usable as a
    control parameter a schedule can hold fixed."""
    meanEdge = _meanEdge(model)
    count = max(_freeCount(model), 1)
    if model.modelType == "depth":
        return count * float(model.depthStiffness) * meanEdge ** 4
    if model.modelType == "softDepth":
        return count * float(model.softStiffness) * meanEdge ** 4
    return count * meanEdge ** 2


# UNVERIFIED(Cam)
def _setDensity(model, phi):
    """Move to density ``phi`` AFFINELY, with the container left at whatever size it already is.

    THE BOX NEVER MOVES. ``setPackingFraction`` resizes the polygons about their own centroids, and in
    a fixed box that is exactly the affine compression, merely written in the box's frame. Shrink the
    box by 1/f about its centre, map every polygon's centroid affinely with it, and leave the polygons
    their own size; then rescale the whole picture by f to put the box back where it was. Centroids
    land back where they started and the polygons come out f times larger -- which is what this call
    does in one step. No SHAKE repair is needed either, since the targets scale with the geometry.

    ``scaleBox`` is NOT the same move and must not be substituted. It walks the wall while leaving the
    centroids at fixed ABSOLUTE positions, so their fractional coordinates spread outward and the
    deformation is not affine. Decompressing that way is pathological: the walls simply retreat from a
    cluster that never expands, the contact energy collapses to the noise floor, and the controller
    reads "unjammed" from a packing whose interior has not moved at all.

    (``compressToJamming`` does move the box, and legitimately -- there the box carries a degree of
    freedom under an applied pressure, so the wall does mechanical work on the packing rather than
    teleporting past it. That is a different protocol, not a different spelling of this one.)"""
    model.setPackingFraction(phi)
    return model


# Step ratio used while the packing is FAR from the requested excess, in either direction -- there is
# no branch worth following there, so the fine ladder buys nothing and costs a relaxation per percent
# of density. Below the target it strides up to first contact; above it, it backs off past the target
# so the final approach can be a compression.
#
# THE TEST IS A MARGIN, NOT ZERO. In principle the contact energy is identically zero below jamming;
# in practice it is not. Measured on 6 equilateral octagons while the box was closed 5% at a time, it
# read 3.8e-11, 1.8e-11, 8.7e-12, 1.8e-11, 4.6e-11, 1.7e-11, 2.2e-12, 2.4e-11, 2.4e-11 across
# phi = 0.43 .. 0.93 -- wandering with NO trend, which is residual settling at the ~3e-12 force noise
# floor rather than contact -- and then jumped four decades to 2.3e-07 at phi = 1.03. A test against
# exactly zero never fires against that floor, and the controller crawls the whole journey at the fine
# step. A test against a fraction of the TARGET does fire, and is the right shape besides: a decade
# down in energy is 2.2x down in indentation under the cubic law, so it is nowhere near the state being
# aimed at. Overshoot is not a risk either way -- the first stride past the target brackets it, and
# everything after that is a clipped secant inside the bracket.
_FREE_FLIGHT_STEP = 1.05
_FREE_FLIGHT_MARGIN = 0.1
# Density past which compressing further cannot be answering the question. At phi > 1 the polygons must
# overlap by (phi-1)/phi of their own area, so at 1.5 a third of the packing is double-covered -- well
# outside the contact law's validity limit dMax/rIn << 1, past which the repulsion reverses sign.
_MAX_USEFUL_DENSITY = 1.5
# How far the density may fall while buying nothing before the packing is called stuck rather than
# merely dense. Halving phi quarters the area available to the overlap, so an energy that has not even
# halved over that is not responding to density at all.
_STUCK_DENSITY_DROP = 2.0
_STUCK_ENERGY_DROP = 2.0
# Share of the contact energy in the WALL above which a failed run is blamed on the boundary rather
# than on the density. Some wall load is normal in a confined packing -- the wall does carry the
# confinement -- but past this the polygons are relieving stress by extruding through it instead of
# bearing on each other, and no amount of compressing will change that. Measured at wallStiffness = 1:
# 94-100%.
_WALL_DOMINANCE = 0.9


# UNVERIFIED(Cam)
def holdExcessEnergy(model, excess, tolerance = 0.05, maxDensityStep = 1.01, maxRounds = 80,
                     maxUnbalancedForce = 1e-8, maxSteps = 20000, minimizer = "lbfgs",
                     innerProgressBar = False, verbose = False):
    """Move the density until the RELAXED contact energy sits a fixed EXCESS above jamming.
    Returns ``(excess, phi)`` as achieved.

    WHY ENERGY RATHER THAN DENSITY. A schedule that holds phi fixed has to be told a density that is
    above jamming, and that density is not knowable in advance -- it is the answer being looked for. It
    also MOVES during a run: compliant 32-gons at ``kappa = 4`` mould around each other and jam far
    denser than the rigid squares they are on their way to becoming, so a phi chosen to overlap the
    squares leaves the compliant shapes rattling in free space with nothing touching and nothing to
    rearrange. The anneal then does no work at all and the sweep reports a lower bound rather than a
    result. Excess energy has no such problem: it is zero below jamming by construction and positive
    above it, so asking for a positive value states the requirement DIRECTLY.

    It is two-sided for the same reason. A packing that is too loose is compressed, one that is
    overjammed is decompressed, and neither case has to be anticipated by the caller.

    THE DENSITY MOVES AFFINELY AND THE BOX NEVER DOES -- see ``_setDensity``, which is where the
    reasoning lives. Walking the wall instead leaves the polygons at fixed absolute positions, and
    decompressing that way retreats the walls from a cluster that never expands.

    THE APPROACH IS SYMMETRIC. Whichever side it starts on, it strides while more than a decade from
    the target and ladders for the last decade.

    THE TWO DIRECTIONS DO NOT AGREE, AND NOT BY A LITTLE. Driven to the same energy from both sides,
    ``tests/excessEnergyCheck.py`` check 3 measured the two ending 137% apart in phi (0.4067 against
    0.9626), the DECOMPRESSING run the denser. An earlier reading of the same check gave 4.3% with the
    opposite sign, but that was taken before the excess counted body contact only and before the wall
    could be stiffened, so it does not stand. What survives is the qualitative point: the state reached
    depends on the history and not only on the target, so a density reported here is a property of the
    route as well as of the shapes. No attempt is made to compensate -- which direction is BETTER is
    not established, and picking one on the strength of a single configuration is how the 4.3% figure
    came to be quoted in the first place.

    THE LAST LEG IS A LADDER, NOT A JUMP. Within a decade of the target every step is capped at
    ``maxDensityStep`` and followed by its own relaxation, so the configuration is carried along a
    branch. Further out -- and in either direction -- it strides, because the packing is not in the
    state being aimed at and there is no branch worth following there.

    Once the target is BRACKETED the next density comes from a secant on ``log E`` against ``log phi``.
    Near jamming ``E ~ C (phi - phiJ)^alpha``, so the slope measured between the last two relaxed states
    is the right local derivative and the step is a Newton step on the scaling law -- no exponent is
    assumed, one is measured. The step cap does NOT apply there: the bracket is never wider than a
    single stride and is itself the constraint, and capping on top of it pinned the controller to the
    cap every round.

    ``tolerance`` is a band on ``log E``, so 0.05 means "within 5% of the requested energy". That is a
    much tighter requirement on the geometry than it sounds: with ``alpha`` around 3 to 4 it pins
    ``phi - phiJ`` to better than 2%.

    CHOOSING THE VALUE. It has a floor and a ceiling, both measured on the depth tier. The floor is the
    residual a converged relaxation leaves behind -- around 1e-9 in these units, and it is NOISE, with
    no trend in density at all (see ``_FREE_FLIGHT_MARGIN`` for the run), so anything within a decade of
    it is unreachable however many rounds are spent. The ceiling is where the overlap stops being a
    nudge: 2.4e-05 came with 13% of a polygon's area overlapping, which is not a packing being pressed
    together but one being crushed. Between them, 1e-06 landed at roughly 3% areal overlap, which is
    the intended "just above jamming". Start there and move by decades."""
    excess = float(excess)
    if excess <= 0.0:
        raise ValueError(f"excessEnergy must be positive, got {excess}. The controller drives the "
                         f"packing to a state that OVERLAPS by a set amount; zero IS jamming, which "
                         f"is what the decompression phase exists to find.")
    step = float(maxDensityStep)
    if step <= 1.0:
        raise ValueError(f"maxDensityStep must exceed 1, got {step}")
    band = abs(float(tolerance))

    # A polygon whose CENTROID has left the box cannot be brought back by a density move. The affine
    # step scales each polygon about its own centroid, so an outside centroid stays outside and only
    # grows -- the density then means something different from what it says, since part of the packing
    # is not in the container at all. Diagnosed here because it is a setup error, not a search failure.
    meanEdge = _meanEdge(model)
    if model.getWallPenetration() > 0.5 * meanEdge:
        warnings.warn(
            f"\n*** a polygon is {model.getWallPenetration() / meanEdge:.2f} edge lengths outside the "
            f"container ***\n"
            f"    the density move scales each polygon about its own centroid, so one that is already "
            f"outside stays outside and only gets bigger. Relax with the container term first and "
            f"check that nothing is left out before asking for a density.", stacklevel = 2)

    phi = model.getPackingFraction()
    _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
    energy = model.getExcessEnergy()
    startPhi, startEnergy = phi, energy
    low = high = None
    previous = None
    reached = False
    stuck = False

    for _ in range(int(maxRounds)):
        if verbose:
            print(f"  hold     phi {phi:.6f}  excess {energy:.4e}  (target {excess:.4e})  "
                  f"pairOverlap {model.getPairOverlapArea():.3e}")
        if energy > 0.0 and abs(np.log(energy / excess)) <= band:
            reached = True
            break
        # EVERY point brackets; only some are fit to extrapolate from. Conflating those two cost a run:
        # refusing a noise-floor point as a bracket end left the controller with a `high` and no `low`
        # across an onset steeper than one stride, so it strode 5% down, 5% up, and oscillated until it
        # ran out of rounds -- ending at exactly 1.0203/1.05^2 twice over. A point below the target
        # bounds the answer from below no matter how small its energy is; what it cannot do is anchor a
        # SECANT, because a slope measured from the noise floor spans the whole onset knee (four decades
        # of energy over 5% of density in one measured run, a local exponent near 190 describing
        # nothing). So it brackets, and the secant is gated separately.
        if energy > excess:
            high = phi
        else:
            low = phi
        if low is not None and high is not None:
            # BRACKETED. Geometric bisection is the fallback and is always valid. The secant on log E
            # against log phi replaces it only when BOTH endpoints of the slope carry real signal --
            # near jamming E ~ C (phi - phiJ)^alpha, so a slope measured between two loaded states is
            # the right local derivative and the step is a Newton step on the scaling law, with no
            # exponent assumed.
            #
            # No step cap applies here, and applying one was a separate bug: the bracket is itself
            # never wider than a single stride, so clipping the secant to maxDensityStep on top of it
            # pinned the controller to the cap every round and it crawled instead of converging.
            nextPhi = np.sqrt(low * high)
            signal = excess * _FREE_FLIGHT_MARGIN
            if (previous is not None and previous[0] != phi
                    and previous[1] >= signal and energy >= signal):
                slope = np.log(energy / previous[1]) / np.log(phi / previous[0])
                if slope > 0.0:
                    nextPhi = phi * np.exp(np.log(excess / energy) / slope)
            # Clip back INSIDE the bracket, leaving a margin so it always shrinks: this is what turns
            # a secant that overshoots into something no worse than a bisection.
            margin = (high / low) ** 0.1
            nextPhi = min(max(nextPhi, low * margin), high / margin)
        elif energy > excess:
            # ABOVE and not bracketed: decompress. The packing can be many decades over -- 2.1e-02
            # against a target of 1e-06 in one measured start, and phi = 1.02 out of the box for
            # amorphous 32-gons -- so creeping down at the fine step just exhausts maxRounds. Stride
            # while more than a decade over, ladder for the last decade.
            nextPhi = phi / (_FREE_FLIGHT_STEP if energy > excess / _FREE_FLIGHT_MARGIN else step)
        else:
            # BELOW and not bracketed: compress, by the mirror-image rule.
            nextPhi = phi * (_FREE_FLIGHT_STEP if energy < excess * _FREE_FLIGHT_MARGIN else step)
        # A RUNAWAY COMPRESSION IS A BROKEN ENERGY, NOT A DENSE PACKING. If the contact energy never
        # responds, the loop above keeps striding: 80 rounds at the coarse stride is a factor of 49 in
        # phi, and the polygons end up many times the size of the box with the intersection scan
        # grinding on the wreckage. Past phi = 1 the shapes must overlap by (phi-1)/phi of their own
        # area, so by 1.5 a third of the packing is double-covered and the contact law is far outside
        # its validity limit dMax/rIn << 1 -- beyond the medial axis its repulsion reverses sign. There
        # is nothing to find up there, so it stops and says why.
        # OVERLAP THAT DECOMPRESSION CANNOT RELIEVE. An affine move scales each polygon about its own
        # centroid, so two bodies sharing a centroid stay concentric at EVERY density -- no affine
        # protocol can separate them, because an affine map takes coincident points to coincident
        # points. A configuration tangled like that reads high forever and the loop would decompress
        # to nothing chasing it: measured, skipping the pre-relax sent phi from 1.0203 to 0.0206, a
        # factor of 50, with the excess still at 1.563e-03 the whole way down. The test is a lack of
        # RESPONSE, not a density: halving phi must buy something.
        if phi < startPhi / _STUCK_DENSITY_DROP and energy > startEnergy / _STUCK_ENERGY_DROP:
            stuck = True
            break
        if nextPhi > phi and nextPhi > _MAX_USEFUL_DENSITY:
            warnings.warn(
                f"\n*** compression ran away: phi would pass {_MAX_USEFUL_DENSITY:g} with the excess "
                f"still at {energy:.3e} ***\n"
                f"    asked for {excess:.3e}. Above phi = 1 the polygons must overlap by (phi-1)/phi "
                f"of their area, so an energy this small there is not measuring contact -- suspect the "
                f"tier or the neighbour list rather than the density.\n"
                f"    Stopping at phi = {phi:.6f}.", stacklevel = 2)
            break
        previous = (phi, energy)
        phi = nextPhi
        _setDensity(model, phi)
        _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
        energy = model.getExcessEnergy()

    if stuck:
        warnings.warn(
            f"\n*** the overlap is not responding to density: this configuration is tangled ***\n"
            f"    phi {startPhi:.5f} -> {phi:.5f} bought only excess {startEnergy:.3e} -> "
            f"{energy:.3e}, against a target of {excess:.3e}.\n"
            f"    A density move is AFFINE, and an affine map takes coincident points to coincident "
            f"points -- so two polygons sharing a centroid stay concentric however far this "
            f"decompresses. Relax the configuration apart first (the container term on the spring "
            f"tier does this); no amount of decompression will.", stacklevel = 2)
    elif not reached:
        # Three failures wear the same symptom and their advice is different, so they are told apart
        # before anything is printed.
        #
        # THE WALL GAVE WAY is checked first because it is the one that looks like progress. The excess
        # counts body contact only, but the packing is free to relieve stress through the boundary
        # instead, and if the wall is the softer route compression simply extrudes the packing: the
        # target is never reached however many rounds are spent, because the density is not what is
        # holding it back. Measured at wallStiffness = 1 on the depth tier, 99.7% of the contact energy
        # sat in the wall with 24 vertices outside the box.
        wallShare = 0.0
        if model.modelType != "softDepth":
            total = model.getContactEnergy()
            if total > 0.0:
                wallShare = 1.0 - model.getPairContactEnergy() / total
        if wallShare > _WALL_DOMINANCE:
            diagnosis = (f"    {100 * wallShare:.1f}% of the contact energy is in the WALL, not "
                         f"between bodies: the packing is relieving stress by extruding through the "
                         f"boundary rather than by bearing on itself, so compressing further only "
                         f"pushes more of it out.\n"
                         f"    Raise the wall stiffness -- setDepthContact(wallStiffness = ...) -- so "
                         f"that escaping costs more than touching a neighbour.")
        elif energy > excess and low is None:
            diagnosis = (f"    it never got below the target, so {excess:.3e} may sit under the noise "
                         f"floor -- a converged relaxation leaves around 1e-9 in these units and that "
                         f"residual has no trend in density at all. Ask for more, not for more rounds.")
        else:
            diagnosis = (f"    steps within a decade of the target are capped at {step:g}x in phi, so "
                         f"a long journey costs rounds -- raise maxRounds, or maxDensityStep to move "
                         f"faster at the price of following the branch less closely.")
        warnings.warn(
            f"\n*** holdExcessEnergy did not reach {excess:.3e} in {int(maxRounds)} rounds ***\n"
            f"    stopped at excess {energy:.3e}, phi {model.getPackingFraction():.6f}.\n"
            f"{diagnosis}", stacklevel = 2)
    return energy, model.getPackingFraction()


def energySweep(model, finalPolydispersity = None, finalEnergy = 0.0, finalSigma = None,
                annealRounds = 10, phiStep = 0.004, refineRounds = 10, minPhi = 0.2,
                maxUnbalancedForce = 1e-8, maxSteps = 20000, progressBar = False,
                innerProgressBar = False, verbose = False, drawEvery = None,
                finishRigid = True, annealShape = True, finalDistortion = 1e-6,
                shapeRounds = 12, sharpDecompress = None, maxSigmaRatio = 2.0,
                wallTolerance = None, compressStep = None, compressRounds = 6,
                minimizer = "cg", maxShapeRatio = 2.0, excessEnergy = None,
                excessTolerance = 0.05, maxDensityStep = 1.01):
    """Anneal the shape distribution and the contact sharpness, then decompress until the packing is
    valid. Returns a ``SweepResult``; the model is left in the packed configuration.

    ``excessEnergy`` REPLACES THE FIXED DENSITY OF PHASE A. Left at None the anneal runs at whatever
    density it was handed, which requires the caller to already know a density above jamming. Set to a
    positive number it instead holds the dimensionless contact energy of ``Model.getExcessEnergy`` at
    that value throughout the anneal, re-establishing it after every round. See ``holdExcessEnergy``
    for why that is the better control parameter -- briefly, the jamming density is the answer being
    searched for and it MOVES as the shapes stiffen, so no fixed phi can stay just above it.

    THE VERDICT IS TWO TESTS. ``finalEnergy`` (default 0.0) applies to the POLYGON-POLYGON overlap
    area, which is exact: measured across a density sweep through jamming it reads identically
    ``0.000000e+00`` at every valid density, so it is tested against zero and needs no tolerance.
    ``wallTolerance`` (default 1e-4 of the mean edge) applies to CONTAINMENT, as a penetration DEPTH.

    Containment cannot be tested against zero. What survives a long relaxation is a corner just
    clipping the wall, whose overlap area goes as ``delta^2`` and whose restoring force goes as
    ``delta^3`` -- measured slopes 1.978 and 2.966 against the predicted 2 and 3. The minimizer stops
    when that force sinks into the ~3e-12 force noise, not when the geometry is clean, and each factor
    of ten in ``delta`` costs a factor of a thousand in the noise floor. The earlier default of a
    1e-12 tolerance on the TOTAL overlap therefore rejected every genuinely jammed state and kept
    decompressing until nothing was touching at all: measured, it returned 0.665692 where the packing
    is valid to at least 0.673692, with ``max|F| = 0.0`` exactly at the density it reported.

    Prefer a depth to an area, because the two differ by a square: the resting state carried 3.56e-10
    of outside area, which sounds negligible, and 1.88e-05 of depth, which is 7.5e-05 of an edge.

    ``finalEnergy`` was the tolerance on residual OVERLAP AREA, not on the relaxation energy -- see the
    module docstring for why the energy cannot be the test.

    KEEP IT NEAR ROUNDOFF. A valid packing has EXACTLY zero overlap, so any nonzero tolerance accepts a
    genuinely overlapping state and biases the answer UPWARD -- it reports a denser packing than is
    actually achievable. Measured: at ``finalEnergy = 1e-5`` a run returned phi = 0.587499 carrying
    8.1e-06 of real overlap, when the last exactly-zero density was 0.58. The default is therefore
    1e-12, and ``result.overlap`` always reports what was left so the answer cannot be misread.

    ``finalSigma`` defaults to the dynamics floor (0.01 of the mean edge); a smaller request is clamped
    with a warning, since relaxing there diverges and the verdict does not need it.

    Three phases:

      A  ANNEAL at fixed density -- ramp polydispersity and sigma to their targets, relaxing at each
         step. The verdict has to be taken at the FINAL values: "it packs" at a wide distribution with
         rounded corners is a statement about rounded polydisperse shapes, not about the ones being
         searched for.
      B  DECOMPRESS -- step the density down by ``phiStep`` until the overlap vanishes, re-relaxing
         each time. Small steps so the branch is followed continuously rather than hopping basins.
      C  REFINE -- bisect the resulting bracket, each trial one relaxation from a good configuration.

    THE SHAPE FREEDOM IS ANNEALED, NOT CONFISCATED (``annealShape``). The shapes are held only by their
    total distortion ``sum_i d_i`` (see ``Model.setShapeBudget``), and that budget is walked down
    ALONGSIDE the density through phase B, so a polygon is still free to stay bent while its neighbor
    straightens, exactly while the overlap is being relieved. Only when the budget reaches
    ``finalDistortion`` do the polygons become rigid.

    The alternative -- the one this replaces -- projected the annealed shapes onto rigid regular
    polygons in a single step before decompression, and measurably gave back everything the anneal had
    won: overlap 2.593e-03 before the anneal, 9.207e-04 after it, and 2.905e-03 the instant the shapes
    were projected, slightly WORSE than the starting state. Set ``annealShape = False`` to get that
    behavior back for comparison.

    THE MOLLIFICATION IS TURNED OFF EARLY (``sharpDecompress``). Phase A needs it -- a smooth landscape
    is what lets the shapes flow -- but phase B is where the answer is decided, and there the Plummer
    contact is an active liability: it does not vanish on a valid packing (measured 8.2e-04 where the
    true overlap was exactly zero), so it keeps pushing polygons apart after they have stopped
    touching, and the density it settles at is looser than the one the shapes actually admit.
    Decompression therefore runs on the SHARP tier, whose energy is identically zero below jamming and
    which needs no sigma at all. ``maxSigmaRatio`` caps how fast sigma may fall per anneal round,
    lengthening phase A rather than letting the contact stiffness outrun the relaxation.
    """
    result = SweepResult()
    # THE VERDICT IS TWO TESTS, because only one of the two quantities is exact.
    #
    # Polygon-polygon overlap is a perfect sign change: measured across a density sweep through
    # jamming it reads identically 0.000000e+00 at every valid density, so it is tested against ZERO
    # and needs no tolerance at all.
    #
    # Containment cannot be. What survives a long relaxation is a CORNER just clipping the wall, whose
    # overlap area goes as delta^2 and whose restoring force goes as delta^3 (measured slopes 1.978 and
    # 2.966 against the predicted 2 and 3). The minimizer stops when that force sinks into the ~3e-12
    # force noise, not when the geometry is clean, and buying a factor of ten in delta costs a factor
    # of a thousand in the noise floor. So containment gets a tolerance -- expressed as a DEPTH, since
    # an area tolerance hides a square root: the resting 3.56e-10 of area is 1.88e-05 of depth.
    meanEdge = _meanEdge(model)
    if wallTolerance is None:
        wallTolerance = _WALL_TOLERANCE_FRACTION * meanEdge
    wallTolerance = float(wallTolerance)
    floor = _MIN_STABLE_SOFTENING_FRACTION * meanEdge

    # THE CONTACT TIERS HAVE NOTHING TO ANNEAL. Phase A exists to walk a numerical regulator down:
    # sigma is a mollification width, and the landscape is smoothed early and made faithful late. Soft
    # depth's ``epsilon`` is not that -- it is a SHAPE parameter setting the corner rounding radius
    # (eq 19), so driving it to zero would sharpen the particles rather than remove an approximation.
    # The exact-distance ``depth`` tier has no regulator AT ALL: it is closed form with no quadrature,
    # no softening and no tolerance, so there is nothing that could be annealed even in principle.
    # Either way the softening ladder is skipped; every other schedule here -- size, width, the phi
    # ladder, decompression, the verdict -- is tier-agnostic and runs unchanged.
    softDepthTier = model.modelType in ("softDepth", "depth")
    if sharpDecompress is None:
        # Sharp decompression exists because the Plummer tail stays repulsive after the shapes have
        # separated and holds the packing open. Soft depth's contact law returns exactly 0.0 for
        # h <= 0 -- no tail -- so it can decompress on its own energy. The VERDICT is unaffected
        # either way: _packs reads exact geometry, not the tier's energy.
        sharpDecompress = not softDepthTier
    if finalSigma is None:
        sigmaTarget = floor
    else:
        sigmaTarget = float(finalSigma)
        if sigmaTarget < floor:
            warnings.warn(
                f"\n*** finalSigma = {sigmaTarget:.2e} is below the dynamics floor ***\n"
                f"    sigma must stay above {floor:.2e} ({_MIN_STABLE_SOFTENING_FRACTION} of the mean "
                f"edge) or the 1/sigma contact force makes FIRE/CG diverge. Clamping to it.\n"
                f"    Nothing is lost: validity is read from the EXACT overlap area, which has no "
                f"sigma in it.", stacklevel = 2)
            sigmaTarget = floor

    # Lengthen the anneal if sigma would otherwise fall too fast; every ramp uses the same count so the
    # three schedules stay in step.
    if not softDepthTier:
        annealRounds = _slowRounds(model.sigma, sigmaTarget, annealRounds, maxSigmaRatio)

    # START ABOVE JAMMING BY CONSTRUCTION, rather than by being told a density that is. The energy
    # controller is two-sided, so this both rescues a packing that was handed a density below its
    # jamming point -- where nothing touches, the anneal has nothing to rearrange, and the sweep can
    # only report the lower bound it was given -- and backs off one that was handed a density far above
    # it. It runs BEFORE startPhi is taken so the compression phase's ceiling is the density actually
    # started from.
    if excessEnergy is not None:
        holdExcessEnergy(model, excessEnergy, tolerance = excessTolerance,
                         maxDensityStep = maxDensityStep, maxUnbalancedForce = maxUnbalancedForce,
                         maxSteps = maxSteps, minimizer = minimizer,
                         innerProgressBar = innerProgressBar, verbose = verbose)

    # The density the sweep starts from, kept because the compression phase must never push past it:
    # that density was already established to fail, so trying it again only wastes relaxations.
    startPhi = model.getPackingFraction()

    bar = _SweepBar(progressBar, annealRounds, model.getPackingFraction(), phiStep, minPhi,
                    refineRounds)
    shapeLimits = _shapeLimits(model) if drawEvery else None
    _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
    valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
    if valid:
        bar.close()
        warnings.warn(
            f"\n*** the packing was already valid at phi = {model.getPackingFraction():.6f} ***\n"
            f"    pair overlap {overlap:.3e} <= {finalEnergy:.1e} and wall penetration "
            f"{penetration:.3e} <= {wallTolerance:.1e} before any decompression, so there "
            f"is nothing to search: this density is a lower bound on what was achievable, not a "
            f"result. Start ABOVE the jamming density.", stacklevel = 2)
        result.phi = model.getPackingFraction()
        result.packed = True
        result.record("start", model, overlap)
        return result
    result.record("start", model, overlap)

    # --- Phase A: anneal polydispersity and sigma at fixed density ------------------------------
    # The SIZE distribution is what "polydispersity" means for a packing: it lives in targetArea, and
    # driving it to zero is what recovers the monodisperse problem. The edge-length moments cannot do
    # this job -- with the areas held rigid the edge lengths are already as equal as those areas allow,
    # so squeezing them only distorts shapes (measured: the edge CV would not move below 0.1117 while
    # the worst shape index went 4.07 -> 4.28).
    sizeStart = model.getSizePolydispersity()
    sizeTarget = 0.0 if finalPolydispersity is None else float(finalPolydispersity)
    # A RAMP WHOSE TARGET IS EXACTLY ZERO MUST BE LINEAR, NOT GEOMETRIC. The geometric form is right for
    # sigma, which stops at a positive floor -- halving it matters as much at 0.05 as at 0.005. It is
    # wrong here: a size CV of 1e-5 and one of 1e-9 are both just "monodisperse", so a multiplicative
    # ramp toward zero spends nearly every step in decades where no geometry changes. Measured on the
    # old form, _geometric(0.2222, 1e-9, 10) cut the spread 6.8x on the FIRST round and reached
    # 4.8e-03 by the second -- the anneal was effectively over after two of its ten rounds, and the
    # remaining eight relaxed at a width indistinguishable from zero. The 1e-9 was an arithmetic guard
    # standing in for a physical target. Linear spreads the rounds across the range that exists.
    if sizeStart <= 0.0:
        sizes = [None] * annealRounds
    elif sizeTarget <= 0.0:
        sizes = list(np.linspace(sizeStart, 0.0, annealRounds + 1)[1:])
    else:
        sizes = _geometric(sizeStart, sizeTarget, annealRounds)

    startWidth = max(model.getPolydispersity().get("edge", 0.0), 0.0) or None
    reachable = _reachableWidth(model)
    if finalPolydispersity is None:
        # PRESERVE the width the packing already has. The old default of driving it to ~0 conflated two
        # different things: the annealing freedom (how much shapes may vary DURING the search) and the
        # physical size distribution of the objects being packed. For equal objects those coincide;
        # for a deliberately polydisperse packing they are opposites, and squeezing the edge moments
        # destroys the size spread the user asked for -- by distorting the shapes, since the areas are
        # held rigid.
        widthTarget = startWidth if startWidth else reachable
    else:
        widthTarget = float(finalPolydispersity)
        if widthTarget < reachable * (1.0 - 1e-9):
            warnings.warn(
                f"\n*** finalPolydispersity = {widthTarget:.3g} is narrower than the fixed areas allow ***\n"
                f"    with the areas rigid, the edge lengths cannot be more equal than sqrt(A0) is: "
                f"the floor is {reachable:.4g}. Asking for less forces the polygons to stop being "
                f"regular -- measured, it tripled the overlap and drove the shape index from 4.07 to "
                f"4.28 before any minimization ran. Clamping.\n"
                f"    To end monodisperse, make the AREAS monodisperse (setMonoPerimeter) rather than "
                f"squeezing the edge moments.", stacklevel = 2)
            widthTarget = reachable
    widths = _geometric(startWidth if startWidth else 1.0, widthTarget, annealRounds) \
        if _hasMomentMechanism(model) else [None] * annealRounds
    sigmas = [None] * annealRounds if softDepthTier \
        else _geometric(model.sigma, sigmaTarget, annealRounds)
    for width, sigma, size in zip(widths, sigmas, sizes):
        if size is not None:
            model.setSizePolydispersity(size)
        elif width is not None:
            model.setTargetPolydispersity(width)
        if sigma is not None:
            model.setMollification(sigma)
        # RE-ESTABLISH the excess after every round, because each round moves the jamming density
        # underneath the packing: narrowing the size spread, sharpening the corners and stiffening the
        # shapes all change how densely these objects can sit. Holding phi instead would let the
        # packing drift below jamming partway through the anneal and coast the rest of it with nothing
        # in contact. The controller does its own relaxation, so it replaces the plain one.
        if excessEnergy is None:
            _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
        else:
            holdExcessEnergy(model, excessEnergy, tolerance = excessTolerance,
                             maxDensityStep = maxDensityStep,
                             maxUnbalancedForce = maxUnbalancedForce, maxSteps = maxSteps,
                             minimizer = minimizer, innerProgressBar = innerProgressBar,
                             verbose = verbose)
        step = result.record("anneal", model, model.getOverlapArea())
        bar.step("anneal", step["phi"], step["overlapArea"], model = model)
        _maybeDraw(model, drawEvery, len(result.history), "anneal", step["phi"],
                   step["overlapArea"],
                   limits = shapeLimits)
        if verbose:
            softening = f"sigma {sigma:.3e}" if sigma is not None \
                else f"eps {model.softEpsilon:.3e} (fixed)"
            print(f"  anneal   phi {step['phi']:.6f}  {softening}  "
                  f"sizeCV {model.getSizePolydispersity():.4f}  "
                  f"overlap {step['overlapArea']:.3e}")

    # --- Handoff: the shapes become rigid GRADUALLY, during the descent -------------------------
    # Whatever holds them, it has to be true by the time the density verdict is taken: a density only a
    # set of distorted quadrilaterals achieves is the same class of false result as letting the shapes
    # shrink to fit. The difference here is WHEN. With annealShape the budget is walked down through
    # phase B and the rigid step happens at the end of that ramp; without it, the projection happens
    # now, in one jump.
    budgets = []
    if finishRigid and annealShape:
        _enableShapeBudget(model)
        # The ramp runs on the isoperimetric DEFICIT, a length, while ``finalDistortion`` is a
        # dimensionless relative distortion. Convert with the same factor that relates them per
        # polygon, ``delta_i = d_i g_i sqrt(A_i)``, so the parameter keeps meaning "every polygon
        # within this much of regular" whichever form is running underneath.
        budgetStart = model.getShapeDeficit()
        counts = np.diff(np.asarray(model.packing.startIndices, dtype = int))[:_freeCount(model)]
        g = np.sqrt(4.0 * counts * np.tan(np.pi / counts))
        scale = float(np.sum(g * np.sqrt(np.abs(model.getAreas()[:_freeCount(model)]))))
        budgetEnd = float(finalDistortion) * scale
        # THE SHAPE RAMP GETS THE SAME RATE CAP SIGMA HAS. Both are annealing knobs walked down while
        # the configuration re-relaxes, and both fail the same way when a step outruns the relaxation:
        # the state that arrives at the next round has not caught up with the last one. Sigma has been
        # capped by ``maxSigmaRatio`` since it was written; the shape budget was not, and its default
        # ladder is STEEPER than sigma is allowed to be -- measured on 11 squares, 6.88e-01 -> 1.21e-05
        # over 12 rounds is a factor 2.49 per step against sigma's permitted 2.0.
        requested = int(shapeRounds)
        shapeRounds = _slowRounds(budgetStart, budgetEnd, requested, maxShapeRatio)
        if budgetStart > budgetEnd:
            budgets = _geometric(budgetStart, budgetEnd, int(shapeRounds))
        step = result.record("shapeAnneal", model, model.getOverlapArea())
        bar.step("shapeAnneal", step["phi"], step["overlapArea"], model = model)
        if shapeRounds > requested:
            # The notches are spent one per DENSITY step, so lengthening the ramp also moves the rigid
            # handoff down by (extra rounds) x phiStep. That is reported rather than silently absorbed:
            # the handoff density is the thing the schedule is really choosing.
            warnings.warn(
                f"\n*** shape anneal lengthened {requested} -> {shapeRounds} rounds ***\n"
                f"    the deficit ramp {budgetStart:.3e} -> {budgetEnd:.3e} would have cut by "
                f"{(budgetEnd / budgetStart) ** (-1.0 / requested):.2f}x per step, past "
                f"maxShapeRatio = {maxShapeRatio:g}.\n"
                f"    One notch is spent per density step, so the rigid handoff moves down by about "
                f"{(shapeRounds - requested) * phiStep:.4f} in phi. Lower phiStep to hold it where it "
                f"was, or raise maxShapeRatio to keep the old (faster) ladder.", stacklevel = 2)
        if verbose:
            print(f"  shape    deficit {budgetStart:.4e} -> {budgetEnd:.3e} "
                  f"(distortion {finalDistortion:.1e}) over {len(budgets)} decompression steps, "
                  f"{(budgetEnd / budgetStart) ** (-1.0 / max(len(budgets), 1)):.2f}x per step")
    elif finishRigid:
        _rigidify(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
        step = result.record("rigidify", model, model.getOverlapArea())
        bar.step("rigidify", step["phi"], step["overlapArea"], model = model)
        _maybeDraw(model, drawEvery, len(result.history), "rigidify", step["phi"],
                   step["overlapArea"],
                   limits = shapeLimits)
        if verbose:
            print(f"  rigidify phi {step['phi']:.6f}  overlap {step['overlapArea']:.3e}  "
                  f"max|C| {step['residual']:.1e}")
    elif not _shapeIsHeld(model):
        warnings.warn(
            "\n*** nothing is holding the SHAPES ***\n"
            "    the edges are neither rigid nor moment-constrained, so only each area is fixed -- and "
            "a fixed-area quadrilateral is any quadrilateral. The packing this returns will not be "
            "made of regular polygons, and its density describes whatever distorted shapes came out. "
            "Use finishRigid = True, or setConstraints(area = True, edge = [1, 2]).", stacklevel = 2)

    # --- Phase B: decompress until the overlap vanishes -----------------------------------------
    # The mollification comes off HERE, not at the end. It exists to smooth the landscape while the
    # shapes are moving; once the question is "do these shapes fit", a contact that stays repulsive
    # after the shapes have separated only holds the packing open.
    if sharpDecompress and model.modelType != "area":
        model.setModelType("area")
        if verbose:
            print(f"  sharp    mollification off (sigma was {model.sigma:.3e}); decompression and the "
                  f"verdict now use the same energy")

    high = model.getPackingFraction()                      # known to overlap
    low = float("nan")
    phi = high
    pending = list(budgets)
    rigid = not (finishRigid and annealShape)
    while phi - phiStep > minPhi:
        phi -= phiStep
        model.setPackingFraction(phi)
        # One notch of shape freedom is surrendered per density step, so the two schedules advance
        # together: the packing opens up and straightens at the same time rather than being straightened
        # first and opened afterwards.
        phase = "decompress"
        if pending:
            model.setShapeDeficit(pending.pop(0))
            phase = "shapeDescend"
        elif not rigid:
            _rigidify(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
            rigid = True
            phase = "rigidify"
        _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
        valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
        step = result.record(phase, model, overlap)
        bar.step(phase, phi, overlap, model = model)
        _maybeDraw(model, drawEvery, len(result.history), phase, phi, overlap,
                   limits = shapeLimits)
        if verbose:
            print(f"  {phase:12s} phi {phi:.6f}  pairOverlap {overlap:.3e}  "
                  f"wallDepth {penetration:.3e}  "
                  f"distortion {model.getMaxShapeDistortion():.3e}"
                  f"{'   <- packs' if (rigid and valid) else ''}")
        # A zero-overlap state reached while the shapes are still free is NOT the answer: it is a
        # statement about distorted quadrilaterals. The verdict waits for the rigid handoff.
        if rigid and valid:
            low = phi
            best = _snapshot(model, phi, overlap)
            break
        high = phi
    if not np.isfinite(low):
        bar.close()
        warnings.warn(
            f"\n*** never packed down to phi = {minPhi} ***\n"
            f"    the branch still overlaps at the lowest density tried. Either the anneal stranded "
            f"it (watch the shape index in the history) or minPhi is too high.", stacklevel = 2)
        return result

    # --- Phase C: refine the bracket ------------------------------------------------------------
    result.bracket = (low, high)
    for _ in range(refineRounds):
        middle = 0.5 * (low + high)
        model.setPackingFraction(middle)
        _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
        valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
        result.record("refine", model, overlap)
        bar.step("refine", middle, overlap, model = model)
        _maybeDraw(model, drawEvery, len(result.history), "refine", middle, overlap,
                   limits = shapeLimits)
        if valid:
            low = middle
            best = _snapshot(model, middle, overlap)
        else:
            high = middle
    # RESTORE the densest valid packing found rather than re-deriving it. Re-relaxing at ``best`` from
    # wherever the last refine trial ended lands in a different basin -- the reported density would
    # then not be backed by a valid configuration at all.
    _restore(model, best)

    # --- Phase D: COMPRESS the found packing back up --------------------------------------------
    # The descent's answer is path dependent and leaves density on the table: the configuration reached
    # by DESCENDING to a density is not the one reached by COMPRESSING into it, because every density
    # gets its own relaxation with its own history. Measured, the packing returned at phi = 0.665692
    # stayed valid -- polygon-polygon overlap identically zero -- when compressed back up to at least
    # 0.673692, about 0.008 of density the descent had given away.
    #
    # This does NOT reopen the decompress-rather-than-compress choice. That is about which BRANCH to
    # follow, and the branch is still the one decompression selected; this only asks how far that
    # branch can be squeezed before it first overlaps. The snapshot is kept at every accepted step, so
    # a failed trial can never leave the model in an invalid state.
    if compressStep is None:
        compressStep = phiStep / 4.0
    if compressRounds > 0:
        low = best["phi"]
        high = None
        phi = low
        # Never compress past where the sweep began: that density was already known to fail.
        while phi + compressStep <= startPhi:
            phi += compressStep
            model.setPackingFraction(phi)
            _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
            valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
            result.record("compress", model, overlap)
            bar.step("compress", phi, overlap, model = model)
            _maybeDraw(model, drawEvery, len(result.history), "compress", phi, overlap,
                       limits = shapeLimits)
            if verbose:
                print(f"  compress     phi {phi:.6f}  pairOverlap {overlap:.3e}  "
                      f"wallDepth {penetration:.3e}"
                      f"{'' if valid else '   <- first overlap'}")
            if not valid:
                high = phi
                break
            low = phi
            best = _snapshot(model, phi, overlap)
        if high is not None:
            for _ in range(compressRounds):
                middle = 0.5 * (low + high)
                model.setPackingFraction(middle)
                _relax(model, maxUnbalancedForce, maxSteps, innerProgressBar, minimizer)
                valid, overlap, penetration = _packs(model, finalEnergy, wallTolerance)
                result.record("compressRefine", model, overlap)
                bar.step("compressRefine", middle, overlap, model = model)
                if valid:
                    low = middle
                    best = _snapshot(model, middle, overlap)
                else:
                    high = middle
        _restore(model, best)

    result.record("final", model, model.getOverlapArea())
    bar.step("done", best["phi"], best["overlap"], model = model)
    bar.close()
    result.phi = best["phi"]
    result.overlap = best["overlap"]
    result.packed = True
    # The final state is always drawn, whatever the interval -- it is the answer.
    _maybeDraw(model, 1 if drawEvery else None, len(result.history), "final", result.phi,
               result.overlap, limits = shapeLimits)
    if verbose:
        print(f"  packed at phi = {result.phi:.6f}  (bracket {result.bracket[0]:.6f} .. "
              f"{result.bracket[1]:.6f})")
    return result


def _shapeIsHeld(model):
    """Whether anything constrains the SHAPE, as opposed to only the size.

    Area alone does not: a fixed-area quadrilateral is any quadrilateral. Shape needs the edges held,
    either rigidly or through their distribution."""
    constraints = model.constraints
    if constraints is None:
        return False
    if bool(getattr(constraints, "edge", False)) or bool(getattr(constraints, "perimeter", False)):
        return True
    distribution = getattr(constraints, "distribution", None)
    return distribution is not None and "edge" in distribution.families()


def _hasMomentMechanism(model):
    """Whether a polydispersity ramp has anything to drive -- transient targets or moment constraints."""
    if model.transient is not None:
        return True
    distribution = getattr(model.constraints, "distribution", None)
    return distribution is not None or bool(getattr(model.constraints, "moments", None))
