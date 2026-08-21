"""FIRE and gradient-descent minimizers (build step 1, reused at step 7).

Both take the packing and a ``forceEnergy`` callable

    forceEnergy(packing) -> (energy: float, force: flat (2N,) array, == -dE/dr)

and relax ``packing.positions`` in place. Hiding the energy model behind this
callable lets step 1 pass eqSoftBody while the later collision model passes its
own force routine (which can refresh neighbors / intersections on each call).

FIRE is the velocity-Verlet variant of Bitzek et al. (2006), mirroring the CUDA
reference's FIRE.h: half-kick / drift / half-kick, then mix the velocity toward
the force and adapt dt / alpha.

Positions are wrapped after every step by ``box.wrapIntoCell`` -- governed by the
box, not a flag. A free-space packing (box=None, the eqSoftBody shape-build) makes
it a no-op; a periodic box wraps into the area-1 cell. So wrapping is always applied
yet correct for both the free-space build (step 1) and the periodic packing (step 7).
"""

import warnings

import numpy as np

from packing import wrapPolygonsIntoCell


def maxForceMagnitude(force):
    """Largest per-vertex force magnitude in a flat (2N,) force array."""
    f = force.reshape(-1, 2)
    return float(np.sqrt(np.einsum("ij,ij->i", f, f)).max())

def applyPins(packing, vector):
    """Zero the entries of a flat (2N,) force/velocity belonging to PINNED vertices, in place.

    A pinned vertex is held fixed: it still pushes on its neighbors and still enters its polygon's
    area / edge terms, it just never moves. Zeroing its force is also what makes the convergence test
    honest -- the reaction force a pin carries is not an unbalanced force, exactly as a constraint's
    normal component is not (see ``_projectForce``). Applied by every minimizer, so pinning works
    with the spring model as well as under constraints."""
    pinned = getattr(packing, "pinned", None)
    if pinned is not None:
        vector.reshape(-1, 2)[pinned] = 0.0
    return vector

def _projectForce(constraints, packing, vector, basis = None):
    """Tangent-space part of a force/velocity when running constrained; the vector unchanged when
    ``constraints`` is None. The tangential force is the TRUE residual on the manifold, so it is what
    the convergence test must see -- the constraint-normal part is carried by the constraints and is
    not an unbalanced force. ``basis`` reuses a decomposition already built at these positions.
    Pins are applied either way, since they are independent of the shape constraints."""
    if constraints is None:
        return applyPins(packing, np.array(vector, dtype = float))
    return applyPins(packing, constraints.projectVector(packing, vector, basis = basis))

def _normalBasis(constraints, packing):
    """The constraint-normal basis at the current positions, or None when unconstrained. Built once
    per step and shared by every projection at that configuration -- it is the expensive part."""
    return None if constraints is None else constraints.normalBasis(packing)

def _retract(constraints, packing, shakeTol, shakeMaxIter):
    """SHAKE ``packing.positions`` back onto the constraint manifold (no-op when unconstrained). A
    first-order step leaves the manifold at O(dt^2); this is the retraction that puts it back."""
    if constraints is not None:
        constraints.projectPositions(packing, tol = shakeTol, maxIter = shakeMaxIter)

# UNVERIFIED(Cam)
def _dumpState(packing, tag, **extra):
    """Write the failing configuration to `data/` and return the path (or a note on why it could not).

    A failure that takes an hour of cascade to reach can only be studied if it survives the traceback.
    Dumping at the raise turns a one-shot event into a file that can be reloaded, plotted and bisected
    offline with no GPU and no rerun -- which matters most for exactly the faults that resist a guess,
    where the alternative is another hour per hypothesis.

    Deliberately NOT in the scratchpad: this is evidence Cam has to be able to find."""
    try:
        import datetime
        import os
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(folder, exist_ok = True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = os.path.join(folder, f"failure-{tag}-{stamp}.npz")
        fields = {"positions": np.asarray(packing.positions, dtype = float)}
        # `startIndices` is the CSR boundary array and is what makes the dump readable at all: vertex
        # counts are NOT uniform across a cascade, so a reader that assumes one n silently merges every
        # polygon into a single ring and reports areas for a shape that does not exist.
        for name in ("velocities", "force", "targetArea", "targetPerimeter", "targetEdgeLength",
                     "targetDiagonal", "startIndices", "shapeId", "next", "prev", "pinned"):
            values = getattr(packing, name, None)
            if values is not None:
                fields[name] = np.asarray(values)
        fields.update({name: np.asarray(value) for name, value in extra.items()})
        np.savez(path, **fields)
        return path
    except Exception as error:
        return f"<state dump failed: {type(error).__name__}: {error}>"


# UNVERIFIED(Cam)
def _checkFinite(packing, step, dt, force, source):
    """Stop AT the step that diverged, naming it, rather than letting the NaN travel.

    A non-finite position is silent where it happens and loud somewhere else: the next constraint
    retraction reports a Jacobian that is not finite for every polygon at once, which points at the
    constraint set -- the one part of the loop that did not do anything wrong. By then ``dt``, the
    force that produced the step and the step number are all gone, and those are what say whether the
    timestep ceiling was exceeded or the state was already sick going in."""
    if np.all(np.isfinite(packing.positions)):
        return
    bad = int((~np.isfinite(packing.positions.reshape(-1, 2)).all(axis = 1)).sum())
    dump = _dumpState(packing, "positions")
    raise FloatingPointError(
        f"{source} produced {bad} non-finite vertices at step {step} (dt = {dt:.4e}, "
        f"max|F| = {force:.4e} going in).\n"
        f"    The positions are gone, so nothing after this point is meaningful -- and the error is "
        f"raised HERE rather than at the next constraint retraction, which would have blamed the "
        f"constraint set instead.\n"
        f"    A large max|F| with dt at its ceiling is the timestep: lower dtMax. A modest max|F| "
        f"means the state was already sick going in -- and since the force evaluation is checked "
        f"separately the step before, that points at the retraction or the transient step, not at "
        f"the contact tier.\n"
        f"    State written to {dump}")


# UNVERIFIED(Cam)
def _checkForce(packing, energy, force, step, source):
    """Catch a non-finite force where it is PRODUCED. Without this it is laundered into the positions
    on the next drift and reported as an integrator divergence, which is the wrong half of the loop.

    ``max|F| < fThreshold`` cannot serve as the guard: the comparison is False for a NaN, so a
    completely broken force reads as "not yet converged" and the loop carries on using it."""
    if np.all(np.isfinite(force)) and np.isfinite(energy):
        return
    bad = int((~np.isfinite(np.asarray(force).reshape(-1, 2)).all(axis = 1)).sum())
    # The FAILING force, saved under its own name: `packing.force` still holds the last GOOD one, so a
    # dump without this shows a healthy force beside a broken energy and reads as no fault at all.
    dump = _dumpState(packing, "force", failingForce = np.asarray(force, dtype = float))
    raise FloatingPointError(
        f"{source} returned a non-finite result at step {step}: energy {energy}, {bad} vertices with "
        f"a non-finite force.\n"
        f"    The GEOMETRY going in was finite -- that is checked separately -- so this is the energy "
        f"tier itself, at this configuration. Nothing downstream is meaningful.\n"
        f"    State written to {dump}\n"
        f"    Reload and dissect it with: python tests/inspectFailureDump.py <path>")

# UNVERIFIED(Cam)
# The smallest residual a force evaluation can resolve. Below this, max|F| is measuring roundoff:
# repeated evaluations of the SAME configuration disagree at this level, and the CUDA path is not
# bit-reproducible (~3e-12 of drift over 120 steps), so no minimizer can drive a residual under it and
# none can tell that it has arrived. A tolerance below it is unreachable BY CONSTRUCTION, which is
# worth saying before the first step rather than after the last.
NOISE_FLOOR = 3e-12


# UNVERIFIED(Cam)
def checkReachable(threshold, name = "the tolerance"):
    """Warn when a requested tolerance is at or under the force noise floor. Returns the threshold.

    Cheap, and it catches the most common way a run burns its whole budget: asking for something the
    arithmetic cannot represent. There is no adaptive stopping rule that rescues this -- the residual
    is not converging slowly, it is already as low as it goes and jittering."""
    if threshold <= NOISE_FLOOR:
        warnings.warn(
            f"\n*** {name} {threshold:.2e} is at or below the force noise floor {NOISE_FLOOR:.1e} "
            f"***\n"
            f"    Repeated evaluations of one configuration disagree at that level, so this target "
            f"can never be met and the run will use its entire step budget. Ask for {10 * NOISE_FLOOR:.0e} "
            f"or more, or treat the run as converged when max|F| stops falling.", stacklevel = 3)
    elif threshold <= 10.0 * NOISE_FLOOR:
        warnings.warn(
            f"\n*** {name} {threshold:.2e} is within a decade of the noise floor "
            f"{NOISE_FLOOR:.1e} ***\n"
            f"    Reachable, but the last decade is mostly roundoff and will cost far more steps than "
            f"the ones before it.", stacklevel = 3)
    return threshold


# UNVERIFIED(Cam)
class _Stall:
    """Stop a minimizer once its residual stops falling fast enough to be worth the steps.

    THE TEST IS ON THE FORCE, NOT THE ENERGY, and the difference matters. Near a minimum the energy
    moves as the SQUARE of the residual, so an energy-stall test goes quiet a whole decade of |F|
    before the run is actually finished -- it loses its signal exactly where the discrimination is
    needed. Relative energy is worse still on the contact tiers, where ``E ~ d^3`` makes ``dE/E`` blow
    up at shallow contact. ``max|F|`` is already computed every step, and it is the number the caller
    asked for.

    BEST-SO-FAR, NOT CURRENT. FIRE is damped dynamics and its residual is not monotone -- it
    overshoots, resets its velocity, and climbs before falling again. A test on the current value
    would fire on an ordinary overshoot. The window minimum is what is actually improving.

    The criterion mirrors the density controller's ``_STUCK_DENSITY_DROP`` in ``anneal.py``: a fixed
    budget must BUY a fixed factor. ``patience`` steps must divide the residual by ``factor``, or the
    run has stopped paying for itself.

    ``reason`` afterwards is a short slug and ``describe()`` a sentence naming which wall was hit,
    which is the part worth having -- stopping early only saves time, while saying whether the floor
    belongs to the arithmetic, the energy tier, or the step budget tells the caller what to change."""

    def __init__(self, patience, factor, threshold, maxSteps = None):
        self.patience = None if patience is None else int(patience)
        self.factor = float(factor)
        self.threshold = float(threshold)
        self.maxSteps = None if maxSteps is None else int(maxSteps)
        self.best = np.inf
        self.mark = np.inf
        self.markStep = 0
        self.rate = 0.0
        self.projected = np.inf
        self.reason = None
        self.exhausted = 0

    def update(self, step, residual, exhausted = 0):
        """Record one step's residual; True when the run should stop.

        ``exhausted`` is the running count of line searches that used every one of their bisections
        without meeting the Wolfe conditions. That is a SHARPER signal than the residual window and
        deserves its own verdict: it says the energy differences along the search direction are below
        roundoff, so the conditions are being tested against noise. Measured on Cam's own run, an
        L-BFGS asked for 1e-12 -- under the noise floor -- ground out 41 evaluations per step, which is
        the 40-bisection cap plus the one bracketing trial, on every single step."""
        if residual < self.best:
            self.best = float(residual)
        if self.patience is None or step < self.markStep + self.patience:
            return False
        # A search that exhausts its bisections on MOST steps of a window is not making progress in a
        # way any budget fixes -- caught before the residual test, since it names the cause.
        exhaustedHere = int(exhausted) - self.exhausted
        self.exhausted = int(exhausted)
        if exhaustedHere > 0.5 * self.patience:
            self.reason = "search"
            return True
        # Decades bought by this window, and the per-step rate they imply.
        bought = np.log10(self.mark / self.best) if np.isfinite(self.mark) and self.best > 0.0 else np.inf
        self.rate = float(bought) / self.patience
        if np.isfinite(bought) and self.best > self.mark / self.factor:
            if self.best <= 10.0 * NOISE_FLOOR:
                self.reason = "noise"
                return True
            if bought <= 0.0:
                self.reason = "flat"
                return True
            # STILL CONVERGING, so the question is not "is this fast?" but "does the BUDGET GET
            # THERE?" -- which is what makes a step lucrative or not. Measured on the notebook's own
            # cell: FIRE converging at 0.35 decades per 1000 steps needed 7,029 more steps out of a
            # 10,000 budget, i.e. it was going to make it, and a fixed-rate test cut it off anyway.
            # A rate threshold cannot know that; the remaining budget can.
            self.projected = np.log10(self.best / self.threshold) / self.rate
            if self.maxSteps is not None and self.projected <= self.maxSteps - step:
                self.mark = self.best
                self.markStep = step
                return False
            self.reason = "slow"
            return True
        self.mark = self.best
        self.markStep = step
        return False

    def describe(self, name):
        """One sentence naming the wall, with the number that makes it actionable."""
        if self.reason == "search":
            # WHY it cannot make progress depends on where it stopped, and conflating the two would
            # give the wrong advice. Near the noise floor the arithmetic is the limit; far above it,
            # the arithmetic is fine and the LANDSCAPE is the limit -- a constrained, kinky objective
            # whose directional energy differences have gone under roundoff long before max|F| has.
            # Measured on the notebook: a search stall at 4.9e-10, which is 135x the floor.
            common = (f"{name} stopped at max|F| = {self.best:.2e}: its line search exhausted every "
                      f"bisection on more than half of the last {self.patience} steps, so each step "
                      f"cost the full ~40 evaluations and bought nothing.")
            if self.best <= 100.0 * NOISE_FLOOR:
                return (f"{common} That is at the force noise floor ({NOISE_FLOOR:.1e}), so the "
                        f"packing is as converged as this arithmetic allows and the requested "
                        f"{self.threshold:.1e} was never reachable. Raise the tolerance.")
            return (f"{common} It is {self.best / NOISE_FLOOR:,.0f}x the noise floor "
                    f"({NOISE_FLOOR:.1e}), so this is NOT an arithmetic limit -- the energy "
                    f"differences along the search direction have gone under roundoff while the "
                    f"residual has not, which is what a stiff constrained landscape does. More steps "
                    f"will not help and neither will a smaller tolerance; {self.best:.0e} is what "
                    f"this configuration and constraint set support. Set the tolerance there, or "
                    f"change the tier or constraints if you need lower.")
        if self.reason == "noise":
            return (f"{name} stopped at max|F| = {self.best:.2e}, which is the force noise floor "
                    f"({NOISE_FLOOR:.1e}) -- this is converged, and the requested "
                    f"{self.threshold:.1e} was below what the arithmetic can resolve.")
        if self.reason == "flat":
            return (f"{name} stopped at max|F| = {self.best:.2e}: {self.patience} steps bought NO "
                    f"reduction at all, and the residual is far above the noise floor "
                    f"({NOISE_FLOOR:.1e}). That is a floor of the ENERGY, not of the minimizer -- a "
                    f"C1 tier has a kink the descent cannot pass, so raise the tolerance or move to a "
                    f"smoother tier rather than spending more steps.")
        return (f"{name} stopped at max|F| = {self.best:.2e}, converging at "
                f"{1000.0 * self.rate:.2f} decades per 1000 steps. Reaching the requested "
                f"{self.threshold:.1e} needs roughly {self.projected:,.0f} more steps, which this "
                f"run's remaining budget does not cover; raise maxSteps if that is worth it, or "
                f"relax the tolerance.")


def _stopStalled(stall, name, bar, packing, energy, f, step):
    """Close out a stalled run: warn with the diagnosis, store state, return the usual tuple.

    ONCE PER REASON PER PACKING. The diagnosis carries live numbers -- the residual, the rate, the
    projected step count -- so it can never de-duplicate on its own, and the callers that trip it are
    typically loops: ``holdExcessEnergy`` relaxes on every density step, so a stall that persists
    would print a slightly different paragraph per round and bury the one that mattered. The first
    of each KIND is the informative one; after that the slug on the packing is the record."""
    if getattr(packing, "_warnedStall", None) != stall.reason:
        packing._warnedStall = stall.reason
        warnings.warn("\n*** " + stall.describe(name) + "\n", stacklevel = 3)
    if bar is not None:
        bar.close()
    packing.force[:] = f
    packing.energy = energy
    packing.stopReason = stall.reason
    return energy, step, False


def _progressBar(total, desc, progress, leave = True):
    """A tqdm progress bar over a minimizer's step loop, or None when ``progress`` is off or tqdm is
    unavailable. Every caller guards its use with ``if bar is not None``. ``leave = False`` for a
    transient inner bar (e.g. Newton's per-step Hessian).

    ``progress`` is True/False, or ``"text"`` to force the PLAIN tqdm instead of the notebook widget.

    WHY THE TEXT OPTION EXISTS. ``tqdm.auto`` picks the ipywidgets bar inside a notebook, and a new
    widget only renders once its model has round-tripped to the frontend over the comm channel -- which
    the kernel services when it returns to its message loop, not during a cell. So the FIRST widget of
    a session, created immediately before a long compute, sits at "Loading widget..." until that cell
    finishes: a 1000-step FIRE run at n = 32 is ~113 s of numpy with no yield in it. The text bar is
    ordinary stream output and needs no handshake, so it always draws."""
    if not progress:
        return None
    try:
        if str(progress).lower() == "text":
            from tqdm import tqdm
        else:
            from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total = total, desc = desc, leave = leave)

def minimizeFIRE(
    packing,
    forceEnergy,
    maxSteps = 100000,
    fThreshold = 1e-10,
    dt = 0.01,
    dtMax = 0.1,
    alphaStart = 0.1,
    fInc = 1.1,
    fDec = 0.5,
    fAlpha = 0.99,
    nMin = 5,
    constraints = None,
    shakeTol = 1e-14,
    shakeMaxIter = 20,
    transient = None,
    callback = None,
    callbackEvery = 200,
    patience = None,
    stallFactor = 2.0,
    progress = True,
):
    """Relax packing.positions with FIRE. Returns (energy, steps, converged).

    ``converged`` is True when the max per-vertex force drops below fThreshold.
    On return the final force/energy are also stored on the packing. If ``callback`` is given it is
    called ``callback(step, energy, force)`` every ``callbackEvery`` steps with the packing's
    current positions/force/energy stored -- the hook the Model uses to checkpoint mid-run.

    With ``constraints`` (a ``constraints.ShapeConstraints``) this runs CONSTRAINED FIRE: the shape is
    held rigid by projection rather than by stiff springs. Each step projects the force and velocity
    onto the constraint tangent space and SHAKE-retracts the positions back onto the manifold
    (velocity projected again after the retraction, i.e. RATTLE). Removing the stiff spring modes from
    the dynamics lifts the timestep ceiling and drops the condition number, which is the whole point.
    Pass the overlap force ALONE in that case -- the springs are what the constraints replace.
    """
    v = packing.velocities
    v[:] = 0.0
    alpha = alphaStart
    nPos = 0
    _retract(constraints, packing, shakeTol, shakeMaxIter)
    energy, f = forceEnergy(packing)
    f = _projectForce(constraints, packing, f)

    stall = _Stall(patience, stallFactor, fThreshold, maxSteps)
    bar = _progressBar(maxSteps, "FIRE" if constraints is None else "FIRE (constrained)", progress)
    for step in range(maxSteps):
        mf = maxForceMagnitude(f)
        if mf < fThreshold:
            if bar is not None:
                bar.close()
            packing.force[:] = f
            packing.energy = energy
            return energy, step, True
        if stall.update(step, mf):
            return _stopStalled(stall, "FIRE", bar, packing, energy, f, step)

        # velocity Verlet: half-kick, drift, recompute force, half-kick
        v += 0.5 * dt * f
        packing.positions += dt * v
        wrapPolygonsIntoCell(packing)
        # DIVERGENCE CAUGHT HERE, BEFORE THE TRANSIENT STEP AND THE RETRACTION. Both of those consume
        # the positions, so a NaN reaching them surfaces as a constraint Jacobian that is not finite
        # for EVERY polygon at once -- which reads as a constraint fault and is not one. Splitting the
        # test either side of the transient call is what separates "the integrator diverged" from "the
        # target step diverged"; they present identically downstream and have opposite fixes.
        _checkFinite(packing, step, dt, mf, "the FIRE step")
        # TRANSIENT target DOF step alongside the positions -- the double optimization is a joint
        # minimization, not an alternating one, so the targets move on the same timestep.
        if transient is not None:
            transient(packing, dt)
            _checkFinite(packing, step, dt, mf, "the TRANSIENT target step")
        _retract(constraints, packing, shakeTol, shakeMaxIter)
        _checkFinite(packing, step, dt, mf, "the constraint RETRACTION")
        energy, f = forceEnergy(packing)
        _checkForce(packing, energy, f, step, "the force evaluation")
        # Force and velocity are projected at the SAME configuration, so they share one basis.
        basis = _normalBasis(constraints, packing)
        f = _projectForce(constraints, packing, f, basis)
        v += 0.5 * dt * f
        v[:] = _projectForce(constraints, packing, v, basis)

        # FIRE: mix velocity toward the force direction, then adapt dt / alpha
        power = float(np.dot(v, f))
        fnorm = float(np.linalg.norm(f))
        vnorm = float(np.linalg.norm(v))
        if fnorm > 1e-300:
            v[:] = (1.0 - alpha) * v + alpha * (vnorm / fnorm) * f
        if power > 0.0:
            nPos += 1
            if nPos > nMin:
                dt = min(dt * fInc, dtMax)
                alpha *= fAlpha
        else:
            nPos = 0
            dt *= fDec
            alpha = alphaStart
            v[:] = 0.0

        if callback is not None and step % callbackEvery == 0:
            packing.force[:] = f
            packing.energy = energy
            callback(step, energy, f)

        if bar is not None:
            bar.set_postfix(maxF = f"{mf:.2e}", refresh = False)
            bar.update(1)

    if bar is not None:
        bar.close()
    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, False

def minimizeGD(
    packing,
    forceEnergy,
    maxSteps = 100000,
    fThreshold = 1e-10,
    step = 0.001,
    callback = None,
    callbackEvery = 200,
    progress = True,
):
    """Relax packing.positions with fixed-step gradient descent.

    Moves along the force (= -dE/dr) by ``step`` each iteration. Returns
    (energy, steps, converged); final force/energy are stored on the packing. ``callback`` /
    ``callbackEvery`` work as in ``minimizeFIRE`` (the checkpoint hook).
    """
    energy, f = forceEnergy(packing)
    f = applyPins(packing, np.array(f, dtype = float))
    bar = _progressBar(maxSteps, "GD", progress)
    for i in range(maxSteps):
        mf = maxForceMagnitude(f)
        if mf < fThreshold:
            if bar is not None:
                bar.close()
            packing.force[:] = f
            packing.energy = energy
            return energy, i, True
        packing.positions += step * f
        wrapPolygonsIntoCell(packing)
        energy, f = forceEnergy(packing)
        f = applyPins(packing, np.array(f, dtype = float))
        if callback is not None and i % callbackEvery == 0:
            packing.force[:] = f
            packing.energy = energy
            callback(i, energy, f)
        if bar is not None:
            bar.set_postfix(maxF = f"{mf:.2e}", refresh = False)
            bar.update(1)
    if bar is not None:
        bar.close()
    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, False

def minimizeCG(
    packing,
    forceEnergy,
    maxSteps = 50000,
    fThreshold = 1e-12,
    c1 = 1e-4,
    c2 = 0.1,
    maxZoom = 40,
    constraints = None,
    shakeTol = 1e-14,
    shakeMaxIter = 20,
    callback = None,
    callbackEvery = 200,
    patience = None,
    stallFactor = 2.0,
    progress = True,
):
    """Relax packing.positions with Polak-Ribiere+ nonlinear conjugate gradient and a STRONG-WOLFE
    line search (Nocedal & Wright, Alg. 3.5/3.6). ``forceEnergy(packing) -> (energy, force = -dE/dr
    flat)``. Each step line-searches along the conjugate direction ``d`` for a step satisfying
    sufficient decrease (c1) and the curvature condition (c2), then updates ``d = f + beta d`` with
    ``beta`` = Polak-Ribiere+ and a steepest-descent restart every 2N steps / on a non-descent
    direction. The Wolfe search (guaranteeing energy decrease, not just a stationary directional
    gradient) is what lets CG polish a stiff, kinky objective. Returns (energy, steps, converged);
    final force/energy are stored on the packing. Positions are wrapped per-polygon after each step.

    With ``constraints`` (a ``constraints.ShapeConstraints``) this becomes RIEMANNIAN CG on the
    constraint manifold: the search direction is projected onto the tangent space and every trial
    point of the line search is SHAKE-retracted back onto the manifold before its energy/force are
    read. This is the second-order-ish polish to reach for once constrained FIRE stalls -- it needs no
    Hessian, so none of the FD-Hessian cost or the singular-null-space trouble of Newton applies."""
    x = packing.positions
    n = x.size
    _retract(constraints, packing, shakeTol, shakeMaxIter)
    energy, f = forceEnergy(packing)
    f = _projectForce(constraints, packing, f)
    d = f.copy()
    kRestart = 0
    stash = {"f": f}

    def phiDphi(x0, a):
        x[:] = x0 + a * d
        _retract(constraints, packing, shakeTol, shakeMaxIter)
        e, force = forceEnergy(packing)
        force = _projectForce(constraints, packing, force)
        stash["f"] = force
        return e, -float(force @ d)          # phi(a), phi'(a) = grad . d = -force . d

    def zoom(x0, aLo, phiLo, aHi, phi0, dphi0):
        aMid = 0.5 * (aLo + aHi)
        for _ in range(maxZoom):
            aMid = 0.5 * (aLo + aHi)
            phiA, dphiA = phiDphi(x0, aMid)
            if phiA > phi0 + c1 * aMid * dphi0 or phiA >= phiLo:
                aHi = aMid
            else:
                if abs(dphiA) <= -c2 * dphi0:
                    return aMid, phiA
                if dphiA * (aHi - aLo) >= 0.0:
                    aHi = aLo
                aLo, phiLo = aMid, phiA
        # RAN OUT OF BISECTIONS. The Wolfe conditions were never met, which on a converged packing
        # means the energy differences along the line are pure roundoff and the tests are reading
        # noise. Recorded because it is a far sharper signal than the residual window: it says the
        # SEARCH cannot make progress, not merely that this window was unproductive.
        stash["exhausted"] = stash.get("exhausted", 0) + 1
        return aMid, phiLo

    alphaGuess = 1.0
    stall = _Stall(patience, stallFactor, fThreshold, maxSteps)
    bar = _progressBar(maxSteps, "CG" if constraints is None else "CG (constrained)", progress)
    for step in range(maxSteps):
        mf = maxForceMagnitude(f)
        if mf < fThreshold:
            if bar is not None:
                bar.close()
            packing.force[:] = f
            packing.energy = energy
            return energy, step, True
        if stall.update(step, mf, stash.get("exhausted", 0)):
            return _stopStalled(stall, "CG", bar, packing, energy, f, step)
        x0 = x.copy()
        phi0 = energy
        fStart = f.copy()
        dphi0 = -float(fStart @ d)
        if dphi0 >= 0.0:                          # not a descent direction -> restart
            d = fStart.copy()
            dphi0 = -float(fStart @ d)
            kRestart = 0
        aPrev, phiPrev = 0.0, phi0
        a = min(1.0, alphaGuess)
        alpha, phiAlpha = None, None
        for it in range(60):
            phiA, dphiA = phiDphi(x0, a)
            if phiA > phi0 + c1 * a * dphi0 or (it > 0 and phiA >= phiPrev):
                alpha, phiAlpha = zoom(x0, aPrev, phiPrev, a, phi0, dphi0)
                break
            if abs(dphiA) <= -c2 * dphi0:
                alpha, phiAlpha = a, phiA
                break
            if dphiA >= 0.0:
                alpha, phiAlpha = zoom(x0, a, phiA, aPrev, phi0, dphi0)
                break
            aPrev, phiPrev = a, phiA
            a = 2.0 * a
        if alpha is None:
            alpha = a
        energy, _ = phiDphi(x0, alpha)           # leave state at the accepted step
        f = stash["f"]
        wrapPolygonsIntoCell(packing)
        alphaGuess = alpha
        beta = max(0.0, float(f @ (f - fStart)) / float(fStart @ fStart))   # Polak-Ribiere+
        # Reprojecting the combined direction is the (cheap) vector transport: ``d`` was tangent at
        # the PREVIOUS point, and the manifold has curved underneath it since.
        d = _projectForce(constraints, packing, f + beta * d)
        kRestart += 1
        if kRestart >= 2 * n or float(f @ d) <= 0.0:
            d = f.copy()
            kRestart = 0
        if callback is not None and step % callbackEvery == 0:
            packing.force[:] = f
            packing.energy = energy
            callback(step, energy, f)
        if bar is not None:
            bar.set_postfix(maxF = f"{mf:.2e}", refresh = False)
            bar.update(1)
    if bar is not None:
        bar.close()
    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, False


# UNVERIFIED(Cam)
def minimizeLBFGS(
    packing,
    forceEnergy,
    maxSteps = 50000,
    fThreshold = 1e-12,
    memory = 10,
    c1 = 1e-4,
    c2 = 0.9,
    maxZoom = 40,
    constraints = None,
    shakeTol = 1e-14,
    shakeMaxIter = 20,
    callback = None,
    callbackEvery = 200,
    patience = None,
    stallFactor = 2.0,
    progress = True,
):
    """Relax packing.positions with limited-memory BFGS (Nocedal & Wright, Alg. 7.4/7.5).

    Same interface and same strong-Wolfe machinery as ``minimizeCG``; what differs is the direction and
    what that does to the COST PER STEP. CG rebuilds its direction from one scalar (Polak-Ribiere beta)
    and needs a tight curvature condition, ``c2 = 0.1``, to keep the direction meaningful, so its line
    search rarely accepts the first trial. L-BFGS carries an approximate inverse Hessian in the last
    ``memory`` (s, y) pairs, which makes the unit step the RIGHT step near a minimum and admits the
    loose quasi-Newton ``c2 = 0.9``. The line search then usually accepts alpha = 1 outright, at one
    force evaluation per step against CG's several. On a stiff contact objective this is the whole
    difference, and it is not a small one: the contact note measures L-BFGS reaching 9.8e-11 in 41
    iterations where FIRE stalled at 5.5e-6 after 150 on the identical gradient.

    ``memory`` pairs cost ``2 * memory`` vectors of storage and ``4 * memory`` dot products per step,
    both negligible beside one force evaluation. The curvature guard ``s . y > 0`` is what keeps the
    approximation positive definite; a pair that fails it is DISCARDED rather than damped, which is the
    conservative choice for an objective whose Hessian is genuinely indefinite between contact events.

    With ``constraints`` this is Riemannian L-BFGS: directions and gradients are projected onto the
    tangent space and every trial point is SHAKE-retracted, exactly as in ``minimizeCG``. The (s, y)
    pairs are then only approximately transported between tangent spaces, which is why the memory is
    cleared on any non-descent direction rather than trusted through a restart.

    Returns (energy, steps, converged); final force/energy are stored on the packing."""
    x = packing.positions
    _retract(constraints, packing, shakeTol, shakeMaxIter)
    energy, f = forceEnergy(packing)
    f = _projectForce(constraints, packing, f)
    sHistory, yHistory, rhoHistory = [], [], []
    stash = {"f": f, "a": None, "e": energy}
    evaluations = [1]

    def phiDphi(x0, a):
        x[:] = x0 + a * d
        _retract(constraints, packing, shakeTol, shakeMaxIter)
        e, force = forceEnergy(packing)
        evaluations[0] += 1
        force = _projectForce(constraints, packing, force)
        stash["f"], stash["a"], stash["e"] = force, a, e
        return e, -float(force @ d)

    def zoom(x0, aLo, phiLo, aHi, phi0, dphi0):
        aMid = 0.5 * (aLo + aHi)
        for _ in range(maxZoom):
            aMid = 0.5 * (aLo + aHi)
            phiA, dphiA = phiDphi(x0, aMid)
            if phiA > phi0 + c1 * aMid * dphi0 or phiA >= phiLo:
                aHi = aMid
            else:
                if abs(dphiA) <= -c2 * dphi0:
                    return aMid, phiA
                if dphiA * (aHi - aLo) >= 0.0:
                    aHi = aLo
                aLo, phiLo = aMid, phiA
        # RAN OUT OF BISECTIONS. The Wolfe conditions were never met, which on a converged packing
        # means the energy differences along the line are pure roundoff and the tests are reading
        # noise. Recorded because it is a far sharper signal than the residual window: it says the
        # SEARCH cannot make progress, not merely that this window was unproductive.
        stash["exhausted"] = stash.get("exhausted", 0) + 1
        return aMid, phiLo

    def direction(force):
        """Two-loop recursion for ``d = H f``, the quasi-Newton descent direction. ``f`` is the FORCE,
        so the gradient is ``-f`` and the usual ``-H g`` is ``+H f``; the recursion below runs on
        ``q = -f`` and negates at the end, which keeps every sign matching the textbook."""
        q = -force
        alphas = []
        for s, y, rho in zip(reversed(sHistory), reversed(yHistory), reversed(rhoHistory)):
            a = rho * float(s @ q)
            alphas.append(a)
            q = q - a * y
        if sHistory:
            yLast = yHistory[-1]
            q = q * (float(sHistory[-1] @ yLast) / float(yLast @ yLast))
        for s, y, rho, a in zip(sHistory, yHistory, rhoHistory, reversed(alphas)):
            b = rho * float(y @ q)
            q = q + s * (a - b)
        return -q

    d = _projectForce(constraints, packing, direction(f))
    stall = _Stall(patience, stallFactor, fThreshold, maxSteps)
    bar = _progressBar(maxSteps, "L-BFGS" if constraints is None else "L-BFGS (constrained)", progress)
    for step in range(maxSteps):
        mf = maxForceMagnitude(f)
        if mf < fThreshold:
            if bar is not None:
                bar.close()
            packing.force[:] = f
            packing.energy = energy
            return energy, step, True
        if stall.update(step, mf, stash.get("exhausted", 0)):
            return _stopStalled(stall, "L-BFGS", bar, packing, energy, f, step)
        x0 = x.copy()
        phi0 = energy
        fStart = f.copy()
        dphi0 = -float(fStart @ d)
        if dphi0 >= 0.0:
            sHistory.clear(); yHistory.clear(); rhoHistory.clear()
            d = fStart.copy()
            dphi0 = -float(fStart @ d)
        # THE UNIT STEP IS THE POINT. L-BFGS scales its own direction, so alpha = 1 is the Newton-like
        # step and is accepted outright once the memory is warm; unlike CG there is nothing to gain
        # from carrying the previous alpha forward, and doing so would throw away that scaling.
        aPrev, phiPrev = 0.0, phi0
        a = 1.0
        alpha, phiAlpha = None, None
        for it in range(60):
            phiA, dphiA = phiDphi(x0, a)
            if phiA > phi0 + c1 * a * dphi0 or (it > 0 and phiA >= phiPrev):
                alpha, phiAlpha = zoom(x0, aPrev, phiPrev, a, phi0, dphi0)
                break
            if abs(dphiA) <= -c2 * dphi0:
                alpha, phiAlpha = a, phiA
                break
            if dphiA >= 0.0:
                alpha, phiAlpha = zoom(x0, a, phiA, aPrev, phi0, dphi0)
                break
            aPrev, phiPrev = a, phiA
            a = 2.0 * a
        if alpha is None:
            alpha = a
        # The accepted step is USUALLY the last point the search evaluated -- with a warm memory the
        # very first trial, alpha = 1, satisfies both Wolfe conditions -- and re-evaluating it would
        # double the cost of the whole minimizer. Only move when the accepted alpha is somewhere else.
        if stash["a"] != alpha:
            energy, _ = phiDphi(x0, alpha)
        else:
            energy = stash["e"]
        f = stash["f"]
        # s is read BEFORE wrapping. A periodic wrap translates a polygon by a box length, and that
        # jump would enter the curvature pair as a displacement that never happened.
        s = x - x0
        y = fStart - f
        curvature = float(s @ y)
        wrapPolygonsIntoCell(packing)
        if curvature > 1e-16 * float(np.sqrt((s @ s) * (y @ y))):
            sHistory.append(s.copy())
            yHistory.append(y.copy())
            rhoHistory.append(1.0 / curvature)
            if len(sHistory) > memory:
                sHistory.pop(0); yHistory.pop(0); rhoHistory.pop(0)
        d = _projectForce(constraints, packing, direction(f))
        if callback is not None and step % callbackEvery == 0:
            packing.force[:] = f
            packing.energy = energy
            callback(step, energy, f)
        if bar is not None:
            bar.set_postfix(maxF = f"{mf:.2e}", evalsPerStep = f"{evaluations[0] / (step + 1):.1f}",
                            refresh = False)
            bar.update(1)
    if bar is not None:
        bar.close()
    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, False


# UNVERIFIED(Cam)
def minimizeNewton(packing, forceEnergy, maxSteps = 40, fThreshold = 1e-12,
                   hStep = 1e-6, constraints = None, shakeTol = 1e-14, shakeMaxIter = 20,
                   callback = None, callbackEvery = 1, progress = True,
                   hessian = None):
    """Newton's method for the smooth (mollified) energy: each step builds the Hessian
    ``H = d(-force)/dx`` by central finite differences of ``forceEnergy``'s force, solves
    ``H dx = force``, and takes it with a FORCE-based backtracking line search (accept if
    ``max|force|`` decreases), falling back to steepest descent when H gives a non-descent
    direction. Pass the EXACT force so the Hessian is accurate enough to reach the machine
    floor. Returns (energy, steps, converged); final force/energy are stored on the packing.

    Cost: the FD Hessian is ``2*(2N)`` force evaluations per step, so this is practical only
    with a fast force -- a small system, or once the exact tier is vectorized.

    With ``constraints`` this becomes PROJECTED Newton: the FD Hessian is built from the projected
    (tangential) force at SHAKE-retracted sample points, so it is the Hessian of the reduced problem
    embedded in the full space. Its null space simply grows by the constraint normals, which the
    rank-truncating solve below already discards; the step is then projected and the new point
    retracted. Given the cost, prefer constrained ``minimizeCG`` -- it needs no Hessian at all."""
    x = packing.positions
    n = x.size

    def evalForce(pk):
        """(energy, force) with the force projected to the tangent space when constrained -- the
        true residual on the manifold, and the right thing to differentiate for the reduced Hessian."""
        e, force = forceEnergy(pk)
        return e, _projectForce(constraints, pk, force)

    _retract(constraints, packing, shakeTol, shakeMaxIter)
    for step in range(maxSteps):
        energy, f = evalForce(packing)
        mf = maxForceMagnitude(f)
        if mf < fThreshold:
            packing.force[:] = f
            packing.energy = energy
            return energy, step, True
        if hessian is not None:
            H = hessian(packing)
        else:
            H = np.zeros((n, n))
            x0 = x.copy()
            hbar = _progressBar(n, f"Newton step {step} Hessian (max|F|={mf:.2e})", progress, leave = False)
            for j in range(n):
                x[j] = x0[j] + hStep
                _retract(constraints, packing, shakeTol, shakeMaxIter); _, fp = evalForce(packing)
                x[:] = x0; x[j] = x0[j] - hStep
                _retract(constraints, packing, shakeTol, shakeMaxIter); _, fm = evalForce(packing)
                x[:] = x0
                # H = d(grad E)/dx = -d(force)/dx
                H[:, j] = -(fp - fm) / (2.0 * hStep)
                if hbar is not None:
                    hbar.update(1)
            if hbar is not None:
                hbar.close()
        H = 0.5 * (H + H.T)
        g = -f
        # H is singular: the energy is translation-invariant, so global x/y shift are exact zero modes.
        # np.linalg.solve amplifies the force's tiny null-space (FD) noise into a huge rigid-translation
        # step the line search can't reduce. Use an SVD least-squares solve that DROPS the near-null
        # singular values (rcond), giving the minimum-norm physical step; then project the residual
        # translation gauge out for good measure.
        try:
            dx = np.linalg.lstsq(H, -g, rcond = 1e-6)[0]
        except np.linalg.LinAlgError:
            dx = -g
        dxr = dx.reshape(-1, 2)
        dxr -= dxr.mean(0)
        dx = _projectForce(constraints, packing, dxr.ravel())
        if float(g @ dx) > 0.0:                            # not a descent direction
            dx = -g
        f0 = maxForceMagnitude(f)
        x0 = x.copy(); alpha = 1.0; fTrial = f
        for _ in range(40):
            x[:] = x0 + alpha * dx
            wrapPolygonsIntoCell(packing)
            _retract(constraints, packing, shakeTol, shakeMaxIter)
            _, fTrial = evalForce(packing)
            if maxForceMagnitude(fTrial) < f0:
                break
            alpha *= 0.5
        if callback is not None and step % callbackEvery == 0:
            packing.force[:] = fTrial
            packing.energy = energy
            callback(step, energy, fTrial)
    energy, f = evalForce(packing)
    packing.force[:] = f
    packing.energy = energy
    return energy, maxSteps, maxForceMagnitude(f) < fThreshold