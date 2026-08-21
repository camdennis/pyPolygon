"""Monte Carlo / basin-hopping search over packing arrangements.

WHY THIS EXISTS. `energySweep` is a QUENCH: it relaxes to the nearest local minimum and reports the
density reached. Measured, that is not a sampling problem -- three unrelated seeds returned 0.699489,
0.699748, 0.699560, a spread of 2.6e-04 -- so restarts do not help. And it is not a MODEL problem
either: `tests/knownOptimumCheck.py` shows the known optimum is a valid, critical, ATTRACTING minimum
of this very energy, held to nine digits under perturbations up to 1e-3. The landscape rewards the
optimum; descent simply cannot reach it.

What blocks it is barriers. The optimal 5-square packing has its middle square at 45 degrees, and no
gradient will ever rotate a square 45 degrees against its neighbors -- that is a barrier, and descent
cannot cross barriers by construction. Nor does the contact law help: a corner approaching a gap feels
a force vanishing as delta^3 while the overlap it must resolve falls only as delta^2 (measured slopes
2.966 and 1.978).

So the missing ingredient is a MOVE SET, not a better energy.

THREE MOVES, and the first two are free of constraint bookkeeping:

  rotate     spin one polygon about its own centroid
  translate  displace one polygon
  swap       exchange the SIZES of two polygons

Rigid motions preserve every area and edge length exactly, so `rotate` and `translate` land on the
constraint manifold with no retraction at all. `swap` rescales the geometry AND the targets by the same
factor, so it too stays on the manifold by construction rather than by projection.

SWAP IS THE INTERESTING ONE. It is the standard trick for exactly this failure -- swap Monte Carlo
equilibrates polydisperse systems that ordinary dynamics cannot, because exchanging two particles'
sizes is an enormous move in configuration space achieved with a purely local operation. This project
already carries per-polygon sizes in `targetArea`, so the move costs nothing structurally. It is
inert on a monodisperse packing, which is worth remembering when reading a result.

WHAT IS BEING MINIMIZED. At FIXED density: the total violation area (polygon-polygon overlap plus
anything outside the container). Zero means a valid packing exists at that density and this search
found it. That makes the question a decision problem with an unambiguous answer, rather than "how
dense did we get" -- and for the 5-square case a zero-overlap arrangement is KNOWN to exist at
phi = 0.68227, so a failure here is a failure of the search and nothing else.
"""

# UNVERIFIED(Cam)

import numpy as np


_DEFAULT_ROTATION = 0.35          # radians, the scale of a trial spin
_DEFAULT_TRANSLATION = 0.06       # fraction of the mean covering radius


class SearchResult:
    """What a search run found, and how it got there."""

    def __init__(self):
        self.objective = float("inf")
        self.solved = False
        self.rounds = 0
        self.accepted = {"rotate": 0, "translate": 0, "swap": 0}
        self.proposed = {"rotate": 0, "translate": 0, "swap": 0}
        self.history = []

    def rate(self, move):
        """Acceptance rate for one move type -- the number to tune amplitudes against."""
        proposed = self.proposed.get(move, 0)
        return (self.accepted.get(move, 0) / proposed) if proposed else 0.0

    def __repr__(self):
        rates = "  ".join(f"{m} {self.accepted[m]}/{self.proposed[m]}" for m in self.proposed)
        return (f"<SearchResult {'SOLVED' if self.solved else 'unsolved'} "
                f"objective {self.objective:.3e} after {self.rounds} rounds; {rates}>")


# UNVERIFIED(Cam)
def _freeCount(model):
    container = getattr(model.packing, "containerIndex", None)
    return model.getNumPolygons() if container is None else int(container)


# UNVERIFIED(Cam)
def _slice(packing, polygon):
    return int(packing.startIndices[polygon]), int(packing.startIndices[polygon + 1])


# UNVERIFIED(Cam)
def rotatePolygon(packing, polygon, angle):
    """Spin one polygon about its own centroid. Exactly constraint-preserving: a rigid motion changes
    no edge length and no area, so nothing needs retracting afterwards."""
    a, b = _slice(packing, polygon)
    r = packing.positions.reshape(-1, 2)
    centroid = r[a:b].mean(axis = 0)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    r[a:b] = centroid + (r[a:b] - centroid) @ rotation.T


# UNVERIFIED(Cam)
def translatePolygon(packing, polygon, delta):
    """Displace one polygon. Also exactly constraint-preserving."""
    a, b = _slice(packing, polygon)
    packing.positions.reshape(-1, 2)[a:b] += delta


# UNVERIFIED(Cam)
def swapSizes(packing, first, second):
    """Exchange the SIZES of two polygons, geometry and targets together.

    Each polygon is rescaled about its own centroid by ``sqrt(A_other / A_own)`` and its targets are
    scaled to match, so the hard area and edge constraints remain satisfied EXACTLY -- no retraction,
    no drift. Positions of the centroids are untouched, so this is a pure size exchange.

    Inert when the two are already the same size, which is why it does nothing on a monodisperse
    packing."""
    areas = np.asarray(packing.targetArea, dtype = float)
    first, second = int(first), int(second)
    if areas[first] <= 0.0 or areas[second] <= 0.0:
        return False
    ratio = np.sqrt(areas[second] / areas[first])
    if not np.isfinite(ratio) or abs(ratio - 1.0) < 1e-15:
        return False
    r = packing.positions.reshape(-1, 2)
    for polygon, factor in ((first, ratio), (second, 1.0 / ratio)):
        a, b = _slice(packing, polygon)
        centroid = r[a:b].mean(axis = 0)
        r[a:b] = centroid + factor * (r[a:b] - centroid)
        packing.targetArea[polygon] *= factor ** 2
        packing.targetEdgeLength[a:b] *= factor
    packing.syncTargetPerimeter()
    return True


# UNVERIFIED(Cam)
def objective(model):
    """Total violation area at the current density: polygon-polygon overlap plus anything outside the
    wall. Zero exactly when the packing is valid, so the search is a decision problem."""
    return model.getPairOverlapArea() + model.getContainerOverlapArea()


# UNVERIFIED(Cam)
def basinHop(model, rounds = 200, temperature = 0.0, moves = ("rotate", "translate", "swap"),
             rotation = _DEFAULT_ROTATION, translation = None, relaxSteps = 2000,
             maxUnbalancedForce = 1e-8, tolerance = 0.0, rng = None, verbose = False):
    """Search for a valid packing AT THE CURRENT DENSITY by perturb-relax-accept. Returns a
    ``SearchResult``; the model is left in the best configuration found.

    Each round proposes one move, relaxes with the same minimizer the sweep uses, and accepts on the
    post-relaxation objective -- that is what makes it BASIN hopping rather than plain Monte Carlo: the
    landscape being explored is the map from a configuration to the minimum it drains into, which has
    no barriers between adjacent basins even though the underlying energy does.

    ``temperature`` 0 accepts only improvements. A positive value accepts a worsening move with
    probability ``exp(-worse/temperature)``, which is what lets it escape a basin that is locally best
    but globally wrong. Units are those of the objective (an AREA), so scale it against the overlap
    you are trying to remove, not against the energy.

    The density is never changed here. That is deliberate: the sweep already establishes what density a
    quench reaches, and the open question is whether a valid arrangement exists slightly ABOVE it. Set
    the density first with ``setPackingFraction``, then ask this whether it can be satisfied."""
    rng = np.random.default_rng() if rng is None else rng
    packing = model.packing
    free = _freeCount(model)
    if free < 2:
        raise ValueError("basinHop needs at least two ordinary polygons.")
    if translation is None:
        from neighbors import meanPolygonRadius
        translation = _DEFAULT_TRANSLATION * meanPolygonRadius(packing)

    result = SearchResult()
    moves = tuple(m for m in moves if m in ("rotate", "translate", "swap"))
    if not moves:
        raise ValueError("basinHop needs at least one of rotate / translate / swap.")

    def relax():
        if model.constraints is not None:
            model.constraints.projectPositions(packing)
        model.minimizeFIRE(maxUnbalancedForce = maxUnbalancedForce, maxSteps = relaxSteps,
                           progressBar = False)
        return objective(model)

    current = relax()
    best = current
    bestState = _capture(model)
    result.objective = best
    result.history.append(dict(round = 0, move = "start", objective = current, accepted = True))
    if current <= tolerance:
        result.solved = True
        return result

    for step in range(1, int(rounds) + 1):
        move = moves[rng.integers(len(moves))]
        saved = _capture(model)
        applied = _propose(packing, move, free, rotation, translation, rng)
        result.proposed[move] += 1
        if not applied:
            _release(model, saved)
            continue

        trial = relax()
        worse = trial - current
        accept = worse <= 0.0 or (temperature > 0.0
                                  and rng.random() < np.exp(-worse / temperature))
        if accept:
            current = trial
            result.accepted[move] += 1
            if trial < best:
                best, bestState = trial, _capture(model)
                result.objective = best
        else:
            _release(model, saved)

        result.rounds = step
        result.history.append(dict(round = step, move = move, objective = trial, accepted = accept))
        if verbose and (step % 10 == 0 or trial <= tolerance):
            print(f"  round {step:4d}  {move:9s}  objective {trial:.6e}  best {best:.6e}"
                  f"{'   <- accepted' if accept else ''}", flush = True)
        if best <= tolerance:
            result.solved = True
            break

    _release(model, bestState)
    result.objective = best
    return result


# UNVERIFIED(Cam)
def _propose(packing, move, free, rotation, translation, rng):
    """Apply one trial move in place. Returns False when the move was a no-op."""
    if move == "rotate":
        rotatePolygon(packing, int(rng.integers(free)), float(rng.normal(0.0, rotation)))
        return True
    if move == "translate":
        delta = translation * rng.standard_normal(2)
        translatePolygon(packing, int(rng.integers(free)), delta)
        return True
    first, second = rng.choice(free, size = 2, replace = False)
    return swapSizes(packing, first, second)


# UNVERIFIED(Cam)
def _capture(model):
    """Everything a move can change: positions AND the size targets, since swap moves both."""
    return (model.packing.positions.copy(),
            np.array(model.packing.targetArea, dtype = float),
            np.array(model.packing.targetEdgeLength, dtype = float))


# UNVERIFIED(Cam)
def _release(model, state):
    positions, targetArea, targetEdgeLength = state
    model.packing.positions[:] = positions
    model.packing.targetArea[:] = targetArea
    model.packing.targetEdgeLength[:] = targetEdgeLength
    model.packing.syncTargetPerimeter()
    model._forces = None
    model._energy = None
