"""The CONTACT GRAPH of a valid packing, and the polynomial system it defines.

A relaxed packing is a numerical object: its side length is good to maybe nine digits, because the
contact law is soft and the final answer comes from a bisection. The contact graph is a COMBINATORIAL
object -- "does square 7's corner touch square 3's face" is a yes/no question that nine digits answers
comfortably -- and it turns the packing into a system of polynomial equations whose solution is exact.
That is the whole move: optimization supplies the combinatorics, and root-finding supplies the digits.
``refine.py`` does the second half.

THE VARIABLES ARE CHOSEN TO KEEP EVERYTHING POLYNOMIAL. Square ``i`` carries ``(x, y, a, b)`` with
``a = cos(theta)``, ``b = sin(theta)`` and ``a^2 + b^2 = 1`` as an explicit equation, rather than an
angle. Add the box side ``s`` and there are ``4N + 1`` unknowns against ``N`` normalizations, so
``3N + 1`` real degrees of freedom.

THERE IS ONLY ONE KIND OF CONTACT EQUATION, which is what keeps this manageable:

    a vertex of i lies on an edge of j:   n_j^m . (v_i^k - c_j) = 1/2
    a vertex of i lies on a wall:         v_i^k.x = 0,  = s,  and likewise in y

A FLUSH FACE-TO-FACE CONTACT IS NOT A SEPARATE CASE. It appears as two vertex-on-edge equations, which
together force both the parallelism and the separation. Treating it that way keeps the system uniform,
and it is why no edge-edge primitive appears below.

THE SQUARES ARE MADE EXACTLY UNIT HERE. A relaxed packing's squares are equal only to the tolerance the
constraints held them to; the contact system imposes exactness by construction, so the packing is
rescaled to unit squares and the box side ``s = boxSide / squareSide`` becomes the quantity to report.
That is also the convention ``records.py`` uses.

# UNVERIFIED(Cam)
"""

import numpy as np

# Local-frame outward normals of a square, in the order used for edge indices.
_NORMALS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
# Local-frame corners, in the order used for vertex indices.
_CORNERS = 0.5 * np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


# UNVERIFIED(Cam)
def squareFrames(packing):
    """``(centers, angles, side, boxSide, boxOrigin)`` from a packing of equal squares in a box.

    The angle is read from the FIRST EDGE of each polygon and folded into ``[0, pi/2)``, because a
    square is invariant under a quarter turn and a frame that is not folded makes two copies of the
    same packing look different. The side is the mean edge length -- the constraints hold the four to
    within their own tolerance, and the refinement will make them exactly equal anyway."""
    vertices = packing.positions.reshape(-1, 2)
    starts = np.asarray(packing.startIndices, dtype = int)
    container = getattr(packing, "containerIndex", None)
    if container is None:
        raise ValueError("a contact graph needs a container: the box is where s comes from")

    centers, angles, sides = [], [], []
    for p in range(len(starts) - 1):
        if p == container:
            continue
        loop = vertices[starts[p]:starts[p + 1]]
        if len(loop) != 4:
            raise ValueError(f"polygon {p} has {len(loop)} vertices; this module is for SQUARES")
        centers.append(loop.mean(axis = 0))
        edge = loop[1] - loop[0]
        angles.append(np.arctan2(edge[1], edge[0]) % (0.5 * np.pi))
        following = np.roll(loop, -1, axis = 0)
        sides.append(float(np.mean(np.hypot(*(following - loop).T))))

    wall = vertices[starts[container]:starts[container + 1]]
    lower, upper = wall.min(axis = 0), wall.max(axis = 0)
    span = upper - lower
    if abs(span[0] - span[1]) > 1e-6 * max(span):
        raise ValueError(f"the container is {span[0]:.6f} by {span[1]:.6f}, not a square")
    return (np.array(centers), np.array(angles), float(np.mean(sides)),
            float(np.mean(span)), lower)


# UNVERIFIED(Cam)
def unitState(packing):
    """``(state, side)`` with the packing rescaled to UNIT squares in a box ``[0, s]^2``.

    ``state`` is the flat vector ``refine`` solves in: ``x, y, a, b`` per square, then ``s``."""
    centers, angles, side, boxSide, origin = squareFrames(packing)
    scaled = (centers - origin) / side
    s = boxSide / side
    state = np.zeros(4 * len(scaled) + 1)
    state[0:4 * len(scaled):4] = scaled[:, 0]
    state[1:4 * len(scaled):4] = scaled[:, 1]
    state[2:4 * len(scaled):4] = np.cos(angles)
    state[3:4 * len(scaled):4] = np.sin(angles)
    state[-1] = s
    return state, s


# UNVERIFIED(Cam)
def _cornersOf(state, i):
    x, y, a, b = state[4 * i:4 * i + 4]
    rotated = np.stack([a * _CORNERS[:, 0] - b * _CORNERS[:, 1],
                        b * _CORNERS[:, 0] + a * _CORNERS[:, 1]], axis = 1)
    return rotated + np.array([x, y])


# UNVERIFIED(Cam)
def _normalsOf(state, j):
    a, b = state[4 * j + 2], state[4 * j + 3]
    return np.stack([a * _NORMALS[:, 0] - b * _NORMALS[:, 1],
                     b * _NORMALS[:, 0] + a * _NORMALS[:, 1]], axis = 1)


# UNVERIFIED(Cam)
def extractContacts(state, count, tolerance = 1e-6):
    """Every contact in a unit-square state, as ``(kind, i, vertex, j, edge)`` tuples.

    ``kind`` is ``"square"`` for a vertex of ``i`` on an edge of ``j``, or ``"wall"`` for a vertex of
    ``i`` on wall edge ``0..3`` (``x = 0``, ``y = 0``, ``x = s``, ``y = s``), in which case ``j`` is
    ``-1``.

    A VERTEX-VERTEX TOUCH IS REPORTED AS ``kind = "corner"`` AND IS NOT AN EQUATION. Two corners
    meeting is a codimension-two coincidence: it is one geometric condition too many for the single
    equation a face contact carries, and including it as one would silently over-constrain the system.
    It is surfaced so the caller can see it rather than dropped."""
    found = []
    for i in range(count):
        corners = _cornersOf(state, i)
        for k, vertex in enumerate(corners):
            for j in range(count):
                if j == i:
                    continue
                center = state[4 * j:4 * j + 2]
                a, b = state[4 * j + 2], state[4 * j + 3]
                delta = vertex - center
                # Into j's own frame, where the square is axis-aligned and the test is a comparison.
                local = np.array([a * delta[0] + b * delta[1], -b * delta[0] + a * delta[1]])
                outside = np.abs(local) - 0.5
                # THE TRUE DISTANCE TO THE SQUARE, not to an edge's infinite plane. Testing the plane
                # alone reports a contact for any vertex that happens to be level with a face however
                # far away it is along the face: measured, that turned four real contacts into
                # thirty-six and made the system look wildly hyperstatic.
                if outside[0] > 0.0 or outside[1] > 0.0:
                    beyond = np.maximum(outside, 0.0)
                    distance = float(np.hypot(beyond[0], beyond[1]))
                else:
                    distance = float(max(outside))      # negative: an overlap, not a contact
                if abs(distance) > tolerance:
                    continue
                if outside[0] > tolerance and outside[1] > tolerance:
                    found.append(("corner", i, k, j, -1))
                elif outside[0] >= outside[1]:
                    found.append(("square", i, k, j, 0 if local[0] > 0.0 else 2))
                else:
                    found.append(("square", i, k, j, 1 if local[1] > 0.0 else 3))
            for edge, gap in enumerate((vertex[0], vertex[1],
                                        state[-1] - vertex[0], state[-1] - vertex[1])):
                if abs(gap) <= tolerance:
                    found.append(("wall", i, k, -1, edge))
    return found


# UNVERIFIED(Cam)
def gaps(state, count):
    """Every vertex-to-square and vertex-to-wall gap in the packing, sorted, in UNIT-SQUARE units.

    THE TOLERANCE IS THE WHOLE GAME AND IT SHOULD BE READ OFF THIS, not guessed. A relaxed packing's
    contacts do not sit at zero: the bisection that ends a protocol stops at the largest VALID size,
    and the final relaxation then disengages everything by whatever slack it left. Too tight a
    tolerance and every square is a rattler; too loose and spurious pairs over-constrain the system,
    which shows up as a Newton residual that never falls or a converged answer that violates a
    non-overlap inequality.

    A real contact set separates: touching pairs cluster together and the next gap up is orders of
    magnitude away. Units are the SQUARE SIDE, since the state has already been rescaled to unit
    squares -- a gap of 1e-6 here is 1e-6 of a square's side, not of the box."""
    found = []
    for i in range(count):
        for vertex in _cornersOf(state, i):
            for j in range(count):
                if j == i:
                    continue
                center = state[4 * j:4 * j + 2]
                a, b = state[4 * j + 2], state[4 * j + 3]
                delta = vertex - center
                local = np.array([a * delta[0] + b * delta[1], -b * delta[0] + a * delta[1]])
                outside = np.abs(local) - 0.5
                if outside[0] > 0.0 or outside[1] > 0.0:
                    beyond = np.maximum(outside, 0.0)
                    found.append(float(np.hypot(beyond[0], beyond[1])))
                else:
                    found.append(float(-max(outside)))
            for gap in (vertex[0], vertex[1], state[-1] - vertex[0], state[-1] - vertex[1]):
                found.append(abs(float(gap)))
    return np.sort(np.array(found))


# UNVERIFIED(Cam)
def toleranceLadder(state, count, levels = None):
    """How the contact set changes as the tolerance is swept. Returns rows of
    ``(tolerance, contacts, rattlers, equations, unknowns, verdict)``.

    THE RIGHT TOLERANCE SITS ON A PLATEAU. A contact set that is stable across two or three decades is
    a real one; a count that climbs steadily with the tolerance means there is no separation and the
    packing is not jammed, which no amount of buffer will fix. Read this before choosing."""
    if levels is None:
        levels = [10.0 ** k for k in range(-10, -1)]
    rows = []
    for level in levels:
        verdict = audit(state, count, tolerance = level)
        rows.append((level, len(verdict["contacts"]), len(verdict["rattlers"]),
                     len(verdict["equations"]), verdict["unknowns"], verdict["verdict"]))
    return rows


# UNVERIFIED(Cam)
def suggestTolerance(state, count, levels = None):
    """A tolerance from the widest PLATEAU in the tolerance ladder, or None if there is no plateau.

    NOT FROM THE BIGGEST GAP IN THE SPECTRUM, which sounds right and is not: a single grazing pair
    makes a large ratio jump all by itself, and the tolerance it suggests then admits that one contact
    and leaves every square a rattler. Measured on a loose n = 11 packing, the spectrum jumped from
    2.6e-08 to 2.4e-04 -- a factor of ten thousand -- and the tolerance it implied produced one contact
    and eleven rattlers.

    A REAL CONTACT SET IS STABLE OVER DECADES. So the ladder is swept, runs of consecutive levels
    giving the identical contact set are found, and the middle of the longest run is returned.
    Returning None is a real answer and usually the important one: it means the gaps grade smoothly
    rather than clustering, so nothing distinguishes a contact from a near-miss, and the packing is not
    jammed. No buffer fixes that -- ``squeeze.squeeze`` does, by closing the slack.

    A RUN IS REJECTED ONLY IF *EVERY* SQUARE IS A RATTLER, not if any is. Demanding a rattler-free run
    was the first version and it is wrong: rattlers are ordinary at real optima, and the squeezed
    26-square packing -- gaps at 2e-14 with the next at 1e-07, about as clean a separation as exists --
    has five of them and returned None, which read as "not jammed" for a packing sitting exactly on the
    record. All squares loose is the condition that actually means unjammed."""
    if levels is None:
        levels = [10.0 ** k for k in range(-11, -1)]
    rows = toleranceLadder(state, count, levels)
    best, run = None, []
    for index, row in enumerate(rows):
        tolerance, contacts, rattlers = row[0], row[1], row[2]
        if rattlers >= count or contacts == 0:
            run = []
            continue
        if run and (contacts, rattlers) == (rows[index - 1][1], rows[index - 1][2]):
            run.append(tolerance)
        else:
            run = [tolerance]
        if best is None or len(run) > len(best):
            best = list(run)
    if best is None or len(best) < 2:
        return None
    return float(np.sqrt(best[0] * best[-1]))


# UNVERIFIED(Cam)
def contactCounts(contacts, count):
    """How many equations each square carries. Corner touches are excluded -- they are not equations."""
    tally = np.zeros(count, dtype = int)
    for kind, i, _, j, _ in contacts:
        if kind == "corner":
            continue
        tally[i] += 1
        if kind == "square":
            tally[j] += 1
    return tally


# UNVERIFIED(Cam)
def contactJacobian(state, equations, count):
    """The contact Jacobian in double precision, rows by ``4 * count + 1`` columns.

    A cheap numpy twin of ``refine.residualAndJacobian``'s Jacobian half, for the audit -- the exact
    version lives there and carries the residual with it. Kept in step by hand; they differ only in
    arithmetic, not in the derivatives."""
    unknowns = 4 * count + 1
    J = np.zeros((len(equations), unknowns))
    for row, (kind, i, k, j, edge) in enumerate(equations):
        ui, vi = _CORNERS[k]
        pi = 4 * i
        a, b = state[pi + 2], state[pi + 3]
        vx = state[pi] + a * ui - b * vi
        vy = state[pi + 1] + b * ui + a * vi
        if kind == "wall":
            if edge in (0, 2):
                J[row, pi], J[row, pi + 2], J[row, pi + 3] = 1.0, ui, -vi
            else:
                J[row, pi + 1], J[row, pi + 2], J[row, pi + 3] = 1.0, vi, ui
            if edge in (2, 3):
                J[row, unknowns - 1] = -1.0
            continue
        em, en = _NORMALS[edge]
        pj = 4 * j
        aj, bj = state[pj + 2], state[pj + 3]
        nx, ny = aj * em - bj * en, bj * em + aj * en
        wx, wy = vx - state[pj], vy - state[pj + 1]
        J[row, pi], J[row, pi + 1] = nx, ny
        J[row, pi + 2] = nx * ui + ny * vi
        J[row, pi + 3] = -nx * vi + ny * ui
        J[row, pj], J[row, pj + 1] = -nx, -ny
        J[row, pj + 2] = em * wx + en * wy
        J[row, pj + 3] = -en * wx + em * wy
    return J


# UNVERIFIED(Cam)
def findRattlers(state, contacts, count):
    """Squares the contacts fail to hold, found by peeling on CONTACT COUNT.

    Fewer than three contacts cannot fix three degrees of freedom, so such a square is loose; and
    removing it can drop a neighbour below three in turn, hence the fixpoint. Rattlers must come OUT
    before the system is solved: each adds three unknowns and too few equations.

    PEELING ON LOCAL RANK INSTEAD IS WRONG, AND WAS TRIED. It looks like the sharper test -- three
    contacts fix three freedoms only if their normals are independent -- but it removes squares that
    are FIRST-ORDER FLEXIBLE AND SECOND-ORDER RIGID, which are not rattlers at all. In the proved
    n = 5 optimum the tilted middle square is exactly that: every contact meets it at the MIDPOINT of
    a face, where w is parallel to n, so its rotation is a null direction of its own block. A rank peel
    calls it a rattler and deletes the one square the packing is built around -- measured, it took the
    self-test from an exact answer to no answer at all. First-order flexibility is handled globally
    instead, by pinning the direction and making s stationary along it; see ``refine.stationary``.

    ``state`` is unused and kept in the signature on purpose, so this and ``localRanks`` are called the
    same way and choosing between them is explicit at the call site."""
    alive = np.ones(count, dtype = bool)
    while True:
        live = [c for c in contacts
                if c[0] != "corner" and alive[c[1]] and (c[3] < 0 or alive[c[3]])]
        tally = np.zeros(count, dtype = int)
        for kind, i, _, j, _ in live:
            tally[i] += 1
            if kind == "square":
                tally[j] += 1
        loose = alive & (tally < 3)
        if not loose.any():
            return np.flatnonzero(~alive)
        alive[loose] = False


# UNVERIFIED(Cam)
def localRanks(state, equations, count):
    """Rank of each square's own constraint block, out of 4. DIAGNOSTIC ONLY, never a peel criterion.

    A deficiency of one usually means SECOND-ORDER RIGID rather than loose -- see ``findRattlers`` --
    so read this as "which squares the contacts do not pin at first order", and expect the genuinely
    extremal ones to show up in it."""
    J = contactJacobian(state, equations, count)
    out = np.zeros(count, dtype = int)
    for body in range(count):
        rows = [r for r, c in enumerate(equations) if c[1] == body or c[3] == body]
        if not rows:
            continue
        block = np.vstack([J[rows, 4 * body:4 * body + 4],
                           [0.0, 0.0, 2.0 * state[4 * body + 2], 2.0 * state[4 * body + 3]]])
        out[body] = np.linalg.matrix_rank(block, tol = 1e-9 * max(1.0, np.abs(block).max()))
    return out


# UNVERIFIED(Cam)
def audit(state, count, tolerance = None):
    """A readable verdict on whether this contact graph determines the packing.

    Returns a dict with the contacts, the rattlers, and the degree-of-freedom count. THE THREE CASES
    MEAN DIFFERENT THINGS and the distinction is the whole point of running this:

      isostatic    equations == 3N + 1. The system pins every coordinate and s. Newton just works.
      hypostatic   fewer. Something can still move, so this is NOT a local maximum and s is not
                   determined -- almost always a rattler that has not been removed.
      hyperstatic  more. Common at real optima, where symmetry makes contacts dependent. The system is
                   overdetermined but consistent; solve it in least squares and CHECK the residual
                   actually reaches zero, because that is the test of whether the contact set is right.
    """
    if tolerance is None:
        tolerance = suggestTolerance(state, count)
    if tolerance is None:
        raise ValueError(
            "no tolerance separates the contacts -- the gaps grade smoothly rather than clustering. "
            "Inspect contacts.gaps and contacts.toleranceLadder, or use refine.search to let "
            "convergence decide, but be aware this usually means the packing was read after its final "
            "decompression rather than at the jamming point.")
    contacts = extractContacts(state, count, tolerance)
    rattlers = findRattlers(state, contacts, count)
    free = np.setdiff1d(np.arange(count), rattlers)
    kept = [c for c in contacts
            if c[0] != "corner" and c[1] in free and (c[3] < 0 or c[3] in free)]
    unknowns = 3 * len(free) + 1
    verdict = ("isostatic" if len(kept) == unknowns
               else "hypostatic" if len(kept) < unknowns else "hyperstatic")
    return {"contacts": contacts, "equations": kept, "rattlers": rattlers, "free": free,
            "unknowns": unknowns, "verdict": verdict,
            "corners": [c for c in contacts if c[0] == "corner"]}
